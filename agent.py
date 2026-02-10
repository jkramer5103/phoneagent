#!/usr/bin/env python3
"""
AI Voice Agent REST API: POST /call with a phone number and objective.
Calls the number via SIP/RTP, transcribes speech with Soniox real-time STT,
generates responses with Gemini via OpenRouter, speaks them back using
Cartesia Sonic TTS. Returns a structured summary when the call ends.
"""

import os
import re
import json
import time
import wave
import socket
import struct
import hashlib
import random
import audioop
import queue
import threading
import asyncio
import httpx
from cartesia import Cartesia
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from websockets.sync.client import connect as ws_connect

load_dotenv()

# --- Configuration ---
SIP_SERVER = "speedport.ip"
SIP_PORT = 5060
SIP_URI_USER = "**71"
AUTH_USER = "nutzer-1@speedport.ip"
AUTH_PASS = "B-fx3$h-7yH4&42"
LOCAL_SIP_PORT = 5060
RTP_PORT = 16384

OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
SONIOX_API_KEY = os.getenv("SONIOX_API_KEY")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
LLM_MODEL = "google/gemini-3-flash-preview"  # Smart LLM for text generation
CARTESIA_VOICE_ID = "afa425cf-5489-4a09-8a3f-d3cb1f82150d"  # Nico - Friendly Agent (German)

HANGUP_TOKEN = "[HANGUP]"
INTERRUPT_GRACE_PERIOD = 3.0  # Seconds after agent starts speaking before interruption is allowed
RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")


# ============================================================
# Call Recording
# ============================================================

class CallRecorder:
    """Records both sides of a call by capturing raw u-law RTP audio with timestamps."""

    def __init__(self):
        self.start_time = None
        self.inbound = []   # (offset_samples, ulaw_bytes) from remote party
        self.outbound = []  # (offset_samples, ulaw_bytes) from agent
        self.lock = threading.Lock()

    def start(self):
        self.start_time = time.time()

    def _offset(self):
        """Current offset in samples (8kHz) since recording started."""
        if self.start_time is None:
            return 0
        return int((time.time() - self.start_time) * 8000)

    def record_inbound(self, ulaw_data):
        if self.start_time is None:
            return
        with self.lock:
            self.inbound.append((self._offset(), ulaw_data))

    def record_outbound(self, ulaw_data):
        if self.start_time is None:
            return
        with self.lock:
            self.outbound.append((self._offset(), ulaw_data))

    def save(self, filepath):
        """Mix inbound + outbound into a single-channel 8kHz 16-bit PCM WAV."""
        if self.start_time is None:
            print("[REC] No recording to save")
            return

        with self.lock:
            inbound = list(self.inbound)
            outbound = list(self.outbound)

        # Find total length in samples
        max_sample = 0
        for offset, data in inbound + outbound:
            end = offset + len(data)
            if end > max_sample:
                max_sample = end

        if max_sample == 0:
            print("[REC] No audio captured")
            return

        # Create PCM buffers (16-bit signed, so 2 bytes per sample)
        pcm_in = bytearray(max_sample * 2)
        pcm_out = bytearray(max_sample * 2)

        # Decode u-law chunks into PCM buffers at their correct offsets
        for offset, data in inbound:
            try:
                pcm = audioop.ulaw2lin(data, 2)
                byte_offset = offset * 2
                end = byte_offset + len(pcm)
                if end <= len(pcm_in):
                    pcm_in[byte_offset:end] = pcm
            except audioop.error:
                pass

        for offset, data in outbound:
            try:
                pcm = audioop.ulaw2lin(data, 2)
                byte_offset = offset * 2
                end = byte_offset + len(pcm)
                if end <= len(pcm_out):
                    pcm_out[byte_offset:end] = pcm
            except audioop.error:
                pass

        # Mix both channels by adding samples (with clipping)
        mixed = audioop.add(bytes(pcm_in), bytes(pcm_out), 2)

        # Write WAV
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(8000)
            wf.writeframes(mixed)

        duration = max_sample / 8000
        print(f"[REC] Saved {duration:.1f}s recording to {filepath}")

# --- API Models ---
class CallRequest(BaseModel):
    number: str
    objective: str

class CallResult(BaseModel):
    status: str
    summary: str
    objective_completed: bool
    transcript: list[dict]

# --- Per-call State ---
class AgentState:
    def __init__(self, call_number: str, objective: str):
        self.call_number = call_number
        self.objective = objective
        self.is_speaking = False
        self.interrupt_flag = False
        self.call_active = True
        self.conversation = []
        self.lock = threading.Lock()
        self.pending_text = ""
        self.silence_start = None
        self.user_speaking = False
        self.speaking_ended_at = 0.0
        self.speaking_started_at = 0.0
        self.llm_busy = False
        self.hangup_requested = False
        self.sip_sock = None
        self.sip_call_info = None
        self.recorder = CallRecorder()
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self):
        return (
            "Du bist ein KI-Telefonassistent von Jaron Kramer. "
            "Du rufst im Auftrag von Jaron an. "
            "Stell dich als KI-Assistent von Jaron Kramer vor, wenn es passt.\n\n"
            f"Dein Auftrag für diesen Anruf: {self.objective}\n\n"
            "Regeln:\n"
            "- Halte deine Antworten kurz und natürlich — normalerweise 1-2 Sätze.\n"
            "- Du sprichst Deutsch.\n"
            "- Sei höflich, direkt und zielgerichtet.\n"
            "- DU bist der Anrufer. DU führst das Gespräch und sagst, was du brauchst. "
            "Warte nicht darauf, dass der andere fragt — sag direkt, worum es geht.\n"
            "- Reagiere natürlich auf Rückfragen und arbeite auf dein Ziel hin.\n"
            "- Wenn dein Anliegen erledigt ist oder das Gespräch zu Ende ist, "
            "verabschiede dich kurz und füge am Ende das Token [HANGUP] hinzu.\n"
            "- Beispiel: 'Alles klar, vielen Dank! Tschüss! [HANGUP]'"
        )

# Lock to prevent concurrent calls (single SIP port)
call_lock = threading.Lock()


# ============================================================
# SIP Helpers (reused from call.py)
# ============================================================

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((SIP_SERVER, SIP_PORT))
        return s.getsockname()[0]
    finally:
        s.close()

def gen_branch():
    return f"z9hG4bK{random.randint(100000000, 999999999)}"

def gen_tag():
    return f"{random.randint(100000000, 999999999)}"

def gen_call_id(local_ip):
    return f"{random.randint(100000000, 999999999)}@{local_ip}"

def digest_auth(auth_user, password, realm, nonce, method, uri, qop=None, nc=None, cnonce=None):
    ha1 = hashlib.md5(f"{auth_user}:{realm}:{password}".encode()).hexdigest()
    ha2 = hashlib.md5(f"{method}:{uri}".encode()).hexdigest()
    if qop == "auth":
        response = hashlib.md5(f"{ha1}:{nonce}:{nc}:{cnonce}:{qop}:{ha2}".encode()).hexdigest()
    else:
        response = hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode()).hexdigest()
    return response

def build_auth_header(auth_user, password, realm, nonce, method, uri, qop=None, algorithm="MD5"):
    header = f'Digest username="{auth_user}",realm="{realm}",nonce="{nonce}",uri="{uri}"'
    if qop == "auth":
        nc = "00000001"
        cnonce = hashlib.md5(os.urandom(16)).hexdigest()[:16]
        resp = digest_auth(auth_user, password, realm, nonce, method, uri, qop, nc, cnonce)
        header += f',qop={qop},nc={nc},cnonce="{cnonce}"'
    else:
        resp = digest_auth(auth_user, password, realm, nonce, method, uri)
    header += f',response="{resp}",algorithm={algorithm}'
    return header

def parse_www_authenticate(header_line):
    result = {}
    for m in re.finditer(r'(\w+)=(?:"([^"]+)"|(\S+?)(?:,|$))', header_line):
        key = m.group(1)
        value = m.group(2) if m.group(2) is not None else m.group(3)
        result[key] = value
    return result

def parse_sip_response(data):
    text = data.decode(errors="replace")
    parts = text.split("\r\n\r\n", 1)
    header_section = parts[0]
    body = parts[1] if len(parts) > 1 else ""
    lines = header_section.split("\r\n")
    status_line = lines[0]
    status_code = int(status_line.split(" ", 2)[1])
    headers = {}
    for line in lines[1:]:
        if ":" in line:
            key, val = line.split(":", 1)
            headers[key.strip()] = val.strip()
    return status_code, headers, body, status_line

def sip_send_recv(sock, msg, server, port, timeout=5):
    sock.sendto(msg.encode(), (server, port))
    sock.settimeout(timeout)
    try:
        data, addr = sock.recvfrom(8192)
        return data
    except socket.timeout:
        return None


# ============================================================
# SIP Registration & Call Setup
# ============================================================

def sip_register(sock, local_ip):
    tag = gen_tag()
    call_id = gen_call_id(local_ip)
    branch = gen_branch()

    msg = (
        f"REGISTER sip:{SIP_SERVER} SIP/2.0\r\n"
        f"Via: SIP/2.0/UDP {local_ip}:{LOCAL_SIP_PORT};branch={branch};rport\r\n"
        f"From: <sip:{SIP_URI_USER}@{SIP_SERVER}>;tag={tag}\r\n"
        f"To: <sip:{SIP_URI_USER}@{SIP_SERVER}>\r\n"
        f"Call-ID: {call_id}\r\n"
        f"CSeq: 1 REGISTER\r\n"
        f"Contact: <sip:{SIP_URI_USER}@{local_ip}:{LOCAL_SIP_PORT};transport=udp>\r\n"
        f"Max-Forwards: 70\r\n"
        f"Expires: 300\r\n"
        f"User-Agent: VoIPAgent/1.0\r\n"
        f"Content-Length: 0\r\n\r\n"
    )

    print("[SIP] Sending initial REGISTER...")
    resp = sip_send_recv(sock, msg, SIP_SERVER, SIP_PORT)
    if resp is None:
        print("[SIP] No response to REGISTER")
        return False

    status, headers, body, status_line = parse_sip_response(resp)
    print(f"[SIP] {status_line}")

    if status == 423:
        min_exp = headers.get("Min-Expires", "300")
        branch = gen_branch()
        msg = (
            f"REGISTER sip:{SIP_SERVER} SIP/2.0\r\n"
            f"Via: SIP/2.0/UDP {local_ip}:{LOCAL_SIP_PORT};branch={branch};rport\r\n"
            f"From: <sip:{SIP_URI_USER}@{SIP_SERVER}>;tag={tag}\r\n"
            f"To: <sip:{SIP_URI_USER}@{SIP_SERVER}>\r\n"
            f"Call-ID: {call_id}\r\n"
            f"CSeq: 2 REGISTER\r\n"
            f"Contact: <sip:{SIP_URI_USER}@{local_ip}:{LOCAL_SIP_PORT};transport=udp>\r\n"
            f"Max-Forwards: 70\r\n"
            f"Expires: {min_exp}\r\n"
            f"User-Agent: VoIPAgent/1.0\r\n"
            f"Content-Length: 0\r\n\r\n"
        )
        resp = sip_send_recv(sock, msg, SIP_SERVER, SIP_PORT)
        if resp is None:
            return False
        status, headers, body, status_line = parse_sip_response(resp)
        print(f"[SIP] {status_line}")

    if status == 200:
        print("[SIP] Registered!")
        return True
    if status != 401:
        print(f"[SIP] Unexpected: {status}")
        return False

    www_auth = headers.get("WWW-Authenticate", "")
    auth_params = parse_www_authenticate(www_auth)
    realm = auth_params.get("realm", SIP_SERVER)
    nonce = auth_params.get("nonce", "")
    qop = auth_params.get("qop", None)
    algorithm = auth_params.get("algorithm", "MD5")

    reg_uri = f"sip:{SIP_SERVER}"
    auth_header = build_auth_header(AUTH_USER, AUTH_PASS, realm, nonce, "REGISTER", reg_uri, qop, algorithm)

    branch = gen_branch()
    msg = (
        f"REGISTER sip:{SIP_SERVER} SIP/2.0\r\n"
        f"Via: SIP/2.0/UDP {local_ip}:{LOCAL_SIP_PORT};branch={branch};rport\r\n"
        f"From: <sip:{SIP_URI_USER}@{SIP_SERVER}>;tag={tag}\r\n"
        f"To: <sip:{SIP_URI_USER}@{SIP_SERVER}>\r\n"
        f"Call-ID: {call_id}\r\n"
        f"CSeq: 3 REGISTER\r\n"
        f"Contact: <sip:{SIP_URI_USER}@{local_ip}:{LOCAL_SIP_PORT};transport=udp>\r\n"
        f"Max-Forwards: 70\r\n"
        f"Expires: 300\r\n"
        f"Authorization: {auth_header}\r\n"
        f"User-Agent: VoIPAgent/1.0\r\n"
        f"Content-Length: 0\r\n\r\n"
    )

    print("[SIP] Sending authenticated REGISTER...")
    resp = sip_send_recv(sock, msg, SIP_SERVER, SIP_PORT)
    if resp is None:
        return False

    status, headers, body, status_line = parse_sip_response(resp)
    print(f"[SIP] {status_line}")
    if status == 200:
        print("[SIP] Registration successful!")
        return True
    print(f"[SIP] Registration failed: {status}")
    return False


def sip_invite(sock, local_ip, call_number):
    tag = gen_tag()
    call_id = gen_call_id(local_ip)
    branch = gen_branch()
    sess_id = str(random.randint(1000000, 9999999))

    sdp_body = (
        f"v=0\r\n"
        f"o=VoIPAgent {sess_id} {sess_id} IN IP4 {local_ip}\r\n"
        f"s=VoIPAgent\r\n"
        f"c=IN IP4 {local_ip}\r\n"
        f"t=0 0\r\n"
        f"m=audio {RTP_PORT} RTP/AVP 0 101\r\n"
        f"a=rtpmap:0 PCMU/8000\r\n"
        f"a=rtpmap:101 telephone-event/8000\r\n"
        f"a=fmtp:101 0-15\r\n"
        f"a=ptime:20\r\n"
        f"a=sendrecv\r\n"
    )

    msg = (
        f"INVITE sip:{call_number}@{SIP_SERVER} SIP/2.0\r\n"
        f"Via: SIP/2.0/UDP {local_ip}:{LOCAL_SIP_PORT};branch={branch};rport\r\n"
        f"From: <sip:{SIP_URI_USER}@{SIP_SERVER}>;tag={tag}\r\n"
        f"To: <sip:{call_number}@{SIP_SERVER}>\r\n"
        f"Call-ID: {call_id}\r\n"
        f"CSeq: 1 INVITE\r\n"
        f"Contact: <sip:{SIP_URI_USER}@{local_ip}:{LOCAL_SIP_PORT}>\r\n"
        f"Max-Forwards: 70\r\n"
        f"Content-Type: application/sdp\r\n"
        f"User-Agent: VoIPAgent/1.0\r\n"
        f"Content-Length: {len(sdp_body)}\r\n\r\n"
        f"{sdp_body}"
    )

    print(f"[SIP] INVITE {call_number}...")
    resp = sip_send_recv(sock, msg, SIP_SERVER, SIP_PORT, timeout=5)
    if resp is None:
        print("[SIP] No response to INVITE")
        return None

    status, headers, body, status_line = parse_sip_response(resp)
    print(f"[SIP] {status_line}")

    cseq = 1
    if status in (401, 407):
        auth_hdr_name = "Proxy-Authenticate" if status == 407 else "WWW-Authenticate"
        www_auth = headers.get(auth_hdr_name, "")
        auth_params = parse_www_authenticate(www_auth)
        realm = auth_params.get("realm", SIP_SERVER)
        nonce = auth_params.get("nonce", "")
        qop = auth_params.get("qop", None)
        algorithm = auth_params.get("algorithm", "MD5")

        invite_uri = f"sip:{call_number}@{SIP_SERVER}"
        auth_header = build_auth_header(AUTH_USER, AUTH_PASS, realm, nonce, "INVITE", invite_uri, qop, algorithm)
        auth_line_name = "Proxy-Authorization" if status == 407 else "Authorization"

        ack_msg = (
            f"ACK sip:{call_number}@{SIP_SERVER} SIP/2.0\r\n"
            f"Via: SIP/2.0/UDP {local_ip}:{LOCAL_SIP_PORT};branch={branch};rport\r\n"
            f"From: <sip:{SIP_URI_USER}@{SIP_SERVER}>;tag={tag}\r\n"
            f"To: {headers.get('To', f'<sip:{call_number}@{SIP_SERVER}>')}\r\n"
            f"Call-ID: {call_id}\r\n"
            f"CSeq: 1 ACK\r\n"
            f"Max-Forwards: 70\r\n"
            f"Content-Length: 0\r\n\r\n"
        )
        sock.sendto(ack_msg.encode(), (SIP_SERVER, SIP_PORT))

        branch = gen_branch()
        cseq = 2
        msg = (
            f"INVITE sip:{call_number}@{SIP_SERVER} SIP/2.0\r\n"
            f"Via: SIP/2.0/UDP {local_ip}:{LOCAL_SIP_PORT};branch={branch};rport\r\n"
            f"From: <sip:{SIP_URI_USER}@{SIP_SERVER}>;tag={tag}\r\n"
            f"To: <sip:{call_number}@{SIP_SERVER}>\r\n"
            f"Call-ID: {call_id}\r\n"
            f"CSeq: {cseq} INVITE\r\n"
            f"Contact: <sip:{SIP_URI_USER}@{local_ip}:{LOCAL_SIP_PORT}>\r\n"
            f"Max-Forwards: 70\r\n"
            f"{auth_line_name}: {auth_header}\r\n"
            f"Content-Type: application/sdp\r\n"
            f"User-Agent: VoIPAgent/1.0\r\n"
            f"Content-Length: {len(sdp_body)}\r\n\r\n"
            f"{sdp_body}"
        )

        print("[SIP] Sending authenticated INVITE...")
        resp = sip_send_recv(sock, msg, SIP_SERVER, SIP_PORT, timeout=5)
        if resp is None:
            return None
        status, headers, body, status_line = parse_sip_response(resp)
        print(f"[SIP] {status_line}")

    while status in (100, 180, 183):
        print(f"[SIP] {status_line}")
        sock.settimeout(60)
        try:
            resp_data, _ = sock.recvfrom(8192)
        except socket.timeout:
            print("[SIP] Timeout waiting for answer")
            return None
        status, headers, body, status_line = parse_sip_response(resp_data)
        print(f"[SIP] {status_line}")

    if status != 200:
        print(f"[SIP] Call failed: {status}")
        return None

    remote_rtp_ip = None
    remote_rtp_port = None
    sdp_lines = body.replace("\r\n", "\n").split("\n")
    for line in sdp_lines:
        if line.startswith("c=IN IP4 "):
            remote_rtp_ip = line.split()[-1].strip()
        if line.startswith("m=audio "):
            remote_rtp_port = int(line.split()[1])

    print(f"[SIP] Call answered! Remote RTP: {remote_rtp_ip}:{remote_rtp_port}")

    to_header = headers.get("To", f"<sip:{call_number}@{SIP_SERVER}>")
    ack_msg = (
        f"ACK sip:{call_number}@{SIP_SERVER} SIP/2.0\r\n"
        f"Via: SIP/2.0/UDP {local_ip}:{LOCAL_SIP_PORT};branch={gen_branch()};rport\r\n"
        f"From: <sip:{SIP_URI_USER}@{SIP_SERVER}>;tag={tag}\r\n"
        f"To: {to_header}\r\n"
        f"Call-ID: {call_id}\r\n"
        f"CSeq: {cseq} ACK\r\n"
        f"Max-Forwards: 70\r\n"
        f"Content-Length: 0\r\n\r\n"
    )
    sock.sendto(ack_msg.encode(), (SIP_SERVER, SIP_PORT))

    return call_id, tag, cseq, remote_rtp_ip, remote_rtp_port, to_header


def sip_bye(sock, local_ip, call_id, tag, cseq, to_header, call_number):
    branch = gen_branch()
    msg = (
        f"BYE sip:{call_number}@{SIP_SERVER} SIP/2.0\r\n"
        f"Via: SIP/2.0/UDP {local_ip}:{LOCAL_SIP_PORT};branch={branch};rport\r\n"
        f"From: <sip:{SIP_URI_USER}@{SIP_SERVER}>;tag={tag}\r\n"
        f"To: {to_header}\r\n"
        f"Call-ID: {call_id}\r\n"
        f"CSeq: {cseq + 1} BYE\r\n"
        f"Max-Forwards: 70\r\n"
        f"Content-Length: 0\r\n\r\n"
    )
    print("[SIP] Sending BYE...")
    resp = sip_send_recv(sock, msg, SIP_SERVER, SIP_PORT, timeout=5)
    if resp:
        status, _, _, sl = parse_sip_response(resp)
        print(f"[SIP] BYE: {sl}")


# ============================================================
# TTS: Cartesia Sonic (ultra-low latency streaming neural TTS)
# ============================================================

CARTESIA_SAMPLE_RATE = 44100  # Synthesize at high quality, downsample for RTP


def cartesia_tts_ulaw(text, tts_ws, tts_client):
    """Convert text to u-law audio via Cartesia Sonic (single shot, for greeting)."""
    if not text.strip():
        return b""

    pcm_chunks = []
    try:
        for output in tts_ws.send(
            model_id="sonic-3",
            transcript=text,
            voice={"mode": "id", "id": CARTESIA_VOICE_ID},
            language="de",
            output_format={"container": "raw", "encoding": "pcm_s16le", "sample_rate": CARTESIA_SAMPLE_RATE},
        ):
            if hasattr(output, 'audio') and output.audio:
                pcm_chunks.append(output.audio)
    except Exception as e:
        print(f"[TTS] Cartesia error: {e}")
        return b""

    if not pcm_chunks:
        return b""

    pcm = b"".join(pcm_chunks)
    pcm8k, _ = audioop.ratecv(pcm, 2, 1, CARTESIA_SAMPLE_RATE, 8000, None)
    return audioop.lin2ulaw(pcm8k, 2)


# ============================================================
# RTP Send (with interruption support) + Keepalive
# ============================================================

class RTPState:
    def __init__(self):
        self.seq = random.randint(0, 0xFFFF)
        self.timestamp = random.randint(0, 0xFFFFFFFF)
        self.ssrc = random.randint(0, 0xFFFFFFFF)
        self.lock = threading.Lock()


def rtp_keepalive_thread(rtp_sock, remote_ip, remote_port, state, rtp_state):
    """Send silence RTP packets to keep the call alive when not speaking."""
    CHUNK_SIZE = 160
    silence = b'\xff' * CHUNK_SIZE  # 0xFF = silence in u-law

    while state.call_active:
        if not state.is_speaking:
            with rtp_state.lock:
                rtp_header = struct.pack("!BBHII", 0x80, 0,
                                         rtp_state.seq & 0xFFFF,
                                         rtp_state.timestamp & 0xFFFFFFFF,
                                         rtp_state.ssrc)
                rtp_state.seq += 1
                rtp_state.timestamp += CHUNK_SIZE
            try:
                rtp_sock.sendto(rtp_header + silence, (remote_ip, remote_port))
            except OSError:
                break
        time.sleep(0.02)

    print("[RTP-KA] Keepalive stopped")


def rtp_send_thread(rtp_sock, remote_ip, remote_port, ulaw_audio, state, rtp_state):
    """Send u-law audio over RTP. Stops if interrupted."""
    CHUNK_SIZE = 160

    state.is_speaking = True
    state.speaking_started_at = time.time()
    state.interrupt_flag = False

    for i in range(0, len(ulaw_audio), CHUNK_SIZE):
        if state.interrupt_flag or not state.call_active:
            print("[RTP-TX] Interrupted!")
            break

        chunk = ulaw_audio[i:i + CHUNK_SIZE]
        if len(chunk) < CHUNK_SIZE:
            chunk += b'\xff' * (CHUNK_SIZE - len(chunk))

        with rtp_state.lock:
            marker = 0x80 if i == 0 else 0x00
            rtp_header = struct.pack("!BBHII", 0x80, 0 | marker,
                                     rtp_state.seq & 0xFFFF,
                                     rtp_state.timestamp & 0xFFFFFFFF,
                                     rtp_state.ssrc)
            rtp_state.seq += 1
            rtp_state.timestamp += CHUNK_SIZE
        try:
            rtp_sock.sendto(rtp_header + chunk, (remote_ip, remote_port))
        except OSError:
            break

        # Record outbound audio
        state.recorder.record_outbound(chunk)

        time.sleep(0.02)

    state.is_speaking = False
    state.speaking_ended_at = time.time()


# ============================================================
# RTP Receive → Soniox STT
# ============================================================

def rtp_receive_thread(rtp_sock, soniox_ws, state):
    """Receive RTP audio and forward to Soniox for transcription."""
    print("[RTP-RX] Listening for incoming audio...")
    rtp_sock.settimeout(0.1)

    while state.call_active:
        try:
            data, addr = rtp_sock.recvfrom(2048)
        except socket.timeout:
            continue
        except OSError:
            break

        if len(data) < 12:
            continue

        # Extract RTP payload (skip 12-byte header)
        payload = data[12:]
        if len(payload) == 0:
            continue

        # Record inbound audio
        state.recorder.record_inbound(payload)

        # Convert u-law to 16-bit PCM for Soniox
        try:
            pcm_data = audioop.ulaw2lin(payload, 2)
        except audioop.error:
            continue

        # Check audio energy for voice activity detection
        try:
            rms = audioop.rms(pcm_data, 2)
        except audioop.error:
            rms = 0

        VOICE_THRESHOLD = 500
        INTERRUPT_THRESHOLD = 2000  # Much louder = definitely the user, not echo
        ECHO_COOLDOWN = 0.8  # Ignore audio for this long after AI stops speaking

        # Suppress VAD during AI speech and echo cooldown period
        in_echo_zone = state.is_speaking or (time.time() - state.speaking_ended_at < ECHO_COOLDOWN)

        # Allow interruption even during echo zone if audio is very loud
        # But not during the grace period at the start of speech
        in_grace_period = state.is_speaking and (time.time() - state.speaking_started_at < INTERRUPT_GRACE_PERIOD)
        if rms > INTERRUPT_THRESHOLD and state.is_speaking and not in_grace_period:
            print(f"[VAD] User interrupted AI! (rms={rms})")
            state.interrupt_flag = True
            state.user_speaking = True
            state.silence_start = None
        elif rms > VOICE_THRESHOLD and not in_echo_zone:
            # User is speaking (normal detection outside echo zone)
            if not state.user_speaking:
                state.user_speaking = True
                print(f"[VAD] User started speaking (rms={rms})")
            state.silence_start = None
        else:
            if state.user_speaking and not in_echo_zone:
                if state.silence_start is None:
                    state.silence_start = time.time()
                elif time.time() - state.silence_start > 1.0:
                    state.user_speaking = False
                    state.silence_start = None

        # Forward PCM to Soniox — skip during AI speech + echo cooldown
        # But if user is interrupting, forward anyway so STT captures it
        if not in_echo_zone or state.interrupt_flag:
            try:
                soniox_ws.send(pcm_data)
            except Exception as e:
                print(f"[RTP-RX] Soniox send error: {e}")
                break

    print("[RTP-RX] Stopped")


# ============================================================
# Soniox STT → Gemini LLM → Cartesia TTS (streaming) → RTP pipeline
# ============================================================

def llm_respond(user_text, rtp_sock, remote_ip, remote_port, state, rtp_state, cartesia_ws):
    """Stream Gemini text → Cartesia continuations → RTP audio, all concurrently."""
    if not user_text.strip():
        state.llm_busy = False
        return

    state.llm_busy = True
    state.interrupt_flag = False
    print(f"[LLM] User said: {user_text}")

    state.conversation.append({"role": "user", "content": user_text})
    messages = [{"role": "system", "content": state.system_prompt}] + state.conversation[-20:]

    full_response = ""
    tts_buffer = ""           # Buffer to catch [HANGUP] split across chunks
    audio_q = queue.Queue()  # PCM audio chunks from Cartesia → RTP sender
    tts_ctx = cartesia_ws.context()

    # --- Thread 1: Receive audio from Cartesia and push to queue ---
    audio_chunk_count = 0

    def audio_receiver():
        nonlocal audio_chunk_count
        try:
            for output in tts_ctx.receive():
                if state.interrupt_flag or not state.call_active:
                    break
                if hasattr(output, 'audio') and output.audio:
                    audio_chunk_count += 1
                    audio_q.put(output.audio)
        except Exception as e:
            print(f"[TTS] Receive error: {e}")
        print(f"[TTS] Audio receiver done — {audio_chunk_count} chunks received")
        audio_q.put(None)  # Sentinel: no more audio

    recv_thread = threading.Thread(target=audio_receiver, daemon=True)
    recv_thread.start()

    # --- Thread 2: Send RTP packets from audio queue ---
    spoken_duration = 0.0  # Track how much audio was actually sent
    rtp_done_event = threading.Event()

    def rtp_sender():
        nonlocal spoken_duration
        try:
            pcm_buffer = b""
            CHUNK_SAMPLES = 160  # 20ms at 8kHz
            CHUNK_BYTES_44K = CHUNK_SAMPLES * 2 * (CARTESIA_SAMPLE_RATE // 8000)  # Proportional input needed
            ratecv_state = None

            while True:
                try:
                    pcm_chunk = audio_q.get(timeout=5)
                except queue.Empty:
                    break
                if pcm_chunk is None:
                    break
                if state.interrupt_flag or not state.call_active:
                    break

                pcm_buffer += pcm_chunk

                # Process in chunks large enough for clean resampling
                PROCESS_SIZE = CARTESIA_SAMPLE_RATE * 2 // 10  # ~100ms of 44.1kHz PCM16
                while len(pcm_buffer) >= PROCESS_SIZE:
                    if state.interrupt_flag or not state.call_active:
                        return

                    segment = pcm_buffer[:PROCESS_SIZE]
                    pcm_buffer = pcm_buffer[PROCESS_SIZE:]

                    pcm8k, ratecv_state = audioop.ratecv(segment, 2, 1, CARTESIA_SAMPLE_RATE, 8000, ratecv_state)
                    ulaw = audioop.lin2ulaw(pcm8k, 2)

                    if not state.is_speaking:
                        state.is_speaking = True
                        state.speaking_started_at = time.time()

                    for j in range(0, len(ulaw), CHUNK_SAMPLES):
                        if state.interrupt_flag or not state.call_active:
                            return
                        pkt = ulaw[j:j + CHUNK_SAMPLES]
                        if len(pkt) < CHUNK_SAMPLES:
                            pkt += b'\xff' * (CHUNK_SAMPLES - len(pkt))
                        with rtp_state.lock:
                            rtp_header = struct.pack("!BBHII", 0x80, 0,
                                                     rtp_state.seq & 0xFFFF,
                                                     rtp_state.timestamp & 0xFFFFFFFF,
                                                     rtp_state.ssrc)
                            rtp_state.seq += 1
                            rtp_state.timestamp += CHUNK_SAMPLES
                        try:
                            rtp_sock.sendto(rtp_header + pkt, (remote_ip, remote_port))
                        except OSError:
                            return
                        state.recorder.record_outbound(pkt)
                        spoken_duration += 0.02
                        time.sleep(0.02)

            # Flush remaining buffer
            if pcm_buffer and not state.interrupt_flag and state.call_active:
                pcm8k, _ = audioop.ratecv(pcm_buffer, 2, 1, CARTESIA_SAMPLE_RATE, 8000, ratecv_state)
                ulaw = audioop.lin2ulaw(pcm8k, 2)
                for j in range(0, len(ulaw), CHUNK_SAMPLES):
                    if state.interrupt_flag or not state.call_active:
                        return
                    pkt = ulaw[j:j + CHUNK_SAMPLES]
                    if len(pkt) < CHUNK_SAMPLES:
                        pkt += b'\xff' * (CHUNK_SAMPLES - len(pkt))
                    with rtp_state.lock:
                        rtp_header = struct.pack("!BBHII", 0x80, 0,
                                                 rtp_state.seq & 0xFFFF,
                                                 rtp_state.timestamp & 0xFFFFFFFF,
                                                 rtp_state.ssrc)
                        rtp_state.seq += 1
                        rtp_state.timestamp += CHUNK_SAMPLES
                    try:
                        rtp_sock.sendto(rtp_header + pkt, (remote_ip, remote_port))
                    except OSError:
                        return
                    state.recorder.record_outbound(pkt)
                    spoken_duration += 0.02
                    time.sleep(0.02)
        finally:
            rtp_done_event.set()
            print(f"[RTP-TX] Finished sending {spoken_duration:.1f}s of audio")

    rtp_thread = threading.Thread(target=rtp_sender, daemon=True)
    rtp_thread.start()

    # --- Main: Stream Gemini text → Cartesia continuations ---
    try:
        with httpx.Client(timeout=30) as client:
            with client.stream(
                "POST",
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": LLM_MODEL,
                    "messages": messages,
                    "stream": True,
                    "max_tokens": 300,
                },
            ) as resp:
                for line in resp.iter_lines():
                    if state.interrupt_flag or not state.call_active:
                        break
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            full_response += content
                            tts_buffer += content

                            # Check if [HANGUP] is fully present
                            if HANGUP_TOKEN in tts_buffer:
                                # Send everything before the token
                                before = tts_buffer.split(HANGUP_TOKEN)[0]
                                if before.strip():
                                    try:
                                        tts_ctx.send(
                                            model_id="sonic-3",
                                            transcript=before,
                                            voice={"mode": "id", "id": CARTESIA_VOICE_ID},
                                            output_format={"container": "raw", "encoding": "pcm_s16le", "sample_rate": CARTESIA_SAMPLE_RATE},
                                            language="de",
                                            continue_=True,
                                        )
                                    except Exception as e:
                                        print(f"[TTS] Send error: {e}")
                                tts_buffer = ""
                                break

                            # Hold back enough chars to catch a partial "[HANGUP"
                            safe_len = len(tts_buffer) - len(HANGUP_TOKEN) + 1
                            if safe_len > 0:
                                to_send = tts_buffer[:safe_len]
                                tts_buffer = tts_buffer[safe_len:]
                                try:
                                    tts_ctx.send(
                                        model_id="sonic-3",
                                        transcript=to_send,
                                        voice={"mode": "id", "id": CARTESIA_VOICE_ID},
                                        output_format={"container": "raw", "encoding": "pcm_s16le", "sample_rate": CARTESIA_SAMPLE_RATE},
                                        language="de",
                                        continue_=True,
                                    )
                                except Exception as e:
                                    print(f"[TTS] Send error: {e}")
                                    break
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue

    except Exception as e:
        print(f"[LLM] Error: {e}")

    # Flush any remaining buffered text to TTS (strip [HANGUP] if present)
    if tts_buffer:
        flush_text = tts_buffer.replace(HANGUP_TOKEN, "").strip()
        if flush_text:
            try:
                tts_ctx.send(
                    model_id="sonic-3",
                    transcript=flush_text,
                    voice={"mode": "id", "id": CARTESIA_VOICE_ID},
                    output_format={"container": "raw", "encoding": "pcm_s16le", "sample_rate": CARTESIA_SAMPLE_RATE},
                    language="de",
                    continue_=True,
                )
            except Exception as e:
                print(f"[TTS] Flush error: {e}")

    # Signal Cartesia that no more text is coming
    try:
        tts_ctx.no_more_inputs()
    except Exception:
        pass

    # Wait for audio to finish playing
    recv_thread.join(timeout=15)
    rtp_thread.join(timeout=15)
    rtp_done_event.wait(timeout=15)  # Ensure rtp_sender truly finished
    print(f"[RTP-TX] All audio sent ({spoken_duration:.1f}s)")

    # Fallback: if streaming TTS produced no audio, re-synthesize and play
    if spoken_duration < 0.1 and full_response.replace(HANGUP_TOKEN, '').strip() and not state.interrupt_flag and state.call_active:
        fallback_text = full_response.replace(HANGUP_TOKEN, '').strip()
        print(f"[TTS] Streaming produced no audio — falling back to non-streaming for: {fallback_text[:60]}...")
        try:
            fallback_audio = cartesia_tts_ulaw(fallback_text, cartesia_ws, None)
            if fallback_audio:
                rtp_send_thread(rtp_sock, remote_ip, remote_port, fallback_audio, state, rtp_state)
        except Exception as e:
            print(f"[TTS] Fallback error: {e}")

    state.is_speaking = False
    state.speaking_ended_at = time.time()

    # Check if AI wants to hang up
    hangup = "[HANGUP]" in full_response
    clean_response = full_response.replace("[HANGUP]", "").strip()

    # Record conversation history
    if clean_response:
        if state.interrupt_flag:
            state.conversation.append({"role": "assistant", "content": f"{clean_response} [user interrupted]"})
            print(f"[LLM] Interrupted. Response was: {clean_response}")
        else:
            state.conversation.append({"role": "assistant", "content": clean_response})
            print(f"[LLM] Response: {clean_response}")

    state.llm_busy = False

    if hangup and not state.interrupt_flag:
        # All RTP packets have been sent (confirmed by rtp_done_event).
        # Wait briefly for the remote phone's jitter buffer to drain before BYE.
        time.sleep(0.5)
        print("[AGENT] AI decided to hang up")
        state.hangup_requested = True
        state.call_active = False
        if state.sip_sock and state.sip_call_info:
            try:
                local_ip, call_id, tag, cseq, to_header = state.sip_call_info
                sip_bye(state.sip_sock, local_ip, call_id, tag, cseq, to_header, state.call_number)
            except OSError:
                pass  # Socket already closed by run_call cleanup


def stt_process_thread(soniox_ws, rtp_sock, remote_ip, remote_port, state, rtp_state, cartesia_ws):
    """Read Soniox STT results and trigger LLM responses."""
    print("[STT] Processing thread started")
    final_text = ""
    last_final_time = time.time()

    while state.call_active:
        try:
            msg = soniox_ws.recv(timeout=0.2)
        except TimeoutError:
            # Check if we have accumulated text and silence has passed
            if final_text.strip() and not state.user_speaking:
                elapsed = time.time() - last_final_time
                if elapsed > 1.5:
                    if state.llm_busy or state.is_speaking:
                        # LLM/TTS still running — keep text buffered, retry next cycle
                        continue
                    # User stopped speaking, process the text
                    text = final_text.strip()
                    final_text = ""
                    state.llm_busy = True  # Set BEFORE spawning to prevent races
                    # Run LLM in a separate thread so STT keeps processing
                    threading.Thread(target=llm_respond,
                                     args=(text, rtp_sock, remote_ip, remote_port,
                                           state, rtp_state, cartesia_ws),
                                     daemon=True).start()
            continue
        except Exception as e:
            print(f"[STT] WebSocket error: {e}")
            break

        if isinstance(msg, str):
            try:
                result = json.loads(msg)
            except json.JSONDecodeError:
                continue

            if result.get("error_message"):
                print(f"[STT] Error: {result['error_message']}")
                break

            tokens = result.get("tokens", [])
            for token in tokens:
                if token.get("is_final"):
                    text = token["text"]
                    # Strip Soniox end-of-speech markers
                    if text.strip() in ("<end>", "<END>"):
                        continue
                    final_text += text
                    last_final_time = time.time()

            # Show non-final text for debugging (use full line to avoid garbled output)
            non_final = "".join(t["text"] for t in tokens if not t.get("is_final"))
            if non_final:
                print(f"[STT] (partial): {non_final.strip()[:60]}          ", end="\r", flush=True)

            if result.get("finished"):
                print("[STT] Stream finished")
                break

    print("[STT] Processing thread stopped")


# ============================================================
# SIP keepalive & BYE detection
# ============================================================

def sip_listener_thread(sip_sock, local_ip, call_id, tag, to_header, state):
    """Listen for BYE from remote end."""
    sip_sock.settimeout(1)
    while state.call_active:
        try:
            data, addr = sip_sock.recvfrom(8192)
        except socket.timeout:
            continue
        except OSError:
            break

        text = data.decode(errors="replace")
        if text.startswith("BYE "):
            print("[SIP] Remote sent BYE — call ended")
            state.call_active = False
            # Send 200 OK for BYE
            lines = text.split("\r\n")
            via = ""
            from_h = ""
            to_h = ""
            cid = ""
            cseq = ""
            for line in lines:
                if line.startswith("Via:"):
                    via = line
                elif line.startswith("From:"):
                    from_h = line
                elif line.startswith("To:"):
                    to_h = line
                elif line.startswith("Call-ID:"):
                    cid = line
                elif line.startswith("CSeq:"):
                    cseq = line

            ok_msg = (
                f"SIP/2.0 200 OK\r\n"
                f"{via}\r\n"
                f"{from_h}\r\n"
                f"{to_h}\r\n"
                f"{cid}\r\n"
                f"{cseq}\r\n"
                f"Content-Length: 0\r\n\r\n"
            )
            sip_sock.sendto(ok_msg.encode(), addr)
            break

    print("[SIP] Listener stopped")


# ============================================================
# Post-call summarization
# ============================================================

def summarize_call(state):
    """Use the LLM to summarize the call and determine if the objective was met."""
    transcript = state.conversation
    if not transcript:
        return "No conversation took place.", False

    summary_messages = [
        {
            "role": "system",
            "content": (
                "Du bist ein Analyst. Du bekommst ein Gesprächsprotokoll eines Telefonats "
                "und den Auftrag, der dem Anrufer gegeben wurde. "
                "Antworte auf Englisch mit genau diesem JSON-Format (kein Markdown, nur rohes JSON):\n"
                '{"summary": "...", "objective_completed": true/false}\n\n'
                "- summary: A concise English summary of what happened in the call (2-4 sentences).\n"
                "- objective_completed: true if the objective was achieved, false otherwise."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Objective: {state.objective}\n\n"
                f"Transcript:\n" +
                "\n".join(f"{m['role'].upper()}: {m['content']}" for m in transcript)
            ),
        },
    ]

    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": LLM_MODEL,
                    "messages": summary_messages,
                    "max_tokens": 500,
                },
            )
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            # Parse JSON from response (strip markdown fences if present)
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(content)
            return result.get("summary", ""), result.get("objective_completed", False)
    except Exception as e:
        print(f"[SUMMARY] Error: {e}")
        return "Failed to generate summary.", False


# ============================================================
# Core call logic
# ============================================================

def run_call(call_number: str, objective: str) -> CallResult:
    """Execute a full phone call and return structured results."""
    state = AgentState(call_number, objective)
    rtp_state = RTPState()

    local_ip = get_local_ip()
    print(f"[*] Local IP: {local_ip}")

    # Create SIP socket
    sip_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sip_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sip_sock.bind((local_ip, LOCAL_SIP_PORT))

    # Create RTP socket
    rtp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rtp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    rtp_sock.bind((local_ip, RTP_PORT))

    try:
        # Step 1: Initialize Cartesia TTS early (needed for pre-synthesis)
        print("[TTS] Connecting to Cartesia Sonic...")
        cartesia_client = Cartesia(api_key=CARTESIA_API_KEY)
        cartesia_ws = cartesia_client.tts.websocket()
        print("[TTS] Cartesia connected")

        # Step 2: Pre-generate opening line + synthesize audio BEFORE calling
        print("[*] Generating opening line...")
        opening_messages = [
            {"role": "system", "content": state.system_prompt},
            {"role": "user", "content": (
                "Der Angerufene hat gerade abgenommen. "
                "Stell dich kurz als KI-Assistent von Jaron Kramer vor "
                "und sag direkt, warum du anrufst. "
                "Maximal 2 Sätze. Kein [HANGUP]."
            )},
        ]
        try:
            with httpx.Client(timeout=15) as client:
                resp = client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={"model": LLM_MODEL, "messages": opening_messages, "max_tokens": 100},
                )
                greeting = resp.json()["choices"][0]["message"]["content"].strip()
                greeting = greeting.replace(HANGUP_TOKEN, "").strip()
        except Exception as e:
            print(f"[LLM] Opening line error: {e}")
            greeting = "Hallo, guten Tag!"

        print(f"[*] Opening: {greeting}")
        print("[*] Pre-synthesizing greeting audio...")
        greeting_audio = cartesia_tts_ulaw(greeting, cartesia_ws, cartesia_client)
        print(f"[*] Greeting audio ready ({len(greeting_audio)} bytes)")

        # Reconnect Cartesia WS — the .send() call above consumed the connection,
        # we need a fresh one for the streaming .context() API used during the call
        try:
            cartesia_ws.close()
        except Exception:
            pass
        cartesia_ws = cartesia_client.tts.websocket()
        print("[TTS] Cartesia reconnected for streaming")

        # Step 3: Register
        if not sip_register(sip_sock, local_ip):
            return CallResult(
                status="failed", summary="SIP registration failed.",
                objective_completed=False, transcript=[],
            )

        # Step 4: Call
        result = sip_invite(sip_sock, local_ip, call_number)
        if result is None:
            return CallResult(
                status="failed", summary="Call setup failed. No answer or rejected.",
                objective_completed=False, transcript=[],
            )

        call_id, tag, cseq, remote_rtp_ip, remote_rtp_port, to_header = result
        state.sip_sock = sip_sock
        state.sip_call_info = (local_ip, call_id, tag, cseq, to_header)
        print("[*] Call established! Starting AI agent...")

        # Step 5: Connect Soniox + start all threads BEFORE greeting
        print("[STT] Connecting to Soniox...")
        soniox_ws = ws_connect("wss://stt-rt.soniox.com/transcribe-websocket")

        soniox_config = {
            "api_key": SONIOX_API_KEY,
            "model": "stt-rt-preview",
            "audio_format": "pcm_s16le",
            "sample_rate": 8000,
            "num_channels": 1,
            "language_hints": ["de", "en"],
            "enable_endpoint_detection": True,
        }
        soniox_ws.send(json.dumps(soniox_config))
        print("[STT] Soniox connected and configured")

        threads = []

        t_rtp_rx = threading.Thread(target=rtp_receive_thread,
                                     args=(rtp_sock, soniox_ws, state), daemon=True)
        t_rtp_rx.start()
        threads.append(t_rtp_rx)

        t_keepalive = threading.Thread(target=rtp_keepalive_thread,
                                       args=(rtp_sock, remote_rtp_ip, remote_rtp_port,
                                             state, rtp_state), daemon=True)
        t_keepalive.start()
        threads.append(t_keepalive)

        t_stt = threading.Thread(target=stt_process_thread,
                                 args=(soniox_ws, rtp_sock, remote_rtp_ip, remote_rtp_port,
                                       state, rtp_state, cartesia_ws), daemon=True)
        t_stt.start()
        threads.append(t_stt)

        t_sip = threading.Thread(target=sip_listener_thread,
                                 args=(sip_sock, local_ip, call_id, tag, to_header, state), daemon=True)
        t_sip.start()
        threads.append(t_sip)

        # Start recording
        state.recorder.start()

        # Step 6: Play greeting IMMEDIATELY (audio was pre-synthesized, threads already listening)
        rtp_send_thread(rtp_sock, remote_rtp_ip, remote_rtp_port, greeting_audio, state, rtp_state)
        state.conversation.append({"role": "assistant", "content": greeting})

        print("[*] AI Agent is live!")
        print("=" * 60)

        # Wait for call to end
        while state.call_active:
            time.sleep(0.5)

        # Wait for any in-flight LLM/TTS thread to finish before closing sockets
        for _ in range(20):  # Up to 10s
            if not state.llm_busy:
                break
            time.sleep(0.5)

        # Cleanup
        print("[*] Shutting down...")
        try:
            soniox_ws.send(b"")
            time.sleep(0.5)
            soniox_ws.close()
        except Exception:
            pass
        try:
            cartesia_ws.close()
        except Exception:
            pass

        if not state.hangup_requested:
            sip_bye(sip_sock, local_ip, call_id, tag, cseq, to_header, call_number)

        for t in threads:
            t.join(timeout=3)

    finally:
        rtp_sock.close()
        sip_sock.close()

    # Save call recording
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_number = call_number.replace("+", "").replace(" ", "")
    recording_path = os.path.join(RECORDINGS_DIR, f"{timestamp}_{safe_number}.wav")
    try:
        state.recorder.save(recording_path)
    except Exception as e:
        print(f"[REC] Error saving recording: {e}")

    print("[*] Call finished. Generating summary...")
    summary, objective_completed = summarize_call(state)
    print(f"[*] Summary: {summary}")
    print(f"[*] Objective completed: {objective_completed}")

    return CallResult(
        status="completed",
        summary=summary,
        objective_completed=objective_completed,
        transcript=state.conversation,
    )


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(title="AI Voice Agent API")


@app.post("/call", response_model=CallResult)
async def make_call(req: CallRequest):
    """Place a phone call with a given objective and return the result."""
    if not OPENROUTER_KEY or not SONIOX_API_KEY or not CARTESIA_API_KEY:
        raise HTTPException(status_code=500, detail="Missing API keys in .env")

    acquired = call_lock.acquire(blocking=False)
    if not acquired:
        raise HTTPException(status_code=409, detail="A call is already in progress. Try again later.")

    try:
        print(f"\n[API] New call request: number={req.number}, objective={req.objective}")
        # Run the blocking call in a thread so FastAPI stays responsive
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_call, req.number, req.objective)
        return result
    finally:
        call_lock.release()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
