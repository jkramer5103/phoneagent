#!/usr/bin/env python3
"""
AI Voice Agent REST API (OpenAI Realtime edition):
POST /call with a phone number and objective.
Calls the number via SIP/RTP, pipes audio bidirectionally through the
OpenAI Realtime API (speech-to-speech), which handles STT + LLM + TTS
in a single WebSocket.  Returns a structured summary when the call ends.
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
import base64
import queue
import threading
import asyncio
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from websockets.sync.client import connect as ws_connect

load_dotenv()

# --- Configuration ---
SIP_SERVER = os.getenv("SIP_SERVER", "speedport.ip")
SIP_PORT = int(os.getenv("SIP_PORT", "5060"))
SIP_URI_USER = os.getenv("SIP_URI_USER", "**71")
AUTH_USER = os.getenv("AUTH_USER", "nutzer-1@speedport.ip")
AUTH_PASS = os.getenv("AUTH_PASS", "")
LOCAL_SIP_PORT = int(os.getenv("LOCAL_SIP_PORT", "5060"))
RTP_PORT = int(os.getenv("RTP_PORT", "16384"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemini-3-flash-preview")
REALTIME_MODEL = os.getenv("REALTIME_MODEL", "gpt-4o-realtime-preview")

HANGUP_TOKEN = os.getenv("HANGUP_TOKEN", "[HANGUP]")
RECORDINGS_DIR = os.getenv("RECORDINGS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings"))


# ============================================================
# Call Recording
# ============================================================

class CallRecorder:
    """Records both sides of a call by capturing raw G.711 RTP audio with timestamps."""

    def __init__(self):
        self.start_time = None
        self.codec = "PCMU"  # Set by caller after codec negotiation
        self.inbound = []   # (offset_samples, g711_bytes) from remote party
        self.outbound = []  # (offset_samples, g711_bytes) from agent
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

        # Decode G.711 chunks into PCM buffers at their correct offsets
        for offset, data in inbound:
            try:
                pcm = g711_to_pcm(data, self.codec)
                byte_offset = offset * 2
                end = byte_offset + len(pcm)
                if end <= len(pcm_in):
                    pcm_in[byte_offset:end] = pcm
            except audioop.error:
                pass

        for offset, data in outbound:
            try:
                pcm = g711_to_pcm(data, self.codec)
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
        self.call_active = True
        self.conversation = []
        self.lock = threading.Lock()
        self.hangup_requested = False
        self.sip_sock = None
        self.sip_call_info = None
        self.codec = "PCMU"  # Negotiated codec: "PCMU" (u-law) or "PCMA" (A-law)
        self.recorder = CallRecorder()
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self):
        return (
            "Du bist ein KI-Telefonassistent von Jaron Kramer. "
            "Du rufst im Auftrag von Jaron an. "
            "Stell dich als KI-Assistent von Jaron Kramer vor, wenn es passt.\n\n"
            f"Dein Auftrag für diesen Anruf: {self.objective}\n\n"
            "Regeln:\n"
            "- WICHTIG: Nenne sofort im allerersten Satz den Grund deines Anrufs bzw. dein Anliegen. "
            "Stell dich vor UND sag direkt, worum es geht — beides in einem Atemzug. "
            "Warte NIEMALS darauf, dass der andere dich fragt, warum du anrufst.\n"
            "- Halte deine Antworten kurz und natürlich — normalerweise 1-2 Sätze.\n"
            "- Du sprichst Deutsch.\n"
            "- Sei höflich, direkt und zielgerichtet.\n"
            "- DU bist der Anrufer. DU führst das Gespräch.\n"
            "- Reagiere natürlich auf Rückfragen und arbeite auf dein Ziel hin.\n"
            "- Wenn dein Anliegen erledigt ist oder das Gespräch zu Ende ist, "
            "verabschiede dich kurz und füge am Ende das Token [HANGUP] hinzu.\n"
            "- Beispiel: 'Alles klar, vielen Dank! Tschüss! [HANGUP]'"
        )

# Lock to prevent concurrent calls (single SIP port)
call_lock = threading.Lock()


# ============================================================
# SIP Helpers (reused from agent.py)
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

def parse_sip_request(data):
    """Parse an incoming SIP request (INVITE, BYE, ACK, etc.)."""
    text = data.decode(errors="replace")
    parts = text.split("\r\n\r\n", 1)
    header_section = parts[0]
    body = parts[1] if len(parts) > 1 else ""
    lines = header_section.split("\r\n")
    request_line = lines[0]
    method = request_line.split(" ", 1)[0]
    headers = {}
    for line in lines[1:]:
        if ":" in line:
            key, val = line.split(":", 1)
            headers[key.strip()] = val.strip()
    return method, headers, body, request_line


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


def parse_negotiated_codec(sdp_body):
    """Parse the first audio codec from a remote SDP answer.
    Returns 'PCMA' if A-law (payload 8) is first, 'PCMU' otherwise."""
    for line in sdp_body.replace("\r\n", "\n").split("\n"):
        if line.startswith("m=audio "):
            # m=audio <port> RTP/AVP <pt1> <pt2> ...
            parts = line.split()
            if len(parts) >= 4:
                first_pt = parts[3]
                if first_pt == "8":
                    return "PCMA"
                elif first_pt == "0":
                    return "PCMU"
    return "PCMU"  # Default fallback


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
        f"m=audio {RTP_PORT} RTP/AVP 8 0 101\r\n"
        f"a=rtpmap:8 PCMA/8000\r\n"
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

    codec = parse_negotiated_codec(body)
    print(f"[SIP] Call answered! Remote RTP: {remote_rtp_ip}:{remote_rtp_port}, codec: {codec}")

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

    return call_id, tag, cseq, remote_rtp_ip, remote_rtp_port, to_header, codec


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
# G.711 Helpers
# ============================================================

def pcm_to_g711(pcm_data, codec="PCMU"):
    """Encode 16-bit PCM to G.711 (u-law or A-law)."""
    if codec == "PCMA":
        return audioop.lin2alaw(pcm_data, 2)
    return audioop.lin2ulaw(pcm_data, 2)


def g711_to_pcm(g711_data, codec="PCMU"):
    """Decode G.711 (u-law or A-law) to 16-bit PCM."""
    if codec == "PCMA":
        return audioop.alaw2lin(g711_data, 2)
    return audioop.ulaw2lin(g711_data, 2)


def g711_silence(codec="PCMU"):
    """Return the silence byte for the given codec."""
    if codec == "PCMA":
        return b'\xd5'  # A-law silence
    return b'\xff'      # u-law silence


def alaw_to_ulaw(alaw_data):
    """Convert A-law to u-law via PCM intermediate."""
    pcm = audioop.alaw2lin(alaw_data, 2)
    return audioop.lin2ulaw(pcm, 2)


def ulaw_to_alaw(ulaw_data):
    """Convert u-law to A-law via PCM intermediate."""
    pcm = audioop.ulaw2lin(ulaw_data, 2)
    return audioop.lin2alaw(pcm, 2)


# ============================================================
# RTP State & Helpers
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
    pt = 8 if state.codec == "PCMA" else 0
    silence = g711_silence(state.codec) * CHUNK_SIZE

    while state.call_active:
        if not state.is_speaking:
            with rtp_state.lock:
                rtp_header = struct.pack("!BBHII", 0x80, pt,
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


# ============================================================
# OpenAI Realtime API ↔ RTP Bridge
# ============================================================

REALTIME_WS_URL = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"


def openai_realtime_bridge(rtp_sock, remote_rtp_ip, remote_rtp_port, state, rtp_state):
    """
    Bridge between RTP audio and OpenAI Realtime API.
    - Inbound RTP G.711 → base64 → input_audio_buffer.append
    - response.audio.delta → base64 decode → G.711 → RTP out
    - Handles turn detection, transcript, and hangup via server events.

    OpenAI Realtime API natively supports g711_ulaw at 8kHz.
    For PCMA (A-law) codecs, we convert at the boundary.
    """
    print("[REALTIME] Connecting to OpenAI Realtime API...")

    # Determine the audio format for OpenAI
    # OpenAI supports: pcm16 (24kHz), g711_ulaw (8kHz), g711_alaw (8kHz)
    if state.codec == "PCMA":
        openai_audio_format = "g711_alaw"
    else:
        openai_audio_format = "g711_ulaw"

    rt_ws = ws_connect(
        REALTIME_WS_URL,
        additional_headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1",
        },
    )

    # Wait for session.created
    try:
        init_msg = rt_ws.recv(timeout=10)
        init_event = json.loads(init_msg)
        print(f"[REALTIME] {init_event.get('type', 'unknown')} — session ready")
    except Exception as e:
        print(f"[REALTIME] Failed to receive session.created: {e}")
        rt_ws.close()
        state.call_active = False
        return

    # Configure session: audio format, VAD, instructions, and initial greeting
    session_config = {
        "type": "session.update",
        "session": {
            "modalities": ["text", "audio"],
            "instructions": state.system_prompt,
            "voice": "echo",
            "input_audio_format": openai_audio_format,
            "output_audio_format": openai_audio_format,
            "input_audio_transcription": {
                "model": "gpt-4o-mini-transcribe",
            },
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": 700,
            },
        },
    }
    rt_ws.send(json.dumps(session_config))
    print(f"[REALTIME] Session configured (audio: {openai_audio_format})")

    # Trigger the initial greeting — the AI should introduce itself AND state the objective
    rt_ws.send(json.dumps({
        "type": "response.create",
        "response": {
            "modalities": ["text", "audio"],
            "instructions": (
                "Der Angerufene hat gerade abgenommen. "
                "Sag in deinem ERSTEN Satz: 'Hallo, hier ist der KI-Assistent von Jaron Kramer.' "
                "Dann sag SOFORT im ZWEITEN Satz dein konkretes Anliegen, nämlich: "
                f"'{state.objective}'. "
                "Formuliere es als direkte Frage oder Aussage. "
                "Insgesamt maximal 2 Sätze. Kein [HANGUP]."
            ),
        },
    }))
    print("[REALTIME] Requested initial greeting")

    # --- Audio output queue: Realtime API → RTP ---
    audio_out_q = queue.Queue()
    audio_drained_event = threading.Event()  # Set when RTP sender finishes playing after a None sentinel

    # --- Thread: Read RTP inbound → send to Realtime API ---
    def rtp_to_realtime():
        """Forward incoming RTP G.711 audio to OpenAI Realtime API."""
        print("[RTP→RT] Forwarding inbound audio to Realtime API...")
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

            payload = data[12:]
            if len(payload) == 0:
                continue

            # Record inbound audio
            state.recorder.record_inbound(payload)

            # Base64-encode and send to OpenAI
            # If codec is PCMA, OpenAI handles g711_alaw natively
            b64_audio = base64.b64encode(payload).decode("ascii")
            try:
                rt_ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": b64_audio,
                }))
            except Exception as e:
                print(f"[RTP→RT] Send error: {e}")
                break

        print("[RTP→RT] Stopped")

    # --- Thread: Read Realtime API events → process ---
    def realtime_event_reader():
        """Read events from OpenAI Realtime API and dispatch audio/text."""
        print("[RT] Event reader started")
        current_text = ""

        while state.call_active:
            try:
                msg = rt_ws.recv(timeout=1.0)
            except TimeoutError:
                continue
            except Exception as e:
                print(f"[RT] WebSocket error: {e}")
                break

            if isinstance(msg, bytes):
                continue

            try:
                event = json.loads(msg)
            except json.JSONDecodeError:
                continue

            etype = event.get("type", "")

            if etype == "response.audio.delta":
                # Audio chunk from the model — queue for RTP sending
                audio_b64 = event.get("delta", "")
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)
                    audio_out_q.put(audio_bytes)

            elif etype == "response.audio.done":
                # Signal end of this audio response
                audio_drained_event.clear()
                audio_out_q.put(None)

            elif etype == "response.audio_transcript.delta":
                # Accumulate assistant text for transcript
                current_text += event.get("delta", "")

            elif etype == "response.audio_transcript.done":
                # Full assistant turn text
                transcript_text = event.get("transcript", current_text)
                current_text = ""
                if transcript_text:
                    print(f"[RT] Assistant: {transcript_text}")
                    with state.lock:
                        state.conversation.append({"role": "assistant", "content": transcript_text})

                    # Check for hangup token
                    if HANGUP_TOKEN in transcript_text:
                        print("[AGENT] AI decided to hang up")
                        state.hangup_requested = True
                        # Don't set call_active=False yet — let audio finish playing

            elif etype == "conversation.item.input_audio_transcription.completed":
                # User speech transcription
                user_text = event.get("transcript", "")
                if user_text and user_text.strip():
                    print(f"[RT] User: {user_text}")
                    with state.lock:
                        state.conversation.append({"role": "user", "content": user_text})

            elif etype == "input_audio_buffer.speech_started":
                print("[RT] User started speaking")
                # If we're currently speaking, the model will handle interruption

            elif etype == "input_audio_buffer.speech_stopped":
                print("[RT] User stopped speaking")

            elif etype == "response.done":
                # A full response cycle completed
                response = event.get("response", {})
                status_info = response.get("status", "")
                print(f"[RT] Response done (status={status_info})")

                # If hangup was requested, wait for audio to finish playing over RTP
                if state.hangup_requested:
                    print("[RT] Waiting for audio to finish playing before hangup...")
                    audio_drained_event.wait(timeout=15)
                    time.sleep(0.5)  # Let remote jitter buffer drain
                    state.call_active = False
                    if state.sip_sock and state.sip_call_info:
                        try:
                            local_ip, call_id, tag, cseq, to_header = state.sip_call_info
                            sip_bye(state.sip_sock, local_ip, call_id, tag, cseq, to_header, state.call_number)
                        except OSError:
                            pass

            elif etype == "error":
                error_info = event.get("error", {})
                print(f"[RT] Error: {error_info.get('message', error_info)}")

            elif etype in ("session.created", "session.updated"):
                print(f"[RT] {etype}")

            elif etype == "rate_limits.updated":
                pass  # Ignore rate limit updates

            else:
                # Log unknown events for debugging
                if etype:
                    print(f"[RT] Event: {etype}")

        print("[RT] Event reader stopped")

    # --- Thread: Send queued audio out over RTP ---
    def realtime_to_rtp():
        """Send audio from OpenAI Realtime API out over RTP."""
        print("[RT→RTP] Audio sender started")
        CHUNK_SIZE = 160  # 20ms at 8kHz
        pt = 8 if state.codec == "PCMA" else 0
        sil = g711_silence(state.codec)

        while state.call_active:
            try:
                audio_chunk = audio_out_q.get(timeout=0.5)
            except queue.Empty:
                continue

            if audio_chunk is None:
                # End of response audio
                state.is_speaking = False
                audio_drained_event.set()
                print("[RT→RTP] Audio response finished")
                continue

            if not state.is_speaking:
                state.is_speaking = True
                print("[RT→RTP] Started sending audio")

            # The audio is already in the correct G.711 format (ulaw or alaw)
            # Send in 160-byte (20ms) RTP packets
            for i in range(0, len(audio_chunk), CHUNK_SIZE):
                if not state.call_active:
                    break

                pkt = audio_chunk[i:i + CHUNK_SIZE]
                if len(pkt) < CHUNK_SIZE:
                    pkt += sil * (CHUNK_SIZE - len(pkt))

                with rtp_state.lock:
                    marker = 0x80 if i == 0 and not state.is_speaking else 0x00
                    rtp_header = struct.pack("!BBHII", 0x80, pt | marker,
                                             rtp_state.seq & 0xFFFF,
                                             rtp_state.timestamp & 0xFFFFFFFF,
                                             rtp_state.ssrc)
                    rtp_state.seq += 1
                    rtp_state.timestamp += CHUNK_SIZE
                try:
                    rtp_sock.sendto(rtp_header + pkt, (remote_rtp_ip, remote_rtp_port))
                except OSError:
                    break

                # Record outbound audio
                state.recorder.record_outbound(pkt)
                time.sleep(0.02)

        print("[RT→RTP] Audio sender stopped")

    # Start all bridge threads
    threads = []

    t_rtp_in = threading.Thread(target=rtp_to_realtime, daemon=True)
    t_rtp_in.start()
    threads.append(t_rtp_in)

    t_events = threading.Thread(target=realtime_event_reader, daemon=True)
    t_events.start()
    threads.append(t_events)

    t_rtp_out = threading.Thread(target=realtime_to_rtp, daemon=True)
    t_rtp_out.start()
    threads.append(t_rtp_out)

    # Wait for call to end
    while state.call_active:
        time.sleep(0.5)

    # Cleanup
    print("[REALTIME] Shutting down bridge...")
    try:
        rt_ws.close()
    except Exception:
        pass

    for t in threads:
        t.join(timeout=3)

    print("[REALTIME] Bridge stopped")


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
# Post-call summarization (uses OpenRouter, same as original)
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
    """Execute a full phone call using OpenAI Realtime API and return structured results."""
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
        # Step 1: Register
        if not sip_register(sip_sock, local_ip):
            return CallResult(
                status="failed", summary="SIP registration failed.",
                objective_completed=False, transcript=[],
            )

        # Step 2: Call
        result = sip_invite(sip_sock, local_ip, call_number)
        if result is None:
            return CallResult(
                status="failed", summary="Call setup failed. No answer or rejected.",
                objective_completed=False, transcript=[],
            )

        call_id, tag, cseq, remote_rtp_ip, remote_rtp_port, to_header, codec = result
        state.codec = codec
        state.recorder.codec = codec
        state.sip_sock = sip_sock
        state.sip_call_info = (local_ip, call_id, tag, cseq, to_header)
        print(f"[*] Call established! Codec: {codec}. Starting OpenAI Realtime bridge...")

        # Start recording
        state.recorder.start()

        # Step 3: Start SIP listener (BYE detection)
        t_sip = threading.Thread(target=sip_listener_thread,
                                 args=(sip_sock, local_ip, call_id, tag, to_header, state), daemon=True)
        t_sip.start()

        # Step 4: Start RTP keepalive
        t_keepalive = threading.Thread(target=rtp_keepalive_thread,
                                       args=(rtp_sock, remote_rtp_ip, remote_rtp_port,
                                             state, rtp_state), daemon=True)
        t_keepalive.start()

        # Step 5: Run the OpenAI Realtime ↔ RTP bridge (blocks until call ends)
        print("[*] AI Agent is live (OpenAI Realtime)!")
        print("=" * 60)
        openai_realtime_bridge(rtp_sock, remote_rtp_ip, remote_rtp_port, state, rtp_state)

        # Cleanup
        print("[*] Shutting down...")
        if not state.hangup_requested:
            sip_bye(sip_sock, local_ip, call_id, tag, cseq, to_header, call_number)

        t_sip.join(timeout=3)
        t_keepalive.join(timeout=3)

    finally:
        rtp_sock.close()
        sip_sock.close()

    # Re-register the incoming listener (outgoing call's REGISTER overwrites it)
    re_register_incoming()

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
# Incoming Call Handling
# ============================================================

INCOMING_MESSAGE = (
    "Diese Rufnummer gehört dem KI-Agenten von Jaron Kramer, "
    "und sie ist bisher nicht programmiert, eingehende Anrufe anzunehmen."
)


def handle_incoming_call(sip_sock, local_ip, invite_data, addr):
    """Answer an incoming SIP INVITE, play a message via OpenAI TTS, then hang up."""
    method, headers, body, request_line = parse_sip_request(invite_data)

    via = ""
    from_h = ""
    to_h = ""
    cid = ""
    cseq = ""
    text = invite_data.decode(errors="replace")
    for line in text.split("\r\n"):
        if line.startswith("Via:") and not via:
            via = line
        elif line.startswith("From:"):
            from_h = line
        elif line.startswith("To:"):
            to_h = line
        elif line.startswith("Call-ID:"):
            cid = line
        elif line.startswith("CSeq:"):
            cseq = line

    call_id = headers.get("Call-ID", "")
    contact = headers.get("Contact", "")

    # Add our tag to the To header
    our_tag = gen_tag()
    if ";tag=" not in to_h:
        to_h = to_h + f";tag={our_tag}"

    # Parse remote SDP for RTP info and negotiate codec
    remote_rtp_ip = None
    remote_rtp_port = None
    sdp_lines = body.replace("\r\n", "\n").split("\n")
    for line in sdp_lines:
        if line.startswith("c=IN IP4 "):
            remote_rtp_ip = line.split()[-1].strip()
        if line.startswith("m=audio "):
            remote_rtp_port = int(line.split()[1])
            print(f"[SIP-IN] Remote SDP m-line: {line.strip()}")

    # Negotiate codec from remote SDP
    codec = parse_negotiated_codec(body)
    print(f"[SIP-IN] Negotiated codec: {codec}")

    # Send 100 Trying
    trying_msg = (
        f"SIP/2.0 100 Trying\r\n"
        f"{via}\r\n"
        f"{from_h}\r\n"
        f"{to_h}\r\n"
        f"{cid}\r\n"
        f"{cseq}\r\n"
        f"Content-Length: 0\r\n\r\n"
    )
    sip_sock.sendto(trying_msg.encode(), addr)
    print(f"[SIP-IN] Sent 100 Trying")

    # Send 180 Ringing
    ringing_msg = (
        f"SIP/2.0 180 Ringing\r\n"
        f"{via}\r\n"
        f"{from_h}\r\n"
        f"{to_h}\r\n"
        f"{cid}\r\n"
        f"{cseq}\r\n"
        f"Contact: <sip:{SIP_URI_USER}@{local_ip}:{LOCAL_SIP_PORT}>\r\n"
        f"Content-Length: 0\r\n\r\n"
    )
    sip_sock.sendto(ringing_msg.encode(), addr)
    print(f"[SIP-IN] Sent 180 Ringing")

    # Pre-synthesize the message using OpenAI TTS API
    print("[SIP-IN] Pre-synthesizing message via OpenAI TTS...")
    g711_audio = b""
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                "https://api.openai.com/v1/audio/speech",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o-mini-tts",
                    "input": INCOMING_MESSAGE,
                    "voice": "ash",
                    "response_format": "pcm",
                },
            )
            if resp.status_code == 200:
                # OpenAI TTS returns 24kHz 16-bit PCM
                pcm_24k = resp.content
                # Resample to 8kHz
                pcm_8k, _ = audioop.ratecv(pcm_24k, 2, 1, 24000, 8000, None)
                g711_audio = pcm_to_g711(pcm_8k, codec)
                # Verify audio has content
                rms = audioop.rms(pcm_8k, 2)
                print(f"[SIP-IN] TTS ready ({len(g711_audio)} bytes, {codec}, PCM RMS={rms})")
                # Log first few bytes of G.711 for debugging
                print(f"[SIP-IN] G.711 first 20 bytes: {g711_audio[:20].hex()}")
            else:
                print(f"[SIP-IN] TTS failed: {resp.status_code}")
    except Exception as e:
        print(f"[SIP-IN] TTS error: {e}")

    if not g711_audio:
        print("[SIP-IN] No audio to play")
        return

    if not remote_rtp_ip or not remote_rtp_port:
        print("[SIP-IN] No remote RTP info in SDP, cannot send audio")
        return

    # Build our SDP answer — put the negotiated codec first
    rtp_port_in = RTP_PORT + 2  # Use a different RTP port for incoming calls
    sess_id = str(random.randint(1000000, 9999999))
    if codec == "PCMA":
        codec_order = "8 0"
        codec_lines = (
            f"a=rtpmap:8 PCMA/8000\r\n"
            f"a=rtpmap:0 PCMU/8000\r\n"
        )
    else:
        codec_order = "0 8"
        codec_lines = (
            f"a=rtpmap:0 PCMU/8000\r\n"
            f"a=rtpmap:8 PCMA/8000\r\n"
        )
    sdp_body = (
        f"v=0\r\n"
        f"o=VoIPAgent {sess_id} {sess_id} IN IP4 {local_ip}\r\n"
        f"s=VoIPAgent\r\n"
        f"c=IN IP4 {local_ip}\r\n"
        f"t=0 0\r\n"
        f"m=audio {rtp_port_in} RTP/AVP {codec_order} 101\r\n"
        f"{codec_lines}"
        f"a=rtpmap:101 telephone-event/8000\r\n"
        f"a=fmtp:101 0-15\r\n"
        f"a=ptime:20\r\n"
        f"a=sendrecv\r\n"
    )

    # Create a dedicated SIP socket for this call's signaling
    # (avoids disrupting the incoming listener's recv loop)
    call_sip_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    call_sip_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    call_sip_sock.bind((local_ip, 0))  # Ephemeral port
    call_sip_port = call_sip_sock.getsockname()[1]

    # Send 200 OK with SDP — answer the call now that audio is ready
    # Use the listener socket to send the 200 OK (must come from the registered port)
    ok_msg = (
        f"SIP/2.0 200 OK\r\n"
        f"{via}\r\n"
        f"{from_h}\r\n"
        f"{to_h}\r\n"
        f"{cid}\r\n"
        f"{cseq}\r\n"
        f"Contact: <sip:{SIP_URI_USER}@{local_ip}:{LOCAL_SIP_PORT + 1}>\r\n"
        f"Content-Type: application/sdp\r\n"
        f"User-Agent: VoIPAgent/1.0\r\n"
        f"Content-Length: {len(sdp_body)}\r\n\r\n"
        f"{sdp_body}"
    )
    sip_sock.sendto(ok_msg.encode(), addr)
    print(f"[SIP-IN] Sent 200 OK — call answered")

    # Wait briefly for ACK — but don't block the listener socket
    # The ACK will arrive on the listener socket; we just proceed after a short delay
    time.sleep(0.3)
    print(f"[SIP-IN] Proceeding (ACK handled by listener)")

    print(f"[SIP-IN] Remote RTP: {remote_rtp_ip}:{remote_rtp_port}")

    # Open RTP socket for this incoming call
    rtp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rtp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    rtp_sock.bind((local_ip, rtp_port_in))

    pt = 8 if codec == "PCMA" else 0
    sil = g711_silence(codec)

    try:
        print(f"[SIP-IN] Playing message ({len(g711_audio)} bytes, {codec})...")

        # Send audio over RTP
        rtp_state = RTPState()
        CHUNK_SIZE = 160

        # Send a burst of silence packets first to open the NAT/media path
        silence = sil * CHUNK_SIZE
        for _ in range(25):  # 500ms of silence
            with rtp_state.lock:
                rtp_header = struct.pack("!BBHII", 0x80, pt,
                                         rtp_state.seq & 0xFFFF,
                                         rtp_state.timestamp & 0xFFFFFFFF,
                                         rtp_state.ssrc)
                rtp_state.seq += 1
                rtp_state.timestamp += CHUNK_SIZE
            try:
                rtp_sock.sendto(rtp_header + silence, (remote_rtp_ip, remote_rtp_port))
            except OSError:
                pass
            time.sleep(0.02)

        for i in range(0, len(g711_audio), CHUNK_SIZE):
            chunk = g711_audio[i:i + CHUNK_SIZE]
            if len(chunk) < CHUNK_SIZE:
                chunk += sil * (CHUNK_SIZE - len(chunk))
            with rtp_state.lock:
                marker = 0x80 if i == 0 else 0x00
                rtp_header = struct.pack("!BBHII", 0x80, pt | marker,
                                         rtp_state.seq & 0xFFFF,
                                         rtp_state.timestamp & 0xFFFFFFFF,
                                         rtp_state.ssrc)
                rtp_state.seq += 1
                rtp_state.timestamp += CHUNK_SIZE
            try:
                rtp_sock.sendto(rtp_header + chunk, (remote_rtp_ip, remote_rtp_port))
            except OSError:
                break
            time.sleep(0.02)

        print("[SIP-IN] Message played, hanging up")
        time.sleep(0.5)  # Let jitter buffer drain

    finally:
        rtp_sock.close()

    # Send BYE using the dedicated call socket
    cseq_num = 1
    branch = gen_branch()
    # Extract the remote URI from the From header for the BYE Request-URI
    remote_uri = ""
    from_val = headers.get("From", "")
    uri_match = re.search(r'<(sip:[^>]+)>', from_val)
    if uri_match:
        remote_uri = uri_match.group(1)
    else:
        remote_uri = f"sip:{SIP_SERVER}"

    bye_msg = (
        f"BYE {remote_uri} SIP/2.0\r\n"
        f"Via: SIP/2.0/UDP {local_ip}:{call_sip_port};branch={branch};rport\r\n"
        f"{to_h.replace('To:', 'From:')}\r\n"
        f"{from_h.replace('From:', 'To:')}\r\n"
        f"{cid}\r\n"
        f"CSeq: {cseq_num} BYE\r\n"
        f"Max-Forwards: 70\r\n"
        f"Content-Length: 0\r\n\r\n"
    )
    call_sip_sock.sendto(bye_msg.encode(), (SIP_SERVER, SIP_PORT))
    print("[SIP-IN] Sent BYE")

    # Wait for 200 OK to BYE on the dedicated socket
    call_sip_sock.settimeout(3)
    try:
        resp_data, _ = call_sip_sock.recvfrom(8192)
        status, _, _, sl = parse_sip_response(resp_data)
        print(f"[SIP-IN] BYE response: {sl}")
    except (socket.timeout, Exception):
        pass

    call_sip_sock.close()
    print("[SIP-IN] Incoming call handled")


def incoming_call_listener(sip_sock, local_ip):
    """Background loop: listen for incoming SIP INVITE and handle them."""
    print("[SIP-IN] Listening for incoming calls...")
    sip_sock.settimeout(1)
    while True:
        try:
            data, addr = sip_sock.recvfrom(8192)
        except socket.timeout:
            continue
        except OSError:
            break

        text = data.decode(errors="replace")
        if text.startswith("INVITE "):
            print(f"[SIP-IN] Incoming INVITE from {addr}")
            # Handle in a thread so the listener stays responsive
            threading.Thread(
                target=handle_incoming_call,
                args=(sip_sock, local_ip, data, addr),
                daemon=True,
            ).start()

    print("[SIP-IN] Listener stopped")


# Global incoming-call socket and thread
_incoming_sip_sock = None
_incoming_listener_thread = None


def register_incoming(sip_sock, local_ip, reg_port):
    """Register the incoming listener with the SIP server. Returns True on success."""
    tag = gen_tag()
    call_id = gen_call_id(local_ip)
    branch = gen_branch()

    msg = (
        f"REGISTER sip:{SIP_SERVER} SIP/2.0\r\n"
        f"Via: SIP/2.0/UDP {local_ip}:{reg_port};branch={branch};rport\r\n"
        f"From: <sip:{SIP_URI_USER}@{SIP_SERVER}>;tag={tag}\r\n"
        f"To: <sip:{SIP_URI_USER}@{SIP_SERVER}>\r\n"
        f"Call-ID: {call_id}\r\n"
        f"CSeq: 1 REGISTER\r\n"
        f"Contact: <sip:{SIP_URI_USER}@{local_ip}:{reg_port};transport=udp>\r\n"
        f"Max-Forwards: 70\r\n"
        f"Expires: 300\r\n"
        f"User-Agent: VoIPAgent/1.0\r\n"
        f"Content-Length: 0\r\n\r\n"
    )

    print("[SIP-IN] Sending REGISTER for incoming calls...")
    resp = sip_send_recv(sip_sock, msg, SIP_SERVER, SIP_PORT)
    if resp is None:
        print("[SIP-IN] No response to REGISTER")
        return False

    status, headers, body, status_line = parse_sip_response(resp)
    print(f"[SIP-IN] {status_line}")

    if status == 401:
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
            f"Via: SIP/2.0/UDP {local_ip}:{reg_port};branch={branch};rport\r\n"
            f"From: <sip:{SIP_URI_USER}@{SIP_SERVER}>;tag={tag}\r\n"
            f"To: <sip:{SIP_URI_USER}@{SIP_SERVER}>\r\n"
            f"Call-ID: {call_id}\r\n"
            f"CSeq: 2 REGISTER\r\n"
            f"Contact: <sip:{SIP_URI_USER}@{local_ip}:{reg_port};transport=udp>\r\n"
            f"Max-Forwards: 70\r\n"
            f"Expires: 300\r\n"
            f"Authorization: {auth_header}\r\n"
            f"User-Agent: VoIPAgent/1.0\r\n"
            f"Content-Length: 0\r\n\r\n"
        )

        print("[SIP-IN] Sending authenticated REGISTER...")
        resp = sip_send_recv(sip_sock, msg, SIP_SERVER, SIP_PORT)
        if resp:
            status, headers, body, status_line = parse_sip_response(resp)
            print(f"[SIP-IN] {status_line}")

    if status == 200:
        print("[SIP-IN] Registered for incoming calls!")
        return True
    else:
        print(f"[SIP-IN] Registration failed: {status}")
        return False


def re_register_incoming():
    """Re-register the incoming listener after an outgoing call overwrites the registration."""
    if _incoming_sip_sock is None:
        return
    try:
        local_ip = get_local_ip()
        reg_port = LOCAL_SIP_PORT + 1
        register_incoming(_incoming_sip_sock, local_ip, reg_port)
    except Exception as e:
        print(f"[SIP-IN] Re-registration error: {e}")


def start_incoming_listener():
    """Register with the SIP server and start listening for incoming calls."""
    global _incoming_sip_sock, _incoming_listener_thread

    local_ip = get_local_ip()
    print(f"[SIP-IN] Local IP: {local_ip}")

    # Use a separate socket so it doesn't conflict with outgoing calls
    sip_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sip_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sip_sock.bind((local_ip, LOCAL_SIP_PORT + 1))

    reg_port = LOCAL_SIP_PORT + 1
    if not register_incoming(sip_sock, local_ip, reg_port):
        sip_sock.close()
        return

    _incoming_sip_sock = sip_sock
    _incoming_listener_thread = threading.Thread(
        target=incoming_call_listener,
        args=(sip_sock, local_ip),
        daemon=True,
    )
    _incoming_listener_thread.start()


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(title="AI Voice Agent API (OpenAI Realtime)")


@app.post("/call", response_model=CallResult)
async def make_call(req: CallRequest):
    """Place a phone call with a given objective and return the result."""
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY in .env")

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


@app.on_event("startup")
async def on_startup():
    """Start the incoming call listener when the server starts."""
    threading.Thread(target=start_incoming_listener, daemon=True).start()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
