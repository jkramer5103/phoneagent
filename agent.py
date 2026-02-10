#!/usr/bin/env python3
"""
AI Voice Agent: Calls a phone number via SIP/RTP, transcribes speech with
Soniox real-time STT, generates responses with Gemini via OpenRouter,
and speaks them back using Cartesia Sonic TTS. Supports interruption.
"""

import os
import re
import json
import time
import socket
import struct
import hashlib
import random
import audioop
import queue
import threading
import httpx
from cartesia import Cartesia
from dotenv import load_dotenv
from websockets.sync.client import connect as ws_connect

load_dotenv()

# --- Configuration ---
SIP_SERVER = "speedport.ip"
SIP_PORT = 5060
SIP_URI_USER = "**71"
AUTH_USER = "nutzer-1@speedport.ip"
AUTH_PASS = "B-fx3$h-7yH4&42"
CALL_NUMBER = "+4915123412098"
LOCAL_SIP_PORT = 5060
RTP_PORT = 16384

OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
SONIOX_API_KEY = os.getenv("SONIOX_API_KEY")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
LLM_MODEL = "google/gemini-3-flash-preview"  # Smart LLM for text generation
CARTESIA_VOICE_ID = "afa425cf-5489-4a09-8a3f-d3cb1f82150d"  # Nico - Friendly Agent (German)

SYSTEM_PROMPT = (
    "Du bist ein freundlicher KI-Sprachassistent am Telefon. "
    "Halte deine Antworten kurz und gesprächig — normalerweise 1-3 Sätze. "
    "Du sprichst Deutsch. "
    "Sei hilfsbereit, natürlich und sympathisch."
)

# --- Shared State ---
class AgentState:
    def __init__(self):
        self.is_speaking = False          # True while AI audio is being sent
        self.interrupt_flag = False       # Set to True when user interrupts
        self.call_active = True           # False when call ends
        self.conversation = []            # Chat history for Gemini
        self.lock = threading.Lock()
        self.pending_text = ""            # Accumulated STT text before sending to LLM
        self.silence_start = None         # When silence started (for endpoint detection)
        self.user_speaking = False        # True when user is currently speaking
        self.speaking_ended_at = 0.0      # Timestamp when AI stopped speaking (echo cooldown)
        self.llm_busy = False             # True while LLM is generating a response

state = AgentState()


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


def sip_invite(sock, local_ip):
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
        f"INVITE sip:{CALL_NUMBER}@{SIP_SERVER} SIP/2.0\r\n"
        f"Via: SIP/2.0/UDP {local_ip}:{LOCAL_SIP_PORT};branch={branch};rport\r\n"
        f"From: <sip:{SIP_URI_USER}@{SIP_SERVER}>;tag={tag}\r\n"
        f"To: <sip:{CALL_NUMBER}@{SIP_SERVER}>\r\n"
        f"Call-ID: {call_id}\r\n"
        f"CSeq: 1 INVITE\r\n"
        f"Contact: <sip:{SIP_URI_USER}@{local_ip}:{LOCAL_SIP_PORT}>\r\n"
        f"Max-Forwards: 70\r\n"
        f"Content-Type: application/sdp\r\n"
        f"User-Agent: VoIPAgent/1.0\r\n"
        f"Content-Length: {len(sdp_body)}\r\n\r\n"
        f"{sdp_body}"
    )

    print(f"[SIP] INVITE {CALL_NUMBER}...")
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

        invite_uri = f"sip:{CALL_NUMBER}@{SIP_SERVER}"
        auth_header = build_auth_header(AUTH_USER, AUTH_PASS, realm, nonce, "INVITE", invite_uri, qop, algorithm)
        auth_line_name = "Proxy-Authorization" if status == 407 else "Authorization"

        ack_msg = (
            f"ACK sip:{CALL_NUMBER}@{SIP_SERVER} SIP/2.0\r\n"
            f"Via: SIP/2.0/UDP {local_ip}:{LOCAL_SIP_PORT};branch={branch};rport\r\n"
            f"From: <sip:{SIP_URI_USER}@{SIP_SERVER}>;tag={tag}\r\n"
            f"To: {headers.get('To', f'<sip:{CALL_NUMBER}@{SIP_SERVER}>')}\r\n"
            f"Call-ID: {call_id}\r\n"
            f"CSeq: 1 ACK\r\n"
            f"Max-Forwards: 70\r\n"
            f"Content-Length: 0\r\n\r\n"
        )
        sock.sendto(ack_msg.encode(), (SIP_SERVER, SIP_PORT))

        branch = gen_branch()
        cseq = 2
        msg = (
            f"INVITE sip:{CALL_NUMBER}@{SIP_SERVER} SIP/2.0\r\n"
            f"Via: SIP/2.0/UDP {local_ip}:{LOCAL_SIP_PORT};branch={branch};rport\r\n"
            f"From: <sip:{SIP_URI_USER}@{SIP_SERVER}>;tag={tag}\r\n"
            f"To: <sip:{CALL_NUMBER}@{SIP_SERVER}>\r\n"
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

    to_header = headers.get("To", f"<sip:{CALL_NUMBER}@{SIP_SERVER}>")
    ack_msg = (
        f"ACK sip:{CALL_NUMBER}@{SIP_SERVER} SIP/2.0\r\n"
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


def sip_bye(sock, local_ip, call_id, tag, cseq, to_header):
    branch = gen_branch()
    msg = (
        f"BYE sip:{CALL_NUMBER}@{SIP_SERVER} SIP/2.0\r\n"
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

# Persistent Cartesia client and WebSocket (initialized in main)
cartesia_client = None
cartesia_ws = None

CARTESIA_SAMPLE_RATE = 44100  # Synthesize at high quality, downsample for RTP


def cartesia_tts_ulaw(text):
    """Convert text to u-law audio via Cartesia Sonic (single shot, for greeting)."""
    global cartesia_ws
    if not text.strip():
        return b""

    pcm_chunks = []
    try:
        for output in cartesia_ws.send(
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
        try:
            cartesia_ws = cartesia_client.tts.websocket()
        except Exception:
            pass
        return b""

    if not pcm_chunks:
        return b""

    pcm = b"".join(pcm_chunks)
    pcm8k, _ = audioop.ratecv(pcm, 2, 1, CARTESIA_SAMPLE_RATE, 8000, None)
    return audioop.lin2ulaw(pcm8k, 2)


# ============================================================
# RTP Send (with interruption support) + Keepalive
# ============================================================

# Shared RTP state for keepalive coordination
class RTPState:
    def __init__(self):
        self.seq = random.randint(0, 0xFFFF)
        self.timestamp = random.randint(0, 0xFFFFFFFF)
        self.ssrc = random.randint(0, 0xFFFFFFFF)
        self.lock = threading.Lock()

rtp_state = RTPState()


def rtp_keepalive_thread(rtp_sock, remote_ip, remote_port):
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


def rtp_send_thread(rtp_sock, remote_ip, remote_port, ulaw_audio):
    """Send u-law audio over RTP. Stops if interrupted."""
    CHUNK_SIZE = 160

    state.is_speaking = True
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

        time.sleep(0.02)

    state.is_speaking = False
    state.speaking_ended_at = time.time()


# ============================================================
# RTP Receive → Soniox STT
# ============================================================

def rtp_receive_thread(rtp_sock, soniox_ws):
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
        if rms > INTERRUPT_THRESHOLD and state.is_speaking:
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

def llm_respond(user_text, rtp_sock, remote_ip, remote_port):
    """Stream Gemini text → Cartesia continuations → RTP audio, all concurrently."""
    if not user_text.strip():
        return

    global cartesia_ws
    state.llm_busy = True
    state.interrupt_flag = False
    print(f"[LLM] User said: {user_text}")

    state.conversation.append({"role": "user", "content": user_text})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + state.conversation[-20:]

    full_response = ""
    audio_q = queue.Queue()  # PCM audio chunks from Cartesia → RTP sender
    tts_ctx = cartesia_ws.context()

    # --- Thread 1: Receive audio from Cartesia and push to queue ---
    def audio_receiver():
        try:
            for output in tts_ctx.receive():
                if state.interrupt_flag or not state.call_active:
                    break
                if hasattr(output, 'audio') and output.audio:
                    audio_q.put(output.audio)
        except Exception as e:
            print(f"[TTS] Receive error: {e}")
        audio_q.put(None)  # Sentinel: no more audio

    recv_thread = threading.Thread(target=audio_receiver, daemon=True)
    recv_thread.start()

    # --- Thread 2: Send RTP packets from audio queue ---
    spoken_duration = 0.0  # Track how much audio was actually sent

    def rtp_sender():
        nonlocal spoken_duration
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
                spoken_duration += 0.02
                time.sleep(0.02)

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
                            # Stream each token directly to Cartesia
                            try:
                                tts_ctx.send(
                                    model_id="sonic-3",
                                    transcript=content,
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

    # Signal Cartesia that no more text is coming
    try:
        tts_ctx.no_more_inputs()
    except Exception:
        pass

    # Wait for audio to finish playing
    recv_thread.join(timeout=15)
    rtp_thread.join(timeout=15)

    state.is_speaking = False
    state.speaking_ended_at = time.time()

    # Record conversation history
    if full_response.strip():
        if state.interrupt_flag:
            state.conversation.append({"role": "assistant", "content": f"{full_response.strip()} [user interrupted]"})
            print(f"[LLM] Interrupted. Response was: {full_response.strip()}")
        else:
            state.conversation.append({"role": "assistant", "content": full_response.strip()})
            print(f"[LLM] Response: {full_response.strip()}")

    state.llm_busy = False


def stt_process_thread(soniox_ws, rtp_sock, remote_ip, remote_port):
    """Read Soniox STT results and trigger LLM responses."""
    print("[STT] Processing thread started")
    final_text = ""
    last_final_time = time.time()

    while state.call_active:
        try:
            msg = soniox_ws.recv(timeout=0.2)
        except TimeoutError:
            # Check if we have accumulated text and silence has passed
            if final_text.strip() and not state.user_speaking and not state.llm_busy:
                elapsed = time.time() - last_final_time
                if elapsed > 1.5:
                    # User stopped speaking, process the text
                    text = final_text.strip()
                    final_text = ""
                    # Run LLM in a separate thread so STT keeps processing
                    threading.Thread(target=llm_respond,
                                     args=(text, rtp_sock, remote_ip, remote_port),
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

def sip_listener_thread(sip_sock, local_ip, call_id, tag, to_header):
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
# Main
# ============================================================

def main():
    if not OPENROUTER_KEY:
        print("[!] OPENROUTER_KEY not set in .env")
        return
    if not SONIOX_API_KEY:
        print("[!] SONIOX_API_KEY not set in .env")
        return

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
            print("[!] Registration failed")
            return

        # Step 2: Call
        result = sip_invite(sip_sock, local_ip)
        if result is None:
            print("[!] Call setup failed")
            return

        call_id, tag, cseq, remote_rtp_ip, remote_rtp_port, to_header = result
        print("[*] Call established! Starting AI agent...")

        # Step 3: Connect to Soniox STT WebSocket
        print("[STT] Connecting to Soniox...")
        soniox_ws = ws_connect("wss://stt-rt.soniox.com/transcribe-websocket")

        # Configure Soniox for raw PCM from phone (8kHz, 16-bit, mono after ulaw decode)
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

        # Step 4: Start all threads BEFORE greeting so we can hear user immediately after
        threads = []

        # RTP receive → Soniox
        t_rtp_rx = threading.Thread(target=rtp_receive_thread, args=(rtp_sock, soniox_ws), daemon=True)
        t_rtp_rx.start()
        threads.append(t_rtp_rx)

        # STT → LLM → TTS → RTP
        t_stt = threading.Thread(target=stt_process_thread,
                                 args=(soniox_ws, rtp_sock, remote_rtp_ip, remote_rtp_port), daemon=True)
        t_stt.start()
        threads.append(t_stt)

        # RTP keepalive (prevents router from dropping the call)
        t_keepalive = threading.Thread(target=rtp_keepalive_thread,
                                       args=(rtp_sock, remote_rtp_ip, remote_rtp_port), daemon=True)
        t_keepalive.start()
        threads.append(t_keepalive)

        # Initialize Cartesia TTS
        global cartesia_client, cartesia_ws
        print("[TTS] Connecting to Cartesia Sonic...")
        cartesia_client = Cartesia(api_key=CARTESIA_API_KEY)
        cartesia_ws = cartesia_client.tts.websocket()
        print("[TTS] Cartesia connected")

        # Play greeting
        print("[*] Playing greeting...")
        greeting = "Hallo! Ich bin dein KI Assistent. Wie kann ich dir helfen?"
        greeting_audio = cartesia_tts_ulaw(greeting)
        rtp_send_thread(rtp_sock, remote_rtp_ip, remote_rtp_port, greeting_audio)
        state.conversation.append({"role": "assistant", "content": greeting})

        # SIP listener (detect BYE)
        t_sip = threading.Thread(target=sip_listener_thread,
                                 args=(sip_sock, local_ip, call_id, tag, to_header), daemon=True)
        t_sip.start()
        threads.append(t_sip)

        print("[*] AI Agent is live! Press Ctrl+C to end the call.")
        print("=" * 60)

        # Wait for call to end
        try:
            while state.call_active:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n[*] Ctrl+C — ending call...")
            state.call_active = False

        # Cleanup
        print("[*] Shutting down...")
        try:
            soniox_ws.send(b"")  # Signal end of stream
            time.sleep(0.5)
            soniox_ws.close()
        except Exception:
            pass
        try:
            cartesia_ws.close()
        except Exception:
            pass

        sip_bye(sip_sock, local_ip, call_id, tag, cseq, to_header)

        for t in threads:
            t.join(timeout=3)

    finally:
        rtp_sock.close()
        sip_sock.close()

    print("[*] Done.")


if __name__ == "__main__":
    main()
