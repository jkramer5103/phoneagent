#!/usr/bin/env python3
"""
VoIP call script: Registers with a Speedport router's SIP server,
calls a mobile number, and plays a TTS message.

Uses raw SIP/RTP over UDP to support Speedport's digest auth (qop=auth)
and separate auth username vs SIP URI username.
"""

import io
import os
import re
import time
import socket
import struct
import hashlib
import random
import audioop
import threading
import miniaudio
from gtts import gTTS

# --- Configuration ---
SIP_SERVER = "speedport.ip"
SIP_PORT = 5060
SIP_URI_USER = "**71"                    # Nutzerkennung (SIP URI)
AUTH_USER = "nutzer-1@speedport.ip"      # Authentifizierungsname (digest auth)
AUTH_PASS = "B-fx3$h-7yH4&42"           # Passwort
CALL_NUMBER = "+4915123412098"
TTS_TEXT = "Test test 1 2 3. Dies ist ein automatischer Anruf. Alles funktioniert einwandfrei."
TTS_LANG = "de"
LOCAL_SIP_PORT = 5060
RTP_PORT = 16384


def get_local_ip():
    """Get the LAN IP used to reach the SIP server."""
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
    """Compute SIP Digest authentication response (MD5, with optional qop=auth)."""
    ha1 = hashlib.md5(f"{auth_user}:{realm}:{password}".encode()).hexdigest()
    ha2 = hashlib.md5(f"{method}:{uri}".encode()).hexdigest()
    if qop == "auth":
        response = hashlib.md5(f"{ha1}:{nonce}:{nc}:{cnonce}:{qop}:{ha2}".encode()).hexdigest()
    else:
        response = hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode()).hexdigest()
    return response


def build_auth_header(auth_user, password, realm, nonce, method, uri, qop=None, algorithm="MD5"):
    """Build a full Authorization header value."""
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
    """Parse WWW-Authenticate header into dict."""
    result = {}
    # Extract key="value" and key=value pairs
    for m in re.finditer(r'(\w+)=(?:"([^"]+)"|(\S+?)(?:,|$))', header_line):
        key = m.group(1)
        value = m.group(2) if m.group(2) is not None else m.group(3)
        result[key] = value
    return result


def parse_sip_response(data):
    """Parse a SIP response into status code, headers dict, and body."""
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
    """Send a SIP message and wait for a response."""
    sock.sendto(msg.encode(), (server, port))
    sock.settimeout(timeout)
    try:
        data, addr = sock.recvfrom(8192)
        return data
    except socket.timeout:
        return None


def linear_to_ulaw(sample):
    """Convert a single 16-bit signed PCM sample to u-law byte."""
    return audioop.lin2ulaw(struct.pack("<h", sample), 2)


def generate_tts_audio(text, lang):
    """Generate TTS audio and return as 8kHz 16-bit mono PCM bytes."""
    print("[*] Generating TTS audio...")
    tts = gTTS(text=text, lang=lang)
    mp3_buffer = io.BytesIO()
    tts.write_to_fp(mp3_buffer)
    mp3_buffer.seek(0)
    mp3_data = mp3_buffer.read()

    decoded = miniaudio.decode(mp3_data, output_format=miniaudio.SampleFormat.SIGNED16,
                               nchannels=1, sample_rate=8000)
    pcm_bytes = bytes(decoded.samples)
    ulaw_audio = audioop.lin2ulaw(pcm_bytes, 2)
    print(f"[*] TTS audio ready: {len(ulaw_audio)} bytes ({len(ulaw_audio) / 8000:.1f}s)")
    return ulaw_audio


def sip_register(sock, local_ip):
    """Register with the SIP server. Returns True on success."""
    tag = gen_tag()
    call_id = gen_call_id(local_ip)
    branch = gen_branch()

    # Step 1: Initial REGISTER (expect 401 or 423)
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
        f"User-Agent: VoIPCaller/1.0\r\n"
        f"Content-Length: 0\r\n"
        f"\r\n"
    )

    print("[*] Sending initial REGISTER...")
    resp = sip_send_recv(sock, msg, SIP_SERVER, SIP_PORT)
    if resp is None:
        print("[!] No response to REGISTER")
        return False

    status, headers, body, status_line = parse_sip_response(resp)
    print(f"[*] Response: {status_line}")

    if status == 423:
        # Interval too brief, retry with Min-Expires
        min_exp = headers.get("Min-Expires", "300")
        print(f"[*] Router requires Min-Expires: {min_exp}, retrying...")
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
            f"User-Agent: VoIPCaller/1.0\r\n"
            f"Content-Length: 0\r\n"
            f"\r\n"
        )
        resp = sip_send_recv(sock, msg, SIP_SERVER, SIP_PORT)
        if resp is None:
            print("[!] No response to REGISTER retry")
            return False
        status, headers, body, status_line = parse_sip_response(resp)
        print(f"[*] Response: {status_line}")

    if status != 401:
        if status == 200:
            print("[*] Registered (no auth required)!")
            return True
        print(f"[!] Unexpected status: {status}")
        return False

    # Step 2: Parse 401 challenge and respond with credentials
    www_auth = headers.get("WWW-Authenticate", "")
    auth_params = parse_www_authenticate(www_auth)
    realm = auth_params.get("realm", SIP_SERVER)
    nonce = auth_params.get("nonce", "")
    qop = auth_params.get("qop", None)
    algorithm = auth_params.get("algorithm", "MD5")

    print(f"[*] Auth challenge: realm={realm}, qop={qop}, algorithm={algorithm}")

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
        f"User-Agent: VoIPCaller/1.0\r\n"
        f"Content-Length: 0\r\n"
        f"\r\n"
    )

    print("[*] Sending authenticated REGISTER...")
    resp = sip_send_recv(sock, msg, SIP_SERVER, SIP_PORT)
    if resp is None:
        print("[!] No response to authenticated REGISTER")
        return False

    status, headers, body, status_line = parse_sip_response(resp)
    print(f"[*] Response: {status_line}")

    if status == 200:
        print("[*] Registration successful!")
        return True
    else:
        print(f"[!] Registration failed with status {status}")
        return False


def sip_invite(sock, local_ip):
    """Send INVITE to make a call. Returns (call_id, tag, remote_rtp_ip, remote_rtp_port) or None."""
    tag = gen_tag()
    call_id = gen_call_id(local_ip)
    branch = gen_branch()
    sess_id = str(random.randint(1000000, 9999999))

    sdp_body = (
        f"v=0\r\n"
        f"o=VoIPCaller {sess_id} {sess_id} IN IP4 {local_ip}\r\n"
        f"s=VoIPCall\r\n"
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
        f"User-Agent: VoIPCaller/1.0\r\n"
        f"Content-Length: {len(sdp_body)}\r\n"
        f"\r\n"
        f"{sdp_body}"
    )

    print(f"[*] Sending INVITE to {CALL_NUMBER}...")
    resp = sip_send_recv(sock, msg, SIP_SERVER, SIP_PORT, timeout=5)
    if resp is None:
        print("[!] No response to INVITE")
        return None

    status, headers, body, status_line = parse_sip_response(resp)
    print(f"[*] Response: {status_line}")

    # Handle 407 Proxy Authentication Required or 401
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

        # Send ACK for the failed INVITE first
        ack_msg = (
            f"ACK sip:{CALL_NUMBER}@{SIP_SERVER} SIP/2.0\r\n"
            f"Via: SIP/2.0/UDP {local_ip}:{LOCAL_SIP_PORT};branch={branch};rport\r\n"
            f"From: <sip:{SIP_URI_USER}@{SIP_SERVER}>;tag={tag}\r\n"
            f"To: {headers.get('To', f'<sip:{CALL_NUMBER}@{SIP_SERVER}>')}\r\n"
            f"Call-ID: {call_id}\r\n"
            f"CSeq: 1 ACK\r\n"
            f"Max-Forwards: 70\r\n"
            f"Content-Length: 0\r\n"
            f"\r\n"
        )
        sock.sendto(ack_msg.encode(), (SIP_SERVER, SIP_PORT))

        # Re-INVITE with auth
        branch = gen_branch()
        msg = (
            f"INVITE sip:{CALL_NUMBER}@{SIP_SERVER} SIP/2.0\r\n"
            f"Via: SIP/2.0/UDP {local_ip}:{LOCAL_SIP_PORT};branch={branch};rport\r\n"
            f"From: <sip:{SIP_URI_USER}@{SIP_SERVER}>;tag={tag}\r\n"
            f"To: <sip:{CALL_NUMBER}@{SIP_SERVER}>\r\n"
            f"Call-ID: {call_id}\r\n"
            f"CSeq: 2 INVITE\r\n"
            f"Contact: <sip:{SIP_URI_USER}@{local_ip}:{LOCAL_SIP_PORT}>\r\n"
            f"Max-Forwards: 70\r\n"
            f"{auth_line_name}: {auth_header}\r\n"
            f"Content-Type: application/sdp\r\n"
            f"User-Agent: VoIPCaller/1.0\r\n"
            f"Content-Length: {len(sdp_body)}\r\n"
            f"\r\n"
            f"{sdp_body}"
        )

        print("[*] Sending authenticated INVITE...")
        resp = sip_send_recv(sock, msg, SIP_SERVER, SIP_PORT, timeout=5)
        if resp is None:
            print("[!] No response to authenticated INVITE")
            return None
        status, headers, body, status_line = parse_sip_response(resp)
        print(f"[*] Response: {status_line}")

    # Wait for 100 Trying / 180 Ringing / 183 Session Progress / 200 OK
    cseq = 2
    while status in (100, 180, 183):
        print(f"[*] Waiting... ({status_line})")
        resp_data = None
        sock.settimeout(60)
        try:
            resp_data, _ = sock.recvfrom(8192)
        except socket.timeout:
            print("[!] Timeout waiting for call progress")
            return None
        if resp_data:
            status, headers, body, status_line = parse_sip_response(resp_data)
            print(f"[*] Response: {status_line}")

    if status != 200:
        print(f"[!] Call failed with status {status}")
        return None

    # Parse remote RTP endpoint from SDP
    remote_rtp_ip = None
    remote_rtp_port = None
    for line in body.split("\r\n"):
        if not line and "\n" in body:
            # Handle \n-only line endings
            for line2 in body.split("\n"):
                if line2.startswith("c=IN IP4 "):
                    remote_rtp_ip = line2.split()[-1].strip()
                if line2.startswith("m=audio "):
                    remote_rtp_port = int(line2.split()[1])
            break
        if line.startswith("c=IN IP4 "):
            remote_rtp_ip = line.split()[-1].strip()
        if line.startswith("m=audio "):
            remote_rtp_port = int(line.split()[1])

    print(f"[*] Call answered! Remote RTP: {remote_rtp_ip}:{remote_rtp_port}")

    # Send ACK
    to_header = headers.get("To", f"<sip:{CALL_NUMBER}@{SIP_SERVER}>")
    ack_msg = (
        f"ACK sip:{CALL_NUMBER}@{SIP_SERVER} SIP/2.0\r\n"
        f"Via: SIP/2.0/UDP {local_ip}:{LOCAL_SIP_PORT};branch={gen_branch()};rport\r\n"
        f"From: <sip:{SIP_URI_USER}@{SIP_SERVER}>;tag={tag}\r\n"
        f"To: {to_header}\r\n"
        f"Call-ID: {call_id}\r\n"
        f"CSeq: {cseq} ACK\r\n"
        f"Max-Forwards: 70\r\n"
        f"Content-Length: 0\r\n"
        f"\r\n"
    )
    sock.sendto(ack_msg.encode(), (SIP_SERVER, SIP_PORT))
    print("[*] ACK sent")

    return call_id, tag, remote_rtp_ip, remote_rtp_port, to_header


def send_rtp_audio(local_ip, remote_ip, remote_port, ulaw_audio):
    """Send u-law audio over RTP."""
    rtp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rtp_sock.bind((local_ip, RTP_PORT))

    ssrc = random.randint(0, 0xFFFFFFFF)
    seq = random.randint(0, 0xFFFF)
    timestamp = random.randint(0, 0xFFFFFFFF)
    CHUNK_SIZE = 160  # 20ms at 8kHz
    PT = 0  # PCMU

    print(f"[*] Streaming {len(ulaw_audio)} bytes of audio to {remote_ip}:{remote_port}...")

    for i in range(0, len(ulaw_audio), CHUNK_SIZE):
        chunk = ulaw_audio[i:i + CHUNK_SIZE]
        if len(chunk) < CHUNK_SIZE:
            chunk += b'\xff' * (CHUNK_SIZE - len(chunk))  # silence padding

        # RTP header: V=2, P=0, X=0, CC=0, M=0, PT, seq, timestamp, SSRC
        marker = 0x80 if i == 0 else 0x00  # marker bit on first packet
        rtp_header = struct.pack("!BBHII",
                                 0x80,           # V=2, P=0, X=0, CC=0
                                 PT | marker,    # M + PT
                                 seq & 0xFFFF,
                                 timestamp & 0xFFFFFFFF,
                                 ssrc)
        rtp_sock.sendto(rtp_header + chunk, (remote_ip, remote_port))

        seq += 1
        timestamp += CHUNK_SIZE
        time.sleep(0.02)  # 20ms pacing

    rtp_sock.close()
    print("[*] Audio streaming complete")


def sip_bye(sock, local_ip, call_id, tag, to_header):
    """Send BYE to end the call."""
    branch = gen_branch()
    msg = (
        f"BYE sip:{CALL_NUMBER}@{SIP_SERVER} SIP/2.0\r\n"
        f"Via: SIP/2.0/UDP {local_ip}:{LOCAL_SIP_PORT};branch={branch};rport\r\n"
        f"From: <sip:{SIP_URI_USER}@{SIP_SERVER}>;tag={tag}\r\n"
        f"To: {to_header}\r\n"
        f"Call-ID: {call_id}\r\n"
        f"CSeq: 3 BYE\r\n"
        f"Max-Forwards: 70\r\n"
        f"Content-Length: 0\r\n"
        f"\r\n"
    )
    print("[*] Sending BYE...")
    resp = sip_send_recv(sock, msg, SIP_SERVER, SIP_PORT, timeout=5)
    if resp:
        status, _, _, status_line = parse_sip_response(resp)
        print(f"[*] BYE response: {status_line}")
    else:
        print("[*] No response to BYE (call may already be ended)")


def main():
    local_ip = get_local_ip()
    print(f"[*] Local IP: {local_ip}")

    ulaw_audio = generate_tts_audio(TTS_TEXT, TTS_LANG)

    # Create SIP UDP socket
    sip_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sip_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sip_sock.bind((local_ip, LOCAL_SIP_PORT))

    try:
        # Step 1: Register
        if not sip_register(sip_sock, local_ip):
            print("[!] Registration failed, aborting.")
            return

        # Step 2: INVITE (make the call)
        result = sip_invite(sip_sock, local_ip)
        if result is None:
            print("[!] Call setup failed, aborting.")
            return

        call_id, tag, remote_rtp_ip, remote_rtp_port, to_header = result

        # Step 3: Stream TTS audio via RTP
        send_rtp_audio(local_ip, remote_rtp_ip, remote_rtp_port, ulaw_audio)

        # Step 4: Hang up
        time.sleep(1)
        sip_bye(sip_sock, local_ip, call_id, tag, to_header)

    finally:
        sip_sock.close()

    print("[*] Done.")


if __name__ == "__main__":
    main()
