#!/usr/bin/env python3
"""Register a Speedport IP phone, place one call, play beeps, and hang up.

This is a small, dependency-free SIP/RTP client intended for a trusted LAN.
It supports the UDP transport, SIP digest authentication, and G.711 mu-law.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import math
import os
import random
import re
import select
import socket
import struct
import sys
import time
import uuid
from dataclasses import dataclass


CRLF = "\r\n"


@dataclass
class SipMessage:
    start: str
    headers: dict[str, str]
    body: str
    source: tuple[str, int]

    @property
    def status(self) -> int:
        match = re.match(r"SIP/2\.0\s+(\d{3})", self.start)
        return int(match.group(1)) if match else 0


def parse_sip(data: bytes, source: tuple[str, int]) -> SipMessage:
    text = data.decode("utf-8", "replace")
    head, _, body = text.partition(CRLF + CRLF)
    lines = head.split(CRLF)
    headers: dict[str, str] = {}
    current = ""
    for line in lines[1:]:
        if line[:1] in (" ", "\t") and current:
            headers[current] += " " + line.strip()
            continue
        name, sep, value = line.partition(":")
        if sep:
            current = name.strip().lower()
            headers[current] = value.strip()
    return SipMessage(lines[0], headers, body, source)


def digest_fields(challenge: str) -> dict[str, str]:
    challenge = re.sub(r"^\s*Digest\s+", "", challenge, flags=re.I)
    fields: dict[str, str] = {}
    for match in re.finditer(r'(\w+)\s*=\s*(?:"([^"]*)"|([^,\s]+))', challenge):
        fields[match.group(1).lower()] = match.group(2) or match.group(3)
    return fields


def md5(value: str) -> str:
    return hashlib.md5(value.encode()).hexdigest()


def digest_authorization(
    challenge: str, username: str, password: str, method: str, uri: str
) -> str:
    values = digest_fields(challenge)
    realm = values.get("realm", "")
    nonce = values.get("nonce", "")
    algorithm = values.get("algorithm", "MD5")
    if algorithm.upper() != "MD5":
        raise RuntimeError(f"Unsupported digest algorithm: {algorithm}")
    if not realm or not nonce:
        raise RuntimeError("Registrar returned an incomplete digest challenge")

    ha1 = md5(f"{username}:{realm}:{password}")
    ha2 = md5(f"{method}:{uri}")
    qop_options = [item.strip() for item in values.get("qop", "").split(",")]
    qop = "auth" if "auth" in qop_options else ""
    cnonce = uuid.uuid4().hex[:16]
    nc = "00000001"
    response = (
        md5(f"{ha1}:{nonce}:{nc}:{cnonce}:{qop}:{ha2}")
        if qop
        else md5(f"{ha1}:{nonce}:{ha2}")
    )
    parts = [
        f'username="{username}"',
        f'realm="{realm}"',
        f'nonce="{nonce}"',
        f'uri="{uri}"',
        f'response="{response}"',
        "algorithm=MD5",
    ]
    if "opaque" in values:
        parts.append(f'opaque="{values["opaque"]}"')
    if qop:
        parts.extend([f"qop={qop}", f"nc={nc}", f'cnonce="{cnonce}"'])
    return "Digest " + ", ".join(parts)


class SipClient:
    def __init__(
        self,
        server: str,
        port: int,
        domain: str,
        extension: str,
        username: str,
        password: str,
        timeout: float,
        advertise_ip: str | None = None,
        local_port: int = 5060,
        interface: str | None = None,
    ) -> None:
        self.server_host = server
        self.server_port = port
        self.server_ip = socket.gethostbyname(server)
        self.domain = domain
        self.extension = extension
        self.username = username
        self.password = password
        self.timeout = timeout
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.interface = interface
        if interface:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, interface.encode() + b"\0")
        self.sock.bind(("0.0.0.0", local_port))
        self.sock.connect((self.server_ip, port))
        self.sock.setblocking(False)
        self.local_ip, self.local_port = self.sock.getsockname()
        self.advertise_ip = advertise_ip or self.local_ip
        self.call_id = f"{uuid.uuid4().hex}@{self.advertise_ip}"
        self.from_tag = uuid.uuid4().hex[:12]
        self.cseq = random.randint(1000, 9000)
        self.register_auth: str | None = None

    @property
    def contact(self) -> str:
        return (
            f"<sip:{self.extension}@{self.advertise_ip}:{self.local_port};transport=udp>"
            ";expires=600"
        )

    def send(
        self,
        method: str,
        uri: str,
        *,
        to_uri: str,
        body: str = "",
        authorization: str | None = None,
        authorization_name: str = "Authorization",
        route: str | None = None,
        call_id: str | None = None,
        cseq: int | None = None,
        from_tag: str | None = None,
        to_tag: str | None = None,
        branch: str | None = None,
    ) -> int:
        if cseq is None:
            self.cseq += 1
            cseq = self.cseq
        from_uri = f"sip:{self.extension}@{self.domain}"
        tag = from_tag or self.from_tag
        to_value = f"<{to_uri}>" + (f";tag={to_tag}" if to_tag else "")
        branch = branch or f"z9hG4bK-{uuid.uuid4().hex[:16]}"
        headers = [
            f"{method} {uri} SIP/2.0",
            f"Via: SIP/2.0/UDP {self.advertise_ip}:{self.local_port};branch={branch};rport",
            "Max-Forwards: 70",
            f"From: <{from_uri}>;tag={tag}",
            f"To: {to_value}",
            f"Call-ID: {call_id or self.call_id}",
            f"CSeq: {cseq} {method}",
            f"Contact: {self.contact}",
            "Expires: 600",
            "User-Agent: TelephoneAgent/1.0",
            "Allow: INVITE, ACK, CANCEL, BYE, OPTIONS",
        ]
        if route:
            headers.append(f"Route: {route}")
        if authorization:
            headers.append(f"{authorization_name}: {authorization}")
        if body:
            headers.append("Content-Type: application/sdp")
        encoded = body.encode()
        headers.append(f"Content-Length: {len(encoded)}")
        packet = (CRLF.join(headers) + CRLF + CRLF).encode() + encoded
        self.sock.send(packet)
        return cseq

    def receive(self, deadline: float) -> SipMessage:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for a SIP response")
            readable, _, _ = select.select([self.sock], [], [], remaining)
            if readable:
                data, source = self.sock.recvfrom(65535)
                message = parse_sip(data, source)
                if message.start.startswith("SIP/2.0"):
                    return message

    def final_response(self, cseq: int, method: str, deadline: float) -> SipMessage:
        while True:
            response = self.receive(deadline)
            response_cseq = response.headers.get("cseq", "")
            if response_cseq != f"{cseq} {method}":
                continue
            if response.status >= 200:
                if os.environ.get("SIP_DEBUG"):
                    safe_headers = {
                        name: value
                        for name, value in response.headers.items()
                        if name not in {"authorization", "proxy-authorization"}
                    }
                    print(f"< {response.start} {safe_headers}", file=sys.stderr)
                return response
            if method == "INVITE" and response.status in (180, 183):
                print("Ringing…" if response.status == 180 else "Call progressing…")

    def register(self) -> None:
        uri = f"sip:{self.domain}"
        to_uri = f"sip:{self.extension}@{self.domain}"
        cseq = self.send("REGISTER", uri, to_uri=to_uri)
        response = self.final_response(cseq, "REGISTER", time.monotonic() + self.timeout)
        if response.status in (401, 407):
            challenge_name = "www-authenticate" if response.status == 401 else "proxy-authenticate"
            challenge = response.headers.get(challenge_name, "")
            self.register_auth = digest_authorization(
                challenge, self.username, self.password, "REGISTER", uri
            )
            auth_name = "Authorization" if response.status == 401 else "Proxy-Authorization"
            cseq = self.send(
                "REGISTER",
                uri,
                to_uri=to_uri,
                authorization=self.register_auth,
                authorization_name=auth_name,
            )
            response = self.final_response(cseq, "REGISTER", time.monotonic() + self.timeout)
        if response.status != 200:
            raise RuntimeError(f"Registration failed: {response.start}")
        print(
            f"Registered {self.extension}@{self.domain} via {self.server_ip} "
            f"(advertised {self.advertise_ip})"
        )


def tag_from_to(value: str) -> str | None:
    match = re.search(r"(?:^|;)\s*tag=([^;>\s]+)", value, re.I)
    return match.group(1) if match else None


def sip_uri(value: str, fallback: str) -> str:
    """Extract a URI from a name-addr header such as <sip:user@host>;param=x."""
    match = re.search(r"<([^>]+)>", value)
    if match:
        return match.group(1)
    bare = value.split(";", 1)[0].strip()
    return bare or fallback


def sdp_for(local_ip: str, rtp_port: int, direction: str = "sendonly") -> str:
    if direction not in {"sendonly", "recvonly", "sendrecv", "inactive"}:
        raise ValueError(f"Unsupported SDP media direction: {direction}")
    session = random.randint(1, 2**31)
    return CRLF.join(
        [
            "v=0",
            f"o=TelephoneAgent {session} {session} IN IP4 {local_ip}",
            "s=TelephoneAgent",
            f"c=IN IP4 {local_ip}",
            "t=0 0",
            f"m=audio {rtp_port} RTP/AVP 0",
            "a=rtpmap:0 PCMU/8000",
            f"a={direction}",
            "a=ptime:20",
            "",
        ]
    )


def remote_rtp(sdp: str, fallback_ip: str) -> tuple[str, int]:
    connection = re.search(r"^c=IN IP4 ([^\s]+)", sdp, re.M)
    media = re.search(r"^m=audio (\d+)\s+RTP/AVP(?:\s+.*)?$", sdp, re.M)
    if not media:
        raise RuntimeError("Answered call did not include an RTP audio port")
    return (connection.group(1) if connection else fallback_ip, int(media.group(1)))


def beep_pcm() -> bytes:
    """Three 700 Hz beeps, with short silences, as 16-bit mono PCM."""
    rate = 8000
    output = bytearray()
    for _ in range(3):
        for index in range(int(rate * 0.28)):
            sample = int(10000 * math.sin(2 * math.pi * 700 * index / rate))
            output += struct.pack("<h", sample)
        output += b"\0\0" * int(rate * 0.22)
    return bytes(output)


def pcm16le_to_ulaw(pcm: bytes) -> bytes:
    """Convert signed 16-bit little-endian PCM to G.711 mu-law."""
    encoded = bytearray()
    for (sample,) in struct.iter_unpack("<h", pcm):
        sign = 0x80 if sample < 0 else 0
        magnitude = min(abs(sample), 32635) + 0x84
        exponent = max(0, min(7, magnitude.bit_length() - 8))
        mantissa = (magnitude >> (exponent + 3)) & 0x0F
        encoded.append((~(sign | (exponent << 4) | mantissa)) & 0xFF)
    return bytes(encoded)


def play_rtp(sock: socket.socket, target: tuple[str, int]) -> None:
    ulaw = pcm16le_to_ulaw(beep_pcm())
    sequence = random.randint(0, 65535)
    timestamp = random.randint(0, 2**32 - 1)
    ssrc = random.randint(0, 2**32 - 1)
    started = time.monotonic()
    for offset in range(0, len(ulaw), 160):
        payload = ulaw[offset : offset + 160]
        if len(payload) < 160:
            payload += b"\xff" * (160 - len(payload))
        header = struct.pack("!BBHII", 0x80, 0, sequence, timestamp, ssrc)
        sock.sendto(header + payload, target)
        sequence = (sequence + 1) & 0xFFFF
        timestamp = (timestamp + 160) & 0xFFFFFFFF
        target_time = started + (offset + 160) / 8000
        time.sleep(max(0, target_time - time.monotonic()))


def place_call(client: SipClient, destination: str, answer_timeout: float) -> None:
    rtp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rtp.bind((client.local_ip, 0))
    rtp_port = rtp.getsockname()[1]
    call_id = f"{uuid.uuid4().hex}@{client.advertise_ip}"
    from_tag = uuid.uuid4().hex[:12]
    target_uri = f"sip:{destination}@{client.domain}"
    body = sdp_for(client.advertise_ip, rtp_port)

    invite_branch = f"z9hG4bK-{uuid.uuid4().hex[:16]}"
    cseq = client.send(
        "INVITE",
        target_uri,
        to_uri=target_uri,
        body=body,
        call_id=call_id,
        from_tag=from_tag,
        branch=invite_branch,
    )
    try:
        response = client.final_response(cseq, "INVITE", time.monotonic() + answer_timeout)
    except TimeoutError:
        client.send(
            "CANCEL",
            target_uri,
            to_uri=target_uri,
            call_id=call_id,
            cseq=cseq,
            from_tag=from_tag,
            branch=invite_branch,
        )
        raise TimeoutError("No answer; cancelled the call")
    if response.status in (401, 407):
        # ACK the rejected INVITE before retrying with its challenge.
        client.send(
            "ACK",
            target_uri,
            to_uri=target_uri,
            call_id=call_id,
            cseq=cseq,
            from_tag=from_tag,
            to_tag=tag_from_to(response.headers.get("to", "")),
            branch=invite_branch,
        )
        challenge_name = "www-authenticate" if response.status == 401 else "proxy-authenticate"
        auth = digest_authorization(
            response.headers.get(challenge_name, ""),
            client.username,
            client.password,
            "INVITE",
            target_uri,
        )
        cseq = client.send(
            "INVITE",
            target_uri,
            to_uri=target_uri,
            body=body,
            authorization=auth,
            authorization_name=(
                "Authorization" if response.status == 401 else "Proxy-Authorization"
            ),
            call_id=call_id,
            from_tag=from_tag,
            branch=(invite_branch := f"z9hG4bK-{uuid.uuid4().hex[:16]}"),
        )
        try:
            response = client.final_response(cseq, "INVITE", time.monotonic() + answer_timeout)
        except TimeoutError:
            client.send(
                "CANCEL",
                target_uri,
                to_uri=target_uri,
                call_id=call_id,
                cseq=cseq,
                from_tag=from_tag,
                branch=invite_branch,
            )
            raise TimeoutError("No answer; cancelled the call")

    to_tag = tag_from_to(response.headers.get("to", ""))
    remote_contact = sip_uri(response.headers.get("contact", ""), target_uri)
    client.send(
        "ACK",
        remote_contact,
        to_uri=target_uri,
        call_id=call_id,
        cseq=cseq,
        from_tag=from_tag,
        to_tag=to_tag,
    )
    if response.status != 200:
        raise RuntimeError(f"Call failed: {response.start}")

    rtp_target = remote_rtp(response.body, response.source[0])
    print(f"Answered; sending three beeps to {rtp_target[0]}:{rtp_target[1]}")
    play_rtp(rtp, rtp_target)
    time.sleep(0.25)

    bye_cseq = client.send(
        "BYE",
        remote_contact,
        to_uri=target_uri,
        call_id=call_id,
        from_tag=from_tag,
        to_tag=to_tag,
    )
    bye = client.final_response(bye_cseq, "BYE", time.monotonic() + client.timeout)
    if bye.status != 200:
        raise RuntimeError(f"Hangup failed: {bye.start}")
    print("Call ended cleanly.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="speedport.ip", help="SIP server hostname or IP")
    parser.add_argument("--port", type=int, default=5060, help="SIP UDP port")
    parser.add_argument("--domain", default="speedport.ip", help="SIP registrar domain")
    parser.add_argument("--extension", default="**72", help="SIP phone number/user ID")
    parser.add_argument(
        "--username", default="nutzer-2@speedport.ip", help="Digest authentication username"
    )
    parser.add_argument(
        "--destination",
        default="+4915123412098",
        help="Single destination to call",
    )
    parser.add_argument("--answer-timeout", type=float, default=45, help="Seconds to wait for answer")
    parser.add_argument("--sip-timeout", type=float, default=8, help="SIP response timeout")
    parser.add_argument("--local-port", type=int, default=5060, help="Local SIP UDP port")
    parser.add_argument(
        "--advertise-ip",
        help="LAN address to put in SIP/SDP when calling through a subnet router",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    password = os.environ.get("SPEEDPORT_SIP_PASSWORD") or getpass.getpass(
        "Speedport SIP password: "
    )
    if not password:
        print("No SIP password supplied.", file=sys.stderr)
        return 2
    try:
        client = SipClient(
            args.server,
            args.port,
            args.domain,
            args.extension,
            args.username,
            password,
            args.sip_timeout,
            args.advertise_ip,
            args.local_port,
        )
        client.register()
        place_call(client, args.destination, args.answer_timeout)
        return 0
    except (OSError, RuntimeError, TimeoutError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
