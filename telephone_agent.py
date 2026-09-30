#!/usr/bin/env python3
"""Call a phone number and bridge the answered SIP call to OpenAI GPT-Live."""

from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import ipaddress
import json
import os
import random
import re
import shlex
import socket
import ssl
import struct
import subprocess
import threading
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from speedport_call import (
    SipClient,
    digest_authorization,
    parse_sip,
    remote_rtp,
    sdp_for,
    sip_uri,
    tag_from_to,
)


PROJECT_DIR = Path(__file__).resolve().parent

VOICE_GUIDANCE = """
Follow the call instructions below. Speak naturally and briefly in the requested
language. Listen to the other person's greeting first; ask when details are
unclear and do not invent facts.
Ask one short question at a time. Check each offer against the task's limits
before accepting it; do not correct an invalid acceptance after saying goodbye.
For appointments, clarify conflicting dates and times, then ask the other
person to confirm the final date, time and name before claiming a booking.
Use the calendar context below; never agree to a conflicting weekday.
Do not invent arrangements such as sorting missing details out on arrival.
Backchannel policy: Acknowledge briefly without taking over.
Interruption policy: Yield, listen, then continue appropriately.
An explicit request to hang up overrides the task: stop immediately.
Delegation policy:
Backend tools:
- end_call: Disconnect the telephone call and record its outcome. Only this
  backend capability hangs up; saying goodbye does not disconnect the call.
Delegate to the backend when:
- The other person asks to hang up: delegate immediately, even if the task is
  unfinished. If you already said goodbye, do not say it again.
- The task is finished or cannot be completed: say one short goodbye, then
  immediately delegate to end_call.
Do not delegate to the backend when:
- You are exchanging task details or waiting for a confirmation, unless the
  other person asks to end the call.
The handoff is terminal. Remain silent while it executes; never narrate tools,
announce hanging up, reopen the task, or speak after the final goodbye.
""".strip()

BACKEND_GUIDANCE = """
The voice agent delegates here only to close the call. This is a terminal
handoff, not a request for conversational advice. Always invoke end_call.
Classify the existing conversation: completed if the requested task was confirmed
as done within its limits, unsuccessful if it could not be done, otherwise other.
The caller's request to hang up overrides any unfinished task. Missing details,
invalid offers, or mistakes do not authorize another question or correction.
For appointments, completed requires the other person's confirmation of the
final date, time and name, including any corrections. Check dates against the
calendar context; a conflicting weekday is unresolved. Do not report success
based only on an offer or the agent's own claim. Unresolved details mean other.
Do not generate conversational text, instructions to the voice agent, or a
spoken summary. Record the truthful outcome with end_call and stop.
""".strip()


def calendar_context(now: datetime | None = None) -> str:
    now = now or datetime.now(ZoneInfo("Europe/Berlin"))
    today = now.astimezone(ZoneInfo("Europe/Berlin")).date()
    weekdays = ("Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag")
    lines = ["Calendar context (Europe/Berlin):"]
    for label, offset in (("Heute / today", 0), ("Morgen / tomorrow", 1),
                          ("Übermorgen / day after tomorrow", 2)):
        day = today + timedelta(days=offset)
        lines.append(f"{label}: {weekdays[day.weekday()]}, {day.isoformat()}.")
    return "\n".join(lines)


def read_text_file(path: str | Path, label: str) -> str:
    try:
        value = Path(path).read_text(encoding="utf-8-sig").strip()
    except OSError as error:
        raise ValueError(f"Could not read {label} file {path}: {error.strerror}") from error
    if not value:
        raise ValueError(f"{label.capitalize()} file is empty: {path}")
    return value


def normalize_number(value: str) -> str:
    # Allow common formatting, but only a phone number may enter the SIP URI.
    number = re.sub(r"[ \t().-]", "", value.strip())
    if not re.fullmatch(r"\+?[0-9]{3,20}|\*{1,2}[0-9]{1,10}", number):
        raise ValueError("number.txt must contain one phone number, e.g. +4915123456789")
    return number


def load_call_config(args: argparse.Namespace) -> None:
    args.instructions = read_text_file(args.instructions_file, "instructions")
    value = args.destination if args.destination is not None else read_text_file(args.number_file, "number")
    args.destination = normalize_number(value)


def load_env_value(name: str, path: str = ".env") -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    try:
        with open(path, encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, candidate = line.split("=", 1)
                if key.strip() != name:
                    continue
                candidate = candidate.strip()
                if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in "'\"":
                    candidate = candidate[1:-1]
                return candidate or None
    except FileNotFoundError:
        return None
    return None


def local_lan_interface(server: str) -> str | None:
    """Prefer a directly attached LAN even when a VPN overrides its route."""
    address = ipaddress.ip_address(socket.gethostbyname(server))
    try:
        result = subprocess.run(
            ["ip", "-j", "-4", "route", "show", "table", "main"],
            capture_output=True, text=True, check=True,
        )
        for route in json.loads(result.stdout):
            if (route.get("scope") == "link" and route.get("prefsrc")
                    and address in ipaddress.ip_network(route["dst"], strict=False)):
                return route["dev"]
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError):
        pass
    return None


def route_source_ip(server: str, port: int, interface: str | None = None) -> str:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        if interface:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, interface.encode() + b"\0")
        probe.connect((socket.gethostbyname(server), port))
        return probe.getsockname()[0]
    finally:
        probe.close()


def run_on_lan_host(
    args: argparse.Namespace, api_key: str, sip_password: str
) -> int:
    if not args.remote_host.replace("-", "").replace(".", "").isalnum():
        raise ValueError("Remote host contains unsupported characters")
    remote_dir = f"/tmp/telephone-agent-{uuid.uuid4().hex}"
    ssh_target = args.remote_host
    create = subprocess.run(
        ["ssh", ssh_target, f"mkdir -m 700 -- {shlex.quote(remote_dir)}"],
        check=False,
    )
    if create.returncode:
        raise RuntimeError(f"Could not prepare remote host {ssh_target}")
    try:
        # Send the already validated values, including custom local paths,
        # under fixed remote filenames. The remote directory is mode 700.
        with tempfile.TemporaryDirectory(prefix="telephone-agent-") as staging:
            instructions_path = Path(staging) / "instructions.txt"
            number_path = Path(staging) / "number.txt"
            instructions_path.write_text(args.instructions, encoding="utf-8")
            number_path.write_text(args.destination, encoding="utf-8")
            copy = subprocess.run(
                ["scp", "-q", os.path.abspath(__file__),
                 str(PROJECT_DIR / "speedport_call.py"),
                 str(instructions_path), str(number_path),
                 f"{ssh_target}:{remote_dir}/"],
                check=False,
            )
        if copy.returncode:
            raise RuntimeError(f"Could not copy caller to {ssh_target}")

        forwarded = [
            "python3",
            "telephone_agent.py",
            "--local",
            "--server",
            args.server,
            "--domain",
            args.domain,
            "--port",
            str(args.port),
            "--local-port",
            str(args.local_port),
            "--extension",
            args.extension,
            "--username",
            args.username,
            "--instructions-file",
            "instructions.txt",
            "--number-file",
            "number.txt",
            "--model",
            args.model,
            "--backend-model",
            args.backend_model,
            "--voice",
            args.voice,
            "--answer-timeout",
            str(args.answer_timeout),
            "--sip-timeout",
            str(args.sip_timeout),
            "--max-call-seconds",
            str(args.max_call_seconds),
        ]
        if args.advertise_ip:
            forwarded.extend(["--advertise-ip", args.advertise_ip])
        remote_command = (
            "set -eu; "
            "IFS= read -r OPENAI_API_KEY; "
            "IFS= read -r SPEEDPORT_SIP_PASSWORD; "
            "export OPENAI_API_KEY SPEEDPORT_SIP_PASSWORD; "
            "export PYTHONUNBUFFERED=1; "
            f"cd {shlex.quote(remote_dir)}; "
            f"exec {shlex.join(forwarded)}"
        )
        print(
            f"Routing the call through LAN host {ssh_target}; "
            "credentials are sent over SSH stdin."
        )
        completed = subprocess.run(
            ["ssh", ssh_target, remote_command],
            input=f"{api_key}\n{sip_password}\n",
            text=True,
            check=False,
        )
        return completed.returncode
    finally:
        subprocess.run(
            [
                "ssh",
                ssh_target,
                f"rm -rf -- {shlex.quote(remote_dir)}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


class RawWebSocket:
    """Small RFC 6455 client sufficient for the Live JSON event stream."""

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self.sock: ssl.SSLSocket | None = None
        self.send_lock = threading.Lock()
        self.closed = False
        self.receive_buffer = bytearray()
        self.fragments = bytearray()
        self.text_message = False
        self.started = False
        self.finalized = False

    def connect(self) -> None:
        host = "api.openai.com"
        path = "/v1/live/sessions"
        key = base64.b64encode(os.urandom(16)).decode()
        raw = socket.create_connection((host, 443), timeout=15)
        self.sock = ssl.create_default_context().wrap_socket(raw, server_hostname=host)
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            f"Authorization: Bearer {self.api_key}\r\n"
            "OpenAI-Safety-Identifier: telephone-agent-jaron\r\n"
            "\r\n"
        )
        self.sock.sendall(request.encode())
        response = bytearray()
        while b"\r\n\r\n" not in response:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("Live WebSocket closed during handshake")
            response.extend(chunk)
            if len(response) > 65536:
                raise ConnectionError("Oversized WebSocket handshake response")
        raw_header, extra = bytes(response).split(b"\r\n\r\n", 1)
        self.receive_buffer.extend(extra)
        header = raw_header.decode("latin-1")
        status = header.split("\r\n", 1)[0]
        if " 101 " not in status:
            raise ConnectionError(f"Live WebSocket handshake failed: {status}")
        expected = base64.b64encode(
            hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()
        ).decode()
        headers = {}
        for line in header.split("\r\n")[1:]:
            name, separator, value = line.partition(":")
            if separator:
                headers[name.lower()] = value.strip()
        if headers.get("sec-websocket-accept") != expected:
            raise ConnectionError("Invalid WebSocket handshake signature")
        self.sock.settimeout(0.5)

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        if not self.sock or self.closed:
            raise ConnectionError("Live WebSocket is closed")
        mask = os.urandom(4)
        length = len(payload)
        header = bytearray([0x80 | opcode])
        if length < 126:
            header.append(0x80 | length)
        elif length <= 65535:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", length))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", length))
        masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        with self.send_lock:
            self.sock.sendall(bytes(header) + mask + masked)

    def send_json(self, event: dict) -> None:
        self._send_frame(0x1, json.dumps(event, separators=(",", ":")).encode())

    def recv_text(self) -> str | None:
        """Retain partial frames/messages across socket timeouts."""
        if not self.sock:
            raise ConnectionError("Live WebSocket is not connected")
        while True:
            data = self.receive_buffer
            if len(data) >= 2:
                first, second = data[:2]
                length = second & 0x7F
                offset = 2
                if length == 126:
                    if len(data) < 4:
                        length = None
                    else:
                        length = struct.unpack("!H", data[2:4])[0]
                        offset = 4
                elif length == 127:
                    if len(data) < 10:
                        length = None
                    else:
                        length = struct.unpack("!Q", data[2:10])[0]
                        offset = 10
                masked = bool(second & 0x80)
                if length is not None:
                    payload_offset = offset + (4 if masked else 0)
                    if len(data) >= payload_offset + length:
                        mask = data[offset:payload_offset]
                        payload = bytes(data[payload_offset:payload_offset + length])
                        del data[:payload_offset + length]
                        if masked:
                            payload = bytes(v ^ mask[i % 4] for i, v in enumerate(payload))
                        opcode = first & 0x0F
                        if opcode == 0x8:
                            self.closed = True
                            return None
                        if opcode == 0x9:
                            self._send_frame(0xA, payload)
                        elif opcode == 0x1:
                            self.text_message = True
                            self.fragments.extend(payload)
                        elif opcode == 0x0:
                            self.fragments.extend(payload)
                        if first & 0x80 and opcode in (0x0, 0x1) and self.text_message:
                            message = self.fragments.decode("utf-8")
                            self.fragments.clear()
                            self.text_message = False
                            return message
                        continue
            # A timeout leaves receive_buffer and fragments intact.
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("Live WebSocket closed unexpectedly")
            self.receive_buffer.extend(chunk)

    def close(self) -> None:
        if not self.closed:
            try:
                self._send_frame(0x8, struct.pack("!H", 1000))
            except (OSError, ConnectionError):
                pass
        self.closed = True
        if self.sock:
            self.sock.close()


def pcmu_has_speech(audio: bytes) -> bool:
    """Small output noise gate for hangup timing, not model turn detection."""
    if not audio:
        return False
    energy = 0
    for value in audio:
        value = (~value) & 0xFF
        magnitude = (((value & 15) << 3) + 132) << ((value >> 4) & 7)
        energy += magnitude - 132
    return energy / len(audio) > 150


class RtpAudioSender:
    def __init__(self, sock: socket.socket, target: tuple[str, int]) -> None:
        self.sock = sock
        self.target = target
        self.buffer = bytearray()
        self.condition = threading.Condition()
        self.stopped = False
        self.accepting_audio = True
        self.in_flight = False
        self.received_bytes = 0
        self.played_bytes = 0
        self.last_speech_byte = 0
        self.last_speech_played_at = time.monotonic()
        self.error: OSError | None = None
        self.sequence = random.randint(0, 65535)
        self.timestamp = random.randint(0, 2**32 - 1)
        self.ssrc = random.randint(0, 2**32 - 1)
        self.thread = threading.Thread(target=self._run, name="rtp-output", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def add(self, pcmu: bytes) -> None:
        with self.condition:
            if not self.accepting_audio:
                return
            self.buffer.extend(pcmu)
            self.received_bytes += len(pcmu)
            if pcmu_has_speech(pcmu):
                self.last_speech_byte = self.received_bytes
            self.condition.notify()

    def finish(self) -> None:
        """Freeze playback at the already queued goodbye, retaining its tail."""
        with self.condition:
            self.accepting_audio = False
            # RTP uses complete 160-byte packets; a final partial packet must
            # be padded rather than waiting indefinitely for more model audio.
            self.buffer.extend(b"\xff" * (-len(self.buffer) % 160))
            self.condition.notify_all()

    def speech_drained(self, quiet_seconds: float = 1.0) -> bool:
        with self.condition:
            return (
                self.played_bytes >= self.last_speech_byte
                and time.monotonic() - self.last_speech_played_at >= quiet_seconds
            )

    def stop(self) -> None:
        with self.condition:
            self.stopped = True
            self.condition.notify()
        self.thread.join(timeout=2)

    def _run(self) -> None:
        next_packet = time.monotonic()
        while True:
            with self.condition:
                while len(self.buffer) < 160 and not self.stopped:
                    self.condition.wait(timeout=0.02)
                    if len(self.buffer) < 160:
                        break
                if self.stopped:
                    return
                model_audio = len(self.buffer) >= 160
                if model_audio:
                    payload = bytes(self.buffer[:160])
                    del self.buffer[:160]
                    self.in_flight = True
                else:
                    # G.711 μ-law 0xFF represents digital silence. Keep RTP
                    # flowing between turns like a normal IP telephone.
                    payload = b"\xff" * 160
            now = time.monotonic()
            if next_packet < now - 0.1:
                next_packet = now
            if next_packet > now:
                time.sleep(next_packet - now)
            with self.condition:
                header = struct.pack(
                    "!BBHII", 0x80, 0, self.sequence, self.timestamp, self.ssrc
                )
                try:
                    self.sock.sendto(header + payload, self.target)
                except OSError as error:
                    self.error = error
                    return
                if model_audio:
                    self.played_bytes += len(payload)
                    if pcmu_has_speech(payload):
                        self.last_speech_played_at = time.monotonic()
                    self.in_flight = False
                self.condition.notify_all()
            self.sequence = (self.sequence + 1) & 0xFFFF
            self.timestamp = (self.timestamp + 160) & 0xFFFFFFFF
            next_packet += 0.02


def rtp_payload(packet: bytes) -> bytes:
    if len(packet) < 12 or packet[0] >> 6 != 2:
        return b""
    if packet[1] & 0x7F != 0:
        return b""
    contributing_sources = packet[0] & 0x0F
    offset = 12 + contributing_sources * 4
    if packet[0] & 0x10:
        if len(packet) < offset + 4:
            return b""
        extension_words = struct.unpack("!H", packet[offset + 2 : offset + 4])[0]
        offset += 4 + extension_words * 4
    if offset > len(packet):
        return b""
    end = len(packet)
    if packet[0] & 0x20 and packet[-1] <= end - offset:
        end -= packet[-1]
    return packet[offset:end]


@dataclass
class ActiveCall:
    rtp: socket.socket
    rtp_target: tuple[str, int]
    call_id: str
    from_tag: str
    to_tag: str | None
    target_uri: str
    remote_contact: str


def invite(client: SipClient, destination: str, answer_timeout: float) -> ActiveCall:
    rtp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if client.interface:
        rtp.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, client.interface.encode() + b"\0")
    rtp.bind((client.local_ip, 0))
    rtp.settimeout(0.02)
    rtp_port = rtp.getsockname()[1]
    call_id = f"{os.urandom(16).hex()}@{client.advertise_ip}"
    from_tag = os.urandom(6).hex()
    target_uri = f"sip:{destination}@{client.domain}"
    body = sdp_for(client.advertise_ip, rtp_port, direction="sendrecv")
    branch = f"z9hG4bK-{os.urandom(8).hex()}"

    cseq = client.send(
        "INVITE",
        target_uri,
        to_uri=target_uri,
        body=body,
        call_id=call_id,
        from_tag=from_tag,
        branch=branch,
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
            branch=branch,
        )
        raise TimeoutError("No answer; cancelled the call")

    if response.status in (401, 407):
        client.send(
            "ACK",
            target_uri,
            to_uri=target_uri,
            call_id=call_id,
            cseq=cseq,
            from_tag=from_tag,
            to_tag=tag_from_to(response.headers.get("to", "")),
            branch=branch,
        )
        challenge_name = "www-authenticate" if response.status == 401 else "proxy-authenticate"
        authorization = digest_authorization(
            response.headers.get(challenge_name, ""),
            client.username,
            client.password,
            "INVITE",
            target_uri,
        )
        branch = f"z9hG4bK-{os.urandom(8).hex()}"
        cseq = client.send(
            "INVITE",
            target_uri,
            to_uri=target_uri,
            body=body,
            authorization=authorization,
            authorization_name=(
                "Authorization" if response.status == 401 else "Proxy-Authorization"
            ),
            call_id=call_id,
            from_tag=from_tag,
            branch=branch,
        )
        try:
            response = client.final_response(
                cseq, "INVITE", time.monotonic() + answer_timeout
            )
        except TimeoutError:
            client.send(
                "CANCEL",
                target_uri,
                to_uri=target_uri,
                call_id=call_id,
                cseq=cseq,
                from_tag=from_tag,
                branch=branch,
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
    return ActiveCall(
        rtp=rtp,
        rtp_target=remote_rtp(response.body, response.source[0]),
        call_id=call_id,
        from_tag=from_tag,
        to_tag=to_tag,
        target_uri=target_uri,
        remote_contact=remote_contact,
    )


def sip_response(client: SipClient, request, status: int, reason: str) -> None:
    headers = [
        f"SIP/2.0 {status} {reason}",
        f"Via: {request.headers.get('via', '')}",
        f"From: {request.headers.get('from', '')}",
        f"To: {request.headers.get('to', '')}",
        f"Call-ID: {request.headers.get('call-id', '')}",
        f"CSeq: {request.headers.get('cseq', '')}",
        "Content-Length: 0",
        "",
        "",
    ]
    client.sock.send("\r\n".join(headers).encode())


def monitor_sip(
    client: SipClient,
    call: ActiveCall,
    call_started_at: float,
    stop: threading.Event,
    remote_hangup: threading.Event,
) -> None:
    while not stop.is_set():
        readable, _, _ = select_with_timeout(client.sock, 0.2)
        if not readable:
            continue
        try:
            data = client.sock.recv(65535)
        except BlockingIOError:
            continue
        message = parse_sip(data, (client.server_ip, client.server_port))
        method = message.start.split(" ", 1)[0]
        if method == "BYE":
            received_call_id = message.headers.get("call-id", "")
            if received_call_id != call.call_id:
                sip_response(client, message, 481, "Call/Transaction Does Not Exist")
                print(
                    "Ignored SIP BYE for a different dialog "
                    f"(Call-ID {received_call_id or 'missing'})."
                )
                continue
            sip_response(client, message, 200, "OK")
            detail = (
                message.headers.get("reason")
                or message.headers.get("warning")
                or message.headers.get("user-agent")
                or "no Reason header"
            )
            from_tag = tag_from_to(message.headers.get("from", ""))
            to_tag = tag_from_to(message.headers.get("to", ""))
            elapsed = time.monotonic() - call_started_at
            print(
                f"Speedport ended the active SIP dialog after {elapsed:.1f}s: "
                f"{detail}"
            )
            print(
                "BYE dialog tags: "
                f"From={from_tag or 'missing'} "
                f"(expected remote {call.to_tag or 'missing'}), "
                f"To={to_tag or 'missing'} (expected local {call.from_tag}); "
                f"CSeq={message.headers.get('cseq', 'missing')}"
            )
            remote_hangup.set()
            stop.set()
            return
        if method == "OPTIONS":
            sip_response(client, message, 200, "OK")


def select_with_timeout(sock: socket.socket, timeout: float) -> tuple[list, list, list]:
    import select

    return select.select([sock], [], [], timeout)


def stream_rtp_to_live(
    rtp: socket.socket, websocket: RawWebSocket, stop: threading.Event,
    caller_speaking: threading.Event, errors: list[Exception],
) -> None:
    audio = bytearray()
    last_send = time.monotonic()
    last_receive = last_send
    received_audio = False
    try:
        while not stop.is_set():
            try:
                packet, _ = rtp.recvfrom(4096)
                payload = rtp_payload(packet)
                if payload:
                    if not received_audio:
                        print("Inbound RTP audio received.", flush=True)
                        received_audio = True
                    audio.extend(payload)
                    last_receive = time.monotonic()
                    if pcmu_has_speech(payload):
                        caller_speaking.set()
            except socket.timeout:
                # Live needs a continuous stream, including when the phone
                # suppresses silence or hasn't sent its first RTP packet yet.
                now = time.monotonic()
                if now - last_receive >= 0.02:
                    audio.extend(b"\xff" * 160)
                    last_receive = now
            now = time.monotonic()
            if audio and (len(audio) >= 800 or now - last_send >= 0.1):
                websocket.send_json({
                    "type": "session.input_audio.append",
                    "audio": base64.b64encode(bytes(audio)).decode(),
                })
                audio.clear()
                last_send = now
    except (OSError, ConnectionError) as error:
        errors.append(error)
        stop.set()


def live_session_event(
    model: str, backend_model: str = "gpt-6-luna", voice: str = "marin",
    instructions: str = "Say hello briefly.",
) -> dict:
    context = calendar_context() + "\n\nCall instructions:\n" + instructions
    return {
        "type": "session.start",
        "session": {
            "model": model,
            "instructions": VOICE_GUIDANCE + "\n\n" + context,
            "audio": {
                "format": {"type": "audio/pcmu", "rate": 8000},
                "output": {"voice": voice},
            },
            "delegation": {
                "type": "responses",
                "responses": {
                    "model": backend_model,
                    "instructions": BACKEND_GUIDANCE + "\n\n" + context,
                    "tools": [{
                        "type": "function", "name": "end_call",
                        "description": "Record the outcome and hang up after the spoken goodbye.",
                        "parameters": {
                            "type": "object",
                            "properties": {"outcome": {
                                "type": "string",
                                "enum": ["completed", "unsuccessful", "other"],
                            }},
                            "required": ["outcome"],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    }],
                    "tool_choice": {"type": "function", "name": "end_call"},
                    "parallel_tool_calls": False,
                },
            },
        },
    }


def live_error(event: dict) -> RuntimeError:
    error = event.get("error", {})
    return RuntimeError(f"Live API error: {error.get('code')}: {error.get('message')}")


def start_live_session(websocket: RawWebSocket, config: dict, timeout: float = 15) -> None:
    websocket.send_json(config)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            raw = websocket.recv_text()
        except socket.timeout:
            continue
        if raw is None:
            raise ConnectionError("Live connection closed before session.started")
        event = json.loads(raw)
        if event.get("type") == "error":
            raise live_error(event)
        if event.get("type") == "session.started":
            websocket.started = True
            print(f"Live session ready: {event['session']['id']}")
            return
    raise TimeoutError("No session.started received from GPT-Live")


def finalize_live_session(websocket: RawWebSocket, timeout: float = 15) -> bool:
    if websocket.finalized:
        return True
    if not websocket.started or websocket.closed:
        return False
    websocket.send_json({"type": "session.close"})
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            raw = websocket.recv_text()
        except socket.timeout:
            continue
        if raw is None:
            break
        event = json.loads(raw)
        if event.get("type") == "session.closed":
            websocket.finalized = True
            print(f"Final Live usage: {json.dumps(event.get('usage', {}))}")
            return True
        if event.get("type") == "error":
            raise live_error(event)
    raise TimeoutError("Incomplete Live finalization: no session.closed event")


class LiveCallEvents:
    """Track delegated tools independently of the continuous voice stream."""

    def __init__(self, websocket: RawWebSocket, sender: RtpAudioSender | None = None) -> None:
        self.websocket = websocket
        self.sender = sender
        self.end_requested_at: float | None = None
        self.end_outcome: str | None = None
        self.conversation_started = False
        self.handled_calls: set[str] = set()
        self.pending_results: set[str] = set()
        self.response_ids: dict[str, str] = {}
        self.transcripts: list[dict] = []

    def handle(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "session.output_audio.delta":
            if self.sender and self.end_requested_at is None:
                self.sender.add(base64.b64decode(event.get("delta", "")))
            return
        if kind == "error":
            raise live_error(event)
        if kind == "session.closed":
            self.websocket.finalized = True
            print(f"Final Live usage: {json.dumps(event.get('usage', {}))}")
            raise ConnectionError("Live session ended during the call")
        if kind in ("session.input_transcript.delta", "session.output_transcript.delta"):
            self.conversation_started = True
            self.transcripts.append(event)
            speaker = "Other person" if kind == "session.input_transcript.delta" else "Agent"
            # Preserve each fragment exactly: Live has no turn-completed event.
            print(f"{speaker} [{event.get('start_ms')}–{event.get('end_ms')} ms]: "
                  f"{event.get('delta', '')}", flush=True)
        elif kind == "response.event":
            nested = event.get("event", {})
            if nested.get("type") == "response.output_text.delta":
                print(f"Backend (not spoken): {nested.get('delta', '')}", flush=True)
            delegation_id = event.get("delegation_id", "")
            response_id = nested.get("response", {}).get("id")
            if nested.get("type") == "response.created" and response_id:
                self.response_ids[delegation_id] = response_id
            if nested.get("type") == "response.output_item.done":
                item = nested.get("item", {})
                if item.get("type") != "function_call":
                    return
                call_id = item.get("call_id")
                if not call_id:
                    raise RuntimeError("Delegated function has no call_id")
                if call_id in self.handled_calls:
                    return
                if self.end_requested_at is not None:
                    return  # A terminal tool cannot start another conversation.
                self.handled_calls.add(call_id)
                tool_response_id = nested.get("response_id") or self.response_ids.get(delegation_id)
                if not tool_response_id:
                    raise RuntimeError("Delegated function has no associated response")
                arguments = json.loads(item.get("arguments", "{}"))
                outcome = arguments.get("outcome")
                if item.get("name") != "end_call" or outcome not in {
                    "completed", "unsuccessful", "other"
                }:
                    result = {"ok": False, "error": "Unsupported function or outcome"}
                else:
                    if self.end_requested_at is None:
                        self.end_requested_at = time.monotonic()
                    self.end_outcome = outcome
                    if self.sender:
                        self.sender.finish()
                    result = {"ok": True, "outcome": outcome}
                    print(f"Agent requested hangup: {outcome}")
                self.websocket.send_json({
                    "type": "response.item.create",
                    "item": {"type": "function_call_output", "call_id": call_id,
                             "output": json.dumps(result)},
                })
                # Forwarded item events don't include a response_id. Use the
                # response.created envelope for this delegation instead.
                if self.end_requested_at is None:
                    self.pending_results.add(tool_response_id)
            elif nested.get("type") == "response.completed":
                if response_id in self.pending_results and self.end_requested_at is None:
                    self.pending_results.remove(response_id)
                    self.websocket.send_json({"type": "response.create"})
            elif nested.get("type") in ("response.failed", "response.incomplete"):
                raise RuntimeError(f"Delegated backend failed: {nested.get('response')}")


def bridge_live(
    client: SipClient, call: ActiveCall, websocket: RawWebSocket, max_seconds: float,
) -> bool:
    stop = threading.Event()
    remote_hangup = threading.Event()
    caller_speaking = threading.Event()
    errors: list[Exception] = []
    sender = RtpAudioSender(call.rtp, call.rtp_target)
    sender.start()
    started = time.monotonic()
    rtp_thread = threading.Thread(
        target=stream_rtp_to_live,
        args=(call.rtp, websocket, stop, caller_speaking, errors),
        name="rtp-input", daemon=True,
    )
    sip_thread = threading.Thread(
        target=monitor_sip, args=(client, call, started, stop, remote_hangup),
        name="sip-monitor", daemon=True,
    )
    rtp_thread.start()
    sip_thread.start()
    events = LiveCallEvents(websocket, sender)
    greeting_sent = False
    try:
        while not stop.is_set() and time.monotonic() - started < max_seconds:
            if sender.error:
                raise sender.error
            now = time.monotonic()
            if (not greeting_sent and not events.conversation_started
                    and not caller_speaking.is_set() and now - started >= 2):
                websocket.send_json({
                    "type": "session.instructions.append", "delegation_id": None,
                    "content": "The other person is silent. Open the call briefly according "
                               "to the call instructions, then listen.",
                })
                greeting_sent = True
            if events.end_requested_at is not None and sender.speech_drained():
                break
            try:
                raw = websocket.recv_text()
            except socket.timeout:
                continue
            if raw is None:
                raise ConnectionError("Live WebSocket closed during the call")
            event = json.loads(raw)
            events.handle(event)
        else:
            if not stop.is_set():
                print("Maximum call duration reached.")
        if errors:
            raise errors[0]
    finally:
        stop.set()
        rtp_thread.join(timeout=2)
        sip_thread.join(timeout=2)
        sender.stop()
    return remote_hangup.is_set()


def check_live_api(args: argparse.Namespace, api_key: str) -> int:
    """Exercise the production session schema/audio without touching SIP."""
    websocket = RawWebSocket(api_key, args.model)
    stop = threading.Event()
    audio_bytes = 0
    transcript = ""
    errors: list[Exception] = []
    def send_silence() -> None:
        try:
            while not stop.wait(0.1):
                websocket.send_json({"type": "session.input_audio.append",
                                     "audio": base64.b64encode(b"\xff" * 800).decode()})
        except (OSError, ConnectionError) as error:
            errors.append(error)
    thread = threading.Thread(target=send_silence, daemon=True)
    try:
        websocket.connect()
        start_live_session(websocket, live_session_event(args.model, args.backend_model, args.voice))
        thread.start()
        websocket.send_json({
            "type": "session.instructions.append", "delegation_id": None,
            "content": "Dies ist nur ein Verbindungstest, kein Telefonat. Sage jetzt "
                       "kurz Guten Tag auf Deutsch. Führe keine Aufgabe aus und delegiere nichts.",
        })
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                raw = websocket.recv_text()
            except socket.timeout:
                continue
            if raw is None:
                raise ConnectionError("Live connection closed during API check")
            event = json.loads(raw)
            if event.get("type") == "error":
                raise live_error(event)
            if event.get("type") == "session.output_audio.delta":
                audio = base64.b64decode(event.get("delta", ""))
                if pcmu_has_speech(audio):
                    audio_bytes += len(audio)
            elif event.get("type") == "session.output_transcript.delta":
                transcript += event.get("delta", "")
            if audio_bytes >= 1600 and transcript.strip():
                break
        stop.set()
        thread.join(timeout=2)
        if errors:
            raise errors[0]
        if not audio_bytes or not transcript.strip():
            raise RuntimeError("API check received no speech audio or transcript")
        print(f"API check passed: {audio_bytes} PCMU speech bytes; Agent: {transcript}")
        finalize_live_session(websocket)
        return 0
    except (OSError, ValueError, RuntimeError, TimeoutError, ConnectionError) as error:
        print(f"Error: {error}")
        return 1
    finally:
        stop.set()
        if thread.ident is not None:
            thread.join(timeout=2)
        if websocket.started and not websocket.finalized and not websocket.closed:
            try:
                finalize_live_session(websocket)
            except (OSError, RuntimeError) as error:
                print(f"Warning: {error}")
        websocket.close()


def hangup(client: SipClient, call: ActiveCall) -> None:
    cseq = client.send(
        "BYE",
        call.remote_contact,
        to_uri=call.target_uri,
        call_id=call.call_id,
        from_tag=call.from_tag,
        to_tag=call.to_tag,
    )
    response = client.final_response(cseq, "BYE", time.monotonic() + client.timeout)
    if response.status != 200:
        raise RuntimeError(f"Hangup failed: {response.start}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="192.168.2.1")
    parser.add_argument("--domain", default="speedport.ip")
    parser.add_argument("--port", type=int, default=5060)
    parser.add_argument("--local-port", type=int, default=5060)
    parser.add_argument("--advertise-ip")
    parser.add_argument("--interface", help="Bind SIP and audio to this network interface; auto-detects a direct LAN")
    parser.add_argument("--extension", default="**72")
    parser.add_argument("--username", default="nutzer-2@speedport.ip")
    parser.add_argument("--instructions-file", default=str(PROJECT_DIR / "instructions.txt"),
                        help="UTF-8 text file containing the call instructions")
    parser.add_argument("--number-file", default=str(PROJECT_DIR / "number.txt"),
                        help="Text file containing the phone number")
    parser.add_argument("--destination", help="Override the number file for this call")
    parser.add_argument("--model", default="gpt-live-1")
    parser.add_argument("--backend-model", default="gpt-6-luna")
    parser.add_argument("--voice", default="marin")
    parser.add_argument("--check-api", action="store_true",
                        help="Test Live audio and session lifecycle without dialing")
    parser.add_argument("--answer-timeout", type=float, default=45)
    parser.add_argument("--sip-timeout", type=float, default=15)
    parser.add_argument("--max-call-seconds", type=float, default=180)
    parser.add_argument(
        "--remote-host",
        default=None,
        help="Opt in to forwarding through this SSH host when outside the router LAN",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = load_env_value("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY is missing from the environment or .env")
        return 2
    if args.check_api:
        return check_live_api(args, api_key)
    try:
        load_call_config(args)
    except ValueError as error:
        print(f"Error: {error}")
        return 2
    sip_password = load_env_value("SPEEDPORT_SIP_PASSWORD") or getpass.getpass(
        "Speedport SIP password: "
    )
    if not sip_password:
        print("Error: no Speedport SIP password supplied")
        return 2

    args.interface = args.interface or local_lan_interface(args.server)
    source_ip = route_source_ip(args.server, args.port, args.interface)
    server_ip = socket.gethostbyname(args.server)
    on_server_lan = source_ip.rsplit(".", 1)[0] == server_ip.rsplit(".", 1)[0]
    if args.remote_host and not args.local and not on_server_lan:
        try:
            return run_on_lan_host(args, api_key, sip_password)
        except (OSError, RuntimeError, ValueError) as error:
            print(f"Error: {error}")
            return 1

    websocket = RawWebSocket(api_key, args.model)
    print(f"Running locally; phone interface: {args.interface or 'system route'} ({source_ip})")
    client = SipClient(
        args.server,
        args.port,
        args.domain,
        args.extension,
        args.username,
        sip_password,
        args.sip_timeout,
        args.advertise_ip,
        args.local_port,
        interface=args.interface,
    )
    call: ActiveCall | None = None
    sip_ended = False
    try:
        client.register()
        call = invite(client, args.destination, args.answer_timeout)
        print(f"Answered; connecting Live model {args.model}…")
        websocket.connect()
        print(
            f"Answered; bridging PCMU audio with {call.rtp_target[0]}:"
            f"{call.rtp_target[1]}"
        )
        start_live_session(websocket, live_session_event(
            args.model, args.backend_model, args.voice, args.instructions
        ))
        remote_hung_up = bridge_live(
            client, call, websocket, args.max_call_seconds
        )
        sip_ended = remote_hung_up
        if not remote_hung_up:
            hangup(client, call)
            sip_ended = True
            print("Call ended cleanly.")
        else:
            print("Remote party ended the call.")
        finalize_live_session(websocket)
        return 0
    except (OSError, ValueError, RuntimeError, TimeoutError, ConnectionError) as error:
        print(f"Error: {error}")
        if call and not sip_ended:
            try:
                hangup(client, call)
            except Exception:
                pass
        return 1
    finally:
        if websocket.started and not websocket.finalized and not websocket.closed:
            try:
                finalize_live_session(websocket)
            except (OSError, RuntimeError) as error:
                print(f"Warning: {error}")
        websocket.close()
        if call:
            call.rtp.close()
        client.sock.close()


if __name__ == "__main__":
    raise SystemExit(main())
