import base64
import json
import socket
import struct
import threading
import time
import unittest
import argparse
import tempfile
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from unittest.mock import Mock, patch

from telephone_agent import (
    LiveCallEvents, RawWebSocket, RtpAudioSender, live_session_event,
    pcmu_has_speech, rtp_payload, start_live_session, stream_rtp_to_live,
    load_call_config, normalize_number,
    local_lan_interface, parse_args,
    calendar_context,
)


def frame(payload, opcode=1, final=True):
    payload = payload.encode() if isinstance(payload, str) else payload
    return bytes([(0x80 if final else 0) | opcode, len(payload)]) + payload


class LiveBridgeTests(unittest.TestCase):
    def test_direct_lan_wins_over_vpn_and_ssh_is_opt_in(self):
        routes = [{'dst': 'default', 'dev': 'enp3s0', 'gateway': '192.168.2.1'},
                  {'dst': '192.168.2.0/24', 'dev': 'enp3s0', 'scope': 'link',
                   'prefsrc': '192.168.2.196'}]
        with patch('telephone_agent.subprocess.run', return_value=Mock(stdout=json.dumps(routes))):
            self.assertEqual(local_lan_interface('192.168.2.1'), 'enp3s0')
            self.assertIsNone(local_lan_interface('192.168.3.1'))
        with patch('sys.argv', ['telephone_agent.py']):
            self.assertIsNone(parse_args().remote_host)

    def test_text_files_supply_task_to_voice_and_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            instructions = Path(directory) / 'task.txt'
            number = Path(directory) / 'phone.txt'
            instructions.write_text('Ask whether my computer repair is finished.', encoding='utf-8')
            number.write_text('+49 (151) 234-56789\n', encoding='utf-8')
            args = argparse.Namespace(instructions_file=instructions, number_file=number,
                                      destination=None)
            load_call_config(args)
            self.assertEqual(args.destination, '+4915123456789')
            session = live_session_event('gpt-live-1', instructions=args.instructions)['session']
            self.assertIn(args.instructions, session['instructions'])
            self.assertIn(args.instructions, session['delegation']['responses']['instructions'])
            instructions.write_text('  \n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'empty'):
                load_call_config(args)

    def test_number_file_rejects_multiple_numbers_and_sip_injection(self):
        for number in ['+491234\n+495678', '+491234@another-host', '', 'call Bob']:
            with self.subTest(number=number), self.assertRaises(ValueError):
                normalize_number(number)
        self.assertEqual(normalize_number('**72'), '**72')

    def test_partial_websocket_frames_survive_timeout_and_ping(self):
        ws = RawWebSocket('unused', 'gpt-live-1')
        ws.sock = Mock()
        ws.sock.recv.side_effect = [b'\x81', socket.timeout(), b'\x05hello']
        with self.assertRaises(socket.timeout):
            ws.recv_text()
        self.assertEqual(ws.recv_text(), 'hello')
        ws.receive_buffer.extend(frame('hel', final=False) + frame(b'!', opcode=9))
        ws.sock.recv.side_effect = [socket.timeout(), frame('lo', opcode=0)]
        with self.assertRaises(socket.timeout):
            ws.recv_text()
        self.assertEqual(ws.recv_text(), 'hello')
        self.assertTrue(ws.sock.sendall.called)  # pong sent while message incomplete

    def test_session_must_start_before_commands(self):
        ws = Mock()
        ws.recv_text.side_effect = [socket.timeout(), json.dumps({
            'type': 'session.started', 'session': {'id': 'live_test'},
        })]
        start_live_session(ws, live_session_event('gpt-live-1'))
        self.assertTrue(ws.started)
        self.assertEqual(ws.send_json.call_args.args[0]['type'], 'session.start')
        failed = Mock()
        failed.recv_text.return_value = json.dumps({
            'type': 'error', 'error': {'code': 'invalid_model', 'message': 'Unavailable'},
        })
        with self.assertRaisesRegex(RuntimeError, 'invalid_model'):
            start_live_session(failed, live_session_event('unavailable'))

    def test_terminal_tool_does_not_restart_backend_without_item_response_id(self):
        ws = Mock()
        handler = LiveCallEvents(ws)
        def backend(event):
            handler.handle({'type': 'response.event', 'delegation_id': 'd1', 'event': event})
        backend({'type': 'response.created', 'response': {'id': 'r1'}})
        item = {'type': 'response.output_item.done', 'item': {
            'type': 'function_call', 'call_id': 'c1', 'name': 'end_call',
            'arguments': '{"outcome":"completed"}',
        }}
        backend(item)
        backend(item)  # duplicate delivery must not execute twice
        self.assertEqual(ws.send_json.call_count, 1)
        result = ws.send_json.call_args.args[0]
        self.assertEqual(result['type'], 'response.item.create')
        self.assertEqual(result['item']['call_id'], 'c1')
        backend({'type': 'response.completed', 'response': {'id': 'r1', 'output': []}})
        self.assertEqual(ws.send_json.call_count, 1)
        self.assertEqual(handler.end_outcome, 'completed')
        backend({'type': 'response.created', 'response': {'id': 'r2'}})
        backend({'type': 'response.completed', 'response': {'id': 'r2', 'output': []}})
        self.assertEqual(ws.send_json.call_count, 1)

    def test_late_audio_cannot_extend_goodbye_or_cut_off_partial_tail(self):
        sender = RtpAudioSender(Mock(), ('127.0.0.1', 12345))
        handler = LiveCallEvents(Mock(), sender)
        goodbye = b'\x90' * 197  # Final speech doesn't fill a 20 ms packet.
        handler.handle({'type': 'session.output_audio.delta',
                        'delta': base64.b64encode(goodbye).decode()})
        handler.handle({'type': 'response.event', 'delegation_id': 'd', 'event': {
            'type': 'response.created', 'response': {'id': 'r'},
        }})
        handler.handle({'type': 'response.event', 'delegation_id': 'd', 'event': {
            'type': 'response.output_item.done', 'item': {
                'type': 'function_call', 'call_id': 'c', 'name': 'end_call',
                'arguments': '{"outcome":"other"}',
            },
        }})
        self.assertEqual(sender.buffer, goodbye + b'\xff' * 123)
        handler.handle({'type': 'session.output_audio.delta',
                        'delta': base64.b64encode(b'\x90' * 8000).decode()})
        sender.add(b'\x90' * 8000)  # Gate also protects direct sender users.
        self.assertEqual(sender.buffer, goodbye + b'\xff' * 123)
        self.assertFalse(sender.speech_drained(0))
        sender.start()
        try:
            deadline = time.monotonic() + 1
            while not sender.speech_drained(0) and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(sender.speech_drained(0))
            packets = [call.args[0] for call in sender.sock.sendto.call_args_list]
            played = b''.join(rtp_payload(packet) for packet in packets)
            self.assertTrue(played.startswith(goodbye))
            self.assertFalse(pcmu_has_speech(played[len(goodbye):]))
        finally:
            sender.stop()

    def test_calendar_uses_berlin_weekdays_and_month_rollover(self):
        context = calendar_context(datetime(2026, 9, 30, 13, tzinfo=ZoneInfo('Europe/Berlin')))
        self.assertIn('Heute / today: Mittwoch, 2026-09-30', context)
        self.assertIn('Morgen / tomorrow: Donnerstag, 2026-10-01', context)
        self.assertIn('Übermorgen / day after tomorrow: Freitag, 2026-10-02', context)
        context = calendar_context(datetime(2026, 9, 30, 23, tzinfo=ZoneInfo('UTC')))
        self.assertIn('Heute / today: Donnerstag, 2026-10-01', context)

    def test_invalid_tool_cannot_schedule_hangup(self):
        ws = Mock()
        handler = LiveCallEvents(ws)
        handler.handle({'type': 'response.event', 'delegation_id': 'd', 'event': {
            'type': 'response.created', 'response': {'id': 'r'},
        }})
        handler.handle({'type': 'response.event', 'delegation_id': 'd', 'event': {
            'type': 'response.output_item.done', 'item': {
                'type': 'function_call', 'call_id': 'c', 'name': 'end_call',
                'arguments': '{"outcome":"invalid"}',
            },
        }})
        self.assertIsNone(handler.end_requested_at)
        self.assertFalse(json.loads(ws.send_json.call_args.args[0]['item']['output'])['ok'])
        handler.handle({'type': 'response.event', 'delegation_id': 'd', 'event': {
            'type': 'response.completed', 'response': {'id': 'r'},
        }})
        self.assertEqual(ws.send_json.call_args.args[0], {'type': 'response.create'})

    def test_both_transcript_streams_keep_exact_fragments_and_times(self):
        handler = LiveCallEvents(Mock())
        for kind, delta in [('input', ' Ja'), ('output', ' Danke'), ('input', ' genau.')]:
            handler.handle({'type': f'session.{kind}_transcript.delta', 'delta': delta,
                            'start_ms': 100, 'end_ms': 200})
        self.assertEqual([e['delta'] for e in handler.transcripts], [' Ja', ' Danke', ' genau.'])
        self.assertEqual(handler.transcripts[-1]['end_ms'], 200)

    def test_silent_phone_still_supplies_continuous_live_audio(self):
        rtp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        rtp.bind(('127.0.0.1', 0))
        rtp.settimeout(.02)
        stop = threading.Event()
        speaking = threading.Event()
        errors = []
        sent = []
        ws = Mock()
        def send(event):
            sent.append(event)
            stop.set()
        ws.send_json.side_effect = send
        thread = threading.Thread(target=stream_rtp_to_live,
                                  args=(rtp, ws, stop, speaking, errors))
        thread.start()
        thread.join(1)
        stop.set()
        thread.join(1)
        rtp.close()
        self.assertFalse(errors)
        self.assertFalse(speaking.is_set())
        self.assertEqual(sent[0]['type'], 'session.input_audio.append')
        self.assertEqual(base64.b64decode(sent[0]['audio']), b'\xff' * 800)

    def test_rtp_playback_preserves_codec_and_waits_for_queued_speech(self):
        sink = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sink.bind(('127.0.0.1', 0))
        sink.settimeout(1)
        source = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sender = RtpAudioSender(source, sink.getsockname())
        speech = b'\x90' * 160
        sender.add(speech * 2)
        self.assertFalse(sender.speech_drained(0))
        sender.start()
        try:
            packets = [sink.recv(4096), sink.recv(4096)]
            self.assertEqual([rtp_payload(p) for p in packets], [speech, speech])
            first = struct.unpack('!BBHII', packets[0][:12])
            second = struct.unpack('!BBHII', packets[1][:12])
            self.assertEqual((second[2] - first[2]) % 65536, 1)
            self.assertEqual((second[3] - first[3]) % 2**32, 160)
            self.assertTrue(sender.speech_drained(0))
            self.assertFalse(sender.speech_drained(1))
        finally:
            sender.stop()
            source.close()
            sink.close()

    def test_output_gate_distinguishes_mulaw_silence(self):
        self.assertFalse(pcmu_has_speech(b'\xff' * 160))
        self.assertFalse(pcmu_has_speech(b'\x7f' * 160))
        self.assertTrue(pcmu_has_speech(b'\x90' * 160))
        self.assertFalse(pcmu_has_speech(b''))


if __name__ == '__main__':
    unittest.main()
