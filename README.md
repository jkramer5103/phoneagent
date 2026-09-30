# Telephone Agent

Edit these two files next to the script:

- **`instructions.txt`** — what the agent should do or say. Write ordinary text
  in your preferred language, including any names, details, or limits it needs.
- **`number.txt`** — the phone number to call, for example `+4915123456789`.
  Spaces, parentheses, and dashes are accepted.

Then run:

```bash
python3 telephone_agent.py
# Or: uv run telephone_agent.py
```

The included test task asks a bicycle workshop for a rear-tire replacement,
a slot tomorrow between 14:00 and 17:00 (or the next day), and a total price of
at most 60 euros, under Kramer. Replace it with your own task. Both the voice
agent and its backend receive your instructions. They also receive the current
Berlin date plus the dates and weekdays for tomorrow and the next day.

Short requests are enough, for example:

- `Reserviere morgen um 19 Uhr einen Tisch für zwei auf Kramer.`
- `Frag, ob mein Laptop auf Kramer abholbereit ist.`
- `Buche einen Haarschnitt morgen nachmittags bis 40 Euro auf Kramer.`

The shared conversation policy applies to every task: use the latest corrected
facts, respect inclusive limits, avoid invented preferences, ask about relevant
uncertainties, and distinguish collecting information from making a booking.
Include personal details or limits when the task needs them. The agent cannot
supply facts you have not provided or learned during the call.

## Credentials

Put these in `.env` in your working directory, or in environment variables:

```dotenv
OPENAI_API_KEY=your-openai-project-key
SPEEDPORT_SIP_PASSWORD=your-speedport-ip-phone-password
```

## Test without calling anyone

```bash
python3 telephone_agent.py --check-api
```

This tests OpenAI connectivity, German speech, G.711 audio, and clean session
finalization. It does not read the task/number files or dial a phone.

## Options

The task and number files default to the script's directory. You can use others:

```bash
python3 telephone_agent.py --instructions-file my-task.txt --number-file my-number.txt
```

`--destination` overrides the number file for one call. `--help` lists all options.
Defaults: Speedport `192.168.2.1:5060`, extension `**72`, SIP username
`nutzer-2@speedport.ip`, voice `marin`, voice model `gpt-live-1`, backend
`gpt-6-luna`. The script waits up to 45 seconds for an answer; calls last at most
180 seconds by default (`--max-call-seconds`).

Calls run on this machine by default. The agent detects a directly attached
router LAN and binds SIP and RTP to that interface, so an overlapping VPN route
does not redirect the call. `--interface enp3s0` selects an interface explicitly.

When away from home, you can explicitly opt in to an SSH relay:

```bash
python3 telephone_agent.py --remote-host codex
```

The relay transfers the scripts and loaded task files to a private temporary
folder, sends credentials through encrypted SSH stdin, and removes remote files
afterward. `--local` forces local execution even when `--remote-host` is supplied.

## How it works

The script registers with the Speedport as an IP telephone and dials through it.
After the answer, it starts a GPT-Live WebSocket session and bridges continuous
G.711 μ-law audio directly between the call and OpenAI, without resampling.
Timed transcript fragments for each speaker appear in the terminal. GPT-Live
handles listening, speech, and interruptions continuously.

A closing handoff records intent; it does not immediately disconnect the phone.
A separate lifecycle reviewer uses the backend model and Structured Outputs to
assess the full conversation after speech pauses. It distinguishes ongoing
conversation, a missing farewell, and a completed closing statement. If a
farewell is missing, it requests one brief statement from GPT-Live. It can also
recognize a spoken closing when GPT-Live omitted its native delegation.

Only a validated model decision ends the call: further model audio is blocked,
the queued final speech drains, and SIP BYE disconnects the telephone. An explicit
request for immediate disconnection can skip the farewell. New caller transcript
fragments invalidate older pending decisions; an already spoken final farewell
remains terminal. There are no transcript keyword or regex hangup rules, and a mention or quotation of a farewell is not a closing decision.
The reviewer makes additional Responses API requests while the conversation runs.

GPT-Live uses client delegation. The reviewer records `completed`, `unsuccessful`,
or `other`; there is no managed terminal function loop to leave waiting. Missing
farewells return through `session.commentary.append`, which supplies spoken content.
Inbound PCMU is sent as 800 samples per 100ms of wall time; receive timeouts
never add extra samples on top of late packets. Partial output RTP packets are
padded with silence so their tails play even before the final closing decision. The final drain uses
an audio-energy estimate because Live has no speech-completed event; this does
not decide whether the conversation has ended. Remote hangups and the maximum
duration also end the call. Finally, the script closes the Live session and
prints final usage. Success for bookings reflects verbal confirmation; no booking
system is connected.

`restaurant_call.py` remains a compatibility launcher for the same general agent.
`speedport_call.py` supplies its SIP primitives and can separately make a test
call that plays three beeps.

## Local verification

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile telephone_agent.py call_lifecycle.py restaurant_call.py speedport_call.py
```

Official docs: [GPT-Live WebSockets](https://developers.openai.com/api/docs/guides/voice-websockets?api=live),
[prompting](https://developers.openai.com/api/docs/guides/live-prompting), and
[delegation](https://developers.openai.com/api/docs/guides/live-delegation).

Earlier prototypes (`agent.py`, `agent_realtime.py`, and `main.py`) remain in
this repository. Use `telephone_agent.py` for the current GPT-Live implementation.
