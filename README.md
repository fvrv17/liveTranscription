# Live Conference Subtitles 

Real-time speech recognition of the speaker, translation into any language,
and a summary at the end of the presentation. Limit: 5 seconds of end-to-end latency.

This covers only the server-side components: the venue agent, the cloud service, and the protocol between
them. Clients (the screen in the hall, the viewer’s web interface, the operator’s console) are consumers of the
WebSocket API and are not included in this repository.

## System Architecture

The boundary lies **between audio and text**, not between the front end and the back end.

```
STAGE (sound engineer’s GPU box)      CLOUD
┌────────────────────────────────┐        ┌───────────────────────── ─────┐
│ control panel ─► ingest ─► VAD ─► ASR  │ text  │ translation ─► pub/sub ─► WS     │──► viewers
│              │       │         │ ~2 kbit │ storage ─► summaries         │
│              │       └► segmenter ───►│ operator API                │
│              └► diarization     │        └──────────────────────────────┘
│                                │
│   ws://:8788 (hall screen) ◄────┤  local socket, independent of the uplink
└────────────────────────────────┘
```

Three consequences:

1. **Audio does not leave the room.** Only text is transmitted via the venue’s unreliable uplink:
   ~2 kbps instead of 48. Text can be buffered and resent, but the audio stream cannot.
2. **The screen in the room experiences an internet outage.** It relies on the agent’s local socket.
   Verified: the cloud went down for 8 seconds in the middle of a presentation—the local circuit
   didn’t notice it; 76 accumulated messages were delivered after recovery,
   and duplicates were filtered out based on `(sid, rev)`.
3. **Audience members are not connected to the venue’s Wi-Fi.** Client traffic goes to the cloud, so
   an overloaded access point doesn’t bring down the entire system.

If there’s only one venue and the internet connection is reliable—everything is consolidated into a single server, and that’s
a perfectly fine solution. A hybrid setup is needed when there are multiple venues or the connection is hit-or-miss.

## What’s non-trivial here

**Segment revisions instead of an append-only log.** The same `sid` is overwritten by
four independent sources: ASR stabilization, late diarization, arrival of
translations, and manual operator edits. All four rely on a single mechanism—but only
because it was in the protocol from the very beginning. The format is in `PROTOCOL.md`.

**`final` and `stable` are different flags.** There is a state where “the prefix is committed,
but the sentence isn’t closed yet”: it’s already possible to return the text to the client as confirmed,
but it’s not yet possible to translate it. Translation starts exactly when the status transitions to `stable && final`.
Without this distinction, either the contrast between hypothesis and fact is lost,
or the MT stutters and rewrites the phrase starting with every new word.

**LocalAgreement-N.** A word is committed when it has matched in the last N hypotheses.
This costs exactly +1 chunk of delay and is the main control knob for “stability versus
speed” (`agreement_n`).

**Diaryization outside the critical path.** A segment leaves with `spk: null`; the label
arrives in the next revision after ~200 ms. `channel` mode (separate channels
from the console) is more accurate and faster than any ML—if the organizers provide a multi-channel
input, you should use it: it handles interruptions and overlap correctly, which online
embedding clustering cannot do in principle. Enrollment during soundcheck
(10 seconds per panelist) provides names instead of “Speaker 2.”

**Lazy translation.** We only translate languages that have an active subscriber.
Out of about fifteen languages in the interface, usually 2–3 are active—MT usage drops
by a factor of 4–6. When a new language is activated, we catch up on the last 30 segments.

**Anti-hallucination measures.** During silence, Whisper confidently outputs “Subtitles created…”
or “To be continued…”—on the screen behind the speaker, this is an incident.
Three filters: a VAD gate before ASR, the `no_speech_prob` threshold, a phrase blocklist, plus
a detector for degenerate decoder repetitions.

**Conference vocabulary.** Speaker names, product names, abbreviations—
exactly the words people came to hear, and these are the ones that break down without
`vocabulary` (hotwords for ASR) and the glossary for MT.

## Latency Budget

| Component | Original | Translation |
|---|---|---|
| capture + chunk buffer | 500–900 ms | same |
| ASR inference (large-v3, GPU) | 250–600 ms | same |
| **LocalAgreement stabilization (+1 chunk)** | 500–900 ms | same |
| waiting for sentence boundary | — | 0–3500 ms |
| text uplink | — | 30–80 ms |
| MT | — | 100–600 ms |
| pub/sub + WS | 60–150 ms | same |
| **total** | **1.3–2.2 s** | **2.2–5.0 s** |

A subtle point that determines the configuration: `max_segment_sec` **does not** add delay
to the original subtitles—they grow in segments as recognition progresses. It affects
only the translation, and only the **first words** of a phrase: the last word
is translated almost immediately after it is spoken. Therefore, the “5-second” worst-case scenario refers
to the start of a long sentence in the translation, and this is precisely what conflicts
with the requirement.

If we don’t meet the deadline: `agreement_n: 1` reduces the delay by ~800 ms at the cost of text stability;
switching from Whisper to a transducer model (Parakeet, Deepgram) saves another ~500 ms
on segment size. Reducing `max_segment_sec` is a last resort; it directly
worsens MT quality: short clauses are translated noticeably worse than full sentences.

## Hardware and Money

**On-premises, one hall:** An RTX 4070 / L4 is sufficient for one large-v3 stream;
an RTX 4090 can handle 2–3 halls. 8-core CPU, 32 GB RAM.
**Cloud:** 4 vCPUs / 8 GB can handle about 10,000 WS connections.

Cost for an 8-hour day, one hall: API translation is the most expensive line item
and the only one that scales linearly with the number of languages (about a hundred dollars a day
for 10 languages with constant speech input). Lazy translation cuts this by a factor of three to four.
A local MT model eliminates this cost entirely at the expense of one additional GPU.

## API

**Ingest from the platform:** `ws://…/ws/ingest` (Bearer token). At-least-once
with acknowledgment via `seq`; the agent holds unacknowledged messages in a circular buffer
(5,000 messages ≈ 40 minutes of speech) and resends them upon reconnection. The server deduplicates
by `(sid, rev)`.

**Delivery to clients:** `ws://…/ws/view?talk=…&lang=…`. Upon connection—a snapshot
of the last 40 segments, then live streaming. Change language with a `setlang` message
without reconnecting.

**Local venue loop:** `ws://<edge>:8788`—same message format,
works without an internet connection.

| REST | |
|---|---|
| `GET /api/talks` | list of talks |
| `GET /api/talks/{id}/transcript?lang=` | transcript, optionally translated |
| `GET/POST /api/talks/{id}/summary` | get / regenerate summary |
| `POST /api/talks/{id}/control` | `mute` (mute) · `unmute` · `end` |
| `POST /api/talks/{id}/segments/{sid}` | edit text → new revision sent to clients |
| `GET /api/talks/{id}/stats` | delay, uplink status, listeners by language, MT usage |

The mute and edit features arose not from the initial requirements but from operational needs:
the platform needed a way to instantly mute subtitles when something was said
that shouldn’t be recorded.

## Deployment

```bash
# cloud
pip install -r requirements-cloud.txt
cd cloud && uvicorn app:app --port 8000

# venue agent (requires a GPU)
pip install -r requirements-edge.txt   # uncomment production dependencies
cd edge && python run_edge.py --config config.venue.yaml
```

MT keys and summaries are passed via environment variables (`DEEPL_KEY` or `OPENAI_API_KEY`).
Without ingest keys, storage and distribution work, but translation is disabled,
and summaries are generated by extracting fragments.

## Limitations

- **Overlapping speech** is not handled by online diarization. This can only be resolved using separate
  channels controlled via remote—this is an organizational requirement for the venue, not an ML task.
- **Automatic language detection** is risky during code-switching (“Russian with English
  terms”); in the production profile, the language is specified in advance.
- **Summaries are marked as drafts.** The LLM hallucinates numbers; the prompt includes an explicit
  prohibition, but an operator must review them before publication. Currently, they are published
  automatically—a deliberate simplification.
- **Partial multi-tenancy.** `talk_id` is used everywhere, but the session registry
  is single-level and pub/sub is in-process. For multiple parallel rooms, an
  external broker is required; the `bus.py` interface intentionally matches NATS/Redis Streams,
  making replacement straightforward.
- **Not implemented:** offline transcript annotation after a talk for archiving,
  audience authorization, recording retention policy, and speaker consent.

## Structure

```
edge/       venue agent
  asr.py          faster-whisper + LocalAgreement + hallucination filters
  segmenter.py    stream segmentation, closure policy
  diarize.py      channel / embed, enrollment, audio ring buffer
  audio.py        capture from the console, VAD gate
  uplink.py       store-and-forward over an unreliable channel
  run_edge.py     main loop + local socket for the hall screen
cloud/      cloud component
  app.py          ingest, fan-out, REST, commands to the venue
  bus.py          pub/sub + active language registry
  store.py        SQLite, deduplication by (sid, rev)
  translate.py    lazy MT, cache, glossary
  summarize.py    map-reduce summaries
PROTOCOL.md   message format and delivery guarantees
```
