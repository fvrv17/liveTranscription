# Subtitle Protocol

One rule from which everything else follows:
**A segment is a mutable entity with a version, not a string in an append-only log.**

ASR partial hypotheses, late speaker tags, manual editor corrections, and translation arrivals—
all of these are revisions of the same `sid`. The client stores a `Map<sid, segment>` and performs an upsert.

## Segment Message

```jsonc
{
  “t”: “seg”,
  “sid”: 128,          // segment ID, monotonically increasing within `talk_id`
  “rev”: 3,            // version; the client ignores `rev` <= a version that has already been applied
  “talk”: “t_042”,
  “final”: false,      // false = ASR may still overwrite the text
  “stable”: true,      // text has been committed to LocalAgreement; can be translated
  “lang”: “ru”,        // language of THIS text
  “spk”: “s_maria”,    // null until diacritization has responded
  “spk_conf”: 0.81,
  “text”: “We measured latency at three sites.”,
  “conf”: 0.91,        // average ASR confidence per segment
  “t0”: 412.30,        // seconds from the start of the presentation
  “t1”: 415.10,
  “emitted_at”: 1721471234.512  // Unix timestamp at the edge, for measuring end-to-end latency
}
```

### Why `final` and `stable` Are Different Flags

| stable | final | what it is | show | translate |
|---|---|---|---|---|
| false | false | “tail” of the hypothesis, words are still fluctuating | yes, muted | **no** |
| true | false | prefix committed, sentence not closed | yes, in full contrast | no |
| true | true | segment closed (punctuation/pause/limit) | yes | **yes** |

Translation starts exactly at the transition to `stable && final`. This is the answer to “translation is jerky and expensive.”

## Other messages

```jsonc
{“t”:“tr”,“sid”:128,“rev”:3,“lang”:“en”,‘text’:“We measured latency at three venues.”}
{“t”:“spk”,“sid”:128,“rev”:4,“spk”:‘s_maria’,“spk_conf”:0.81}  // late diarization
{“t”:“talk”,“talk”:“t_042”,“state”:“live”,‘title’:“...”," speakers“:[...],‘src_lang’:”ru"}
{“t”:‘state’,“muted”:true}          // operator kill switch
{“t”:“health”,“asr_rtf”:0.42,“lag_ms”:1840,“uplink”:‘ok’,“dropped”:0}
{“t”:“summary”,“talk”:“t_042”,‘md’:“...”}
{“t”:“hello”,“talk”:“t_042”,“lang”:‘en’,“from_sid”:110}  // client → server upon (re)connection
```

## Delivery Guarantees

Edge → cloud: Each message has a `seq` (monotonic). The cloud acknowledges `{“t”:‘ack’,“seq”:N}`.
The edge stores unacknowledged messages in a circular buffer (default: 5,000 messages ≈ 40 minutes of text)
and, upon reconnection, resends starting from `last_ack+1`. The cloud deduplicates based on `(sid, rev)`.

Cloud → client: The client sends `from_sid` in the `hello` message—the server returns a snapshot of the last
N segments, then switches to live mode. A 10-second Wi-Fi outage in the room does not result in text loss.