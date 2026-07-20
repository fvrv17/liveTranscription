"""
Diarization — who is speaking.

There are two modes, and the choice between them is organizational, not technical:

  mode=channel   Separate channels come from the console (Dante / multi-channel
                 USB interface, via a loop-through to the panelist). Channel == speaker.
                 Zero latency, zero errors, works correctly with
                 interruptions and overlapping speech. THIS IS THE CORRECT SOLUTION
                 for a panel discussion—insist on it from the organizers.

  mode=embed     Only a master mix is available. ECAPA embedding on a committed
                 speech segment + online cosine clustering. It cannot
                 distinguish overlaps at all; it makes mistakes
                 with short responses (“yes,” “I agree”).

Key techniques that make mode=embed suitable:
  1. Enrollment during soundcheck: 10 seconds of speech from each panelist -> named
     centroids. The output is “Maria Petrova,” not “Speaker 2.”
  2. The speaker tag does NOT block the subtitle. The segment is sent to the audience with spk=null,
     and the tag arrives as a separate revision after ~200 ms. It’s exactly the same
     revision mechanism as for partial/final.
"""
from __future__ import annotations

import logging
import os
import wave
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("diarize")
SAMPLE_RATE = 16000
MIN_EMBED_SEC = 0.8      

@dataclass
class SpeakerGuess:
    spk: str | None
    conf: float


class ChannelRouter:
    """mode=channel. Channel Index Mapping -> Speaker ID."""

    def __init__(self, mapping: dict[int, str]):
        self.mapping = {int(k): v for k, v in mapping.items()}

    def assign_channel(self, ch: int) -> SpeakerGuess:
        return SpeakerGuess(self.mapping.get(ch, f"ch{ch}"), 1.0)

    def assign(self, audio: np.ndarray) -> SpeakerGuess:      
        return SpeakerGuess(None, 0.0)


class EmbeddingRouter:
    """mode=embed. Online clustering with optional enrollment."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.threshold = cfg.diarize_threshold
        self.max_speakers = cfg.diarize_max_speakers
        self.centroids: dict[str, np.ndarray] = {}
        self.counts: dict[str, int] = {}
        self.enrolled: set[str] = set()
        self._encoder = None
        self._anon = 0
        if cfg.diarize_enroll_dir and os.path.isdir(cfg.diarize_enroll_dir):
            self._enroll(cfg.diarize_enroll_dir)

    @property
    def encoder(self):
        if self._encoder is None:
            from speechbrain.inference.speaker import EncoderClassifier
            log.info("загружаю ECAPA-TDNN")
            self._encoder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="/tmp/ecapa",
                run_opts={"device": self.cfg.device},
            )
        return self._encoder

    def embed(self, audio: np.ndarray) -> np.ndarray | None:
        if len(audio) < MIN_EMBED_SEC * SAMPLE_RATE:
            return None
        try:
            import torch
            with torch.no_grad():
                t = torch.from_numpy(audio).float().unsqueeze(0)
                e = self.encoder.encode_batch(t).squeeze().cpu().numpy()
            n = np.linalg.norm(e)
            return e / n if n else None
        except Exception as exc:                      
            log.warning("эмбеддинг не посчитан: %s", exc)
            return None

    def _enroll(self, path: str) -> None:
        for fn in sorted(os.listdir(path)):
            if not fn.lower().endswith(".wav"):
                continue
            name = os.path.splitext(fn)[0]           
            audio = _read_wav(os.path.join(path, fn))
            emb = self.embed(audio)
            if emb is not None:
                self.centroids[name] = emb
                self.counts[name] = 8                
                self.enrolled.add(name)
                log.info("enrollment: %s (%.1f c)", name, len(audio) / SAMPLE_RATE)

    def assign(self, audio: np.ndarray) -> SpeakerGuess:
        emb = self.embed(audio)
        if emb is None:
            return SpeakerGuess(None, 0.0)

        best, best_sim = None, -1.0
        for spk, c in self.centroids.items():
            sim = float(np.dot(emb, c))
            if sim > best_sim:
                best, best_sim = spk, sim

        if best is not None and best_sim >= self.threshold:
            if best not in self.enrolled:            
                n = self.counts[best]
                upd = (self.centroids[best] * n + emb) / (n + 1)
                self.centroids[best] = upd / (np.linalg.norm(upd) or 1.0)
                self.counts[best] = n + 1
            return SpeakerGuess(best, best_sim)

        if len(self.centroids) >= self.max_speakers:
            return SpeakerGuess(best, best_sim)      
        self._anon += 1
        spk = f"S{self._anon}"
        self.centroids[spk] = emb
        self.counts[spk] = 1
        log.info("new speaker: %s (best similarity was %.2f)", spk, best_sim)
        return SpeakerGuess(spk, 1.0 - max(best_sim, 0.0))


class NullRouter:
    def assign(self, audio: np.ndarray) -> SpeakerGuess:
        return SpeakerGuess(None, 0.0)


def build_router(cfg):
    if cfg.diarize_mode == "channel":
        return ChannelRouter(cfg.diarize_channels)
    if cfg.diarize_mode == "embed":
        return EmbeddingRouter(cfg)
    return NullRouter()


def _read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        data = w.readframes(w.getnframes())
        a = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            a = a.reshape(-1, w.getnchannels()).mean(axis=1)
    return a


class AudioTape:
    """
    A circular buffer containing the last N seconds of audio.
    The diary function operates on a segment that has ALREADY been marked, cutting
    that time interval from the stream—which is why it does not introduce any delay
    into the main subtitle path.
    """

    def __init__(self, seconds: float = 120.0):
        self.cap = int(seconds * SAMPLE_RATE)
        self.buf = np.zeros(0, dtype=np.float32)
        self.origin = 0.0

    def write(self, pcm: np.ndarray, t_start: float) -> None:
        if len(self.buf) == 0:
            self.origin = t_start
        self.buf = np.concatenate([self.buf, pcm])
        if len(self.buf) > self.cap:
            drop = len(self.buf) - self.cap
            self.buf = self.buf[drop:]
            self.origin += drop / SAMPLE_RATE

    def slice(self, t0: float, t1: float) -> np.ndarray:
        i0 = int(max(0.0, t0 - self.origin) * SAMPLE_RATE)
        i1 = int(max(0.0, t1 - self.origin) * SAMPLE_RATE)
        return self.buf[i0:min(i1, len(self.buf))]
