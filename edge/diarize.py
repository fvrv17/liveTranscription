"""
Диаризация — «кто говорит».

Два режима, и выбор между ними — организационный, а не технический:

  mode=channel   Реализован в channels.py: спикер известен синхронно,
                 асинхронная диаризация не нужна вовсе.

  mode=embed     Есть только суммарный микс. ECAPA-эмбеддинг на закоммиченном
                 куске речи + онлайн-кластеризация по косинусу. Overlap не
                 разбирает в принципе; на коротких репликах («да», «согласен»)
                 ошибается.

Ключевые приёмы, которые делают mode=embed пригодным:
  1. Энроллмент на саундчеке: 10 с речи каждого панелиста -> именованные
     центроиды. На выходе «Мария Петрова», а не «Speaker 2».
  2. Метка спикера НЕ блокирует субтитр. Сегмент уходит в зал с spk=null,
     а метка прилетает отдельной ревизией через ~200 мс. Ровно тот же
     механизм ревизий, что и для partial/final.
"""
from __future__ import annotations

import logging
import os
import wave
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("diarize")
SAMPLE_RATE = 16000
MIN_EMBED_SEC = 0.8      # короче — эмбеддинг бессмысленный


@dataclass
class SpeakerGuess:
    spk: str | None
    conf: float


class EmbeddingRouter:
    """mode=embed. Онлайн-кластеризация с опциональным энроллментом."""

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

    # -- модель --------------------------------------------------------------
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
        except Exception as exc:                       # прототип не должен падать из-за диаризации
            log.warning("эмбеддинг не посчитан: %s", exc)
            return None

    # -- энроллмент ----------------------------------------------------------
    def _enroll(self, path: str) -> None:
        for fn in sorted(os.listdir(path)):
            if not fn.lower().endswith(".wav"):
                continue
            name = os.path.splitext(fn)[0]           # имя файла = имя спикера
            audio = _read_wav(os.path.join(path, fn))
            emb = self.embed(audio)
            if emb is not None:
                self.centroids[name] = emb
                self.counts[name] = 8                # энроллмент весит больше живых наблюдений
                self.enrolled.add(name)
                log.info("энроллмент: %s (%.1f c)", name, len(audio) / SAMPLE_RATE)

    # -- назначение ----------------------------------------------------------
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
            if best not in self.enrolled:            # энроллмент-центроиды не размываем
                n = self.counts[best]
                upd = (self.centroids[best] * n + emb) / (n + 1)
                self.centroids[best] = upd / (np.linalg.norm(upd) or 1.0)
                self.counts[best] = n + 1
            return SpeakerGuess(best, best_sim)

        if len(self.centroids) >= self.max_speakers:
            return SpeakerGuess(best, best_sim)      # лучше спорная метка, чем бесконечные новые
        self._anon += 1
        spk = f"S{self._anon}"
        self.centroids[spk] = emb
        self.counts[spk] = 1
        log.info("новый спикер: %s (лучшее сходство было %.2f)", spk, best_sim)
        return SpeakerGuess(spk, 1.0 - max(best_sim, 0.0))


class NullRouter:
    def assign(self, audio: np.ndarray) -> SpeakerGuess:
        return SpeakerGuess(None, 0.0)


def build_router(cfg):
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
    Кольцевой буфер последних N секунд аудио.
    Диаризация работает по УЖЕ закоммиченному сегменту, вырезая его
    временной интервал из ленты — поэтому она и не добавляет задержки
    в основной путь субтитра.
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
