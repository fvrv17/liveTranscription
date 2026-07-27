"""
Захват звука и VAD-гейт.

VAD стоит ДО ASR, а не внутри него: тишину нельзя отдавать Whisper,
иначе он галлюцинирует титры («Субтитры сделал ...») прямо на экран в зале.
Silero, если установлен; иначе — энергетический гейт с гистерезисом,
которого для прототипа достаточно.
"""
from __future__ import annotations

import logging
import queue
import wave

import numpy as np

log = logging.getLogger("audio")
SAMPLE_RATE = 16000
FRAME = 512          # 32 мс


class VAD:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.speech = False
        self.hang = 0
        if cfg.vad == "silero":
            try:
                import torch
                self.model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
                log.info("VAD: silero")
            except Exception as exc:
                log.warning("silero недоступен (%s), падаю на энергетический VAD", exc)

    def is_speech(self, frame: np.ndarray) -> bool:
        if self.model is not None:
            import torch
            with torch.no_grad():
                p = float(self.model(torch.from_numpy(frame), SAMPLE_RATE).item())
            active = p >= self.cfg.vad_threshold
        else:
            rms = float(np.sqrt(np.mean(frame ** 2)) + 1e-9)
            db = 20 * np.log10(rms)
            active = db > self.cfg.vad_energy_db

        # гистерезис: не рвём фразу на коротких паузах между словами
        if active:
            self.speech = True
            self.hang = int(self.cfg.vad_hangover_sec * SAMPLE_RATE / FRAME)
        elif self.hang > 0:
            self.hang -= 1
        else:
            self.speech = False
        return self.speech


class MicSource:
    """Захват с устройства. На площадке — линейный вход с микшерного пульта,
    а не микрофон ноутбука."""

    def __init__(self, device=None, channels: int = 1):
        import sounddevice as sd
        self.q: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self.channels = channels
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, blocksize=FRAME, device=device,
            channels=channels, dtype="float32", callback=self._cb,
        )

    def _cb(self, indata, frames, time_info, status):
        if status:
            log.warning("audio status: %s", status)
        try:
            self.q.put_nowait(indata.copy())
        except queue.Full:
            log.warning("очередь захвата переполнена, кадр отброшен")

    def __enter__(self):
        self.stream.start()
        return self

    def __exit__(self, *a):
        self.stream.stop()
        self.stream.close()

    def read(self) -> np.ndarray:
        """Возвращает (кадры, каналы). Схлопывать в моно здесь нельзя:
        именно из разницы между каналами и определяется говорящий."""
        block = self.q.get()
        return block if block.ndim > 1 else block.reshape(-1, 1)


class FileSource:
    """Проигрывание WAV в реальном времени — для тестов и демо."""

    def __init__(self, path: str):
        self.wav = wave.open(path, "rb")
        self.ch = self.wav.getnchannels()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.wav.close()

    def read(self) -> np.ndarray:
        data = self.wav.readframes(FRAME)
        if not data:
            raise EOFError
        a = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        return a.reshape(-1, self.ch)
