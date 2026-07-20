"""
Streaming ASR: faster-whisper + LocalAgreement-N.

LocalAgreement commits a word only when it has matched in the last N
consecutive hypotheses. This transforms Whisper’s “flickering” output into a stable
stream and costs us exactly +1 chunk of latency.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("asr")

SAMPLE_RATE = 16000


@dataclass
class Word:
    text: str
    t0: float
    t1: float
    prob: float = 1.0


def _norm(s: str) -> str:
    """Нормализация для сравнения гипотез: пунктуация и регистр не должны
    мешать признать слово стабильным."""
    return re.sub(r"[^\w]", "", s.lower())



# LocalAgreement

class LocalAgreement:
    """
    Гипотезы приходят над одним и тем же незакоммиченным буфером аудио,
    поэтому всегда начинаются с одной точки — общий префикс определён корректно.
    """

    def __init__(self, n: int = 2):
        self.n = max(1, n)
        self.history: list[list[Word]] = []

    def insert(self, hypothesis: list[Word]) -> list[Word]:
        """Возвращает слова, которые только что стали стабильными."""
        self.history.append(list(hypothesis))
        if len(self.history) > self.n:
            self.history.pop(0)
        if len(self.history) < self.n:
            return []

        committed: list[Word] = []
        limit = min(len(h) for h in self.history)
        for i in range(limit):
            cands = [h[i] for h in self.history]
            if all(_norm(c.text) == _norm(cands[0].text) for c in cands):
                # go with the option from the last hypothesis: its punctuation is better, 
                # since the model has already seen the right-hand context
                committed.append(cands[-1])
            else:
                break

        if committed:
            k = len(committed)
            self.history = [h[k:] for h in self.history]
        return committed

    def reset(self) -> None:
        self.history.clear()



# protection against hallucinations


HALLUCINATION_PATTERNS = [
    r"субтитры\s+(сделал|создавал|подготовил)",
    r"продолжение\s+следует",
    r"спасибо\s+за\s+(просмотр|внимание)$",
    r"подписывайтесь\s+на\s+канал",
    r"редактор\s+субтитров",
    r"^\s*ПРОДОЛЖЕНИЕ\s+СЛЕДУЕТ\s*\.{0,3}\s*$",
    r"subtitles?\s+by",
    r"thanks?\s+for\s+watching",
    r"^\s*\[?\s*(музыка|music|аплодисменты|applause)\s*\]?\s*$",
    r"^\s*(you|thank you)\.?\s*$",
]
_HALLU = [re.compile(p, re.IGNORECASE) for p in HALLUCINATION_PATTERNS]


def looks_hallucinated(text: str, no_speech_prob: float, cfg) -> bool:
    if no_speech_prob >= cfg.no_speech_threshold:
        return True
    t = text.strip()
    if not t:
        return True
    if any(p.search(t) for p in _HALLU):
        log.warning("отброшена галлюцинация: %r", t)
        return True
    toks = [_norm(w) for w in t.split() if _norm(w)]
    if len(toks) >= 6 and len(set(toks)) <= 2:
        log.warning("отброшен повтор-луп: %r", t)
        return True
    return False


class WhisperBackend:
    def __init__(self, cfg):
        from faster_whisper import WhisperModel 

        self.cfg = cfg
        log.info("загружаю faster-whisper %s (%s/%s)", cfg.model, cfg.device, cfg.compute_type)
        self.model = WhisperModel(cfg.model, device=cfg.device, compute_type=cfg.compute_type)
        # hotwords 
        self.hotwords = " ".join(cfg.vocabulary) if cfg.vocabulary else None

    def transcribe(self, audio: np.ndarray, prompt: str, offset: float) -> list[Word]:
        segments, _info = self.model.transcribe(
            audio,
            language=self.cfg.language,          
            task="transcribe",
            word_timestamps=True,
            condition_on_previous_text=False,    
            initial_prompt=prompt or None,
            hotwords=self.hotwords,
            beam_size=self.cfg.beam_size,
            temperature=0.0,
            vad_filter=False,                    
        )
        out: list[Word] = []
        for seg in segments:
            if looks_hallucinated(seg.text, seg.no_speech_prob, self.cfg):
                continue
            for w in seg.words or []:
                out.append(
                    Word(text=w.word.strip(), t0=offset + w.start, t1=offset + w.end,
                         prob=getattr(w, "probability", 1.0))
                )
        return out



class StreamingASR:
    """
    feed()  — подкладываем PCM (float32, 16 кГц, mono)
    poll()  — раз в chunk_sec декодируем весь незакоммиченный буфер и
              возвращаем (новые стабильные слова, нестабильный хвост)
    """

    def __init__(self, backend, cfg):
        self.backend = backend
        self.cfg = cfg
        self.agree = LocalAgreement(cfg.agreement_n)
        self.buf = np.zeros(0, dtype=np.float32)
        self.buf_origin = 0.0         
        self.since_poll = 0            
        self.committed_tail: list[str] = []   
        self.last_partial: list[Word] = []

    def feed(self, pcm: np.ndarray) -> None:
        self.buf = np.concatenate([self.buf, pcm])
        self.since_poll += len(pcm)

    def ready(self) -> bool:
        return self.since_poll >= self.cfg.chunk_sec * SAMPLE_RATE

    def poll(self) -> tuple[list[Word], list[Word]]:
        if not self.ready() or len(self.buf) < SAMPLE_RATE * 0.4:
            return [], self.last_partial
        self.since_poll = 0

        prompt = " ".join(self.committed_tail[-40:])
        hyp = self.backend.transcribe(self.buf, prompt, self.buf_origin)
        new_stable = self.agree.insert(hyp)

        if new_stable:
            self.committed_tail.extend(w.text for w in new_stable)
            self.committed_tail = self.committed_tail[-60:]
            
            cut = new_stable[-1].t1 - self.buf_origin
            n = int(max(0.0, cut) * SAMPLE_RATE)
            if n > 0:
                self.buf = self.buf[n:]
                self.buf_origin += n / SAMPLE_RATE

        
        if len(self.buf) > self.cfg.max_buffer_sec * SAMPLE_RATE:
            log.warning("буфер переполнен, принудительный сброс")
            keep = int(self.cfg.max_buffer_sec * 0.5 * SAMPLE_RATE)
            self.buf_origin += (len(self.buf) - keep) / SAMPLE_RATE
            self.buf = self.buf[-keep:]
            self.agree.reset()

        stable_norm = {_norm(w.text) for w in new_stable}
        self.last_partial = [w for w in hyp if _norm(w.text) not in stable_norm][-12:]
        return new_stable, self.last_partial

    def flush(self) -> list[Word]:
        """Конец доклада: коммитим всё, что осталось в гипотезе."""
        tail = self.last_partial
        self.last_partial = []
        self.agree.reset()
        return tail
