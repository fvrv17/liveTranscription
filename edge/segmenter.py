"""
Сегментатор: превращает поток слов от ASR в сегменты, пригодные для перевода.

Отвечает на один вопрос — КОГДА закрывать сегмент.

Компромисс, который здесь зашит: MT хочет длинные законченные предложения,
бюджет задержки хочет короткие. Закрываем по первому из событий:
пунктуация / пауза / лимит слов / лимит длительности. Последние два —
страховка от спикера, который говорит десять минут без единой точки.

Важно для настройки: лимит длительности НЕ добавляет задержки оригинальным
субтитрам (они растут партиалами по мере распознавания). Он влияет только
на перевод, и только на первые слова фразы — последнее слово переводится
почти сразу после произнесения.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from asr import Word

_WS = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.!?;:…])")
TERMINAL = re.compile(r"[.!?…]+[»\"')\]]*$")


def join_words(words: list[Word]) -> str:
    """Склейка word-токенов ASR в строку. Речь не редактируется —
    чинятся только пробелы, которые расставляет токенизатор."""
    t = " ".join(w.text for w in words)
    t = _WS.sub(" ", t).strip()
    return _SPACE_BEFORE_PUNCT.sub(r"\1", t)


@dataclass
class Segment:
    sid: int
    words: list[Word] = field(default_factory=list)
    rev: int = 0
    final: bool = False
    spk: str | None = None
    spk_conf: float = 0.0
    opened_at: float = field(default_factory=time.monotonic)

    @property
    def text(self) -> str:
        return join_words(self.words)

    @property
    def t0(self) -> float:
        return self.words[0].t0 if self.words else 0.0

    @property
    def t1(self) -> float:
        return self.words[-1].t1 if self.words else 0.0

    @property
    def conf(self) -> float:
        """Средняя уверенность ASR по сегменту: видно, где модель плыла."""
        return sum(w.prob for w in self.words) / len(self.words) if self.words else 0.0


@dataclass
class SegEvent:
    seg: Segment
    stable: bool
    partial_tail: str = ""


class Segmenter:
    def __init__(self, cfg):
        self.cfg = cfg
        self._sid = 0
        self.open: Segment | None = None
        self.last_word_end: float | None = None

    def _new(self) -> Segment:
        self._sid += 1
        return Segment(sid=self._sid)

    # -- вход ----------------------------------------------------------------
    def push_stable(self, words: list[Word]) -> list[SegEvent]:
        """Закоммиченные слова: только они могут закрывать сегмент."""
        events: list[SegEvent] = []
        for w in words:
            pause = (w.t0 - self.last_word_end) if self.last_word_end is not None else 0.0
            self.last_word_end = w.t1

            if self.open is None:
                self.open = self._new()
            elif pause >= self.cfg.pause_split_sec and self.open.words:
                events.append(self._close())
                self.open = self._new()

            self.open.words.append(w)

            if self._should_close(self.open):
                events.append(self._close())
                self.open = None

        if self.open and self.open.words:
            self.open.rev += 1
            events.append(SegEvent(seg=self.open, stable=True))
        return events

    def push_partial(self, tail: list[Word]) -> SegEvent | None:
        """Нестабильный хвост гипотезы: отдаём, но не переводим."""
        if not tail:
            return None
        if self.open is None:
            self.open = self._new()
        self.open.rev += 1
        return SegEvent(seg=self.open, stable=False, partial_tail=join_words(tail))

    def tick(self) -> SegEvent | None:
        """Вызывать по таймеру: закрывает сегмент, если спикер замолчал."""
        if not self.open or not self.open.words:
            return None
        if time.monotonic() - self.open.opened_at >= self.cfg.max_segment_sec:
            ev = self._close()
            self.open = None
            return ev
        return None

    def flush(self) -> SegEvent | None:
        if self.open and self.open.words:
            ev = self._close()
            self.open = None
            return ev
        return None

    # -- политика ------------------------------------------------------------
    def _should_close(self, seg: Segment) -> bool:
        last = seg.words[-1].text
        n = len(seg.words)
        if TERMINAL.search(last) and n >= self.cfg.min_words:
            return True
        if n >= self.cfg.max_words:
            return True
        if seg.t1 - seg.t0 >= self.cfg.max_segment_sec:
            return True
        # по запятой режем только длинный сегмент: короткая клауза
        # переводится заметно хуже целого предложения
        if last.endswith(",") and n >= self.cfg.clause_split_words:
            return True
        return False

    def _close(self) -> SegEvent:
        seg = self.open
        seg.final = True
        seg.rev += 1
        return SegEvent(seg=seg, stable=True)
