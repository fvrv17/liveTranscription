"""
Две шкалы времени и перевод между ними.

В ASR попадает только речь: тишину мы отсекаем до модели, иначе Whisper
галлюцинирует. Значит, внутренние часы ASR — и таймкоды слов, которые он
возвращает, — идут ТОЛЬКО по речи. Лента аудио для диаризации, таймкоды
в протоколе и архивная стенограмма живут в абсолютном времени доклада.

Эти шкалы расходятся ровно на суммарную длительность пауз. На часовом
докладе это десятки минут. Расхождение молчаливое: ничего не падает,
просто диаризация режет чужой кусок аудио, а таймкоды в архиве врут.

TimeMap хранит участки непрерывной речи и переводит одну шкалу в другую.
"""
from __future__ import annotations

from bisect import bisect_right

SAMPLE_RATE = 16000


class TimeMap:
    def __init__(self, gap_tolerance: float = 0.005):
        # участки: (начало по часам речи, начало по абсолютным, длительность)
        self.runs: list[list[float]] = []
        self.starts: list[float] = []          # для двоичного поиска
        self.speech_clock = 0.0
        self.gap_tolerance = gap_tolerance
        self._last_abs_end: float | None = None

    def feed(self, n_samples: int, abs_t0: float) -> None:
        """Вызывать ровно тогда, когда кадр уходит в ASR."""
        dur = n_samples / SAMPLE_RATE
        contiguous = (self._last_abs_end is not None
                      and abs(abs_t0 - self._last_abs_end) <= self.gap_tolerance)
        if contiguous and self.runs:
            self.runs[-1][2] += dur
        else:
            self.runs.append([self.speech_clock, abs_t0, dur])
            self.starts.append(self.speech_clock)
        self.speech_clock += dur
        self._last_abs_end = abs_t0 + dur

    def to_abs(self, speech_t: float) -> float:
        """Часы речи -> абсолютное время доклада."""
        if not self.runs:
            return speech_t
        i = bisect_right(self.starts, speech_t) - 1
        if i < 0:
            return self.runs[0][1]
        s0, a0, dur = self.runs[i]
        # за концом участка (хвост гипотезы) — линейная экстраполяция
        return a0 + min(speech_t - s0, dur) + max(0.0, speech_t - s0 - dur)

    def span_abs(self, t0: float, t1: float) -> tuple[float, float]:
        return self.to_abs(t0), self.to_abs(t1)

    @property
    def silence_dropped(self) -> float:
        """Сколько тишины отсеяно — то самое расхождение шкал."""
        if not self.runs:
            return 0.0
        last = self.runs[-1]
        return (last[1] + last[2]) - self.speech_clock

    def compact(self, keep_last: int = 4000) -> None:
        """Участков накапливается по одному на паузу; на длинном докладе
        подрезаем историю, оставляя запас глубже любого открытого сегмента."""
        if len(self.runs) > keep_last * 2:
            self.runs = self.runs[-keep_last:]
            self.starts = [r[0] for r in self.runs]
