"""
Метрики диаризации.

Основная — DER (diarization error rate), стандарт NIST: доля эталонной речи,
размеченной неправильно. Складывается из трёх слагаемых, и разделять их
обязательно, потому что чинятся они разным:

  miss       речь есть, метки нет      -> отбор речи, порог min_conf, окна
  false_alarm метки нет речи           -> нарезка сегментов, VAD
  confusion  метка есть, но чужая      -> кластеризация

Одно число DER без разбивки скрывает главное: система, которая молчит,
и система, которая уверенно врёт, могут дать один и тот же DER. Для зала
это совершенно разные продукты, поэтому wrong_name_rate считается отдельно.

Имена кластеров произвольны (S1, S2, R3), поэтому перед подсчётом confusion
метки гипотезы сопоставляются с эталонными — оптимально и строго один к
одному. Жадное сопоставление здесь занижало бы ошибку.

Коллар: NIST не штрафует ±0.25 с вокруг границ реплик. Точность границы
упирается в разметку, а не в алгоритм, и без коллара метрика меряет шум.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STEP = 0.01                    # шаг сетки, с


@dataclass
class DER:
    der: float
    miss: float
    false_alarm: float
    confusion: float
    wrong_name_rate: float     # доля речи с УВЕРЕННОЙ чужой меткой
    ref_speakers: int
    hyp_speakers: int
    mapping: dict[str, str]
    scored_sec: float

    def line(self) -> str:
        return (f"DER {self.der:6.1%}  (miss {self.miss:5.1%}  FA {self.false_alarm:5.1%}  "
                f"conf {self.confusion:5.1%})  чужое имя {self.wrong_name_rate:5.1%}  "
                f"спикеров {self.hyp_speakers}/{self.ref_speakers}")


def _grid(spans: list[tuple[str, float, float]], n: int) -> np.ndarray:
    """Метка на каждый шаг сетки; -1 = никого. Индексы меток — в порядке появления."""
    lab = np.full(n, -1, dtype=np.int32)
    names: dict[str, int] = {}
    for spk, t0, t1 in spans:
        if spk is None:
            continue
        k = names.setdefault(spk, len(names))
        lab[int(t0 / STEP):int(t1 / STEP)] = k
    return lab, names


def _optimal_map(ref: np.ndarray, hyp: np.ndarray, n_ref: int, n_hyp: int) -> dict[int, int]:
    """Сопоставление меток гипотезы эталонным, максимизирующее совпадение.

    Точный перебор по подмножествам, а не жадность: жадный выбор проигрывает
    ровно там, где интереснее всего — когда два кластера борются за одного
    эталонного спикера, и правильный ответ виден только по сумме.
    """
    if n_ref == 0 or n_hyp == 0:
        return {}
    ov = np.zeros((n_hyp, n_ref), dtype=np.int64)
    both = (ref >= 0) & (hyp >= 0)
    np.add.at(ov, (hyp[both], ref[both]), 1)

    full = 1 << n_ref
    dp = np.zeros(full, dtype=np.int64)
    choice = np.full((n_hyp, full), -1, dtype=np.int32)
    for h in range(n_hyp):
        nxt = dp.copy()                      # -1: этот кластер вообще не сопоставлен
        for mask in range(full):
            base = dp[mask]
            for r in range(n_ref):
                bit = 1 << r
                if mask & bit:
                    continue
                cand = base + ov[h, r]
                if cand > nxt[mask | bit]:
                    nxt[mask | bit] = cand
                    choice[h, mask | bit] = r
        dp = nxt

    mask = int(np.argmax(dp))
    out: dict[int, int] = {}
    for h in range(n_hyp - 1, -1, -1):
        r = int(choice[h, mask])
        if r >= 0:
            out[h] = r
            mask &= ~(1 << r)
    return out


def score(reference: list[tuple[str, float, float]],
          hypothesis: list[tuple[str, float, float]],
          duration: float,
          collar: float = 0.25,
          confident: set[str] | None = None) -> DER:
    """reference/hypothesis: [(спикер, t0, t1)]. spk=None в гипотезе = «не знаю»."""
    n = int(duration / STEP) + 1
    ref, ref_names = _grid(reference, n)
    hyp, hyp_names = _grid([h for h in hypothesis if h[0] is not None], n)

    # коллар вокруг каждой границы эталона
    scored = np.ones(n, dtype=bool)
    c = int(collar / STEP)
    if c > 0:
        for _, t0, t1 in reference:
            for t in (t0, t1):
                i = int(t / STEP)
                scored[max(0, i - c):min(n, i + c + 1)] = False

    ref_speech = (ref >= 0) & scored
    total = int(ref_speech.sum())
    if total == 0:
        return DER(0, 0, 0, 0, 0, len(ref_names), len(hyp_names), {}, 0.0)

    mapping = _optimal_map(ref[scored], hyp[scored], len(ref_names), len(hyp_names))
    inv_ref = {v: k for k, v in ref_names.items()}
    inv_hyp = {v: k for k, v in hyp_names.items()}

    mapped = np.full(n, -1, dtype=np.int32)
    for h, r in mapping.items():
        mapped[hyp == h] = r

    miss = int((ref_speech & (hyp < 0)).sum())
    fa = int((scored & (ref < 0) & (hyp >= 0)).sum())
    conf = int((ref_speech & (hyp >= 0) & (mapped != ref)).sum())

    # «Уверенно чужое имя» — только там, где метка пришла из энроллмента,
    # то есть на экране стояла настоящая фамилия, а не безобидное S3.
    if confident is None:
        wrong = conf
    else:
        conf_mask = np.zeros(n, dtype=bool)
        for h, name in inv_hyp.items():
            if name in confident:
                conf_mask |= (hyp == h)
        wrong = int((ref_speech & conf_mask & (mapped != ref)).sum())

    return DER(
        der=(miss + fa + conf) / total,
        miss=miss / total,
        false_alarm=fa / total,
        confusion=conf / total,
        wrong_name_rate=wrong / total,
        ref_speakers=len(ref_names),
        hyp_speakers=len(hyp_names),
        mapping={inv_hyp[h]: inv_ref[r] for h, r in mapping.items()},
        scored_sec=total * STEP,
    )
