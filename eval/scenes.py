"""
Синтетические сцены: голоса, планы реплик и нарезка на сегменты.

Голос здесь — это гармоники основного тона под заданной спектральной
огибающей. Звучит как робот и не годится для ASR, но для диаризации важно
ровно одно свойство: у одного человека огибающая устойчива, у разных —
различна. На этом же стоит и настоящий ECAPA, поэтому логика кластеризации
проверяется честно, а расстояния между голосами задаются нами и не зависят
ни от весов модели, ни от железа.

Замеренные на этом наборе косинусы (см. tests/test_diarize.py, там это
зафиксировано проверкой): один голос с собой — 0.995+, разные голоса —
не выше 0.50, и отдельно пара Мария/Марина — 0.88, то есть заведомо выше
порога назначения. Последняя пара нужна нарочно: система обязана либо
развести её, либо честно признать неуверенность.

Что здесь СОЗНАТЕЛЬНО сделано неудобно для системы:

  * Сегменты режутся из АУДИО по паузам, а не из плана реплик. Значит, когда
    двое говорят встык, граница сегмента не совпадает с границей говорящего —
    и появляется сегмент на два голоса. Нарезка по плану замела бы этот
    случай под ковёр, а он самый частый на панельной дискуссии.
  * Часть сцен начинается с шумной реплики. Онлайн-кластеризация обязана на
    ней ошибиться — это вход для проверки офлайн-прохода.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SAMPLE_RATE = 16000
FRAME = 512


# Контрольные точки спектральной огибающей. Через них задаётся тембр.
#
# Первая версия описывала голос набором формант, и это не сработало: полосы
# анализатора на верхних частотах шире самих резонансов, форманта целиком
# укладывается в одну полосу и перестаёт различаться. Косинус в итоге
# определялся только общим наклоном спектра, и любые два «светлых» голоса
# слипались. Огибающая, заданная напрямую, свободна от этой связи: расстояние
# между голосами становится параметром сцены, а не побочным эффектом того,
# как устроен банк фильтров.
CTRL_HZ = np.geomspace(120.0, 7000.0, 10)


@dataclass
class Voice:
    name: str
    f0: float                    # основной тон, Гц
    env_db: tuple                # усиление в дБ в точках CTRL_HZ — это и есть тембр
    gain_db: float = 0.0         # уровень канала на пульте


def render(v: Voice, dur: float, rng: np.random.Generator,
           noise_db: float = -45.0) -> np.ndarray:
    """Гармоники, продавленные формантами, под слоговой огибающей."""
    n = int(dur * SAMPLE_RATE)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.arange(n) / SAMPLE_RATE

    # лёгкое вибрато: без него все окна одного голоса идентичны до бита,
    # и разделимость получается нереалистично идеальной
    f0 = v.f0 * (1.0 + 0.02 * np.sin(2 * np.pi * 4.7 * t))
    phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE

    ctrl = np.log(CTRL_HZ)
    env_db = np.asarray(v.env_db, dtype=float)
    sig = np.zeros(n, dtype=np.float64)
    for k in range(1, int(7800 / v.f0) + 1):
        fk = k * v.f0
        amp = 10 ** (np.interp(np.log(fk), ctrl, env_db) / 20.0)
        sig += amp * np.sin(k * phase + k * 0.7)

    # слоговая огибающая ~4 Гц с микропаузами между словами
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)
    env *= (rng.random(n) > 0.05)
    sig *= env
    sig += rng.normal(0, 10 ** (noise_db / 20), n) * env      # шипящие

    peak = np.abs(sig).max()
    if peak > 0:
        sig /= peak
    return (sig * 0.5 * 10 ** (v.gain_db / 20)).astype(np.float32)


def silence(dur: float, rng: np.random.Generator, floor_db: float = -58.0) -> np.ndarray:
    n = int(dur * SAMPLE_RATE)
    return rng.normal(0, 10 ** (floor_db / 20), n).astype(np.float32)


@dataclass
class Scene:
    name: str
    audio: np.ndarray
    turns: list[tuple[str, float, float]]      # (спикер, t0, t1) — эталон
    voices: dict[str, Voice] = field(default_factory=dict)
    note: str = ""

    @property
    def duration(self) -> float:
        return len(self.audio) / SAMPLE_RATE


def build(name: str, plan: list[tuple], voices: dict[str, Voice],
          seed: int = 0, note: str = "", noise_db: float = -45.0) -> Scene:
    """plan: [(ключ голоса или None для паузы, длительность[, шум в дБ]), ...]

    Ключ в plan и ИМЯ спикера — разные вещи: «Мария на петличке» и «Мария на
    трибунном» это два разных ключа с разным гейном и один и тот же человек
    в эталоне. Иначе сцену с перепадом уровней не собрать.
    """
    rng = np.random.default_rng(seed)
    parts, turns, t = [], [], 0.0
    for item in plan:
        who, dur = item[0], item[1]
        noise = item[2] if len(item) > 2 else noise_db
        if who is None:
            parts.append(silence(dur, rng))
        else:
            parts.append(render(voices[who], dur, rng, noise))
            turns.append((voices[who].name, t, t + dur))
        t += dur
    audio = np.concatenate(parts).astype(np.float32)
    # общий шум зала поверх всего: вентиляция, кашель в зале
    audio += silence(len(audio) / SAMPLE_RATE, rng, -60.0)[:len(audio)]
    return Scene(name=name, audio=audio, turns=turns, voices=voices, note=note)


# ── нарезка на сегменты ─────────────────────────────────────────────────────
def segment_audio(audio: np.ndarray, pause_split_sec: float = 0.6,
                  max_segment_sec: float = 3.5,
                  range_db: float = 40.0) -> list[tuple[int, float, float]]:
    """Грубая копия политики segmenter.py, но по звуку, а не по словам.

    Настоящий сегментатор режет по пунктуации и паузам в потоке слов от ASR.
    ASR здесь нет, а важна одна его черта: граница сегмента ставится по ПАУЗЕ,
    и про смену говорящего сегментатор в embed-режиме не знает ничего. Значит,
    два спикера подряд без паузы обязаны попасть в один сегмент — и попадают.
    """
    n = (len(audio) // FRAME) * FRAME
    frames = audio[:n].reshape(-1, FRAME)
    db = 20.0 * np.log10(np.sqrt(np.mean(frames * frames, axis=1)) + 1e-12)
    speech = db >= db.max() - range_db
    fdur = FRAME / SAMPLE_RATE
    gap_need = max(1, int(pause_split_sec / fdur))

    runs: list[list[float]] = []
    i = 0
    while i < len(speech):
        if not speech[i]:
            i += 1
            continue
        j = i
        quiet = 0
        while j < len(speech):
            if speech[j]:
                quiet = 0
            else:
                quiet += 1
                if quiet >= gap_need:
                    break
            j += 1
        end = j - quiet + 1
        runs.append([i * fdur, min(end, len(speech)) * fdur])
        i = j + 1

    segs: list[tuple[int, float, float]] = []
    sid = 0
    for t0, t1 in runs:
        while t1 - t0 > max_segment_sec:
            sid += 1
            segs.append((sid, t0, t0 + max_segment_sec))
            t0 += max_segment_sec
        if t1 - t0 > 0.15:
            sid += 1
            segs.append((sid, t0, t1))
    return segs


# ── набор голосов ───────────────────────────────────────────────────────────
def cast() -> dict[str, Voice]:
    """Четверо различимых + один похожий на Марию (см. scenario similar)."""
    return {
        "Мария":  Voice("Мария",  196.0, (-6, -2, 2, 6, 8, 4, -2, -8, -14, -20)),
        "Иван":   Voice("Иван",   104.0, (4, 8, 6, 2, -4, -10, -16, -22, -26, -30)),
        "Ольга":  Voice("Ольга",  232.0, (-18, -14, -8, -2, 2, 6, 8, 6, 0, -6)),
        "Пётр":   Voice("Пётр",   126.0, (0, 4, 2, -6, 2, 6, 0, -6, -12, -18)),
        # Тембр в пределах естественного разброса от Марии. Этот случай
        # диаризация обязана либо решить, либо честно признать неуверенность;
        # молча выдать одну из двух фамилий — худший из исходов.
        "Марина": Voice("Марина", 190.0, (-5, -3, 3, 5, 7.5, 5, -1, -9, -13, -21)),
    }
