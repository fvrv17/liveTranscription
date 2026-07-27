"""
Кто сейчас говорит при многоканальном входе с пульта.

Наивное решение — «самый громкий канал побеждает» — не работает, и это
главное, что нужно понимать про этот модуль. Причины по убыванию вредности:

  1. ПРОСОЧКА. Спикер за трибуной слышен во ВСЕХ микрофонах на сцене,
     просто тише. Абсолютных «активных» каналов всегда несколько.
  2. РАЗНЫЙ ГЕЙН. Петличка, ручной, трибунный, микрофон в зале — разная
     чувствительность, разное расстояние до рта, разный уровень на пульте.
     Единый абсолютный порог бессмыслен, нужен свой шумовой пол на канал.
  3. ДРОЖАНИЕ. Между словами уровень проваливается. Без гистерезиса
     владение словом скачет по каналам на каждом смыкании губ.

Отсюда алгоритм: адаптивный шумовой пол на канал → SNR вместо уровня →
относительный отрыв лидера от второго места вместо порога → выдержка на
переключение и на отпускание → разделение просочки и настоящего наложения
по корреляции огибающих.

Отдельно: это ещё и VAD. Никто не держит слово — значит, речи нет, и в ASR
кормить нечего. В канальном режиме отдельный VAD не нужен.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("floor")

SAMPLE_RATE = 16000
EPS = 1e-10


def _db(x: np.ndarray) -> float:
    return 20.0 * math.log10(float(np.sqrt(np.mean(x * x))) + EPS)


@dataclass
class Channel:
    idx: int
    name: str
    noise_db: float = -90.0
    level_db: float = -90.0
    snr: float = 0.0
    env: deque = field(default_factory=lambda: deque(maxlen=24))   # ~0.8 c огибающей
    quiet_frames: int = 0
    ready: bool = False
    dead_reported: bool = False


@dataclass
class FloorEvent:
    """Смена владельца слова. Пустой owner = слово отпущено (тишина)."""
    t: float
    owner: str | None
    channel: int | None
    overlap: list[str] = field(default_factory=list)
    reason: str = ""


class FloorDetector:
    def __init__(self, cfg, names: dict[int, str], n_channels: int):
        self.cfg = cfg
        names = {int(k): v for k, v in (names or {}).items()}
        self.chs = [Channel(i, names.get(i, f"ch{i}")) for i in range(n_channels)]
        self.owner: int | None = None
        self.candidate: int | None = None
        self.attack = 0
        self.hold = 0
        self.frames = 0
        self.overlap: list[str] = []

        fr = cfg.floor_frame_sec
        self.attack_need = max(1, int(cfg.floor_attack_sec / fr))
        self.hold_need = max(1, int(cfg.floor_hold_sec / fr))
        self.warmup = max(1, int(cfg.floor_warmup_sec / fr))
        self.dead_need = max(1, int(cfg.floor_dead_sec / fr))

    # -- шумовой пол -------------------------------------------------------
    def _update_noise(self, ch: Channel) -> None:
        """Асимметричное слежение: к тишине падаем быстро, вверх ползём еле-еле.
        Иначе долгая реплика утащит пол за собой и канал «оглохнет»."""
        if not ch.ready:
            ch.noise_db = ch.level_db
            ch.ready = True
            return
        d = ch.level_db - ch.noise_db
        rate = self.cfg.floor_noise_fall if d < 0 else self.cfg.floor_noise_rise
        ch.noise_db += d * rate

    # -- просочка или настоящее наложение ----------------------------------
    def _is_bleed(self, leader: Channel, other: Channel) -> bool:
        """Просочка — это ослабленная копия речи лидера: огибающие идут
        синхронно. Настоящее наложение двух голосов синхронности не даёт.
        Задержка между микрофонами на сцене меньше кадра, так что хватает
        корреляции при нулевом лаге."""
        n = min(len(leader.env), len(other.env))
        if n < self.cfg.floor_corr_min_frames:
            return True                       # мало данных — считаем просочкой
        a = np.fromiter(leader.env, float, len(leader.env))[-n:]
        b = np.fromiter(other.env, float, len(other.env))[-n:]
        sa, sb = a.std(), b.std()
        if sa < 1e-6 or sb < 1e-6:
            return True
        corr = float(np.corrcoef(a, b)[0, 1])
        return corr >= self.cfg.floor_bleed_corr

    # -- основной вход -----------------------------------------------------
    def push(self, block: np.ndarray, t_abs: float) -> FloorEvent | None:
        """block: (frames, n_channels) float32. Возвращает событие только
        в момент смены владельца."""
        self.frames += 1
        for ch in self.chs:
            ch.level_db = _db(block[:, ch.idx])
            self._update_noise(ch)
            ch.snr = ch.level_db - ch.noise_db
            ch.env.append(ch.level_db)
            if ch.snr < self.cfg.floor_open_snr:
                ch.quiet_frames += 1
            else:
                ch.quiet_frames = 0
            self._check_dead(ch)

        if self.frames < self.warmup:
            return None                        # калибровка шумового пола

        active = [c for c in self.chs if c.snr >= self.cfg.floor_open_snr]
        if not active:
            return self._maybe_release(t_abs)

        active.sort(key=lambda c: c.snr, reverse=True)
        leader = active[0]
        rivals = [c for c in active[1:] if not self._is_bleed(leader, c)]
        self.overlap = [c.name for c in rivals
                        if leader.snr - c.snr < self.cfg.floor_margin_db]

        self.hold = self.hold_need
        if self.owner == leader.idx:
            self.candidate = None
            self.attack = 0
            return None

        # смена владельца — только после выдержки, иначе кашель уводит слово
        if self.candidate == leader.idx:
            self.attack += 1
        else:
            self.candidate = leader.idx
            self.attack = 1
        if self.attack < self.attack_need:
            return None

        prev = self.owner
        self.owner = leader.idx
        self.candidate = None
        self.attack = 0
        second = f", второй {active[1].name} {active[1].snr:.0f} dB" if len(active) > 1 else ""
        log.info("слово переходит: %s -> %s (SNR %.0f dB%s)",
                 self.chs[prev].name if prev is not None else "—", leader.name, leader.snr, second)
        return FloorEvent(t=t_abs, owner=leader.name, channel=leader.idx,
                          overlap=list(self.overlap),
                          reason=f"snr={leader.snr:.1f}{second}")

    def _maybe_release(self, t_abs: float) -> FloorEvent | None:
        if self.owner is None:
            return None
        self.hold -= 1
        if self.hold > 0:
            return None                        # пауза между словами, не отпускаем
        prev = self.chs[self.owner].name
        self.owner = None
        self.candidate = None
        self.attack = 0
        self.overlap = []
        log.debug("слово отпущено (%s замолчал)", prev)
        return FloorEvent(t=t_abs, owner=None, channel=None, reason="тишина")

    def _check_dead(self, ch: Channel) -> None:
        """Выключенная петличка или севшая батарейка: канал есть, речи в нём
        не бывает. Оператор должен узнать об этом до начала доклада."""
        if ch.dead_reported or ch.quiet_frames < self.dead_need:
            return
        ch.dead_reported = True
        log.warning("канал %s (%d) молчит %d c — микрофон выключен или отвалился?",
                    ch.name, ch.idx, self.cfg.floor_dead_sec)

    # -- аудио для ASR -----------------------------------------------------
    def asr_input(self, block: np.ndarray) -> np.ndarray | None:
        """dominant — только канал владельца: чистая речь без просочки соседей.
        mix — сумма каналов: хуже качество, но переживает наложение."""
        if self.cfg.asr_source == "mix":
            return block.mean(axis=1)
        if self.frames < self.warmup:
            # Идёт калибровка шумового пола: владелец ещё не определён, но
            # ронять начало доклада нельзя. Кормим микс — атрибуция появится
            # с первым же событием, текст не теряется.
            return block.mean(axis=1)
        if self.owner is None:
            return None                        # никто не говорит — в Whisper не идёт
        return block[:, self.owner]

    def status(self) -> dict:
        return {
            "owner": self.chs[self.owner].name if self.owner is not None else None,
            "overlap": list(self.overlap),
            "channels": [{"name": c.name, "snr": round(c.snr, 1),
                          "noise_db": round(c.noise_db, 1),
                          "dead": c.dead_reported} for c in self.chs],
        }
