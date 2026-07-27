"""
Проверка детектора владения словом на синтетической сцене.

Сцена воспроизводит ровно те условия, на которых ломается наивное
«побеждает самый громкий»:
  ch0 «Мария»    — трибунный, громкий
  ch1 «Иван»     — петличка, тише на 12 dB (другой гейн)
  ch2 «Модератор» — выключенный микрофон, только шум
Речь каждого просачивается во все остальные каналы на -18 dB.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "edge"))

import numpy as np
from channels import FloorDetector
from config import Config

SR, FRAME = 16000, 512
rng = np.random.default_rng(7)


def speech(dur, gain_db=0.0):
    """Слоговая огибающая поверх шума — для логики по огибающим этого хватает."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)          # ~4 слога/сек
    env *= (rng.random(n) > 0.06)                           # микропаузы между словами
    return (rng.normal(0, 0.25, n) * env * 10 ** (gain_db / 20)).astype(np.float32)


def silence(dur, floor_db=-60.0):
    n = int(dur * SR)
    return (rng.normal(0, 10 ** (floor_db / 20), n)).astype(np.float32)


# ── сборка сцены ────────────────────────────────────────────────────────────
BLEED = 10 ** (-18 / 20)
plan = [("Мария", 0, 4.0), (None, None, 1.5), ("Иван", 1, 3.0),
        (None, None, 0.4), ("Мария", 0, 2.0)]

parts = []
for who, ch, dur in plan:
    src = silence(dur) if ch is None else speech(dur, gain_db=0 if ch == 0 else -12)
    block = np.stack([
        src if ch == 0 else src * BLEED,
        src if ch == 1 else src * BLEED,
        np.zeros_like(src),
    ], axis=1)
    block += np.stack([silence(dur, -62), silence(dur, -58), silence(dur, -55)], axis=1)
    parts.append(block)
scene = np.concatenate(parts).astype(np.float32)

# ── прогон ──────────────────────────────────────────────────────────────────
cfg = Config()
cfg.floor_dead_sec = 5.0
det = FloorDetector(cfg, {0: "Мария", 1: "Иван", 2: "Модератор"}, 3)

events = []
for i in range(0, len(scene) - FRAME, FRAME):
    t = i / SR
    ev = det.push(scene[i:i + FRAME], t)
    if ev:
        events.append((round(t, 2), ev.owner, ev.overlap))

print("события смены владельца слова:")
for t, owner, ov in events:
    print(f"  {t:6.2f} c  ->  {owner or 'тишина'}" + (f"   наложение: {ov}" if ov else ""))

# ── проверки ────────────────────────────────────────────────────────────────
owners = [o for _, o, _ in events]
speakers = [o for o in owners if o]
ok = True

if speakers != ["Мария", "Иван", "Мария"]:
    print("FAIL: последовательность владельцев", speakers); ok = False
else:
    print("OK: порядок владельцев верный, несмотря на разницу гейна 12 dB")

# после захвата слова и до конца реплики переключений быть не должно
claim = events[0][0]
inside = [t for t, o, _ in events if claim < t < 3.8]
if inside:
    print("FAIL: дребезг внутри реплики на", inside); ok = False
else:
    print("OK: микропаузы между словами не отнимают слово")

# пауза 1.5 c должна отпустить слово, пауза 0.4 c — нет
released = [t for t, o, _ in events if o is None]
if not any(4.0 <= t <= 5.2 for t in released):
    print("FAIL: длинная пауза не отпустила слово", released); ok = False
elif any(8.4 <= t <= 8.9 for t in released):
    print("FAIL: короткая пауза 0.4 c ошибочно отпустила слово"); ok = False
else:
    print("OK: длинная пауза отпускает слово, короткая — нет")

if any(ov for _, _, ov in events):
    print("FAIL: просочка -18 dB принята за наложение"); ok = False
else:
    print("OK: просочка не спутана с наложением речи")

if not det.chs[2].dead_reported:
    print("FAIL: выключенный микрофон не обнаружен"); ok = False
else:
    print("OK: выключенный микрофон обнаружен")

print("\nшумовой пол по каналам:", [round(c.noise_db, 1) for c in det.chs])
sys.exit(0 if ok else 1)
