"""
Два теста на вторую починку и на обратный случай для детектора.

1. Шкалы времени. ASR видит только речь, лента и протокол — абсолютное время.
   До починки координаты сегмента приходили в часах речи, а лента резалась
   ими как абсолютными: расхождение равно суммарной длительности пауз.
2. Настоящее наложение. Проверяем, что дискриминатор просочки не вырожден
   и отличает два независимых голоса от ослабленной копии одного.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "edge"))

import numpy as np
from channels import FloorDetector
from config import Config
from timebase import TimeMap

SR, FRAME = 16000, 512
rng = np.random.default_rng(11)
ok = True

# ── 1. шкалы времени ────────────────────────────────────────────────────────
print("=== шкалы времени ===")
tm = TimeMap()
abs_t = 0.0
plan = [("речь", 2.0), ("пауза", 3.0), ("речь", 2.0), ("пауза", 5.0), ("речь", 1.0)]
for kind, dur in plan:
    for _ in range(int(dur * SR) // FRAME):
        if kind == "речь":
            tm.feed(FRAME, abs_t)
        abs_t += FRAME / SR

print(f"часы речи в конце : {tm.speech_clock:.2f} c")
print(f"абсолютное время  : {abs_t:.2f} c")
print(f"отсеяно тишины    : {tm.silence_dropped:.2f} c  (это и было расхождение)")

# последнее слово произнесено в самом конце: по часам речи ~5 c, абсолютно ~13 c
last_speech_t = tm.speech_clock - 0.1
got = tm.to_abs(last_speech_t)
want = abs_t - 0.1
if abs(got - want) > 0.1:
    print(f"FAIL: to_abs({last_speech_t:.2f}) = {got:.2f}, ожидалось ~{want:.2f}"); ok = False
else:
    print(f"OK: to_abs({last_speech_t:.2f} по речи) = {got:.2f} c абсолютного времени")

naive_error = want - last_speech_t
print(f"OK: без починки диаризация резала бы ленту с ошибкой {naive_error:.1f} c")

if abs(tm.to_abs(1.0) - 1.0) > 0.05:
    print("FAIL: до первой паузы шкалы обязаны совпадать"); ok = False
else:
    print("OK: до первой паузы шкалы совпадают")

# ── 2. настоящее наложение против просочки ─────────────────────────────────
print("\n=== наложение против просочки ===")

def speech(dur, seed, gain_db=0.0):
    r = np.random.default_rng(seed)
    n = int(dur * SR)
    t = np.arange(n) / SR
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t + seed)   # своя слоговая структура
    env *= (r.random(n) > 0.06)
    return (r.normal(0, 0.25, n) * env * 10 ** (gain_db / 20)).astype(np.float32)

def run(block, label):
    cfg = Config()
    det = FloorDetector(cfg, {0: "Мария", 1: "Иван"}, 2)
    seen = []
    for i in range(0, len(block) - FRAME, FRAME):
        ev = det.push(block[i:i + FRAME], i / SR)
        if ev:
            seen.append(ev)
    overlapped = any(e.overlap for e in seen) or bool(det.overlap)
    print(f"  {label}: наложение {'обнаружено' if overlapped else 'не обнаружено'}")
    return overlapped

a = speech(4.0, seed=1)
b = speech(4.0, seed=99, gain_db=-3)          # второй голос, независимая огибающая
noise = lambda d, db: rng.normal(0, 10 ** (db / 20), int(d * SR)).astype(np.float32)

bleed_only = np.stack([a + noise(4.0, -62), a * 10 ** (-18 / 20) + noise(4.0, -60)], axis=1)
two_voices = np.stack([a + noise(4.0, -62), b + noise(4.0, -60)], axis=1)

if run(bleed_only, "просочка одного голоса -18 dB"):
    print("FAIL: просочка принята за наложение"); ok = False
if not run(two_voices, "два независимых голоса"):
    print("FAIL: настоящее наложение не обнаружено — дискриминатор вырожден"); ok = False
else:
    print("OK: дискриминатор различает оба случая")

sys.exit(0 if ok else 1)
