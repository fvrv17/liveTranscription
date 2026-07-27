"""
Проверки embed-диаризации.

Числа DER живут в eval/baseline.json и меняются вместе с настройками. Здесь —
только инварианты, которые ломать нельзя ни при какой настройке:

  1. подменный энкодер разделяет голоса так, как заявлено в eval/scenes.py;
  2. эмбеддинг не зависит от гейна микрофона;
  3. spk_conf — одна шкала, а не две склеенные;
  4. упор в max_speakers не приводит к уверенной чужой метке;
  5. кластеры одного человека склеиваются, и склейка чинит историю;
  6. двух опознанных панелистов не сливает никакая перекластеризация;
  7. офлайн-проход действительно чинит то, что жадный онлайн испортил;
  8. смена голоса внутри сегмента находится и режет уверенность.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "edge"), os.path.join(HERE, "..", "eval")]

import numpy as np

from config import Config
from diarize import OnlineClusterer, _confidence, agglomerative, build_router, voiced
from scenes import Voice, cast, render
from stub_encoder import StubEncoder

ok = True


def check(cond: bool, good: str, bad: str) -> None:
    global ok
    if cond:
        print("OK:", good)
    else:
        print("FAIL:", bad)
        ok = False


def cfg_for_eval() -> Config:
    c = Config()
    c.diarize_mode = "embed"
    c.diarize_refine_threshold = 0.07
    return c


enc = StubEncoder()
V = cast()


def emb_of(v: Voice, seed: int = 5, dur: float = 1.5) -> np.ndarray:
    return enc.encode(render(v, dur, np.random.default_rng(seed)))


# ── 1. разделимость подменного энкодера ─────────────────────────────────────
print("=== разделимость энкодера ===")
mean = {}
for name, v in V.items():
    rng = np.random.default_rng(11)
    es = [enc.encode(render(v, 1.5, rng)) for _ in range(5)]
    m = np.mean(es, axis=0)
    mean[name] = m / np.linalg.norm(m)
    intra = min(float(np.dot(es[i], es[j])) for i in range(5) for j in range(i + 1, 5))
    check(intra >= 0.99, f"{name}: один голос с собой {intra:.3f}",
          f"{name}: разброс внутри голоса слишком велик ({intra:.3f})")

hard = float(np.dot(mean["Мария"], mean["Марина"]))
others = [(a, b, float(np.dot(mean[a], mean[b])))
          for i, a in enumerate(V) for b in list(V)[i + 1:]
          if {a, b} != {"Мария", "Марина"}]
worst = max(others, key=lambda p: p[2])
check(worst[2] <= 0.55, f"разные голоса не ближе {worst[2]:.3f} ({worst[0]}/{worst[1]})",
      f"голоса {worst[0]} и {worst[1]} неразличимы: {worst[2]:.3f}")
check(hard >= 0.80, f"пара Мария/Марина осталась трудной: {hard:.3f}",
      f"пара Мария/Марина перестала быть трудной ({hard:.3f}) — сценарий similar обесценен")

# ── 2. независимость от гейна ───────────────────────────────────────────────
print("\n=== гейн микрофона ===")
loud = emb_of(V["Мария"], seed=9)
quiet = emb_of(Voice("Мария", V["Мария"].f0, V["Мария"].env_db, gain_db=-18.0), seed=9)
sim = float(np.dot(loud, quiet))
check(sim >= 0.99, f"тот же голос тише на 18 dB: {sim:.3f}",
      f"гейн меняет эмбеддинг ({sim:.3f}) — один человек распадётся на двух")

# ── 3. одна шкала уверенности ───────────────────────────────────────────────
print("\n=== шкала spk_conf ===")
thr, temp, ref = 0.68, 0.06, 0.08
sure = _confidence(0.95, 0.30, thr, temp, ref)
edge = _confidence(0.69, 0.66, thr, temp, ref)
miss = _confidence(0.20, 0.10, thr, temp, ref)
check(sure > edge > miss, f"монотонна: уверенно {sure:.2f} > спорно {edge:.2f} > мимо {miss:.2f}",
      f"немонотонна: {sure:.2f} / {edge:.2f} / {miss:.2f}")
check(0.0 <= miss and sure <= 1.0, "остаётся в [0,1]", "вышла за [0,1]")
check(edge < 0.5, f"похожий сосед сбивает уверенность до {edge:.2f}",
      f"при отрыве 0.03 уверенность всё ещё {edge:.2f}")

# ── 4. упор в max_speakers ──────────────────────────────────────────────────
print("\n=== мест больше нет ===")
c = cfg_for_eval()
c.diarize_max_speakers = 2
cl = OnlineClusterer(c)
cl.observe(mean["Мария"])
cl.observe(mean["Иван"])
spk, conf = cl.observe(mean["Ольга"])         # третий лишний
check(conf < c.diarize_min_conf,
      f"метка отдана с уверенностью {conf:.2f} < порога {c.diarize_min_conf} — наружу уйдёт null",
      f"чужая метка {spk} с уверенностью {conf:.2f} прошла бы на экран")

# ── 5. склейка кластеров ────────────────────────────────────────────────────
print("\n=== склейка дублей ===")
c = cfg_for_eval()
cl = OnlineClusterer(c)
a, _ = cl.observe(emb_of(V["Мария"], seed=1))
cl.centroids["S_dup"] = emb_of(V["Мария"], seed=2)   # тот же голос, отдельный кластер
cl.counts["S_dup"] = 1
before = len(cl.centroids)
cl.merge_pass()
check(len(cl.centroids) == before - 1, f"дубль схлопнут: {before} -> {len(cl.centroids)}",
      f"дубль одного голоса не склеен ({before} -> {len(cl.centroids)})")
check(bool(cl.take_merges()), "алиас записан — прошлые сегменты можно переименовать",
      "склейка не оставила алиаса, история осталась с мёртвой меткой")

# ── 6. запрет на слияние опознанных ─────────────────────────────────────────
print("\n=== два панелиста не сливаются ===")
X = np.vstack([mean["Мария"], mean["Марина"], mean["Мария"], mean["Марина"]])
w = np.ones(4)
free = agglomerative(X, w, threshold=0.5, block=None)
held = agglomerative(X, w, threshold=0.5, block=np.array([0, 1, 0, 1]))
check(len(set(free.tolist())) == 1, "без запрета порог 0.5 действительно сливает их в один",
      "проверка вырождена: порог 0.5 и так их не сливает")
check(len(set(held.tolist())) == 2, "с запретом остались двумя кластерами",
      "запрет не сработал: Мария и Марина слиты")

# ── 7. офлайн чинит жадность онлайна ────────────────────────────────────────
print("\n=== офлайн-проход против жадности ===")
from run_eval import make_cfg, run_scenario                       # noqa: E402
from scenarios import build_all                                   # noqa: E402

scn = next(s for s in build_all() if s.scene.name == "greedy_merge")
r = run_scenario(scn, make_cfg())
on, re_ = r["online"].der, r["refined"].der
check(on > 0.20, f"жадный онлайн ожидаемо провалился: DER {on:.1%}",
      f"онлайн справился сам (DER {on:.1%}) — сценарий больше не проверяет офлайн-проход")
check(re_ < on - 0.15, f"офлайн-проход починил: {on:.1%} -> {re_:.1%}",
      f"офлайн-проход не помог: {on:.1%} -> {re_:.1%}")

# ── 8. смена голоса внутри сегмента ─────────────────────────────────────────
print("\n=== смена голоса внутри сегмента ===")
rng = np.random.default_rng(21)
half = render(V["Мария"], 2.5, rng)
mixed = np.concatenate([half, render(V["Иван"], 2.5, rng)])
router = build_router(cfg_for_eval(), encoder=StubEncoder())
router.enroll_audio({"Мария": render(V["Мария"], 8.0, np.random.default_rng(1234)),
                     "Иван": render(V["Иван"], 8.0, np.random.default_rng(1234))})
g_mixed = router.assign(1, mixed, 0.0)
g_clean = router.assign(2, render(V["Мария"], 5.0, np.random.default_rng(31)), 10.0)
check(g_mixed.change_at is not None and abs(g_mixed.change_at - 2.5) <= 0.4,
      f"момент смены найден: {g_mixed.change_at} c (истина 2.5)",
      f"смена голоса не найдена или далеко: {g_mixed.change_at}")
check(g_mixed.conf < g_clean.conf,
      f"уверенность срезана: смешанный {g_mixed.conf:.2f} < чистого {g_clean.conf:.2f}",
      f"смешанный сегмент так же уверен, как чистый ({g_mixed.conf:.2f})")

# ── 9. отбор речи ───────────────────────────────────────────────────────────
print("\n=== выброс пауз перед эмбеддингом ===")
speech = render(V["Мария"], 1.0, np.random.default_rng(41))
padded = np.concatenate([np.zeros(16000, dtype=np.float32), speech,
                         np.zeros(16000, dtype=np.float32)])
kept = voiced(padded, range_db=35.0, floor_db=-55.0)
check(len(kept) < len(padded) * 0.75, f"тишина выброшена: {len(padded)} -> {len(kept)} отсчётов",
      f"тишина осталась в окне: {len(padded)} -> {len(kept)}")
sim = float(np.dot(enc.encode(kept), enc.encode(speech)))
check(sim >= 0.95, f"вектор по отобранной речи совпал с чистым: {sim:.3f}",
      f"отбор речи испортил вектор: {sim:.3f}")

print()
sys.exit(0 if ok else 1)
