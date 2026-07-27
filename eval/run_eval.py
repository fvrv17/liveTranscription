#!/usr/bin/env python3
"""
Прогон диаризации по синтетическим сценам.

    python eval/run_eval.py                 # таблица + сверка с baseline.json
    python eval/run_eval.py --scenario panel --verbose
    python eval/run_eval.py --update-baseline

Что здесь считается результатом. Не «DER стал меньше» сам по себе, а две
величины рядом: онлайн (что зал увидел сразу) и после офлайн-прохода (что
осталось в архиве). Расхождение между ними — это ровно та польза, ради
которой офлайн-проход написан, и она должна быть видна числом.

Энкодер подменён (eval/stub_encoder.py), поэтому прогон детерминирован,
идёт секунды и не требует ни torch, ни GPU, ни скачивания моделей. Он меряет
логику: нарезку, отбор речи, кластеризацию, склейку и перекластеризацию —
всё, что ошибается независимо от того, кто считает вектор. Качество самой
ECAPA на живом голосе он не меряет и не претендует.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "edge"),
                os.path.dirname(os.path.abspath(__file__))]

import numpy as np

from config import Config
from diarize import build_router
from metrics import DER, score
from scenarios import Scenario, build_all
from scenes import SAMPLE_RATE, render
from stub_encoder import StubEncoder

BASELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline.json")
TOLERANCE = 0.02          # на столько DER может ухудшиться без объявления регрессии


def make_cfg() -> Config:
    """Пороги здесь СВОИ, и это не подгонка результата.

    Порог — свойство пары «энкодер + шкала его косинусов», а не алгоритма.
    У ECAPA один голос даёт ~0.7-0.8, разные ~0.2-0.4, и дефолты в config.py
    расставлены под это. У подменного энкодера шкала жёстче: один голос 0.995+,
    разные — 0.5 и ниже. Оставить боевые числа значило бы мерить не логику,
    а несовпадение двух шкал.

    Проверяется здесь то, что от шкалы не зависит: порядок решений, склейка,
    запреты, перекластеризация. Калибровку боевого порога синтетика не
    заменяет — её делают на записи реального зала.
    """
    cfg = Config()
    cfg.diarize_mode = "embed"
    cfg.diarize_refine_threshold = 0.07   # между «свой голос» (~0.01) и Мария/Марина (~0.12)
    return cfg


def run_scenario(scn: Scenario, cfg: Config) -> dict:
    """Гоняет сцену через настоящий роутер и возвращает обе разметки."""
    from scenes import segment_audio

    router = build_router(cfg, encoder=StubEncoder())
    if scn.enroll:
        rng = np.random.default_rng(1234)
        router.enroll_audio({scn.scene.voices[k].name: render(scn.scene.voices[k], 8.0, rng)
                             for k in scn.enroll})

    segs = segment_audio(scn.scene.audio, cfg.pause_split_sec, cfg.max_segment_sec)
    labels: dict[int, str | None] = {}
    changes: dict[int, float] = {}
    confs: dict[int, float] = {}

    for sid, t0, t1 in segs:
        audio = scn.scene.audio[int(t0 * SAMPLE_RATE):int(t1 * SAMPLE_RATE)]
        g = router.assign(sid, audio, t0)
        labels[sid] = g.spk
        confs[sid] = g.conf
        if g.change_at is not None:
            changes[sid] = g.change_at
        # то же, что делает run_edge: склейка кластеров правит и прошлое
        for old, new in router.take_merges().items():
            for k, v in labels.items():
                if v == old:
                    labels[k] = new

    online = dict(labels)
    fixes = router.refine()
    refined = dict(labels)
    for sid, g in fixes.items():
        refined[sid] = g.spk

    spans = {sid: (t0, t1) for sid, t0, t1 in segs}
    enrolled = {scn.scene.voices[k].name for k in scn.enroll}
    hyp_on = [(online[s], *spans[s]) for s in spans]
    hyp_re = [(refined[s], *spans[s]) for s in spans]
    dur = scn.scene.duration

    return {
        "online": score(scn.scene.turns, hyp_on, dur, confident=enrolled),
        "refined": score(scn.scene.turns, hyp_re, dur, confident=enrolled),
        "segments": len(segs),
        "changed_by_refine": len(fixes),
        "changes": changes,
        "confs": confs,
        "labels_online": online,
        "labels_refined": refined,
        "spans": spans,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None, help="прогнать только один")
    ap.add_argument("--verbose", action="store_true", help="посегментно")
    ap.add_argument("--update-baseline", action="store_true")
    args = ap.parse_args()

    cfg = make_cfg()
    scenarios = [s for s in build_all()
                 if args.scenario is None or s.scene.name == args.scenario]
    if not scenarios:
        print(f"нет такого сценария: {args.scenario}", file=sys.stderr)
        return 2

    baseline = {}
    if os.path.exists(BASELINE):
        with open(BASELINE, encoding="utf-8") as f:
            baseline = json.load(f).get("scenarios", {})

    results: dict[str, dict] = {}
    print(f"{'сценарий':<15} {'сегм':>5} {'DER онлайн':>11} {'DER архив':>10} "
          f"{'чужое имя':>10} {'правок':>7}")
    print("─" * 66)

    for scn in scenarios:
        r = run_scenario(scn, cfg)
        on, re_ = r["online"], r["refined"]
        results[scn.scene.name] = {"online": round(on.der, 4), "refined": round(re_.der, 4)}
        arrow = "→" if re_.der < on.der - 1e-9 else (" " if abs(re_.der - on.der) < 1e-9 else "↑")
        print(f"{scn.scene.name:<15} {r['segments']:>5} {on.der:>10.1%} "
              f"{arrow}{re_.der:>9.1%} {re_.wrong_name_rate:>10.1%} {r['changed_by_refine']:>7}")

        if args.verbose:
            print(f"    вопрос: {scn.asks}")
            print(f"    онлайн:  {on.line()}")
            print(f"    архив:   {re_.line()}")
            if r["changes"]:
                pretty = ", ".join(f"sid{s}@{t:.1f}c" for s, t in r["changes"].items())
                print(f"    смена голоса внутри сегмента: {pretty}")
            for sid in sorted(r["spans"]):
                t0, t1 = r["spans"][sid]
                o, rf = r["labels_online"][sid], r["labels_refined"][sid]
                mark = "" if o == rf else f"  ->  {rf}"
                print(f"      sid{sid:<3} {t0:6.2f}–{t1:6.2f}  {str(o):<8} "
                      f"conf {r['confs'][sid]:.2f}{mark}")
            print()

    if args.update_baseline:
        with open(BASELINE, "w", encoding="utf-8") as f:
            json.dump({"tolerance": TOLERANCE, "scenarios": results}, f,
                      ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        print(f"\nbaseline обновлён: {BASELINE}")
        return 0

    if not baseline:
        print("\nbaseline.json нет — сравнивать не с чем. "
              "Зафиксировать текущее: --update-baseline")
        return 0

    regressions = []
    for name, cur in results.items():
        base = baseline.get(name)
        if base is None:
            print(f"\nновый сценарий {name}, в baseline его нет")
            continue
        for kind in ("online", "refined"):
            if cur[kind] > base[kind] + TOLERANCE:
                regressions.append(f"{name}/{kind}: {base[kind]:.1%} -> {cur[kind]:.1%}")

    print()
    if regressions:
        print("РЕГРЕССИЯ:")
        for r in regressions:
            print("  " + r)
        return 1
    print(f"регрессий нет (допуск {TOLERANCE:.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
