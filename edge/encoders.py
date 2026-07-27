"""
Энкодер голоса: аудио -> вектор, близкий для одного человека и далёкий для разных.

Зачем отдельный модуль, если реализация по сути одна (ECAPA). Причина не в
том, что завтра появится вторая модель, а в том, что без подменяемого
энкодера диаризацию нельзя проверить. ECAPA тянет torch, весит сотни
мегабайт и требует GPU, поэтому в CI её не будет никогда — а логика
кластеризации, оконного разбиения и склейки кластеров ошибается ровно так
же независимо от того, кто считает вектор. Отсюда контракт ниже и реестр:
eval подставляет детерминированный энкодер и гоняет тот же самый код.

Контракт:
  * на выходе L2-нормированный вектор -> косинус равен скалярному произведению;
  * None, если по этому куску вектор считать бессмысленно (слишком коротко,
    модель упала) — вызывающий обязан это пережить, диаризация не имеет
    права ронять субтитр.
"""
from __future__ import annotations

import logging
from typing import Callable, Protocol

import numpy as np

log = logging.getLogger("encoder")
SAMPLE_RATE = 16000


class Encoder(Protocol):
    dim: int

    def encode(self, audio: np.ndarray) -> np.ndarray | None:
        """audio: моно float32 @16 кГц. Возврат: L2-нормированный вектор или None."""


def l2(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    return (v / n) if n > 1e-9 else None


class EcapaEncoder:
    """ECAPA-TDNN из speechbrain. Загружается лениво: на площадке модель
    поднимается на саундчеке, а в тестах не поднимается вовсе."""

    dim = 192

    def __init__(self, cfg):
        self.cfg = cfg
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from speechbrain.inference.speaker import EncoderClassifier
            log.info("загружаю ECAPA-TDNN (device=%s)", self.cfg.device)
            self._model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=self.cfg.diarize_model_dir,
                run_opts={"device": self.cfg.device},
            )
        return self._model

    def encode(self, audio: np.ndarray) -> np.ndarray | None:
        try:
            import torch
            with torch.no_grad():
                t = torch.from_numpy(np.ascontiguousarray(audio)).float().unsqueeze(0)
                e = self.model.encode_batch(t).squeeze().cpu().numpy()
            return l2(e)
        except Exception as exc:
            # Диаризация — не критический путь: сегмент уже ушёл в зал без метки.
            # Здесь нельзя падать, но и молчать нельзя: без счётчика в телеметрии
            # сломанная модель выглядит как «просто никого не узнаём».
            log.warning("эмбеддинг не посчитан: %s", exc)
            return None


# ── реестр ──────────────────────────────────────────────────────────────────
_REGISTRY: dict[str, Callable[[object], Encoder]] = {"ecapa": EcapaEncoder}


def register_encoder(name: str, factory: Callable[[object], Encoder]) -> None:
    _REGISTRY[name] = factory


def build_encoder(cfg) -> Encoder:
    name = getattr(cfg, "diarize_encoder", "ecapa")
    if name not in _REGISTRY:
        raise ValueError(f"неизвестный энкодер {name!r}; есть: {sorted(_REGISTRY)}")
    return _REGISTRY[name](cfg)
