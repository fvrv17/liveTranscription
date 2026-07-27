"""
Стриминговый ASR: faster-whisper + LocalAgreement-N.

LocalAgreement — слово коммитим только когда оно совпало в N последних
гипотезах подряд. Это превращает "мигающий" вывод Whisper в стабильный
поток и стоит нам ровно +1 чанк задержки.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger("asr")

SAMPLE_RATE = 16000


@dataclass
class Word:
    text: str
    t0: float
    t1: float
    prob: float = 1.0


def _norm(s: str) -> str:
    """Нормализация для сравнения гипотез: пунктуация и регистр не должны
    мешать признать слово стабильным."""
    return re.sub(r"[^\w]", "", s.lower())


# ─────────────────────────────────────────────────────────────────────────────
# LocalAgreement
# ─────────────────────────────────────────────────────────────────────────────
class LocalAgreement:
    """
    Гипотезы приходят над одним и тем же незакоммиченным буфером аудио,
    поэтому всегда начинаются с одной точки — общий префикс определён корректно.
    """

    def __init__(self, n: int = 2):
        self.n = max(1, n)
        self.history: list[list[Word]] = []

    def insert(self, hypothesis: list[Word]) -> list[Word]:
        """Возвращает слова, которые только что стали стабильными."""
        self.history.append(list(hypothesis))
        if len(self.history) > self.n:
            self.history.pop(0)
        if len(self.history) < self.n:
            return []

        committed: list[Word] = []
        limit = min(len(h) for h in self.history)
        for i in range(limit):
            cands = [h[i] for h in self.history]
            if all(_norm(c.text) == _norm(cands[0].text) for c in cands):
                # берём вариант из последней гипотезы: у неё лучше пунктуация,
                # т.к. модель уже видела правый контекст
                committed.append(cands[-1])
            else:
                break

        if committed:
            k = len(committed)
            self.history = [h[k:] for h in self.history]
        return committed

    def reset(self) -> None:
        self.history.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Защита от галлюцинаций
# ─────────────────────────────────────────────────────────────────────────────
# Whisper обучался на веб-аудио, значительная часть которого — ролики
# с субтитровыми дорожками. Служебная обвязка этих дорожек запомнилась моделью,
# и на тишине декодер, лишённый акустических опор, скатывается в неё как
# в наиболее вероятный априор. Это артефакт модели, а не предметной области.
#
# Архитектура фильтра принципиальна: решение принимает АКУСТИКА, текстовые
# правила могут только усилить уже имеющееся подозрение. Обратный порядок
# опасен — блоклист не переносится между языками и моделями, и любая фраза
# в нём рискует оказаться нормальной репликой спикера.
#
#   1. no_speech_prob        — прямая оценка модели «речи здесь нет»
#   2. avg_logprob           — декодер не уверен ни в одном токене
#   3. compression_ratio     — вырожденный, самоповторяющийся текст
#   4. детектор ран-повторов — луп декодера, ловится без словарей
#   5. боилерплейт           — ТОЛЬКО как тайбрейкер к п.1-3

# Фразы, которые в докладе не звучат никогда. Список намеренно короткий:
# он не должен и не может быть полным, это лишь добивание пограничных случаев.
# Расширяется через vocabulary_blocklist в конфиге под конкретную модель и язык.
#
# Чего здесь СОЗНАТЕЛЬНО нет: «спасибо за внимание», «вопросы?», «аплодисменты».
# Это нормальные реплики конференции; фильтр по ним гасит экран ровно в тот
# момент, когда спикер заканчивает доклад.
DEFAULT_BOILERPLATE = [
    r"субтитр(ы|ов)\s+(сделал|создавал|подготовил|редактор)",
    r"подписывайтесь\s+на\s+канал",
    r"subtitles?\s+(by|provided\s+by)",
    r"thanks?\s+for\s+watching",
]


def _compile_boilerplate(cfg) -> list:
    pats = list(getattr(cfg, "boilerplate_patterns", None) or DEFAULT_BOILERPLATE)
    return [re.compile(p, re.IGNORECASE) for p in pats]


def is_degenerate(text: str) -> bool:
    """Луп декодера: мало уникальных токенов на длинном отрезке.
    Работает на любом языке и не требует списка фраз."""
    toks = [_norm(w) for w in text.split() if _norm(w)]
    if len(toks) < 6:
        return False
    if len(set(toks)) <= 2:
        return True
    # повтор биграммы четыре раза подряд
    for i in range(len(toks) - 7):
        bg = toks[i:i + 2]
        if all(toks[i + 2 * k:i + 2 * k + 2] == bg for k in range(1, 4)):
            return True
    return False


def hallucination_verdict(seg, text: str, boilerplate: list, cfg) -> str | None:
    """Возвращает причину отбраковки или None. Логируется на пульт оператора:
    молчаливое проглатывание речи страшнее ложного пропуска."""
    t = text.strip()
    if not t:
        return "пусто"

    nsp = getattr(seg, "no_speech_prob", 0.0)
    logp = getattr(seg, "avg_logprob", 0.0)
    cr = getattr(seg, "compression_ratio", 1.0)

    # --- акустика: самостоятельные основания ---
    if nsp >= cfg.no_speech_threshold:
        return f"no_speech_prob={nsp:.2f}"
    if cr >= cfg.compression_ratio_threshold:
        return f"compression_ratio={cr:.2f}"
    if is_degenerate(t):
        return "вырожденный повтор"

    # --- боилерплейт: только поверх акустического сомнения ---
    marginal = nsp >= cfg.marginal_no_speech or logp <= cfg.logprob_threshold
    if marginal and any(p.search(t) for p in boilerplate):
        return f"боилерплейт при no_speech={nsp:.2f}, logprob={logp:.2f}"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Бэкенды
# ─────────────────────────────────────────────────────────────────────────────
class WhisperBackend:
    def __init__(self, cfg):
        from faster_whisper import WhisperModel  # ленивый импорт

        self.cfg = cfg
        log.info("загружаю faster-whisper %s (%s/%s)", cfg.model, cfg.device, cfg.compute_type)
        self.model = WhisperModel(cfg.model, device=cfg.device, compute_type=cfg.compute_type)
        # hotwords = словарь терминов конференции. Без него имена докладчиков,
        # названия продуктов и аббревиатуры распознаются позорно.
        self.hotwords = " ".join(cfg.vocabulary) if cfg.vocabulary else None
        self.boilerplate = _compile_boilerplate(cfg)
        self.dropped = 0

    def transcribe(self, audio: np.ndarray, prompt: str, offset: float) -> list[Word]:
        segments, _info = self.model.transcribe(
            audio,
            language=self.cfg.language,          # None => автодетект (рискованно при code-switching)
            task="transcribe",
            word_timestamps=True,
            condition_on_previous_text=False,    # иначе луп-галлюцинации копятся
            initial_prompt=prompt or None,
            hotwords=self.hotwords,
            beam_size=self.cfg.beam_size,
            temperature=0.0,
            vad_filter=False,                    # VAD у нас свой, до ASR
        )
        out: list[Word] = []
        for seg in segments:
            reason = hallucination_verdict(seg, seg.text, self.boilerplate, self.cfg)
            if reason:
                self.dropped += 1
                log.warning("отброшен сегмент (%s): %r", reason, seg.text.strip()[:80])
                continue
            for w in seg.words or []:
                out.append(
                    Word(text=w.word.strip(), t0=offset + w.start, t1=offset + w.end,
                         prob=getattr(w, "probability", 1.0))
                )
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Стриминговая обёртка
# ─────────────────────────────────────────────────────────────────────────────
class StreamingASR:
    """
    feed()  — подкладываем PCM (float32, 16 кГц, mono)
    poll()  — раз в chunk_sec декодируем весь незакоммиченный буфер и
              возвращаем (новые стабильные слова, нестабильный хвост)
    """

    def __init__(self, backend, cfg):
        self.backend = backend
        self.cfg = cfg
        self.agree = LocalAgreement(cfg.agreement_n)
        self.buf = np.zeros(0, dtype=np.float32)
        self.buf_origin = 0.0          # абсолютное время начала буфера, сек
        self.since_poll = 0            # сэмплов накоплено с прошлого poll
        self.committed_tail: list[str] = []   # для initial_prompt
        self.last_partial: list[Word] = []

    def feed(self, pcm: np.ndarray) -> None:
        self.buf = np.concatenate([self.buf, pcm])
        self.since_poll += len(pcm)

    def ready(self) -> bool:
        return self.since_poll >= self.cfg.chunk_sec * SAMPLE_RATE

    def poll(self) -> tuple[list[Word], list[Word]]:
        if not self.ready() or len(self.buf) < SAMPLE_RATE * 0.4:
            return [], self.last_partial
        self.since_poll = 0

        prompt = " ".join(self.committed_tail[-40:])
        hyp = self.backend.transcribe(self.buf, prompt, self.buf_origin)
        new_stable = self.agree.insert(hyp)

        if new_stable:
            self.committed_tail.extend(w.text for w in new_stable)
            self.committed_tail = self.committed_tail[-60:]
            # подрезаем буфер по концу последнего стабильного слова
            cut = new_stable[-1].t1 - self.buf_origin
            n = int(max(0.0, cut) * SAMPLE_RATE)
            if n > 0:
                self.buf = self.buf[n:]
                self.buf_origin += n / SAMPLE_RATE

        # аварийный сброс: буфер разросся (длинная фраза без стабилизации)
        if len(self.buf) > self.cfg.max_buffer_sec * SAMPLE_RATE:
            log.warning("буфер переполнен, принудительный сброс")
            keep = int(self.cfg.max_buffer_sec * 0.5 * SAMPLE_RATE)
            self.buf_origin += (len(self.buf) - keep) / SAMPLE_RATE
            self.buf = self.buf[-keep:]
            self.agree.reset()

        stable_norm = {_norm(w.text) for w in new_stable}
        self.last_partial = [w for w in hyp if _norm(w.text) not in stable_norm][-12:]
        return new_stable, self.last_partial

    def flush(self) -> list[Word]:
        """Конец доклада: коммитим всё, что осталось в гипотезе."""
        tail = self.last_partial
        self.last_partial = []
        self.agree.reset()
        return tail

    def hard_reset(self, new_origin: float) -> list[Word]:
        """Смена входного канала. Буфер содержит аудио ПРЕДЫДУЩЕГО микрофона —
        дописывать в него речь нового спикера нельзя, гипотеза склеит двоих.
        Забираем хвост, чистим всё, переносим начало отсчёта."""
        tail = self.flush()
        self.buf = np.zeros(0, dtype=np.float32)
        self.buf_origin = new_origin
        self.since_poll = 0
        self.committed_tail.clear()          # initial_prompt чужого спикера вредит
        return tail
