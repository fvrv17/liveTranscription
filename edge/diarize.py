"""
Диаризация — «кто говорит».

Два режима, и выбор между ними — организационный, а не технический:

  mode=channel   Реализован в channels.py: спикер известен синхронно,
                 асинхронная диаризация не нужна вовсе.

  mode=embed     Есть только суммарный микс. Эмбеддинг голоса на закоммиченном
                 куске речи + кластеризация по косинусу.

Ключевые приёмы, которые делают mode=embed пригодным:
  1. Энроллмент на саундчеке: 10 с речи каждого панелиста -> именованные
     центроиды. На выходе «Мария Петрова», а не «Speaker 2». Там же —
     единственный момент, когда ещё можно переставить микрофоны: если два
     голоса неразличимы, оператор узнаёт об этом до начала, а не после.
  2. Метка спикера НЕ блокирует субтитр. Сегмент уходит в зал с spk=null,
     а метка прилетает отдельной ревизией через ~200 мс. Ровно тот же
     механизм ревизий, что и для partial/final.

Почему одного онлайн-прохода мало. Жадная кластеризация принимает решение по
первым секундам голоса и живёт с ним до конца доклада: шумный первый сэмпл
навсегда порождает лишний кластер, а поздний похожий голос навсегда
приклеивается к чужому. Оба случая чинятся только взглядом назад, поэтому
здесь два уровня:

  ОНЛАЙН (assign)   — решение за ~200 мс, право на ошибку есть.
  ОФЛАЙН (refine)   — агломеративная перекластеризация по накопленным
                      эмбеддингам; расходится с онлайном — шлём ревизии.

Ревизии прошлых сегментов ничего не стоят: протокол с самого начала считает
сегмент изменяемой сущностью с версией (см. PROTOCOL.md), а клиент делает
upsert по sid. То есть исправление истории — не костыль, а штатный ход.

Чего этот модуль не делает: overlap. Два одновременных голоса дают один
смешанный вектор, и он не близок ни к одному из них. В канальном режиме
наложение видно (channels.py), здесь — нет.
"""
from __future__ import annotations

import logging
import os
import threading
import wave
from dataclasses import dataclass

import numpy as np

from encoders import build_encoder, l2

log = logging.getLogger("diarize")
SAMPLE_RATE = 16000
FRAME = 512                      # 32 мс — кадр энергетического гейта


@dataclass
class SpeakerGuess:
    spk: str | None
    conf: float
    change_at: float | None = None   # смена говорящего ВНУТРИ сегмента, абс. время
    n_windows: int = 0


# ── окна и отбор речи ───────────────────────────────────────────────────────
@dataclass
class Window:
    t0: float
    t1: float
    emb: np.ndarray


def voiced(audio: np.ndarray, range_db: float, floor_db: float) -> np.ndarray:
    """Выбросить паузы внутри куска.

    Эмбеддинг считается по тому, что ему скормили. Кусок ленты между t0 и t1 —
    это не непрерывная речь: там межсловные паузы, вдохи и хвост тишины до
    закрытия сегмента. Половина окна из тишины смещает вектор в сторону
    «тихой комнаты», и все спикеры начинают походить друг на друга.

    Порог берём ОТНОСИТЕЛЬНО максимума этого же куска, а не абсолютный:
    гейн петлички и трибунного микрофона отличаются на десяток децибел,
    и единый абсолютный порог отрезал бы тихого спикера целиком.
    """
    n = (len(audio) // FRAME) * FRAME
    if n == 0:
        return audio
    frames = audio[:n].reshape(-1, FRAME)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    db = 20.0 * np.log10(rms)
    keep = (db >= db.max() - range_db) & (db >= floor_db)
    if not keep.any():
        return np.zeros(0, dtype=np.float32)
    return frames[keep].reshape(-1)


class WindowEmbedder:
    """Режет кусок на перекрывающиеся окна и считает вектор на каждом.

    Одного вектора на сегмент не хватает по двум причинам. Во-первых, сегмент
    закрывается по пунктуации и паузе, а не по смене голоса, — в embed-режиме
    некому поставить границу на переходе, и реплика двух человек попадает в
    один сегмент. Во-вторых, по набору окон видно разброс: устойчивая метка и
    метка, которая скачет от окна к окну, не должны иметь одинаковый spk_conf.
    """

    def __init__(self, cfg, encoder):
        self.cfg = cfg
        self.enc = encoder

    def __call__(self, audio: np.ndarray, t0_abs: float) -> list[Window]:
        win = int(self.cfg.diarize_window_sec * SAMPLE_RATE)
        hop = int(self.cfg.diarize_hop_sec * SAMPLE_RATE)
        need = int(self.cfg.diarize_min_voiced_sec * SAMPLE_RATE)
        out: list[Window] = []

        # Кусок короче окна — не выбрасываем: короткие реплики («да», «согласен»)
        # это половина диалога на панели. Считаем один укороченный вектор.
        starts = range(0, max(1, len(audio) - win + 1), hop) if len(audio) >= win else [0]
        for s in starts:
            chunk = audio[s:s + win]
            speech = voiced(chunk, self.cfg.diarize_voiced_range_db, self.cfg.diarize_voiced_floor_db)
            if len(speech) < need:
                continue
            emb = self.enc.encode(speech)
            if emb is None:
                continue
            out.append(Window(t0=t0_abs + s / SAMPLE_RATE,
                              t1=t0_abs + min(s + win, len(audio)) / SAMPLE_RATE,
                              emb=emb))
        return out


# ── онлайн-кластеризация ────────────────────────────────────────────────────
def _sigmoid(x: float, temp: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x / max(temp, 1e-6))))


def _confidence(best: float, second: float, thr: float, temp: float, margin_ref: float) -> float:
    """Одна шкала [0,1] на все случаи.

    В прошлой версии совпадение возвращало косинус, а новый спикер —
    (1 - косинус), то есть в одном поле протокола жили две несовместимые
    величины: 0.8 значило то «уверенно узнали», то «уверенно не узнали».
    Здесь conf везде означает одно — «насколько можно верить этой метке».

    Складываются два признака: запас над порогом (узнали ли вообще) и отрыв
    от второго места (не спутали ли с соседом). Отрыв сравнивается с
    margin_ref, и это ключевое: он обязан уметь наложить ВЕТО. Иначе
    достаточно чуть переползти порог, чтобы получить conf > 0.5 — даже когда
    второй кандидат отстал на сотую и выбор между двумя фамилиями по сути
    подброшен монетой. Для зала это худший из возможных исходов, хуже
    отсутствия подписи.
    """
    return _sigmoid((best - thr) + (best - second - margin_ref), temp)


class OnlineClusterer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.centroids: dict[str, np.ndarray] = {}
        self.counts: dict[str, int] = {}
        self.enrolled: set[str] = set()
        self.merges: dict[str, str] = {}     # алиас -> победитель, для правки истории
        self._anon = 0

    # -- поиск ---------------------------------------------------------------
    def _rank(self, emb: np.ndarray) -> list[tuple[str, float]]:
        return sorted(((s, float(np.dot(emb, c))) for s, c in self.centroids.items()),
                      key=lambda p: p[1], reverse=True)

    def _new_cluster(self, emb: np.ndarray) -> str:
        self._anon += 1
        spk = f"S{self._anon}"
        self.centroids[spk] = emb.copy()
        self.counts[spk] = 1
        return spk

    def _update(self, spk: str, emb: np.ndarray) -> None:
        """Скользящее среднее с ПОЛОМ на скорость обучения.

        Обычное «среднее по всем наблюдениям» выглядит разумно, но через сотню
        реплик вес нового наблюдения падает до 1/100, и центроид перестаёт
        реагировать. Спикер отвернулся от микрофона или взял другой — центроид
        остался в прошлом. Поэтому вес не опускается ниже 1/adapt_window.
        """
        if spk in self.enrolled:
            return                       # энроллмент — якорь, его не размываем
        n = self.counts.get(spk, 1)
        w = 1.0 / min(n + 1, max(2, self.cfg.diarize_adapt_window))
        upd = l2((1.0 - w) * self.centroids[spk] + w * emb)
        if upd is not None:
            self.centroids[spk] = upd
        self.counts[spk] = n + 1

    # -- склейка -------------------------------------------------------------
    def merge_pass(self) -> None:
        """Схлопнуть кластеры, которые оказались одним человеком.

        Порог склейки ВЫШЕ порога назначения. Это не описка: назначение
        ошибается в сторону дробления (сомневаешься — заведи новый кластер),
        а склейка должна быть консервативной, иначе она сольёт двух реальных
        людей, и это уже не чинится ничем.
        """
        for a in [s for s in self.centroids if s not in self.enrolled]:
            if a not in self.centroids:
                continue
            for b in list(self.centroids):
                if b == a or b not in self.centroids:
                    continue
                if float(np.dot(self.centroids[a], self.centroids[b])) < self.cfg.diarize_merge_threshold:
                    continue
                # выживает более представительный; энролленный не проигрывает никогда
                loser, winner = ((a, b) if (b in self.enrolled or self.counts[b] >= self.counts[a])
                                 else (b, a))
                if loser in self.enrolled:
                    continue
                nl, nw = self.counts[loser], self.counts[winner]
                if winner not in self.enrolled:
                    m = l2(self.centroids[winner] * nw + self.centroids[loser] * nl)
                    if m is not None:
                        self.centroids[winner] = m
                self.counts[winner] = nl + nw
                del self.centroids[loser], self.counts[loser]
                # старые метки тоже переклеиваем: сегменты с loser уже в зале
                self.merges[loser] = winner
                for k, v in list(self.merges.items()):
                    if v == loser:
                        self.merges[k] = winner
                log.info("склеены кластеры: %s -> %s", loser, winner)
                break

    # -- основной вход -------------------------------------------------------
    def observe(self, emb: np.ndarray, learn: bool = True) -> tuple[str | None, float]:
        rank = self._rank(emb)
        best, best_sim = rank[0] if rank else (None, -1.0)
        second_sim = rank[1][1] if len(rank) > 1 else -1.0
        thr, temp = self.cfg.diarize_threshold, self.cfg.diarize_conf_temp
        ref = self.cfg.diarize_margin_ref

        if best is not None and best_sim >= thr:
            if learn:
                self._update(best, emb)
            return best, _confidence(best_sim, second_sim, thr, temp, ref)

        if len(self.centroids) >= self.cfg.diarize_max_speakers:
            # Мест нет. Раньше здесь молча возвращалась лучшая метка, какой бы
            # далёкой она ни была, — на экране появлялось уверенное чужое имя.
            # Возвращаем её же, но с честно низкой уверенностью: выше по стеку
            # diarize_min_conf решит, показывать метку или оставить null.
            return best, _confidence(best_sim, second_sim, thr, temp, ref)

        if not learn:
            return best, _confidence(best_sim, second_sim, thr, temp, ref)

        spk = self._new_cluster(emb)
        # Уверенность нового кластера ограничена сверху: за ним ровно одно
        # наблюдение. Наберёт ещё — пойдёт обычным путём и получит нормальный
        # conf. Считается по тому, насколько уверенно голос НЕ похож ни на кого
        # известного: заведённый впритык к порогу кластер уверенным не бывает.
        conf = min(self.cfg.diarize_new_conf_cap, _sigmoid(thr - best_sim - ref, temp))
        log.info("новый спикер: %s (лучшее сходство было %.2f)", spk, best_sim)
        return spk, conf

    def take_merges(self) -> dict[str, str]:
        m, self.merges = self.merges, {}
        return m


# ── офлайн-перекластеризация ────────────────────────────────────────────────
def agglomerative(embs: np.ndarray, weights: np.ndarray, threshold: float,
                  block: np.ndarray | None = None) -> np.ndarray:
    """Average-linkage по косинусному расстоянию. Возвращает метки кластеров.

    Своя реализация вместо scipy: на edge не должно быть лишней зависимости
    ради полусотни строк, а входов здесь тысячи, не миллионы — кластеризуем
    по одному вектору на сегмент, а не по каждому окну.

    Average linkage, а не single: single склеивает через цепочку промежуточных
    векторов и на голосах регулярно сливает всех в один кластер.

    Числа кластеров на входе НЕТ, и это осознанно. Соблазн подставить сюда
    количество панелистов с саундчека велик, но оно не годится ни как точное
    значение, ни даже как граница: на саундчеке записаны не все, кто возьмёт
    микрофон (вопрос из зала), и не все записанные обязательно заговорят.
    Обе ошибки проверяются сценариями late_joiner и backchannels.

    block — запрет на слияние: сегменты с РАЗНЫМИ неотрицательными значениями
    в один кластер не попадут никогда. Через него заходит единственное, что
    саундчек действительно знает наверняка: это Мария, а это Иван, и они
    разные люди. Слить двух панелистов — самая дорогая ошибка здесь, и лучше
    запретить её структурно, чем надеяться на порог.
    """
    n = len(embs)
    if n < 2:
        return np.zeros(n, dtype=int)

    d = 1.0 - np.einsum("ik,jk->ij", embs, embs)
    np.fill_diagonal(d, np.inf)
    tag = (np.full(n, -1, dtype=int) if block is None else np.asarray(block, dtype=int).copy())
    conflict = (tag[:, None] >= 0) & (tag[None, :] >= 0) & (tag[:, None] != tag[None, :])
    d[conflict] = np.inf

    size = weights.astype(float).copy()
    alive = np.ones(n, dtype=bool)
    parent = np.arange(n)

    while alive.sum() > 1:
        masked = np.where(alive[:, None] & alive[None, :], d, np.inf)
        i, j = divmod(int(np.argmin(masked)), n)
        best = masked[i, j]
        if not np.isfinite(best) or best > threshold:
            break

        # Lance-Williams для average linkage
        si, sj = size[i], size[j]
        d[i, :] = (si * d[i, :] + sj * d[j, :]) / (si + sj)
        d[:, i] = d[i, :]
        d[i, i] = np.inf
        size[i] = si + sj
        alive[j] = False
        parent[parent == j] = i

        # объединённый кластер наследует метку и вместе с ней все запреты
        tag[i] = max(tag[i], tag[j])
        if tag[i] >= 0:
            bad = alive & (tag >= 0) & (tag != tag[i])
            d[i, bad] = np.inf
            d[bad, i] = np.inf

    _, labels = np.unique(parent, return_inverse=True)
    return labels


class OfflineRefiner:
    """Копит по одному вектору на сегмент и в конце пересобирает разметку."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.sids: list[int] = []
        self.embs: list[np.ndarray] = []
        self.weights: list[float] = []
        self.online: list[str | None] = []

    def note(self, sid: int, emb: np.ndarray, dur: float, spk: str | None) -> None:
        self.sids.append(sid)
        self.embs.append(emb)
        self.weights.append(max(dur, 0.1))
        self.online.append(spk)

    def refine(self, enrolled: dict[str, np.ndarray]) -> dict[int, SpeakerGuess]:
        """Возвращает ТОЛЬКО расхождения с онлайновой разметкой."""
        if len(self.sids) < 2:
            return {}
        X = np.vstack(self.embs)
        w = np.asarray(self.weights, dtype=float)

        # Сегменты, уверенно опознанные по саундчеку, размечаем заранее —
        # это и есть запрет на слияние разных панелистов (см. agglomerative).
        roster = sorted(enrolled)
        block = None
        if roster:
            E = np.vstack([enrolled[n] for n in roster])
            sims = np.einsum("ik,jk->ij", X, E)
            block = np.where(sims.max(axis=1) >= self.cfg.diarize_threshold,
                             sims.argmax(axis=1), -1)

        labels = agglomerative(X, w, self.cfg.diarize_refine_threshold, block)

        centroids: dict[int, np.ndarray] = {}
        for k in np.unique(labels):
            m = labels == k
            c = l2((X[m] * w[m, None]).sum(axis=0))
            if c is not None:
                centroids[int(k)] = c

        names: dict[int, str] = {}
        taken_n: set[str] = set()
        if block is not None:
            for k in centroids:
                tags = {int(t) for t in block[labels == k] if t >= 0}
                if tags:                       # по построению их не больше одной
                    names[k] = roster[tags.pop()]
                    taken_n.add(names[k])
        # Кластер без опознанных сегментов ещё может оказаться панелистом,
        # который весь доклад говорил тихо: добираем жадно, строго один к одному.
        if enrolled:
            pairs = sorted(((float(np.dot(c, enrolled[name])), k, name)
                            for k, c in centroids.items() if k not in names
                            for name in roster),
                           key=lambda p: p[0], reverse=True)
            for sim, k, name in pairs:
                if k in names or name in taken_n or sim < self.cfg.diarize_threshold:
                    continue
                names[k] = name
                taken_n.add(name)
        for k in centroids:
            names.setdefault(k, f"R{k + 1}")

        out: dict[int, SpeakerGuess] = {}
        for sid, k, emb, prev in zip(self.sids, labels, self.embs, self.online):
            spk = names[int(k)]
            if spk == prev:
                continue
            sim = float(np.dot(emb, centroids[int(k)]))
            out[sid] = SpeakerGuess(spk=spk, conf=round(min(1.0, max(0.0, sim)), 3))
        return out


# ── роутер ──────────────────────────────────────────────────────────────────
class EmbeddingRouter:
    def __init__(self, cfg, encoder=None):
        self.cfg = cfg
        self.enc = encoder if encoder is not None else build_encoder(cfg)
        self.win = WindowEmbedder(cfg, self.enc)
        self.clust = OnlineClusterer(cfg)
        self.refiner = OfflineRefiner(cfg)
        self.enroll_embs: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()      # assign живёт в executor-потоке
        self._since_merge = 0
        if cfg.diarize_enroll_dir and os.path.isdir(cfg.diarize_enroll_dir):
            self.enroll(cfg.diarize_enroll_dir)

    # -- энроллмент ----------------------------------------------------------
    def enroll(self, path: str) -> None:
        """Имя файла = имя спикера."""
        samples = {os.path.splitext(fn)[0]: read_wav(os.path.join(path, fn))
                   for fn in sorted(os.listdir(path)) if fn.lower().endswith(".wav")}
        self.enroll_audio(samples)

    def enroll_audio(self, samples: dict[str, np.ndarray]) -> None:
        """Вектор усредняем по окнам: одна точка, снятая с одного вдоха,
        ловит интонацию этой секунды, а не голос."""
        for name, audio in samples.items():
            wins = self.win(audio, 0.0)
            if not wins:
                log.warning("энроллмент %s: речи не найдено, пропускаю", name)
                continue
            c = l2(np.mean([w.emb for w in wins], axis=0))
            if c is None:
                continue
            self.enroll_embs[name] = c
            self.clust.centroids[name] = c
            self.clust.counts[name] = 1
            self.clust.enrolled.add(name)
            log.info("энроллмент: %s (%.1f c, окон %d)", name, len(audio) / SAMPLE_RATE, len(wins))
        self._report_separability()

    def _report_separability(self) -> None:
        """Единственный момент, когда ещё можно что-то сделать.

        Если голоса двух панелистов ближе порога назначения, диаризация будет
        путать их весь доклад — и никакая настройка в рантайме этого не спасёт.
        Оператор должен увидеть это на саундчеке, пока микрофоны ещё можно
        развести по каналам (mode=channel) и снять вопрос совсем.
        """
        names = sorted(self.enroll_embs)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                sim = float(np.dot(self.enroll_embs[a], self.enroll_embs[b]))
                if sim >= self.cfg.diarize_threshold:
                    log.warning("энроллмент: %s и %s почти неразличимы (косинус %.2f >= порога %.2f) "
                                "— развести по каналам пульта или перезаписать образцы",
                                a, b, sim, self.cfg.diarize_threshold)

    # -- назначение ----------------------------------------------------------
    def assign(self, sid: int, audio: np.ndarray, t0_abs: float = 0.0) -> SpeakerGuess:
        wins = self.win(audio, t0_abs)
        if not wins:
            return SpeakerGuess(None, 0.0)

        with self._lock:
            # Метки окон — только чтобы увидеть смену голоса внутри сегмента,
            # центроиды по ним не двигаем: иначе одно наблюдение учтётся столько
            # раз, сколько в нём окон, и длинная реплика перевесит всё остальное.
            per_win = [self.clust.observe(w.emb, learn=False) for w in wins]
            mean = l2(np.mean([w.emb for w in wins], axis=0))
            if mean is None:
                return SpeakerGuess(None, 0.0)
            spk, conf = self.clust.observe(mean)

            self._since_merge += 1
            if self._since_merge >= self.cfg.diarize_merge_every:
                self._since_merge = 0
                self.clust.merge_pass()
                spk = self.clust.merges.get(spk, spk)

            self.refiner.note(sid, mean, wins[-1].t1 - wins[0].t0, spk)

        change_at = self._change_point(wins, per_win)
        if change_at is not None:
            # Метку не дробим: sid уже уехал в зал, а sid'ы монотонные и задним
            # числом между ними не вставить. Но врать про уверенность на
            # смешанном сегменте нельзя — режем её и сообщаем момент смены.
            conf *= self.cfg.diarize_mixed_penalty
        if conf < self.cfg.diarize_min_conf:
            # Пустая метка честнее уверенно неправильной: зал простит «—»,
            # но не простит чужую фамилию под цитатой.
            return SpeakerGuess(None, round(conf, 3), change_at, len(wins))
        return SpeakerGuess(spk, round(conf, 3), change_at, len(wins))

    def _change_point(self, wins: list[Window],
                      per_win: list[tuple[str | None, float]]) -> float | None:
        """Сегмент на двух голосах: метки окон делятся на префикс и суффикс.

        Точность момента ограничена шагом окна и лучше него не будет. Окна
        перекрываются, поэтому первое окно с новой меткой содержит ещё и хвост
        предыдущего голоса — его начало как оценка смены уходит в прошлое почти
        на целое окно. Смена лежит между концом последнего окна старого голоса
        и концом первого окна нового; берём середину этого интервала.
        """
        labels = [s for s, _ in per_win]
        if len(labels) < 2 or len(set(labels)) < 2:
            return None
        for i in range(1, len(labels)):
            head, tail = set(labels[:i]), set(labels[i:])
            if len(head) != 1 or len(tail) != 1 or head == tail:
                continue
            a = l2(np.mean([w.emb for w in wins[:i]], axis=0))
            b = l2(np.mean([w.emb for w in wins[i:]], axis=0))
            if a is not None and b is not None and float(np.dot(a, b)) < self.cfg.diarize_threshold:
                return round((wins[i - 1].t1 + wins[i].t1) / 2.0, 2)
        return None

    # -- ревизии -------------------------------------------------------------
    def take_merges(self) -> dict[str, str]:
        with self._lock:
            return self.clust.take_merges()

    def refine(self) -> dict[int, SpeakerGuess]:
        with self._lock:
            return self.refiner.refine(self.enroll_embs)


class NullRouter:
    def assign(self, sid: int, audio: np.ndarray, t0_abs: float = 0.0) -> SpeakerGuess:
        return SpeakerGuess(None, 0.0)

    def take_merges(self) -> dict[str, str]:
        return {}

    def refine(self) -> dict[int, SpeakerGuess]:
        return {}


def build_router(cfg, encoder=None):
    if cfg.diarize_mode == "embed":
        return EmbeddingRouter(cfg, encoder)
    return NullRouter()


def read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        data = w.readframes(w.getnframes())
        a = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            a = a.reshape(-1, w.getnchannels()).mean(axis=1)
    return a


class AudioTape:
    """
    Кольцевой буфер последних N секунд аудио.
    Диаризация работает по УЖЕ закоммиченному сегменту, вырезая его
    временной интервал из ленты — поэтому она и не добавляет задержки
    в основной путь субтитра.
    """

    def __init__(self, seconds: float = 120.0):
        self.cap = int(seconds * SAMPLE_RATE)
        self.buf = np.zeros(0, dtype=np.float32)
        self.origin = 0.0

    def write(self, pcm: np.ndarray, t_start: float) -> None:
        if len(self.buf) == 0:
            self.origin = t_start
        self.buf = np.concatenate([self.buf, pcm])
        if len(self.buf) > self.cap:
            drop = len(self.buf) - self.cap
            self.buf = self.buf[drop:]
            self.origin += drop / SAMPLE_RATE

    def slice(self, t0: float, t1: float) -> np.ndarray:
        i0 = int(max(0.0, t0 - self.origin) * SAMPLE_RATE)
        i1 = int(max(0.0, t1 - self.origin) * SAMPLE_RATE)
        return self.buf[i0:min(i1, len(self.buf))]
