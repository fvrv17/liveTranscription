"""
Pub/Sub + реестр активных языков.

В прототипе — in-process (asyncio). Интерфейс намеренно совпадает с NATS/Redis
Streams: publish(topic, msg) / subscribe(topic) -> async iterator, чтобы замена
на реальный брокер при масштабировании на несколько залов была механической.

Реестр языков — это то, на чём экономятся основные деньги: переводим только
те языки, которые кто-то реально слушает. Из 12 предложенных в UI активными
на докладе обычно оказываются 2-3.
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

log = logging.getLogger("bus")


class Bus:
    def __init__(self):
        self._subs: dict[str, set[asyncio.Queue]] = defaultdict(set)
        self._lang_counts: dict[tuple[str, str], int] = defaultdict(int)
        self.on_lang_activated = None      # колбэк: язык стал востребован

    async def publish(self, topic: str, msg: dict) -> None:
        for q in list(self._subs.get(topic, ())):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                log.warning("подписчик отстаёт, сообщение отброшено: %s", topic)

    def subscribe(self, topic: str, q: asyncio.Queue | None = None) -> asyncio.Queue:
        q = q or asyncio.Queue(maxsize=500)
        self._subs[topic].add(q)
        return q

    def unsubscribe(self, topic: str, q: asyncio.Queue) -> None:
        self._subs[topic].discard(q)
        if not self._subs[topic]:
            self._subs.pop(topic, None)

    # -- учёт языков ---------------------------------------------------------
    async def lang_attach(self, talk: str, lang: str) -> None:
        key = (talk, lang)
        self._lang_counts[key] += 1
        if self._lang_counts[key] == 1:
            log.info("язык активирован: %s/%s", talk, lang)
            if self.on_lang_activated:
                await self.on_lang_activated(talk, lang)

    def lang_detach(self, talk: str, lang: str) -> None:
        key = (talk, lang)
        self._lang_counts[key] = max(0, self._lang_counts[key] - 1)
        if self._lang_counts[key] == 0:
            self._lang_counts.pop(key, None)
            log.info("язык больше не слушают: %s/%s", talk, lang)

    def active_langs(self, talk: str) -> set[str]:
        return {lang for (t, lang), n in self._lang_counts.items() if t == talk and n > 0}

    def listeners(self, talk: str) -> dict[str, int]:
        return {lang: n for (t, lang), n in self._lang_counts.items() if t == talk and n > 0}


def topic_seg(talk: str, lang: str) -> str:
    return f"talk:{talk}:lang:{lang}"


def topic_ctl(talk: str) -> str:
    return f"talk:{talk}:ctl"
