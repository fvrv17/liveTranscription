"""
Аплинк edge -> cloud.

Смысл разреза системы именно здесь: через ненадёжный канал площадки идёт
только ТЕКСТ (~1-2 кбит/с), а не аудио. Текст легко буферизуется и досылается.
Пока аплинк лежит, экран в зале продолжает работать — он питается локально.

Гарантия: at-least-once с дедупликацией на стороне cloud по (sid, rev).
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import deque

log = logging.getLogger("uplink")


class Uplink:
    def __init__(self, url: str, token: str, buffer_size: int = 5000):
        self.url = url
        self.token = token
        self.pending: deque[dict] = deque(maxlen=buffer_size)   # неподтверждённые
        self.seq = 0
        self.last_ack = 0
        self.ws = None
        self.connected = asyncio.Event()
        self.dropped = 0
        self.control_cb = None      # команды оператора приходят обратной дорогой

    async def run(self) -> None:
        import websockets
        backoff = 0.5
        while True:
            try:
                async with websockets.connect(
                    self.url, additional_headers={"Authorization": f"Bearer {self.token}"},
                    ping_interval=10, ping_timeout=10, max_queue=None,
                ) as ws:
                    self.ws = ws
                    self.connected.set()
                    backoff = 0.5
                    log.info("аплинк поднят: %s", self.url)
                    await self._resend()
                    async for raw in ws:
                        msg = json.loads(raw)
                        if msg.get("t") == "ack":
                            self._ack(msg["seq"])
                        elif self.control_cb:
                            await self.control_cb(msg)
            except Exception as exc:
                self.connected.clear()
                self.ws = None
                log.warning("аплинк упал (%s), переподключение через %.1f c; в буфере %d",
                            exc, backoff, len(self.pending))
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 10.0)

    def _ack(self, seq: int) -> None:
        self.last_ack = max(self.last_ack, seq)
        while self.pending and self.pending[0]["seq"] <= self.last_ack:
            self.pending.popleft()

    async def _resend(self) -> None:
        if not self.pending:
            return
        log.info("досылаю %d сообщений с seq=%d", len(self.pending), self.pending[0]["seq"])
        for msg in list(self.pending):
            await self._raw_send(msg)

    async def _raw_send(self, msg: dict) -> None:
        if self.ws is None:
            return
        try:
            await self.ws.send(json.dumps(msg, ensure_ascii=False))
        except Exception as exc:
            log.warning("отправка не прошла: %s", exc)

    async def send(self, msg: dict) -> None:
        self.seq += 1
        msg["seq"] = self.seq
        if len(self.pending) == self.pending.maxlen:
            self.dropped += 1
        self.pending.append(msg)
        await self._raw_send(msg)

    @property
    def healthy(self) -> bool:
        return self.ws is not None
