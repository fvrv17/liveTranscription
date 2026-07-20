#!/usr/bin/env python3
"""
Edge-агент площадки.

Отвечает за всё, что должно пережить обрыв интернета:
  захват -> VAD -> ASR -> сегментация -> (а) локальный экран сцены
                                         (б) аплинк текста в облако

Диаризация идёт отдельной дорожкой и НЕ задерживает субтитр:
сегмент уходит сразу, метка спикера прилетает следующей ревизией.

    python run_edge.py --config config.venue.yaml
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import threading
import time

import numpy as np

from asr import StreamingASR, WhisperBackend, SAMPLE_RATE
from audio import FRAME, VAD, FileSource, MicSource
from config import Config
from diarize import AudioTape, SpeakerGuess, build_router
from segmenter import Segmenter
from uplink import Uplink

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-9s %(levelname)-7s %(message)s")
log = logging.getLogger("edge")


# ─────────────────────────────────────────────────────────────────────────────
# Локальный сокет для экрана сцены
# ─────────────────────────────────────────────────────────────────────────────
class LocalStage:
    """Экран за спикером питается ОТСЮДА, а не из облака.
    Упал аплинк — зал продолжает читать субтитры."""

    def __init__(self, port: int):
        self.port = port
        self.clients: set = set()
        self.recent: list[dict] = []

    async def serve(self) -> None:
        import websockets
        async def handler(ws):
            self.clients.add(ws)
            log.info("экран сцены подключён (всего %d)", len(self.clients))
            try:
                for m in self.recent[-40:]:
                    await ws.send(json.dumps(m, ensure_ascii=False))
                async for _ in ws:
                    pass
            finally:
                self.clients.discard(ws)
        async with websockets.serve(handler, "0.0.0.0", self.port, ping_interval=20):
            log.info("локальный экран сцены: ws://0.0.0.0:%d", self.port)
            await asyncio.Future()

    async def broadcast(self, msg: dict) -> None:
        if msg.get("t") == "seg" and msg.get("final"):
            self.recent.append(msg)
            self.recent = self.recent[-200:]
        if not self.clients:
            return
        payload = json.dumps(msg, ensure_ascii=False)
        for ws in list(self.clients):
            try:
                await ws.send(payload)
            except Exception:
                self.clients.discard(ws)


# ─────────────────────────────────────────────────────────────────────────────
# Захват в отдельном потоке
# ─────────────────────────────────────────────────────────────────────────────
def capture_thread(source, loop, queue: asyncio.Queue, stop: threading.Event) -> None:
    def offer(item):
        # Потребитель отстал — отбрасываем кадр, но не роняем захват.
        # В проде это же место должно инкрементить счётчик в телеметрии.
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            pass

    with source as src:
        while not stop.is_set():
            try:
                frame = src.read()
            except EOFError:
                loop.call_soon_threadsafe(offer, None)
                return
            except Exception as exc:
                log.error("ошибка захвата: %s", exc)
                return
            loop.call_soon_threadsafe(offer, frame)


# ─────────────────────────────────────────────────────────────────────────────
class Edge:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.muted = False
        self.stopping = False

        self.backend = WhisperBackend(cfg)
        self.asr = StreamingASR(self.backend, cfg)
        self.seg = Segmenter(cfg)
        self.vad = VAD(cfg) if cfg.vad != "off" else None
        self.tape = AudioTape(120.0)
        self.router = build_router(cfg)
        self.stage = LocalStage(cfg.local_stage_port)
        self.uplink = Uplink(cfg.cloud_ws, cfg.token)
        self.uplink.control_cb = self.on_control

        self.diar_q: asyncio.Queue = asyncio.Queue()
        self.samples = 0                 # абсолютный счётчик, задаёт шкалу времени доклада
        self.lag_ms = 0.0

    # -- команды оператора ---------------------------------------------------
    async def on_control(self, msg: dict) -> None:
        if msg.get("t") != "ctl":
            return
        cmd = msg.get("cmd")
        log.info("команда оператора: %s", cmd)
        if cmd == "mute":                # kill switch: спикер сказал что-то off-record
            self.muted = True
            await self.emit({"t": "state", "muted": True})
        elif cmd == "unmute":
            self.muted = False
            await self.emit({"t": "state", "muted": False})
        elif cmd == "end":
            self.stopping = True

    # -- отправка ------------------------------------------------------------
    async def emit(self, msg: dict) -> None:
        msg.setdefault("talk", self.cfg.talk_id)
        msg.setdefault("emitted_at", time.time())
        await self.stage.broadcast(msg)
        await self.uplink.send(msg)

    async def emit_seg(self, ev) -> None:
        if self.muted:
            return
        s = ev.seg
        text = s.text
        if not text:
            return
        await self.emit({
            "t": "seg", "sid": s.sid, "rev": s.rev,
            "final": s.final, "stable": ev.stable,
            "lang": self.cfg.language or "auto",
            "spk": s.spk, "spk_conf": round(s.spk_conf, 2),
            "text": text + ((" " + ev.partial_tail) if ev.partial_tail else ""),
            "stable_len": len(text),          # всё правее — нестабильный хвост гипотезы
            "conf": round(s.conf, 3),
            "t0": round(s.t0, 2), "t1": round(s.t1, 2),
        })
        if s.final and self.cfg.diarize_mode != "off":
            await self.diar_q.put(s)

    # -- диаризация вне критического пути ------------------------------------
    async def diarizer(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            s = await self.diar_q.get()
            audio = self.tape.slice(s.t0, s.t1)
            guess = await loop.run_in_executor(None, self.router.assign, audio)
            if guess.spk and guess.spk != s.spk:
                s.spk, s.spk_conf, s.rev = guess.spk, guess.conf, s.rev + 1
                await self.emit({"t": "spk", "sid": s.sid, "rev": s.rev,
                                 "spk": guess.spk, "spk_conf": round(guess.conf, 2)})

    # -- телеметрия ----------------------------------------------------------
    async def health(self) -> None:
        while True:
            await asyncio.sleep(2)
            await self.emit({"t": "health",
                             "lag_ms": round(self.lag_ms),
                             "uplink": "ok" if self.uplink.healthy else "down",
                             "pending": len(self.uplink.pending),
                             "dropped": self.uplink.dropped,
                             "muted": self.muted})

    # -- главный цикл --------------------------------------------------------
    async def main_loop(self) -> None:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=400)
        stop = threading.Event()
        source = FileSource(self.cfg.audio_file) if self.cfg.audio_file else MicSource()
        threading.Thread(target=capture_thread, args=(source, loop, q, stop), daemon=True).start()

        await self.emit({"t": "talk", "state": "live", "title": self.cfg.title,
                         "src_lang": self.cfg.language or "auto"})

        last_tick = time.monotonic()
        try:
            while not self.stopping:
                try:
                    frame = await asyncio.wait_for(q.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    frame = None

                if frame is None:
                    if isinstance(source, FileSource):
                        break
                else:
                    t_abs = self.samples / SAMPLE_RATE
                    self.samples += len(frame)
                    self.tape.write(frame, t_abs)
                    speech = self.vad.is_speech(frame) if self.vad else True
                    if speech:
                        self.asr.feed(frame)   # тишину в Whisper не отдаём — галлюцинирует

                if self.asr.ready():
                    t_start = time.monotonic()
                    stable, partial = await loop.run_in_executor(None, self.asr.poll)
                    self.lag_ms = (time.monotonic() - t_start) * 1000 + self.cfg.chunk_sec * 1000

                    for ev in self.seg.push_stable(stable):
                        await self.emit_seg(ev)
                    ev = self.seg.push_partial(partial)
                    if ev:
                        await self.emit_seg(ev)

                if time.monotonic() - last_tick > 0.3:
                    last_tick = time.monotonic()
                    ev = self.seg.tick()
                    if ev:
                        await self.emit_seg(ev)

        finally:
            stop.set()

        # конец доклада: добираем хвост и просим облако собрать саммари
        for w in [self.asr.flush()]:
            if w:
                for ev in self.seg.push_stable(w):
                    await self.emit_seg(ev)
        ev = self.seg.flush()
        if ev:
            await self.emit_seg(ev)
        await asyncio.sleep(1.0)                     # дать диаризации доехать
        await self.emit({"t": "talk", "state": "ended"})
        await asyncio.sleep(2.0)                     # дать аплинку слить буфер
        log.info("доклад завершён")

    async def run(self) -> None:
        tasks = [asyncio.create_task(self.uplink.run()),
                 asyncio.create_task(self.stage.serve()),
                 asyncio.create_task(self.diarizer()),
                 asyncio.create_task(self.health())]
        try:
            await self.main_loop()
        finally:
            for t in tasks:
                t.cancel()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.venue.yaml")
    ap.add_argument("--talk", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    if args.talk:
        cfg.talk_id = args.talk
    log.info("edge стартует: model=%s diarize=%s talk=%s", cfg.model, cfg.diarize_mode, cfg.talk_id)
    asyncio.run(Edge(cfg).run())


if __name__ == "__main__":
    main()
