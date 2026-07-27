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
from channels import FloorDetector
from config import Config
from diarize import AudioTape, SpeakerGuess, build_router
from segmenter import Segmenter
from timebase import TimeMap
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
        self.timemap = TimeMap()
        self.tape = AudioTape(120.0)

        # В канальном режиме владелец слова сам работает детектором речи:
        # никто не говорит — значит, кормить ASR нечем. Отдельный VAD не нужен.
        self.floor = (FloorDetector(cfg, cfg.diarize_channels, cfg.channels)
                      if cfg.diarize_mode == "channel" else None)
        self.vad = VAD(cfg) if (cfg.vad != "off" and self.floor is None) else None
        self.router = build_router(cfg)
        self.stage = LocalStage(cfg.local_stage_port)
        self.uplink = Uplink(cfg.cloud_ws, cfg.token)
        self.uplink.control_cb = self.on_control

        self.diar_q: asyncio.Queue = asyncio.Queue()
        self.samples = 0                 # абсолютный счётчик, задаёт шкалу времени доклада
        self.lag_ms = 0.0
        # Закрытые сегменты: sid -> {rev, spk}. Нужен, чтобы задним числом
        # переименовать спикера — при склейке кластеров и после офлайн-прохода.
        # Держим только версию и метку, тексты живут у клиента.
        self.sent: dict[int, dict] = {}

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
        t0_abs, t1_abs = self.timemap.span_abs(s.t0, s.t1)
        await self.emit({
            "t": "seg", "sid": s.sid, "rev": s.rev,
            "final": s.final, "stable": ev.stable,
            "lang": self.cfg.language or "auto",
            "spk": s.spk, "spk_conf": round(s.spk_conf, 2),
            "text": text + ((" " + ev.partial_tail) if ev.partial_tail else ""),
            "stable_len": len(text),          # всё правее — нестабильный хвост гипотезы
            "conf": round(s.conf, 3),
            # наружу — абсолютное время доклада, а не часы ASR (см. timebase.py)
            "t0": round(t0_abs, 2), "t1": round(t1_abs, 2),
        })
        if s.final:
            self.sent[s.sid] = {"rev": s.rev, "spk": s.spk, "conf": s.spk_conf}
        # в канальном режиме спикер уже проставлен синхронно, догонять нечего
        if s.final and self.cfg.diarize_mode == "embed":
            await self.diar_q.put((s.sid, t0_abs, t1_abs))

    # -- диаризация вне критического пути ------------------------------------
    async def revise_speaker(self, sid: int, spk: str | None, conf: float, src: str) -> None:
        """Поздняя правка метки. Ревизия того же sid, а не новый сегмент:
        клиент делает upsert по sid и просто перерисует подпись (PROTOCOL.md)."""
        rec = self.sent.get(sid)
        if rec is None or rec["spk"] == spk:
            return
        rec["rev"] += 1
        rec["spk"] = spk
        rec["conf"] = conf
        await self.emit({"t": "spk", "sid": sid, "rev": rec["rev"],
                         "spk": spk, "spk_conf": round(conf, 2), "src": src})

    async def diarizer(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            sid, t0_abs, t1_abs = await self.diar_q.get()
            try:
                audio = self.tape.slice(t0_abs, t1_abs)  # лента живёт в абсолютном времени
                guess = await loop.run_in_executor(None, self.router.assign, sid, audio, t0_abs)
                if guess.spk:
                    await self.revise_speaker(sid, guess.spk, guess.conf, "online")
                if guess.change_at is not None:
                    # Текст задним числом не режем: sid'ы монотонные, между ними
                    # не вставить. Отдаём момент смены — архив и оператор разберут.
                    log.warning("сегмент %d содержит смену голоса на %.2f c", sid, guess.change_at)
                    await self.emit({"t": "spk_change", "sid": sid, "at": guess.change_at})
                # Кластеры, оказавшиеся одним человеком: чиним всю их историю.
                for old, new in self.router.take_merges().items():
                    for s_id, rec in self.sent.items():
                        if rec["spk"] == old:
                            await self.revise_speaker(s_id, new, rec["conf"], "merge")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Диаризация — не критический путь, и умереть целиком она не
                # имеет права: сегмент уже в зале, просто останется без подписи.
                # Без этого одно исключение оставляло доклад без меток до конца,
                # причём молча — очередь продолжала расти.
                log.exception("диаризация упала на сегменте %d: %s", sid, exc)
            finally:
                self.diar_q.task_done()

    async def drain_diarizer(self) -> None:
        """Дождаться, пока разберут очередь. Офлайн-проход обязан видеть
        последние сегменты — иначе он пересоберёт кластеры без них."""
        try:
            await asyncio.wait_for(self.diar_q.join(), timeout=15.0)
        except asyncio.TimeoutError:
            log.warning("диаризация не успела разобрать очередь (%d в хвосте)", self.diar_q.qsize())

    async def refine_speakers(self) -> None:
        """Офлайн-проход в конце доклада. Онлайн решал за 200 мс по первым
        секундам голоса; здесь виден весь доклад целиком, и ранние ошибки —
        лишний кластер, приклеенный чужой голос — наконец видно и можно снять."""
        if not (self.cfg.diarize_refine and self.cfg.diarize_mode == "embed"):
            return
        loop = asyncio.get_running_loop()
        fixes = await loop.run_in_executor(None, self.router.refine)
        if not fixes:
            log.info("офлайн-проход: расхождений с онлайном нет")
            return
        log.info("офлайн-проход: переразмечено %d сегментов из %d", len(fixes), len(self.sent))
        for sid, g in sorted(fixes.items()):
            await self.revise_speaker(sid, g.spk, g.conf, "refine")

    # -- смена владельца слова -----------------------------------------------
    async def on_floor(self, ev) -> None:
        """Порядок важен: сначала добираем хвост ПРЕДЫДУЩЕГО спикера,
        затем закрываем его сегмент, и только потом переключаем имя."""
        tail = self.asr.hard_reset(self.timemap.speech_clock)
        for e in self.seg.push_stable(tail):
            await self.emit_seg(e)
        e = self.seg.flush()
        if e:
            await self.emit_seg(e)

        self.seg.set_speaker(ev.owner)
        if ev.overlap:
            log.warning("наложение речи: %s поверх %s", ", ".join(ev.overlap), ev.owner)
        await self.emit({"t": "floor", "owner": ev.owner, "channel": ev.channel,
                         "overlap": ev.overlap, "at": round(ev.t, 2)})

    # -- телеметрия ----------------------------------------------------------
    async def health(self) -> None:
        while True:
            await asyncio.sleep(2)
            extra = self.floor.status() if self.floor else {}
            await self.emit({"t": "health",
                             "lag_ms": round(self.lag_ms),
                             "silence_dropped": round(self.timemap.silence_dropped, 1),
                             **extra,
                             "uplink": "ok" if self.uplink.healthy else "down",
                             "pending": len(self.uplink.pending),
                             "dropped": self.uplink.dropped,
                             "muted": self.muted})

    # -- главный цикл --------------------------------------------------------
    async def main_loop(self) -> None:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=400)
        stop = threading.Event()
        source = (FileSource(self.cfg.audio_file) if self.cfg.audio_file
                  else MicSource(channels=self.cfg.channels))
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
                    block = frame if frame.ndim > 1 else frame.reshape(-1, 1)
                    t_abs = self.samples / SAMPLE_RATE
                    self.samples += len(block)
                    # Лента пишется ЦЕЛИКОМ и в абсолютном времени: по ней режет
                    # диаризация, и пропуски сделали бы её координаты враньём.
                    self.tape.write(block.mean(axis=1), t_abs)

                    if self.floor is not None:
                        fev = self.floor.push(block, t_abs)
                        if fev:
                            await self.on_floor(fev)
                        pcm = self.floor.asr_input(block)
                    else:
                        pcm = block.mean(axis=1)
                        if self.vad and not self.vad.is_speech(pcm):
                            pcm = None

                    if pcm is not None:
                        self.timemap.feed(len(pcm), t_abs)
                        self.asr.feed(pcm)     # тишину в Whisper не отдаём — галлюцинирует

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
        await self.drain_diarizer()
        await self.refine_speakers()
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
