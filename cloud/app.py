#!/usr/bin/env python3
"""
Cloud component.

Handles text reception from Edge, translation, fan-out via WebSocket, storage,
summaries, and the operator console. Audio is not sent here at all—only text is transmitted via
the site’s unreliable uplink.

    uvicorn app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse

from bus import Bus, topic_ctl, topic_seg
from store import Store
from summarize import Summarizer
from translate import LANGS, Glossary, Translator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)-9s %(levelname)-7s %(message)s")
log = logging.getLogger("cloud")

TOKEN = os.getenv("INGEST_TOKEN", "dev-token")

app = FastAPI(title="Live Captions")
bus = Bus()
store = Store(os.getenv("DB_PATH", "captions.db"))

glossary = Glossary({
    # Conference terminology. Do not translate, or translate strictly 
    "Anthropic": {}, "Whisper": {}, "LocalAgreement": {},
    "аплинк": {"en": "uplink"}, "диаризация": {"en": "diarization"},
})
translator = Translator(store, bus, glossary)
summarizer = Summarizer(store)

edge_sockets: dict[str, WebSocket] = {}     # talk_id -> socket edge-agent (for the back commands)
health: dict[str, dict] = {}


async def on_lang_activated(talk: str, lang: str) -> None:
    """Первый слушатель включил язык — догоняем последние сегменты."""
    src = (store.get_talk(talk) or {}).get("src_lang")
    if lang != src:
        asyncio.create_task(translator.translate_backlog(talk, lang))


bus.on_lang_activated = on_lang_activated



# received from edge

@app.websocket("/ws/ingest")
async def ws_ingest(ws: WebSocket):
    auth = ws.headers.get("authorization", "")
    if auth != f"Bearer {TOKEN}":
        await ws.close(code=4401)
        return
    await ws.accept()
    talk = None
    log.info("edge подключён")
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            talk = msg.get("talk", talk)
            if talk:
                edge_sockets[talk] = ws
            await handle_edge_message(msg)
            if "seq" in msg:
                await ws.send_text(json.dumps({"t": "ack", "seq": msg["seq"]}))
    except WebSocketDisconnect:
        log.warning("edge отключился (talk=%s)", talk)
    finally:
        if talk and edge_sockets.get(talk) is ws:
            edge_sockets.pop(talk, None)


async def handle_edge_message(msg: dict) -> None:
    t, talk = msg.get("t"), msg.get("talk")
    if not talk:
        return

    if t == "talk":
        store.upsert_talk(talk, title=msg.get("title"), src_lang=msg.get("src_lang"),
                          state=msg.get("state"))
        await bus.publish(topic_ctl(talk), msg)
        await bus.publish(topic_seg(talk, msg.get("src_lang", "ru")), msg)
        if msg.get("state") == "ended":
            store.upsert_talk(talk, ended=time.time())
            asyncio.create_task(make_summary(talk))

    elif t == "seg":
        fresh = store.put_segment(msg)          # deduplication of redeliveries
        src = msg.get("lang", "ru")
        await bus.publish(topic_seg(talk, src), msg)
        if fresh and msg.get("final") and msg.get("stable"):
            asyncio.create_task(translator.on_segment(msg))

    elif t == "spk":
        store.set_speaker(talk, msg["sid"], msg["rev"], msg["spk"], msg.get("spk_conf", 0))
        src = (store.get_talk(talk) or {}).get("src_lang", "ru")
        await bus.publish(topic_seg(talk, src), msg)

    elif t in ("health", "state"):
        health[talk] = {**msg, "at": time.time()}
        await bus.publish(topic_ctl(talk), msg)


async def make_summary(talk: str) -> None:
    log.info("формирую саммари: %s", talk)
    md = await summarizer.build(talk)
    store.upsert_talk(talk, summary=md, state="ended")
    await bus.publish(topic_ctl(talk), {"t": "summary", "talk": talk, "md": md})
    for lang in bus.active_langs(talk):
        await bus.publish(topic_seg(talk, lang), {"t": "summary", "talk": talk, "md": md})
    log.info("саммари готово: %s (%d символов)", talk, len(md))



# viewers

@app.websocket("/ws/view")
async def ws_view(ws: WebSocket):
    await ws.accept()
    talk = ws.query_params.get("talk", "")
    lang = ws.query_params.get("lang", "ru")
    if not talk:
        await ws.close(code=4400)
        return

    info = store.get_talk(talk) or {}
    src = info.get("src_lang") or "ru"
    q: asyncio.Queue = asyncio.Queue(maxsize=500)
    subscribed: set[str] = set()

    async def attach(new_lang: str):
        nonlocal lang
        for t in list(subscribed):
            bus.unsubscribe(t, q)
            subscribed.discard(t)
        bus.lang_detach(talk, lang)
        lang = new_lang
        await bus.lang_attach(talk, lang)
        for t in {topic_seg(talk, lang), topic_seg(talk, src), topic_ctl(talk)}:
            bus.subscribe(t, q)
            subscribed.add(t)
        # snapshot person who joins in the middle of a presentation can see the context
        rows = store.translated_segments(talk, lang)[-40:]
        await ws.send_text(json.dumps({"t": "snapshot", "talk": talk, "lang": lang,
                                       "src_lang": src, "title": info.get("title"),
                                       "state": info.get("state"), "segments": rows},
                                      ensure_ascii=False))

    await bus.lang_attach(talk, lang)
    bus.lang_detach(talk, lang)     # reset the counter the attach command below will increment it again
    await attach(lang)

    async def pump():
        while True:
            msg = await q.get()
            await ws.send_text(json.dumps(msg, ensure_ascii=False))

    pump_task = asyncio.create_task(pump())
    try:
        while True:
            data = json.loads(await ws.receive_text())
            if data.get("t") == "setlang":       # changing the language without reconnecting

                new = data.get("lang", lang)
                if new != lang:
                    await attach(new)
    except WebSocketDisconnect:
        pass
    finally:
        pump_task.cancel()
        for t in subscribed:
            bus.unsubscribe(t, q)
        bus.lang_detach(talk, lang)



# REST

@app.get("/api/langs")
def api_langs():
    return LANGS


@app.get("/api/talks")
def api_talks():
    return store.list_talks()


@app.get("/api/talks/{talk}/transcript")
def api_transcript(talk: str, lang: str = ""):
    return store.translated_segments(talk, lang) if lang else store.segments(talk)


@app.get("/api/talks/{talk}/summary")
def api_summary(talk: str):
    info = store.get_talk(talk)
    if not info:
        raise HTTPException(404, "доклад не найден")
    return {"talk": talk, "state": info["state"], "summary": info["summary"]}


@app.post("/api/talks/{talk}/summary")
async def api_make_summary(talk: str):
    if not store.get_talk(talk):
        raise HTTPException(404, "доклад не найден")
    await make_summary(talk)
    return {"ok": True, "summary": store.get_talk(talk)["summary"]}


@app.post("/api/talks/{talk}/control")
async def api_control(talk: str, payload: dict):
    """Пульт оператора: mute (kill switch), unmute, end."""
    ws = edge_sockets.get(talk)
    if not ws:
        raise HTTPException(409, "edge-агент не подключён")
    await ws.send_text(json.dumps({"t": "ctl", "cmd": payload.get("cmd")}))
    return {"ok": True}


@app.post("/api/talks/{talk}/segments/{sid}")
async def api_edit(talk: str, sid: int, payload: dict):
    """Ручная правка оператором: улетает клиентам новой ревизией."""
    msg = store.edit_segment(talk, sid, payload.get("text", ""))
    if not msg:
        raise HTTPException(404, "сегмент не найден")
    src = (store.get_talk(talk) or {}).get("src_lang", "ru")
    await bus.publish(topic_seg(talk, src), msg)
    asyncio.create_task(translator.on_segment(msg))
    return {"ok": True, "rev": msg["rev"]}


@app.get("/api/talks/{talk}/stats")
def api_stats(talk: str):
    return {"listeners": bus.listeners(talk),
            "health": health.get(talk, {}),
            "mt": translator.stats,
            "edge_connected": talk in edge_sockets}


@app.get("/healthz")
def healthz():
    return PlainTextResponse("ok")
