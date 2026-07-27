"""
Хранилище транскриптов.

SQLite достаточно для прототипа и для одного зала: запись — десятки строк
в минуту. Дедупликация по (talk, sid): приходящая ревизия перезаписывает
предыдущую, поэтому повторная доставка после реконнекта аплинка безопасна.

Вместе с текстом храним среднюю уверенность ASR по сегменту: по ней видно,
на каких участках доклада модель плыла.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS talks (
  talk TEXT PRIMARY KEY, title TEXT, src_lang TEXT,
  started REAL, ended REAL, state TEXT, summary TEXT
);
CREATE TABLE IF NOT EXISTS segments (
  talk TEXT, sid INTEGER, rev INTEGER, final INTEGER,
  spk TEXT, spk_conf REAL, lang TEXT, text TEXT, conf REAL,
  t0 REAL, t1 REAL, emitted_at REAL,
  PRIMARY KEY (talk, sid)
);
CREATE TABLE IF NOT EXISTS translations (
  talk TEXT, sid INTEGER, rev INTEGER, lang TEXT, text TEXT,
  PRIMARY KEY (talk, sid, lang)
);
CREATE INDEX IF NOT EXISTS idx_seg_talk ON segments(talk, sid);
"""


class Store:
    def __init__(self, path: str = "captions.db"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript(SCHEMA)
            self.db.commit()

    # -- доклады -------------------------------------------------------------
    def upsert_talk(self, talk: str, **kw) -> None:
        with self.lock:
            self.db.execute("INSERT OR IGNORE INTO talks(talk, started, state) VALUES(?,?,?)",
                            (talk, time.time(), "live"))
            for k, v in kw.items():
                if v is None:                     # частичное обновление не должно затирать поля
                    continue
                if k in ("title", "src_lang", "state", "summary", "ended"):
                    self.db.execute(f"UPDATE talks SET {k}=? WHERE talk=?", (v, talk))
            self.db.commit()

    def get_talk(self, talk: str) -> dict | None:
        with self.lock:
            r = self.db.execute("SELECT * FROM talks WHERE talk=?", (talk,)).fetchone()
        return dict(r) if r else None

    def list_talks(self) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT talk,title,src_lang,state,started FROM talks "
                                   "ORDER BY started DESC").fetchall()
        return [dict(r) for r in rows]

    # -- сегменты ------------------------------------------------------------
    def put_segment(self, m: dict) -> bool:
        """Возвращает False, если пришла устаревшая ревизия (дубль после реконнекта)."""
        with self.lock:
            cur = self.db.execute("SELECT rev FROM segments WHERE talk=? AND sid=?",
                                  (m["talk"], m["sid"])).fetchone()
            if cur and cur["rev"] >= m["rev"]:
                return False
            self.db.execute(
                "INSERT OR REPLACE INTO segments"
                "(talk,sid,rev,final,spk,spk_conf,lang,text,conf,t0,t1,emitted_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (m["talk"], m["sid"], m["rev"], int(m.get("final", False)),
                 m.get("spk"), m.get("spk_conf", 0.0), m.get("lang"), m.get("text", ""),
                 m.get("conf", 0.0), m.get("t0", 0.0), m.get("t1", 0.0), m.get("emitted_at", 0.0)))
            self.db.commit()
        return True

    def set_speaker(self, talk: str, sid: int, rev: int, spk: str, conf: float) -> None:
        with self.lock:
            self.db.execute("UPDATE segments SET spk=?, spk_conf=?, rev=? "
                            "WHERE talk=? AND sid=? AND rev<?", (spk, conf, rev, talk, sid, rev))
            self.db.commit()

    def edit_segment(self, talk: str, sid: int, text: str) -> dict | None:
        """Ручная правка оператором: уезжает клиентам новой ревизией."""
        with self.lock:
            r = self.db.execute("SELECT * FROM segments WHERE talk=? AND sid=?",
                                (talk, sid)).fetchone()
            if not r:
                return None
            rev = r["rev"] + 1
            self.db.execute("UPDATE segments SET text=?, rev=? WHERE talk=? AND sid=?",
                            (text, rev, talk, sid))
            self.db.execute("DELETE FROM translations WHERE talk=? AND sid=?", (talk, sid))
            self.db.commit()
        return {"t": "seg", "talk": talk, "sid": sid, "rev": rev, "final": True, "stable": True,
                "lang": r["lang"], "spk": r["spk"], "text": text, "stable_len": len(text),
                "t0": r["t0"], "t1": r["t1"]}

    def segments(self, talk: str, since_sid: int = 0, finals_only: bool = True) -> list[dict]:
        q = "SELECT * FROM segments WHERE talk=? AND sid>?"
        if finals_only:
            q += " AND final=1"
        q += " ORDER BY sid"
        with self.lock:
            rows = self.db.execute(q, (talk, since_sid)).fetchall()
        return [dict(r) for r in rows]

    # -- переводы ------------------------------------------------------------
    def get_translation(self, talk: str, sid: int, lang: str, rev: int) -> str | None:
        with self.lock:
            r = self.db.execute("SELECT rev,text FROM translations WHERE talk=? AND sid=? AND lang=?",
                                (talk, sid, lang)).fetchone()
        return r["text"] if r and r["rev"] >= rev else None

    def put_translation(self, talk: str, sid: int, rev: int, lang: str, text: str) -> None:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO translations(talk,sid,rev,lang,text) "
                            "VALUES(?,?,?,?,?)", (talk, sid, rev, lang, text))
            self.db.commit()

    def translated_segments(self, talk: str, lang: str, since_sid: int = 0) -> list[dict]:
        with self.lock:
            rows = self.db.execute(
                "SELECT s.sid, s.rev, s.spk, s.t0, s.t1, "
                "COALESCE(tr.text, s.text) AS text, (tr.text IS NULL) AS untranslated "
                "FROM segments s LEFT JOIN translations tr "
                "ON tr.talk=s.talk AND tr.sid=s.sid AND tr.lang=? "
                "WHERE s.talk=? AND s.sid>? AND s.final=1 ORDER BY s.sid",
                (lang, talk, since_sid)).fetchall()
        return [dict(r) for r in rows]
