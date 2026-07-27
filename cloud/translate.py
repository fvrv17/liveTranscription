"""
Перевод.

Три правила, каждое из которых напрямую влияет на счёт и на UX:

  1. Переводим ТОЛЬКО final && stable сегменты. Перевод partial-гипотез стоит
     в разы дороже и на экране «дёргается»: MT переписывает всю фразу от
     одного нового слова.
  2. Переводим ТОЛЬКО языки, у которых есть живой подписчик (bus.active_langs).
     Из 12 языков в UI активны обычно 2-3 — счёт падает в 4-6 раз.
  3. Кэш по (talk, sid, lang) и глоссарий. Термины конференции защищаются
     плейсхолдерами до перевода и возвращаются после — иначе MT переведёт
     название продукта.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re

log = logging.getLogger("mt")

LANGS = {
    "en": "English", "ru": "Русский", "sr": "Srpski", "de": "Deutsch", "fr": "Français",
    "es": "Español", "it": "Italiano", "pt": "Português", "tr": "Türkçe",
    "zh": "中文", "ja": "日本語", "ar": "العربية", "uk": "Українська", "pl": "Polski",
}


class Glossary:
    """Термины, которые нельзя переводить и нельзя коверкать."""

    def __init__(self, terms: dict[str, dict[str, str]] | None = None):
        # {"термин в оригинале": {"en": "Term", "de": "Term"}}; пустой словарь => не переводить
        self.terms = terms or {}
        self._re = self._compile()

    def _compile(self):
        if not self.terms:
            return None
        pat = "|".join(sorted((re.escape(k) for k in self.terms), key=len, reverse=True))
        return re.compile(rf"(?<!\w)({pat})(?!\w)", re.IGNORECASE)

    def protect(self, text: str) -> tuple[str, dict[str, str]]:
        if not self._re:
            return text, {}
        found: dict[str, str] = {}
        def sub(m):
            key = f"§{len(found)}§"
            found[key] = m.group(1)
            return key
        return self._re.sub(sub, text), found

    def restore(self, text: str, found: dict[str, str], lang: str) -> str:
        for key, original in found.items():
            repl = self.terms.get(original, {}).get(lang) \
                or self.terms.get(original.lower(), {}).get(lang) or original
            text = text.replace(key, repl)
        return text


# ─────────────────────────────────────────────────────────────────────────────
class DeepLEngine:
    def __init__(self, key: str):
        self.key = key
        self.url = ("https://api-free.deepl.com/v2/translate" if key.endswith(":fx")
                    else "https://api.deepl.com/v2/translate")

    async def translate(self, text: str, src: str, tgt: str) -> str:
        import httpx
        async with httpx.AsyncClient(timeout=8.0) as c:
            r = await c.post(self.url,
                             headers={"Authorization": f"DeepL-Auth-Key {self.key}"},
                             data={"text": text, "target_lang": tgt.upper(),
                                   "source_lang": src.upper() if src != "auto" else None})
            r.raise_for_status()
            return r.json()["translations"][0]["text"]


class OpenAICompatEngine:
    """Любой OpenAI-совместимый endpoint, включая локальный vLLM с NLLB/Qwen."""

    def __init__(self, base: str, key: str, model: str):
        self.base, self.key, self.model = base.rstrip("/"), key, model

    async def translate(self, text: str, src: str, tgt: str) -> str:
        import httpx
        prompt = (f"Translate from {LANGS.get(src, src)} to {LANGS.get(tgt, tgt)}. "
                  f"This is a live conference subtitle: keep it short, keep the register, "
                  f"do not add commentary, output only the translation.\n\n{text}")
        async with httpx.AsyncClient(timeout=8.0) as c:
            r = await c.post(f"{self.base}/chat/completions",
                             headers={"Authorization": f"Bearer {self.key}"},
                             json={"model": self.model, "temperature": 0.1, "max_tokens": 300,
                                   "messages": [{"role": "user", "content": prompt}]})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()


def build_engine():
    if os.getenv("DEEPL_KEY"):
        log.info("MT: DeepL")
        return DeepLEngine(os.environ["DEEPL_KEY"])
    if os.getenv("OPENAI_API_KEY"):
        log.info("MT: OpenAI-совместимый endpoint")
        return OpenAICompatEngine(os.getenv("OPENAI_BASE", "https://api.openai.com/v1"),
                                  os.environ["OPENAI_API_KEY"],
                                  os.getenv("MT_MODEL", "gpt-4o-mini"))
    log.warning("MT: ключи не заданы — перевод отключён, клиенты получают оригинал")
    return None


# ─────────────────────────────────────────────────────────────────────────────
class Translator:
    def __init__(self, store, bus, glossary: Glossary | None = None, concurrency: int = 6):
        self.store = store
        self.bus = bus
        self.engine = build_engine()
        self.glossary = glossary or Glossary()
        self.sem = asyncio.Semaphore(concurrency)
        self.stats = {"calls": 0, "chars": 0, "cache_hits": 0, "errors": 0}

    async def on_segment(self, msg: dict) -> None:
        """Точка входа: вызывается на final && stable сегменте."""
        if self.engine is None:
            return
        talk, sid, rev = msg["talk"], msg["sid"], msg["rev"]
        src = msg.get("lang", "auto")
        targets = {l for l in self.bus.active_langs(talk) if l != src}
        if not targets:
            return
        await asyncio.gather(*(self._one(talk, sid, rev, src, l, msg["text"]) for l in targets))

    async def translate_backlog(self, talk: str, lang: str, limit: int = 30) -> None:
        """Кто-то первым включил язык — догоняем последние N сегментов,
        чтобы человек не смотрел в пустой экран."""
        if self.engine is None:
            return
        rows = [r for r in self.store.segments(talk)[-limit:]]
        for r in rows:
            await self._one(talk, r["sid"], r["rev"], r["lang"], lang, r["text"], publish=False)

    async def _one(self, talk, sid, rev, src, tgt, text, publish: bool = True) -> None:
        cached = self.store.get_translation(talk, sid, tgt, rev)
        if cached is not None:
            self.stats["cache_hits"] += 1
            return
        protected, found = self.glossary.protect(text)
        try:
            async with self.sem:
                out = await self.engine.translate(protected, src, tgt)
            self.stats["calls"] += 1
            self.stats["chars"] += len(text)
        except Exception as exc:
            self.stats["errors"] += 1
            log.warning("перевод %s->%s не удался: %s", src, tgt, exc)
            return                       # деградация: клиент увидит оригинал, а не пустоту
        out = self.glossary.restore(out, found, tgt)
        self.store.put_translation(talk, sid, rev, tgt, out)
        if publish:
            from bus import topic_seg
            await self.bus.publish(topic_seg(talk, tgt),
                                   {"t": "tr", "talk": talk, "sid": sid, "rev": rev,
                                    "lang": tgt, "text": out})
