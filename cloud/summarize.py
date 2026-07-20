"""
Presentation Summary.

It is based on a transcript from the repository (rather than generated on the fly), so:
  * We split a long presentation into segments and perform a map-reduce;
  * If there was a diary-style transcript, we count the remarks by speaker and create a Q&A section;
  * if there are no keys—we use extractive fallback so the prototype works offline.

It’s important to note what’s often overlooked: LLMs hallucinate NUMBERS. Therefore, the prompt explicitly prohibits
numbers that aren’t in the transcript, and in the UI, the summary is marked as a draft
until confirmed by an operator.
"""
from __future__ import annotations

import logging
import os
import re
from collections import Counter

log = logging.getLogger("sum")

MAP_PROMPT = """Ниже — фрагмент стенограммы доклада с конференции (распознан автоматически, возможны ошибки).
Выпиши сжатые заметки: ключевые тезисы, названные технологии/термины, приведённые цифры и факты.
Не выдумывай ничего, чего нет в тексте. Числа приводи только те, что есть дословно.
Пиши на языке оригинала. Только заметки, без предисловий.

--- ФРАГМЕНТ ---
{chunk}"""

REDUCE_PROMPT = """Ты готовишь итоговое саммари доклада «{title}» для сайта конференции.
Ниже — заметки по фрагментам, по порядку.

Собери итог в Markdown по структуре:
## О чём доклад
(2-3 предложения)
## Ключевые тезисы
(4-7 пунктов)
## Термины и технологии
(перечисление, если были)
## Цифры и факты
(только те, что явно есть в заметках; если их нет — напиши «не прозвучали»)
## Вопросы из зала
(если в заметках есть следы Q&A; иначе опусти секцию)

Не добавляй ничего, чего нет в заметках. Не выдумывай цифры.

--- ЗАМЕТКИ ---
{notes}"""


def chunk_text(segments: list[dict], window_chars: int = 6000) -> list[str]:
    chunks, cur, n = [], [], 0
    for s in segments:
        line = (f"[{s['spk']}] " if s.get("spk") else "") + (s.get("text") or "")
        if n + len(line) > window_chars and cur:
            chunks.append("\n".join(cur))
            cur, n = [], 0
        cur.append(line)
        n += len(line)
    if cur:
        chunks.append("\n".join(cur))
    return chunks


class Summarizer:
    def __init__(self, store):
        self.store = store
        self.key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        self.base = os.getenv("OPENAI_BASE", "https://api.openai.com/v1")
        self.model = os.getenv("SUMMARY_MODEL", "gpt-4o-mini")

    async def _llm(self, prompt: str, max_tokens: int = 900) -> str:
        import httpx
        async with httpx.AsyncClient(timeout=90.0) as c:
            r = await c.post(f"{self.base}/chat/completions",
                             headers={"Authorization": f"Bearer {self.key}"},
                             json={"model": self.model, "temperature": 0.2,
                                   "max_tokens": max_tokens,
                                   "messages": [{"role": "user", "content": prompt}]})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()

    async def build(self, talk: str) -> str:
        info = self.store.get_talk(talk) or {}
        segs = self.store.segments(talk)
        if not segs:
            return "_Стенограмма пуста — саммари не сформировано._"

        title = info.get("title") or talk
        stats = self._stats(segs)

        if not self.key:
            body = self._extractive(segs)
            log.warning("ключа LLM нет — экстрактивное саммари")
        else:
            try:
                chunks = chunk_text(segs)
                notes = []
                for i, ch in enumerate(chunks, 1):
                    notes.append(await self._llm(MAP_PROMPT.format(chunk=ch), 600))
                    log.info("map %d/%d", i, len(chunks))
                body = await self._llm(
                    REDUCE_PROMPT.format(title=title, notes="\n\n".join(notes)), 1200)
            except Exception as exc:
                log.error("LLM-саммари не удалось (%s), фолбэк", exc)
                body = self._extractive(segs)

        return f"# {title}\n\n{body}\n\n---\n{stats}\n\n" \
               f"_Черновик: сформировано автоматически по распознанной речи, требует проверки оператором._"

    def _stats(self, segs: list[dict]) -> str:
        dur = (segs[-1]["t1"] - segs[0]["t0"]) / 60 if segs else 0
        words = sum(len((s.get("text") or "").split()) for s in segs)
        by_spk = Counter(s["spk"] for s in segs if s.get("spk"))
        line = f"**Длительность:** {dur:.0f} мин · **Слов:** {words} · **Сегментов:** {len(segs)}"
        if by_spk:
            share = " · ".join(f"{k}: {round(v * 100 / sum(by_spk.values()))}%" for k, v in by_spk.most_common())
            line += f"\n\n**Распределение реплик:** {share}"
        return line

    def _extractive(self, segs: list[dict]) -> str:
        """Фолбэк без LLM: частотные термины + предложения, их содержащие."""
        text = " ".join(s.get("text") or "" for s in segs)
        words = re.findall(r"\b[\w-]{5,}\b", text.lower())
        stop = {"который", "поэтому", "который", "потому", "например", "сейчас", "может",
                "нужно", "будет", "こちら", "should", "because", "example"}
        top = [w for w, _ in Counter(words).most_common(40) if w not in stop][:10]
        picked, seen = [], set()
        for term in top[:6]:
            for s in segs:
                t = s.get("text") or ""
                if term in t.lower() and s["sid"] not in seen and len(t.split()) > 5:
                    picked.append(f"- {t}")
                    seen.add(s["sid"])
                    break
        return ("## Ключевые термины\n" + ", ".join(top) +
                "\n\n## Показательные фрагменты\n" + "\n".join(picked) +
                "\n\n_LLM недоступна: саммари собрано извлечением фрагментов._")
