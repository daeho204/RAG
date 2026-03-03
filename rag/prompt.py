# rag/prompting.py
from __future__ import annotations

from typing import List
from rag.retriever import RetrievedChunk


def build_system_prompt(chunks: List[RetrievedChunk]) -> str:
    # chunk를 짧게 붙이고, 출처 표시
    lines = [
        "You are a helpful assistant for EMC/standard documents.",
        "Use ONLY the provided context to answer. If context is insufficient, say you don't know.",
        "Cite sources using [source_file:page] format.",
        "",
        "CONTEXT:",
    ]
    for i, c in enumerate(chunks, start=1):
        src = c.payload.get("source_file", "UNKNOWN")
        page = c.payload.get("page", None)
        page_str = "?" if page is None else str(page)
        snippet = c.text.strip()
        lines.append(f"\n---\n#{i} [{src}:{page_str}] (score={c.score:.4f})\n{snippet}")
    return "\n".join(lines)