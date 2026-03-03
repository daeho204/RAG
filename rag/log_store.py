# rag/log_store.py
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ChatLogRecord:
    ts: str
    model: str
    user_text: str
    system_text: Optional[str]
    answer_text: str
    usage: Optional[Dict[str, Any]]
    retrieval: Optional[Dict[str, Any]]  # diagnostics + top chunks summary


def _jsonable(obj: Any) -> Any:
    """Convert unknown objects to JSON-serializable form."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    # Pydantic / OpenAI objects may have model_dump()
    if hasattr(obj, "model_dump"):
        return _jsonable(obj.model_dump())
    # dataclass-like
    if hasattr(obj, "__dict__"):
        return _jsonable(vars(obj))
    return str(obj)


class JsonlChatStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: ChatLogRecord) -> None:
        d = asdict(record)
        d = _jsonable(d)
        line = json.dumps(d, ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    @staticmethod
    def now_iso() -> str:
        return datetime.now().isoformat(timespec="seconds")