# rag/llm_client.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

from openai import OpenAI


@dataclass(frozen=True)
class VllmConfig:
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    model: str = "google/gemma-3-1b-it"
    timeout: float = 120.0


class VllmChatClient:
    def __init__(self, cfg: VllmConfig) -> None:
        self.cfg = cfg
        self.client = OpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            timeout=cfg.timeout,
        )

    def chat(
        self,
        user_text: str,
        system_text: str | None = None,
        temperature: float = 0.2,
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        messages = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": user_text})

        resp = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=messages,
            temperature=temperature,
        )

        answer = resp.choices[0].message.content or ""

        # usage can be CompletionUsage object -> convert to dict safely
        usage_obj = getattr(resp, "usage", None)
        usage: Optional[Dict[str, Any]] = None
        if usage_obj is not None:
            if hasattr(usage_obj, "model_dump"):
                usage = usage_obj.model_dump()
            else:
                # fallback
                usage = {
                    "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
                    "completion_tokens": getattr(usage_obj, "completion_tokens", None),
                    "total_tokens": getattr(usage_obj, "total_tokens", None),
                }

        return answer, usage