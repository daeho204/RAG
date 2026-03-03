# rag/config.py
from __future__ import annotations

from dataclasses import dataclass
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

def _get_env(name: str, default: str | None = None) -> str:
    v = os.getenv(name, default)
    if v is None or v == "":
        raise RuntimeError(f"Missing env: {name}")
    return v


@dataclass(frozen=True)
class Settings:
    # vLLM
    vllm_base_url: str = _get_env("VLLM_BASE_URL", "http://localhost:8000/v1")
    vllm_api_key: str = _get_env("VLLM_API_KEY", "EMPTY")
    vllm_model: str = _get_env("VLLM_MODEL", "google/gemma-3-1b-it")

    # Qdrant
    qdrant_url: str = _get_env("QDRANT_URL", "http://localhost:6333")
    qdrant_collection: str = _get_env("QDRANT_COLLECTION", "rag_pdfs")

    # Embeddings
    embedding_model: str = _get_env("EMBEDDING_MODEL", "google/embeddinggemma-300m")

    # Chunking
    chunk_size: int = int(_get_env("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(_get_env("CHUNK_OVERLAP", "150"))
    top_k: int = int(_get_env("TOP_K", "5"))