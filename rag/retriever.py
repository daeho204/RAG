# rag/retriever.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint

from rag.config import Settings


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    score: float
    text: str
    payload: Dict[str, Any]


class QdrantRetriever:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        self.emb = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        self.qc = QdrantClient(url=settings.qdrant_url)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        k = top_k or self.s.top_k
        qvec = self.emb.embed_query(query)

        res: List[ScoredPoint] = self.qc.search(
            collection_name=self.s.qdrant_collection,
            query_vector=qvec,
            limit=k,
            with_payload=True,
            with_vectors=False,
        )

        out: List[RetrievedChunk] = []
        for p in res:
            payload = p.payload or {}
            text = payload.get("text", "")
            out.append(
                RetrievedChunk(
                    id=str(p.id),
                    score=float(p.score),
                    text=text,
                    payload=dict(payload),
                )
            )
        return out