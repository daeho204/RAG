# rag/retrieval_eval.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import math
import json
from pathlib import Path

from rag.retriever import RetrievedChunk


@dataclass(frozen=True)
class RetrievalDiagnostics:
    top_k: int
    score_min: float
    score_max: float
    score_mean: float
    by_source_file: Dict[str, int]


def diagnostics(chunks: Sequence[RetrievedChunk]) -> RetrievalDiagnostics:
    if not chunks:
        return RetrievalDiagnostics(top_k=0, score_min=0.0, score_max=0.0, score_mean=0.0, by_source_file={})

    scores = [c.score for c in chunks]
    by_src: Dict[str, int] = {}
    for c in chunks:
        src = str(c.payload.get("source_file", "UNKNOWN"))
        by_src[src] = by_src.get(src, 0) + 1

    return RetrievalDiagnostics(
        top_k=len(chunks),
        score_min=min(scores),
        score_max=max(scores),
        score_mean=sum(scores) / len(scores),
        by_source_file=by_src,
    )


# ---------- Optional offline evaluation (qrels) ----------

@dataclass(frozen=True)
class Qrel:
    query: str
    relevant_ids: List[str]


@dataclass(frozen=True)
class OfflineMetrics:
    recall_at_k: float
    mrr: float
    map: float


def load_qrels(path: str = "data/qrels.jsonl") -> List[Qrel]:
    p = Path(path)
    if not p.exists():
        return []
    out: List[Qrel] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        out.append(Qrel(query=obj["query"], relevant_ids=[str(x) for x in obj["relevant_ids"]]))
    return out


def recall_at_k(retrieved_ids: List[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hit = 0
    for rid in retrieved_ids[:k]:
        if rid in relevant:
            hit += 1
    # standard recall: hits / |relevant|
    return hit / len(relevant)


def mrr_at_k(retrieved_ids: List[str], relevant: set[str], k: int) -> float:
    for rank, rid in enumerate(retrieved_ids[:k], start=1):
        if rid in relevant:
            return 1.0 / rank
    return 0.0


def ap_at_k(retrieved_ids: List[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hit = 0
    s = 0.0
    for i, rid in enumerate(retrieved_ids[:k], start=1):
        if rid in relevant:
            hit += 1
            s += hit / i
    return s / len(relevant)


def eval_offline(
    qrels: List[Qrel],
    retrieve_fn,  # callable(query)->List[RetrievedChunk]
    k: int = 5
) -> OfflineMetrics:
    if not qrels:
        return OfflineMetrics(recall_at_k=0.0, mrr=0.0, map=0.0)

    r_sum = 0.0
    mrr_sum = 0.0
    ap_sum = 0.0

    for q in qrels:
        chunks = retrieve_fn(q.query)
        retrieved_ids = [c.id for c in chunks]
        rel = set(q.relevant_ids)

        r_sum += recall_at_k(retrieved_ids, rel, k)
        mrr_sum += mrr_at_k(retrieved_ids, rel, k)
        ap_sum += ap_at_k(retrieved_ids, rel, k)

    n = len(qrels)
    return OfflineMetrics(
        recall_at_k=r_sum / n,
        mrr=mrr_sum / n,
        map=ap_sum / n,
    )