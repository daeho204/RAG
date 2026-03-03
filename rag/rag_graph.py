# rag/graph.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from rag.config import Settings
from rag.retriever import QdrantRetriever, RetrievedChunk
from rag.retrieval_eval import diagnostics, RetrievalDiagnostics
from rag.prompting import build_system_prompt
from rag.llm_client import VllmChatClient, VllmConfig
from rag.log_store import JsonlChatStore, ChatLogRecord


class RAGState(TypedDict, total=False):
    user_query: str
    retrieved: List[RetrievedChunk]
    diag: Dict[str, Any]
    answer: str
    usage: Dict[str, Any]


def make_graph(settings: Settings, store: Optional[JsonlChatStore] = None):
    retriever = QdrantRetriever(settings)
    llm = VllmChatClient(
        VllmConfig(
            base_url=settings.vllm_base_url,
            api_key=settings.vllm_api_key,
            model=settings.vllm_model,
            timeout=settings.vllm_timeout,
        )
    )

    def node_retrieve(state: RAGState) -> RAGState:
        q = state["user_query"]
        chunks = retriever.retrieve(q, top_k=settings.top_k)
        return {"retrieved": chunks}

    def node_eval(state: RAGState) -> RAGState:
        chunks = state.get("retrieved", [])
        d = diagnostics(chunks)

        # Keep it JSON-friendly
        diag = {
            "top_k": d.top_k,
            "score_min": d.score_min,
            "score_max": d.score_max,
            "score_mean": d.score_mean,
            "by_source_file": d.by_source_file,
            "top_chunks": [
                {
                    "id": c.id,
                    "score": c.score,
                    "source_file": c.payload.get("source_file"),
                    "page": c.payload.get("page"),
                    "chunk_index": c.payload.get("chunk_index"),
                }
                for c in chunks
            ],
        }
        return {"diag": diag}

    def node_generate(state: RAGState) -> RAGState:
        q = state["user_query"]
        chunks = state.get("retrieved", [])
        system_prompt = build_system_prompt(chunks)

        answer, usage = llm.chat(user_text=q, system_text=system_prompt, temperature=0.2)
        out: RAGState = {"answer": answer}
        if usage:
            out["usage"] = usage
        return out

    def node_log(state: RAGState) -> RAGState:
        if store is None:
            return {}

        rec = ChatLogRecord(
            ts=store.now_iso(),
            model=settings.vllm_model,
            user_text=state.get("user_query", ""),
            system_text=None,  # system prompt is big; keep optional
            answer_text=state.get("answer", ""),
            usage=state.get("usage"),
            retrieval=state.get("diag"),
        )
        store.append(rec)
        return {}

    g = StateGraph(RAGState)
    g.add_node("retrieve", node_retrieve)
    g.add_node("eval", node_eval)
    g.add_node("generate", node_generate)
    g.add_node("log", node_log)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "eval")
    g.add_edge("eval", "generate")
    g.add_edge("generate", "log")
    g.add_edge("log", END)

    return g.compile()