# rag/ingest.py
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any

from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from rag.config import Settings


def iter_pdfs(pdf_dir: Path) -> Iterable[Path]:
    for p in sorted(pdf_dir.glob("*.pdf")):
        if p.is_file():
            yield p


def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def make_chunk_id(source_file: str, page: int | None, chunk_index: int, text: str) -> str:
    # Stable ID: same PDF/page/chunk_index/text => same id
    # If chunking settings change, chunk_index/text will change => new ids created (expected).
    key = f"{source_file}|p={page}|c={chunk_index}|t={sha1_hex(text)}"
    return sha1_hex(key)


def ensure_collection(client: QdrantClient, collection: str, vector_size: int) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if collection in existing:
        # Optional: verify dimension matches
        info = client.get_collection(collection)
        current_size = info.config.params.vectors.size  # type: ignore[attr-defined]
        if current_size != vector_size:
            raise RuntimeError(
                f"Collection '{collection}' exists but vector size mismatch "
                f"(expected={vector_size}, got={current_size}). "
                f"Use a new collection name or recreate the collection."
            )
        return

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    payload: Dict[str, Any]


def load_and_chunk_pdfs(pdf_dir: Path, splitter: RecursiveCharacterTextSplitter) -> List[ChunkRecord]:
    pdf_files = list(iter_pdfs(pdf_dir))
    if not pdf_files:
        return []

    all_records: List[ChunkRecord] = []

    for pdf in tqdm(pdf_files, desc="Load PDFs"):
        loader = PyPDFLoader(str(pdf))
        pages = loader.load()  # one Document per page

        # Add stable metadata
        for d in pages:
            d.metadata["source_file"] = pdf.name

        chunks = splitter.split_documents(pages)

        # Create ChunkRecord with stable ids
        for idx, ch in enumerate(chunks):
            text = ch.page_content or ""
            source_file = ch.metadata.get("source_file", pdf.name)
            page = ch.metadata.get("page", None)

            cid = make_chunk_id(source_file=source_file, page=page, chunk_index=idx, text=text)

            payload = {
                "source_file": source_file,
                "page": page,
                "chunk_index": idx,
                # Keep useful metadata if exists
                **{k: v for k, v in ch.metadata.items() if k not in {"source_file", "page"}},
                # Optionally store text for debugging / trace
                "text": text,
            }

            all_records.append(ChunkRecord(chunk_id=cid, text=text, payload=payload))

    return all_records


def upsert_records(
    qc: QdrantClient,
    collection: str,
    emb: HuggingFaceEmbeddings,
    records: List[ChunkRecord],
    embed_batch: int = 64,
    upsert_batch: int = 128,
) -> None:
    # We embed texts in batches, then upsert points in batches.
    # This avoids huge memory spikes and is fast.
    n = len(records)
    if n == 0:
        return

    # Embed in batches
    vectors: List[List[float]] = []
    for i in tqdm(range(0, n, embed_batch), desc="Embedding"):
        batch_texts = [r.text for r in records[i:i + embed_batch]]
        batch_vecs = emb.embed_documents(batch_texts)
        vectors.extend(batch_vecs)

    # Upsert in batches
    for i in tqdm(range(0, n, upsert_batch), desc="Upserting"):
        pts: List[PointStruct] = []
        for r, v in zip(records[i:i + upsert_batch], vectors[i:i + upsert_batch]):
            pts.append(
                PointStruct(
                    id=r.chunk_id,     # stable id (string is allowed)
                    vector=v,
                    payload=r.payload,
                )
            )
        qc.upsert(collection_name=collection, points=pts)


def ingest_pdfs(pdf_dir: str = "data/pdfs") -> None:
    s = Settings()
    pdf_path = Path(pdf_dir)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF folder not found: {pdf_path.resolve()}")

    print("Loading embedding model...")
    emb = HuggingFaceEmbeddings(model_name=s.embedding_model)

    qc = QdrantClient(url=s.qdrant_url)

    # Determine embedding dimension once
    dummy = emb.embed_query("dimension probe")
    ensure_collection(qc, s.qdrant_collection, vector_size=len(dummy))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    records = load_and_chunk_pdfs(pdf_path, splitter)
    if not records:
        print("No PDF found or no text extracted.")
        return

    print(f"Total chunks: {len(records)}")

    upsert_records(
        qc=qc,
        collection=s.qdrant_collection,
        emb=emb,
        records=records,
        embed_batch=64,
        upsert_batch=128,
    )

    print(f"[OK] Upserted {len(records)} chunks into '{s.qdrant_collection}'")


if __name__ == "__main__":
    ingest_pdfs()