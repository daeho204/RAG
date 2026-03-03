# rag/app.py
from __future__ import annotations

import argparse
import json

from rag.config import Settings
from rag.graph import make_graph
from rag.log_store import JsonlChatStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="User query")
    ap.add_argument("--log", default="logs/chat_history.jsonl", help="jsonl log path")
    args = ap.parse_args()

    s = Settings()
    store = JsonlChatStore(args.log)

    app = make_graph(s, store=store)
    out = app.invoke({"user_query": args.query})

    # pretty print diagnostics
    diag = out.get("diag", {})
    print("\n=== Retrieval Diagnostics ===")
    print(json.dumps(diag, ensure_ascii=False, indent=2))

    print("\n=== Answer ===")
    print(out.get("answer", ""))


if __name__ == "__main__":
    main()