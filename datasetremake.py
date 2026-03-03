# convert_to_gemma_chat_jsonl.py
# -*- coding: utf-8 -*-

"""
Gemma chat format converter
No CLI args required. Just run:

    python convert_to_gemma_chat_jsonl.py
"""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# =========================
# CONFIG (여기만 수정하면 됨)
# =========================

INPUT_PATH = Path("dataset.jsonl")
OUTPUT_PATH = Path("dataset_gemma_chat.jsonl")

SYSTEM_PREFIX = (
    "You must answer strictly based on the excerpt. "
    "If the answer is not present in the excerpt, answer '없음'."
)


# =========================
# REGEX
# =========================

RE_SECTION_RULES = re.compile(r"^###\s*규칙\s*$", re.MULTILINE)
RE_SECTION_EXCERPT = re.compile(r"^###\s*발췌문\s*$", re.MULTILINE)
RE_SECTION_QA = re.compile(r"^###\s*문제와\s*정답\s*$", re.MULTILINE)

RE_QA_PAIR = re.compile(
    r"^Q(?P<qnum>\d+)\.\s*(?P<qtext>.*?)\n"
    r"^A(?P<anum>\d+)\.\s*(?P<atext>.*?)(?=\n^Q\d+\.|\Z)",
    re.MULTILINE | re.DOTALL,
)


# =========================
# PARSERS
# =========================

def split_sections(text: str):
    m_rules = RE_SECTION_RULES.search(text)
    m_excerpt = RE_SECTION_EXCERPT.search(text)
    m_qa = RE_SECTION_QA.search(text)

    if not (m_rules and m_excerpt and m_qa):
        return None, None, None, ""

    header = text[:m_rules.start()]
    rules = text[m_rules.end():m_excerpt.start()]
    excerpt = text[m_excerpt.end():m_qa.start()]
    qa_block = text[m_qa.end():]

    return rules.strip(), excerpt.strip(), qa_block.strip(), header.strip()


def parse_meta(header: str) -> Dict:
    meta = {}
    if header.startswith("[DOC="):
        header = header[1:-1]
        for part in header.split("|"):
            if "=" in part:
                k, v = part.split("=", 1)
                meta[k] = v
    return meta


def parse_qa(qa_block: str) -> List[Tuple[str, str]]:
    results = []
    for m in RE_QA_PAIR.finditer(qa_block):
        if m.group("qnum") == m.group("anum"):
            q = m.group("qtext").strip()
            a = m.group("atext").strip()
            results.append((q, a))
    return results


# =========================
# CONVERTER
# =========================

def convert():

    print(f"[INPUT] {INPUT_PATH}")
    print(f"[OUTPUT] {OUTPUT_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_in = 0
    total_out = 0

    with open(INPUT_PATH, encoding="utf-8") as fin, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as fout:

        for line in fin:

            total_in += 1

            obj = json.loads(line)
            text = obj["text"]

            rules, excerpt, qa_block, header = split_sections(text)

            if not excerpt or not qa_block:
                continue

            meta = parse_meta(header)
            qa_pairs = parse_qa(qa_block)

            system = f"{SYSTEM_PREFIX}\n\n{rules}" if rules else SYSTEM_PREFIX

            for idx, (q, a) in enumerate(qa_pairs):

                sample = {
                    "messages": [
                        {"role": "system", "content": system},
                        {
                            "role": "user",
                            "content": f"### 발췌문\n{excerpt}\n\n### 질문\n{q}"
                        },
                        {"role": "assistant", "content": a}
                    ],
                    "meta": {
                        **meta,
                        "qa_index": idx
                    }
                }

                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

                total_out += 1

    print(f"[DONE] in={total_in}, out={total_out}")


# =========================
# ENTRY
# =========================

if __name__ == "__main__":
    convert()