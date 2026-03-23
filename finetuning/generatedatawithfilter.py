# generate_and_verify_qa_final.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================================================
# Config
# =========================================================
MODEL_ID = "openai/gpt-oss-20b"

INPUT_FILE = "rawdata/test.jsonl"              # JSON or JSONL
OUTPUT_FILE = "dataset/sft_dataset.jsonl"
FAILED_FILE = "dataset/rejected_qa.jsonl"

QA_PER_CHUNK = 3
MAX_INPUT_TOKENS = 1400               # truncate document for stable generation
SAVE_EVERY = 100                      # flush every N accepted/rejected additions

SYSTEM_PROMPT = "You are a medical expert assistant."

GEN_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.3,
    "top_p": 0.9,
    "do_sample": True,
}

VERIFY_CONFIG = {
    "max_new_tokens": 160,
    "temperature": 0.0,
    "do_sample": False,
}


# =========================================================
# Load model/tokenizer
# =========================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.eval()

GEN_CONFIG["pad_token_id"] = tokenizer.pad_token_id
GEN_CONFIG["eos_token_id"] = tokenizer.eos_token_id
VERIFY_CONFIG["pad_token_id"] = tokenizer.pad_token_id
VERIFY_CONFIG["eos_token_id"] = tokenizer.eos_token_id


# =========================================================
# Prompt templates
# =========================================================
GEN_PROMPT_TEMPLATE = """
Generate exactly {n} high-quality question-answer pairs from the document.

Rules:
- Use only the content of the document.
- Do not use metadata.
- Do not ask about file name, source, or document id.
- Make the questions diverse:
  1) one factual question
  2) one comparison or reasoning question
  3) one recommendation or guideline interpretation question
- Avoid yes/no questions.
- Avoid trivial wording copied directly from headings.
- Answers must be fully supported by the document.
- Return only valid JSON.
- Do not include markdown fences.
- Use Korean for both questions and answers.

Document:
{document}

Output format:
[
  {{"question":"...", "answer":"..."}},
  {{"question":"...", "answer":"..."}},
  {{"question":"...", "answer":"..."}}
]
""".strip()


VERIFY_PROMPT_TEMPLATE = """
Determine whether the answer is fully supported by the document and whether the question is answerable from the document.

Rules:
- Be strict.
- If the answer includes unsupported information, set supported=false.
- If the question cannot be answered from the document alone, set answerable=false.
- quality_score must be an integer from 1 to 5.
- Return only valid JSON.
- Do not include markdown fences.

Document:
{document}

Question:
{question}

Answer:
{answer}

Return this exact schema:
{{
  "supported": true,
  "answerable": true,
  "quality_score": 4,
  "reason": "short explanation"
}}
""".strip()


# =========================================================
# Utility
# =========================================================
def load_rows(path: str) -> List[Dict[str, Any]]:
    """
    Supports both:
    1) JSON array file
    2) JSONL file
    """
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)

        if first == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON array file must contain a list.")
            return data

        rows = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows


def append_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def truncate_document(text: str, max_tokens: int = MAX_INPUT_TOKENS) -> str:
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(token_ids) <= max_tokens:
        return text
    token_ids = token_ids[:max_tokens]
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def safe_parse_json_array(text: str) -> List[Dict[str, Any]]:
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError("JSON array not found in generated text.")
    parsed = json.loads(text[start:end])
    if not isinstance(parsed, list):
        raise ValueError("Parsed JSON is not a list.")
    return parsed


def safe_parse_json_object(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("JSON object not found in generated text.")
    parsed = json.loads(text[start:end])
    if not isinstance(parsed, dict):
        raise ValueError("Parsed JSON is not an object.")
    return parsed


# =========================================================
# Generation
# =========================================================
def run_generation(messages: List[Dict[str, str]], config: Dict[str, Any]) -> str:
    """
    Use chat template for chat/instruct models.
    Slice generated tokens by token length, not by decoded string length.
    """
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **config,
        )

    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return generated_text


def generate_qa_pairs(document: str, n: int) -> List[Dict[str, str]]:
    prompt = GEN_PROMPT_TEMPLATE.format(document=document, n=n)
    messages = [
        {"role": "system", "content": "You generate grounded medical QA datasets."},
        {"role": "user", "content": prompt},
    ]

    text = run_generation(messages, GEN_CONFIG)
    data = safe_parse_json_array(text)

    clean_pairs: List[Dict[str, str]] = []

    for item in data:
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()

        if not question or not answer:
            continue

        clean_pairs.append({
            "question": question,
            "answer": answer,
        })

    return clean_pairs


def verify_qa(document: str, question: str, answer: str) -> Dict[str, Any]:
    prompt = VERIFY_PROMPT_TEMPLATE.format(
        document=document,
        question=question,
        answer=answer,
    )
    messages = [
        {"role": "system", "content": "You are a strict medical QA verifier."},
        {"role": "user", "content": prompt},
    ]

    text = run_generation(messages, VERIFY_CONFIG)
    result = safe_parse_json_object(text)

    return {
        "supported": bool(result.get("supported", False)),
        "answerable": bool(result.get("answerable", False)),
        "quality_score": int(result.get("quality_score", 0)),
        "reason": str(result.get("reason", "")).strip(),
    }


def build_sft_sample(question: str, answer: str) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


# =========================================================
# Rule-based filters
# =========================================================
def apply_rule_filters(question: str, answer: str) -> tuple[bool, str, str, str]:
    """
    Returns:
        passed, normalized_question, normalized_answer, reject_reason
    """
    q = question.strip()
    a = answer.strip()

    if not q or not a:
        return False, q, a, "empty_field"

    if not q.endswith("?"):
        q = q + "?"

    # character-length quick filter
    if len(q) < 8 or len(a) < 5:
        return False, q, a, "too_short"

    # token/word-length filter
    if len(q.split()) < 3:
        return False, q, a, "question_too_short_by_words"

    if len(a.split()) < 2:
        return False, q, a, "answer_too_short_by_words"

    # meaningless self-repeat
    if normalize_text(q).replace("?", "") in normalize_text(a):
        return False, q, a, "question_subsumed_by_answer"

    return True, q, a, ""


# =========================================================
# Main
# =========================================================
def main() -> None:
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)
    failed_path = Path(FAILED_FILE)

    # clean previous outputs for fresh run
    if output_path.exists():
        output_path.unlink()
    if failed_path.exists():
        failed_path.unlink()

    rows = load_rows(INPUT_FILE)

    accepted_buffer: List[Dict[str, Any]] = []
    rejected_buffer: List[Dict[str, Any]] = []
    seen_pairs = set()

    accepted_count = 0
    rejected_count = 0

    for row_idx, row in enumerate(tqdm(rows, desc="Processing chunks"), start=1):
        document = row.get("text")
        if not document:
            rejected_buffer.append({
                "row_index": row_idx,
                "reason": "missing_text",
                "row_preview": str(row)[:300],
            })
            rejected_count += 1
            continue

        document = truncate_document(document, MAX_INPUT_TOKENS)

        try:
            qa_pairs = generate_qa_pairs(document, QA_PER_CHUNK)
        except Exception as e:
            rejected_buffer.append({
                "row_index": row_idx,
                "document": document[:500],
                "reason": f"generation_failed: {str(e)}",
            })
            rejected_count += 1
            continue

        for qa in qa_pairs:
            q_raw = qa["question"]
            a_raw = qa["answer"]

            passed, q, a, reject_reason = apply_rule_filters(q_raw, a_raw)

            if not passed:
                rejected_buffer.append({
                    "row_index": row_idx,
                    "document": document[:500],
                    "question": q_raw,
                    "answer": a_raw,
                    "reason": reject_reason,
                })
                rejected_count += 1
                continue

            # dedup after normalization
            pair_key = (normalize_text(q), normalize_text(a))
            if pair_key in seen_pairs:
                rejected_buffer.append({
                    "row_index": row_idx,
                    "document": document[:300],
                    "question": q,
                    "answer": a,
                    "reason": "duplicate_pair",
                })
                rejected_count += 1
                continue
            seen_pairs.add(pair_key)

            try:
                verdict = verify_qa(document, q, a)
            except Exception as e:
                rejected_buffer.append({
                    "row_index": row_idx,
                    "document": document[:500],
                    "question": q,
                    "answer": a,
                    "reason": f"verification_failed: {str(e)}",
                })
                rejected_count += 1
                continue

            if (
                verdict["supported"] is True
                and verdict["answerable"] is True
                and verdict["quality_score"] >= 3
            ):
                accepted_buffer.append(build_sft_sample(q, a))
                accepted_count += 1
            else:
                rejected_buffer.append({
                    "row_index": row_idx,
                    "document": document[:500],
                    "question": q,
                    "answer": a,
                    "reason": verdict,
                })
                rejected_count += 1

            # periodic flush
            if len(accepted_buffer) >= SAVE_EVERY:
                append_jsonl(OUTPUT_FILE, accepted_buffer)
                accepted_buffer.clear()

            if len(rejected_buffer) >= SAVE_EVERY:
                append_jsonl(FAILED_FILE, rejected_buffer)
                rejected_buffer.clear()

    # final flush
    append_jsonl(OUTPUT_FILE, accepted_buffer)
    append_jsonl(FAILED_FILE, rejected_buffer)

    print(f"accepted: {accepted_count}")
    print(f"rejected: {rejected_count}")
    print(f"saved accepted to: {OUTPUT_FILE}")
    print(f"saved rejected to: {FAILED_FILE}")


if __name__ == "__main__":
    main()