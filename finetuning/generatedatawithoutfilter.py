# generate_qa_fast.py

import json
import re
from typing import List, Dict, Any
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =====================================
# Config
# =====================================

MODEL_ID = "openai/gpt-oss-20b"

INPUT_FILE = "rawdata/test.jsonl"
OUTPUT_FILE = "dataset/sft_dataset_3.jsonl"
FAILED_FILE = "dataset/rejected_qa_3.jsonl"

# Token수가 너무 낮으면 raw text데이터가 너무 길면 jsonl을 array로 변경할 때 ']' 가 짤려서 에러남
MAX_INPUT_TOKENS = 2048
SAVE_EVERY = 100

SYSTEM_PROMPT = "You are a medical expert assistant."

GEN_CONFIG = {
    "max_new_tokens": 2048,
    "temperature": 0.3,
    "top_p": 0.9,
    "do_sample": True,
}


# =====================================
# Load model
# =====================================

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


# =====================================
# Prompt
# =====================================

GEN_PROMPT_TEMPLATE = """
Generate exactly {n} question-answer pairs from the document.

Rules:
- Use only the document
- Do not use metadata
- Use Korean
- Return only valid JSON
- No markdown

Document:
{document}

Output format:
[
  {{"question":"...", "answer":"..."}}
]
""".strip()


# =====================================
# Utils
# =====================================

def append_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_text(text: str):
    return " ".join(text.strip().lower().split())


def truncate_document(text: str):

    ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    if len(ids) <= MAX_INPUT_TOKENS:
        return text

    ids = ids[:MAX_INPUT_TOKENS]

    return tokenizer.decode(ids, skip_special_tokens=True)


def safe_parse_json_array(text: str):

    start = text.find("[")
    end = text.rfind("]") + 1

    if start == -1 or end == 0:
        raise ValueError(f"JSON array not found. start={start}, end={end}, raw='{text[:500]}'")

    candidate = text[start:end]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse failed: {e}\ncandidate='{candidate[:500]}'")


def count_sentences(text: str):

    parts = re.split(r"[.!?]\s+|\n+", text)

    parts = [p.strip() for p in parts if p.strip()]

    return max(1, len(parts))


def count_tokens(text: str):

    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def decide_qa_count(text: str):

    s = count_sentences(text)
    t = count_tokens(text)

    if s <= 3 or t <= 120:
        return 1

    elif s <= 7 or t <= 300:
        return 2

    else:
        return 3


# =====================================
# Generation
# =====================================

def run_generation(messages):

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
            **GEN_CONFIG
        )

    input_len = inputs["input_ids"].shape[1]

    generated_ids = outputs[0][input_len:]

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def generate_qa(document, n):

    prompt = GEN_PROMPT_TEMPLATE.format(
        document=document,
        n=n
    )

    messages = [
        {"role": "system", "content": "You generate medical QA datasets."},
        {"role": "user", "content": prompt},
    ]

    text = run_generation(messages)

    # 디버깅: 모델 raw 출력 확인
    print("=" * 60)
    print("[RAW OUTPUT]")
    print(repr(text))  # repr로 찍으면 \n 등 특수문자도 보임
    print("=" * 60)

    return safe_parse_json_array(text)


def build_sft_sample(q, a):

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
    }


# =====================================
# Main
# =====================================

def main():

    seen_pairs = set()

    accepted_buffer = []
    rejected_buffer = []

    accepted_count = 0
    rejected_count = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:

        rows = [json.loads(line) for line in f if line.strip()]

    for row in tqdm(rows):

        document = row.get("text")

        if not document:
            continue

        document = truncate_document(document)

        qa_count = decide_qa_count(document)

        try:

            qa_pairs = generate_qa(document, qa_count)

        except Exception as e:

            rejected_buffer.append({
                "document": document[:300],
                "reason": f"generation_failed:{str(e)}"
            })

            rejected_count += 1
            continue

        for qa in qa_pairs:

            q = qa.get("question", "").strip()
            a = qa.get("answer", "").strip()

            if not q or not a:
                continue

            if not q.endswith("?"):
                q += "?"

            key = (normalize_text(q), normalize_text(a))

            if key in seen_pairs:

                rejected_buffer.append({
                    "question": q,
                    "answer": a,
                    "reason": "duplicate"
                })

                rejected_count += 1
                continue

            seen_pairs.add(key)

            accepted_buffer.append(build_sft_sample(q, a))

            accepted_count += 1

        if len(accepted_buffer) >= SAVE_EVERY:

            append_jsonl(OUTPUT_FILE, accepted_buffer)

            accepted_buffer.clear()

        if len(rejected_buffer) >= SAVE_EVERY:

            append_jsonl(FAILED_FILE, rejected_buffer)

            rejected_buffer.clear()

    append_jsonl(OUTPUT_FILE, accepted_buffer)
    append_jsonl(FAILED_FILE, rejected_buffer)

    print("accepted:", accepted_count)
    print("rejected:", rejected_count)


if __name__ == "__main__":
    main()