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

INPUT_FILE = "train.jsonl"
OUTPUT_FILE = "sft_train_dataset_2.jsonl"
FAILED_FILE = "rejected_qa_train_2.jsonl"

MAX_INPUT_TOKENS = 4096
SAVE_EVERY = 100

SYSTEM_PROMPT = "You are a medical expert assistant."

GEN_CONFIG = {
    "max_new_tokens": 4096,
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
    max_memory={
    0: "0MiB",
    1: "70000MiB",  
    2: "70000MiB",  
    3: "0MiB",  
    },
    
)
# max_memory={
    # 0: "0MiB",
    # 1: "58000MiB",  # 60559 - 2000
    # 2: "62000MiB",  # 64613 - 2000
    # 3: "60000MiB",  # 62603 - 2000
    # },

model.eval()

GEN_CONFIG["pad_token_id"] = tokenizer.pad_token_id
GEN_CONFIG["eos_token_id"] = tokenizer.eos_token_id


# =====================================
# Prompt
# =====================================


GEN_PROMPT_TEMPLATE = """
Generate exactly {n} question-answer pairs from the document.
Output ONLY a valid JSON array. No explanations, no reasoning, no analysis, no extra text.
Start your response with '[' and end with ']'.
All questions and answers must be written in Korean.

Document:
{document}
""".strip()

# GEN_PROMPT_TEMPLATE = """
# Generate exactly {n} question-answer pairs from the document.

# Rules:
# - Use only the document
# - Do not use metadata
# - Use Korean
# - Return only valid JSON
# - No markdown

# Document:
# {document}

# Output format:
# [
#   {{"question":"...", "answer":"..."}}
# ]
# """.strip()

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

# def safe_parse_json_array(text: str):
#     # findall로 모든 [...] 찾고 마지막 것을 사용
#     matches = list(re.finditer(r'\[.*?\]', text, re.DOTALL))
#     if not matches:
#         raise ValueError(f"JSON array not found. raw='{text[:500]}'")
    
#     candidate = matches[-1].group(0)  # 마지막 [...] 선택
    
#     try:
#         return json.loads(candidate)
#     except json.JSONDecodeError as e:
#         raise ValueError(f"JSON parse failed: {e}\ncandidate='{candidate[:500]}'")

#  수정: find("[") + rfind("]") 방식으로 변경
# def safe_parse_json_array(text: str):
#     start = text.find("[")
#     if start == -1:
#         raise ValueError(f"JSON array not found. raw='{text[:500]}'")
    
#     candidate = text[start:]
#     end = candidate.rfind("]") + 1
#     if end == 0:
#         raise ValueError(f"JSON array not found. raw='{text[:500]}'")
    
#     candidate = candidate[:end]
    
#     try:
#         return json.loads(candidate)
#     except json.JSONDecodeError as e:
#         raise ValueError(f"JSON parse failed: {e}\ncandidate='{candidate[:500]}'")

def safe_parse_json_array(text: str):
    # 모델이 항상 assistantfinal 이후에 실제 JSON을 출력하는 패턴 처리
    marker = "assistantfinal"
    if marker in text:
        text = text[text.index(marker) + len(marker):]
    
    start = text.find("[")
    if start == -1:
        raise ValueError(f"JSON array not found. raw='{text[:500]}'")
    
    candidate = text[start:]
    end = candidate.rfind("]") + 1
    if end == 0:
        raise ValueError(f"JSON array not found. raw='{text[:500]}'")
    
    candidate = candidate[:end]
    
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

    return safe_parse_json_array(text)  # 2번 수정: 디버깅 print 제거


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

    print("model_max_length: ", tokenizer.model_max_length)
    print("max_position_embeddings: ", model.config.max_position_embeddings)
    
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

        raw_output = None  # 3번 수정: raw_output 추적용

        try:

            prompt = GEN_PROMPT_TEMPLATE.format(document=document, n=qa_count)
            messages = [
                {"role": "system", "content": "You are a medical QA generator. Output only valid JSON array. No reasoning, no analysis, no explanation, no draft."},
                {"role": "user", "content": prompt},
            ]
            raw_output = run_generation(messages)
            qa_pairs = safe_parse_json_array(raw_output)

        # json관련 에러 발생시 디버깅용으로 stop
        except Exception as e:
            debug_info = {
                "input_prompt": prompt,          # 입력으로 들어간 전체 프롬프트
                "input_document": document,      # 원본 문서 전체
                "qa_count": qa_count,            # 요청한 QA 개수
                "input_tokens": count_tokens(document),   # 입력 토큰 수
                "sentence_count": count_sentences(document),  # 문장 수
                "raw_output": raw_output,        # 모델 출력 전체 (잘리지 않게)
                "reason": str(e),                # 실패 이유
            }

            with open("debug_first_rejected_2.json", "w", encoding="utf-8") as f:
                json.dump(debug_info, f, ensure_ascii=False, indent=2)

            print("\n[STOP] 첫 번째 rejected 발생, debug_first_rejected.json 저장 후 중단")
            print(f"reason: {e}")
            break


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