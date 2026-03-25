# compare_models_param_knowledge.py
# -*- coding: utf-8 -*-

import json
import os
import time
from datetime import datetime
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ==============================
# 🔧 CONFIG (edit only here)
# ==============================

BASE_MODEL_ID = "google/gemma-3-27b-it"
# ADAPTER_DIR = "/dataset/finetune/outputs/gemma3_27b_medical_AdaptLLM_260305(remakeTest2)"
ADAPTER_DIR = "/home/gpuuser/daeho/emc_translation_adapter_epoch3"

OUT_JSONL = "out_details_remake_4.jsonl"

DTYPE = "bf16"  # "bf16" or "fp16"

MAX_NEW_TOKENS = 128
TEMPERATURE = 0.0
TOP_P = 1.0

SYSTEM_PROMPT = (
    # "You are a careful assistant. "
    # "If you are not sure or do not know, say '모르겠습니다'. "
    # "Do not invent facts or guess."
    "다음 문장을 한국어로 번역해줘"
)

# QUERIES = [
#     "피부 절개 전 항생제 투여 권고 시간은 몇 분 이내로 명시되어 있는가?",
#     "피부 절개 전 항생제의 60분 이내 투여가 60~120분 사이 투여보다 우월하다는 근거가 충분한가?",
#     "반감기가 짧은 항생제의 경우, 투여 시점은 어떻게 권고되는가?",
#     "수술실에서 항생제를 투여할 수 있는 경우, 적절한 투여 시점은 언제인가?",
#     "120분 이내 항생제 투여의 효과에 대한 근거는 무엇이라고 설명되어 있는가?",
#     "H. pylori 감염군과 비감염군 사이의 혈청 가스트린 수치에는 통계적으로 유의한 차이가 있는가?",
#     "혈청 가스트린 수치 상승과 가장 직접적으로 관련된 요인은 무엇으로 기술되어 있는가?",
#     "H. pylori 감염 여부는 어떤 검사 방법으로 판정되었는가?",
#     "H. pylori 감염과 고가스트린혈증 간에는 직접적인 상관관계가 있는가?",
#     "H. pylori 감염 환자의 혈청 가스트린 수치는 비감염 환자에 비해 몇 퍼센트 증가한다고 보고되었는가?",
    
# ]
QUERIES = [
#    "The EMI receiver measured the disturbance voltage using a quasi-peak value detector, which weights signals according to their repetition rate as specified in CISPR standards.​",
#    "absorber-lined OATS/SAC: OATS or SAC with ground plane partially covered by RF-energy absorbing material.",
#    "for the average detector, the effective time to average the signal envelope.",
#    "for pure continuous broadband disturbances, e.g. from ignition motors, arc welding equipment, and collector motors, a stepped scan (with peak or even quasi-peak detection) for sampling of the emission spectrum may be used. In this case the knowledge of the type of disturbance is used to draw a polyline (piecewise) curve as the spectrum envelope (see Figure 3). The step size shall be chosen so that no significant variations in the spectrum envelope are missed. A single swept measurement, if performed slowly enough, will also yield the spectrum envelope",
#    "The record shall also include an indication upon which conductor of the mains port carried the observed disturbance(s)",
#    "In order to simulate the influence of the user’s hand, application of the artificial hand is required for hand-held equipment during the mains disturbance voltage measurement.",
#    "The artificial hand consists of metal foil which is connected to one terminal (terminal M) of an RC element consisting of a capacitor of 220 pF ± 20 % in series with a resistance of 510 Ω ± 10 % (see Figure 6); the other terminal of the RC element shall be connected to the reference ground of the measuring system (see CISPR 16-1-2). The RC element of the artificial hand may be incorporated in the housing of the artificial mains network.",
#    "The artificial network is required to provide a defined impedance at radio frequencies across the mains supply at the point of measurement and also to provide for isolation of the equipment under test from ambient noise on the power lines.",
]

def torch_dtype():
    return torch.bfloat16 if DTYPE == "bf16" else torch.float16


def load_tokenizer():
    # Prefer adapter dir tokenizer if exists; else base
    tok_src = ADAPTER_DIR if os.path.isdir(ADAPTER_DIR) else BASE_MODEL_ID
    tok = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_only_model():
    """Independent BASE-only model instance."""
    m = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch_dtype(),
        device_map="auto",
    )
    m.eval()
    return m


def load_base_plus_peft_model():
    """Independent model instance: base (fresh) + adapter attached."""
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch_dtype(),
        device_map="auto",
    )
    base.eval()
    peft = PeftModel.from_pretrained(base, ADAPTER_DIR)
    peft.eval()
    return peft


# def build_prompt(question: str) -> str:
#     """
#     Keep it plain to ensure identical prompt for both models.
#     (No extra context. Only a fixed system rule + question.)
#     """
#     return (
#         f"SYSTEM:\n{SYSTEM_PROMPT}\n\n"
#         f"USER:\n{question}\n\n"
#         f"ASSISTANT:"
#     )

def build_messages(question: str) -> List[Dict[str, str]]:
    messages = []
    if SYSTEM_PROMPT.strip():
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": question})
    return messages

def postprocess_output(text: str) -> str:
    """
    Cut off obvious role-continuation artifacts if they appear.
    This is only a safety net. The main fix is using apply_chat_template().
    """
    stop_markers = [
        "\nmodel",
        "\nuser",
        "\nassistant",
        "\nSYSTEM:",
        "\nUSER:",
        "\nASSISTANT:",
        "<start_of_turn>user",
        "<start_of_turn>model",
        "<start_of_turn>assistant",
    ]

    cut_pos = len(text)
    lower_text = text.lower()

    for marker in stop_markers:
        idx = lower_text.find(marker.lower())
        if idx != -1:
            cut_pos = min(cut_pos, idx)

    return text[:cut_pos].strip()

@torch.no_grad()
def generate(tokenizer, model, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Generate response from chat-template formatted messages.
    Returns both prompt text and generated output text for logging.
    """
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    gen_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": (TEMPERATURE > 0),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if TEMPERATURE > 0:
        gen_kwargs["temperature"] = TEMPERATURE
        gen_kwargs["top_p"] = TOP_P

    out = model.generate(**inputs, **gen_kwargs)

    # Only decode newly generated tokens
    new_tokens = out[0][input_len:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    decoded = postprocess_output(decoded)

    return {
        "prompt_text": prompt_text,
        "output_text": decoded,
    }


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def run_batch(queries: List[str]) -> None:
    print("Loading tokenizer...")
    tokenizer = load_tokenizer()

    print("Loading BASE-only model (independent)...")
    base_model = load_base_only_model()

    print("Loading BASE+PEFT model (independent)...")
    peft_model = load_base_plus_peft_model()

    # PEFT 설정 확인
    # print("\n[PEFT CONFIG]")
    # print(peft_model.peft_config)
    # print("\n=== LoRA MODULE CHECK ===")

    # count = 0
    # for name, module in peft_model.named_modules():
    #     if "lora" in name.lower():
    #         print(name)
    #         count += 1

    # print(f"\n[LoRA] total lora modules: {count}")

    # print("\n=== TRAINABLE PARAMS (LoRA) ===")
    # for name, param in peft_model.named_parameters():
    #     if "lora" in name.lower():
    #         print(name, param.shape)
    
    # print("\n=== MODEL SUMMARY ===")
    # peft_model.print_trainable_parameters()

    print("\nStart batch inference (NO CONTEXT, CHAT TEMPLATE).\n")

    for idx, q in enumerate(queries):
        messages = build_messages(q)

        t0 = time.time()
        base_ret = generate(tokenizer, base_model, messages)
        t1 = time.time()
        peft_ret = generate(tokenizer, peft_model, messages)
        t2 = time.time()

        base_out = base_ret["output_text"]
        peft_out = peft_ret["output_text"]
        prompt_used = base_ret["prompt_text"]  # same messages for both models

        print(f"[{idx}] Q: {q}")
        print("\n--- PROMPT USED ---")
        print(prompt_used)
        print("\n--- BASE OUTPUT ---")
        print(base_out)
        print("\n--- PEFT OUTPUT ---")
        print(peft_out)
        print("\n" + "=" * 100 + "\n")

        record = {
            "idx": idx,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "base_model": BASE_MODEL_ID,
            "adapter_dir": ADAPTER_DIR,
            "gen": {
                "dtype": DTYPE,
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
            },
            "system_prompt": SYSTEM_PROMPT,
            "question": q,
            "messages_used": messages,
            "prompt_used": prompt_used,
            "base_output": base_out,
            "peft_output": peft_out,
            "latency_sec": {
                "base": round(t1 - t0, 3),
                "peft": round(t2 - t1, 3),
            },
        }
        append_jsonl(OUT_JSONL, record)

    print(f"✅ Done. Saved to: {OUT_JSONL}")


def main():
    run_batch(QUERIES)


if __name__ == "__main__":
    main()