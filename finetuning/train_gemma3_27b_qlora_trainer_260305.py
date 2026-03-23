# train_gemma3_27b_qlora_promptlen_collator.py
# -*- coding: utf-8 -*-
"""
Gemma3 QLoRA training (completion-only loss) using prompt-length masking.

[CHANGES APPLIED]
1) TrainingArguments: add tf32=True (speed on Ampere/Ada/Hopper)
2) Tokenizer/Model: explicitly set pad_token_id (stability with padding/masking)
3) Gradient checkpointing: use_reentrant=False (often more stable with PEFT + 4bit + DDP)
4) LoRA target_modules: attention-only (more stable / lower VRAM)  [optional but recommended]
5) DataCollator: ALWAYS provide token_type_ids (Gemma3 requires it during training)
   - NOTE: We DO NOT rely on tokenizer(..., return_token_type_ids=...) because many tokenizers won't return it.
   - We create token_type_ids explicitly and set it to 0 everywhere.

실행 예:
accelerate launch --num_processes 4 train_gemma3_27b_qlora_promptlen_collator.py \
  --data_files dataset_gemma_chat.jsonl \
  --output_dir outputs/gemma3_27b_qlora_promptlen
"""

from __future__ import annotations

import os
import argparse
import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# ============================================================
# DDP / device
# ============================================================
def set_device_from_local_rank() -> int:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank


# ============================================================
# TrainingArguments (HF 버전 호환)
# ============================================================
def build_training_args(args):
    sig = inspect.signature(TrainingArguments.__init__)
    extra = {"eval_strategy": "no"} if "eval_strategy" in sig.parameters else {"evaluation_strategy": "no"}

    return TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,

        # NOTE: warmup_ratio is deprecated warning in your logs; kept as-is for compatibility.
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",

        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        bf16=True,
        fp16=False,

        # =========================
        # [CHANGED] Enable TF32 for faster matmul on Ampere/Ada/Hopper (A100/H100/RTX40xx).
        # =========================
        tf32=True,

        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        optim="paged_adamw_8bit",
        report_to="none",

        # IMPORTANT:
        # - For LoRA + gradient checkpointing + DDP, False is generally preferred for speed/stability.
        # - But you observed True makes it "run"; that means some params are not used in some forward paths.
        # - Keep True for now if needed to avoid crashes; later try False again after stabilizing.
        ddp_find_unused_parameters=True,

        # DataCollator uses raw text columns -> keep False
        remove_unused_columns=False,
        **extra,
    )


# ============================================================
# Dataset normalization
# ============================================================
def normalize_messages_to_text(tokenizer, msgs) -> str:
    """
    msgs: list[{"role":..., "content":...}] -> chat template 문자열로 변환
    """
    if not isinstance(msgs, list) or len(msgs) == 0:
        return ""

    cleaned = []
    for m in msgs:
        if isinstance(m, dict) and "role" in m and "content" in m:
            cleaned.append({"role": m["role"], "content": str(m["content"])})
    if not cleaned:
        return ""

    return tokenizer.apply_chat_template(
        cleaned,
        tokenize=False,
        add_generation_prompt=False,
    )


# ============================================================
# PromptLen 기반 completion-only collator
# ============================================================
@dataclass
class PromptLenCompletionCollator:
    """
    labels 마스킹을 'prompt 길이'로 하는 안정적인 completion-only collator
    - full_text: system+user+assistant
    - prompt_text: system+user (assistant 제외)
    - labels[:prompt_len] = -100
    """
    tokenizer: Any
    max_length: int
    pad_to_multiple_of: Optional[int] = 8
    debug_kept_threshold: int = 5

    # =========================
    # [ADDED] kept 분포 로깅 옵션
    # =========================
    log_every_n_steps: int = 50          # N step마다 분포 출력
    print_suspicious_samples: bool = False  # 분포만 볼 거면 False 추천

    # =========================
    # [ADDED] state (DDP rank0 로그용)
    # init에서 받지 않도록 field(init=False)로 고정
    # =========================
    step: int = field(default=0, init=False)
    kept_zero: int = field(default=0, init=False)
    kept_total: int = field(default=0, init=False)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        self.step += 1  # [ADDED]

        full_texts = [f["full_text"] for f in features]
        prompt_texts = [f["prompt_text"] for f in features]

        # full tokenize (이게 실제 input_ids가 됨)
        full_enc = self.tokenizer(
            full_texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # prompt tokenize (길이만 필요)
        prompt_enc = self.tokenizer(
            prompt_texts,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        input_ids = full_enc["input_ids"]

        # =========================
        # [CHANGED] Gemma3 training REQUIRES token_type_ids.
        # =========================
        full_enc["token_type_ids"] = torch.zeros_like(input_ids)

        labels = input_ids.clone()

        for i in range(len(features)):
            prompt_ids = prompt_enc["input_ids"][i]  # list[int]
            prompt_len = len(prompt_ids)

            seq_len = input_ids.size(1)
            prompt_len = min(prompt_len, seq_len)

            # prompt 부분은 학습 제외
            labels[i, :prompt_len] = -100

            # pad 토큰도 학습 제외
            if self.tokenizer.pad_token_id is not None:
                labels[i, input_ids[i] == self.tokenizer.pad_token_id] = -100

            # =========================
            # [ADDED] kept 통계 누적 (분포/비율 확인)
            # 실제로 loss 계산에 사용된 assistant 토큰수임
            # 0.6-0.8 수준으로 치명적인 수준은 아님 -> 대부분의 샘플이 정상적으로 completion only loss가 걸리고 있다.
            # =========================
            kept = (labels[i] != -100).sum().item()
            self.kept_total += 1
            if kept == 0:
                self.kept_zero += 1

            # =========================
            # [CHANGED] suspicious sample 출력은 옵션으로
            # =========================
            if (
                self.print_suspicious_samples
                and os.environ.get("RANK", "0") == "0"
                and kept <= self.debug_kept_threshold
            ):
                print("==== suspicious sample (kept<=threshold) ====", flush=True)
                print("prompt_len:", prompt_len, "kept:", kept, flush=True)
                print(full_texts[i][:400], flush=True)

        # =========================
        # [ADDED] N step마다 kept==0 비율 출력 (rank0만)
        # =========================
        if os.environ.get("RANK", "0") == "0" and self.log_every_n_steps > 0:
            if self.step % self.log_every_n_steps == 0:
                ratio = 100.0 * self.kept_zero / max(1, self.kept_total)
                print(
                    f"[kept==0 ratio] step={self.step} "
                    f"zero={self.kept_zero}/{self.kept_total} ({ratio:.2f}%)",
                    flush=True,
                )

        full_enc["labels"] = labels
        return full_enc


# ============================================================
# Main
# ============================================================
def main():
    local_rank = set_device_from_local_rank()

    parser = argparse.ArgumentParser()

    # Model / Data
    parser.add_argument("--model_id", type=str, default="google/gemma-3-27b-it")
    parser.add_argument("--data_files", type=str, required=True, help='예: "dataset_gemma_chat.jsonl" 또는 "/data/*.jsonl"')

    # input format (이 스크립트는 messages 전용)
    parser.add_argument("--messages_column", type=str, default="messages")

    parser.add_argument("--subset_size", type=int, default=0, help="0이면 전체 사용, 아니면 N개만 사용")
    parser.add_argument("--seed", type=int, default=42)

    # Train
    parser.add_argument("--output_dir", type=str, default="outputs/gemma3_27b_qlora_promptlen")
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=30)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Debug
    parser.add_argument("--debug_kept_threshold", type=int, default=5)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    print(
        f"[DDP] pid={os.getpid()} | LOCAL_RANK={local_rank} | RANK={rank} | WORLD_SIZE={world_size} | "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} | "
        f"torch.cuda.current_device={torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'}",
        flush=True,
    )

    # =========================
    # [CHANGED] Enable TF32 globally (works even if TrainingArguments tf32 is set).
    # =========================
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ------------------------
    # Tokenizer
    # ------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    tokenizer.padding_side = "right"

    # =========================
    # [CHANGED] Force pad_token + pad_token_id and also set model.config.pad_token_id later.
    # Completion-only masking + padding 안정성에 도움.
    # =========================
    
    # Keep native pad token if defined by tokenizer/model
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------
    # QLoRA: NF4 4bit 로드
    # ------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # DDP에서 각 프로세스가 자기 GPU로만 로드하도록 강제
    device_map = {"": local_rank} if torch.cuda.is_available() else None

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

    # =========================
    # [CHANGED] Make pad_token_id consistent between tokenizer and model.
    # =========================
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    # ------------------------
    # Prepare model for k-bit training + Gradient Checkpointing
    # ------------------------
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # =========================
    # [CHANGED] Use non-reentrant gradient checkpointing (often more stable with PEFT + 4bit + DDP).
    # =========================
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.config.use_cache = False

    # ------------------------
    # LoRA config
    # ------------------------
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",

        # =========================
        # [CHANGED] Recommend attention-only LoRA for stability/VRAM.
        # If you really want MLP LoRA too, revert to gate/up/down as before.
        # =========================
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
        ],
    )
    model = get_peft_model(model, lora_config)

    if rank == 0:
        model.print_trainable_parameters()

    # ------------------------
    # Dataset 로드
    # ------------------------
    ds = load_dataset("json", data_files=args.data_files, split="train")
    ds = ds.shuffle(seed=args.seed)
    if args.subset_size and args.subset_size > 0:
        ds = ds.select(range(min(args.subset_size, len(ds))))

    if args.messages_column not in ds.column_names:
        raise ValueError(f"messages_column='{args.messages_column}'이 없습니다. 현재 컬럼: {ds.column_names}")

    if rank == 0:
        print("Using input_format = messages", flush=True)

    # ------------------------
    # messages -> prompt_text / full_text 생성
    # ------------------------
    def to_prompt_full(example: Dict[str, Any]) -> Dict[str, str]:
        msgs = example.get(args.messages_column)

        if not isinstance(msgs, list) or len(msgs) < 2:
            return {"prompt_text": "", "full_text": ""}

        last = msgs[-1]
        if not (isinstance(last, dict) and last.get("role") in ("assistant", "model")):
            # assistant turn이 없으면 completion-only 정의가 불가
            return {"prompt_text": "", "full_text": ""}

        prompt_msgs = msgs[:-1]
        full_msgs = msgs

        prompt_text = normalize_messages_to_text(tokenizer, prompt_msgs)
        full_text = normalize_messages_to_text(tokenizer, full_msgs)

        return {"prompt_text": prompt_text, "full_text": full_text}

    ds = ds.map(to_prompt_full, desc="Building prompt_text/full_text")
    ds = ds.filter(
        lambda ex: isinstance(ex["prompt_text"], str)
        and isinstance(ex["full_text"], str)
        and len(ex["prompt_text"].strip()) > 0
        and len(ex["full_text"].strip()) > 0
    )

    if len(ds) == 0:
        raise ValueError("전처리 후 학습 가능한 샘플이 0개입니다. messages 구조/내용을 확인하세요.")

    if rank == 0:
        print("==== SAMPLE AFTER NORMALIZATION (PROMPT/FULL) ====")
        for ex in ds.select(range(min(2, len(ds)))):
            print("=" * 80)
            print("---- PROMPT ----")
            print(ex["prompt_text"][:800])
            print("---- FULL ----")
            print(ex["full_text"][:800])

    # ------------------------
    # Data collator: completion-only labels 마스킹 (prompt_len 기반)
    # ------------------------
    data_collator = PromptLenCompletionCollator(
        tokenizer=tokenizer,
        max_length=args.max_seq_len,
        debug_kept_threshold=args.debug_kept_threshold,
        log_every_n_steps=50,                             # [ADDED]
        print_suspicious_samples=False, 
    )
    # data_collator = PromptLenCompletionCollator(
    #     tokenizer=tokenizer,
    #     max_length=args.max_seq_len,
    #     debug_kept_threshold=args.debug_kept_threshold,
    # )

    train_args = build_training_args(args)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        data_collator=data_collator,
    )

    trainer.train()

    if rank == 0:
        trainer.model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Done. LoRA adapter saved to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()