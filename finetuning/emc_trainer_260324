# train_gemma3_27b_qlora_promptlen_collator_stable.py
# -*- coding: utf-8 -*-

"""
Gemma3 QLoRA training (completion-only loss) using prompt-length masking.

Key improvements in this version:
1) Keep native pad token if tokenizer/model already defines it
2) Always provide token_type_ids for Gemma3 training
3) Use non-reentrant gradient checkpointing explicitly
4) Use attention-only LoRA target modules for better stability / lower VRAM
5) Track kept-token statistics:
   - zero ratio
   - average kept
   - small kept ratio (<= 16)
6) Filter samples with zero trainable assistant tokens BEFORE training
7) Check whether prompt token ids are a true prefix of full token ids
8) Use ddp_find_unused_parameters=False by default
9) Use a slightly safer default LR for 27B QLoRA

Example:
accelerate launch --num_processes 4 train_gemma3_27b_qlora_promptlen_collator_stable.py \
  --data_files dataset_gemma_chat.jsonl \
  --output_dir outputs/gemma3_27b_qlora_promptlen_stable
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
# TrainingArguments builder
# ============================================================
def build_training_args(args):
    sig = inspect.signature(TrainingArguments.__init__)
    extra = (
        {"eval_strategy": "no"}
        if "eval_strategy" in sig.parameters
        else {"evaluation_strategy": "no"}
    )

    return TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True,
        fp16=False,
        tf32=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        optim="paged_adamw_8bit",
        report_to="none",

        # More efficient / usually preferred once training is stable.
        ddp_find_unused_parameters=True,

        # Required because collator uses raw text columns.
        remove_unused_columns=False,

        # Can reduce padding waste when sequence lengths vary.
        group_by_length=False,

        **extra,
    )


# ============================================================
# Dataset normalization
# ============================================================
def normalize_content(content: Any) -> str:
    """
    Convert content into plain text as safely as possible.

    Supported cases:
    - str
    - list of dicts like [{"type": "text", "text": "..."}]
    - other values fall back to str(...)
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(content)


def normalize_messages_to_text(tokenizer, msgs) -> str:
    """
    msgs: list[{"role":..., "content":...}] -> chat template string
    """
    if not isinstance(msgs, list) or len(msgs) == 0:
        return ""

    cleaned = []
    for m in msgs:
        if isinstance(m, dict) and "role" in m and "content" in m:
            cleaned.append(
                {
                    "role": m["role"],
                    "content": normalize_content(m["content"]),
                }
            )

    if not cleaned:
        return ""

    return tokenizer.apply_chat_template(
        cleaned,
        tokenize=False,
        add_generation_prompt=False,
    )

#gemma3의 module들 중에 language관련 모델만 추출한다.
def find_text_lora_target_modules(model):
    allowed_suffixes = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    targets = []
    for module_name, _ in model.named_modules():
        if not module_name.startswith("language_model"):
            continue
        if module_name.endswith(allowed_suffixes):
            targets.append(module_name)
    return sorted(set(targets))

# ============================================================
# PromptLen-based completion-only collator
# ============================================================
@dataclass
class PromptLenCompletionCollator:
    """
    Stable completion-only collator using prompt length masking.

    - full_text: system + user + assistant
    - prompt_text: system + user
    - labels[:prompt_len] = -100
    """
    tokenizer: Any
    max_length: int
    pad_to_multiple_of: Optional[int] = 8
    debug_kept_threshold: int = 5
    small_kept_threshold: int = 16
    log_every_n_steps: int = 50
    print_suspicious_samples: bool = False

    # Runtime stats
    step: int = field(default=0, init=False)
    kept_zero: int = field(default=0, init=False)
    kept_total: int = field(default=0, init=False)
    kept_sum: int = field(default=0, init=False)
    kept_small: int = field(default=0, init=False)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        self.step += 1

        full_texts = [f["full_text"] for f in features]
        prompt_texts = [f["prompt_text"] for f in features]

        # Tokenize full text -> actual model inputs
        full_enc = self.tokenizer(
            full_texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
            add_special_tokens=False,
        )

        # Tokenize prompt only -> only length is needed
        prompt_enc = self.tokenizer(
            prompt_texts,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )

        input_ids = full_enc["input_ids"]

        # Gemma3 training expects token_type_ids.
        full_enc["token_type_ids"] = torch.zeros_like(input_ids)

        labels = input_ids.clone()

        for i in range(len(features)):
            prompt_ids = prompt_enc["input_ids"][i]
            prompt_len = len(prompt_ids)

            seq_len = input_ids.size(1)
            prompt_len = min(prompt_len, seq_len)

            # Mask prompt tokens out of loss.
            labels[i, :prompt_len] = -100

            # Mask padding tokens out of loss.
            if self.tokenizer.pad_token_id is not None:
                labels[i, input_ids[i] == self.tokenizer.pad_token_id] = -100

            kept = (labels[i] != -100).sum().item()

            self.kept_total += 1
            self.kept_sum += kept

            if kept == 0:
                self.kept_zero += 1
            if kept <= self.small_kept_threshold:
                self.kept_small += 1

            if (
                self.print_suspicious_samples
                and os.environ.get("RANK", "0") == "0"
                and kept <= self.debug_kept_threshold
            ):
                print("==== suspicious sample (kept<=threshold) ====", flush=True)
                print(f"prompt_len={prompt_len} kept={kept}", flush=True)
                print(full_texts[i][:600], flush=True)

        if os.environ.get("RANK", "0") == "0" and self.log_every_n_steps > 0:
            if self.step % self.log_every_n_steps == 0:
                zero_ratio = 100.0 * self.kept_zero / max(1, self.kept_total)
                small_ratio = 100.0 * self.kept_small / max(1, self.kept_total)
                avg_kept = self.kept_sum / max(1, self.kept_total)

                print(
                    f"[kept stats] step={self.step} "
                    f"avg={avg_kept:.2f} "
                    f"small<={self.small_kept_threshold}={self.kept_small}/{self.kept_total} ({small_ratio:.2f}%) "
                    f"zero={self.kept_zero}/{self.kept_total} ({zero_ratio:.2f}%)",
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

    # Model / data
    parser.add_argument("--model_id", type=str, default="google/gemma-3-27b-it")
    parser.add_argument(
        "--data_files",
        type=str,
        required=True,
        help='Example: "dataset_gemma_chat.jsonl" or "/data/*.jsonl"',
    )
    parser.add_argument("--messages_column", type=str, default="messages")
    parser.add_argument("--subset_size", type=int, default=0, help="0 means use all")
    parser.add_argument("--seed", type=int, default=42)

    # Training
    parser.add_argument("--output_dir", type=str, default="outputs/gemma3_27b_qlora_promptlen_stable")
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=30)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Debug / stats
    parser.add_argument("--debug_kept_threshold", type=int, default=5)
    parser.add_argument("--small_kept_threshold", type=int, default=16)
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--print_suspicious_samples", action="store_true")
    parser.add_argument("--prefix_check_samples", type=int, default=5)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    print(
        f"[DDP] pid={os.getpid()} | "
        f"LOCAL_RANK={local_rank} | RANK={rank} | WORLD_SIZE={world_size} | "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} | "
        f"torch.cuda.current_device={torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'}",
        flush=True,
    )

    # Enable TF32 globally as well.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    tokenizer.padding_side = "right"

    # Keep native pad token if already defined.
    # Only fall back to EOS when absolutely necessary.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------------
    # QLoRA 4-bit config
    # --------------------------------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Force each DDP process to load model on its own GPU only.
    device_map = {"": local_rank} if torch.cuda.is_available() else None

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    # --------------------------------------------------------
    # Prepare for k-bit training + explicit gradient checkpointing
    # --------------------------------------------------------
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=False,
    )

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.config.use_cache = False

    # --------------------------------------------------------
    # LoRA config
    # --------------------------------------------------------
    
    target_modules = find_text_lora_target_modules(model)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules
    )
    model = get_peft_model(model, lora_config)

    if rank == 0:
        model.print_trainable_parameters()

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------
    ds = load_dataset("json", data_files=args.data_files, split="train")
    ds = ds.shuffle(seed=args.seed)

    if args.subset_size and args.subset_size > 0:
        ds = ds.select(range(min(args.subset_size, len(ds))))

    if args.messages_column not in ds.column_names:
        raise ValueError(
            f"messages_column='{args.messages_column}' not found. "
            f"Available columns: {ds.column_names}"
        )

    if rank == 0:
        print("Using input_format = messages", flush=True)

    # --------------------------------------------------------
    # messages -> prompt_text / full_text
    # --------------------------------------------------------
    def to_prompt_full(example: Dict[str, Any]) -> Dict[str, str]:
        msgs = example.get(args.messages_column)

        if not isinstance(msgs, list) or len(msgs) < 2:
            return {"prompt_text": "", "full_text": ""}

        last = msgs[-1]
        if not (isinstance(last, dict) and last.get("role") in ("assistant", "model")):
            # Completion-only requires the final turn to be assistant/model.
            return {"prompt_text": "", "full_text": ""}

        prompt_msgs = msgs[:-1]
        full_msgs = msgs

        prompt_text = normalize_messages_to_text(tokenizer, prompt_msgs)
        full_text = normalize_messages_to_text(tokenizer, full_msgs)

        return {
            "prompt_text": prompt_text,
            "full_text": full_text,
        }

    ds = ds.map(to_prompt_full, desc="Building prompt_text/full_text")

    ds = ds.filter(
        lambda ex: isinstance(ex["prompt_text"], str)
        and isinstance(ex["full_text"], str)
        and len(ex["prompt_text"].strip()) > 0
        and len(ex["full_text"].strip()) > 0,
        desc="Filtering empty normalized samples",
    )

    if len(ds) == 0:
        raise ValueError("No trainable samples remain after normalization.")

    # --------------------------------------------------------
    # Prefix check: verify prompt token ids are a true prefix of full token ids
    # --------------------------------------------------------
    if rank == 0:
        print("==== PREFIX CHECK ====", flush=True)
        n_check = min(args.prefix_check_samples, len(ds))
        mismatch_count = 0

        for ex in ds.select(range(n_check)):
            prompt_ids = tokenizer(
                ex["prompt_text"],
                truncation=True,
                max_length=args.max_seq_len,
                padding=False,
            )["input_ids"]

            full_ids = tokenizer(
                ex["full_text"],
                truncation=True,
                max_length=args.max_seq_len,
                padding=False,
            )["input_ids"]

            is_prefix = full_ids[:len(prompt_ids)] == prompt_ids
            if not is_prefix:
                mismatch_count += 1

            print(
                f"prefix_match={is_prefix} "
                f"prompt_len={len(prompt_ids)} full_len={len(full_ids)}",
                flush=True,
            )

        if mismatch_count > 0:
            print(
                f"[WARNING] prefix mismatches detected: {mismatch_count}/{n_check}. "
                "Prompt-length masking may not align perfectly with assistant boundaries.",
                flush=True,
            )

    # --------------------------------------------------------
    # Filter samples with zero trainable assistant tokens
    # --------------------------------------------------------
    def has_trainable_tokens(example: Dict[str, Any]) -> bool:
        prompt_ids = tokenizer(
            example["prompt_text"],
            truncation=True,
            max_length=args.max_seq_len,
            padding=False,
        )["input_ids"]

        full_ids = tokenizer(
            example["full_text"],
            truncation=True,
            max_length=args.max_seq_len,
            padding=False,
        )["input_ids"]

        prompt_len = min(len(prompt_ids), len(full_ids))
        kept = len(full_ids) - prompt_len
        return kept > 0

    before_len = len(ds)
    ds = ds.filter(
        has_trainable_tokens,
        desc="Filtering samples with no trainable assistant tokens",
    )
    after_len = len(ds)

    if len(ds) == 0:
        raise ValueError("All samples were removed by trainable-token filtering.")

    if rank == 0:
        removed = before_len - after_len
        print(
            f"Trainable-token filter: kept={after_len}, removed={removed}, "
            f"removed_ratio={(removed / max(1, before_len)) * 100:.2f}%",
            flush=True,
        )

    # --------------------------------------------------------
    # Print sample previews
    # --------------------------------------------------------
    if rank == 0:
        print("==== SAMPLE AFTER NORMALIZATION (PROMPT/FULL) ====", flush=True)
        for ex in ds.select(range(min(2, len(ds)))):
            print("=" * 80, flush=True)
            print("---- PROMPT ----", flush=True)
            print(ex["prompt_text"][:800], flush=True)
            print("---- FULL ----", flush=True)
            print(ex["full_text"][:800], flush=True)

    # --------------------------------------------------------
    # Data collator
    # --------------------------------------------------------
    data_collator = PromptLenCompletionCollator(
        tokenizer=tokenizer,
        max_length=args.max_seq_len,
        debug_kept_threshold=args.debug_kept_threshold,
        small_kept_threshold=args.small_kept_threshold,
        log_every_n_steps=args.log_every_n_steps,
        print_suspicious_samples=args.print_suspicious_samples,
    )

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