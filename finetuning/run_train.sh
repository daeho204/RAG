#!/usr/bin/env bash
set -euo pipefail


############################################
# Model / Data
############################################
MODEL_ID="google/gemma-3-27b-it"

# 🔥 지식 QA jsonl (messages 포함)
# DATA_FILES="/dataset/jeongwoo/MEDICAL_TL_DATASET/Processed/TS_국문_학술 논문 및 저널/train_adaptllm_medical.jsonl"
# 데이터셋 형태 수정 테스트(02.24)
DATA_FILES="/dataset/train_1.jsonl"

# 0 이면 전체 데이터 사용
SUBSET_SIZE=0

############################################
# Train config
############################################
MAX_SEQ_LEN=2048
# OUT_DIR="/dataset/finetune/outputs/gemma3_27b_medical_AdaptLLM_260206_2"
OUT_DIR="/dataset/outputs/gemma3_27b_medical_AdaptLLM_260305(kept checker)"

PER_DEVICE_BS=1
GRAD_ACCUM=8
NUM_EPOCHS=1.0
LR=1e-4
# 2026-03-09 2e-4는 aggressive adapation에 가까워서 수치 변경. 안정 SFT는 5e-5, QA FT는 1e-4가 적절

############################################
# LoRA
############################################
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

############################################
# DDP / CUDA
############################################
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

############################################
# Launch (accelerate)
############################################
accelerate launch \
  --num_processes 4 \
  --num_machines 1 \
  --mixed_precision bf16 \
  train_gemma3_27b_qlora_trainer_260309.py \
  --model_id "${MODEL_ID}" \
  --data_files "${DATA_FILES}" \
  --subset_size "${SUBSET_SIZE}" \
  --max_seq_len "${MAX_SEQ_LEN}" \
  --output_dir "${OUT_DIR}" \
  --per_device_train_batch_size "${PER_DEVICE_BS}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --num_train_epochs "${NUM_EPOCHS}" \
  --learning_rate "${LR}" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --weight_decay 0.01 \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  
  ##초반 학습 제어용 scheduler_type, warmup_ratio, weigt_decay : trainer_260309부터
