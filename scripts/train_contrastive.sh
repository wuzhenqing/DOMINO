#!/bin/bash
# Example script for training DOMINO contrastive soft tokens
# Customize the paths below for your setup

LLM_MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"   # HuggingFace model name or local path
TRAIN_DATA="./data/train.jsonl"               # Path to training data (jsonl)
VALID_DATA="./data/valid.jsonl"               # Path to validation data (jsonl)
OUTPUT_DIR="./outputs/contrastive"            # Output directory

PUBLIC_SOFT_TOKEN_COUNT=256
PRIVATE_SOFT_TOKEN_COUNT=256
EPOCH=10

deepspeed --master_port 1113 --include localhost:0 ../domino/contrastive/train.py \
    --model_name_or_path "${LLM_MODEL}" \
    --use_fast_tokenizer True \
    --use_flash_attn False \
    --train_data_path "${TRAIN_DATA}" \
    --valid_data_path "${VALID_DATA}" \
    --save_strategy "no" \
    --max_seq_len 1024 \
    --public_soft_token_count ${PUBLIC_SOFT_TOKEN_COUNT} \
    --private_soft_token_count ${PRIVATE_SOFT_TOKEN_COUNT} \
    --output_dir "${OUTPUT_DIR}" \
    --overwrite_output_dir True \
    --deepspeed ../configs/ds_z2.json \
    --gradient_checkpointing True \
    --num_train_epochs ${EPOCH} \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-3 \
    --lr_scheduler_type "linear" \
    --warmup_steps 0 \
    --bf16 True \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --logging_dir "${OUTPUT_DIR}/logs" \
    --report_to "tensorboard" \
    --evaluation_strategy "no" \
    --per_device_eval_batch_size 1 \
    --eval_accumulation_steps 10
