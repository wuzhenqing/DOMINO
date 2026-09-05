#!/bin/bash
# Example script for training baseline soft prompt tuning

LLM_MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
TRAIN_DATA="./data/train.jsonl"
VALID_DATA="./data/valid.jsonl"
OUTPUT_DIR="./outputs/soft_prompt"

SOFT_TOKEN_COUNT=256
EPOCH=10

deepspeed --master_port 1114 --include localhost:0 ../domino/soft_prompt/train.py \
    --model_name_or_path "${LLM_MODEL}" \
    --use_fast_tokenizer True \
    --use_flash_attn False \
    --train_data_path "${TRAIN_DATA}" \
    --valid_data_path "${VALID_DATA}" \
    --max_seq_len 1024 \
    --soft_token_count ${SOFT_TOKEN_COUNT} \
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
