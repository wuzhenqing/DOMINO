#!/bin/bash
# Pipeline: Synthetic Ascend C Kernel Data via DOMINO contrastive soft tokens
# Customize paths for your environment.

LLM_MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct"
SOFT_PROMPT_DIR="./outputs/contrastive_ascend_c"
PUBLIC_SOFT_TOKEN_COUNT=256
TEMP=0.8
TARGET_COUNT=80000
BATCH_SIZE=200
DEVICES='0'

## Step 1: Generate synthetic Ascend C operator instructions using learned public soft tokens
python -m domino.contrastive.generate \
    --pretrained_model_name_or_path "${LLM_MODEL}" \
    --tokenizer_name_or_path "${LLM_MODEL}" \
    --soft_prompt_dir "${SOFT_PROMPT_DIR}" \
    --inference_engine vllm \
    --public_soft_token_count "${PUBLIC_SOFT_TOKEN_COUNT}" \
    --temp "${TEMP}" \
    --target_count "${TARGET_COUNT}" \
    --tensor_parallel_size 1 \
    --device "cuda:0"
echo "======Synthetic Ascend C Instruction Generation Done! $(date) ========"

## Step 2: Instruction quality assessment (Ascend C operator-kernel prompt)
python -m domino.pipeline.ascend_c.quality_assessment \
    --process_stage quality_assessment \
    --synthetic_instruct_path "${SOFT_PROMPT_DIR}/vllm_generated_${TARGET_COUNT}_samples_temp${TEMP}.jsonl" \
    --GRADE_MODEL "${LLM_MODEL}" \
    --batch_size "${BATCH_SIZE}" \
    --devices "${DEVICES}"
echo "======Ascend C Instruction Assessment Done! $(date) ========"

## Step 3: Generate Ascend C kernel responses for excellent instructions
python -m domino.pipeline.ascend_c.generate_response \
    --process_stage inference_response \
    --synthetic_instruct_path "${SOFT_PROMPT_DIR}/vllm_generated_${TARGET_COUNT}_samples_temp${TEMP}_instruct_quality.jsonl" \
    --inference_model "${LLM_MODEL}" \
    --batch_size "${BATCH_SIZE}" \
    --devices "${DEVICES}"
echo "======Ascend C Response Generation Done! $(date) ========"

# Optional next steps (run after response generation):
# ## Step 4: Compiler-anchored syntax check via CANN CPU-debug build
# python -m domino.pipeline.ascend_c.compile_check \
#     --process_stage compile_check \
#     --instruct_response_path "${SOFT_PROMPT_DIR}/vllm_generated_${TARGET_COUNT}_samples_temp${TEMP}_instruct_quality_excellent_response.jsonl" \
#     --num_workers 4 \
#     --timeout 300
# echo "======Ascend C Compile Check Done! $(date) ========"
#
# ## Step 5: Format filtered pairs for LLaMA-Factory SFT
# python -m domino.pipeline.ascend_c.format_for_sft \
#     "${SOFT_PROMPT_DIR}/vllm_generated_${TARGET_COUNT}_samples_temp${TEMP}_instruct_quality_excellent_response.jsonl_syntex_quality.jsonl"
# echo "======Ascend C SFT Formatting Done! $(date) ========"
