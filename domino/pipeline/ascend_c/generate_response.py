import json
import os
import argparse
import re
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm


def process_response_batch(batch, llm, params, tokenizer):
    synthetic_instructions = [item['synthetic_text'] for item in batch]

    ASCEND_C_FORMATTING = (
        "You will implement the described operator as a single-file Ascend C kernel "
        "(`.asc` style). Provide a kernel class with Init() and Process() methods, "
        "and an `extern \"C\" __global__ __aicore__` entry function. "
        "Use `kernel_operator.h` and standard Ascend C APIs. "
        "Enclose the complete kernel code within ```cpp and ``` fences."
    )

    prompts = []
    for synthetic_instruction in synthetic_instructions:
        format_prompt = f"### Instruction:\n{synthetic_instruction}\n{ASCEND_C_FORMATTING}\n\n### Response:\n"
        chat = [{"role": "user", "content": format_prompt}]
        template = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompts.append(template)

    outputs = llm.generate(prompts, params)

    for i, item in enumerate(batch):
        item["original_response"] = outputs[i].outputs[0].text.strip()

    return batch


def extract_instruct_quality(instruction_quality_string):
    valid_quality = {'unknown', 'very poor', 'poor', 'average', 'good', 'excellent'}
    try:
        cleaned_instruction_quality_string = instruction_quality_string.replace("```json", "").replace("```", "").strip()
        instruction_quality_json = json.loads(cleaned_instruction_quality_string)
        input_quality = instruction_quality_json['input_quality']
    except Exception:
        match = re.search(r'"input_quality"\s*:\s*"([^"]+)"', instruction_quality_string)
        if match:
            input_quality = match.group(1)
        else:
            input_quality = "unknown"

    input_quality = input_quality.lower()
    if input_quality not in valid_quality:
        input_quality = 'unknown'

    return input_quality


def get_vllm_configuration(inference_model, devices, max_tokens=4096, gpu_memory_utilization=0.85):
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    tokenizer = AutoTokenizer.from_pretrained(inference_model, trust_remote_code=True)
    stop_token_ids = [tokenizer.eos_token_id, tokenizer.pad_token_id]

    print("Start Local vllm engine...")
    llm = LLM(model=inference_model,
              dtype="bfloat16",
              trust_remote_code=True,
              max_model_len=8192,
              tensor_parallel_size=len(devices.split(',')),
              gpu_memory_utilization=gpu_memory_utilization,
              enforce_eager=True)

    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0,
        top_p=1,
        repetition_penalty=1,
        stop_token_ids=stop_token_ids,
    )

    return llm, params, tokenizer


def generate_response(synthetic_instruct_path, inference_model, BATCH_SIZE, devices, max_count=None, max_tokens=4096, gpu_memory_utilization=0.85, quality_threshold=None):
    synthetic_instruct_datasets = []
    with open(synthetic_instruct_path, 'r') as f:
        for line in f:
            synthetic_instruct_datasets.append(json.loads(line))

    print(f"len of synthetic instruct dataset = {len(synthetic_instruct_datasets)}")

    if quality_threshold is None:
        threshold_instruct_quality = {'excellent'}
    else:
        threshold_instruct_quality = set(quality_threshold)
    excellent_instruct_datasets = []
    for item in synthetic_instruct_datasets:
        instruction_quality_string = item['instruction_quality']
        input_quality = extract_instruct_quality(instruction_quality_string)

        if input_quality in threshold_instruct_quality:
            excellent_instruct_datasets.append(item)

    if max_count is not None and max_count > 0:
        excellent_instruct_datasets = excellent_instruct_datasets[:max_count]

    print(f"above threshold instruct count = {len(excellent_instruct_datasets)}")

    llm, params, tokenizer = get_vllm_configuration(inference_model, devices, max_tokens=max_tokens, gpu_memory_utilization=gpu_memory_utilization)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_batches = (len(excellent_instruct_datasets) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in tqdm(range(num_batches)):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, len(excellent_instruct_datasets))
        batch = excellent_instruct_datasets[start_idx:end_idx]

        batch = process_response_batch(batch, llm, params, tokenizer)

        excellent_instruct_datasets[start_idx:end_idx] = batch

    saved_path = synthetic_instruct_path.replace(".jsonl", "_excellent_response.jsonl")
    with open(saved_path, 'w') as g:
        for item in excellent_instruct_datasets:
            g.write(json.dumps(item) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_stage", type=str, default='inference_response', choices=["inference_response"])
    parser.add_argument("--synthetic_instruct_path", type=str, required=True)
    parser.add_argument("--inference_model", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--devices", type=str, default='0,1,2,3')
    parser.add_argument("--max_count", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--quality_threshold", type=str, nargs='+', default=None,
                        help="Quality labels to accept (default: excellent). Pass multiple values, e.g. average good excellent.")
    args = parser.parse_args()

    if args.process_stage == "inference_response":
        generate_response(synthetic_instruct_path=args.synthetic_instruct_path,
                          inference_model=args.inference_model,
                          BATCH_SIZE=args.batch_size,
                          devices=args.devices,
                          max_count=args.max_count,
                          max_tokens=args.max_tokens,
                          gpu_memory_utilization=args.gpu_memory_utilization,
                          quality_threshold=args.quality_threshold)
    else:
        print("Check process stage.")
