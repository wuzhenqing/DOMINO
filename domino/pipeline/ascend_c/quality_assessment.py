import os
import json
import re
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def remove_duplicate_words(text):
    return re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)


def input_quality_rating(input):
    user_message = f'''
# Instruction

You need to assess the quality of the given Ascend C operator-kernel instruction based on its clarity, specificity, completeness, and challenge for an Ascend NPU kernel author.

The rating scale is as follows:

- very poor: The instruction is unclear, vague, or disorganized and the examples are wrong. It lacks essential information and context, contains many unreadable characters for humans or has large scale repetitive parts.
- poor: The instruction is somewhat unclear or lacks important details, requiring significant clarification, or contains irrelevant content and repetition. The examples are partially wrong. It also contains a small number of human unreadable characters and minor repetition.
- average: The instruction has moderate clarity and specificity. It is basically complete but may require some additional information for full understanding, and the difficulty is relatively easy. The examples are right.
- good: The instruction is clear and specific, and it is generally well-articulated. It is complete, providing sufficient context to understand its intent, with little to no irrelevant content or noise, and has a moderate level of difficulty. The examples are reasonable and right.
- excellent: The instruction is very clear, specific, and well-articulated. It contains all the necessary information and context for providing a comprehensive Ascend C kernel implementation and also has a certain level of challenge, requiring multi-step reasoning, careful tiling, queue management, or vector compute. The corresponding examples are reasonable and right.

## Ascend C Operator-Kernel Instruction
```
{input}
```

## Output Format
Given the instruction, you first need to give an assessment, highlighting the strengths and/or weaknesses of the Ascend C operator-kernel instruction.
Then, you need to output a rating from very poor to excellent by filling in the placeholders in [...]:
{{   
    "explanation": "[...]",
    "input_quality": "[very poor/poor/average/good/excellent]"
}}
'''
    return user_message


def get_vllm_configuration(GRADE_MODEL, devices, max_tokens=1024, gpu_memory_utilization=0.85):
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    with open(os.path.join(os.path.dirname(__file__), '..', 'llm_configs.json'), "r") as f:
        model_configs = json.load(f)
        model_config = model_configs[GRADE_MODEL.split('/')[-1]]
        stop_token_ids = model_config["stop_token_ids"]

    print("Start Local vllm engine...")
    llm = LLM(model=GRADE_MODEL,
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

    tokenizer = AutoTokenizer.from_pretrained(GRADE_MODEL)

    return llm, params, tokenizer


def quality_process_batch(batch, llm, params, tokenizer):
    synthetic_instructions = [item['synthetic_text'] for item in batch]
    prompts = []
    for instruction in synthetic_instructions:
        instruction = input_quality_rating(input=instruction)
        chat = [{"role": "user", "content": instruction}]
        template = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompts.append(template)

    outputs = llm.generate(prompts, params)

    for i, item in enumerate(batch):
        item["instruction_quality"] = outputs[i].outputs[0].text.strip()

    return batch


def instruct_quality_assessment(synthetic_instruct_path, GRADE_MODEL, BATCH_SIZE, devices, max_tokens=1024, gpu_memory_utilization=0.85):
    synthetic_instruct_dataset = []
    with open(synthetic_instruct_path, 'r') as f:
        for line in f:
            synthetic_instruct_dataset.append(json.loads(line))
    print(f"synthetic instruction dataset length = {len(synthetic_instruct_dataset)}")

    llm, params, tokenizer = get_vllm_configuration(GRADE_MODEL, devices, max_tokens=max_tokens, gpu_memory_utilization=gpu_memory_utilization)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_batches = (len(synthetic_instruct_dataset) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in tqdm(range(num_batches)):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, len(synthetic_instruct_dataset))
        batch = synthetic_instruct_dataset[start_idx:end_idx]

        batch = quality_process_batch(batch, llm, params, tokenizer)

        synthetic_instruct_dataset[start_idx:end_idx] = batch

    saved_path = synthetic_instruct_path.replace(".jsonl", "_instruct_quality.jsonl")
    with open(saved_path, 'w') as g:
        for item in synthetic_instruct_dataset:
            g.write(json.dumps(item) + '\n')


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_stage", type=str, default='quality_assessment', choices=["quality_assessment"])
    parser.add_argument("--synthetic_instruct_path", type=str, required=True)
    parser.add_argument("--GRADE_MODEL", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--devices", type=str, default='0,1,2,3')
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = parser.parse_args()

    if args.process_stage == "quality_assessment":
        instruct_quality_assessment(
            synthetic_instruct_path=args.synthetic_instruct_path,
            GRADE_MODEL=args.GRADE_MODEL,
            BATCH_SIZE=args.batch_size,
            devices=args.devices,
            max_tokens=args.max_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
    else:
        print("Check process stage.")
