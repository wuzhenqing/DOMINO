import argparse
import json
import os
import sys

# Avoid proxy/HF lookup issues before importing vLLM/transformers.
for key in [
    "all_proxy",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
]:
    os.environ.pop(key, None)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from .ascend_c_metrics import ascend_c_metrics, extract_code


SYSTEM_MESSAGE = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user"


def format_ascend_c_prompt(task: dict) -> str:
    prompt = f"{SYSTEM_MESSAGE}\n\n"
    prompt += (
        "You will be given a specification for an Ascend C operator kernel. "
        "Generate a complete, compilable Ascend C kernel implementation that matches the specification. "
        "Enclose the kernel code within ```cpp fences. "
        "Do not return anything except the code inside the fence.\n\n"
    )
    prompt += f"Specification:\n{task['question_content']}\n\n"
    if task.get("main", "").strip():
        prompt += (
            "The kernel will be linked with a separate self-verifying main function. "
            "Implement only the kernel class, the extern \"C\" entry function, and any necessary includes/macros; "
            "do not include a main function.\n\n"
        )
    prompt += "```cpp\n# YOUR CODE HERE\n```\n\n<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


def run_batch(
    prompts,
    model_path,
    tensor_parallel_size=1,
    n_samples=10,
    temperature=0.2,
    max_tokens=4096,
    max_model_len=4096,
    gpu_memory_utilization=0.75,
    max_num_seqs=16,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=max_model_len,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        max_num_seqs=max_num_seqs,
    )
    sampling_params = SamplingParams(
        n=n_samples,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1.0,
        frequency_penalty=0,
        presence_penalty=0,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.pad_token_id],
    )

    outputs = [None for _ in prompts]
    remaining_prompts = []
    remaining_indices = []
    for prompt_index, prompt in enumerate(prompts):
        remaining_prompts.append(prompt)
        remaining_indices.append(prompt_index)

    if remaining_prompts:
        vllm_outputs = llm.generate(remaining_prompts, sampling_params)
        for index, vllm_output in zip(remaining_indices, vllm_outputs):
            outputs[index] = [o.text for o in vllm_output.outputs]

    return outputs


def evaluate_ascend_c(
    data_path,
    model_path,
    tensor_parallel_size=1,
    n_samples=10,
    temperature=0.2,
    output_path=None,
    max_tokens=4096,
    max_model_len=4096,
    gpu_memory_utilization=0.75,
    max_num_seqs=16,
):
    benchmark = []
    with open(data_path, "r") as f:
        for line in f:
            benchmark.append(json.loads(line))

    benchmark = sorted(benchmark, key=lambda x: x.get("question_id", ""))
    prompts = [format_ascend_c_prompt(task) for task in benchmark]
    outputs = run_batch(
        prompts,
        model_path,
        tensor_parallel_size=tensor_parallel_size,
        n_samples=n_samples,
        temperature=temperature,
        max_tokens=max_tokens,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
    )

    generations = [
        [extract_code(o) for o in outputs_list] for outputs_list in outputs
    ]

    metrics = ascend_c_metrics(
        benchmark,
        generations,
        k_list=[1, 5, 10],
        num_process_evaluate=4,
        build_timeout=120,
        run_timeout=60,
        debug=False,
    )

    compile_pass = metrics["compile"]
    functional_pass = metrics["functional"]

    print(f"compile pass@1 = {compile_pass.get('pass@1', 'N/A')}")
    print(f"compile pass@5 = {compile_pass.get('pass@5', 'N/A')}")
    print(f"compile pass@10 = {compile_pass.get('pass@10', 'N/A')}")
    if functional_pass:
        print(f"functional pass@1 = {functional_pass.get('pass@1', 'N/A')}")
        print(f"functional pass@5 = {functional_pass.get('pass@5', 'N/A')}")
        print(f"functional pass@10 = {functional_pass.get('pass@10', 'N/A')}")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(
                {
                    "compile": compile_pass,
                    "functional": functional_pass,
                    "functional_compile": metrics["functional_compile"],
                },
                f,
                indent=4,
            )

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.75)
    parser.add_argument("--max_num_seqs", type=int, default=16)
    args = parser.parse_args()

    evaluate_ascend_c(
        data_path=args.data_path,
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        n_samples=args.n_samples,
        temperature=args.temperature,
        output_path=args.output_path,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
    )
