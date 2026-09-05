import os
import json


def format_ascend_c_for_sft(path):
    datasets = []
    with open(path, 'r') as f:
        for line in f:
            datasets.append(json.loads(line))

    print(f"length of datasets = {len(datasets)}")

    system = (
        "You are an expert Ascend C kernel engineer. "
        "Implement the described operator as a single-file Ascend C kernel "
        "with a kernel class (Init/Process) and an `extern \"C\" __global__ __aicore__` entry function. "
        "Enclose the complete kernel code within ```cpp and ``` fences."
    )

    formatted_samples = []
    for item in datasets:
        instruction = item.get('synthetic_text', '')
        output = item.get('original_response', '')

        point = {
            "instruction": instruction,
            "input": "",
            "output": output,
            "system": system
        }

        formatted_samples.append(point)

    output_path = path.replace(".jsonl", "_sft_format.json")
    with open(output_path, 'w') as g:
        json.dump(formatted_samples, g, indent=4)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        format_ascend_c_for_sft(sys.argv[1])
