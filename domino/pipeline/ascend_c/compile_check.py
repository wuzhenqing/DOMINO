import os
import re
import json
import shutil
import argparse
import tempfile
import subprocess
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


EXAMPLE_PROJECT_DIR = Path(__file__).resolve().parents[3] / "data" / "cann_cpu_debug_example"
CANN_SET_ENV = "/usr/local/Ascend/ascend-toolkit/set_env.sh"
CMAKE_TIMEOUT = 300


def extract_ascend_c_code(response):
    """Extract the first ```cpp or ``` fenced Ascend C code block."""
    pattern = r"```(?:cpp|c\+\+)?\s*\n(.*?)\n```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def has_main_function(code):
    return re.search(r"\bint\s+main\s*\(", code) is not None


def prepare_kernel_source(generated_code):
    header = '''#include <cstdint>
#include "acl/acl.h"
#ifdef ASCENDC_CPU_DEBUG
#include "cpu_debug_launch.h"
#endif
#include "kernel_operator.h"
#include "add_custom_tiling.h"

'''
    source = header + generated_code
    if not has_main_function(generated_code):
        source += "\n\nint main() { return 0; }\n"
    return source


def build_kernel(generated_code, timeout=CMAKE_TIMEOUT):
    """
    Compile the generated Ascend C kernel in a temporary CPU-debug project.
    Returns (verdict, log) where verdict is one of {'right', 'wrong', 'unknown'}.
    """
    if generated_code is None or not generated_code.strip():
        return "unknown", ""

    example_dir = EXAMPLE_PROJECT_DIR
    if not example_dir.exists():
        return "unknown", ""

    temp_dir = tempfile.mkdtemp(prefix="ascend_c_compile_")
    log = ""
    try:
        src_dir = Path(temp_dir)
        build_dir = src_dir / "build"

        shutil.copy2(example_dir / "add_custom_tiling.h", src_dir / "add_custom_tiling.h")
        shutil.copytree(example_dir / "shim", src_dir / "shim")

        kernel_source = prepare_kernel_source(generated_code)
        with open(src_dir / "kernel.asc", "w") as f:
            f.write(kernel_source)

        cmake_lists = '''cmake_minimum_required(VERSION 3.16.0)
find_package(ASC REQUIRED)
project(ascend_c_kernel LANGUAGES ASC CXX)
add_executable(kernel kernel.asc)
target_compile_options(kernel PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:-isystem${CMAKE_CURRENT_SOURCE_DIR}/shim>
    $<$<COMPILE_LANGUAGE:ASC>:-DDTYPE_X=float>
    $<$<COMPILE_LANGUAGE:ASC>:-DDTYPE_Y=float>
    $<$<COMPILE_LANGUAGE:ASC>:-DDTYPE_Z=float>
)
'''
        with open(src_dir / "CMakeLists.txt", "w") as f:
            f.write(cmake_lists)

        configure_cmd = (
            f"source {CANN_SET_ENV} && "
            f"cmake -B {build_dir} -S {src_dir} "
            f"-DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201"
        )
        build_cmd = f"source {CANN_SET_ENV} && cmake --build {build_dir} -j4"

        try:
            config_res = subprocess.run(
                configure_cmd,
                shell=True,
                executable="/bin/bash",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
            )
            log += f"=== CONFIGURE (rc={config_res.returncode}) ===\n{config_res.stdout}\n"
            if config_res.returncode != 0:
                return "wrong", log

            build_res = subprocess.run(
                build_cmd,
                shell=True,
                executable="/bin/bash",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
            )
            log += f"=== BUILD (rc={build_res.returncode}) ===\n{build_res.stdout}\n"
            if build_res.returncode == 0 and (build_dir / "kernel").exists():
                return "right", log
            return "wrong", log
        except subprocess.TimeoutExpired:
            log += "=== TIMEOUT ===\n"
            return "unknown", log
        except Exception as e:
            log += f"=== EXCEPTION ===\n{repr(e)}\n"
            return "unknown", log
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def check_one(args):
    item, timeout = args
    generated_code = extract_ascend_c_code(item.get("original_response", ""))
    verdict, log = build_kernel(generated_code, timeout=timeout)
    item["response_syntex"] = verdict
    item["compile_log"] = log
    return item


def compile_check(instruct_response_path, num_workers=4, timeout=CMAKE_TIMEOUT):
    instruct_response_dataset = []
    with open(instruct_response_path, 'r') as f:
        for line in f:
            instruct_response_dataset.append(json.loads(line))
    print(f"instruct & response dataset length = {len(instruct_response_dataset)}")

    syntex_correct_count = 0
    syntex_wrong_count = 0
    syntex_unknown_count = 0

    results = [None] * len(instruct_response_dataset)
    args_list = [(item, timeout) for item in instruct_response_dataset]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {
            executor.submit(check_one, args): idx
            for idx, args in enumerate(args_list)
        }
        for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx)):
            idx = future_to_idx[future]
            try:
                item = future.result(timeout=timeout + 30)
            except Exception:
                item = instruct_response_dataset[idx].copy()
                item["response_syntex"] = "unknown"

            verdict = item.get("response_syntex", "unknown")
            if verdict == "right":
                syntex_correct_count += 1
            elif verdict == "wrong":
                syntex_wrong_count += 1
            else:
                syntex_unknown_count += 1
            results[idx] = item

    print(f"syntex correct count = {syntex_correct_count}")
    print(f"syntex wrong count = {syntex_wrong_count}")
    print(f"syntex unknown count = {syntex_unknown_count}")

    saved_path = instruct_response_path.replace(".jsonl", "_syntex_quality.jsonl")
    with open(saved_path, 'w') as g:
        for item in results:
            g.write(json.dumps(item) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--process_stage", type=str, default='compile_check', choices=["compile_check"])
    parser.add_argument("--instruct_response_path", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=CMAKE_TIMEOUT)
    args = parser.parse_args()

    if args.process_stage == "compile_check":
        compile_check(
            instruct_response_path=args.instruct_response_path,
            num_workers=args.num_workers,
            timeout=args.timeout
        )
    else:
        print("Check process stage.")
