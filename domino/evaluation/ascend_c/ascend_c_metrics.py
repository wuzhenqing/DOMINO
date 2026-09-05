import json
import multiprocessing
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

from tqdm import tqdm

from domino.evaluation.code_generation.pass_k import compute_metrics_from_results


CANN_SET_ENV = "/usr/local/Ascend/ascend-toolkit/set_env.sh"
ASC_ARCH = "dav-2201"


def extract_code(model_output: str) -> str:
    outputlines = model_output.split("\n")
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[0] + 1 : indexlines[1]])


def _strip_main_block(source: str) -> str:
    pattern = re.compile(r"\b(?:int|int32_t|int16_t|int8_t)\s+main\s*\(")
    m = pattern.search(source)
    if not m:
        return source
    start = m.start()
    brace_start = source.find("{", m.end())
    if brace_start == -1:
        return source[:start]
    depth = 1
    i = brace_start + 1
    while i < len(source) and depth > 0:
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
        i += 1
    return source[:start] + source[i:]


def _cmake_target_name(name: str) -> str:
    sanitized = re.sub(r"[^0-9a-zA-Z_]+", "_", name)
    sanitized = re.sub(r"^[^a-zA-Z_]", "_", sanitized)
    return sanitized or "kernel"


def _write_project(
    project_dir: str,
    kernel_source: str,
    task: dict,
):
    op_name = task.get("operator_name", "kernel")
    target = _cmake_target_name(op_name)
    asc_file = f"{target}.asc"

    # Write kernel source.
    with open(os.path.join(project_dir, asc_file), "w") as f:
        f.write(kernel_source)

    # Write tiling header if provided.
    tiling_name = task.get("tiling_header_name")
    tiling_content = task.get("tiling_header")
    if tiling_name and tiling_content:
        with open(os.path.join(project_dir, tiling_name), "w") as f:
            f.write(tiling_content)

    # Write float128 shim.
    shim_dir = os.path.join(project_dir, "shim", "bits")
    os.makedirs(shim_dir, exist_ok=True)
    with open(os.path.join(shim_dir, "floatn.h"), "w") as f:
        f.write(
            "#ifndef _BITS_FLOATN_H\n"
            "#define _BITS_FLOATN_H\n"
            "#define __HAVE_FLOAT128 0\n"
            "#define __HAVE_DISTINCT_FLOAT128 0\n"
            "#endif\n"
        )

    # Write CMakeLists.txt.
    dtype_flags = task.get("dtype_flags", [])
    options_lines = [
        "    $<$<COMPILE_LANGUAGE:ASC>:-isystem${CMAKE_CURRENT_SOURCE_DIR}/shim>",
    ]
    for flag in dtype_flags:
        flag = flag.strip()
        if flag:
            options_lines.append(f"    $<$<COMPILE_LANGUAGE:ASC>:-D{flag}>")

    cmake = (
        "cmake_minimum_required(VERSION 3.16.0)\n"
        "\n"
        "find_package(ASC REQUIRED)\n"
        "\n"
        f"project({target}_proj LANGUAGES ASC CXX)\n"
        "\n"
        f"add_executable({target} {asc_file})\n"
        "\n"
        f"target_compile_options({target} PRIVATE\n"
        + "\n".join(options_lines)
        + "\n)"
    )
    with open(os.path.join(project_dir, "CMakeLists.txt"), "w") as f:
        f.write(cmake)


def _run_shell(command: str, cwd: str, timeout: int) -> Tuple[int, str, str]:
    proc = subprocess.run(
        command,
        cwd=cwd,
        shell=True,
        executable="/bin/bash",
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _score_single(
    task: dict,
    generation: str,
    build_timeout: int = 120,
    run_timeout: int = 60,
    debug: bool = False,
) -> Tuple[int, int, dict]:
    metadata = {
        "compile_ok": 0,
        "run_ok": 0,
        "compile_returncode": None,
        "run_returncode": None,
        "compile_stdout": "",
        "compile_stderr": "",
        "run_stdout": "",
        "run_stderr": "",
        "error": None,
    }

    kernel_source = extract_code(generation)
    if not kernel_source.strip():
        metadata["error"] = "No code extracted from model output"
        return 0, 0, metadata

    attached_main = task.get("main", "")
    if attached_main.strip():
        kernel_source = _strip_main_block(kernel_source)
        kernel_source = kernel_source.rstrip() + "\n\n" + attached_main.strip() + "\n"

    project_dir = tempfile.mkdtemp(prefix="ascend_c_eval_")
    try:
        _write_project(project_dir, kernel_source, task)
        target = _cmake_target_name(task.get("operator_name", "kernel"))

        configure_cmd = (
            f"source {CANN_SET_ENV} && "
            f"cmake -B build -S . "
            f"-DCMAKE_ASC_RUN_MODE=cpu "
            f"-DCMAKE_ASC_ARCHITECTURES={ASC_ARCH}"
        )
        build_cmd = f"source {CANN_SET_ENV} && cmake --build build -j4"

        if debug:
            print(f"[ascend_c eval] configuring in {project_dir}")
        rc, out, err = _run_shell(configure_cmd, project_dir, build_timeout)
        if rc != 0:
            metadata.update(
                {
                    "compile_returncode": rc,
                    "compile_stdout": out,
                    "compile_stderr": err,
                    "error": "CMake configuration failed",
                }
            )
            return 0, 0, metadata

        if debug:
            print(f"[ascend_c eval] building in {project_dir}")
        rc, out, err = _run_shell(build_cmd, project_dir, build_timeout)
        metadata.update(
            {
                "compile_returncode": rc,
                "compile_stdout": out,
                "compile_stderr": err,
            }
        )
        if rc != 0:
            metadata["error"] = "Build failed"
            return 0, 0, metadata

        binary_path = os.path.join(project_dir, "build", target)
        if not os.path.exists(binary_path):
            metadata["error"] = "Build succeeded but binary not found"
            return 0, 0, metadata

        metadata["compile_ok"] = 1

        if not attached_main.strip():
            return 1, 0, metadata

        if debug:
            print(f"[ascend_c eval] running {binary_path}")
        rc, out, err = _run_shell(
            f"source {CANN_SET_ENV} && ./{target}",
            os.path.join(project_dir, "build"),
            run_timeout,
        )
        metadata.update(
            {
                "run_returncode": rc,
                "run_stdout": out,
                "run_stderr": err,
            }
        )

        pass_marker = task.get("pass_marker", "PASS")
        fail_marker = task.get("fail_marker", "FAIL")
        run_ok = rc == 0 and pass_marker in out and fail_marker not in out
        if rc == 0 and pass_marker not in out and fail_marker not in out:
            run_ok = True  # accept clean exit if no explicit markers are required

        metadata["run_ok"] = int(run_ok)
        if not run_ok:
            metadata["error"] = "Run did not produce expected PASS output"

        return 1, int(run_ok), metadata

    except subprocess.TimeoutExpired as e:
        metadata["error"] = f"Timeout during {e.cmd if hasattr(e, 'cmd') else 'build/run'}"
        return metadata["compile_ok"], metadata["run_ok"], metadata
    except Exception as e:
        metadata["error"] = repr(e)
        return 0, 0, metadata
    finally:
        shutil.rmtree(project_dir, ignore_errors=True)


def _evaluate_generations_by_problem(args):
    problem_generations, sample, debug, build_timeout, run_timeout = args
    compile_results = []
    run_results = []
    metadata_list = []

    for o_idx, generation in enumerate(problem_generations):
        try:
            compile_ok, run_ok, metadata = _score_single(
                sample,
                generation,
                build_timeout=build_timeout,
                run_timeout=run_timeout,
                debug=debug,
            )
            if debug:
                print(f"\nTask {sample.get('question_id', o_idx)} generation {o_idx}: compile={compile_ok} run={run_ok}")
        except Exception as e:
            if debug:
                print(f"\nTask {sample.get('question_id', o_idx)} generation {o_idx} exception: {repr(e)}")
            compile_ok, run_ok = 0, 0
            metadata = {"error": repr(e)}
        compile_results.append(compile_ok)
        run_results.append(run_ok)
        metadata_list.append(metadata)

    return compile_results, run_results, metadata_list


def evaluate_generations(
    samples_list: List[dict],
    generations_list: List[List[str]],
    debug: bool = False,
    num_process_evaluate: int = 4,
    build_timeout: int = 120,
    run_timeout: int = 60,
):
    inputs = [
        (
            generations_list[idx],
            samples_list[idx],
            debug,
            build_timeout,
            run_timeout,
        )
        for idx in range(len(generations_list))
    ]

    results_compile: Dict[int, List[int]] = {}
    results_run: Dict[int, List[int]] = {}
    metadata: Dict[int, List[dict]] = {}

    with tqdm(total=len(inputs)) as pbar:
        with ProcessPoolExecutor(
            max_workers=1 if debug else num_process_evaluate
        ) as executor:
            futures = {
                executor.submit(_evaluate_generations_by_problem, arg): index
                for index, arg in enumerate(inputs)
            }
            for future in as_completed(futures):
                index = futures[future]
                compile_res, run_res, meta = future.result()
                results_compile[index] = compile_res
                results_run[index] = run_res
                metadata[index] = meta
                pbar.update(1)

    return results_compile, results_run, metadata


def ascend_c_metrics(
    samples_list: List[dict],
    generations_list: List[List[str]],
    k_list=None,
    num_process_evaluate: int = 4,
    build_timeout: int = 120,
    run_timeout: int = 60,
    debug: bool = False,
):
    if k_list is None:
        k_list = [1, 5, 10]

    results_compile, results_run, metadata = evaluate_generations(
        samples_list,
        generations_list,
        debug=debug,
        num_process_evaluate=num_process_evaluate,
        build_timeout=build_timeout,
        run_timeout=run_timeout,
    )

    compile_metrics = compute_metrics_from_results(results_compile, k_list=k_list)

    runnable_indices = [
        i for i, s in enumerate(samples_list) if s.get("main", "").strip()
    ]
    if runnable_indices:
        runnable_compile = {i: results_compile[i] for i in runnable_indices}
        runnable_run = {i: results_run[i] for i in runnable_indices}
        functional_compile_metrics = compute_metrics_from_results(
            runnable_compile, k_list=k_list
        )
        functional_metrics = compute_metrics_from_results(runnable_run, k_list=k_list)
    else:
        functional_compile_metrics = {}
        functional_metrics = {}

    return {
        "compile": compile_metrics,
        "functional": functional_metrics,
        "functional_compile": functional_compile_metrics,
        "results_compile": results_compile,
        "results_run": results_run,
        "metadata": metadata,
    }
