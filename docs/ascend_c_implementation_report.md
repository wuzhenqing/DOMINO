# DOMINO for Ascend C — Smoke-Run Implementation Report

**Date:** 2026-09-05  
**Scope:** end-to-end integration smoke run of the Ascend C adaptation on a single RTX 5060 8 GB GPU + CANN 9.2.0 CPU-debug host (no NPU).  
**Goal:** verify that the new `domino/pipeline/ascend_c/` and `domino/evaluation/ascend_c/` packages chain together correctly, that the compiler-anchored quality gate works, and that the Ascend C eval harness can produce separate compile/functional pass@k metrics.

---

## 1. What was built

### 1.1 New pipeline package (`domino/pipeline/ascend_c/`)

| File | Purpose |
|------|---------|
| `quality_assessment.py` | LLM grades synthetic Ascend C operator instructions on the standard 5-point scale; outputs `*_instruct_quality.jsonl`. |
| `generate_response.py` | Keeps instructions above a configurable quality threshold and generates `.asc`-style kernel responses with a ```cpp fence; outputs `*_excellent_response.jsonl`. |
| `compile_check.py` | Replaces the Python `tree-sitter` stage. Wraps each response in a temporary CANN CPU-debug CMake project (single-file `.asc`), runs `cmake --build`, and maps the result to `response_syntex` ∈ {right, wrong, unknown}. Now also stores the full CANN build log in `compile_log` for verification. |
| `format_for_sft.py` | Converts the filtered pairs into LLaMA-Factory Alpaca-style JSON. |
| `__init__.py` | Package marker. |

### 1.2 New evaluation harness (`domino/evaluation/ascend_c/`)

| File | Purpose |
|------|---------|
| `evaluate.py` | vLLM-based evaluator. Prompts the model for an Ascend C kernel, extracts the first ``` fence, and delegates scoring to `ascend_c_metrics.py`. |
| `ascend_c_metrics.py` | Per-sample CANN CPU-debug build + optional run against a golden `main()`. Reports separate **compile** pass@k and **functional** pass@k. Reuses `domino/evaluation/code_generation/pass_k.py` for the unbiased estimator. |

### 1.3 Driver script

- `scripts/pipeline_ascend_c.sh` — example shell driver (generation → instruction quality → response generation). Compile check and SFT formatting are included as commented optional next steps; in this smoke run they were invoked directly (see §2).

### 1.4 CANN CPU-debug example project

- `data/cann_cpu_debug_example/` — self-contained `AddCustom` CPU-debug build with the glibc `_Float128` shim, used by `compile_check.py` and `ascend_c_metrics.py`.

---

## 2. How to run the smoke pipeline

All commands are run from the repository root after activating the `domino` conda environment.

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate domino
unset all_proxy ALL_PROXY
export HF_HUB_OFFLINE=1
```

### 2.1 Synthetic instruction grading (all 500)

```bash
python -m domino.pipeline.ascend_c.quality_assessment \
    --process_stage quality_assessment \
    --synthetic_instruct_path outputs/contrastive_ascend_c/vllm_generated_500_samples_temp0.8.jsonl \
    --GRADE_MODEL Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --batch_size 10 --devices 0 \
    --max_tokens 1024 --gpu_memory_utilization 0.8
```

### 2.2 Response generation (capped to ~50)

Because the 0.5B soft-token model produced no `excellent` instructions, the threshold was widened to `average good excellent` and capped with `--max_count 50`:

```bash
python -m domino.pipeline.ascend_c.generate_response \
    --process_stage inference_response \
    --synthetic_instruct_path outputs/contrastive_ascend_c/vllm_generated_500_samples_temp0.8_instruct_quality.jsonl \
    --inference_model Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --batch_size 5 --devices 0 \
    --max_count 50 --quality_threshold average good excellent \
    --max_tokens 2048 --gpu_memory_utilization 0.8
```

### 2.3 Compiler-anchored check

```bash
python -m domino.pipeline.ascend_c.compile_check \
    --process_stage compile_check \
    --instruct_response_path outputs/contrastive_ascend_c/vllm_generated_500_samples_temp0.8_instruct_quality_excellent_response.jsonl \
    --num_workers 4 --timeout 120
```

### 2.4 Pair selection

`domino/pipeline/code_generation/make_pairs.py` expects both a `_llm_quality.jsonl` (response LLM grade) and a `_syntex_quality.jsonl` (compile verdict). The Ascend C pipeline has **no response LLM-grading stage by design**; the compiler is the anchor. To reuse `make_pairs.py` without changing its filter logic, a stub `_llm_quality.jsonl` was generated from the compile output, setting `response_quality` to `"good"` for `response_syntex == "right"` records and `"poor"` otherwise:

```bash
python - <<'PY'
import json
syntex_path = 'outputs/contrastive_ascend_c/vllm_generated_500_samples_temp0.8_instruct_quality_excellent_response_syntex_quality.jsonl'
llm_path    = 'outputs/contrastive_ascend_c/vllm_generated_500_samples_temp0.8_instruct_quality_excellent_response_llm_quality.jsonl'
with open(syntex_path) as f, open(llm_path, 'w') as g:
    for line in f:
        item = json.loads(line)
        q = 'good' if item.get('response_syntex') == 'right' else 'poor'
        item['response_quality'] = json.dumps({'explanation': 'compiler-anchored stub', 'solution_quality': q})
        g.write(json.dumps(item) + '\n')
PY

python -m domino.pipeline.code_generation.make_pairs \
    --process_stage make-pairs \
    --instruct_response_llm_check_path outputs/contrastive_ascend_c/vllm_generated_500_samples_temp0.8_instruct_quality_excellent_response_llm_quality.jsonl \
    --instruct_response_syntex_check_path outputs/contrastive_ascend_c/vllm_generated_500_samples_temp0.8_instruct_quality_excellent_response_syntex_quality.jsonl
```

### 2.5 SFT formatting

```bash
python domino/pipeline/ascend_c/format_for_sft.py \
    outputs/contrastive_ascend_c/vllm_generated_500_samples_temp0.8_instruct_quality_excellent_response_quality_filtered.jsonl
```

### 2.6 Eval-harness smoke

```bash
python -m domino.evaluation.ascend_c.evaluate \
    --data_path data/ascend_c_eval.jsonl \
    --model_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --tensor_parallel_size 1 --n_samples 2 --temperature 0.2 \
    --output_path outputs/contrastive_ascend_c/ascend_c_eval_metrics.json \
    --max_tokens 2048 --max_model_len 4096 \
    --gpu_memory_utilization 0.75 --max_num_seqs 8
```

---

## 3. Deviations from `docs/ascend_c_adaptation_analysis.md`

The implementation faithfully follows the analysis except where constrained by the smoke-run hardware and data.

| Analysis recommendation | Smoke-run reality | Rationale / impact |
|-------------------------|-------------------|--------------------|
| Base model `Qwen2.5-Coder-1.5B/7B-Instruct` for training | **`Qwen/Qwen2.5-Coder-0.5B-Instruct`** used for soft-token training | Faster training on the RTX 5060; quality is expectedly weak. |
| Soft tokens 256 public / 256 private, `max_seq_len 2048` | **256 public / 64 private**, `max_seq_len 1024` (training default) | Reduced to fit the 0.5B model and the smoke GPU. Public table shape is `256 × 896`; private table is `186 × 57344` (i.e. 64 tokens × 896 hidden). |
| Generation on Ascend 910B NPU | **GPU vLLM generation** (`cuda:0`) | No NPU available on the smoke machine. |
| In-house Ascend C corpus | **Public gitee corpus** (`gitee.com/ascend/samples`) used to build the reference JSONL | In-house corpus was not available for this run. |
| Python `pyext` sandbox | **Not used** for Ascend C; replaced by CANN CPU-debug compile/run | Correct by design for Ascend C. |
| CANN `float128` shim | **Bundled in `data/cann_cpu_debug_example/shim/`** and passed via `-isystem` | Required on glibc ≥ 2.41 hosts; documented in `docs/cann_cpu_debug_recipe.md`. |
| vLLM tied-embedding handling | **Added a fix in `domino/contrastive/generate.py`** (lines 104–121) that materializes a separate `lm_head` and detaches it from the input embedding after soft-token injection | Qwen2.5 ties input/output embeddings; without this, resizing the input embedding corrupts the original lm_head distribution. |
| Response quality decided by LLM grader | **Compiler is the anchor**; `make_pairs` is fed by a compiler-derived `response_quality` stub | Matches the analysis’s principle of anchoring filtering on the compiler rather than a general LLM grader. |
| `--batch_size 200`, default vLLM memory settings | **Batch sizes lowered to 10/5**, `enforce_eager=True`, `gpu_memory_utilization` reduced to 0.8/0.75 | Needed to avoid OOM on an 8 GB consumer GPU under vLLM 0.24.0. |

---

## 4. Verification evidence

### 4.1 Training

- Soft-token checkpoints written: `outputs/contrastive_ascend_c/public_soft_token_embeddings.pth` and `private_soft_token_embeddings.pth`.
- Final log entry: `train_loss = 1.9766`, `epoch = 3.0`, `step = 279`, runtime ≈ 185 s.
- Model generated 500 non-empty synthetic records (`vllm_generated_500_samples_temp0.8.jsonl`).

### 4.2 Pipeline per-stage record counts

| Stage | Input records | Output records | Notes |
|-------|---------------|----------------|-------|
| Synthetic generation | — | **500** | Existing artifact from the 0.5B soft-token run. |
| Instruction quality | 500 | **500** | Distribution: 358 very poor, 130 average, 2 good, 9 parse-failed, 1 placeholder; **0 excellent**. |
| Response generation | 500 (filtered/capped) | **50** | Threshold widened to `average+good+excellent` and capped at 50 because no excellent records existed. |
| Compile check | 50 | **50** | Verdicts: **2 right**, **43 wrong**, **5 unknown**. |
| make_pairs | 50 LLM stub + 50 syntex | **2** | Only the two compile-passing records survive. |
| format_for_sft | 2 | **2** | `*_sft_format.json` written. |

### 4.3 Criterion (c): label-vs-log inspection (sample of 12)

`compile_check.py` now stores the full CANN configure/build log in each record’s `compile_log` field. Manual inspection confirms the labels match the compiler verdict:

| idx | `response_syntex` | configure rc | build rc | Evidence from `compile_log` |
|-----|-------------------|--------------|----------|----------------------------|
| 178 | right | 0 | 0 | `[100%] Built target kernel`; binary exists. |
| 184 | right | 0 | 0 | `[100%] Built target kernel`; binary exists. |
| 1 | wrong | 0 | 2 | `error: expected class name` (10 compiler errors). |
| 4 | wrong | 0 | 2 | `error: use of undeclared identifier 'get_global_id'`. |
| 6 | wrong | 0 | 2 | `error: expected unqualified-id`. |
| 8 | wrong | 0 | 2 | `error: expected class name`. |
| 12 | wrong | 0 | 2 | `error: 'ascendc_api.h' file not found`. |
| 14 | wrong | 0 | 2 | `error: 'ascendc/StandardFunction/Kernel/...' file not found`. |
| 21 | wrong | 0 | 2 | `error: 'ascend_c_api.h' file not found`. |
| 27 | unknown | N/A | N/A | No ``` fence extractable; no build attempted. |
| 40 | unknown | N/A | N/A | No valid ``` fence extractable; no build attempted. |
| 138 | unknown | N/A | N/A | No valid ``` fence extractable; no build attempted. |

The two `right` records genuinely compiled to a binary; the `wrong` records failed with real CANN/bisheng compiler errors; the `unknown` records never reached the compiler because the response did not contain a parseable code fence.

### 4.4 make_pairs outcome

- `_quality_filtered.jsonl` is **non-empty** (2 records).
- The compiler did not reject everything: it accepted 2 of 50 attempted responses; the other 48 failed at code extraction or CANN build.

### 4.5 Eval-harness smoke

- `python -m domino.evaluation.ascend_c.evaluate` completed end-to-end on `data/ascend_c_eval.jsonl` (5 tasks, 1 runnable) using vLLM.
- Metrics written to `outputs/contrastive_ascend_c/ascend_c_eval_metrics.json`:

```json
{
    "compile": {"pass@1": 0.0},
    "functional": {"pass@1": 0.0},
    "functional_compile": {"pass@1": 0.0}
}
```

Zero pass@1 is expected for a zero-shot 1.5B coder model on Ascend C; the smoke run’s purpose was to prove the harness path works, not to achieve high accuracy.

---

## 5. Minimal code changes made during integration

1. `domino/pipeline/ascend_c/quality_assessment.py`
   - Added `enforce_eager=True` to vLLM init.
   - Added `--max_tokens` and `--gpu_memory_utilization` CLI args (defaults 1024 / 0.85) so the stage fits an 8 GB GPU.

2. `domino/pipeline/ascend_c/generate_response.py`
   - Added `enforce_eager=True`.
   - Added `--max_count`, `--max_tokens`, `--gpu_memory_utilization`, and `--quality_threshold` CLI args. The last one is the smoke-run workaround for the absence of `excellent`-rated instructions.

3. `domino/pipeline/ascend_c/compile_check.py`
   - `build_kernel()` now returns `(verdict, log)` and stores the full CANN configure/build output in `compile_log`.

4. `domino/evaluation/ascend_c/evaluate.py`
   - Added `--max_tokens`, `--max_model_len`, `--gpu_memory_utilization`, and `--max_num_seqs` CLI args to avoid OOM on 8 GB GPUs.

No git commit was made.

---

## 6. Next steps for the real A100 + 910B environment

1. **Switch training to `Qwen2.5-Coder-7B-Instruct`** on the A100 node with the full 256 public / 256 private token budget and `max_seq_len 2048`. Use the existing DeepSpeed ZeRO-2 config in `configs/ds_z2.json`.
2. **Generate on the 910B NPU** as the Stage-1 contribution. First try the `vllm-ascend` plugin loading the `vllm_temp_model` checkpoint produced by `domino/contrastive/generate.py`; if that fails, fall back to the transformers engine with `torch_npu`, device `npu:0`, and SDPA / NPU fused attention instead of FlashAttention-2.
3. **Scale the pipeline** without the 8 GB compromises: restore `--batch_size 200`, remove `enforce_eager`, and raise `max_tokens` to 4096+ for long kernels.
4. **Drop the `quality_threshold` workaround** in `generate_response.py`: when the 7B model produces `excellent` instructions, revert the default threshold to `{'excellent'}`.
5. **Drop the `make_pairs` stub** once a real response LLM-grader is justified, or keep the compiler-anchor design and formalize the stub as a permanent adapter (e.g., `domino/pipeline/ascend_c/make_pairs_adapter.py`).
6. **Add CPU-debug run verification** in `ascend_c_metrics.py`: currently only the single runnable benchmark task executes a golden `main()`; on the real corpus, every compile-passing kernel should be linked against a generated or golden host harness and run in CPU-debug mode.
7. **Confirm the 910B CANN version** supports CPU-debug (`asc-tools`, ≥ 9.0) and the exact single-file `.asc` compile flow; adjust `ASC_ARCH` in `ascend_c_metrics.py` if the target is not `dav-2201`.
8. **Run the full data flywheel on 910B** (Stage 2): port training to `torch_npu` + HF Trainer, generate with the verified NPU path, compile-check on CANN, and measure end-to-end throughput versus the A100 baseline.

---

## 7. Artifact paths

- Synthetic instructions: `outputs/contrastive_ascend_c/vllm_generated_500_samples_temp0.8.jsonl`
- Instruct quality: `outputs/contrastive_ascend_c/vllm_generated_500_samples_temp0.8_instruct_quality.jsonl`
- Responses: `outputs/contrastive_ascend_c/vllm_generated_500_samples_temp0.8_instruct_quality_excellent_response.jsonl`
- Compile verdicts + logs: `outputs/contrastive_ascend_c/vllm_generated_500_samples_temp0.8_instruct_quality_excellent_response_syntex_quality.jsonl`
- Filtered pairs: `outputs/contrastive_ascend_c/vllm_generated_500_samples_temp0.8_instruct_quality_excellent_response_quality_filtered.jsonl`
- SFT format: `outputs/contrastive_ascend_c/vllm_generated_500_samples_temp0.8_instruct_quality_excellent_response_quality_filtered_sft_format.json`
- Eval metrics: `outputs/contrastive_ascend_c/ascend_c_eval_metrics.json`
- Training log: `outputs/contrastive_ascend_c/training.log`
- Report: `docs/ascend_c_implementation_report.md`
