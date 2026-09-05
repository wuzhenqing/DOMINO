# DOMINO for Ascend C: Project Analysis and Adaptation Roadmap

This document analyzes the DOMINO codebase and lays out a roadmap for using it to
synthesize domain-specific training data for **Huawei Ascend C** (the C/C++-derived
kernel programming language for Ascend NPUs).

**Confirmed constraints** (from the project owner):

- Hardware: NVIDIA A100 GPU node + Ascend 910B NPU node.
- Two-stage strategy: Stage 1 = train on GPU, generate data on the Ascend NPU;
  Stage 2 = migrate training to the NPU too, proving the Ascend platform can be part
  of its own data flywheel (the intended academic contribution).
- Reference data: Ascend C documentation and many Ascend C operator implementations
  are already available in-house.
- Target artifact: kernel-side code only at first (CANN now supports single-file
  `.asc` operator implementations); full operator projects (host + tiling +
  packaging) come later, after an SFT'd model exists.

## 1. What DOMINO is

DOMINO (Domain-Specific Data Synthesis for LLMs via Minimal Sufficient
Representation Learning) learns a compact representation of a target domain from a
**few reference examples**, then uses it to synthesize large amounts of novel
in-domain training data with a frozen base LLM. The shipped pipelines target Python
code generation (problem → solution) and code execution (function + input →
predicted output), with `Qwen/Qwen2.5-Coder-7B-Instruct` as the reference base model.

The method trains two kinds of continuous soft tokens on top of the frozen LLM:

- **Public (domain-level) soft tokens** — one shared
  `nn.Embedding(public_count, hidden)` table capturing generalizable domain patterns
  (`domino/contrastive/model.py:22-24`).
- **Private (sample-level) soft tokens** — one
  `nn.Embedding(domain_samples, private_count * hidden)` table with a unique row per
  reference sample (`domino/contrastive/model.py:25-26`).

## 2. How the core method works (`domino/contrastive/`)

### 2.1 Model and loss (`model.py`)

- The base LLM is frozen (`model.py:14-15`); typically <1% of parameters train.
- Each forward pass runs the frozen LM **three ways** (`model.py:37-64`):
  1. public tokens + sample embeddings → `loss_public_only`
  2. public + the sample's **own** private tokens → `loss_public_private`
  3. public + **every other** sample's private tokens (cross-sample) → per-sample CE
     matrix `denominator_matrix[i,j]`
- Contrastive loss (`model.py:118-124`):

  ```python
  numerator   = loss_public_only + loss_public_private
  denominator = mean of off-diagonal exp(-denominator_matrix)
  loss        = numerator + log(denominator)
  ```

  Intuition: public tokens must reconstruct *every* reference (numerator), while
  private tokens must help *only their own* sample (the log of the cross-sample
  denominator pushes private information out of the public tokens). At synthesis time
  only public tokens are used, so generation produces novel in-domain samples rather
  than memorized copies.
- Soft-token positions get attention mask 1 and labels `-100`; only real text is
  supervised (`model.py:70-101`).
- Note: `loss_public_only`/`loss_public_private` are batch-mean scalars from HF while
  the denominator is a per-sample mean — the terms are not strictly dimensionally
  homogeneous. Worth remembering if Ascend C training looks unstable.

### 2.2 Training data and training (`dataset.py`, `train.py`)

- Input: JSONL, one record per line; the only required field is
  **`question_content`** (`dataset.py:41-46`). Raw text is tokenized directly — **no
  chat template**.
- Defaults from `scripts/train_contrastive.sh`: 256 public + 256 private tokens, 10
  epochs, lr 1e-3 (linear), per-device batch 3, bf16, gradient checkpointing,
  DeepSpeed ZeRO-2 (`configs/ds_z2.json`), `max_seq_len 1024`.
- Only the two tables are saved: `public_soft_token_embeddings.pth` and
  `private_soft_token_embeddings.pth` (`train.py:139-152`).

### 2.3 Synthesis (`generate.py`)

- **vLLM path (default):** adds `<soft_0>…<soft_255>` as new special tokens, resizes
  the embedding matrix, writes the trained **public** embeddings into the last rows,
  and saves a temporary HF model to `<soft_prompt_dir>/vllm_temp_model/`
  (`generate.py:80-103`). vLLM then loads that temp model (`dtype=bfloat16`,
  `gpu_memory_utilization=0.95`, TP=4).
- The prompt is literally the concatenation of all soft tokens —
  `<soft_0><soft_1>…<soft_255>` (`generate.py:128`). The entire synthetic distribution
  is therefore controlled by the learned public tokens, i.e. by the reference set.
- Sampling: `max_tokens=2048, temperature=0.8, top_p=1`, batches of 200
  (`generate.py:120-134`). Output: `vllm_generated_{N}_samples_temp{T}.jsonl`,
  records `{"idx", "synthetic_text"}`.
- A slower **transformers engine** exists as fallback (`generate.py:9-66`), feeding
  `inputs_embeds` directly.

### 2.4 Baseline and utilities

- `domino/soft_prompt/` is the single-shared-soft-prompt baseline (one table, plain
  CE reconstruction, checkpoint `soft_token_embeddings.pth`); same generation
  mechanics.
- `domino/utils/embedding.py` extracts `Salesforce/codet5p-110m-embedding` embeddings
  → `.npy` for diversity/t-SNE analysis (`examples/tsne_viz.py`); `statistics.py`
  counts JSONL sizes/word counts; `query_llm.py` is a one-prompt vLLM helper.

## 3. Quality-control pipeline (`domino/pipeline/`)

Each stage reads a JSONL, appends one field, and writes a new suffixed file.
Code-generation stages:

| Stage | Module | What it does | Output suffix |
|---|---|---|---|
| Instruction grading | `code_generation/quality_assessment.py` | LLM grades each synthetic instruction on a 5-point scale (very poor…excellent), temp 0 | `_instruct_quality.jsonl` |
| Response generation | `code_generation/generate_response.py` | Keeps only `excellent`; generates solutions with a **hard-coded Python prompt** ("You will solve the code problem using Python…") | `_excellent_response.jsonl` |
| Response grading | `code_generation/filter_by_quality.py --process_stage response_llm_assessment` | LLM grades responses poor/average/good → `response_quality` | `_llm_quality.jsonl` |
| Syntax check | `filter_by_quality.py --process_stage tree-sitter` | **`tree_sitter_python`** parse of the first ``` fence; sets `response_syntex` ∈ {right, wrong, unknown} (the misspelled key is the stable data format — keep it) | `_syntex_quality.jsonl` |
| Pair selection | `code_generation/make_pairs.py` | Keeps records with `response_quality == 'good'` **and** `response_syntex == 'right'` | `_quality_filtered.jsonl` |

- Grader stop tokens are looked up in `domino/pipeline/llm_configs.json` by model
  name — add an entry per new grader model.
- The `code_execution/` pipeline mirrors this but is Python-only throughout:
  `BASE_IMPORTS` + `exec()` sandbox with a 3 s multiprocessing timeout
  (`code_execution/utils.py`), a novelty filter keyed on the Python regex
  `def\s+(\w+)\s*\(` (`filter.py`), and `check_correctness.py` is a known-incomplete
  snippet (undefined `synthetic_path`, no CLI).

## 4. Evaluation harnesses (`domino/evaluation/`)

- **Code generation:** LiveCodeBench-style. `codeproblem.py` expects
  `question_content, starter_code, public/private_test_cases, metadata.func_name`
  etc.; `evaluate.py:15-25` hard-codes a Qwen chat prompt asking for a **Python**
  program; output is extracted between the first two ``` lines (already
  language-tag-agnostic); pass@k uses the unbiased estimator in `pass_k.py:4-23`;
  execution runs generated Python in-process via `pyext.RuntimeModule` with
  `reliability_guard()` (destructive builtins disabled), SIGALRM per-test timeouts,
  mocked stdin / captured stdout, and `ProcessPoolExecutor` parallelism
  (`testing_utils.py`, `codegen_metrics.py`).
- **Code execution (output prediction):** one-shot CoT prompt with
  `[PYTHON]/[THOUGHT]/[ANSWER]` tags (`code_execution/evaluate.py:8-36`); correctness
  = `exec()` of `assert expected == generated` in a guarded subprocess; pass@1.
- **Reusable for Ascend C:** the pass@k estimator, the process-pool + timeout + kill
  scaffolding, and the fence-based code extraction. **Must be replaced:** everything
  that compiles or runs code (all Python-`exec`-based).

## 5. Operational notes

- Example hyperparameters: 80k synthetic samples at temp 0.8 for codegen
  (`scripts/pipeline_codegen.sh`), 40k at temp 0.6 for codeexe with the soft-prompt
  baseline; grading batch 200 on 4 GPUs.
- DeepSpeed configs are ZeRO-2/ZeRO-3 with `"auto"` batch/precision blocks, no
  offload.
- `requirements.txt`: torch≥2.0, transformers≥4.36, vllm≥0.4, deepspeed≥0.12,
  tree-sitter + tree-sitter-python, pyext.
- `scripts/train_contrastive.sh` and `scripts/train_soft_prompt.sh` now call
  `../domino/contrastive/train.py` and `../domino/soft_prompt/train.py`; pipeline
  scripts must still be run from the repo root.
- The MAGPIE baseline (`domino/baselines/magpie/`) requires cloning the external repo
  and patching `exp/gen_ins.py`; its prompt templates are Python-centric.

## 6. Ascend C domain brief

- **Program shape:** a kernel class with `Init()` (binds `GM_ADDR` pointers, tiling,
  allocates `TPipe`/`TQue` queues) and `Process()` (CopyIn → Compute → CopyOut loop
  using `GlobalTensor`/`LocalTensor`, `DataCopy`, vector ops like `AscendC::Add`);
  entry point `extern "C" __global__ __aicore__ void kernel(GM_ADDR…, GM_ADDR
  tiling)` with `GET_TILING_DATA`. The host side provides tiling/registration.
  Reference: the official `AddCustom` sample at
  <https://gitee.com/ascend/samples/blob/master/operator/ascendc/tutorials/AddCustomSample/FrameworkLaunch/AddCustom/op_kernel/add_custom.cpp>.
- **Build/run:** CANN toolkit (`msopgen` project generator → CMake/`build.sh` →
  `.run` package; launch via aclnn/aclop/MindSpore `ops.Custom`). Targets Ascend 910B
  (training) / 310P (inference). CANN now also supports **single-file `.asc`
  operator implementations** — the natural first target artifact; verify the exact
  `.asc` compile flow against the CANN version installed on the 910B node.
- **Data scarcity:** the public Ascend C corpus is small — on the order of ~84
  end-to-end sample projects in `gitee.com/ascend/samples/operator/ascendc`, plus
  `ascendc-api-adv` and `op-plugin`. This scarcity is exactly the regime DOMINO is
  designed for; combined with the in-house corpus, the reference-set requirement is
  comfortably met.
- **General LLMs are near-zero on Ascend C** (which motivates the whole exercise):
  - AscendKernelGen / NPUKernelBench (<https://arxiv.org/abs/2601.07160>):
    Qwen2.5-Coder-7B — L1 compile success 9.19%, L2/L3 ≈ 0.40%, functional
    correctness ≈ 0%. Their domain-adapted KernelGen-LM (Qwen3-32B base) reaches
    95.5% Pass@10 compile success on L2 kernels and 64.3% functional correctness;
    they also release the Ascend-CoT corpus (83,916 raw samples → 9,955 packed 32k
    SFT sequences).
  - MultiKernelBench (<https://arxiv.org/abs/2507.17773>): the best zero-shot AscendC
    result is DeepSeek-V3 at Pass@1 ≈ 2.5%; **category-aware one-shot prompting
    (in-domain exemplars) gives large relative gains, especially for AscendC** —
    direct external evidence that in-domain conditioning, DOMINO's core premise, pays
    off for Ascend C.
- **Validation options:** `tree-sitter-cpp` is **not** a reliable Ascend C validator
  (vendor keywords `__aicore__`/`__global__`, `GM_ADDR` macros, unresolved CANN
  headers, invisible queue/tiling semantics). Practical ground-truth checks:
  1. **CANN compile check** on the 910B host (msopgen/CMake build; inspect the
     compile log);
  2. **CPU debug mode** via `asc-tools` (CANN ≥ 9.0): builds the kernel with gcc on
     the host (`cmake -DCMAKE_ASC_RUN_MODE=cpu …; make`), runs it without consuming
     the NPU, and `npu_check` can additionally validate memory/sync/queue usage
     (<https://gitcode.com/cann/asc-tools>);
  3. Real NPU execution for final correctness (output comparison vs golden tensors).

## 7. Fitness assessment: what transfers, what must change

| Component | Status for Ascend C | Required change |
|---|---|---|
| Contrastive soft-token core (`contrastive/model.py`, `train.py`, `dataset.py`) | **Language-agnostic** — trainable as-is | None in method; bump `max_seq_len` (kernels are long); watch private-table memory: `#references × private_count × hidden` (e.g. 500 refs × 256 × 3584 ≈ 0.46B trainable params, fine under ZeRO-2) |
| Generation mechanics (`contrastive/generate.py`) | Temp-model + embedding-injection trick is model-agnostic | vLLM is CUDA-only here; for NPU generation use `vllm-ascend` or the transformers engine + `torch_npu`; raise `max_tokens` (2048 → 4096+) |
| Quality pipeline prompts | Python wording hard-coded | Rewrite instruction/response prompts for Ascend C; keep the 5-scale/3-scale labels and the `response_syntex` key |
| Syntax check | `tree-sitter-python` | Replace with CANN compile / CPU-debug check (Section 6); tree-sitter-cpp only as a cheap pre-filter if at all |
| Correctness execution | Python `exec()` sandbox | Replace with CANN compile + NPU/CPU-debug run + golden-tensor comparison; reuse the timeout/process-pool scaffolding |
| Evaluation harness | Python prompt + pyext sandbox | New Ascend C evaluator; reuse pass@k estimator and extraction |
| Base model choice | Qwen2.5-Coder-7B is ~9% compile success zero-shot on Ascend C | Acceptable *by design* — DOMINO exists to close this gap; but prefer a compile-check-anchored pipeline over LLM grading, and consider KernelGen-LM-32B as a stronger grader |
| Device/dtype defaults | `cuda:0`, `CUDA_VISIBLE_DEVICES`, bf16 | Parameterize device; bf16 is supported on 910B; remove/gate hard-coded `flash_attention_2` (`soft_prompt/model.py:12-13`, `--use_flash_attn` flag) — no FlashAttention-2 on NPU; use SDPA / `npu_fusion_attention` |

## 8. Roadmap

### Stage 0 — Reference data preparation (CPU only)

1. Build `data/train.jsonl` + `data/valid.jsonl` from the in-house Ascend C operator
   corpus + documentation, one `question_content` per line (kernel source, optionally
   prefixed with a short spec: operator name, I/O tensors, dtypes, target SoC).
2. Run `domino/utils/statistics.py` and a tokenizer-fertility check (tokens/line vs
   Python) to pick `max_seq_len` (recommend 2048).
3. Keep the reference count moderate (hundreds) — every sample gets its own
   private-token row.

### Stage 1 — GPU training + NPU-side generation/validation

1. **Env (A100 node):** `pip install -r requirements.txt`; fix the script path bugs
   when copying the scripts.
2. **Train:** `deepspeed domino/contrastive/train.py --model_name_or_path
   Qwen/Qwen2.5-Coder-7B-Instruct --public_soft_token_count 256
   --private_soft_token_count 256 --deepspeed configs/ds_z2.json --bf16 True …` →
   `public_soft_token_embeddings.pth`.
3. **Generate on the 910B** (the stage-1 requirement): two candidate paths, verify
   (a) first with a tiny run —
   (a) `vllm-ascend` plugin loading the `vllm_temp_model` checkpoint produced by
       `generate.py` (the embedding injection is plain HF format and is expected to
       load; confirm vllm-ascend handles the resized embedding matrix);
   (b) the transformers engine + `torch_npu`, with small edits: device `npu:0`,
       remove `flash_attention_2`.
   Fallback if both block: generate on A100 with stock vLLM and validate on Ascend —
   flag this clearly as a deviation from the stage-1 goal.
4. **Ascend C quality pipeline** — new sibling package `domino/pipeline/ascend_c/`
   following the existing stage conventions:
   - `quality_assessment.py`: prompt rewritten for "Ascend C operator kernel
     question"; same labels + `excellent` gate; register the grader in
     `llm_configs.json`.
   - `generate_response.py`: prompt asks for Ascend C kernel code (kernel class +
     `extern "C" __global__ __aicore__` entry, `.asc`-style single file), ```cpp
     fences, temp 0.
   - `compile_check.py` (replaces the tree-sitter stage): CANN CPU-debug build
     (`asc-tools`, CANN ≥ 9.0) or direct compile on the 910B; map the result to
     `response_syntex` ∈ {right, wrong, unknown} for format compatibility.
   - `make_pairs.py`: reuse unchanged (good + right).
   - Principle: **anchor filtering on the compiler, not on the LLM grader** —
     general LLMs grade Ascend C no better than they write it.
5. **SFT formatting:** emit Alpaca-style JSON for LLaMA-Factory, mirroring
   `code_execution/format_for_sft.py`.
6. **Evaluation:** new `domino/evaluation/ascend_c/` harness — prompt the SFT'd model
   for kernel code, compile with CANN on the 910B, run against golden inputs, compare
   outputs; reuse `pass_k.py` and the process-pool/timeout scaffolding. Report pass@k
   on compile success and functional correctness separately (as NPUKernelBench does).

### Stage 2 — NPU-native data flywheel (the academic point)

1. Port training to the 910B: `torch_npu` + HF Trainer; SDPA/NPU fused attention
   instead of FlashAttention-2; DeepSpeed-NPU or MindSpeed for ZeRO-2-equivalent
   sharding; verify loss curves match the GPU run on the same data.
2. Generation fully on NPU via the verified Stage-1 path (vllm-ascend or
   transformers+torch_npu).
3. Run the complete loop — train → generate → grade → compile-check → SFT pairs — on
   the Ascend stack and measure end-to-end throughput vs the GPU baseline. This
   substantiates the claim that the Ascend NPU platform can itself close the data
   flywheel for Ascend C.
4. **Later extension:** full operator projects (kernel + host tiling/registration +
   prototype JSON, msopgen packaging) once the SFT'd model reliably writes kernels;
   the pipeline stages generalize by swapping prompts and the validation step.

## 9. Risks and open questions

- **CANN version on the 910B node** determines whether CPU-debug validation (≥ 9.0)
  and vllm-ascend are available — check first (`npu-smi info`, CANN version).
- **vllm-ascend + patched-embedding checkpoint** is unverified; test with ~100
  generations before committing.
- **Grader quality:** keep LLM grading on the *instruction* side only; let the CANN
  compiler decide response quality.
- **Sequence lengths:** Ascend C kernels with tiling code are long; expect to raise
  `max_seq_len`/`max_tokens`.
- **Loss construction:** the numerator terms are batch-mean while the denominator is
  a per-sample mean (`model.py:118-124`) — if Ascend C training is unstable,
  normalize per-sample first.
- **`.asc` flow:** confirm the exact single-file operator compile/run workflow
  against the installed CANN docs before writing `compile_check.py`.
- **Verification limits:** this repo's full workflow needs GPUs/NPUs and model
  downloads; this analysis is code- and literature-based, not an executed run.
