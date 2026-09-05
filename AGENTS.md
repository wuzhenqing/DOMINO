# AGENTS.md

Guidance for AI coding agents working in this repository.

## Project Overview

**DOMINO** (Domain-Specific Data Synthesis for LLMs via Minimal Sufficient Representation Learning) is a research codebase that learns a minimal sufficient representation of a target domain from limited reference examples, then uses it to synthesize diverse in-domain training data. The primary application domain is **code**: code generation (problem → solution) and code execution (function + input → predicted output).

The method trains two kinds of learnable continuous soft tokens on top of a frozen base LLM:

- **Domain-level (public) soft tokens** — shared across all reference samples, capturing generalizable domain patterns.
- **Sample-level (private) soft tokens** — unique per sample, encoding sample-specific information.

Training combines a reconstruction objective (public tokens must reconstruct every reference sample) with a contrastive objective (private tokens must help only their own sample). At synthesis time only the public tokens are used, so generation produces novel in-domain samples instead of memorized copies.

License: Apache 2.0. All documentation and comments are in English; keep it that way.

## Tech Stack

- Python 3.10+, PyTorch 2.0+
- HuggingFace `transformers` (Trainer, HfArgumentParser) + `accelerate` + `datasets`
- **DeepSpeed** (ZeRO stage 2/3) for distributed training — configs in `configs/`
- **vLLM** for all large-scale inference (generation, grading, evaluation)
- `tree-sitter` + `tree-sitter-python` for syntax checking of generated code
- `pyext` for runtime module compilation in the code-generation evaluator
- scikit-learn / matplotlib / numpy / scipy for analysis and figures

There is **no** `pyproject.toml`, `setup.py`, or other packaging file. The `domino` package is used in-place from the repository root; dependencies come from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Repository Layout

```
domino/
├── contrastive/            # Core DOMINO method (public + private soft tokens)
│   ├── model.py            # PublicPrivateContrastiveModel: frozen LLM + two nn.Embedding tables,
│   │                       # contrastive loss over public-only / public+private / cross-sample CE losses
│   ├── train.py            # HF Trainer entry point (launch via DeepSpeed)
│   ├── dataset.py          # SoftDataset: JSONL loader, uses the "question_content" field
│   └── generate.py         # Synthetic sample generation; engines: transformers or vllm
├── soft_prompt/            # Baseline: single shared (task-level) soft prompt
│   └── model/train/dataset/generate.py   # Same layout as contrastive/
├── pipeline/               # Synthetic-data quality control pipelines
│   ├── llm_configs.json    # Per-model stop tokens/token ids for the grading LLMs
│   ├── code_generation/    # quality_assessment → generate_response → filter_by_quality → make_pairs
│   └── code_execution/     # quality_assessment → generate_response → check_correctness → filter → format_for_sft
├── evaluation/
│   ├── code_generation/    # LiveCodeBench-style harness: evaluate.py, codegen_metrics.py,
│   │                       # pass_k.py, testing_utils.py (sandboxed execution), codeproblem.py
│   └── code_execution/     # Output-prediction eval: evaluate.py (CoT [PYTHON]/[ANSWER] prompt), metrics.py, utils.py
├── baselines/magpie/       # Configs/scripts + README for the MAGPIE baseline (external repo, see its README)
└── utils/                  # embedding.py (e.g. codet5p embeddings), statistics.py, query_llm.py
scripts/                    # Example end-to-end shell scripts (templates — customize paths before running)
configs/                    # ds_z2.json, ds_z3.json — DeepSpeed ZeRO configs
examples/                   # Paper figures: pass-rate plots, t-SNE visualization
assets/                     # Architecture diagram (arch.png / arch.pdf)
```

## Common Commands

All Python entry points are run **as modules from the repository root** (`python -m domino...`). The shell scripts in `scripts/` are examples; edit the paths (model, data, output dir, devices) before use.

### 1. Train DOMINO soft tokens

Input: JSONL reference data with a `question_content` field per line.

```bash
# See scripts/train_contrastive.sh for the full flag set; it is launched with DeepSpeed:
deepspeed --master_port 1113 --include localhost:0 domino/contrastive/train.py \
    --model_name_or_path Qwen/Qwen2.5-Coder-7B-Instruct \
    --train_data_path ./data/train.jsonl --valid_data_path ./data/valid.jsonl \
    --public_soft_token_count 256 --private_soft_token_count 256 \
    --deepspeed configs/ds_z2.json --bf16 True --gradient_checkpointing True \
    --num_train_epochs 10 --per_device_train_batch_size 3 --learning_rate 1e-3 \
    --output_dir ./outputs/contrastive ...
```

Training saves only the soft-token tables as raw state dicts into the output dir: `public_soft_token_embeddings.pth` and `private_soft_token_embeddings.pth`. The base LLM stays frozen (typically <1% trainable parameters).

The baseline variant is `domino/soft_prompt/train.py` (see `scripts/train_soft_prompt.sh`).

### 2. Generate synthetic data

```bash
python -m domino.contrastive.generate \
    --pretrained_model_name_or_path <MODEL> --tokenizer_name_or_path <MODEL> \
    --soft_prompt_dir ./outputs/contrastive --inference_engine vllm \
    --public_soft_token_count 256 --temp 0.8 --target_count 80000 \
    --tensor_parallel_size 4 --device cuda:0
```

The vLLM path adds the soft tokens as new special tokens (`<soft_0>`…), writes their trained embeddings into the input embedding matrix, and saves a temp model under `<soft_prompt_dir>/vllm_temp_model/`. Output goes to `<soft_prompt_dir>/vllm_generated_{N}_samples_temp{T}.jsonl`.

### 3. Quality-control pipeline

See `scripts/pipeline_codegen.sh` (code generation) and `scripts/pipeline_codeexe.sh` (code execution). Stages, each appending a suffix to the JSONL filename:

- `domino.pipeline.code_generation.quality_assessment` — an LLM grades each synthetic instruction (very poor … excellent) → `*_instruct_quality.jsonl`
- `domino.pipeline.code_generation.generate_response` — generate responses → `*_response.jsonl`
- `domino.pipeline.code_generation.filter_by_quality` — LLM grades responses and/or tree-sitter syntax check (`--process_stage response_llm_assessment|tree-sitter`)
- `domino.pipeline.code_generation.make_pairs` — keep only pairs where the LLM rating is `good` and syntax is `right` → `*_quality_filtered.jsonl`

The grading LLM's stop tokens are looked up in `domino/pipeline/llm_configs.json` by model name; add an entry there when grading with a new model.

### 4. Downstream SFT and evaluation

SFT with the synthesized pairs is done **outside** this repo (e.g. LLaMA-Factory). Evaluation:

```bash
# Code generation (pass@1/5/10)
python -m domino.evaluation.code_generation.evaluate \
    --data_path ./data/test.jsonl --model_path <MODEL> \
    [--tensor_parallel_size 4 --n_samples 10 --temperature 0.2 --output_path metrics.json]

# Code execution / output prediction (pass@1)
python -m domino.evaluation.code_execution.evaluate \
    --test_path ./data/test_codeexe.jsonl --model_path <MODEL>
```

## Coding Conventions

- Plain functions and argparse CLIs; training scripts use `HfArgumentParser` dataclasses (`ModelArguments`, `DataArguments`, `TrainingArguments`).
- JSONL everywhere; pipeline stages read a file, add one field per record, and write a new suffixed file rather than mutating in place.
- vLLM defaults used throughout: `dtype="bfloat16"`, `gpu_memory_utilization=0.95`, `trust_remote_code=True`; batch sizes around 200.
- The default/reference base model is `Qwen/Qwen2.5-Coder-7B-Instruct`; the 14B variant and DeepSeek-Coder-V2-Lite also appear in configs.
- Labels for soft-token positions are masked with `-100`; tokenizers fall back to `pad_token_id = eos_token_id` with right padding.
- The key `response_syntex` (misspelled) is part of the pipeline's data format — keep it for compatibility.
- Data and outputs are not committed: `.gitignore` excludes `data/`, `outputs/`, `*.pth`, `*.safetensors`, `*.bin`, `*.log`.

## Testing

There is **no test suite and no CI** in this repository — do not look for pytest configurations or add test scaffolding unprompted. Changes are verified by running the relevant module end-to-end on a small data slice, and ultimately by evaluation metrics (pass@k on LiveCodeBench-style benchmarks). Note the full workflow requires GPUs (DeepSpeed/vLLM) and HuggingFace model downloads, so many changes cannot be fully exercised on a CPU-only machine — say so when that limits verification.

## Security Considerations

- **The evaluation harness executes untrusted LLM-generated Python code** (`evaluation/code_generation/testing_utils.py`, `evaluation/code_execution/utils.py::check_correctness`, `pipeline/code_execution/check_correctness.py`). Mitigations already in place: a `reliability_guard()` that disables destructive builtins, execution in temporary directories, SIGALRM timeouts, and multiprocessing isolation. Run evaluations inside a container or disposable environment anyway.
- Models are loaded with `trust_remote_code=True` — only load checkpoints from trusted sources.
- Soft-token checkpoints are loaded with `torch.load` (`.pth`) — only open files produced by your own training runs.
- Keep secrets and API keys out of the repo; the grading/generation stages call local vLLM instances, not external APIs, so no credentials should be needed.

## Known Gotchas

- `scripts/train_contrastive.sh` and `scripts/train_soft_prompt.sh` now call `../domino/contrastive/train.py` and `../domino/soft_prompt/train.py`; run them from the `scripts/` directory (the DeepSpeed config path is `../configs/...`).
- Pipeline scripts mix `python -m domino...` (requires repo root on `PYTHONPATH`) with relative `./outputs` paths — run them from the repository root.
- `domino/pipeline/code_execution/check_correctness.py` is incomplete (references an undefined `synthetic_path` at module function scope and lacks a CLI); treat it as a work-in-progress snippet.
- The MAGPIE baseline under `domino/baselines/magpie/` is not self-contained: it requires cloning the external `magpie-align/magpie` repo and patching its `exp/gen_ins.py` (see that directory's README).
