#!/usr/bin/env python3
"""Build an Ascend C reference corpus from ascend/samples for DOMINO Stage 0.

The script scans the shallow-cloned samples repository (operator/ascendc and
operator_contrib), extracts kernel-side source files that contain Ascend C
markers (__aicore__, __global__, GM_ADDR), attaches a short spec comment when
derivable from the path or nearby README/JSON prototype files, and writes
JSONL train/valid splits.
"""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path


KERNEL_MARKERS = re.compile(r"__aicore__|__global__|GM_ADDR|AscendC::")

# Path segments that are host-side or framework-invocation wrappers, not kernel source.
HOST_SEGMENTS = (
    "/op_host/",
    "/AclNNInvocation/",
    "/PytorchInvocation/",
    "/TensorflowInvocation/",
    "/CppExtensionInvocation/",
    "/AclOfflineModel/",
    "/AclOnlineModel/",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build Ascend C reference corpus")
    parser.add_argument(
        "--samples-repo",
        default="data/raw/ascend-samples",
        help="Path to the shallow-cloned ascend/samples repository",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory to write train.jsonl and valid.jsonl",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.1,
        help="Fraction of records to hold out as validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/valid split",
    )
    parser.add_argument(
        "--include-operator-contrib",
        action="store_true",
        default=True,
        help="Also scan operator_contrib for additional kernel sources (default: True)",
    )
    parser.add_argument(
        "--no-include-operator-contrib",
        dest="include_operator_contrib",
        action="store_false",
        help="Restrict scan to operator/ascendc only",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Limit total records (0 = unlimited)",
    )
    return parser.parse_args()


def find_candidate_dirs(repo_root: Path):
    candidates = [repo_root / "operator" / "ascendc"]
    if args.include_operator_contrib:
        candidates.append(repo_root / "operator_contrib")
    return [d for d in candidates if d.exists()]


def is_kernel_source(path: Path) -> bool:
    """Return True if the file appears to contain Ascend C kernel code."""
    if path.stat().st_size == 0:
        return False
    # Only read a reasonable prefix; kernel markers are usually early.
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return bool(KERNEL_MARKERS.search(text))


def discover_kernel_files(repo_root: Path):
    dirs = find_candidate_dirs(repo_root)
    files = []
    for base in dirs:
        for ext in ("*.cpp", "*.h", "*.hpp", "*.asc"):
            for path in base.rglob(ext):
                if any(seg in path.as_posix() for seg in HOST_SEGMENTS):
                    continue
                if is_kernel_source(path):
                    files.append(path)
    # Deterministic ordering.
    files.sort()
    return files


def find_nearest_readme_and_json(kernel_path: Path, repo_root: Path):
    """Walk upward from kernel_path looking for README*.md and *.json files."""
    readmes = []
    jsons = []
    current = kernel_path.parent
    # Stop at the repository root or after a sane number of levels.
    for _ in range(6):
        if not current.is_relative_to(repo_root) or current == repo_root:
            break
        for child in current.iterdir():
            if child.is_file():
                if child.name.lower().startswith("readme") and child.suffix == ".md":
                    readmes.append(child)
                elif child.suffix == ".json":
                    jsons.append(child)
        current = current.parent
    return readmes, jsons


def strip_html_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def parse_readme_spec(readme_path: Path) -> dict:
    """Best-effort parser for the operator spec table in README.md files."""
    try:
        text = readme_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {}

    spec = {}
    # Try to find a short description under 算子描述.
    desc_match = re.search(
        r"##?\s*算子描述\s*\n(.*?)(?:\n##|\n###|\n##\s*算子规格|$)",
        text,
        re.DOTALL,
    )
    if desc_match:
        desc = " ".join(strip_html_tags(desc_match.group(1)).split())[:300]
        spec["description"] = desc

    # Find the first table after 算子规格描述.
    spec_match = re.search(r"算子规格描述(.*?)\n##", text, re.DOTALL)
    if not spec_match:
        spec_match = re.search(r"算子规格描述(.*?)$", text, re.DOTALL)
    if not spec_match:
        return spec

    table_text = spec_match.group(1)
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_text, re.DOTALL)
    inputs = []
    outputs = []
    current = None
    for row in rows:
        cells = [strip_html_tags(c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)]
        if not cells:
            continue
        first = cells[0]
        if "算子输入" in first:
            current = "input"
            continue
        if "算子输出" in first:
            current = "output"
            continue
        if "核函数名" in first:
            spec["kernel_name"] = cells[-1] if cells else ""
            continue
        # Skip header rows (name / shape / data type / format).
        if any(h in cells for h in ("name", "shape", "data type", "format", "名称")):
            continue
        # Data rows should have 4 cells; malformed tables may repeat section labels in rowspan.
        if len(cells) == 4:
            name, shape, dtype, fmt = cells
            entry = f"{name}({dtype})[{fmt}]" if fmt else f"{name}({dtype})"
            if current == "input":
                inputs.append(entry)
            elif current == "output":
                outputs.append(entry)

    if inputs:
        spec["inputs"] = inputs
    if outputs:
        spec["outputs"] = outputs
    return spec


def parse_json_spec(json_path: Path) -> dict:
    """Best-effort parser for operator prototype JSON files."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}

    if not isinstance(data, list) or not data:
        return {}
    entry = data[0]
    spec = {}
    if "op" in entry:
        spec["op_type"] = entry["op"]

    def fmt_tensor(t):
        name = t.get("name", "")
        dtypes = t.get("type", [])
        fmts = t.get("format", [])
        dtype = dtypes[0] if dtypes else "?"
        fmt = fmts[0] if fmts else "?"
        return f"{name}({dtype})[{fmt}]"

    if "input_desc" in entry:
        spec["inputs"] = [fmt_tensor(t) for t in entry["input_desc"]]
    if "output_desc" in entry:
        spec["outputs"] = [fmt_tensor(t) for t in entry["output_desc"]]
    return spec


def derive_spec(kernel_path: Path, repo_root: Path, source_code: str) -> dict:
    rel = kernel_path.relative_to(repo_root)
    parts = rel.parts
    op_name = kernel_path.stem

    # Heuristic operator category.
    category = "unknown"
    if "operator_contrib" in parts:
        category = "operator_contrib"
    elif "operator" in parts and "ascendc" in parts:
        idx = parts.index("ascendc")
        if len(parts) > idx + 1:
            category = parts[idx + 1]

    spec = {
        "operator_name": op_name,
        "source_path": str(rel),
        "category": category,
    }

    readmes, jsons = find_nearest_readme_and_json(kernel_path, repo_root)

    # Prefer JSON prototype files for structured I/O.
    for jp in jsons:
        parsed = parse_json_spec(jp)
        if parsed.get("inputs") or parsed.get("outputs"):
            spec.update(parsed)
            break

    # Fall back to README tables / descriptions.
    if not (spec.get("inputs") or spec.get("outputs")):
        for rp in readmes:
            parsed = parse_readme_spec(rp)
            if parsed.get("inputs") or parsed.get("outputs") or parsed.get("description"):
                spec.update(parsed)
                break

    # If README/JSON gave a kernel name, use it as the operator name.
    if "kernel_name" in spec and spec["kernel_name"]:
        spec["operator_name"] = spec["kernel_name"]
    elif "op_type" in spec and spec["op_type"]:
        spec["operator_name"] = spec["op_type"]

    return spec


def build_question_content(source_code: str, spec: dict) -> str:
    lines = ["// Ascend C kernel source"]
    lines.append(f"// Operator: {spec.get('operator_name', 'unknown')}")
    lines.append(f"// Source: {spec.get('source_path', '')}")
    if spec.get("category"):
        lines.append(f"// Category: {spec['category']}")
    if spec.get("inputs"):
        lines.append(f"// Inputs: {', '.join(spec['inputs'])}")
    if spec.get("outputs"):
        lines.append(f"// Outputs: {', '.join(spec['outputs'])}")
    if spec.get("description"):
        desc = spec["description"].replace("\n", " ")
        lines.append(f"// Description: {desc}")
    lines.append("")
    return "\n".join(lines) + source_code


def main():
    global args
    args = parse_args()

    repo_root = Path(args.samples_repo).expanduser().resolve()
    if not repo_root.exists():
        raise SystemExit(f"Samples repo not found: {repo_root}")

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {repo_root} for Ascend C kernel sources ...")
    files = discover_kernel_files(repo_root)
    print(f"Found {len(files)} candidate kernel source files")

    records = []
    seen_hashes = set()
    skipped_dup = 0
    for path in files:
        try:
            source_code = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            print(f"  warn: could not read {path}: {exc}")
            continue

        h = hashlib.sha256(source_code.encode("utf-8")).hexdigest()
        if h in seen_hashes:
            skipped_dup += 1
            continue
        seen_hashes.add(h)

        spec = derive_spec(path, repo_root, source_code)
        question_content = build_question_content(source_code, spec)
        records.append(
            {
                "question_content": question_content,
                "operator_name": spec.get("operator_name"),
                "source_path": spec.get("source_path"),
                "category": spec.get("category"),
            }
        )

    if args.max_files and args.max_files > 0:
        records = records[: args.max_files]

    print(f"Unique records: {len(records)} (skipped {skipped_dup} exact duplicates)")

    # Deterministic shuffle and split.
    rng = random.Random(args.seed)
    rng.shuffle(records)
    split_at = int(len(records) * (1 - args.valid_ratio))
    train = records[:split_at]
    valid = records[split_at:]

    train_path = out_dir / "train.jsonl"
    valid_path = out_dir / "valid.jsonl"

    for path, subset in ((train_path, train), (valid_path, valid)):
        with path.open("w", encoding="utf-8") as f:
            for rec in subset:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote {len(subset)} records to {path}")

    # Persist a small manifest for reproducibility.
    manifest = {
        "samples_repo": str(repo_root),
        "valid_ratio": args.valid_ratio,
        "seed": args.seed,
        "include_operator_contrib": args.include_operator_contrib,
        "train_count": len(train),
        "valid_count": len(valid),
        "duplicate_count": skipped_dup,
    }
    with (out_dir / "ascend_corpus_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
