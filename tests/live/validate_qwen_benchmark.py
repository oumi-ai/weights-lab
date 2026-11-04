"""Frozen-input MATH-500 evaluation of the released Qwen edit; no training.

Run on an allocated CUDA GPU. The parent starts a fresh Python process for each
arm. --prepare-only freezes inputs without importing vLLM or requiring CUDA.
Full validation additionally requires higher edited accuracy and lower token-cap
share. These directional criteria are declared before either arm executes; exact
published percentages are not required because upstream dataset pins are absent.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import types
import urllib.request
from pathlib import Path

MODELS = {
    "base": {"model": "Qwen/Qwen3.5-0.8B", "revision": "2fc06364715b967f1860aea9cf38778875588b17"},
    "edited": {
        "model": "RikiyaT/Qwen3.5-0.8B-weight-space-programming",
        "revision": "d1671a178705bcab7644806d23c535517f35723b",
    },
}
LOGS_REVISION = "4d09ba6d6a7e891dd8063974dfe81b3d78e3494f"
GRADER_URL = f"https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/{LOGS_REVISION}/exp8_math/code/mathqa.py"
GRADER_SHA256 = "f0d576a1adc625dede7bff749b3bf8c9a5a8f6bae43d51e2d0ee36d694e89843"
DATASET = "HuggingFaceH4/MATH-500"
SAMPLING = {
    "n": 4,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "presence_penalty": 1.5,
    "max_tokens": 32768,
    "seed": 0,
}
REQUIRED = {
    "vllm": "0.33.0",
    "transformers": "5.20.2",
    "math-verify": "0.9.0",
    "latex2sympy2_extended": "1.11.0",
    "antlr4-python3-runtime": "4.13.2",
}
SHIM_NOTE = (
    "mcqa.loop_signals shim raises if called; omitted upstream loop diagnostic is unused by grading"
)
VALIDATION_CRITERIA = {
    "requires_full_protocol": True,
    "problems_per_arm": 500,
    "samples_per_problem": 4,
    "edited_accuracy_must_be": "strictly greater than base accuracy",
    "edited_cut_share_must_be": "strictly less than base cut share",
    "exact_published_percentage_required": False,
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_child_path():
    """Expose executables installed beside an absolute virtualenv Python."""
    directory = str(Path(sys.executable).parent)
    current = os.environ.get("PATH", "")
    entries = current.split(os.pathsep) if current else []
    os.environ["PATH"] = os.pathsep.join([directory, *[p for p in entries if p != directory]])
    return directory


def build_metadata():
    metadata = {"ninja_executable": shutil.which("ninja"), "nvidia_driver_version": None}
    try:
        metadata["ninja_package_version"] = importlib.metadata.version("ninja")
    except importlib.metadata.PackageNotFoundError:
        metadata["ninja_package_version"] = None
    if shutil.which("nvidia-smi"):
        try:
            driver = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if driver.returncode == 0:
                metadata["nvidia_driver_version"] = driver.stdout.strip().splitlines()
        except (OSError, subprocess.TimeoutExpired):
            pass
    return metadata


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def log(root, event, **fields):
    record = {"time": time.time(), "event": event, **fields}
    with (root / "progress.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
        stream.flush()
    print(json.dumps(record), flush=True)


def load_grader(path, expected_hash=GRADER_SHA256):
    if sha256(path) != expected_hash:
        raise ValueError("Upstream grader SHA256 mismatch")
    shim = types.ModuleType("mcqa")

    def unavailable(*args, **kwargs):
        raise NotImplementedError(SHIM_NOTE)

    shim.loop_signals = unavailable
    previous = sys.modules.get("mcqa")
    sys.modules["mcqa"] = shim
    module = types.ModuleType("pinned_qwen_mathqa")
    module.__file__ = str(path)
    try:
        exec(compile(Path(path).read_bytes(), str(path), "exec"), module.__dict__)
    finally:
        if previous is None:
            sys.modules.pop("mcqa", None)
        else:
            sys.modules["mcqa"] = previous
    return module


def prepare(root, dataset_revision=None, max_problems=500, grader_source=None):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    if not 1 <= max_problems <= 500:
        raise ValueError("max-problems must be between 1 and 500")
    root.mkdir(parents=True, exist_ok=False)
    log(root, "preparing")
    source = (
        Path(grader_source).read_bytes()
        if grader_source
        else urllib.request.urlopen(GRADER_URL, timeout=60).read()
    )
    grader_path = root / "upstream_mathqa.py"
    grader_path.write_bytes(source)
    grader = load_grader(grader_path)
    # Resolve once, including when the caller supplies a tag or branch.
    revision = HfApi().dataset_info(DATASET, revision=dataset_revision).sha
    loaded = load_dataset(DATASET, revision=revision, split="test")
    if len(loaded) != 500:
        raise ValueError(f"Expected all 500 MATH-500 rows, received {len(loaded)}")
    rows = [
        {"id": row["unique_id"], "problem": row["problem"], "answer": row["answer"]}
        for row in loaded
    ]
    if len({row["id"] for row in rows}) != 500:
        raise ValueError("MATH-500 problem IDs are not unique")
    rows = rows[:max_problems]
    tokenizer = AutoTokenizer.from_pretrained(
        MODELS["base"]["model"], revision=MODELS["base"]["revision"], trust_remote_code=False
    )
    prompts = [
        {
            "id": row["id"],
            "prompt": tokenizer.apply_chat_template(
                [{"role": "user", "content": grader.build_prompt(row["problem"])}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            ),
        }
        for row in rows
    ]
    for filename, records in (("dataset.jsonl", rows), ("prompts.jsonl", prompts)):
        (root / filename).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
        )
    manifest = {
        "state": "prepared",
        "full_protocol": max_problems == 500,
        "full_validated": False,
        "problem_count": len(rows),
        "dataset": DATASET,
        "dataset_revision": revision,
        "dataset_split": "test",
        "upstream_dataset_revision_was_unpinned": True,
        "models": MODELS,
        "tokenizer": MODELS["base"],
        "thinking": True,
        "sampling": SAMPLING,
        "dtype": "bfloat16",
        "max_model_len": 34816,
        "gpu_memory_utilization": 0.85,
        "logs_revision": LOGS_REVISION,
        "grader_url": GRADER_URL,
        "grader_sha256": GRADER_SHA256,
        "grader_shim": SHIM_NOTE,
        "required_versions": REQUIRED,
        "validation_criteria": VALIDATION_CRITERIA,
        "files": {
            name: sha256(root / name)
            for name in ("dataset.jsonl", "prompts.jsonl", "upstream_mathqa.py")
        },
        "prepare_versions": {
            name: importlib.metadata.version(name)
            for name in ("transformers", "datasets", "huggingface_hub")
        },
    }
    write_json(root / "manifest.json", manifest)
    log(root, "prepared", problems=len(rows), full_protocol=manifest["full_protocol"])
    return manifest


def read_prepared(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha256(root / name) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    if manifest["models"] != MODELS or manifest["sampling"] != SAMPLING:
        raise ValueError("Prepared model pins or sampling differ from the fixed protocol")
    rows = [json.loads(line) for line in (root / "dataset.jsonl").read_text().splitlines()]
    prompts = [json.loads(line) for line in (root / "prompts.jsonl").read_text().splitlines()]
    if (
        len(rows) != manifest["problem_count"]
        or len(prompts) != len(rows)
        or [row["id"] for row in rows] != [prompt["id"] for prompt in prompts]
    ):
        raise ValueError("Frozen dataset and prompt rows do not align")
    return manifest, rows, prompts


def grade_outputs(rows, outputs, grader, stream, samples=4):
    if len(outputs) != len(rows) or not rows:
        raise ValueError("Generation did not return every problem")
    correct = cut = tokens = total = 0
    for row, output in zip(rows, outputs, strict=True):
        if len(output.outputs) != samples or sorted(s.index for s in output.outputs) != list(
            range(samples)
        ):
            raise ValueError(f"Incomplete or duplicated samples for {row['id']}")
        for sample in output.outputs:
            if sample.finish_reason not in {"stop", "length"}:
                raise ValueError(f"Unexpected finish reason: {sample.finish_reason}")
            prediction, _ = grader.extract(sample.text)
            good = bool(grader.is_correct(prediction, row["answer"]))
            record = {
                "id": row["id"],
                "sample_index": sample.index,
                "text": sample.text,
                "extracted": prediction,
                "answer": row["answer"],
                "correct": good,
                "token_count": len(sample.token_ids),
                "finish_reason": sample.finish_reason,
            }
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            correct += good
            cut += sample.finish_reason == "length"
            tokens += len(sample.token_ids)
            total += 1
    return {
        "problems": len(rows),
        "samples_per_problem": samples,
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "cut": cut,
        "cut_share": cut / total,
        "tokens_sum": tokens,
        "mean_tokens": tokens / total,
    }


def run_arm(root, arm):
    child_bin = prepare_child_path()
    manifest, rows, prompts = read_prepared(root)
    folder = root / arm
    folder.mkdir(exist_ok=False)
    summary = {
        "state": "running",
        "arm": arm,
        "full_protocol": manifest["full_protocol"],
        "full_validated": False,
        "protocol_complete": False,
        "manifest_sha256": sha256(root / "manifest.json"),
        "python": sys.version,
        "platform": platform.platform(),
        "child_executable_directory": child_bin,
        "model": MODELS[arm],
        "provenance": manifest,
        "started_at": time.time(),
    }
    write_json(folder / "summary.json", summary)
    try:
        versions = {name: importlib.metadata.version(name) for name in REQUIRED}
        if versions != REQUIRED:
            raise RuntimeError(
                f"Benchmark dependency versions differ: {versions}; expected {REQUIRED}"
            )
        summary["build_metadata"] = build_metadata()
        import torch
        from vllm import LLM, SamplingParams

        if not torch.cuda.is_available():
            raise RuntimeError("Full benchmark requires a CUDA GPU")
        gpu = torch.cuda.get_device_properties(0)
        summary.update(
            versions=versions,
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu={
                "name": gpu.name,
                "total_memory": gpu.total_memory,
                "capability": list(torch.cuda.get_device_capability(0)),
            },
        )
        grader = load_grader(root / "upstream_mathqa.py")
        log(root, "loading_arm", arm=arm, gpu=summary["gpu"])
        llm = LLM(
            **MODELS[arm],
            tokenizer=MODELS["base"]["model"],
            tokenizer_revision=MODELS["base"]["revision"],
            trust_remote_code=False,
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
            max_model_len=34816,
            seed=0,
        )
        finished = threading.Event()

        def heartbeat():
            while not finished.wait(30):
                log(
                    root, "generating", arm=arm, elapsed_seconds=time.time() - summary["started_at"]
                )

        watcher = threading.Thread(target=heartbeat, daemon=True)
        log(root, "generation_started", arm=arm, problems=len(rows), samples_per_problem=4)
        watcher.start()
        try:
            outputs = llm.generate([p["prompt"] for p in prompts], SamplingParams(**SAMPLING))
        finally:
            finished.set()
            watcher.join()
        with (folder / "samples.jsonl").open("x") as stream:
            scores = grade_outputs(rows, outputs, grader, stream)
        if scores["total"] != len(rows) * 4:
            raise RuntimeError("Incomplete benchmark sample count")
        summary.update(
            scores, state="passed", protocol_complete=manifest["full_protocol"] and len(rows) == 500
        )
        log(root, "arm_finished", arm=arm, **scores)
    except BaseException as error:
        summary.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary["duration_seconds"] = time.time() - summary["started_at"]
        write_json(folder / "summary.json", summary)
    return summary


def compare_arms(arms, full_protocol):
    """Apply the predeclared directional criteria after both arms finish."""
    base, edited = arms["base"], arms["edited"]
    protocol_complete = full_protocol and all(
        arm["state"] == "passed"
        and arm["full_protocol"]
        and arm["problems"] == 500
        and arm["samples_per_problem"] == 4
        and arm["total"] == 2000
        for arm in (base, edited)
    )
    accuracy_delta = edited["accuracy"] - base["accuracy"]
    cut_share_delta = edited["cut_share"] - base["cut_share"]
    reproduced = protocol_complete and accuracy_delta > 0 and cut_share_delta < 0
    return {
        "accuracy_delta": accuracy_delta,
        "cut_share_delta": cut_share_delta,
        "edited_higher_accuracy": accuracy_delta > 0,
        "edited_lower_cut_share": cut_share_delta < 0,
        "protocol_complete": protocol_complete,
        "qualitative_effect_reproduced": reproduced,
        "full_validated": reproduced,
    }


def run_parent(root):
    manifest, _, _ = read_prepared(root)
    summary = {
        "state": "running",
        "full_protocol": manifest["full_protocol"],
        "full_validated": False,
        "protocol_complete": False,
        "qualitative_effect_reproduced": False,
        "validation_criteria": VALIDATION_CRITERIA,
        "arms": {},
    }
    write_json(root / "summary.json", summary)
    log(root, "validation_criteria_declared", criteria=VALIDATION_CRITERIA)
    try:
        for arm in MODELS:
            log(root, "child_starting", arm=arm)
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--output-dir",
                    str(root),
                    "--arm",
                    arm,
                ],
                check=True,
            )
            result = json.loads((root / arm / "summary.json").read_text())
            if result["state"] != "passed":
                raise RuntimeError(f"Arm did not finish: {arm}")
            summary["arms"][arm] = result
        if summary["arms"]["base"]["gpu"] != summary["arms"]["edited"]["gpu"]:
            raise RuntimeError("The two arms ran on different GPU configurations")
        summary.update(compare_arms(summary["arms"], manifest["full_protocol"]))
        summary["state"] = (
            "validation_failed"
            if manifest["full_protocol"] and not summary["full_validated"]
            else "passed"
        )
        log(
            root,
            "comparison_finished",
            state=summary["state"],
            accuracy_delta=summary["accuracy_delta"],
            cut_share_delta=summary["cut_share_delta"],
            full_validated=summary["full_validated"],
        )
    except BaseException as error:
        summary.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(root / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--grader-source", type=Path)
    parser.add_argument(
        "--max-problems",
        type=int,
        default=500,
        help="Below 500 is diagnostic, never full validation",
    )
    parser.add_argument("--arm", choices=list(MODELS), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = args.output_dir.expanduser().resolve()
    if args.arm:
        return run_arm(root, args.arm)
    manifest = prepare(root, args.dataset_revision, args.max_problems, args.grader_source)
    if not args.prepare_only:
        return run_parent(root)
    return {
        "state": "prepared",
        "full_protocol": manifest["full_protocol"],
        "full_validated": False,
    }


if __name__ == "__main__":
    result = main()
    if result["state"] in {"failed", "validation_failed"}:
        raise SystemExit(1)
