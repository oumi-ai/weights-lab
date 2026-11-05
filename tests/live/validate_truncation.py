"""Check the lab's truncation share against vLLM's on a MATH-500 subset.

This is the Phase 1 verification in #8. Both engines sample the frozen prompts from
validate_qwen_benchmark.py, with its protocol minus presence_penalty, which
transformers lacks, and both stop on the same tokens. Run it on one CUDA GPU. Each
engine and model runs in a fresh process. The check passes when, for each model,
the 95% bootstrap interval for the paired per-problem difference in truncation
share (lab minus vLLM) contains zero, and both engines saw identical prompt tokens.

    python tests/live/validate_truncation.py --output-dir DIR --max-problems 40
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_qwen_benchmark import (  # noqa: E402
    MODELS,
    REQUIRED,
    SAMPLING,
    build_metadata,
    load_grader,
    log,
    prepare,
    prepare_child_path,
    read_prepared,
    write_json,
)

ENGINES = ("vllm", "lab")
# The published protocol without presence_penalty, which only vLLM implements.
PROTOCOL = {key: value for key, value in SAMPLING.items() if key != "presence_penalty"}
SAMPLES = PROTOCOL["n"]
# <|im_end|> ends the assistant turn and is the tokenizer's EOS, which vLLM stops on.
# <|endoftext|> is the checkpoint's own EOS, which generate() stops on by default.
STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")
CHAT = {"kwargs": {"enable_thinking": True}}
BOOTSTRAP = {"resamples": 10000, "seed": 0, "level": 0.95}
CRITERIA = {
    "statistic": "mean over problems of (lab cut fraction - vLLM cut fraction)",
    "interval": "95% percentile bootstrap over problems, 10000 resamples, seed 0",
    "passes_when": "the interval contains zero for every model, and prompt tokens match",
}


def stop_ids(tokenizer):
    ids = [tokenizer.convert_tokens_to_ids(token) for token in STOP_TOKENS]
    if any(not isinstance(i, int) or i == tokenizer.unk_token_id for i in ids):
        found = dict(zip(STOP_TOKENS, ids, strict=True))
        raise ValueError(f"Tokenizer lacks a stop token: {found}")
    return ids


def grade(grader, text, answer):
    """Return (extracted, correct, error); a grader failure counts as incorrect."""
    try:
        prediction, _ = grader.extract(text)
        return prediction, bool(grader.is_correct(prediction, answer)), None
    except Exception as error:  # noqa: BLE001 - recorded with the sample
        return None, False, f"{type(error).__name__}: {error}"


def run_vllm(root, arm, rows, prompts, record, max_tokens):
    import torch
    from vllm import LLM, SamplingParams

    versions = {name: importlib.metadata.version(name) for name in REQUIRED}
    if versions != REQUIRED:
        raise RuntimeError(f"Benchmark dependency versions differ: {versions}; expected {REQUIRED}")
    if not torch.cuda.is_available():
        raise RuntimeError("vLLM needs a CUDA GPU")
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
    stops = stop_ids(llm.get_tokenizer())
    params = SamplingParams(**{**PROTOCOL, "max_tokens": max_tokens}, stop_token_ids=stops)
    record.update(versions=versions, stop_token_ids=stops, sampling_params=repr(params))
    log(root, "generation_started", arm=arm, engine="vllm", problems=len(rows))
    outputs = llm.generate([p["prompt"] for p in prompts], params)
    samples, prompt_ids = [], {}
    for row, output in zip(rows, outputs, strict=True):
        if sorted(s.index for s in output.outputs) != list(range(SAMPLES)):
            raise ValueError(f"Incomplete or duplicated samples for {row['id']}")
        prompt_ids[row["id"]] = list(output.prompt_token_ids)
        for sample in sorted(output.outputs, key=lambda s: s.index):
            if sample.finish_reason not in {"stop", "length"}:
                raise ValueError(f"Unexpected finish reason: {sample.finish_reason}")
            samples.append(
                {
                    "id": row["id"],
                    "sample_index": sample.index,
                    "text": sample.text,
                    "truncated": sample.finish_reason == "length",
                    "finish_reason": sample.finish_reason,
                    "token_count": len(sample.token_ids),
                }
            )
    return samples, prompt_ids


def run_lab(root, arm, rows, prompts, grader, folder, record, max_tokens, batch_size, device):
    import torch

    from oumi_weights_lab.lab import load_lab

    session = folder / "session"
    session.mkdir()
    lab = load_lab({**MODELS[arm], "device": device, "dtype": "bfloat16", "seed": 0}, session)
    stops = stop_ids(lab.tokenizer)
    # The lab's own rendering of each prompt, through the path evaluate() uses.
    chat = lab._chat_settings(CHAT)
    assert chat is not None
    prompt_ids = {}
    for row, prompt in zip(rows, prompts, strict=True):
        rendered = lab._render_chat(grader.build_prompt(row["problem"]), chat)
        if rendered != prompt["prompt"]:
            raise ValueError(f"The lab renders {row['id']} differently from the vLLM prompt")
        prompt_ids[row["id"]] = lab._prompt_ids(rendered)
    config = {
        "do_sample": True,
        "temperature": PROTOCOL["temperature"],
        "top_p": PROTOCOL["top_p"],
        "top_k": PROTOCOL["top_k"],
        "max_new_tokens": max_tokens,
        "eos_token_id": stops,
    }
    record.update(
        stop_token_ids=stops,
        generation_config=config,
        batch_size=batch_size,
        lab_version=importlib.metadata.version("oumi-weights-lab"),
        transformers=importlib.metadata.version("transformers"),
        model_class=type(lab.model).__name__,
    )
    log(root, "generation_started", arm=arm, engine="lab", problems=len(rows))
    summary = lab.evaluate(
        [
            {
                "id": row["id"],
                "prompt": grader.build_prompt(row["problem"]),
                "answer": row["answer"],
            }
            for row in rows
        ],
        name=f"truncation-{arm}",
        mode="sampled",
        checker=lambda text, answer: grade(grader, text, answer)[1],
        checker_id="pinned-upstream-mathqa",
        generation_config=config,
        seed=0,
        batch_size=batch_size,
        num_samples=SAMPLES,
        chat_template=CHAT,
    )
    record["lab_summary"] = {
        key: summary.get(key)
        for key in (
            "truncation_share",
            "mean_generated_tokens",
            "accuracy",
            "scoring_config",
            "prompt_format",
            "forward_passes",
        )
    }
    samples = []
    items = Path(summary["run_dir"]) / "items.jsonl"
    for row, line in zip(rows, items.read_text().splitlines(), strict=True):
        item = json.loads(line)
        if item["id"] != row["id"] or len(item["samples"]) != SAMPLES:
            raise ValueError(f"The lab returned unexpected samples for {row['id']}")
        for index, sample in enumerate(item["samples"]):
            samples.append(
                {
                    "id": row["id"],
                    "sample_index": index,
                    "text": sample["generated"],
                    "truncated": sample["finish_reason"] == "max_tokens",
                    "finish_reason": sample["finish_reason"],
                    "token_count": sample["generated_tokens"],
                }
            )
    record["peak_memory_bytes"] = torch.cuda.max_memory_allocated() if device == "cuda" else None
    return samples, prompt_ids


def run_child(root, arm, engine, max_tokens, batch_size, device):
    """Generate one model with one engine; write its samples and summary."""
    prepare_child_path()
    _, rows, prompts = read_prepared(root)
    grader = load_grader(root / "upstream_mathqa.py")
    folder = root / f"{arm}-{engine}"
    folder.mkdir(exist_ok=False)
    record = {
        "state": "running",
        "arm": arm,
        "engine": engine,
        "model": MODELS[arm],
        "max_tokens": max_tokens,
        "python": sys.version,
        "platform": platform.platform(),
        "build_metadata": build_metadata(),
        "started_at": time.time(),
    }
    write_json(folder / "summary.json", record)
    try:
        import torch

        if torch.cuda.is_available():
            record["gpu"] = torch.cuda.get_device_name(0)
        record["torch"] = torch.__version__
        log(root, "child_started", arm=arm, engine=engine)
        if engine == "vllm":
            samples, prompt_ids = run_vllm(root, arm, rows, prompts, record, max_tokens)
        else:
            samples, prompt_ids = run_lab(
                root, arm, rows, prompts, grader, folder, record, max_tokens, batch_size, device
            )
        answers = {row["id"]: row["answer"] for row in rows}
        with (folder / "samples.jsonl").open("x") as stream:
            for sample in samples:
                extracted, correct, error = grade(grader, sample["text"], answers[sample["id"]])
                sample.update(extracted=extracted, correct=correct, grader_error=error)
                stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
        write_json(folder / "prompt_ids.json", prompt_ids)
        total = len(samples)
        record.update(
            state="passed",
            samples=total,
            cut_share=sum(s["truncated"] for s in samples) / total,
            accuracy=sum(s["correct"] for s in samples) / total,
            mean_tokens=sum(s["token_count"] for s in samples) / total,
            grader_errors=sum(s["grader_error"] is not None for s in samples),
        )
        log(root, "child_finished", arm=arm, engine=engine, cut_share=record["cut_share"])
    except BaseException as error:
        record.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["duration_seconds"] = time.time() - record["started_at"]
        write_json(folder / "summary.json", record)


def per_problem(folder):
    cuts = {}
    for line in (folder / "samples.jsonl").read_text().splitlines():
        sample = json.loads(line)
        cuts.setdefault(sample["id"], []).append(sample["truncated"])
    return {key: sum(values) / len(values) for key, values in cuts.items()}


def bootstrap_interval(differences, resamples, seed, level):
    rng = random.Random(seed)
    n = len(differences)
    means = sorted(sum(rng.choices(differences, k=n)) / n for _ in range(resamples))
    tail = (1 - level) / 2
    return means[int(tail * resamples)], means[int((1 - tail) * resamples) - 1]


def compare(root, arm):
    lab, vllm = per_problem(root / f"{arm}-lab"), per_problem(root / f"{arm}-vllm")
    if set(lab) != set(vllm):
        raise ValueError(f"The engines sampled different problems for {arm}")
    ids = sorted(lab)
    differences = [lab[key] - vllm[key] for key in ids]
    low, high = bootstrap_interval(differences, **BOOTSTRAP)
    prompts = [json.loads((root / f"{arm}-{e}" / "prompt_ids.json").read_text()) for e in ENGINES]
    summaries = {e: json.loads((root / f"{arm}-{e}" / "summary.json").read_text()) for e in ENGINES}
    return {
        "problems": len(ids),
        "lab_cut_share": sum(lab.values()) / len(ids),
        "vllm_cut_share": sum(vllm.values()) / len(ids),
        "difference": sum(differences) / len(ids),
        "interval": [low, high],
        "interval_contains_zero": low <= 0 <= high,
        "prompt_ids_identical": prompts[0] == prompts[1],
        "accuracy": {e: summaries[e]["accuracy"] for e in ENGINES},
        "mean_tokens": {e: summaries[e]["mean_tokens"] for e in ENGINES},
        "seconds": {e: summaries[e]["duration_seconds"] for e in ENGINES},
    }


def run_engine(root, arm, engine, max_tokens, batch_size, device):
    """Run one child; if the lab runs out of GPU memory, retry at half the batch size."""
    env = dict(os.environ)
    if engine == "lab":
        # The eager KV cache grows by concatenation every step.
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    folder = root / f"{arm}-{engine}"
    while True:
        command = [sys.executable, str(Path(__file__).resolve()), "--output-dir", str(root)]
        command += ["--arm", arm, "--engine", engine, "--max-tokens", str(max_tokens)]
        command += ["--lab-batch-size", str(batch_size), "--device", device]
        if subprocess.run(command, env=env, check=False).returncode == 0:
            return
        error = json.loads((folder / "summary.json").read_text()).get("error", "")
        if engine != "lab" or "OutOfMemoryError" not in error or batch_size == 1:
            raise RuntimeError(f"{arm} on {engine} failed: {error}")
        shutil.move(folder, root / f"{arm}-{engine}-out-of-memory-at-{batch_size}")
        batch_size //= 2
        log(root, "retrying_smaller_batch", arm=arm, batch_size=batch_size)


def run_parent(root, arms, engines, max_tokens, batch_size, device):
    summary = {
        "state": "running",
        "criteria": CRITERIA,
        "protocol": PROTOCOL,
        "stop_tokens": STOP_TOKENS,
        "max_tokens": max_tokens,
        "models": {},
    }
    write_json(root / "truncation-summary.json", summary)
    log(root, "criteria_declared", criteria=CRITERIA)
    try:
        for arm in arms:
            for engine in engines:
                run_engine(root, arm, engine, max_tokens, batch_size, device)
            if set(engines) == set(ENGINES):
                summary["models"][arm] = compare(root, arm)
                log(root, "compared", arm=arm, **summary["models"][arm])
        complete = set(arms) == set(MODELS) and set(engines) == set(ENGINES)
        matched = all(
            m["interval_contains_zero"] and m["prompt_ids_identical"]
            for m in summary["models"].values()
        )
        full = complete and max_tokens == PROTOCOL["max_tokens"]
        summary["state"] = "passed" if matched else "mismatch"
        summary["verification_complete"] = full and matched
    except BaseException as error:
        summary.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(root / "truncation-summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-problems", type=int, default=40)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=PROTOCOL["max_tokens"],
        help="Below the protocol's 32768 is a smoke test, never verification",
    )
    parser.add_argument("--lab-batch-size", type=int, default=160)
    parser.add_argument("--device", default="cuda", help="The lab's device; vLLM always uses CUDA")
    parser.add_argument("--arms", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--engines", nargs="+", choices=ENGINES, default=list(ENGINES))
    parser.add_argument("--grader-source", type=Path)
    parser.add_argument("--arm", choices=list(MODELS), help=argparse.SUPPRESS)
    parser.add_argument("--engine", choices=ENGINES, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = args.output_dir.expanduser().resolve()
    if args.arm:
        run_child(root, args.arm, args.engine, args.max_tokens, args.lab_batch_size, args.device)
        return {"state": "passed"}
    prepare(root, max_problems=args.max_problems, grader_source=args.grader_source)
    return run_parent(
        root, args.arms, args.engines, args.max_tokens, args.lab_batch_size, args.device
    )


if __name__ == "__main__":
    result = main()
    if result["state"] != "passed":
        raise SystemExit(1)
