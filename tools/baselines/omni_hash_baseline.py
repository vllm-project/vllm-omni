# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Record / verify a deterministic text+audio output hash baseline for an Omni model.

This is the cheap "output is bit-identical" gate for the behaviour-preserving
worker/model-runner changes: record a baseline on the bench box
BEFORE any code moves, then re-run with ``--check`` after each refactor.

Determinism: **batch size 1** (one prompt per ``generate`` call) + **fixed seed**.
Thinker and code2wav run greedy (temperature 0); the talker's speech sampler needs
temperature to emit valid codec tokens, so it runs at its natural sampling with a
fixed seed (reproducible via seed + batch-1, not pure-greedy). 3 fixed prompts, run
with async output OFF and ON (6 entries total). Hashes are only meaningful WITHIN a single recorded
environment (driver/cuda/torch/gpu), so the baseline JSON records the env and
``--check`` refuses to compare across a changed env unless ``--force-env``.

Baseline JSON schema (results keyed by BOTH prompt_id and async_mode so the
matrix is self-identifying)::

    {
      "model": "...",
      "commit": "<git sha>",
      "env": {"driver": "<nvidia driver>", "cuda": "<cuda build>", "torch": "...", "gpu": "..."},
      "results": [
        {"prompt_id": 0, "async_mode": false, "text_sha": "...", "wav_sha": "..."},
        ...  # exactly len(FIXED_PROMPTS) * 2 entries
      ]
    }

The file content is JSON, but committed baselines use a NON-``.json`` extension
(``.baseline``) because the repo ``.gitignore`` excludes ``*.json``; ``--check``
loads by content, so the extension does not matter to the script.

Usage::

    # record (run twice, confirm the two runs are identical before committing)
    python tools/baselines/omni_hash_baseline.py <model> --yaml <deploy.yaml> \
        --out tools/baselines/<model>_hash.baseline
    # verify after a refactor (same bench box)
    python tools/baselines/omni_hash_baseline.py <model> --yaml <deploy.yaml> \
        --check tools/baselines/<model>_hash.baseline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys

import torch
from vllm.sampling_params import SamplingParams

from vllm_omni.entrypoints.omni import Omni

SEED = 42
DEFAULT_SYSTEM = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)

# 3 fixed prompts — do NOT edit once a baseline is recorded (prompt_id is positional).
FIXED_PROMPTS = [
    "Explain the system architecture for a scalable audio generation pipeline. Answer in 15 words.",
    "Describe how ocean tides work. Answer in 15 words.",
    "List three benefits of unit tests. Answer in 15 words.",
]
ASYNC_MODES = (False, True)


def _text_prompt(question: str) -> dict:
    return {
        "prompt": (
            f"<|im_start|>system\n{DEFAULT_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    }


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _driver_version() -> str:
    """The actual NVIDIA driver version (NOT torch.version.cuda, which is the build)."""
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True)
        return out.strip().splitlines()[0].strip()
    except Exception:
        return "unknown"


def _env_fingerprint() -> dict:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {
        "driver": _driver_version(),
        "cuda": str(torch.version.cuda),
        "torch": torch.__version__,
        "gpu": gpu,
    }


def _sampling_params() -> list[SamplingParams]:
    # thinker + code2wav: greedy. talker: seeded (temperature>0) — its speech sampler
    # needs temperature to emit valid codec tokens; the fixed seed keeps it reproducible.
    thinker = SamplingParams(
        temperature=0.0, top_p=1.0, top_k=-1, max_tokens=128, seed=SEED, detokenize=True, repetition_penalty=1.1
    )
    talker = SamplingParams(
        temperature=0.9, top_p=0.8, top_k=40, max_tokens=256, seed=SEED, detokenize=True, repetition_penalty=1.05
    )
    code2wav = SamplingParams(
        temperature=0.0, top_p=1.0, top_k=-1, max_tokens=256, seed=SEED, detokenize=True, repetition_penalty=1.1
    )
    return [thinker, talker, code2wav]


def _audio_bytes(audio) -> bytes:
    """Deterministic bytes for one request's audio, handling chunked (list) output.

    async-chunk mode can emit a list of audio chunks instead of a single tensor;
    concatenate them along the last axis in emit order.
    """
    if isinstance(audio, (list, tuple)):
        chunks = [a for a in audio if a is not None]
        if not chunks:
            return b""
        audio = torch.cat([torch.as_tensor(a).detach().to(torch.float32).cpu() for a in chunks], dim=-1)
    if audio is None:
        return b""
    return torch.as_tensor(audio).detach().to(torch.float32).cpu().numpy().tobytes()


def _run_once(model: str, yaml_path: str, *, async_mode: bool) -> list[dict]:
    """One deterministic pass, batch size 1 per prompt; returns per-prompt hashes."""
    # ``async_chunk`` is the repo's documented async-output toggle for streaming
    # (see tests/e2e/accuracy/qwen3_omni). Confirm the exact knob at bench time.
    omni = Omni(
        model,
        stage_configs_path=yaml_path,
        stage_init_timeout=1800,
        init_timeout=3600,
        async_chunk=async_mode,
    )
    results: list[dict] = []
    try:
        for prompt_id, question in enumerate(FIXED_PROMPTS):
            text: str | None = None
            wav: bytes = b""
            # Batch size 1: submit a single prompt so scheduling is deterministic.
            for step in omni.generate([_text_prompt(question)], _sampling_params(), py_generator=False):
                out = step.request_output
                if step.final_output_type == "text":
                    text = out.outputs[0].text  # exact text, no strip -> truly bit-identical
                elif step.final_output_type == "audio":
                    wav = _audio_bytes(out.outputs[0].multimodal_output["audio"])
            if text is None:
                raise RuntimeError(f"prompt_id={prompt_id} async={async_mode}: produced no text output")
            if not wav:
                raise RuntimeError(f"prompt_id={prompt_id} async={async_mode}: produced no/empty audio output")
            results.append(
                {
                    "prompt_id": prompt_id,
                    "async_mode": async_mode,
                    "text_sha": _sha(text.encode("utf-8")),
                    "wav_sha": _sha(wav),
                }
            )
    finally:
        omni.close()
    return results


def _record(model: str, yaml_path: str) -> dict:
    results: list[dict] = []
    for async_mode in ASYNC_MODES:
        results.extend(_run_once(model, yaml_path, async_mode=async_mode))
    return {"model": model, "commit": _git_commit(), "env": _env_fingerprint(), "results": results}


def _key(record: dict):
    return (record["prompt_id"], record["async_mode"])


def _expected_keys() -> set:
    return {(pid, mode) for pid in range(len(FIXED_PROMPTS)) for mode in ASYNC_MODES}


def _check(model: str, yaml_path: str, baseline_path: str, *, force_env: bool) -> int:
    with open(baseline_path) as fh:
        baseline = json.load(fh)

    now = _record(model, yaml_path)
    if not force_env and baseline.get("env") != now["env"]:
        print(
            f"ENV MISMATCH: baseline={baseline.get('env')} now={now['env']} — hashes are "
            "env-specific. Re-record on this box or pass --force-env.",
            file=sys.stderr,
        )
        return 2

    problems: list[str] = []

    def _index_unique(records: list[dict], label: str) -> dict:
        by_key: dict = {}
        for record in records:
            key = _key(record)
            if key in by_key:
                problems.append(f"{label}: duplicate entry for (prompt_id={key[0]}, async={key[1]})")
            by_key[key] = record
        return by_key

    base_by_key = _index_unique(baseline["results"], "baseline")
    now_by_key = _index_unique(now["results"], "current-run")
    expected = _expected_keys()

    # Guard against duplicate-collapse: the record count must match the key count.
    if len(baseline["results"]) != len(expected):
        problems.append(f"baseline has {len(baseline['results'])} records, expected exactly {len(expected)}")
    if set(now_by_key) != expected:
        problems.append(f"current-run key set {sorted(set(now_by_key))} != expected {sorted(expected)}")
    if set(base_by_key) != set(now_by_key):
        missing = sorted(set(base_by_key) - set(now_by_key))
        extra = sorted(set(now_by_key) - set(base_by_key))
        problems.append(f"baseline/run key-set mismatch: missing-from-run={missing} extra-in-run={extra}")
    for key in sorted(set(base_by_key) & set(now_by_key)):
        b, r = base_by_key[key], now_by_key[key]
        if (b["text_sha"], b["wav_sha"]) != (r["text_sha"], r["wav_sha"]):
            problems.append(
                f"HASH MISMATCH (prompt_id={key[0]}, async={key[1]}): "
                f"text {b['text_sha'][:8]}!={r['text_sha'][:8]} / wav {b['wav_sha'][:8]}!={r['wav_sha'][:8]}"
            )

    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1
    print(f"OK: all {len(expected)} (prompt_id, async_mode) entries match the baseline.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model")
    p.add_argument("--yaml", required=True, help="Stage deploy config path")
    p.add_argument("--out", help="Write a new baseline JSON to this path")
    p.add_argument("--check", metavar="BASELINE_JSON", help="Verify current output against this baseline")
    p.add_argument("--force-env", action="store_true", help="Compare hashes even if the recorded env differs")
    args = p.parse_args()

    if args.check:
        return _check(args.model, args.yaml, args.check, force_env=args.force_env)

    baseline = _record(args.model, args.yaml)
    payload = json.dumps(baseline, indent=2)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(payload)
        print(f"Wrote baseline: {args.out}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
