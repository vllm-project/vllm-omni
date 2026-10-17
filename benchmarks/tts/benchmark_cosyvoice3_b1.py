# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""V1 RAS correctness + paired sampler replay, using the real installed runtime.

Run from the repository root with ``python -m benchmarks.tts.benchmark_cosyvoice3_b1``.
No model weights are needed. This measures the sampler, not Stage 0 or TTS E2E.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Case:
    batch: int
    mixed: bool
    history: str
    seeds: str
    dtype: str

    @property
    def name(self):
        return f"b{self.batch}-{'mixed' if self.mixed else 'random'}-{self.history}-{self.seeds}-{self.dtype}"


def load_runtime():
    # A fresh benchmark process must bind both arms AFTER Omni patches vLLM.
    import vllm_omni  # isort: skip # noqa: F401

    from vllm.sampling_params import SamplingParams
    from vllm.v1.sample.logits_processor.state import LogitsProcessors
    from vllm.v1.sample.metadata import SamplingMetadata
    from vllm.v1.sample.ops.topk_topp_sampler import random_sample

    from benchmarks.tts.cosyvoice3_b1_reference import BASELINE_COMMIT, ReferenceCosyVoice3Sampler
    from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import CosyVoice3Model
    from vllm_omni.worker.sampling_utils import call_model_sampler

    if not (
        ReferenceCosyVoice3Sampler._ras_sample_batch.__globals__["random_sample"]
        is CosyVoice3Model._ras_sample_batch.__globals__["random_sample"]
        is random_sample
    ):
        raise RuntimeError("Baseline and head must use the same initialized Omni random_sample")

    return SimpleNamespace(
        head=CosyVoice3Model,
        reference=ReferenceCosyVoice3Sampler,
        params=SamplingParams,
        metadata=SamplingMetadata,
        processors=LogitsProcessors,
        call=call_model_sampler,
        random_sample=random_sample,
        baseline_commit=BASELINE_COMMIT,
    )


def make_arm(runtime, cls, case, device):
    model = object.__new__(cls)
    torch.nn.Module.__init__(model)
    model.model_stage = "cosyvoice3_talker"
    model.config = SimpleNamespace(
        vocab_size=151923,
        llm={"speech_token_size": 6561, "sampling": {"top_p": 0.8, "top_k": 25, "win_size": 10, "tau_r": 0.1}},
    )
    params = [
        runtime.params(
            temperature=0.0 if case.mixed and i % 2 == 0 else 0.7 + 0.1 * (i % 3),
            seed=100 + i if case.seeds == "all" or (case.seeds == "partial" and i % 2 == 1) else None,
            top_p=0.8,
            top_k=25,
            repetition_penalty=1.0001,
        )
        for i in range(case.batch)
    ]
    histories = {
        "empty": [],
        "quiet": [0] * 10,
        "repeat": [1] * 10,
        "single_valid": [1] * 10,
    }
    metadata = runtime.metadata(
        temperature=torch.tensor([p.temperature for p in params], device=device),
        all_greedy=False,
        all_random=not case.mixed,
        top_p=torch.full((case.batch,), 0.8, device=device),
        top_k=torch.full((case.batch,), 25, dtype=torch.int32, device=device),
        generators={
            i: torch.Generator(device=device).manual_seed(p.seed) for i, p in enumerate(params) if p.seed is not None
        },
        max_num_logprobs=None,
        no_penalties=False,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(case.batch, device=device),
        presence_penalties=torch.zeros(case.batch, device=device),
        repetition_penalties=torch.full((case.batch,), 1.0001, device=device),
        output_token_ids=[list(histories[case.history]) for _ in params],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=runtime.processors(),
    )
    batch = SimpleNamespace(req_ids=[f"r{i}" for i in range(case.batch)])
    requests = {req: SimpleNamespace(sampling_params=p) for req, p in zip(batch.req_ids, params)}
    return SimpleNamespace(model=model, metadata=metadata, batch=batch, requests=requests)


def inputs(case, device):
    values = torch.randn(case.batch, 6761, generator=torch.Generator().manual_seed(42))
    values[:, -1] = float("-inf")
    if case.history == "single_valid":
        values.fill_(float("-inf"))
    # Force primary token 1. Repeat/quiet isolate whether conditional RAS fires;
    # replacement still samples a broad distribution (except single_valid).
    values[:, 1] = 12.0
    return values.to(device=device, dtype=getattr(torch, case.dtype))


def sample(runtime, arm, logits):
    return runtime.call(
        arm.model, arm.model.sample, logits, arm.metadata, input_batch=arm.batch, requests=arm.requests
    ).sampled_token_ids


def check_case(runtime, case, device, steps):
    """Compare seeded token/RNG trajectories; unseeded rows only have a distribution contract."""
    arms = [make_arm(runtime, cls, case, device) for cls in (runtime.reference, runtime.head)]
    logits = inputs(case, device)
    for step in range(steps):
        results = [sample(runtime, arm, logits.clone()) for arm in arms]
        for arm, result in zip(arms, results):
            assert result.dtype == torch.int32 and result.shape == (case.batch, 1), case.name
            assert bool(((result >= 0) & (result < logits.shape[1])).all()), case.name
            assert bool(torch.isfinite(logits.gather(1, result.long())).all()), case.name
            for history, token in zip(arm.metadata.output_token_ids, result[:, 0].tolist()):
                history.append(token)
        for row in range(case.batch):
            if row in arms[0].metadata.generators or (case.mixed and row % 2 == 0):
                assert torch.equal(results[0][row], results[1][row]), (case.name, step, row, "token mismatch")
            if row in arms[0].metadata.generators:
                assert torch.equal(
                    arms[0].metadata.generators[row].get_state(), arms[1].metadata.generators[row].get_state()
                ), (case.name, step, row, "RNG state mismatch")


def paired_summary(pairs):
    saved = np.array([p["baseline"]["completed_ms"] - p["head"]["completed_ms"] for p in pairs])
    # Resample PAIRS, not the two arms independently. Descriptive bootstrap CI;
    # serial correlation and external contention still need trace inspection.
    draws = np.random.default_rng(42).choice(saved, size=(5000, len(saved)), replace=True).mean(axis=1)
    return {
        "baseline_median_ms": statistics.median(p["baseline"]["completed_ms"] for p in pairs),
        "head_median_ms": statistics.median(p["head"]["completed_ms"] for p in pairs),
        "paired_mean_saved_ms": float(saved.mean()),
        "paired_median_saved_ms": float(np.median(saved)),
        "paired_mean_saved_ci95_ms": np.quantile(draws, [0.025, 0.975]).tolist(),
        "positive_pairs": int((saved > 0).sum()),
        "pairs": len(pairs),
    }


def measure(runtime, arm, logits, steps, device):
    for row, generator in arm.metadata.generators.items():
        generator.manual_seed(100 + row)
    torch.manual_seed(7654)
    # Cloning and RNG reset must not be billed to sampler execution.
    batches = [logits.clone() for _ in range(steps)]
    if device == "cuda":
        torch.accelerator.synchronize()
    start = time.perf_counter_ns()
    for batch in batches:
        sample(runtime, arm, batch)
    enqueued = time.perf_counter_ns()
    if device == "cuda":
        torch.accelerator.synchronize()
    completed = time.perf_counter_ns()
    return {"host_ms": (enqueued - start) / 1e6 / steps, "completed_ms": (completed - start) / 1e6 / steps}


def profile_case(runtime, case, device, output):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    traces = {}
    for name, cls in (("baseline", runtime.reference), ("head", runtime.head)):
        arm = make_arm(runtime, cls, case, device)
        logits = inputs(case, device)
        measure(runtime, arm, logits, 4, device)
        with torch.profiler.profile(activities=activities) as prof:
            measure(runtime, arm, logits, 8, device)
        path = output / f"{case.name}-{name}.json"
        prof.export_chrome_trace(str(path))
        traces[name] = str(path)
    return traces


def environment(runtime, device):
    head_file = Path(inspect.getfile(runtime.head)).resolve()
    expected = ROOT / "vllm_omni/model_executor/models/cosyvoice3/cosyvoice3.py"
    if head_file != expected:
        raise RuntimeError(
            f"Imported another checkout: {head_file}. Install this checkout with pip install -e . --no-deps"
        )
    files = [head_file, Path(inspect.getfile(runtime.reference)), Path(inspect.getfile(runtime.random_sample))]
    if hasattr(runtime.random_sample, "__wrapped__"):
        from vllm_omni.utils import seeded_exponential

        files.extend([Path(inspect.getfile(runtime.random_sample.__wrapped__)), Path(seeded_exponential.__file__)])
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "vllm": importlib.metadata.version("vllm"),
        "cuda_runtime": torch.version.cuda,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "baseline_commit": runtime.baseline_commit,
        "head_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output", type=Path, default=Path("b1-results/sampler.json"))
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--check-steps", type=int, default=40)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--dtypes", nargs="+", choices=["float32", "bfloat16"], default=["float32", "bfloat16"])
    parser.add_argument("--trace", action="store_true", help="separate CPU/CUDA traces for two representative cases")
    parser.add_argument("--quick", action="store_true", help="small smoke matrix; not a performance acceptance run")
    args = parser.parse_args()
    if min(args.repeats, args.steps, args.check_steps, *args.batches) < 1 or args.warmup < 0:
        parser.error("counts must be positive; warmup may be zero")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is required; this command never silently switches to CPU")
    runtime = load_runtime()
    torch.set_num_threads(1)
    cases = [
        Case(batch, mixed, history, seeds, dtype)
        for batch in args.batches
        for mixed in (False, True)
        if not mixed or batch > 1
        for history in ("empty", "quiet", "repeat", "single_valid")
        for seeds in ("all", "none", "partial")
        if seeds != "partial" or batch > 1
        for dtype in args.dtypes
    ]
    if args.quick:
        cases = [c for c in cases if c.batch <= 4 and c.history in ("empty", "repeat") and c.seeds == "all"]
        args.repeats, args.warmup, args.steps, args.check_steps = 3, 1, 4, 8
    report = {
        "scope": "fixed-logits/history sampler replay, including host parameter gathering; not Stage-0/E2E",
        "environment": environment(runtime, args.device),
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "status": "running",
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    save()
    try:
        # Finish every correctness case before timing any case.
        for case in cases:
            check_case(runtime, case, args.device, args.check_steps)
        report["correctness_cases_passed"] = len(cases)
        for case in cases:
            arms = [make_arm(runtime, cls, case, args.device) for cls in (runtime.reference, runtime.head)]
            logits = inputs(case, args.device)
            for _ in range(args.warmup):
                for arm in arms:
                    measure(runtime, arm, logits, args.steps, args.device)
            pairs = []
            for round_id in range(args.repeats):
                pair = {}
                for index in (0, 1) if round_id % 2 == 0 else (1, 0):
                    pair[("baseline", "head")[index]] = measure(runtime, arms[index], logits, args.steps, args.device)
                pairs.append(pair)
            result = {"case": asdict(case), "summary": paired_summary(pairs), "raw_pairs": pairs}
            report["cases"].append(result)
            save()
            print(case.name, json.dumps(result["summary"]), flush=True)
        # Profiling is outside ALL timing rounds and does not contaminate them.
        if args.trace:
            trace_dir = args.output.parent / "traces"
            trace_dir.mkdir(exist_ok=True)
            report["traces"] = {
                case.name: profile_case(runtime, case, args.device, trace_dir)
                for case in cases
                if case.dtype == "float32"
                and case.seeds in ("all", "none")
                and case.history == "repeat"
                and ((case.batch == 1 and not case.mixed) or (case.batch == 4 and case.mixed))
            }
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
