# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reproduce PR #7477 host-observed embedding-wrapper latency on one CUDA GPU.

Run from the PR checkout with PYTHONPATH=$PWD. Baseline and fixed methods share
weights and inputs. Only the historical wrapper is loaded from --baseline-ref.
"""

import argparse
import ast
import gc
import json
import os
import random
import statistics
import subprocess
import textwrap
import time
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from transformers import Qwen2_5OmniThinkerConfig
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.multimodal.utils import set_mm_embedding_modality

from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_thinker import (
    Qwen2_5OmniThinkerForConditionalGeneration as Thinker,
)
from vllm_omni.platforms import current_omni_platform


def baseline_method(cls, ref):
    path = "vllm_omni/model_executor/models/qwen2_5_omni/qwen2_5_omni_thinker.py"
    source = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[2]), "show", f"{ref}:{path}"], text=True
    )
    tree = ast.parse(source)
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls.__name__)
    node = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "embed_input_ids")
    method = textwrap.dedent(ast.get_source_segment(source, node))
    # Preserve zero-argument super() against the real model class and MRO.
    factory = "from __future__ import annotations\ndef make_method(cls):\n    __class__ = cls\n"
    factory += textwrap.indent(method, "    ") + "\n    return embed_input_ids\n"
    namespace = dict(cls.embed_input_ids.__globals__)
    exec(compile(factory, f"{ref}:{path}", "exec"), namespace)
    return namespace["make_method"](cls)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--baseline-ref", default="01a2f93256975c7ff9565c5414b0fe0225bc4765")
parser.add_argument("--cpu-core", type=int, default=sorted(os.sched_getaffinity(0))[0])
parser.add_argument("--lengths", type=int, nargs="+", default=[1172])
parser.add_argument("--warmup", type=int, default=50)
parser.add_argument("--blocks", type=int, default=20)
parser.add_argument("--iterations", type=int, default=20)
parser.add_argument("--output-dir", type=Path, default=Path("micro-results"))
args = parser.parse_args()
if min(args.lengths) < 300 or min(args.blocks, args.iterations) < 1 or args.warmup < 0:
    parser.error("Require lengths >= 300, positive blocks/iterations, and nonnegative warmup")
if args.cpu_core not in os.sched_getaffinity(0):
    parser.error("CPU core is outside the allowed affinity set")
ROOT, MODEL, CORE = args.output_dir, args.model, args.cpu_core
ROOT.mkdir(parents=True, exist_ok=True)
CONFIG = json.loads((MODEL / "config.json").read_text())["thinker_config"]
HIDDEN = CONFIG["text_config"]["hidden_size"]
VOCAB = CONFIG["text_config"]["vocab_size"]
IDS = {m: CONFIG[m + "_token_index"] for m in ("audio", "image", "video")}
METHODS = {"baseline": baseline_method(Thinker, args.baseline_ref), "fixed": Thinker.embed_input_ids}
RESULTS, VALIDATIONS = [], []
os.sched_setaffinity(0, {CORE})
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
gc.disable()


def cpu_frequency():
    path = Path(f"/sys/devices/system/cpu/cpu{CORE}/cpufreq/scaling_cur_freq")
    if path.exists():
        return float(path.read_text()) / 1000
    for section in Path("/proc/cpuinfo").read_text().split("\n\n"):
        fields = dict(line.split(":", 1) for line in section.splitlines() if ":" in line)
        fields = {k.strip(): v.strip() for k, v in fields.items()}
        if fields.get("processor") == str(CORE):
            return float(fields.get("cpu MHz", "nan"))


def payload(case, length, device):
    generator = torch.Generator().manual_seed(7451 + length)
    ids = torch.randint(2, 10000, (length,), generator=generator)
    specs = {
        "text": [],
        "audio": [("audio", 100)],
        "image": [("image", 16)],
        "video": [("video", 128)],
        "mixed": [("audio", 100), ("image", 16), ("video", 128)],
        "interleaved": [("video", 64), ("audio", 50), ("video", 64), ("audio", 50)],
    }[case]
    offset = 44
    counts = {}
    for modality, count in specs:
        ids[offset : offset + count] = IDS[modality]
        counts[modality] = counts.get(modality, 0) + count
        offset += count + (0 if case == "interleaved" else 2)
    mask = torch.zeros(length, dtype=torch.bool)
    for token in IDS.values():
        mask |= ids == token
    mm = [
        set_mm_embedding_modality(torch.full((count, HIDDEN), i + 1, device=device, dtype=torch.bfloat16), modality)
        for i, (modality, count) in enumerate(counts.items())
    ]
    return ids.to(device=device, dtype=torch.int32), None if case == "text" else mm, None if case == "text" else mask


def once(fn, cuda):
    # Drain preceding work before the host timer; the production function's
    # own D2H synchronizations remain inside the timed call.
    if cuda:
        current_omni_platform.synchronize()
    w0 = time.perf_counter_ns()
    c0 = time.thread_time_ns()
    value = fn()
    c1 = time.thread_time_ns()
    w1 = time.perf_counter_ns()
    if cuda:
        current_omni_platform.synchronize()
    w2 = time.perf_counter_ns() if cuda else w1
    del value
    return [(w1 - w0) / 1000, (c1 - c0) / 1000, (w2 - w0) / 1000]


def measure(functions, device, case, length, scope, rounds=20, iterations=20, warmups=50):
    cuda = device == "cuda"
    for fn in functions.values():
        for _ in range(warmups):
            once(fn, cuda)
    order_rng = random.Random(7451 + length + len(case) + len(scope))
    directions = [i % 2 for i in range(rounds)]
    order_rng.shuffle(directions)
    names = list(functions)
    for block, direction in enumerate(directions):
        order = names if direction == 0 else names[::-1]
        for name in order:
            frequency_start = cpu_frequency()
            wall_start = time.time()
            samples = [once(functions[name], cuda) for _ in range(iterations)]
            wall_end = time.time()
            RESULTS.append(
                dict(
                    device=device,
                    case=case,
                    tokens=length,
                    scope=scope,
                    variant=name,
                    block=block,
                    iterations=iterations,
                    samples_us=samples,
                    wall_start=wall_start,
                    wall_end=wall_end,
                    cpu_mhz_start=frequency_start,
                    cpu_mhz_end=cpu_frequency(),
                )
            )
    (ROOT / "micro-raw.json").write_text(json.dumps(RESULTS))
    current = [r for r in RESULTS if (r["device"], r["case"], r["tokens"], r["scope"]) == (device, case, length, scope)]
    print(
        "RESULT",
        json.dumps(
            {
                "device": device,
                "case": case,
                "tokens": length,
                "scope": scope,
                "host_median_us": {
                    v: statistics.median(t[0] for r in current if r["variant"] == v for t in r["samples_us"])
                    for v in names
                },
            }
        ),
        flush=True,
    )


def build_model(weights, device):
    model = Thinker.__new__(Thinker)
    nn.Module.__init__(model)
    model.config = Qwen2_5OmniThinkerConfig(**CONFIG)
    model._has_oov_mm_tokens = False
    table = VocabParallelEmbedding(VOCAB, HIDDEN, params_dtype=torch.bfloat16, disable_tp=True)
    table.weight = nn.Parameter(weights.to(device), requires_grad=False)

    class LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = table

        def embed_input_ids(self, ids):
            return self.embed_tokens(ids)

    model.language_model = LM()
    return model


def validate(model, args, case, length, device):
    ids, mm, mask = args
    lm = model.get_language_model()
    original = lm.embed_input_ids
    outputs, counts = {}, {}
    for variant, method in METHODS.items():
        calls = []

        def counted(tokens):
            calls.append(tokens.numel())
            return original(tokens)

        lm.embed_input_ids = counted
        try:
            outputs[variant] = method(model, ids, mm, is_multimodal=mask)
        finally:
            lm.embed_input_ids = original
        counts[variant] = len(calls)
    assert torch.equal(outputs["baseline"], outputs["fixed"])
    assert counts == {"baseline": 1 if case == "text" else 2, "fixed": 1}
    VALIDATIONS.append(dict(case=case, tokens=length, device=device, calls=counts, same_input_wrapper_torch_equal=True))
    (ROOT / "validation.json").write_text(json.dumps(VALIDATIONS, indent=2))


@torch.inference_mode()
def main():
    import transformers
    import vllm

    metadata = dict(
        baseline_ref=args.baseline_ref,
        checkout=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        torch=torch.__version__,
        vllm=vllm.__version__,
        transformers=transformers.__version__,
        gpu=torch.cuda.get_device_name(),
        cpu_core=CORE,
        torch_cpu_threads=1,
        warmup=args.warmup,
        blocks=args.blocks,
        iterations=args.iterations,
        lengths=args.lengths,
        embedding_shape=[VOCAB, HIDDEN],
        dtype="bfloat16",
        input_ids_dtype="int32",
        mask_device="cpu",
        start_time=time.time(),
        metric_order=["host_observed_us", "calling_thread_cpu_us", "synchronized_completion_us"],
    )
    index = json.loads((MODEL / "model.safetensors.index.json").read_text())["weight_map"]
    key = "thinker.model.embed_tokens.weight"
    with safe_open(MODEL / index[key], framework="pt", device="cpu") as f:
        weights = f.get_tensor(key)
    assert weights.shape == (VOCAB, HIDDEN) and weights.dtype == torch.bfloat16
    model = build_model(weights, "cuda")
    measure({"timer_floor": lambda: None}, "cuda", "timer", 0, "overhead", rounds=10, iterations=100)
    for length in args.lengths:
        for case in ("text", "audio", "image", "video", "mixed", "interleaved"):
            ids, mm, mask = payload(case, length, "cuda")
            validate(model, (ids, mm, mask), case, length, "cuda")
            functions = {
                variant: (lambda method=method: method(model, ids, mm, is_multimodal=mask))
                for variant, method in METHODS.items()
            }
            measure(
                functions,
                "cuda",
                case,
                length,
                "whole_wrapper",
                rounds=args.blocks,
                iterations=args.iterations,
                warmups=args.warmup,
            )
    summary = []
    for length in args.lengths:
        for case in ("text", "audio", "image", "video", "mixed", "interleaved"):
            medians = {
                variant: statistics.median(
                    statistics.median(sample[0] for sample in record["samples_us"])
                    for record in RESULTS
                    if record["tokens"] == length and record["case"] == case and record["variant"] == variant
                )
                for variant in METHODS
            }
            summary.append(
                dict(
                    tokens=length,
                    case=case,
                    host_us=medians,
                    reduction_pct=100 * (1 - medians["fixed"] / medians["baseline"]),
                )
            )
    metadata["end_time"] = time.time()
    (ROOT / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
