# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare original/candidate Snake ops inside IndexTTS2's pretrained BigVGAN.

    git show <base>:vllm_omni/model_executor/models/common/snake_activation.py > /tmp/snake_before.py
    python benchmarks/kernels/benchmark_snake_decoder.py --baseline-source /tmp/snake_before.py

Loads the baseline Python module supplied by the caller. Uses fixed synthetic
log-mel inputs, not text-to-speech serving or perceptual audio evaluation.
Only Snake activations differ between the two copies of the same decoder.
"""

import argparse
import copy
import importlib.util
import json
import statistics

import torch
from vllm.triton_utils import triton

from vllm_omni.model_executor.models.common.snake_activation import Snake, SnakeBeta
from vllm_omni.model_executor.models.indextts2.s2mel.modules.bigvgan import BigVGAN, load_hparams_from_json
from vllm_omni.transformers_utils.repo_utils import hf_api


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", required=True)
    parser.add_argument("--model", default="nvidia/bigvgan_v2_22khz_80band_256x")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--frames", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if min(args.rounds, *args.frames) <= 0:
        parser.error("rounds and frames must be positive")
    revision = hf_api().model_info(args.model, revision=args.revision).sha
    config = hf_api().hf_hub_download(args.model, "config.json", revision=revision)
    checkpoint = hf_api().hf_hub_download(args.model, "bigvgan_generator.pt", revision=revision)
    model = BigVGAN(load_hparams_from_json(config))
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)["generator"]
    # Omni regenerates these fixed filters as non-persistent buffers, and
    # DownSample1d no longer wraps its filter in a separate lowpass module.
    buffers = dict(model.named_buffers())
    for key in list(weights):
        if key.endswith((".upsample.filter", ".downsample.lowpass.filter")):
            buffer_key = key.replace(".downsample.lowpass.filter", ".downsample.filter")
            torch.testing.assert_close(weights.pop(key), buffers[buffer_key], rtol=0, atol=1e-7)
    model.load_state_dict(weights, strict=True)
    model.remove_weight_norm()
    model = model.eval().to("cuda")
    original = copy.deepcopy(model)

    spec = importlib.util.spec_from_file_location("snake_before", args.baseline_source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load baseline-source")
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    count = 0
    for parent in original.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, SnakeBeta):
                cls = baseline.Snake if isinstance(child, Snake) else baseline.SnakeBeta
                replacement = cls(child.in_features, alpha_logscale=child.alpha_logscale).to("cuda").eval()
                replacement.load_state_dict(child.state_dict(), strict=True)
                setattr(parent, name, replacement)
                count += 1
    assert count > 0
    print(
        json.dumps(
            {
                "model": args.model,
                "revision": revision,
                "device": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "dtype": "float32",
                "activation_count": count,
                "input": "synthetic log-mel, seed 42",
                "rounds": args.rounds,
                "warmup": 10,
                "rep_ms": 200,
            }
        ),
        flush=True,
    )
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    for frames in args.frames:
        mel = torch.randn(1, model.h.num_mels, frames, device="cuda") * 1.5 - 3
        expected, actual = original(mel), model(mel)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.isfinite(actual).all()
        for candidate in (original, model):
            for _ in range(10):
                candidate(mel)
        samples: dict[str, list[float]] = {"original": [], "candidate": []}
        for round_index in range(args.rounds):
            order = [("original", original), ("candidate", model)]
            if round_index % 2:
                order.reverse()
            for name, candidate in order:
                samples[name].append(
                    triton.testing.do_bench_cudagraph(lambda: candidate(mel), rep=200, return_mode="median")
                )
        assert SnakeBeta._triton_kernel and baseline.SnakeBeta._triton_kernel, "Snake Triton fallback occurred"
        medians = {key: statistics.median(values) for key, values in samples.items()}
        print(
            json.dumps(
                {
                    "mel_frames": frames,
                    "output_samples": actual.numel(),
                    "max_abs_error": (actual - expected).abs().max().item(),
                    "round_ms": samples,
                    "median_ms": medians,
                    "speedup": medians["original"] / medians["candidate"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
