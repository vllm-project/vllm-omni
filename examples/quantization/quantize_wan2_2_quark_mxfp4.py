#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantize Wan2.2-T2V-A14B to a calibrated Quark MXFP4 checkpoint (offline).

Produces a diffusers-style transformer directory whose weights are SmoothQuant-
calibrated and MXFP4-rounded (bf16 tensors carrying FP4-grid values + a
quantization config marking the checkpoint serialized). vllm-omni's ROCm offline
loader (ROCmMxfp4OfflineLinearMethod, gfx950) reads it, packs to the AITER FP4
layout at load, and runs the same gemm_a4w4 as the online path - only the weights
are calibrated.

The A14B cascade has TWO transformers (transformer + transformer_2); both are
quantized with the same config.

With --pack, weights are packed to the AITER FP4 layout (weight_shuffle +
weight_scale, ~3.7x smaller on disk, no pack at load); vllm-omni's
ROCmMxfp4PackedLinearMethod (gfx950) loads them directly. Without --pack, the export
stays calibrated bf16 and is packed at load - portable across AITER versions.

Example:
    python examples/quantization/quantize_wan2_2_quark_mxfp4.py \\
        --model /path/to/Wan2.2-T2V-A14B-Diffusers \\
        --output /path/to/wan2.2-t2v-a14b-quark-mxfp4 \\
        --n-prompts 4 --n-steps 4 [--pack]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time

import torch
from quark_mxfp4_scaling_maps import (
    get_decoder_layers_attr,
    get_exclude_patterns,
    get_rotation_map,
    get_scaling_map,
    shim_rotation_config,
)

DEFAULT_CALIB_PROMPTS = [
    "A serene lakeside sunrise with mist over the water.",
    "A bustling city street at night with neon signs and rain.",
    "A close-up of a hummingbird hovering over a red flower.",
    "Aerial view of ocean waves crashing on a rocky coast.",
    "A cat walking through tall grass in a sunny meadow.",
    "Time-lapse of clouds moving over snowy mountain peaks.",
]

# Keys Quark adds that vllm-omni's WanTransformer3DModel does not expect. The offline
# loader re-derives the per-32 scale from the calibrated weight via AITER, so these
# are redundant and are stripped from the saved checkpoint.
_DROP_KEY_MARKERS = ("_weight_quantizer", "_input_quantizer", "._amax")


class _ToSafe:
    """Make a non-tensor value survive Quark's cache_model_inps `.to(device)` call.

    Quark's calib input-caching (built for LLMs whose block kwargs are all tensors)
    calls .to(device) on EVERY value. Wan DiT blocks receive non-tensor kwargs
    (bool/None/int); wrapping them so .to() returns the original primitive lets the
    block forward receive the real value.
    """

    __slots__ = ("v",)

    def __init__(self, v):
        self.v = v

    def to(self, *a, **k):
        return self.v


class _SafeCalibLoader:
    """Re-iterable wrapper making non-tensor calib kwargs .to()-safe."""

    def __init__(self, dl):
        self._dl = dl

    def __len__(self):
        return len(self._dl)

    def __iter__(self):
        for sample in self._dl:
            if isinstance(sample, dict):
                yield {k: (v if isinstance(v, torch.Tensor) else _ToSafe(v)) for k, v in sample.items()}
            else:
                yield sample


def build_qconfig(model, alpha: float, smooth: bool, r2: bool):
    from quark.torch.quantization.config.config import (
        OCP_MXFP4Spec,
        QConfig,
        QLayerConfig,
        RotationConfig,
        SmoothQuantConfig,
    )

    scaling_layers = get_scaling_map(model)
    decoder_layers = get_decoder_layers_attr(model)

    w_spec = OCP_MXFP4Spec(ch_axis=-1, is_dynamic=False).to_quantization_spec()
    global_cfg = QLayerConfig(weight=w_spec)
    exclude = get_exclude_patterns(model)

    algo = []
    if r2:
        # R2 folds into v_proj/o_proj weights offline (no runtime op). Uses the
        # dict-form rotation map; R1 is excluded (Wan R1 requires an online rotation
        # op the inference path does not have).
        rot = get_rotation_map(model)
        shim_rotation_config(model.config)
        algo.append(
            RotationConfig(
                scaling_layers=rot["scaling_layers"],
                model_decoder_layers=decoder_layers,
                r1=False,
                r2=True,
                r3=False,
                r4=False,
                v_proj=rot["v_proj"],
                o_proj=rot["o_proj"],
                self_attn=rot["self_attn"],
                mlp=rot["mlp"],
            )
        )
    if smooth:
        algo.append(
            SmoothQuantConfig(
                scaling_layers=scaling_layers,
                model_decoder_layers=decoder_layers,
                alpha=alpha,
            )
        )
    return QConfig(global_quant_config=global_cfg, algo_config=algo or None, exclude=exclude)


def _pack_linear_weight(w_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a calibrated bf16 weight (N, K) to the AITER FP4 layout.

    Must match ROCmMxfp4LinearMethod.process_weights_after_loading (per_1x32 quant +
    shuffle_weight(16, 16)) so the loader can consume the result as-is. Returns
    weight_shuffle (float4_e2m1fn_x2, N, K/2) and weight_scale (float8_e8m0fnu, N, K/32),
    both as uint8 (same bit width, lossless through safetensors).
    """
    import aiter
    from aiter.ops.shuffle import shuffle_weight

    quant_func = aiter.get_hip_quant(aiter.QuantType.per_1x32)
    weight_quant, weight_scale = quant_func(w_bf16, shuffle=True)
    weight_shuffle = shuffle_weight(weight_quant, layout=(16, 16))
    return weight_shuffle.view(torch.uint8).contiguous(), weight_scale.view(torch.uint8).contiguous()


def _quantized_linear_names(tq) -> set[str]:
    """Names of frozen linears that Quark actually quantized (have a weight quantizer)."""
    names = set()
    for name, mod in tq.named_modules():
        if hasattr(mod, "_weight_quantizer") and getattr(mod, "weight", None) is not None:
            names.add(name)
    return names


def quantize_component(args, comp: str) -> dict:
    from diffusers import WanPipeline
    from quark.torch import ModelQuantizer
    from quark.torch.utils.diffusers.calibration import get_calib_dataloader
    from safetensors.torch import save_file

    dev = "cuda"
    t0 = time.time()
    pipe = WanPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    tf = getattr(pipe, comp)
    print(f"[quark-mxfp4] {comp}: {type(tf).__name__} loaded in {time.time() - t0:.0f}s")

    qconfig = build_qconfig(tf, args.alpha, not args.no_smooth, args.r2)
    pipe.to(dev)
    prompts = DEFAULT_CALIB_PROMPTS[: args.n_prompts]
    dl = _SafeCalibLoader(get_calib_dataloader(pipe, tf, prompts, n_steps=args.n_steps, device=dev))

    quantizer = ModelQuantizer(qconfig)
    tq = quantizer.freeze(quantizer.quantize_model(tf, dataloader=dl))
    dt = time.time() - t0

    out = os.path.join(args.output, comp)
    os.makedirs(out, exist_ok=True)

    if args.pack:
        export_format = "mxfp4_packed"
        quantized = _quantized_linear_names(tq)
        state = dict(tq.state_dict())
        sd, packed, dropped = {}, 0, 0
        for k, v in state.items():
            if not isinstance(v, torch.Tensor):
                continue
            if any(m in k for m in _DROP_KEY_MARKERS):
                dropped += 1
                continue
            mod_name = k.rsplit(".", 1)[0]
            if k.endswith(".weight") and mod_name in quantized:
                # Pack into weight_shuffle + weight_scale.
                wsh, wsc = _pack_linear_weight(v.to(torch.bfloat16).cuda())
                sd[f"{mod_name}.weight_shuffle"] = wsh.cpu()
                sd[f"{mod_name}.weight_scale"] = wsc.cpu()
                packed += 1
            else:
                sd[k] = v.to(torch.bfloat16).contiguous() if v.is_floating_point() else v.contiguous()
        # Non-quantized linears load unquantized (bf16) at runtime.
        all_linears = {
            n for n, m in tq.named_modules() if isinstance(m, torch.nn.Linear) or hasattr(m, "_weight_quantizer")
        }
        ignored = sorted(all_linears - quantized)
        note = f"packed {packed} linears"
    else:
        export_format = "mxfp4"
        sd, dropped = {}, 0
        for k, v in tq.state_dict().items():
            if not isinstance(v, torch.Tensor):
                continue
            if any(m in k for m in _DROP_KEY_MARKERS):
                dropped += 1
                continue
            sd[k] = v.to(torch.bfloat16).contiguous() if v.is_floating_point() else v.contiguous()
        ignored = None
        note = f"{len(sd)} tensors"

    save_file(sd, os.path.join(out, "diffusion_pytorch_model.safetensors"))
    # config.json stanza so vllm-omni auto-selects the right offline MXFP4 loader.
    qc = {
        "quant_method": "quark",
        "quark_export_format": export_format,
        "is_checkpoint_mxfp4_serialized": True,
        "producer": "quark",
        "algo": {"smoothquant": not args.no_smooth, "alpha": args.alpha, "rotation_r2": args.r2},
    }
    if export_format == "mxfp4_packed":
        # Loader must match this preshuffle layout.
        qc["packing"] = "aiter_per_1x32_shuffle16x16"
        if ignored:
            qc["ignored_layers"] = ignored
    json.dump({"quantization_config": qc}, open(os.path.join(out, "quant_config.json"), "w"), indent=2)
    print(
        f"[quark-mxfp4] {comp}: saved {len(sd)} tensors ({note}, dropped {dropped} "
        f"quantizer keys) format={export_format} -> {out} in {dt:.0f}s"
    )

    del pipe, tf, tq
    gc.collect()
    torch.accelerator.empty_cache()
    return {"saved": out, "tensors": len(sd), "dropped": dropped, "format": export_format, "seconds": round(dt, 1)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a calibrated Quark MXFP4 Wan2.2 checkpoint.")
    p.add_argument(
        "--model", default="Wan-AI/Wan2.2-T2V-A14B-Diffusers", help="Source BF16 diffusers model path or id."
    )
    p.add_argument("--output", required=True, help="Export root directory.")
    p.add_argument(
        "--components",
        nargs="+",
        default=["transformer", "transformer_2"],
        help="Transformer components to quantize (A14B cascade = both).",
    )
    p.add_argument("--n-prompts", type=int, default=4, help="Calibration prompts.")
    p.add_argument("--n-steps", type=int, default=4, help="Denoise steps per calib prompt.")
    p.add_argument("--alpha", type=float, default=0.5, help="SmoothQuant alpha.")
    p.add_argument("--no-smooth", action="store_true", help="Disable SmoothQuant (plain MXFP4).")
    p.add_argument(
        "--r2",
        action="store_true",
        help="Enable R2 Hadamard rotation (folds into attn v/o proj offline; "
        "no runtime op). R1 is not supported on Wan (needs an online "
        "rotation op in the inference path).",
    )
    p.add_argument(
        "--pack",
        action="store_true",
        help="Pack weights offline to the AITER FP4 layout (weight_shuffle + "
        "weight_scale, ~3.7x smaller, no pack at load). ROCm gfx950 loader "
        "only. Omit for the portable calibrated-bf16 export.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    summary = {
        "model": args.model,
        "output": args.output,
        "alpha": args.alpha,
        "smoothquant": not args.no_smooth,
        "r2": args.r2,
        "n_prompts": args.n_prompts,
        "n_steps": args.n_steps,
        "components": {},
    }
    for comp in args.components:
        print(f"\n{'=' * 60}\n[quark-mxfp4] component: {comp}\n{'=' * 60}")
        summary["components"][comp] = quantize_component(args, comp)
    json.dump(summary, open(os.path.join(args.output, "export_summary.json"), "w"), indent=2)
    print(f"\n[quark-mxfp4] DONE\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
