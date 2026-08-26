# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline W4A8 export: a Quark-calibrated Wan2.2 checkpoint for vLLM-Omni.

Two variants, both consumed by ``quantization="quark_w4a8"`` with
``is_checkpoint_w4a8_serialized=True``:

* ``svdquant`` -- runs Quark's ``SVDQuantProcessor`` (activation-aware SmoothQuant
  smoothing + exact SVD on the smoothed weight), emitting the BF16 residual ``R``
  plus low-rank factors ``proj_down`` (=L1) / ``proj_up`` (=L2). This is the
  *calibrated* counterpart to vLLM-Omni's online ``torch.svd_lowrank`` path, which
  sees only the raw weight. ``--gptq`` additionally quantizes the residual with
  Hessian-based GPTQ (using the calibration data) instead of RTN.
* ``plain`` -- the **RTN tier** (round-to-nearest MXFP4 weight; no low-rank branch,
  no calibration -- always uncalibrated). The export just copies the stock weights;
  vLLM-Omni RTN-quantizes them to MXFP4 at load, exactly as the *online* plain path
  does -- so a default-format plain export is the stock checkpoint round-tripped,
  useful only as a portable/reproducible artifact. "BF16" here is the on-disk
  storage, not the served precision.

By default (``--pack-format bf16``) the residual is written **unpacked BF16** and
vLLM-Omni RTN-packs it to the FlyDSL MXFP4 layout at load; ``--pack-format
packed``/``unshuffled`` instead store the residual already MXFP4-packed (~4x
smaller). Self-attention ``to_q/to_k/to_v`` are pre-fused into ``to_qkv``
here (residual concatenated, ``proj_down`` stacked, ``proj_up`` block-diagonal)
because the fused layer's low-rank factors cannot be reassembled by the runtime
shard loader the way a plain weight can.

Layout mirrors ``quantize_wan2_2_quark_mxfp4.py`` (PR #5693):

    <output_dir>/<comp>/diffusion_pytorch_model.safetensors
    <output_dir>/<comp>/quant_config.json      # {"quantization_config": {...}}

where ``<comp>`` is ``transformer`` (+ ``transformer_2`` for the A14B cascade).

Usage:

    python examples/quantization/export_quark_svdquant_w4a8.py \\
        --model /workspace/Wan2.2-TI2V-5B-Diffusers --variant svdquant \\
        --svd_rank 32 --n_calib_prompts 2 --n_calib_steps 20 \\
        --output_dir /path/to/wan5b-w4a8-svd

Requires an AMD Quark checkout with the SVDQuant algorithm and a GPU (this box's
torch has no CPU LAPACK, so ``torch.linalg.svd`` must run on device).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time

import torch

DEFAULT_CALIB_PROMPTS = [
    "A cinematic shot of a serene mountain lake at sunrise, low mist over the water.",
    "A golden retriever running across a green meadow, realistic style.",
    "Neon city street at night in the rain, reflections on the asphalt.",
    "A hot air balloon drifting over a patchwork of autumn fields.",
]

# Quark's SVDQuant exclude set for Wan (input/output embedders and the tiny
# out-projection that the FlyDSL SVD epilogue refuses anyway).
WAN_SVDQUANT_EXCLUDE = [
    "*time_embedder*",
    "*patch_embedding*",
    "*condition_embedder*",
    "*norm_out*",
    "*proj_out*",
]

# Bare tokens for the checkpoint's ignored_layers (is_layer_skipped substring
# match); these layers stay BF16 at load.
_IGNORED_TOKENS = ["time_embedder", "patch_embedding", "condition_embedder", "norm_out", "proj_out"]

# CLI --pack-format -> quark_export_format written into the checkpoint config.
_EXPORT_FORMAT = {"bf16": "bf16", "packed": "mxfp4_packed", "unshuffled": "mxfp4_unshuffled"}

# ErrorCorrectedModule state_dict -> vLLM-Omni canonical key. Order matters: the
# correction/layer suffixes are checked before the bare ``.weight`` they contain.
_ECM_REMAP = (
    (".correction.l1.weight", ".proj_down"),
    (".correction.l2.weight", ".proj_up"),
    (".layer.weight", ".weight"),
    (".layer.bias", ".bias"),
)

_QKV_RE = re.compile(r"^(.*\.attn1\.)to_q\.(weight|bias|proj_down|proj_up)$")

# Mirrors of flydsl_w4a8.supports_svd_shape / supports_shape -- kept local so the
# state-dict shaping needs no vLLM import. K (in_features) is the strict axis: the
# kernel requires K >= 256 and a multiple of 256, which also makes the packed E8M0
# scale exactly (N, K/32) with no padding, so the packed loader can use clean shapes.
_SVD_DIM_MULTIPLE = 256
_TILE_N_MULTIPLE = 32


def _omni_svd_ok(out_features: int, in_features: int) -> bool:
    return (
        out_features >= _SVD_DIM_MULTIPLE
        and in_features >= _SVD_DIM_MULTIPLE
        and out_features % _SVD_DIM_MULTIPLE == 0
        and in_features % _SVD_DIM_MULTIPLE == 0
    )


def _omni_plain_ok(out_features: int, in_features: int) -> bool:
    return (
        in_features >= _SVD_DIM_MULTIPLE
        and in_features % _SVD_DIM_MULTIPLE == 0
        and out_features % _TILE_N_MULTIPLE == 0
    )


def _remap_key(key: str) -> str:
    for suffix, canonical in _ECM_REMAP:
        if key.endswith(suffix):
            return key[: -len(suffix)] + canonical
    return key


def _fuse_qkv(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Pre-fuse self-attention to_q/to_k/to_v into a single to_qkv.

    residual/bias/proj_down concatenate along dim 0; proj_up is block-diagonal so
    each head's correction only feeds its own output slice. Verified equal to the
    three independent corrections summed.
    """
    prefixes = {m.group(1) for k in sd if (m := _QKV_RE.match(k))}
    for prefix in sorted(prefixes):
        for sub in ("weight", "bias", "proj_down", "proj_up"):
            keys = [f"{prefix}to_{x}.{sub}" for x in ("q", "k", "v")]
            if not all(k in sd for k in keys):
                continue
            parts = [sd.pop(k) for k in keys]
            fused = torch.block_diag(*parts) if sub == "proj_up" else torch.cat(parts, dim=0)
            sd[f"{prefix}to_qkv.{sub}"] = fused.contiguous()
    return sd


def build_omni_state_dict(model: torch.nn.Module, variant: str, fuse_qkv: bool) -> dict[str, torch.Tensor]:
    """Remap a (post-``apply()``) model's state_dict into vLLM-Omni's layout."""
    sd: dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        if not isinstance(value, torch.Tensor):
            continue
        tensor = value.detach().to("cpu")
        if tensor.is_floating_point():
            tensor = tensor.to(torch.bfloat16)
        sd[_remap_key(key)] = tensor.contiguous()
    if variant == "svdquant" and fuse_qkv:
        sd = _fuse_qkv(sd)
    if variant == "svdquant":
        sd = _fold_unsupported_factors(sd)
    return sd


def _fold_unsupported_factors(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Reconstruct the full weight for any SVD layer the vLLM-Omni gate rejects.

    Keeps the invariant *factors present iff the (fused) shape passes
    ``supports_svd_shape``* -- otherwise the loader would route the layer to a
    non-SVD method and silently drop the correction, leaving a residual-only
    weight. Wan's dims are 256-aligned so this normally folds nothing.
    """
    prefixes = {k[: -len(".proj_down")] for k in sd if k.endswith(".proj_down")}
    for prefix in sorted(prefixes):
        proj_down = sd[f"{prefix}.proj_down"]
        proj_up = sd[f"{prefix}.proj_up"]
        if _omni_svd_ok(proj_up.shape[0], proj_down.shape[1]):
            continue
        weight = sd[f"{prefix}.weight"]
        sd[f"{prefix}.weight"] = (weight + proj_up @ proj_down).contiguous()
        del sd[f"{prefix}.proj_down"], sd[f"{prefix}.proj_up"]
    return sd


def _pack_residuals(sd: dict[str, torch.Tensor], unshuffled: bool) -> dict[str, torch.Tensor]:
    """Pre-pack each quantized residual weight to MXFP4 on disk.

    ``packed`` (``unshuffled=False``): ``X.weight`` -> ``X.weight_shuffle`` (N, K/2)
    + ``X.weight_scale`` (N, K/32), preshuffled into the kernel layout. Fastest
    load, but the shuffle bakes in K/N so it is **TP=1 only**.

    ``unshuffled`` (``unshuffled=True``): ``X.weight`` -> ``X.weight_packed`` (N, K/2)
    + ``X.weight_scale`` (N, K/32) in *natural* order, so vLLM can shard it for
    **TP>1**; each rank shuffles its shard at load.

    Both are ~4x smaller than BF16. Only weights the loader will quantize are
    packed; ignored/untileable layers stay BF16. Low-rank ``proj_down``/``proj_up``
    stay BF16. Either format couples the checkpoint to the kernel's pack version.
    """
    from vllm_omni.quantization import flydsl_w4a8

    out: dict[str, torch.Tensor] = {}
    for key, value in sd.items():
        base = key[: -len(".weight")] if key.endswith(".weight") else None
        if (
            base is not None
            and value.ndim == 2
            and not any(tok in base for tok in _IGNORED_TOKENS)
            and _omni_plain_ok(value.shape[0], value.shape[1])
        ):
            if unshuffled:
                w_q, w_s = flydsl_w4a8.pack_weight_unshuffled(value.to("cuda"))
                out[f"{base}.weight_packed"] = w_q.cpu().contiguous()
            else:
                w_q, w_s = flydsl_w4a8.pack_weight(value.to("cuda"))
                out[f"{base}.weight_shuffle"] = w_q.cpu().contiguous()
            out[f"{base}.weight_scale"] = w_s.cpu().contiguous()
        else:
            out[key] = value
    return out


def quantize_component(pipe, comp: str, args: argparse.Namespace) -> dict[str, torch.Tensor]:
    transformer = getattr(pipe, comp)
    print(f"[export] {comp}: {type(transformer).__name__}")

    if args.variant == "plain":
        # A portable pre-quant artifact; RTN happens at load. Smoothing is a
        # no-op without a low-rank branch, so nothing calibrated to do here.
        # (--gptq + plain is rejected at parse time.)
        return build_omni_state_dict(transformer, variant="plain", fuse_qkv=False)

    from quark.torch.algorithm.svdquant.svdquant import SVDQuantProcessor
    from quark.torch.quantization.config.config import SVDQuantConfig
    from quark.torch.utils.diffusers.calibration import get_calib_dataloader

    prompts = DEFAULT_CALIB_PROMPTS[: args.n_calib_prompts]
    t0 = time.time()
    dataloader = get_calib_dataloader(
        pipe,
        transformer,
        prompts,
        n_steps=args.n_calib_steps,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
    )
    print(f"[export] {comp}: captured calibration in {time.time() - t0:.0f}s")

    cfg = SVDQuantConfig(
        name="svdquant",
        svd_rank=args.svd_rank,
        smooth_alpha=args.smooth_alpha,
        search_alpha=args.search_alpha,
        exclude_patterns=list(WAN_SVDQUANT_EXCLUDE),
        min_layer_size=256,
        use_gptq=args.gptq,
    )
    t0 = time.time()
    SVDQuantProcessor(model=transformer, quant_algo_config=cfg, calib_data=dataloader).apply()
    print(f"[export] {comp}: SVDQuant apply() in {time.time() - t0:.0f}s")

    return build_omni_state_dict(transformer, variant="svdquant", fuse_qkv=not args.no_fuse_qkv)


def write_component(out_dir: str, comp: str, sd: dict[str, torch.Tensor], args: argparse.Namespace) -> None:
    from safetensors.torch import save_file

    comp_dir = os.path.join(out_dir, comp)
    os.makedirs(comp_dir, exist_ok=True)
    path = os.path.join(comp_dir, "diffusion_pytorch_model.safetensors")
    save_file(sd, path)
    size_gb = os.path.getsize(path) / 1e9
    print(f"[export] {comp}: wrote {len(sd)} tensors ({size_gb:.2f} GB) -> {path}")

    quant_config: dict = {
        "quant_method": "quark_w4a8",
        "is_checkpoint_w4a8_serialized": True,
        "producer": "quark",
        "variant": args.variant,
        # bf16: unpacked residual (packed at load). mxfp4_packed: preshuffled on
        # disk (TP=1). mxfp4_unshuffled: natural-order MXFP4 on disk (shardable, TP>1).
        "quark_export_format": _EXPORT_FORMAT[args.pack_format],
        "ignored_layers": list(_IGNORED_TOKENS),
        "algo": {
            "svdquant": args.variant == "svdquant",
            "smooth_alpha": args.smooth_alpha,
            "search_alpha": args.search_alpha,
            "gptq": args.gptq,
        },
    }
    if args.pack_format != "bf16":
        quant_config["packing"] = "flydsl_a8w4_preshuffle"
    if args.variant == "svdquant":
        quant_config["svd_rank"] = args.svd_rank
    with open(os.path.join(comp_dir, "quant_config.json"), "w") as handle:
        json.dump({"quantization_config": quant_config}, handle, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Path to a diffusers Wan2.2 model directory.")
    p.add_argument("--variant", default="svdquant", choices=["svdquant", "plain"])
    p.add_argument("--svd_rank", type=int, default=32)
    p.add_argument("--smooth_alpha", type=float, default=0.5)
    p.add_argument("--search_alpha", action="store_true", help="Per-layer alpha search (slow).")
    p.add_argument("--gptq", action="store_true", help="Residual GPTQ (svdquant only).")
    p.add_argument("--no_fuse_qkv", action="store_true", help="Debug: emit separate attn1 to_q/k/v.")
    p.add_argument(
        "--pack-format",
        dest="pack_format",
        default="bf16",
        choices=["bf16", "packed", "unshuffled"],
        help="On-disk residual format. bf16: unpacked, packed at load (default). "
        "packed: preshuffled MXFP4 (~4x smaller, fastest load, TP=1 only). "
        "unshuffled: natural-order MXFP4 (~4x smaller, shardable for TP>1, shuffled per shard at load).",
    )
    p.add_argument("--n_calib_prompts", type=int, default=2)
    p.add_argument("--n_calib_steps", type=int, default=20)
    p.add_argument("--num_frames", type=int, default=17)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=448)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()
    # GPTQ acts on the SVD residual; plain has no residual. Reject at parse time
    # so it fails before the pipeline loads, not tens of minutes into calibration.
    if args.gptq and args.variant == "plain":
        p.error("--gptq requires --variant svdquant (plain has no residual to quantize)")
    return args


def main() -> int:
    args = parse_args()
    print("[export] args:", vars(args))

    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    comps = ["transformer"]
    if getattr(pipe, "transformer_2", None) is not None:
        comps.append("transformer_2")
    print(f"[export] components: {comps}")

    os.makedirs(args.output_dir, exist_ok=True)
    for comp in comps:
        sd = quantize_component(pipe, comp, args)
        if args.pack_format != "bf16":
            sd = _pack_residuals(sd, unshuffled=args.pack_format == "unshuffled")
        write_component(args.output_dir, comp, sd, args)

    print(f"[export] DONE -> {args.output_dir}")
    print(
        "[export] copy/symlink the non-transformer pipeline components "
        "(vae, text_encoder, scheduler, model_index.json) into the output dir to make it loadable."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
