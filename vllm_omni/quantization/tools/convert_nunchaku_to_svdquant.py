#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert a nunchaku-published merged-safetensors NVFP4 SVDQuant checkpoint
into a vLLM-loadable diffusers pipeline folder.

On-disk format: canonical row-major + FP4 nibble pack. This is the
layout the SM_100 native (CuTe) kernel consumes directly. For the
nunchaku kernel backend (consumer GPUs), vLLM repacks to fragment at
load time via `SVDQuantLinearMethod.process_weights_after_loading`.
The on-disk format is the same regardless of target backend; users do
not need to know about nunchaku-vs-native layout.

What this does:
  1. Resolve inputs (local paths or HuggingFace repo ids → snapshot_download).
  2. Stream tensors from the nunchaku merged safetensors, grouping by linear
     layer prefix (those with a `.qweight` sibling).
  3. Unpack fragment layout → row-major for every layer:
       qweight via `unpack_nunchaku_qweight_fp4` → [N, K/2] uint8 (FP4 nibbles)
       wscales via `unpack_nunchaku_wscales_fp4` → [K/16, N] fp8_e4m3fn
       proj_up via `unpack_lowrank_weight(down=False)` → [N, R]
       proj_down via `unpack_lowrank_weight(down=True)` → unpack returns
         [R, K] (transpose-quirk in nunchaku), transpose back to [K, R]
  4. For each fused gate-up linear (suffix `.feed_forward.net.0.proj` in
     Z-Image), do a bit-preserving N-axis half-swap so the on-disk layout
     matches vLLM's standard `[gate; hidden]` SiluAndMul convention.
  5. Ensure each linear's state-dict block carries `wtscale` as a `(1,)`
     bf16 tensor (default 1.0 if missing).
  6. Write a complete diffusers folder at `--output-dir`:
       <out>/model_index.json                  (linked from base)
       <out>/scheduler/, text_encoder/, ...    (linked from base)
       <out>/transformer/config.json           (base config + injected
                                                "quantization_config" field
                                                so vllm-omni auto-picks SVDQuant)
       <out>/transformer/diffusion_pytorch_model.safetensors  (converted weights)

All transforms are pure permute+view (bit-preserving). Round-trip
verified: `pack(unpack(x)) == x` bit-exactly for proj_down and proj_up
across shape stress; half-swap pipeline verified end-to-end against
`svdq_gemm_w4a4_cuda`.

Usage:
  python -m vllm_omni.quantization.tools.convert_nunchaku_to_svdquant \\
      --nunchaku-checkpoint nunchaku-tech/nunchaku-z-image-turbo/svdq-fp4_r128-z-image-turbo.safetensors \\
      --base-pipeline Tongyi-MAI/Z-Image-Turbo \\
      --output-dir ~/.cache/huggingface/hub/models--ultranationalism--nunchaku-z-image-turbo-svdq/

Both `--nunchaku-checkpoint` and `--base-pipeline` accept either a local
path or an HF repo id; HF auto-download happens only on cache miss.

Non-transformer subfolders + top-level files are hard-linked from the
base pipeline by default (saves 35+ GB). `huggingface_hub.upload_folder`
reads file content, so hard links upload fine. Use `--copy` to disable.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vllm_omni.quantization.tools.svdquant_nvfp4_layout import (
    _unpack_nibbles,
    unpack_nunchaku_qweight_fp4,
    unpack_nunchaku_wscales_fp4,
)

# ---------------------------------------------------------------------------
# Z-Image-specific knowledge
# ---------------------------------------------------------------------------

# Fused gate-up linear suffix in nunchaku-published Z-Image NVFP4 checkpoints.
# diffusers `FeedForward(activation_fn="swiglu")` stores the fused up
# projection at `.net.0.proj`; the `feed_forward` parent name is from
# Z-Image's transformer block. Identified empirically from
# `svdq-fp4_r128-z-image-turbo.safetensors` (34 such layers; shape
# [2*hidden, K]).
ZIMAGE_FUSED_GATE_UP_TAIL = "feed_forward.net.0.proj"

# Key-prefix renames from diffusers FF naming (in nunchaku checkpoint) to
# vllm-omni Z-Image's `MergedColumnParallelLinear` + `RowParallelLinear` naming.
# See `vllm_omni/diffusion/models/z_image/z_image_transformer.py:368-394` —
# `FeedForward.__init__` registers `self.w13` (fused gate-up) and `self.w2`
# (down). The diffusers FF block uses `net = ModuleList([gate_up, act, down])`
# instead, giving `net.0.proj` and `net.2`. Both quantized.
ZIMAGE_LAYER_RENAMES: dict[str, str] = {
    "feed_forward.net.0.proj": "feed_forward.w13",
    "feed_forward.net.2": "feed_forward.w2",
}


def is_fused_gate_up_zimage(layer_prefix: str) -> bool:
    return layer_prefix.endswith(ZIMAGE_FUSED_GATE_UP_TAIL)


def rename_key_zimage(key: str) -> str:
    """Apply the Z-Image FF layer-name renames (nunchaku diffusers → vllm-omni).

    Only matches `.<src>.` substrings (with both bounding dots) to avoid
    false matches on substrings.
    """
    for src, dst in ZIMAGE_LAYER_RENAMES.items():
        marker = f".{src}."
        if marker in key:
            return key.replace(marker, f".{dst}.", 1)
    return key


# ---------------------------------------------------------------------------
# Per-linear nunchaku-fragment → row-major (with optional half-swap)
# ---------------------------------------------------------------------------


# nunchaku.lora.flux is the canonical home of unpack_lowrank_weight; import
# lazily so the script can at least argparse-help without nunchaku installed.
def _lowrank_unpack():
    from nunchaku.lora.flux.nunchaku_converter import unpack_lowrank_weight

    return unpack_lowrank_weight


def _pack_qweight_row_major(nibs: torch.Tensor) -> torch.Tensor:
    """`[N, K] uint8 nibbles → [N, K/2] uint8`, low nibble = even k.

    Inverse of `_unpack_nibbles`. The on-disk canonical `qweight` is the
    pair-packed nibble byte exactly as the SM_100 CuTe kernel expects.
    """
    assert nibs.shape[-1] % 2 == 0
    lo = nibs[..., 0::2]
    hi = nibs[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8)


def unpack_nvfp4_layer(
    params: dict[str, torch.Tensor],
    *,
    half_swap_n: bool,
) -> dict[str, torch.Tensor]:
    """nunchaku fragment → canonical row-major for one NVFP4 SVDQuant linear.

    Pure permute+view (bit-preserving) for `qweight`, `wscales`, `proj_up`,
    `proj_down`. `wcscales`, `bias`, `smooth_factor`, etc. are already
    layout-agnostic and copy through.

    When `half_swap_n=True`, additionally swap the two N-axis halves on
    `qweight`, `wscales`, `proj_up`, `wcscales`, `bias` — the SiluAndMul
    `[gate; hidden]` reorder. Swap happens on row-major intermediates,
    which is free (it's a slice + cat).
    """
    unpack_lowrank_weight = _lowrank_unpack()

    qweight = params["qweight"]  # [N, K/2] int8 (nunchaku fragment)
    wscales = params["wscales"]  # [K/16, N] fp8 (nunchaku fragment)
    proj_up = params["proj_up"]  # [N, R] bf16 (nunchaku fragment)
    proj_down = params["proj_down"]  # [K, R] bf16 (nunchaku fragment)
    wcscales = params.get("wcscales")  # [N] bf16 (optional)
    bias = params.get("bias")  # [N] bf16 (optional)

    N = qweight.shape[0]
    if half_swap_n:
        assert N % 2 == 0, f"fused gate-up N must be even; got {N}"
    half = N // 2

    # qweight: unpack fragment → [N, K/2] uint8 nibble bytes (low = even-k);
    # then `_unpack_nibbles` → [N, K] full-nibble form so we can slice on N
    # then repack to [N, K/2] for storage.
    qw_rm = unpack_nunchaku_qweight_fp4(qweight.view(torch.int8))  # [N, K/2] uint8
    if half_swap_n:
        nibs = _unpack_nibbles(qw_rm)  # [N, K] uint8
        nibs = torch.cat([nibs[half:], nibs[:half]], dim=0).contiguous()
        qw_rm = _pack_qweight_row_major(nibs)
    qweight_out = qw_rm.contiguous()

    # wscales: unpack to [K/16, N] row-major fp8.
    ws_rm = unpack_nunchaku_wscales_fp4(wscales)
    if half_swap_n:
        ws_rm = torch.cat([ws_rm[:, half:], ws_rm[:, :half]], dim=1).contiguous()
    wscales_out = ws_rm.contiguous()

    # proj_up: down=False → unpack returns [N, R] directly.
    pu_rm = unpack_lowrank_weight(proj_up, down=False)
    if half_swap_n:
        pu_rm = torch.cat([pu_rm[half:], pu_rm[:half]], dim=0).contiguous()
    proj_up_out = pu_rm.contiguous()

    # proj_down: down=True. nunchaku's unpack returns [R, K]; canonical
    # row-major is [K, R] (matches SM_100 CuTe kernel's expected layout).
    # Transpose to [K, R].
    pd_rm = unpack_lowrank_weight(proj_down, down=True)
    K = proj_down.shape[0]
    R = proj_down.shape[1]
    if pd_rm.shape == (R, K):
        pd_rm = pd_rm.transpose(0, 1).contiguous()
    assert pd_rm.shape == (K, R), f"proj_down expected ({K}, {R}); got {tuple(pd_rm.shape)}"
    proj_down_out = pd_rm

    out = dict(params)
    out["qweight"] = qweight_out
    out["wscales"] = wscales_out
    out["proj_up"] = proj_up_out
    out["proj_down"] = proj_down_out
    if half_swap_n:
        if wcscales is not None:
            out["wcscales"] = torch.cat([wcscales[half:], wcscales[:half]]).contiguous()
        if bias is not None:
            out["bias"] = torch.cat([bias[half:], bias[:half]]).contiguous()
    return out


# ---------------------------------------------------------------------------
# Input materialization
# ---------------------------------------------------------------------------


def _resolve_nunchaku_checkpoint(arg: str) -> Path:
    """Accept a local file path OR an HF spec `<repo_id>/<filename>`.

    Local path is returned as-is. Otherwise the trailing component is treated
    as the filename within the repo, and the rest is the repo id. Downloads
    only on cache miss.
    """
    p = Path(arg)
    if p.exists() and p.is_file():
        return p
    # Treat as HF spec: split into (repo_id, filename).
    parts = arg.split("/")
    if len(parts) < 3:
        raise ValueError(
            f"--nunchaku-checkpoint {arg!r} is not a local file and not a "
            "<repo_id>/<filename> spec (need owner/name/file.safetensors)"
        )
    repo_id = "/".join(parts[:2])
    filename = "/".join(parts[2:])
    from huggingface_hub import hf_hub_download

    print(f"resolving nunchaku checkpoint: repo={repo_id} file={filename}")
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    return Path(path)


def _resolve_base_pipeline(arg: str) -> Path:
    """Accept a local diffusers folder OR an HF repo id."""
    p = Path(arg)
    if p.exists() and p.is_dir():
        # Local diffusers folder.
        return p
    # HF repo id → snapshot_download (uses cache if present).
    from huggingface_hub import snapshot_download

    print(f"resolving base pipeline: repo={arg}")
    path = snapshot_download(repo_id=arg)
    return Path(path)


# ---------------------------------------------------------------------------
# Filesystem mirror (hard-link with copy fallback)
# ---------------------------------------------------------------------------


def _link_or_copy_file(src: Path, dst: Path, prefer_copy: bool) -> None:
    """Hard-link src → dst, falling back to copy. Resolves source symlinks
    (the HF cache uses symlink-from-snapshot-to-blob; we want the blob).
    """
    real = src.resolve()
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if prefer_copy:
        shutil.copy2(real, dst)
        return
    try:
        os.link(real, dst)
    except OSError:
        # Cross-fs or permissions: fall back to copy.
        shutil.copy2(real, dst)


def _link_or_copy_tree(src: Path, dst: Path, prefer_copy: bool) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        d = dst / item.name
        if item.is_dir():
            _link_or_copy_tree(item, d, prefer_copy)
        else:
            _link_or_copy_file(item, d, prefer_copy)


# ---------------------------------------------------------------------------
# Conversion driver
# ---------------------------------------------------------------------------


# Suffixes nunchaku publishes alongside every quantized linear that the
# vLLM SVDQuant LinearMethod does not consume — they bloat the output
# checkpoint without serving any backend. Filter them at group time so
# downstream conversion / save never touches them.
#
# `smooth_factor_orig`: declared by nunchaku as "(Unused)" (see
# `nunchaku/models/linear.py:54`) and never read by any quantize/forward
# path in either int4 or nvfp4. ~0.001 GB across a Z-Image checkpoint —
# trivially small, but keeping it triggers a KeyError at load time since
# vLLM does not register a `smooth_factor_orig` parameter.
_DROPPED_NUNCHAKU_SUFFIXES: frozenset[str] = frozenset({"smooth_factor_orig"})


def _group_keys_by_layer(
    keys: list[str],
) -> tuple[dict[str, list[str]], list[str]]:
    """Return (layer_prefix → list-of-suffixes, leftover-keys).

    A "linear" is any key prefix that has a `.qweight` sibling. Suffixes
    in `_DROPPED_NUNCHAKU_SUFFIXES` are filtered out entirely.
    """
    qweight_prefixes = {k.rsplit(".", 1)[0] for k in keys if k.endswith(".qweight")}
    layer_to_suffixes: dict[str, list[str]] = {p: [] for p in qweight_prefixes}
    leftover: list[str] = []
    for k in keys:
        prefix, _, suffix = k.rpartition(".")
        if prefix in layer_to_suffixes:
            if suffix in _DROPPED_NUNCHAKU_SUFFIXES:
                continue
            layer_to_suffixes[prefix].append(suffix)
        else:
            leftover.append(k)
    return layer_to_suffixes, leftover


def _detect_rank_precision(f, sample_prefix: str) -> tuple[int, str]:
    proj_down = f.get_tensor(f"{sample_prefix}.proj_down")
    wscales = f.get_tensor(f"{sample_prefix}.wscales")
    rank = int(proj_down.shape[1])
    if wscales.dtype == torch.float8_e4m3fn:
        precision = "nvfp4"
    elif wscales.dtype in (torch.float16, torch.bfloat16):
        precision = "int4"
    else:
        raise ValueError(f"unexpected wscales dtype {wscales.dtype}")
    return rank, precision


def convert(
    nunchaku_checkpoint: Path,
    base_pipeline: Path,
    output_dir: Path,
    *,
    prefer_copy: bool,
    is_fused_gate_up=is_fused_gate_up_zimage,
    rename_key=rename_key_zimage,
    progress: bool = True,
) -> None:
    """Drive the full conversion. See module docstring for behavior."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----- Mirror base pipeline (everything except transformer/) -----
    base_top_level = sorted(base_pipeline.iterdir(), key=lambda p: p.name)
    for item in base_top_level:
        if item.name == "transformer":
            continue
        d = output_dir / item.name
        if item.is_dir():
            _link_or_copy_tree(item, d, prefer_copy)
        else:
            _link_or_copy_file(item, d, prefer_copy)
    print(f"mirrored {len(base_top_level) - 1} top-level entries from base ({'copy' if prefer_copy else 'hard-link'})")

    # ----- transformer/ -----
    transformer_dir = output_dir / "transformer"
    transformer_dir.mkdir(exist_ok=True)
    base_transformer = base_pipeline / "transformer"

    # ----- Scan nunchaku checkpoint -----
    with safe_open(str(nunchaku_checkpoint), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        metadata = f.metadata() or {}
        layer_to_suffixes, leftover = _group_keys_by_layer(keys)
        if not layer_to_suffixes:
            raise RuntimeError("no quantized linears found (no .qweight keys)")
        sample_prefix = next(iter(layer_to_suffixes))
        rank, precision = _detect_rank_precision(f, sample_prefix)

        n_linears = len(layer_to_suffixes)
        n_fused = sum(1 for p in layer_to_suffixes if is_fused_gate_up(p))
        print(
            f"nunchaku checkpoint: {n_linears} quantized linears, "
            f"{n_fused} fused gate-up (to swap); {len(leftover)} other keys"
        )
        print(f"detected rank={rank} precision={precision}")
        if "model_class" in metadata:
            print(f"nunchaku metadata model_class={metadata['model_class']!r}")

        # ----- Build output state_dict via streaming reads -----
        out_sd: dict[str, torch.Tensor] = {}

        for i, (prefix, suffixes) in enumerate(sorted(layer_to_suffixes.items())):
            params: dict[str, torch.Tensor] = {}
            for suf in suffixes:
                params[suf] = f.get_tensor(f"{prefix}.{suf}")

            # ---- normalize: make the per-layer state-dict self-contained ----
            # vllm-omni's diffusers_loader doesn't whitelist SVDQuant-specific
            # suffixes in `_QUANTIZED_WEIGHT_SUFFIXES`, so missing wcscales /
            # wtscale would be treated as unexpected_missing → ValueError.
            # Fill with the kernel-identity defaults vLLM uses in create_weights.
            qweight = params["qweight"]
            N = qweight.shape[0]
            lora_dtype = params["proj_up"].dtype  # bf16 for NVFP4 per vLLM convention

            # wcscales (NVFP4 only): default ones = identity per-channel scale
            if precision == "nvfp4" and "wcscales" not in params:
                params["wcscales"] = torch.ones(N, dtype=lora_dtype)
            # wtscale (NVFP4 only): default 1.0 = identity per-tensor scale.
            # Also normalize 0-D → 1-D for the entries that are present.
            if precision == "nvfp4":
                if "wtscale" not in params:
                    params["wtscale"] = torch.tensor([1.0], dtype=lora_dtype)
                elif params["wtscale"].dim() == 0:
                    params["wtscale"] = params["wtscale"].view(1).contiguous()

            # ---- transform: nunchaku fragment → row-major (+ SwiGLU
            # half-swap for fused gate-up layers). On-disk format is
            # canonical row-major regardless of target backend; nunchaku
            # backend repacks at load time in vLLM.
            params = unpack_nvfp4_layer(params, half_swap_n=is_fused_gate_up(prefix))

            # ---- emit: rename source prefix to vllm-omni's param naming ----
            out_prefix = rename_key(f"{prefix}.dummy")[: -len(".dummy")]
            for suf, t in params.items():
                out_sd[f"{out_prefix}.{suf}"] = t
            if progress and (i % 20 == 0 or i == n_linears - 1):
                print(f"  [{i + 1}/{n_linears}] {prefix}" + (f"  ->  {out_prefix}" if out_prefix != prefix else ""))

        # Leftover (unquantized) keys: rename too (most are no-ops; safer to apply uniformly).
        for k in leftover:
            out_sd[rename_key(k)] = f.get_tensor(k)

    # ----- transformer/config.json: inject quantization_config -----
    # vllm-omni reads `transformer/config.json["quantization_config"]` to
    # auto-detect the quant method (see `OmniDiffusionConfig` /
    # `TransformerConfig.from_dict`); a sidecar `quantization_config.json`
    # is *not* consulted. Mirror what `merge_mxfp8_checkpoint.py` does:
    # load base config.json, inject the dict, write back.
    #
    # Per-component routing (`ComponentQuantizationConfig`): SVDQuant is an
    # *offline* method — checkpoints have `.qweight`/`.wscales`/... keys,
    # not `.weight`. If we apply it to ZImagePipeline globally, the Qwen3
    # text encoder (BF16, ships with `.weight` keys) gets its linears
    # wrapped in SVDQuant slots too and refuses to load. The PR #1034
    # (Z-Image FP8) path got away with this because FP8 *online* mode
    # accepts plain `.weight` and converts at load-time; SVDQuant can't.
    #
    # Scope: prefix `"model"` matches the Qwen3 text encoder layers
    # (`model.layers.X.{self_attn,mlp}.*`) — masked to None. Everything
    # else (Z-Image DiT prefixes `layers.X.*` / `noise_refiner.X.*` /
    # `context_refiner.X.*`) falls through to the `default` SVDQuant rule.
    with open(base_transformer / "config.json") as fp:
        tf_config = json.load(fp)
    tf_config["quantization_config"] = {
        # Text encoder (Qwen3): unquantized. Its only nn.Linear instances
        # live under `model.layers.*` and get prefixed accordingly by
        # `recursive_replace_linear` (utils.py:96 starts prefix="").
        "model": None,
        "default": {
            # Use `quant_method` (HF convention; matches MXFP8 path).
            # `vllm-omni`'s factory accepts either `quant_method` or `method`.
            "quant_method": "svdquant",
            "rank": rank,
            "precision": precision,
            "act_unsigned": False,
            # `lm_head` covers Qwen3's text-encoder language-modeling
            # head, which lives at the **top level** of `Qwen3ForCausalLM`
            # — *not* under `model.*` — so it is not caught by the
            # `"model": None` prefix rule above. Without this substring
            # skip it falls through to SVDQuant and hits a tied-weight
            # `data_ptr` error on the first forward (see vllm-omni
            # diffusers_loader handling of Qwen3 text encoder).
            #
            # Other precision-sensitive Z-Image DiT linears
            # (cap_embedder, x_embedder, adaLN_modulation, t_embedder,
            # FinalLayer.linear) already pass `quant_config=None` in
            # the model class itself, so they need no entry here.
            "modules_to_not_convert": ["lm_head"],
        },
    }
    out_config_path = transformer_dir / "config.json"
    # Defensive: a previous run may have hard-linked config.json from the base
    # snapshot; open(..., "w") would truncate the shared inode and corrupt
    # the base's cached blob. Unlink first to detach.
    if out_config_path.exists() or out_config_path.is_symlink():
        out_config_path.unlink()
    with open(out_config_path, "w") as fp:
        json.dump(tf_config, fp, indent=2)
    print(f"wrote {out_config_path} (with embedded quantization_config)")

    # ----- Write the converted single safetensors -----
    out_path = transformer_dir / "diffusion_pytorch_model.safetensors"
    # Preserve nunchaku metadata so downstream can still inspect provenance.
    out_metadata = {k: v for k, v in metadata.items() if isinstance(v, str)}
    out_metadata["conversion"] = json.dumps(
        {
            "tool": "vllm_omni.quantization.tools.convert_nunchaku_to_svdquant",
            "layout": "row_major",  # canonical; vLLM repacks for nunchaku backend at load
            "half_swapped_layers": [p for p in layer_to_suffixes if is_fused_gate_up(p)],
        }
    )
    save_file(out_sd, str(out_path), metadata=out_metadata)
    print(f"wrote {out_path} ({out_path.stat().st_size / 2**30:.2f} GiB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--nunchaku-checkpoint",
        required=True,
        help="Local path to nunchaku merged .safetensors OR HF spec "
        "<repo_id>/<filename> (e.g. nunchaku-tech/nunchaku-z-image-turbo"
        "/svdq-fp4_r128-z-image-turbo.safetensors).",
    )
    parser.add_argument(
        "--base-pipeline",
        default="Tongyi-MAI/Z-Image-Turbo",
        help="Local diffusers folder OR HF repo id of the unquantized base "
        "pipeline. Default: Tongyi-MAI/Z-Image-Turbo.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output diffusers folder path.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy non-transformer files instead of hard-linking (slower, "
        "uses ~35 GiB extra). Default: hard-link (HF upload-safe).",
    )
    args = parser.parse_args()

    nunchaku_path = _resolve_nunchaku_checkpoint(args.nunchaku_checkpoint)
    base_path = _resolve_base_pipeline(args.base_pipeline)
    output_dir = Path(args.output_dir).expanduser()

    print(f"nunchaku checkpoint: {nunchaku_path}")
    print(f"base pipeline:       {base_path}")
    print(f"output:              {output_dir}")
    print()

    convert(
        nunchaku_checkpoint=nunchaku_path,
        base_pipeline=base_path,
        output_dir=output_dir,
        prefer_copy=args.copy,
    )
    print("\ndone.")


if __name__ == "__main__":
    main()
