# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline, data-free rank-32 SVD/RTN export. Quality must be validated separately.

Consumes a standard FL2VA or Ref2VA partition, never downloads or publishes
weights. This is a reproducible candidate recipe, not an official calibrated
MiniMax release or a performance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vllm_omni.diffusion.models.minimax_h3.adaln_lookup import canonical_lookup_timesteps
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
    MiniMaxH3DiTArchConfig,
    _reorder_grouped_qkv_to_qkv,
)
from vllm_omni.diffusion.models.minimax_h3.time_request import _time_shift_sigmas


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TensorSource:
    def __init__(self, directory: Path):
        self.files: dict[str, Path] = {}
        for path in sorted(directory.glob("*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as source:
                for name in source.keys():
                    if name in self.files:
                        raise ValueError(f"duplicate source tensor {name}")
                    self.files[name] = path
        if not self.files:
            raise ValueError(f"no safetensors in {directory}")

    def get(self, name: str) -> torch.Tensor:
        with safe_open(self.files[name], framework="pt", device="cpu") as source:
            return source.get_tensor(name)


class ShardWriter:
    def __init__(self, directory: Path, limit_bytes: int = 2 * 1024**3):
        directory.mkdir(parents=True)
        self.directory = directory
        self.limit_bytes = limit_bytes
        self.pending: dict[str, torch.Tensor] = {}
        self.weight_map: dict[str, str] = {}
        self.total_bytes = 0
        self.pending_bytes = 0
        self.shards: list[dict[str, object]] = []

    def add(self, name: str, tensor: torch.Tensor):
        if name in self.weight_map or name in self.pending:
            raise ValueError(f"duplicate output tensor {name}")
        size = tensor.numel() * tensor.element_size()
        if self.pending and self.pending_bytes + size > self.limit_bytes:
            self.flush()
        self.pending[name] = tensor.detach().cpu().contiguous()
        self.pending_bytes += size
        self.total_bytes += size

    def flush(self):
        if not self.pending:
            return
        filename = f"model-{len(self.shards) + 1:05d}.safetensors"
        path = self.directory / filename
        save_file(self.pending, str(path), metadata={"format": "pt"})
        self.weight_map.update(dict.fromkeys(self.pending, filename))
        self.shards.append({"file": filename, "sha256": sha256(path), "bytes": path.stat().st_size})
        self.pending.clear()
        self.pending_bytes = 0

    def finish(self):
        self.flush()
        (self.directory / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": self.total_bytes}, "weight_map": self.weight_map}, indent=2) + "\n"
        )
        return self.shards


def pack_nvfp4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """E2M1 RN-even residual, E4M3 block scales and BF16 global scale."""
    if weight.ndim != 2 or weight.shape[1] % 16:
        raise ValueError("NVFP4 requires a 2D weight with K divisible by 16")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("nonfinite source weight")
    blocks = weight.float().reshape(weight.shape[0], -1, 16)
    outer = (blocks.abs().amax() / (6 * 448)).clamp(min=torch.finfo(torch.bfloat16).tiny).to(torch.bfloat16)
    scales = (blocks.abs().amax(dim=-1) / (6 * outer.float())).to(torch.float8_e4m3fn)
    denominator = scales.float() * outer.float()
    normalized = blocks / denominator.clamp(min=torch.finfo(torch.float32).tiny).unsqueeze(-1)
    boundaries = weight.new_tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)
    absolute = normalized.abs().contiguous()
    codes = torch.bucketize(absolute, boundaries)
    ties = (codes < 7) & (absolute == boundaries[codes.clamp(max=6)]) & (codes % 2 == 1)
    codes = (codes + ties.to(torch.int64)).to(torch.uint8)
    codes |= (normalized < 0).to(torch.uint8) << 3
    codes = codes.reshape(weight.shape)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed.view(torch.int8), scales.T.contiguous(), outer.reshape(1)


def quantize_linear(weight: torch.Tensor, rank: int, seed: int) -> dict[str, torch.Tensor]:
    if rank > min(weight.shape):
        raise ValueError("correction rank exceeds weight dimensions")
    torch.manual_seed(seed)
    dense = weight.float()
    u, s, v = torch.svd_lowrank(dense, q=rank, niter=2)
    # Subtract the serialized BF16 factors, rather than their FP32 precursors.
    down = v.to(torch.bfloat16)
    up = (u * s).to(torch.bfloat16)
    residual = dense - up.float() @ down.float().T
    packed, scales, outer = pack_nvfp4(residual)
    return {
        "qweight": packed,
        "wscales": scales,
        "wtscale": outer,
        "wcscales": torch.ones(weight.shape[0], device=weight.device, dtype=torch.bfloat16),
        "smooth_factor": torch.ones(weight.shape[1], device=weight.device, dtype=torch.bfloat16),
        "proj_down": down,
        "proj_up": up,
    }


def export_timesteps(steps: list[int], video_shift: float, audio_shift: float) -> list[float]:
    values = {0.0, 1.0, float(torch.tensor(0.999, dtype=torch.float32))}
    for count in steps:
        for shift, anchor in ((video_shift, 0.999), (audio_shift, 1.0)):
            for sigma in _time_shift_sigmas(num_steps=count, shift_scale=shift)[:-1]:
                time = float(torch.tensor(1.0 - sigma, dtype=torch.float32))
                values.update((time, float(torch.tensor(max(time, anchor), dtype=torch.float32))))
    return canonical_lookup_timesteps(sorted(values))


def time_embeddings(source: TensorSource, times: torch.Tensor, arch: MiniMaxH3DiTArchConfig) -> torch.Tensor:
    half = arch.timestep_input_dim // 2
    frequencies = torch.exp(-math.log(10000.0) * torch.arange(half, device=times.device).float() / half)
    arguments = times[:, None] * frequencies[None]
    rows = torch.cat((arguments.cos(), arguments.sin()), dim=-1)
    for projection in ("proj_in", "proj_out"):
        prefix = f"time_embedder.{projection}"
        rows = torch.nn.functional.linear(
            rows, source.get(prefix + ".weight").to(times.device), source.get(prefix + ".bias").to(times.device)
        )
        if projection == "proj_in":
            rows = torch.nn.functional.silu(rows)
    return rows


def export_transformer(source_dir: Path, destination: Path, times: list[float], device: torch.device, seed: int):
    config = json.loads((source_dir / "config.json").read_text())
    if config.get("quantization_config") or config.get("adaln_curve_rank"):
        raise ValueError("export requires standard full-precision AdaLN and DiT weights")
    arch = MiniMaxH3DiTArchConfig.from_mapping(config)
    source = TensorSource(source_dir)
    writer = ShardWriter(destination)
    embeddings = time_embeddings(source, torch.tensor(times, device=device), arch)
    activated = torch.nn.functional.silu(embeddings).to(torch.bfloat16)
    skipped: set[str] = set()
    for block in [*(f"blocks.{index}" for index in range(arch.num_layers)), "final_layer"]:
        prefix = block + ".adaln_proj.linear"
        weight_name, bias_name = prefix + ".weight", prefix + ".bias"
        weight, bias = source.get(weight_name).to(device), source.get(bias_name).to(device)
        if weight.dtype != torch.bfloat16 or bias.dtype != torch.bfloat16:
            raise ValueError(f"{prefix} must be BF16")
        writer.add(block + ".adaln_proj.lookup_values", torch.nn.functional.linear(activated, weight, bias))
        skipped.update((weight_name, bias_name))
    for index, name in enumerate(sorted(source.files)):
        if name in skipped:
            continue
        tensor = source.get(name)
        active = name.startswith("blocks.") and name.endswith(
            (".attn.qkv_proj.weight", ".attn.out_proj.weight", ".mlp.fc1.weight", ".mlp.fc2.weight")
        )
        if active:
            tensor = tensor.to(device)
            if name.endswith(".attn.qkv_proj.weight"):
                tensor = _reorder_grouped_qkv_to_qkv(
                    tensor,
                    num_query_groups=arch.num_attention_heads,
                    heads_per_group=1,
                    head_dim=arch.attention_head_dim,
                )
            for suffix, value in quantize_linear(tensor, rank=32, seed=seed + index).items():
                writer.add(name.removesuffix(".weight") + "." + suffix, value)
            print(f"quantized {name}", flush=True)
        else:
            writer.add(name, tensor)
    config["adaln_lookup_timesteps"] = times
    config["quantization_config"] = {
        "quant_method": "svdquant",
        "rank": 32,
        "precision": "nvfp4",
        "activation_bits": 4,
        "modules_to_not_convert": ["token_refiner", "condition_proj", "adaln_proj"],
    }
    shards = writer.finish()
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return shards


def export_encoder(source_dir: Path, destination: Path, device: torch.device, seed: int):
    source = TensorSource(source_dir)
    config = json.loads((source_dir / "config.json").read_text())
    if config.get("quantization_config"):
        raise ValueError("export requires BF16 text encoder weights")
    writer = ShardWriter(destination)
    consumed = set()
    for index in range(50):
        prefix = f"model.language_model.layers.{index}"
        groups = {
            ".self_attn.qkv_proj": [".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj"],
            ".mlp.gate_up_proj": [".mlp.gate_proj", ".mlp.up_proj"],
            ".self_attn.o_proj": [".self_attn.o_proj"],
            ".mlp.down_proj": [".mlp.down_proj"],
        }
        for group_index, (target, inputs) in enumerate(groups.items()):
            names = [prefix + path + ".weight" for path in inputs]
            weight = torch.cat([source.get(name) for name in names]).to(device)
            for suffix, value in quantize_linear(weight, rank=32, seed=seed + 4 * index + group_index).items():
                writer.add(prefix + target + "." + suffix, value)
            consumed.update(names)
        print(f"quantized encoder layer {index}", flush=True)
    for name in sorted(source.files):
        if name in consumed or name in ("lm_head.weight", "model.language_model.norm.weight"):
            continue
        if name.startswith("model.language_model.layers.") and int(name.split(".")[3]) >= 50:
            continue
        writer.add(name, source.get(name))
    shards = writer.finish()
    for path in source_dir.iterdir():
        if (
            path.is_file()
            and path.suffix != ".safetensors"
            and path.name not in ("config.json", "model.safetensors.index.json")
        ):
            shutil.copy2(path, destination / path.name)
    config["quantization_config"] = {
        "quant_method": "svdquant",
        "rank": 32,
        "precision": "nvfp4",
        "activation_bits": 16,
    }
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return shards


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="standard FL2VA or Ref2VA partition")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[50])
    parser.add_argument("--video-shift", type=float, default=12.0)
    parser.add_argument("--audio-shift", type=float, default=3.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=6493)
    args = parser.parse_args()
    if any(count < 2 for count in args.steps):
        parser.error("--steps values must be >=2")
    index = json.loads((args.source / "model_index.json").read_text())
    release = index.get("_minimax_h3") or {}
    if release.get("partition") not in ("fl2va", "ref2va") or release.get("base_schedule"):
        parser.error("requires a standard, undistilled FL2VA or Ref2VA partition")
    times = export_timesteps(args.steps, args.video_shift, args.audio_shift)
    args.destination.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    transformer = export_transformer(
        args.source / "transformer", args.destination / "transformer", times, device, args.seed
    )
    encoder = export_encoder(args.source / "text_encoder", args.destination / "text_encoder", device, args.seed)
    for component in ("audio_vae", "video_vae", "tokenizer", "processor"):
        shutil.copytree(args.source / component, args.destination / component)
    # Keep keyframe encoding FP32. Store decoder weights FP16, matching its
    # native FP16 autocast compute; audio VAE precision is unchanged.
    vae_dir = args.destination / "video_vae"
    for path in vae_dir.rglob("*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as source:
            tensors = {
                name: source.get_tensor(name).half() if "decoder." in name else source.get_tensor(name)
                for name in source.keys()
            }
        temporary = path.with_suffix(".safetensors.tmp")
        save_file(tensors, str(temporary), metadata={"format": "pt"})
        temporary.replace(path)
    vae_config = json.loads((vae_dir / "config.json").read_text())
    vae_config["decode_dtype"] = "float16"
    (vae_dir / "config.json").write_text(json.dumps(vae_config, indent=2) + "\n")
    (args.destination / "model_index.json").write_text(json.dumps(index, indent=2) + "\n")
    if (args.source.parent / "LICENSE").is_file():
        shutil.copy2(args.source.parent / "LICENSE", args.destination / "LICENSE")
    manifest = {
        "recipe": "data-free-svd-rank32-rtn-nvfp4",
        "calibrated": False,
        "quality_validated": False,
        "source_revision": args.source_revision,
        "exporter_sha256": sha256(Path(__file__)),
        "steps": args.steps,
        "video_shift": args.video_shift,
        "audio_shift": args.audio_shift,
        "seed": args.seed,
        "torch": torch.__version__,
        "transformer": transformer,
        "text_encoder": encoder,
    }
    (args.destination / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
