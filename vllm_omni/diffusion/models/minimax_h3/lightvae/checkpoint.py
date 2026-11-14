# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Portable readers and native-layout conversion for standard H3 Lite weights."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path

import torch
from safetensors import safe_open

_DIFFUSERS_WEIGHTS_INDEX = "diffusion_pytorch_model.safetensors.index.json"
_KNOWN_TEMPORAL_DECODE_PROTOCOLS = frozenset({"released-segment-prepadding-v3"})
_KNOWN_TILE_LAYOUTS = frozenset({"native"})
_H3_CLIP_LENGTH = 17
_H3_TOKEN_DROP = 3


@dataclass(frozen=True)
class H3LiteDecodeProfile:
    """Decoder tiling contract for the 26-layer LightVAE release.

    The release uses an anisotropic, frame-kind-aware geometry:
    ``tile_size``/``short_axis_overlap`` on the short spatial axis,
    ``long_axis_tile``/``long_axis_overlap`` on the long axis, and
    ``image_tile_size``/``image_long_axis_tile`` for a single image frame.
    Values are in pixel space and are applied by the native decoder adapter.

    ``tile_overlap`` is the release's base/legacy overlap. It is used when a
    per-axis override is absent and when the official VAE config is the
    fallback source.
    """

    tile_size: int | None = None
    tile_overlap: int | None = None
    long_axis_tile: int | None = None
    short_axis_overlap: int | None = None
    long_axis_overlap: int | None = None
    image_tile_size: int | None = None
    image_long_axis_tile: int | None = None
    image_context_tokens: int | None = None
    image_context_output_phase: int | None = None
    tile_layout: str | None = None
    temporal_decode_protocol: str | None = None


@dataclass(frozen=True)
class H3LiteDecoderCheckpoint:
    config: dict[str, object]
    kept_layers: tuple[int, ...]
    state_dict: dict[str, torch.Tensor]
    decode_profile: H3LiteDecodeProfile | None = None


def resolve_h3_lightvae_decoder_path(additional: Mapping[str, object] | None) -> str | None:
    """Resolve a LightVAE decoder path.

    The value is either a compact 12-layer ``.safetensors`` file or a
    26-layer Diffusers directory. ``vae_path`` and ``vae_decoder_path`` are
    aliases for framework integrations that expose a generic VAE path.
    """
    if not additional:
        return None
    present = [key for key in ("vae_path", "lightvae_decoder_path", "vae_decoder_path") if key in additional]
    if not present:
        return None
    if len(present) > 1:
        raise ValueError("use only one of vae_path/lightvae_decoder_path/vae_decoder_path")
    key = present[0]
    value = additional[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a nonempty local path")
    if additional.get("video_vae_decoder") is not None:
        raise ValueError(f"{key} cannot be combined with video_vae_decoder")
    return value


def _validate_config(config: dict[str, object]) -> tuple[int, ...]:
    if config.get("family", "h3") != "h3":
        raise ValueError("LightVAE family must be h3")
    for field in (
        "latent_channels",
        "decoder_num_layers",
        "decoder_num_attention_heads",
        "decoder_attention_head_dim",
    ):
        value = config.get(field)
        if type(value) is not int or value <= 0:
            raise ValueError(f"LightVAE {field} must be a positive integer")
    kept = config.get("kept_layers")
    depth = config["decoder_num_layers"]
    if (
        not isinstance(kept, list)
        or not kept
        or any(type(index) is not int for index in kept)
        or kept != sorted(set(kept))
        or kept[0] < 0
        or kept[-1] >= depth
    ):
        raise ValueError("LightVAE kept_layers must be ordered unique teacher indices")
    for field in ("latents_mean", "latents_std"):
        values = config.get(field)
        if (
            not isinstance(values, list)
            or len(values) != config["latent_channels"]
            or any(type(x) not in (float, int) or not math.isfinite(x) for x in values)
            or (field == "latents_std" and any(x <= 0 for x in values))
        ):
            raise ValueError(f"LightVAE {field} does not match latent_channels")
    for field, expected in (("clip_length", _H3_CLIP_LENGTH), ("token_drop", _H3_TOKEN_DROP)):
        if config.get(field) != expected:
            raise ValueError(f"LightVAE {field} must be {expected}")
    return tuple(kept)


def read_h3_lite_decoder_checkpoint(
    path: str | Path,
    *,
    teacher_num_layers: int | None = None,
) -> H3LiteDecoderCheckpoint:
    """Read decoder tensors on the CPU.

    ``path`` is either a compact 12-layer LightVAE ``.safetensors`` file with
    ``architecture`` metadata, or a 26-layer Diffusers
    ``AutoencoderKLMiniMaxH3`` directory with ``config.json`` and sharded
    safetensors. For the directory form, ``teacher_num_layers`` must name the
    native decoder depth the student is installed into.
    """
    path = Path(path)
    if path.is_dir():
        if teacher_num_layers is None:
            raise ValueError(f"{path}: teacher_num_layers is required for a Diffusers LightVAE directory")
        config, kept, state, decode_profile = _read_diffusers_directory(path, teacher_num_layers)
    elif path.suffix == ".safetensors":
        config, kept, state = _read_compact_file(path)
        decode_profile = None
    else:
        raise ValueError(f"{path}: expected a LightVAE safetensors file or a Diffusers directory")
    indices = {
        int(name.split(".")[2])
        for name in state
        if name.startswith("decoder.transformer_blocks.") and name.split(".")[2].isdigit()
    }
    if indices != set(range(len(kept))):
        raise ValueError(f"{path}: decoder.transformer_blocks must use compact indices")
    return H3LiteDecoderCheckpoint(config, kept, state, decode_profile)


def _read_compact_file(path: Path) -> tuple[dict[str, object], tuple[int, ...], dict[str, torch.Tensor]]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        try:
            architecture = json.loads((handle.metadata() or {})["architecture"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}: missing or invalid architecture metadata") from exc
        if not isinstance(architecture, dict):
            raise ValueError(f"{path}: architecture must be an object")
        config = architecture.get("teacher_config", architecture)
        if not isinstance(config, dict):
            raise ValueError(f"{path}: architecture.teacher_config must be an object")
        config = dict(config)
        config["kept_layers"] = architecture.get("kept_layers")
        kept = _validate_config(config)
        state = {}
        for name in handle.keys():
            if name in ("latents_mean", "latents_std"):
                continue
            if not name.startswith(("decoder.", "post_quant_conv.")):
                raise ValueError(f"{path}: unsupported checkpoint tensor {name}")
            tensor = handle.get_tensor(name)
            if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise ValueError(f"{path}: unsupported or quantized tensor {name}")
            state[name] = tensor
    return config, kept, state


def _diffusers_shards(path: Path) -> list[Path]:
    index_path = path / _DIFFUSERS_WEIGHTS_INDEX
    if index_path.is_file():
        try:
            with open(index_path) as f:
                index = json.load(f)
            weight_map = index["weight_map"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}: invalid {index_path.name}") from exc
        if not isinstance(weight_map, dict):
            raise ValueError(f"{path}: {index_path.name} weight_map must be an object")
        names = sorted({str(name) for name in weight_map.values()})
        shards = [path / name for name in names]
        missing = [shard.name for shard in shards if not shard.is_file()]
        if missing:
            raise ValueError(f"{path}: missing safetensors shards {missing}")
        return shards
    shards = sorted(path.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"{path}: no safetensors shards found")
    return shards


def _validate_profile_int(path: Path, field: str, value: object, *, allow_zero: bool) -> int | None:
    if value is None:
        return None
    low = 0 if allow_zero else 1
    if type(value) is not int or value < low:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{path}: {field} must be a {qualifier} integer")
    return value


def _validate_profile_overlaps(path: Path, profile: H3LiteDecodeProfile) -> None:
    def require(field: str, overlap: int | None, tile: int | None) -> None:
        if overlap is None or tile is None:
            return
        if overlap >= tile:
            raise ValueError(f"{path}: {field}={overlap} must be smaller than its tile size {tile}")

    short_tile = profile.tile_size
    short_overlap = profile.short_axis_overlap if profile.short_axis_overlap is not None else profile.tile_overlap
    long_tile = profile.long_axis_tile if profile.long_axis_tile is not None else short_tile
    long_overlap = profile.long_axis_overlap if profile.long_axis_overlap is not None else profile.tile_overlap
    image_tile = profile.image_tile_size if profile.image_tile_size is not None else short_tile
    image_long_tile = profile.image_long_axis_tile if profile.image_long_axis_tile is not None else image_tile
    image_overlap = profile.tile_overlap if profile.tile_overlap is not None else short_overlap

    require("student_short_axis_overlap", short_overlap, short_tile)
    require("student_long_axis_overlap", long_overlap, long_tile)
    require("student_tile_overlap", image_overlap, image_tile)
    require("student_tile_overlap", image_overlap, image_long_tile)


def _profile_from_official_config(path: Path, raw: Mapping[str, object]) -> H3LiteDecodeProfile | None:
    """Build a fallback profile from the official H3 VAE config.

    The official component stores ``vae_tile_size`` / ``vae_tile_overlap_min``
    in ``config.json``. Accept the legacy ``tile_size`` names too so a caller
    can point ``vae_path`` at either the component directory or a converted
    config. If the config does not contain tiling fields, return ``None`` and
    let the native loader keep its own defaults.
    """
    tile = raw.get("vae_tile_size", raw.get("tile_size"))
    overlap = raw.get("vae_tile_overlap_min", raw.get("tile_overlap_min"))
    if tile is None and overlap is None:
        return None
    tile = _validate_profile_int(path, "vae_tile_size", tile, allow_zero=False)
    overlap = _validate_profile_int(path, "vae_tile_overlap_min", overlap, allow_zero=True)
    if tile is None or overlap is None:
        return None
    return H3LiteDecodeProfile(
        tile_size=tile,
        tile_overlap=overlap,
        long_axis_tile=tile,
        short_axis_overlap=overlap,
        long_axis_overlap=overlap,
        image_tile_size=tile,
        image_long_axis_tile=tile,
        tile_layout="native",
    )


def _merge_profile(
    primary: H3LiteDecodeProfile,
    fallback: H3LiteDecodeProfile | None,
) -> H3LiteDecodeProfile:
    if fallback is None:
        return primary
    values = {
        field.name: getattr(primary, field.name)
        if getattr(primary, field.name) is not None
        else getattr(fallback, field.name)
        for field in fields(H3LiteDecodeProfile)
    }
    return replace(primary, **values)


def _read_decode_profile(
    path: Path,
    *,
    fallback: H3LiteDecodeProfile | None = None,
) -> H3LiteDecodeProfile | None:
    decode_path = path / "decode_config.json"
    if not decode_path.is_file():
        if fallback is not None:
            _validate_profile_overlaps(path, fallback)
        return fallback
    try:
        with open(decode_path) as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid decode_config.json") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: decode_config.json must be an object")

    def _int_field(name: str, *, allow_zero: bool) -> int | None:
        return _validate_profile_int(path, f"decode_config.json {name}", raw.get(name), allow_zero=allow_zero)

    protocol = raw.get("temporal_decode_protocol")
    if protocol is not None and protocol not in _KNOWN_TEMPORAL_DECODE_PROTOCOLS:
        raise ValueError(f"{path}: unsupported temporal_decode_protocol {protocol!r}")
    tile_layout = raw.get("tile_layout")
    if tile_layout is not None and tile_layout not in _KNOWN_TILE_LAYOUTS:
        raise ValueError(f"{path}: unsupported tile_layout {tile_layout!r}")

    profile = H3LiteDecodeProfile(
        tile_size=_int_field("student_tile_size", allow_zero=False),
        tile_overlap=_int_field("student_tile_overlap", allow_zero=True),
        long_axis_tile=_int_field("student_long_axis_tile", allow_zero=False),
        short_axis_overlap=_int_field("student_short_axis_overlap", allow_zero=True),
        long_axis_overlap=_int_field("student_long_axis_overlap", allow_zero=True),
        image_tile_size=_int_field("image_tile_size", allow_zero=False),
        image_long_axis_tile=_int_field("image_long_axis_tile", allow_zero=False),
        image_context_tokens=_int_field("image_context_tokens", allow_zero=False),
        image_context_output_phase=_int_field("image_context_output_phase", allow_zero=True),
        tile_layout=tile_layout,
        temporal_decode_protocol=protocol,
    )
    merged = _merge_profile(profile, fallback)
    _validate_profile_overlaps(path, merged)
    return merged


def _read_diffusers_directory(
    path: Path,
    teacher_num_layers: int,
) -> tuple[dict[str, object], tuple[int, ...], dict[str, torch.Tensor], H3LiteDecodeProfile | None]:
    if type(teacher_num_layers) is not int or teacher_num_layers <= 0:
        raise ValueError("teacher_num_layers must be a positive integer")
    config_path = path / "config.json"
    try:
        with open(config_path) as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: missing or invalid config.json") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: config.json must be an object")
    class_name = raw.get("_class_name")
    if class_name not in (None, "AutoencoderKLMiniMaxH3"):
        raise ValueError(f"{path}: config.json is not an AutoencoderKLMiniMaxH3 component")
    student_num_layers = raw.get("decoder_num_layers")
    if type(student_num_layers) is not int or student_num_layers <= 0:
        raise ValueError(f"{path}: decoder_num_layers must be a positive integer")
    if student_num_layers > teacher_num_layers:
        raise ValueError(
            f"{path}: student decoder depth {student_num_layers} exceeds teacher depth {teacher_num_layers}"
        )
    config: dict[str, object] = {
        "family": "h3",
        "latent_channels": raw.get("latent_channels"),
        "decoder_num_layers": teacher_num_layers,
        "decoder_num_attention_heads": raw.get("decoder_num_attention_heads"),
        "decoder_attention_head_dim": raw.get("decoder_attention_head_dim"),
        "decoder_norm_eps": raw.get("decoder_norm_eps", raw.get("norm_eps")),
        "decoder_rope_theta": raw.get("decoder_rope_theta"),
        "decoder_rope_dim_ratio": raw.get("decoder_rope_dim_ratio"),
        "latents_mean": raw.get("latents_mean"),
        "latents_std": raw.get("latents_std"),
        "clip_length": raw.get("clip_length"),
        "token_drop": raw.get("token_drop"),
        "kept_layers": raw.get("kept_layers", list(range(student_num_layers))),
    }
    kept = _validate_config(config)
    state: dict[str, torch.Tensor] = {}
    for shard in _diffusers_shards(path):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if not name.startswith(("decoder.", "post_quant_conv.")):
                    continue
                tensor = handle.get_tensor(name)
                if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                    raise ValueError(f"{path}: unsupported or quantized tensor {name}")
                if name in state:
                    raise ValueError(f"{path}: duplicate tensor {name}")
                state[name] = tensor
    if not state:
        raise ValueError(f"{path}: no decoder or post_quant_conv tensors found")
    official_profile = _profile_from_official_config(path, raw)
    return config, kept, state, _read_decode_profile(path, fallback=official_profile)


def convert_h3_lite_decoder_state_dict(
    state: Mapping[str, torch.Tensor],
    *,
    num_heads: int,
) -> dict[str, torch.Tensor]:
    """Map Diffusers H3 tensors to the supported native H3 decoder layout."""
    if type(num_heads) is not int or num_heads <= 0:
        raise ValueError("num_heads must be a positive integer")
    converted = {}
    projections: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in state.items():
        parts = name.split(".")
        target = name
        if name.startswith("decoder.transformer_blocks.") and len(parts) >= 4 and parts[2].isdigit():
            suffix = ".".join(parts[3:])
            if len(parts) == 6 and parts[3] == "attn" and parts[4] in ("to_q", "to_k", "to_v"):
                if parts[5] not in ("weight", "bias"):
                    raise ValueError(f"unsupported decoder parameter {name}")
                projection_key = ".".join(parts[:3]) + f".attn.to_qkv.{parts[5]}"
                projections.setdefault(projection_key, {})[parts[4]] = tensor
                continue
            if suffix in ("ff.net.0.proj.weight", "ff.net.0.proj.bias"):
                if tensor.ndim not in (1, 2) or tensor.shape[0] % 2:
                    raise ValueError(f"invalid MLP projection {name}")
                up, gate = tensor.chunk(2, dim=0)
                tensor = torch.cat((gate, up), dim=0)
                target = name.replace(".ff.net.0.proj.", ".ff.w1.")
            elif suffix in ("ff.net.2.weight", "ff.net.2.bias"):
                target = name.replace(".ff.net.2.", ".ff.w2.")
            elif suffix in ("attn.to_out.0.weight", "attn.to_out.0.bias"):
                target = name.replace(".attn.to_out.0.", ".attn.to_out.")
            elif suffix not in ("norm1.weight", "norm2.weight", "scale1", "scale2"):
                raise ValueError(f"unsupported decoder parameter {name}")
        elif name in ("decoder.proj_in.weight", "decoder.proj_in.bias"):
            target = name.replace("decoder.proj_in.", "decoder.x_embedder.")
        elif name not in (
            "decoder.register_tokens",
            "decoder.norm_out.weight",
            "decoder.norm_out.bias",
            "decoder.proj_out.weight",
            "decoder.proj_out.bias",
            "post_quant_conv.weight",
            "post_quant_conv.bias",
        ):
            raise ValueError(f"unsupported decoder parameter {name}")
        converted[target] = tensor
    for target, group in projections.items():
        if set(group) != {"to_q", "to_k", "to_v"}:
            raise ValueError(f"{target}: missing to_q, to_k or to_v")
        tensors = [group[key] for key in ("to_q", "to_k", "to_v")]
        first = tensors[0]
        expected_ndim = 2 if target.endswith("weight") else 1
        if (
            first.ndim != expected_ndim
            or first.shape[0] % num_heads
            or any(t.shape != first.shape or t.dtype != first.dtype for t in tensors)
        ):
            raise ValueError(f"{target}: incompatible to_q/to_k/to_v shapes")
        packed = torch.stack([t.reshape(num_heads, -1, *t.shape[1:]) for t in tensors], dim=1)
        converted[target] = packed.reshape(3 * first.shape[0], *first.shape[1:])
    return converted
