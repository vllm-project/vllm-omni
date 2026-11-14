# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Install standard Lite weights into the supported native H3 VAE structure."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from typing import cast

import torch
import torch.distributed as dist
from torch import nn

from .checkpoint import (
    H3LiteDecodeProfile,
    H3LiteDecoderCheckpoint,
    convert_h3_lite_decoder_state_dict,
)


def install_h3_lite_decoder(
    model: nn.Module,
    checkpoint: H3LiteDecoderCheckpoint,
    *,
    native_config: Mapping[str, object],
) -> None:
    """Validate and replace decoder weights before residency hooks are built."""
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("H3 Lite decoder installation must happen on the CPU")
    decoder = getattr(model, "decoder", None)
    blocks = getattr(decoder, "transformer_blocks", None)
    post_quant = getattr(model, "post_quant_conv", None)
    if (
        not isinstance(decoder, nn.Module)
        or not isinstance(blocks, nn.ModuleList)
        or not isinstance(post_quant, nn.Module)
        or not isinstance(getattr(decoder, "x_embedder", None), nn.Linear)
        or len(blocks) != checkpoint.config["decoder_num_layers"]
    ):
        raise ValueError("unsupported native H3 decoder structure or teacher depth")
    for field in ("latent_channels", "latents_mean", "latents_std"):
        if native_config.get(field) != checkpoint.config[field]:
            raise ValueError(f"H3 Lite configuration disagrees on {field}")
    for field in ("clip_length", "token_drop"):
        actual = native_config.get(f"vae_{field}", native_config.get(field))
        if actual != checkpoint.config[field]:
            raise ValueError(f"H3 Lite configuration disagrees on {field}")
    heads = cast(int, checkpoint.config["decoder_num_attention_heads"])
    head_dim = cast(int, checkpoint.config["decoder_attention_head_dim"])
    hidden = heads * head_dim
    for field in ("decoder_norm_eps", "decoder_rope_theta", "decoder_rope_dim_ratio"):
        value = checkpoint.config.get(field)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
            or (field in native_config and native_config[field] != value)
        ):
            raise ValueError(f"H3 Lite configuration disagrees on {field}")
    eps = checkpoint.config["decoder_norm_eps"]
    if getattr(getattr(decoder, "norm_out", None), "eps", None) != eps:
        raise ValueError("H3 Lite decoder_norm_eps disagrees with norm_out")
    ratio = cast(float, checkpoint.config["decoder_rope_dim_ratio"])
    rotary_dim = int(head_dim * ratio)
    if ratio > 1 or rotary_dim <= 0 or rotary_dim % 6:
        raise ValueError("unsupported H3 decoder_rope_dim_ratio")
    pos_embed = getattr(decoder, "pos_embed", None)
    inv_freq = getattr(pos_embed, "inv_freq", None)
    expected_freq = 1.0 / cast(float, checkpoint.config["decoder_rope_theta"]) ** torch.arange(
        0, 1, 6 / rotary_dim, dtype=torch.float32, device="cpu"
    )
    if (
        getattr(pos_embed, "n_dim", None) != 3
        or getattr(pos_embed, "angle_scale", None) != 2 * math.pi
        or not isinstance(inv_freq, torch.Tensor)
        or inv_freq.device.type != "cpu"
        or inv_freq.shape != expected_freq.shape
        or not torch.allclose(inv_freq.float(), expected_freq, rtol=1e-6, atol=0)
    ):
        raise ValueError("unsupported H3 decoder rope frequencies")
    if decoder.x_embedder.in_features != checkpoint.config["latent_channels"]:
        raise ValueError("H3 Lite x_embedder latent_channels mismatch")
    for block in blocks:
        attention = getattr(block, "attn", None)
        for owner, name in (
            (block, "norm1"),
            (block, "norm2"),
            (attention, "norm_q"),
            (attention, "norm_k"),
        ):
            if getattr(getattr(owner, name, None), "eps", None) != eps:
                raise ValueError(f"H3 Lite decoder_norm_eps disagrees with {name}")
        qkv = getattr(attention, "to_qkv", None)
        if (
            not isinstance(qkv, nn.Linear)
            or getattr(attention, "dim_head", None) != head_dim
            or qkv.in_features != hidden
            or qkv.out_features != 3 * hidden
        ):
            raise ValueError("unsupported H3 attention structure or head dimensions")
    converted = convert_h3_lite_decoder_state_dict(checkpoint.state_dict, num_heads=heads)
    # Keep only shapes here. Retaining a second full state-dict reference while
    # the checkpoint tensors and converted tensors are alive increases the
    # install-time CPU peak for the 26-layer decoder.
    expected_shapes: dict[str, tuple[int, ...]] = {}
    for name, tensor in decoder.state_dict().items():
        if name.startswith("transformer_blocks."):
            parts = name.split(".")
            teacher_index = int(parts[1])
            if teacher_index not in checkpoint.kept_layers:
                continue
            parts[1] = str(checkpoint.kept_layers.index(teacher_index))
            name = ".".join(parts)
        expected_shapes[f"decoder.{name}"] = tuple(tensor.shape)
    expected_shapes.update(
        {f"post_quant_conv.{key}": tuple(value.shape) for key, value in post_quant.state_dict().items()}
    )
    mask = getattr(decoder, "mask_token", None)
    if mask is not None:
        if tuple(mask.shape) != (1, 1, hidden):
            raise ValueError("unsupported native H3 training mask_token shape")
        # The native implementation retains this unused training parameter.
        converted["decoder.mask_token"] = torch.zeros_like(mask)
    missing, unexpected = (
        expected_shapes.keys() - converted.keys(),
        converted.keys() - expected_shapes.keys(),
    )
    if missing or unexpected:
        raise ValueError(
            f"H3 Lite decoder parameter mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    for name, tensor in converted.items():
        if tuple(tensor.shape) != expected_shapes[name]:
            raise ValueError(
                f"H3 Lite tensor shape mismatch for {name}: {tuple(tensor.shape)} != {expected_shapes[name]}"
            )
    # The official VAE config is the fallback for fields the release profile
    # does not override (and for a 26-layer directory with no decode_config).
    profile = checkpoint.decode_profile
    if profile is not None:
        profile = _merge_profile_fallback(profile, _official_profile_from_native_config(native_config))
    # Validate the profile before mutating the model. The actual commit happens
    # only after the strict weight load succeeds.
    _decode_profile_updates(model, profile)
    original_blocks = decoder.transformer_blocks
    try:
        decoder.transformer_blocks = nn.ModuleList([blocks[index] for index in checkpoint.kept_layers])
        decoder.load_state_dict(
            {key.removeprefix("decoder."): value for key, value in converted.items() if key.startswith("decoder.")},
            strict=True,
        )
        post_quant.load_state_dict(
            {
                key.removeprefix("post_quant_conv."): value
                for key, value in converted.items()
                if key.startswith("post_quant_conv.")
            },
            strict=True,
        )
    except Exception:
        decoder.transformer_blocks = original_blocks
        raise
    decoder.to(dtype=torch.float32)
    post_quant.to(dtype=torch.float32)
    apply_h3_lite_decode_profile(model, profile)


def _profile_value(model: nn.Module, profile: H3LiteDecodeProfile, name: str, fallback_name: str) -> int | None:
    value = getattr(profile, name)
    if value is not None:
        return value
    value = getattr(model, fallback_name, None)
    return value if type(value) is int else None


def _official_profile_from_native_config(native_config: Mapping[str, object]) -> H3LiteDecodeProfile | None:
    tile = native_config.get("vae_tile_size", native_config.get("tile_size"))
    overlap = native_config.get("vae_tile_overlap_min", native_config.get("tile_overlap_min"))
    if type(tile) is not int or type(overlap) is not int or tile <= 0 or overlap < 0:
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


def _merge_profile_fallback(profile: H3LiteDecodeProfile, fallback: H3LiteDecodeProfile | None) -> H3LiteDecodeProfile:
    if fallback is None:
        return profile
    values = {
        name: getattr(profile, name) if getattr(profile, name) is not None else getattr(fallback, name)
        for name in H3LiteDecodeProfile.__dataclass_fields__
    }
    return H3LiteDecodeProfile(**values)


def _require_overlap_below_tile(field: str, overlap: int | None, tile: int | None) -> None:
    if overlap is None or tile is None:
        return
    if overlap >= tile:
        raise ValueError(f"H3 Lite {field}={overlap} must be smaller than its tile size {tile}")


def _decode_profile_updates(model: nn.Module, profile: H3LiteDecodeProfile | None) -> dict[str, object]:
    """Validate a profile and return the attributes to set, without mutating."""
    if profile is None:
        return {}
    decoder = getattr(model, "decoder", None)
    if not isinstance(decoder, nn.Module):
        raise ValueError("H3 Lite decode profile requires a native decoder")
    ratio = getattr(model, "vae_ratio", None)
    if type(ratio) is not int or ratio <= 0:
        ratio = None

    short_tile = _profile_value(model, profile, "tile_size", "decoder_tile_size")
    short_overlap = profile.short_axis_overlap
    if short_overlap is None:
        short_overlap = profile.tile_overlap
    if short_overlap is None:
        short_overlap = getattr(model, "decoder_tile_overlap_min", None)
    long_tile = _profile_value(model, profile, "long_axis_tile", "decoder_tile_size")
    long_overlap = profile.long_axis_overlap
    if long_overlap is None:
        long_overlap = profile.tile_overlap
    if long_overlap is None:
        long_overlap = getattr(model, "decoder_tile_overlap_min", None)
    image_tile = profile.image_tile_size or short_tile
    image_long_tile = profile.image_long_axis_tile or image_tile
    image_overlap = profile.tile_overlap if profile.tile_overlap is not None else short_overlap

    _require_overlap_below_tile("decoder_short_axis_overlap", short_overlap, short_tile)
    _require_overlap_below_tile("decoder_long_axis_overlap", long_overlap, long_tile)
    if short_tile is not None and short_overlap is not None and long_overlap is not None:
        _require_overlap_below_tile(
            "decoder_square_axis_overlap",
            max(short_overlap, long_overlap),
            short_tile,
        )
    _require_overlap_below_tile("decoder_image_overlap", image_overlap, image_tile)
    _require_overlap_below_tile("decoder_image_overlap", image_overlap, image_long_tile)

    values = {
        "decoder_tile_overlap_min": short_overlap,
        "decoder_tile_size": short_tile,
        "decoder_long_axis_tile": long_tile,
        "decoder_short_axis_overlap": short_overlap,
        "decoder_long_axis_overlap": long_overlap,
        "decoder_image_tile_size": image_tile,
        "decoder_image_long_axis_tile": image_long_tile,
        "decoder_image_overlap": image_overlap,
    }
    updates: dict[str, object] = {}
    for name, value in values.items():
        if value is None:
            continue
        if ratio is not None and value % ratio:
            raise ValueError(f"H3 Lite {name}={value} must be a multiple of the spatial ratio {ratio}")
        if name not in (
            "decoder_short_axis_overlap",
            "decoder_long_axis_overlap",
            "decoder_tile_overlap_min",
            "decoder_image_overlap",
        ):
            if value <= 0:
                raise ValueError(f"H3 Lite {name} must be positive")
        if value < 0:
            raise ValueError(f"H3 Lite {name} must be non-negative")
        updates[name] = value

    if profile.image_context_tokens is not None:
        tokens_chunk_size = getattr(model, "tokens_chunk_size", None)
        if type(tokens_chunk_size) is int and profile.image_context_tokens != tokens_chunk_size:
            raise ValueError(
                "H3 Lite temporal decode protocol disagrees with the native token window: "
                f"{profile.image_context_tokens} != {tokens_chunk_size}"
            )
        updates["image_context_tokens"] = profile.image_context_tokens
    if profile.image_context_output_phase is not None:
        tokens = profile.image_context_tokens or getattr(model, "tokens_chunk_size", None)
        if type(tokens) is int and not 0 <= profile.image_context_output_phase < tokens:
            raise ValueError("H3 Lite image_context_output_phase must be inside the image context window")
        updates["image_context_output_phase"] = profile.image_context_output_phase
    if profile.tile_layout is not None:
        updates["tile_layout"] = profile.tile_layout
    if profile.temporal_decode_protocol is not None:
        updates["temporal_decode_protocol"] = profile.temporal_decode_protocol

    for name in ("decoder_tile_size", "decoder_tile_overlap_min"):
        if name in updates and not hasattr(model, name):
            raise ValueError(f"native H3 VAE does not expose {name}")
    return updates


def _split_tiles_for_profile(
    model: nn.Module,
    length: int,
    size: int,
    min_overlap: int,
    *,
    layout: str,
) -> tuple[list[int], list[int], list[int]]:
    if size <= 0 or min_overlap < 0 or min_overlap >= size:
        raise ValueError(f"H3 Lite tile size {size} must be positive and overlap {min_overlap} must be smaller than it")
    ratio = int(getattr(model, "vae_ratio", 1) or 1)
    if layout == "adaptive":
        if length <= size + min_overlap:
            return [0], [length], []
        count = math.ceil(length / size)
        required = math.ceil((length + min_overlap * (count - 1)) / count / ratio) * ratio
        if required > size + min_overlap:
            raise ValueError("adaptive tile expansion exceeds one overlap")
        size = max(size, required)
        while size * count - min_overlap * (count - 1) < length:
            count += 1
    else:
        if size >= length:
            return [0], [length], []
        count = math.ceil(length / size)
        while size * count - min_overlap * (count - 1) < length:
            count += 1
    overlaps = [min_overlap] * (count - 1)
    remaining = size * count - sum(overlaps) - length
    for i in range(remaining // ratio):
        overlaps[i % (count - 1)] += ratio
    starts = [0]
    for value in overlaps:
        starts.append(starts[-1] + size - value)
    return starts, [size] * count, overlaps


def _lightvae_geometry(
    model: nn.Module,
    profile: H3LiteDecodeProfile,
    frames: int,
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    short = _profile_value(model, profile, "tile_size", "decoder_tile_size")
    long_tile = _profile_value(model, profile, "long_axis_tile", "decoder_tile_size")
    short_overlap = _profile_value(model, profile, "short_axis_overlap", "decoder_tile_overlap_min")
    long_overlap = _profile_value(model, profile, "long_axis_overlap", "decoder_tile_overlap_min")
    if short is None or long_tile is None or short_overlap is None or long_overlap is None:
        raise ValueError("H3 Lite profile is missing decoder tiling geometry")
    if frames == 1:
        image_tile = _profile_value(model, profile, "image_tile_size", "decoder_image_tile_size") or short
        image_long = (
            _profile_value(model, profile, "image_long_axis_tile", "decoder_image_long_axis_tile") or image_tile
        )
        image_overlap = profile.tile_overlap if profile.tile_overlap is not None else short_overlap
        return image_tile, image_long, image_overlap, image_overlap
    if height < width:
        return short, long_tile, short_overlap, long_overlap
    if height > width:
        return long_tile, short, long_overlap, short_overlap
    overlap = max(short_overlap, long_overlap)
    return short, short, overlap, overlap


def _blend_tiles(
    model: nn.Module,
    previous: torch.Tensor,
    current: torch.Tensor,
    extent: int,
    dim: int,
) -> torch.Tensor:
    if extent <= 0:
        return current
    blend = getattr(model, "blend", None) or getattr(model, "_blend", None)
    if not callable(blend):
        raise ValueError("native H3 VAE does not expose a tile blend function")
    return blend(previous, current, extent, dim)


def _native_parallel_state(model: nn.Module) -> Mapping[str, object]:
    module = importlib.import_module(model.__class__.__module__)
    getter = getattr(module, "get_parallel_state", None)
    if callable(getter):
        return getter()
    return {"sp_rank": 0, "sp_size": 1, "sp_process_group": None}


def _tiled_decode_lightvae(
    model: nn.Module,
    profile: H3LiteDecodeProfile,
    z: torch.Tensor,
    *,
    is_image: bool = False,
) -> torch.Tensor:
    """Native H3 tiled decode with the release's per-axis LightVAE geometry."""
    if getattr(model, "parallel_tiling", False):
        state = _native_parallel_state(model)
        sp_rank = int(state["sp_rank"])
        sp_size = int(state["sp_size"])
    else:
        state = {"sp_process_group": None}
        sp_rank, sp_size = 0, 1

    ratio = int(getattr(model, "vae_ratio", 1) or 1)
    height, width = int(z.shape[-2]) * ratio, int(z.shape[-1]) * ratio
    frames = 1 if is_image else int(z.shape[2])
    tile_h, tile_w, overlap_h, overlap_w = _lightvae_geometry(model, profile, frames, height, width)
    layout = profile.tile_layout or "native"
    y_idx, y_len, y_overlap = _split_tiles_for_profile(model, height, tile_h, overlap_h, layout=layout)
    x_idx, x_len, x_overlap = _split_tiles_for_profile(model, width, tile_w, overlap_w, layout=layout)

    i_max, j_max = len(y_idx), len(x_idx)
    num_tiles = i_max * j_max
    z_tiles = []
    for i_pos, i_len in zip(y_idx, y_len):
        i_pos, i_len = i_pos // ratio, i_len // ratio
        for j_pos, j_len in zip(x_idx, x_len):
            j_pos, j_len = j_pos // ratio, j_len // ratio
            z_tiles.append(z[..., i_pos : i_pos + i_len, j_pos : j_pos + j_len])

    local_tile_indices = model._local_tile_indices(num_tiles, sp_rank, sp_size)
    stack_tiling = bool(getattr(model, "stack_tiling", False)) and not (
        bool(getattr(model, "training", False))
        and bool(getattr(getattr(model, "decoder", None), "mask_enabled", False))
    )
    decoded_tasks = model._run_tile_tasks(z_tiles, local_tile_indices, model.decode, stack_tiling)
    if sp_size > 1:
        dist.barrier(group=state.get("sp_process_group"))
        all_decoded = model._all_gather_tiled_results(decoded_tasks, num_tiles)
    else:
        all_decoded = decoded_tasks

    rows = [[None for _ in range(j_max)] for _ in range(i_max)]
    for idx, decoded in enumerate(all_decoded):
        i, j = idx // j_max, idx % j_max
        rows[i][j] = decoded.to(z.device)

    result_rows = []
    for i, row in enumerate(rows):
        result_row = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = _blend_tiles(model, rows[i - 1][j], tile, y_overlap[i - 1], dim=-2)
            if j > 0:
                tile = _blend_tiles(model, row[j - 1], tile, x_overlap[j - 1], dim=-1)
            if i < len(rows) - 1 and y_overlap[i] > 0:
                tile = tile[..., : -y_overlap[i], :]
            if j < len(row) - 1 and x_overlap[j] > 0:
                tile = tile[..., :, : -x_overlap[j]]
            result_row.append(tile)
        result_rows.append(torch.cat(result_row, dim=-1))
    return torch.cat(result_rows, dim=-2)


def lightvae_decoder_tile_count(model: nn.Module, latent: torch.Tensor) -> int | None:
    """Return the release tile count for a latent, or None for the native path."""
    profile = getattr(model, "_h3_lite_decode_profile", None)
    if not isinstance(profile, H3LiteDecodeProfile):
        return None
    ratio = int(getattr(model, "vae_ratio", 1) or 1)
    height, width = int(latent.shape[-2]) * ratio, int(latent.shape[-1]) * ratio
    tile_h, tile_w, overlap_h, overlap_w = _lightvae_geometry(model, profile, int(latent.shape[2]), height, width)
    layout = profile.tile_layout or "native"
    rows, _, _ = _split_tiles_for_profile(model, height, tile_h, overlap_h, layout=layout)
    cols, _, _ = _split_tiles_for_profile(model, width, tile_w, overlap_w, layout=layout)
    return len(rows) * len(cols)


def _install_lightvae_tiled_decode(model: nn.Module, profile: H3LiteDecodeProfile | None) -> None:
    if profile is None or getattr(model, "_h3_lite_tiled_decode_installed", False):
        return
    if not callable(getattr(model, "tiled_decode", None)):
        # Profile-only unit doubles do not own the native decode entry point.
        return
    model._h3_lite_tiled_decode = model.tiled_decode

    def tiled_decode(z: torch.Tensor, *, is_image: bool = False) -> torch.Tensor:
        return _tiled_decode_lightvae(model, profile, z, is_image=is_image)

    model.tiled_decode = tiled_decode
    model._h3_lite_tiled_decode_installed = True


def apply_h3_lite_decode_profile(model: nn.Module, profile: H3LiteDecodeProfile | None) -> None:
    """Apply the 26-layer LightVAE release geometry to one native VAE.

    The official VAE config remains the fallback source. The release's
    anisotropic short/long axis and image fields are installed on this decoder
    instance only; the official path never calls this function.
    """
    updates = _decode_profile_updates(model, profile)
    if profile is None:
        return
    # Commit only after every field has passed validation.
    for name, value in updates.items():
        setattr(model, name, value)
    model._h3_lite_decode_profile = profile
    _install_lightvae_tiled_decode(model, profile)
