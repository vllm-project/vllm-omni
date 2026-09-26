# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-mode compatibility and dense conditioning batches for SenseNova."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.cache_utils import DynamicCache


def image_count(sampling) -> int:
    """Legacy batch_size aliases the standard output count's default of one.

    Sampling params cannot distinguish an omitted count from an explicit one.
    A non-default standard count must agree with the legacy alias.
    """
    count = sampling.num_outputs_per_prompt
    legacy = (sampling.extra_args or {}).get("batch_size")
    for name, value in (("num_outputs_per_prompt", count), ("batch_size", legacy)):
        if name == "batch_size" and value is None:
            continue
        if type(value) is not int or value <= 0:
            raise ValueError(f"SenseNova {name} must be a positive integer, got {value!r}")
    if legacy is not None:
        if count != 1 and count != legacy:
            raise ValueError("SenseNova batch_size conflicts with num_outputs_per_prompt")
        count = legacy
    return count


def denoise_options(sampling) -> dict:
    extra = sampling.extra_args or {}
    return dict(
        cfg_scale=float(extra.get("cfg_scale", 4.0)),
        img_cfg_scale=float(extra.get("img_cfg_scale", 1.0)),
        cfg_norm=str(extra.get("cfg_norm", "none")),
        timestep_shift=float(extra.get("timestep_shift", 3.0)),
        cfg_interval=tuple(float(t) for t in extra.get("cfg_interval", (0.0, 1.0))),
        t_eps=float(extra.get("t_eps", 0.02)),
    )


def request_mode(prompt) -> str:
    if isinstance(prompt, str):
        return "t2i"
    if "text" in prompt.get("modalities", []):
        return "text"
    mm = prompt.get("multi_modal_data") or {}
    return "it2i" if mm.get("image") or mm.get("img2img") else "t2i"


def request_condition_key(prompt, sampling) -> tuple:
    # Prompt, seed and think are private to each request. Both think modes
    # have the same denoise contract after their prefix has been prepared.
    return (request_mode(prompt), image_count(sampling), *denoise_options(sampling).values())


@dataclass(frozen=True)
class _PackedVarlenPlan:
    offsets: list[int]
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    image_positions: torch.Tensor
    max_seqlen_k: int

    def stamp(self, layer) -> None:
        layer.sensenova_packed_varlen = True
        layer.sensenova_cu_seqlens_q = self.cu_seqlens_q
        layer.sensenova_cu_seqlens_k = self.cu_seqlens_k
        layer.sensenova_image_positions = self.image_positions
        layer.sensenova_max_seqlen_k = self.max_seqlen_k


def _packed_varlen_plan(prefix_lengths: list[int], image_tokens: int, device: torch.device) -> _PackedVarlenPlan:
    """Build the shared [prefix, image] layout for requests and CFG branches."""
    sample_lengths = [length + image_tokens for length in prefix_lengths]
    offsets = [0]
    for length in sample_lengths:
        offsets.append(offsets[-1] + length)
    return _PackedVarlenPlan(
        offsets=offsets,
        cu_seqlens_q=torch.arange(
            0, (len(prefix_lengths) + 1) * image_tokens, image_tokens, dtype=torch.int32, device=device
        ),
        cu_seqlens_k=torch.tensor(offsets, dtype=torch.int32, device=device),
        image_positions=torch.cat(
            [
                torch.arange(start + length, end, device=device)
                for start, end, length in zip(offsets[:-1], offsets[1:], prefix_lengths, strict=True)
            ]
        ),
        max_seqlen_k=max(sample_lengths),
    )


def merge_conditioning(
    caches: list[dict],
    counts: list[int],
    image_tokens: int,
    *,
    packed_varlen: bool = False,
) -> dict:
    """Merge each CFG prefix, preserving every image's 3-D positions.

    Dense storage is [sum(counts), Hkv, max_prefix, D], with padding masked
    by a 2-D boolean key mask. On a packed-varlen backend, ragged requests
    instead store each [prefix, image] sequence contiguously. No AR context or
    scheduler row is consumed here.
    """
    merged = {}
    for branch in ("cond", "uncond", "img_cond"):
        if branch not in caches[0]:
            continue
        prefixes = [cache[branch] for cache in caches]
        lengths = [prefix.get_seq_length() for prefix in prefixes]
        max_len = max(lengths)
        total = sum(counts)
        kv = DynamicCache()
        use_packed = packed_varlen and len(set(lengths)) > 1
        plan = None
        if use_packed:
            plan = _packed_varlen_plan(
                [length for count, length in zip(counts, lengths, strict=True) for _ in range(count)],
                image_tokens,
                prefixes[0].layers[0].keys.device,
            )
        for layer_idx in range(len(prefixes[0].layers)):
            example = prefixes[0].layers[layer_idx].keys
            if plan is not None:
                keys = example.new_zeros(1, example.shape[1], plan.offsets[-1], example.shape[3])
            else:
                keys = example.new_zeros(total, example.shape[1], max_len, example.shape[3])
            values = torch.zeros_like(keys)
            batch_start = packed_start = 0
            for prefix, count, length in zip(prefixes, counts, lengths, strict=True):
                layer = prefix.layers[layer_idx]
                if plan is not None:
                    for row in range(count):
                        keys[0, :, packed_start : packed_start + length].copy_(layer.keys[row])
                        values[0, :, packed_start : packed_start + length].copy_(layer.values[row])
                        packed_start += length + image_tokens
                else:
                    keys[batch_start : batch_start + count, :, :length].copy_(layer.keys)
                    values[batch_start : batch_start + count, :, :length].copy_(layer.values)
                batch_start += count
            kv.update(keys, values, layer_idx)
            if plan is not None:
                plan.stamp(kv.layers[layer_idx])
        mask = None
        if len(set(lengths)) > 1 and not use_packed:
            mask = torch.ones(total, max_len + image_tokens, dtype=torch.bool, device=keys.device)
            start = 0
            for count, length in zip(counts, lengths, strict=True):
                mask[start : start + count, length:max_len] = False
                start += count
        merged[branch] = kv
        merged[f"mask_{branch}"] = {"full_attention": mask}
        merged[f"idx_{branch}"] = torch.cat(
            [
                cache[f"idx_{branch}"].unsqueeze(1).expand(-1, count, -1)
                for cache, count in zip(caches, counts, strict=True)
            ],
            dim=1,
        )
    return merged


def merge_cfg_branches(
    branch_caches: list[DynamicCache],
    branch_indexes: list[torch.Tensor],
    image_tokens: int,
    *,
    packed_varlen: bool = False,
) -> tuple[DynamicCache, torch.Tensor, dict[str, torch.Tensor | None]]:
    """Fuse CFG branches, including request-packed and dense prefix caches.

    A packed source includes reserved image slots. Copy only its prefix tokens;
    the fused cache gets fresh image slots in branch-major order.
    """
    if not branch_caches:
        raise ValueError("SenseNova CFG fusion requires at least one branch")
    # Each entry holds (is_packed, prefix_length_per_image, packed_k_offsets).
    layouts: list[tuple[bool, list[int], list[int] | None]] = []
    batch_size = None
    for cache in branch_caches:
        layer = cache.layers[0]
        is_packed = bool(getattr(layer, "sensenova_packed_varlen", False))
        if is_packed:
            if not packed_varlen:
                raise ValueError("SenseNova packed CFG source requires a packed-varlen attention backend")
            source_offsets = layer.sensenova_cu_seqlens_k.tolist()
            lengths = [end - start - image_tokens for start, end in zip(source_offsets[:-1], source_offsets[1:])]
            if any(length < 0 for length in lengths) or source_offsets[-1] != layer.keys.shape[2]:
                raise ValueError("SenseNova packed CFG source has invalid K/V offsets")
            rows = len(lengths)
            if layer.sensenova_cu_seqlens_q.numel() != rows + 1:
                raise ValueError("SenseNova packed CFG source has mismatched Q/K rows")
        else:
            rows = layer.keys.shape[0]
            lengths = [cache.get_seq_length()] * rows
            source_offsets = None
        if batch_size is None:
            batch_size = rows
        elif rows != batch_size:
            raise ValueError("SenseNova CFG branch cache batch sizes must match")
        layouts.append((is_packed, lengths, source_offsets))

    assert batch_size is not None
    prefix_lengths = [length for _, lengths, _ in layouts for length in lengths]
    max_len = max(prefix_lengths)
    use_packed = packed_varlen and (any(is_packed for is_packed, _, _ in layouts) or len(set(prefix_lengths)) > 1)
    plan = None
    if use_packed:
        plan = _packed_varlen_plan(prefix_lengths, image_tokens, branch_caches[0].layers[0].keys.device)
    merged = DynamicCache()
    for layer_idx in range(len(branch_caches[0].layers)):
        example = branch_caches[0].layers[layer_idx].keys
        if plan is not None:
            keys = example.new_zeros(1, example.shape[1], plan.offsets[-1], example.shape[3])
        else:
            keys = example.new_zeros(len(branch_caches) * batch_size, example.shape[1], max_len, example.shape[3])
        values = torch.zeros_like(keys)
        packed_start = 0
        for branch_idx, (cache, layout) in enumerate(zip(branch_caches, layouts, strict=True)):
            source_packed, row_lengths, source_offsets = layout
            if plan is not None:
                for row, length in enumerate(row_lengths):
                    if source_packed:
                        assert source_offsets is not None
                        source_start = source_offsets[row]
                        source_key = cache.layers[layer_idx].keys[0, :, source_start : source_start + length]
                        source_value = cache.layers[layer_idx].values[0, :, source_start : source_start + length]
                    else:
                        source_key = cache.layers[layer_idx].keys[row, :, :length]
                        source_value = cache.layers[layer_idx].values[row, :, :length]
                    keys[0, :, packed_start : packed_start + length].copy_(source_key)
                    values[0, :, packed_start : packed_start + length].copy_(source_value)
                    packed_start += length + image_tokens
            else:
                rows = slice(branch_idx * batch_size, (branch_idx + 1) * batch_size)
                keys[rows, :, : row_lengths[0]].copy_(cache.layers[layer_idx].keys)
                values[rows, :, : row_lengths[0]].copy_(cache.layers[layer_idx].values)
        merged.update(keys, values, layer_idx)
        if plan is not None:
            plan.stamp(merged.layers[layer_idx])

    mask = None
    if len(set(prefix_lengths)) > 1 and not use_packed:
        mask = torch.ones(
            len(branch_caches) * batch_size,
            max_len + image_tokens,
            dtype=torch.bool,
            device=branch_caches[0].layers[0].keys.device,
        )
        for row, length in enumerate(prefix_lengths):
            mask[row, length:max_len] = False

    normalized_indexes = [
        idx.unsqueeze(1).expand(-1, batch_size, -1) if idx.ndim == 2 else idx for idx in branch_indexes
    ]
    return merged, torch.cat(normalized_indexes, dim=1), {"full_attention": mask}
