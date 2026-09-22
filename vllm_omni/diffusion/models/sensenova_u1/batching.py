# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-mode compatibility and dense conditioning batches for SenseNova."""

from __future__ import annotations

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


def merge_conditioning(caches: list[dict], counts: list[int], image_tokens: int) -> dict:
    """Pad each CFG prefix independently, preserving each image's 3-D positions.

    Dense storage is [sum(counts), Hkv, max_prefix, D]. Padding is masked out
    by a 2-D boolean key mask that compatible native backends can unpad into a
    varlen attention call; the following image tokens remain bidirectional. No
    AR context or scheduler row is consumed here.
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
        for layer_idx in range(len(prefixes[0].layers)):
            example = prefixes[0].layers[layer_idx].keys
            keys = example.new_zeros(total, example.shape[1], max_len, example.shape[3])
            values = torch.zeros_like(keys)
            start = 0
            for prefix, count, length in zip(prefixes, counts, lengths, strict=True):
                layer = prefix.layers[layer_idx]
                keys[start : start + count, :, :length].copy_(layer.keys)
                values[start : start + count, :, :length].copy_(layer.values)
                start += count
            kv.update(keys, values, layer_idx)
        mask = None
        if len(set(lengths)) > 1:
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
