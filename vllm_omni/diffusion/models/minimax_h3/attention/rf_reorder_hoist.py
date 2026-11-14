# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-local RF permutation before SP sharding, migrated from feature/h3-opt."""

from dataclasses import replace

import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.platforms import current_omni_platform

from .mindiesd_parallel import supports_mindiesd_ulysses


def relabel_branch(branch, permutation: torch.Tensor, inverse: torch.Tensor) -> None:
    """Move packed labels once; keep each modality's rows and update masks ordered."""
    used = permutation.numel()
    if inverse.numel() != used or branch.seq_len < used:
        raise ValueError("RF permutation/inverse length does not match the packed sequence")
    tail = torch.arange(used, branch.seq_len, device=permutation.device, dtype=torch.long)
    permutation = torch.cat((permutation, tail))
    inverse = torch.cat((inverse, tail))
    device = branch.device
    branch.img_pos = inverse.index_select(0, branch.img_pos.cpu())
    branch.audio_pos = inverse.index_select(0, branch.audio_pos.cpu())
    branch.img_pos_dev = branch.img_pos.to(device)
    branch.audio_pos_dev = branch.audio_pos.to(device)
    index, back = permutation.to(device), inverse.to(device)
    branch.text_pos_dev = back.index_select(0, branch.text_pos_dev)
    branch.token_tags_dev = branch.token_tags_dev.index_select(0, index)
    branch.img_position_ids_dev = branch.img_position_ids_dev.index_select(0, index)
    kw = branch.static_kwargs
    kw["img_position_ids"] = branch.img_position_ids_dev[None]
    kw["token_tags"] = branch.token_tags_dev
    kw["text_pos_info"] = {"position_ids": branch.text_pos_dev}
    kw["img_pos_info"] = {"position_ids": branch.img_pos_dev}
    kw["audio_pos_info"] = {"position_ids": branch.audio_pos_dev}
    kw["img_pos_for_infer_output_info"] = {"position_ids": branch.img_pos_dev}


def maybe_hoist_branch(branch, model) -> bool:
    layout = branch.static_kwargs["video_token_layout"]
    if layout.pre_rearranged:
        return True
    blocks = getattr(model, "blocks", ())
    layers = [block.attn.attention for block in blocks if hasattr(block, "attn")]
    if not layers:
        return False
    records = [layer.effective_attention() for layer in layers]
    configs = [
        impl.rainfusion.parallel if backend.get_name() == "RAINFUSION_ATTN" else None for impl, backend, _ in records
    ]
    if not any(config is not None and config.reorder_hoist for config in configs):
        return False
    if getattr(getattr(model, "od_config", None), "kv_offload_config", None) is not None:
        raise ValueError("RF reorder hoist cannot relabel a reference-KV reuse request")
    if not current_omni_platform.is_npu() or any(config is None or not config.reorder_hoist for config in configs):
        raise ValueError("RF reorder hoist requires MindIE parallel on every main-DiT attention layer")
    for layer, (impl, _, _) in zip(layers, records, strict=True):
        strategy = layer._get_active_parallel_strategy()
        if (
            not supports_mindiesd_ulysses(strategy)
            or layer._has_custom_attention
            or layer.is_paged_kv_active()
            or impl.rainfusion.precision != "bf16"
            or impl.rainfusion.sparse_type != "rf_v3"
            or layer._with_kv_cache_dtype(AttentionMetadata()).extra.get("kv_cache_dtype", "bf16")
            not in ("float", "bf16")
            or getattr(layer, "_schedule_candidates", {})
        ):
            raise ValueError("RF reorder hoist requires BF16 pure strict Ulysses without schedules or cached KV")
    metadata = AttentionMetadata(video_layout=layout, extra={"max_seqlen_q": branch.used_len})
    plan = records[0][0]._resolve_plan(metadata)
    if plan is None:
        return False
    from mindiesd import build_rf_attention_plan

    spans = plan.video_spans or [{"start": plan.prefix_len, "latent_shape": plan.latent_shape}]
    sdk_plan = build_rf_attention_plan(
        video_spans=spans,
        sequence_len=branch.used_len,
        pool_size=128,
        input_layout="BSND",
        device="cpu",
    )
    relabel_branch(branch, sdk_plan.permutation, sdk_plan.inverse)
    branch.static_kwargs["video_token_layout"] = replace(layout, pre_rearranged=True)
    branch.static_kwargs["packed_seq_params"].pop("vsa_prefix_segments", None)
    return True
