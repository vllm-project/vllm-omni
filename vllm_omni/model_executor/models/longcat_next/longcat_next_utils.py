"""Shared helpers for the LongCat-Next stages.

Token-id layout: text + special tokens occupy [0, 131125); visual codes
start at 150581 (8 levels x 16384 each); audio codes start at 131125 (8
levels, non-uniform sizes -- see modeling_longcat_next.py's offset table).
Per-level codes never ride the visible token stream -- they come from
talker_mtp via ``multimodal_output["codes"]`` (see
stage_input_processors/longcat_next.py). The visible stream only carries
one placeholder id per generation step, which is all ``infer_visual_grid``
below reads.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

# Special-token ids (tokenizer_config.json added_tokens_decoder)
AUDIO_START_TOKEN_ID = 131103  # <longcat_audio_start>
AUDIO_END_TOKEN_ID = 131104  # <longcat_audio_end>
AUDIO_PAD_TOKEN_ID = 131105  # <longcat_audio_pad>
IMG_START_TOKEN_ID = 131106  # <longcat_img_start>
IMG_END_TOKEN_ID = 131107  # <longcat_img_end>
IMG_PAD_TOKEN_ID = 131108  # <longcat_img_pad>
IMG_NEWLINE_TOKEN_ID = 131109  # <longcat_img_newline>
AUDIOTEXT_START_TOKEN_ID = 131120  # <longcat_audiotext_start>
AUDIOTEXT_END_TOKEN_ID = 131121  # <longcat_audiotext_end>
AUDIOTEXT_PAD_TOKEN_ID = 131122  # <longcat_audiotext_pad>
AUDIOGEN_START_TOKEN_ID = 131123  # <longcat_audiogen_start>
AUDIOGEN_END_TOKEN_ID = 131124  # <longcat_audiogen_end>

NUM_CODEBOOKS = 8

_WEIGHT_PATH_PLACEHOLDER = "WEIGHT_PATH_TO_LONGCAT_NEXT"


def resolve_single_request_additional_info(kwargs: dict[str, Any], decoder_name: str) -> dict[str, Any]:
    """Unwrap a decoder-stage forward()'s model_intermediate_buffer/
    runtime_additional_information kwarg down to the single request's info
    dict.

    Shared by the audio/image/multi decoder stages, all of which run with
    max_num_seqs: 1 -- warns (rather than silently dropping) if more than
    one request's info dict shows up in a batch.
    """
    model_intermediate_buffer = (
        kwargs.get("model_intermediate_buffer") or kwargs.get("runtime_additional_information") or {}
    )
    if isinstance(model_intermediate_buffer, dict):
        info_dicts = [info for info in model_intermediate_buffer.values() if isinstance(info, dict)]
    else:
        info_dicts = [info for info in model_intermediate_buffer if isinstance(info, dict)]
    if len(info_dicts) > 1:
        logger.warning(
            "%s got %d requests in one batch; only the first is decoded (max_num_seqs should be 1 for this stage).",
            decoder_name,
            len(info_dicts),
        )
    return info_dicts[0] if info_dicts else {}


def resolve_checkpoint_relative_path(configured_path: str, model_path: str) -> str:
    """Resolve config.json's WEIGHT_PATH_TO_LONGCAT_NEXT placeholder against
    the local model directory, instead of requiring users to edit the
    checkpoint."""
    if configured_path.startswith(_WEIGHT_PATH_PLACEHOLDER):
        relative = configured_path[len(_WEIGHT_PATH_PLACEHOLDER) :].lstrip("/")
        return os.path.join(model_path, relative)
    if os.path.isabs(configured_path):
        return configured_path
    return os.path.join(model_path, configured_path)


def _apply_transformers_qwen2_5_vl_compat() -> None:
    """The checkpoint's remote code imports ``Qwen2RMSNorm`` from
    ``transformers.models.qwen2_5_vl.modeling_qwen2_5_vl``, renamed to
    ``Qwen2_5_VLRMSNorm`` in newer transformers. Alias it back; safe to call
    repeatedly."""
    try:
        from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as _mod
    except ImportError:
        return
    if not hasattr(_mod, "Qwen2RMSNorm") and hasattr(_mod, "Qwen2_5_VLRMSNorm"):
        _mod.Qwen2RMSNorm = _mod.Qwen2_5_VLRMSNorm


def get_remote_attr(model_path: str, module_file: str, attr_name: str) -> Any:
    """Fetch a class or function from the checkpoint's remote code."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    _apply_transformers_qwen2_5_vl_compat()
    return get_class_from_dynamic_module(f"{module_file}.{attr_name}", model_path)


def load_remote_hf_config(model_path: str) -> Any:
    """Load the checkpoint's own LongcatNextConfig, bypassing AutoConfig.

    vllm-omni's registered config shim always wins over the remote one, but
    its plain-dict visual_config/audio_config break the checkpoint's remote
    code (which expects real config objects). Load the checkpoint's own
    config class directly instead.
    """
    config_cls = get_remote_attr(model_path, "configuration_longcat_next", "LongcatNextConfig")
    return config_cls.from_pretrained(model_path)


def load_weight_subtree(
    module: torch.nn.Module,
    model_path: str,
    prefix: str,
    *,
    dtype: torch.dtype | None = None,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Load only the ``prefix``-scoped tensors from a sharded checkpoint, so
    a decoder stage never materialises the full backbone in memory."""
    from safetensors import safe_open

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map: dict[str, str] = json.load(f)["weight_map"]

    dotted = prefix if prefix.endswith(".") else prefix + "."
    shard_to_keys: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        if key.startswith(dotted):
            shard_to_keys.setdefault(shard, []).append(key)

    if not shard_to_keys:
        raise ValueError(f"No weights under prefix '{prefix}' in {index_path}")

    state_dict: dict[str, torch.Tensor] = {}
    for shard, keys in shard_to_keys.items():
        shard_path = os.path.join(model_path, shard)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in keys:
                tensor = f.get_tensor(key)
                if dtype is not None and tensor.is_floating_point():
                    tensor = tensor.to(dtype)
                state_dict[key[len(dotted) :]] = tensor

    missing, unexpected = module.load_state_dict(state_dict, strict=strict)
    if missing:
        logger.warning("load_weight_subtree(%s): %d missing keys (e.g. %s)", prefix, len(missing), missing[:3])
    if unexpected:
        logger.warning("load_weight_subtree(%s): %d unexpected keys (e.g. %s)", prefix, len(unexpected), unexpected[:3])
    return list(missing), list(unexpected)


def infer_visual_grid(output_ids: Sequence[int]) -> tuple[int, int] | None:
    """Infer (token_h, token_w) from the newline structure of the first
    image segment. Counts visible IMG_PAD placeholders per row -- one per
    real-pixel step -- not the real per-level codes, which ride
    multimodal_output instead of this stream."""
    in_segment = False
    row_len = 0
    width: int | None = None
    height = 0
    for tid in output_ids:
        if tid == IMG_START_TOKEN_ID:
            in_segment = True
            row_len = 0
            width = None
            height = 0
        elif not in_segment:
            continue
        elif tid == IMG_NEWLINE_TOKEN_ID:
            if width is None:
                width = row_len
            height += 1
            row_len = 0
        elif tid == IMG_END_TOKEN_ID:
            if row_len and width is None:
                width = row_len
            if row_len and width:
                height += 1
            if width and height:
                return height, width
            return None
        else:
            row_len += 1
    return None
