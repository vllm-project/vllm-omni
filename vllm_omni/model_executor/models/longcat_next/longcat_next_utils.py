"""Shared helpers for the LongCat-Next stages.

Token-id layout (verified against the checkpoint's config.json and
tokenizer_config.json):

- Text + special tokens occupy [0, 131125).
- Audio codes occupy 8 codebook levels of 16384 entries each, with level
  offsets ``cumsum([131125] + [16384] * 7)``.
- Visual codes likewise start at 150581 with the same per-level layout.
- Level-0 code ``16384`` (== codebook size) is the audio chunk-end marker.

In the flattened thinker output stream, each multimodal position contributes
``NUM_CODEBOOKS`` consecutive ids (one per level, each carrying its level
offset), delimited by the start/end marker tokens below.
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
CODEBOOK_SIZE = 16384  # visual codebook only -- every level is 16384-wide.

VISUAL_OFFSET = 150581
AUDIO_OFFSET = 131125

# Audio codebook sizes are NOT uniform across levels (unlike visual's flat
# 16384), per config.json's audio_config.vq_config.codebook_sizes.
AUDIO_CODEBOOK_SIZES = [8192, 4096, 2048, 1024, 1024, 1024, 1024, 1024]


def _cumulative_offsets(base: int, codebook_sizes: Sequence[int]) -> list[int]:
    """cumsum([base] + codebook_sizes[:-1]), mirroring the model's
    visual_offset_vals/audio_offset_vals buffers (modeling_longcat_next.py)."""
    offsets = [base]
    for size in codebook_sizes[:-1]:
        offsets.append(offsets[-1] + size)
    return offsets

# Per-level cumulative offsets, mirroring the model's visual/audio_offset_vals
# buffers (cumsum of [base] + codebook_sizes[:-1]).
VISUAL_LEVEL_OFFSETS = [VISUAL_OFFSET + level * CODEBOOK_SIZE for level in range(NUM_CODEBOOKS)]
AUDIO_LEVEL_OFFSETS = [AUDIO_OFFSET + level * CODEBOOK_SIZE for level in range(NUM_CODEBOOKS)]

_WEIGHT_PATH_PLACEHOLDER = "WEIGHT_PATH_TO_LONGCAT_NEXT"


def resolve_checkpoint_relative_path(configured_path: str, model_path: str) -> str:
    """Resolve the checkpoint's WEIGHT_PATH_TO_LONGCAT_NEXT placeholder.

    The released config.json stores auxiliary weight paths as
    ``WEIGHT_PATH_TO_LONGCAT_NEXT/<subpath>``; resolve them against the
    local model directory instead of requiring users to edit the checkpoint.
    """
    if configured_path.startswith(_WEIGHT_PATH_PLACEHOLDER):
        relative = configured_path[len(_WEIGHT_PATH_PLACEHOLDER):].lstrip("/")
        return os.path.join(model_path, relative)
    if os.path.isabs(configured_path):
        return configured_path
    return os.path.join(model_path, configured_path)


def _apply_transformers_qwen2_5_vl_compat() -> None:
    """Alias renamed transformers internals the checkpoint's remote code expects.

    The checkpoint's ``modular_longcat_next_visual.py`` was written against an
    older transformers release and imports ``Qwen2RMSNorm`` from
    ``transformers.models.qwen2_5_vl.modeling_qwen2_5_vl``. Newer transformers
    (5.x) renamed that class to ``Qwen2_5_VLRMSNorm``. Patching the installed
    transformers module (not the checkpoint) so the remote code's import
    resolves; safe to call repeatedly.
    """
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
    """Load the checkpoint's own LongcatNextConfig (with nested sub-configs).

    Bypasses ``AutoConfig``: vllm-omni's registered ``longcat_next`` config
    shim (``transformers_utils/configs/longcat_next.py``) always wins over
    the remote one once its model_type is registered, even with
    ``trust_remote_code=True``, and its plain-dict ``visual_config``/
    ``audio_config`` break the checkpoint's remote code (e.g.
    ``VisualEncoder.__init__`` expects a real config object). Loading the
    checkpoint's own config class directly keeps those as its own
    ``LongcatNextVisualConfig``/``LongcatNextAudioConfig`` objects.
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
    """Load only the ``prefix``-scoped tensors from a sharded checkpoint.

    Reads model.safetensors.index.json to locate the shards holding
    ``prefix.*`` keys and pulls just those tensors via safe_open, so a decoder
    stage never materialises the 74B backbone shards in memory.
    """
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
                state_dict[key[len(dotted):]] = tensor

    missing, unexpected = module.load_state_dict(state_dict, strict=strict)
    if missing:
        logger.warning("load_weight_subtree(%s): %d missing keys (e.g. %s)", prefix, len(missing), missing[:3])
    if unexpected:
        logger.warning(
            "load_weight_subtree(%s): %d unexpected keys (e.g. %s)", prefix, len(unexpected), unexpected[:3]
        )
    return list(missing), list(unexpected)


def _extract_code_segments(
    output_ids: Sequence[int],
    start_id: int,
    end_id: int,
    level_offsets: Sequence[int],
    skip_ids: frozenset[int],
) -> tuple[list[list[int]], list[list[int]]]:
    """Split a flat id stream into per-segment [n, NUM_CODEBOOKS] code grids.

    Returns (codes, raw_rows): ``codes`` holds de-offset codebook indices,
    ``raw_rows`` the original ids (used by callers that need markers back).
    Ids inside a segment that are not codes (e.g. newline markers) must be
    listed in ``skip_ids``.
    """
    segments: list[list[int]] = []
    current: list[int] | None = None
    for tid in output_ids:
        if tid == start_id:
            current = []
        elif tid == end_id:
            if current is not None:
                segments.append(current)
            current = None
        elif current is not None and tid not in skip_ids:
            current.append(tid)

    codes: list[list[int]] = []
    raw_rows: list[list[int]] = []
    for segment in segments:
        usable = len(segment) - len(segment) % NUM_CODEBOOKS
        if usable != len(segment):
            logger.warning(
                "Multimodal segment length %d is not a multiple of %d codebook levels; truncating",
                len(segment),
                NUM_CODEBOOKS,
            )
        for row_start in range(0, usable, NUM_CODEBOOKS):
            row = segment[row_start:row_start + NUM_CODEBOOKS]
            raw_rows.append(list(row))
            codes.append([tid - level_offsets[level] for level, tid in enumerate(row)])
    return codes, raw_rows


def extract_visual_codes(output_ids: Sequence[int]) -> list[list[int]]:
    """Pull visual codebook indices out of a flat thinker output stream.

    Visual code rows live between <longcat_img_start> and <longcat_img_end>;
    <longcat_img_newline> markers are structural and skipped.
    """
    codes, _ = _extract_code_segments(
        output_ids,
        IMG_START_TOKEN_ID,
        IMG_END_TOKEN_ID,
        VISUAL_LEVEL_OFFSETS,
        frozenset({IMG_NEWLINE_TOKEN_ID, IMG_PAD_TOKEN_ID}),
    )
    return codes


def infer_visual_grid(output_ids: Sequence[int]) -> tuple[int, int] | None:
    """Infer (token_h, token_w) from newline structure of the first image segment.

    Counts VISIBLE placeholder tokens per row directly (row_len), with no
    NUM_CODEBOOKS division: each real-pixel generation step contributes
    exactly one IMG_PAD_TOKEN_ID to the visible stream (forced in
    compute_logits, modeling_longcat_next.py), not NUM_CODEBOOKS ids --
    the real per-level codes ride multimodal_output, not this stream (see
    extract_visual_codes's docstring for the same point). An earlier
    version of this function divided by NUM_CODEBOOKS, matching
    extract_visual_codes's now-corrected wrong assumption.
    """
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


def extract_audio_codes(output_ids: Sequence[int]) -> list[list[int]]:
    """Pull audio codebook indices out of a flat thinker output stream.

    Audio-generation rows live between <longcat_audiogen_start> and
    <longcat_audiogen_end>. Level-0 code 16384 (chunk-end) rows are kept —
    the audio decoder uses them for chunking, mirroring lazy_decode_and_save.
    """
    codes, _ = _extract_code_segments(
        output_ids,
        AUDIOGEN_START_TOKEN_ID,
        AUDIOGEN_END_TOKEN_ID,
        AUDIO_LEVEL_OFFSETS,
        frozenset({AUDIO_PAD_TOKEN_ID}),
    )
    return codes
