# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared payload builder utilities for stage input processors.

This module provides common helpers to eliminate code duplication across
stage input processors (Qwen2.5, Qwen3, Qwen3-TTS, MiMo, GLM-TTS, Fish, CosyVoice).

Common patterns:
- Extract model output from pooling_output/multimodal_output
- Normalize token/code rows (filter invalid, strip boundaries, trim stop tokens)
- Build OmniPayloadStruct (embed, hidden_states, ids, meta, codes)
- Choose shape (async_chunk, full_payload, token_only)
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm_omni.data_entry_keys import (
    EmbeddingsStruct,
    HiddenStatesStruct,
    IdsStruct,
    MetaStruct,
    OmniPayloadStruct,
    to_dict,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Common Helper Functions
# ============================================================================


def ensure_list(x: Any) -> list[Any]:
    """Convert ConstantList / tensor-like to Python list."""
    if hasattr(x, "_x"):
        return list(x._x)
    if isinstance(x, list):
        return list(x)
    if isinstance(x, tuple):
        return list(x)
    if x is None:
        return []
    try:
        return list(x)
    except TypeError:
        return [x]


def layer_tensor(layers: dict[Any, Any], key: str) -> torch.Tensor | None:
    """Fetch layer tensor with tolerant key lookup (str/int)."""
    if not isinstance(layers, dict):
        return None
    key_int = int(key)
    val = layers.get(key_int)
    if val is None:
        val = layers.get(key)
    return val if isinstance(val, torch.Tensor) else None


def to_cpu_tensor(value: Any) -> torch.Tensor | None:
    """Convert value to CPU tensor if possible."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list):
        if not value:
            return None
        if isinstance(value[0], torch.Tensor):
            return value[0].detach().cpu()
    return None


def to_tensor_or_none(value: Any) -> torch.Tensor | None:
    """Convert value to tensor or return None."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
        return value[0].detach().cpu()
    return None


def strip_boundary_tokens(
    token_ids: list[int],
    start_token_id: int | None = None,
    pad_token_id: int | None = None,
    end_token_id: int | None = None,
) -> list[int]:
    """Strip boundary tokens (START/PAD/END) from token list."""
    tids = list(token_ids)

    if start_token_id is not None and tids and tids[0] == start_token_id:
        tids = tids[1:]

    if end_token_id is not None and tids and tids[-1] == end_token_id:
        tids = tids[:-1]

    if pad_token_id is not None:
        tids = [tid for tid in tids if tid != pad_token_id]

    return tids


def filter_invalid_tokens(
    token_ids: list[int],
    min_valid: int = 0,
    max_valid: int | None = None,
) -> list[int]:
    """Filter invalid tokens (negative, out-of-range) from token list."""
    filtered = []
    for tid in token_ids:
        if tid < min_valid:
            continue
        if max_valid is not None and tid >= max_valid:
            continue
        filtered.append(tid)
    return filtered


def count_trailing_placeholders(token_ids: list[int], placeholder: int = -1) -> int:
    """Count trailing placeholder tokens in token list."""
    count = 0
    while count < len(token_ids) and token_ids[-1 - count] == placeholder:
        count += 1
    return count


# ============================================================================
# TransitionSpec: Configuration for payload extraction/normalization
# ============================================================================


@dataclass
class TransitionSpec:
    """Configuration for stage transition payload building.

    Defines how to extract and normalize data from pooling_output/multimodal_output
    and build OmniPayloadStruct for different shapes (async_chunk, full_payload, token_only).
    """

    # Source key mappings (pooling_output keys -> target struct fields)
    embed_layer_keys: dict[str, str] = field(default_factory=dict)
    hidden_state_layer_keys: dict[str, str] = field(default_factory=dict)
    code_keys: dict[str, str] = field(default_factory=dict)

    # Normalization rules
    token_range: tuple[int, int] | None = None  # (min_valid, max_valid)
    codebook_size: int | None = None
    boundary_tokens: dict[str, int] = field(default_factory=dict)  # start, pad, end

    # Layer specifications
    embed_layer: str | None = None
    hidden_layer: str | None = None

    # Shape selection
    trim_stop_token: bool = False
    trim_trailing_placeholders: bool = False

    # Metadata extraction
    extract_speaker: bool = False
    extract_language: bool = False

    # Custom validators
    token_validator: Callable[[int], bool] | None = None
    code_validator: Callable[[torch.Tensor], torch.Tensor] | None = None


# ============================================================================
# StagePayloadBuilder: Builder for OmniPayloadStruct
# ============================================================================


class StagePayloadBuilder:
    """Builder for stage transition payloads.

    Provides methods to extract, normalize, and build OmniPayloadStruct
    for different shapes (async_chunk, full_payload, token_only).
    """

    def __init__(self, spec: TransitionSpec):
        self.spec = spec

    def extract_from_pooling_output(
        self,
        pooling_output: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract tensors from pooling_output based on spec."""
        extracted = {}

        # Extract embeddings
        if self.spec.embed_layer:
            layers = pooling_output.get("hidden_states", {}).get("layers", {})
            if isinstance(layers, dict):
                extracted["embed"] = layer_tensor(layers, self.spec.embed_layer)

        # Extract hidden states
        if self.spec.hidden_layer:
            layers = pooling_output.get("hidden_states", {}).get("layers", {})
            if isinstance(layers, dict):
                extracted["hidden"] = layer_tensor(layers, self.spec.hidden_layer)

        # Extract codes
        for source_key, target_key in self.spec.code_keys.items():
            val = pooling_output.get(source_key)
            if val is not None:
                extracted[target_key] = val

        # Extract additional fields
        for source_key, target_key in self.spec.embed_layer_keys.items():
            val = pooling_output.get(source_key)
            if val is not None:
                extracted[target_key] = val

        return extracted

    def normalize_tokens(
        self,
        token_ids: list[int],
    ) -> list[int]:
        """Normalize token IDs based on spec."""
        normalized = list(token_ids)

        # Apply custom validator if provided
        if self.spec.token_validator:
            normalized = [tid for tid in normalized if self.spec.token_validator(tid)]

        # Filter by range
        if self.spec.token_range:
            min_valid, max_valid = self.spec.token_range
            normalized = filter_invalid_tokens(normalized, min_valid, max_valid)

        # Strip boundary tokens
        if self.spec.boundary_tokens:
            normalized = strip_boundary_tokens(
                normalized,
                start_token_id=self.spec.boundary_tokens.get("start"),
                pad_token_id=self.spec.boundary_tokens.get("pad"),
                end_token_id=self.spec.boundary_tokens.get("end"),
            )

        return normalized

    def normalize_codes(
        self,
        codes: torch.Tensor,
        output_token_ids: list[int] | None = None,
    ) -> tuple[torch.Tensor, dict[str, int]]:
        """Normalize codec frames based on spec.

        Returns:
            (filtered_codes, stats_dict) where stats_dict contains:
            - raw_rows: original number of rows
            - aligned_rows: rows aligned with output_token_ids
            - valid_rows: rows after filtering
            - trailing_placeholder_count: count of trailing placeholders
        """
        if codes.ndim != 2 or codes.numel() == 0:
            return codes, {
                "raw_rows": int(codes.shape[0]) if codes.ndim > 0 else 0,
                "aligned_rows": 0,
                "valid_rows": 0,
                "trailing_placeholder_count": 0,
            }

        raw_rows = int(codes.shape[0])

        # Count trailing placeholders
        trailing_placeholder_count = 0
        if output_token_ids is not None and self.spec.trim_trailing_placeholders:
            trailing_placeholder_count = count_trailing_placeholders(output_token_ids)

        # Align with output token IDs if provided
        aligned_len = raw_rows
        if output_token_ids is not None:
            aligned_len = min(raw_rows, len(output_token_ids))

        # Apply custom code validator if provided
        if self.spec.code_validator:
            codes = self.spec.code_validator(codes)

        # Filter by codebook size
        if self.spec.codebook_size is not None:
            valid_mask = (codes.max(dim=1).values < self.spec.codebook_size) & (codes.min(dim=1).values >= 0)
            codes = codes[valid_mask]

        # Align with output token IDs
        if output_token_ids is not None and aligned_len > 0:
            codes = codes[-aligned_len:]

        valid_rows = int(codes.shape[0]) if codes.ndim > 0 else 0

        return codes, {
            "raw_rows": raw_rows,
            "aligned_rows": aligned_len,
            "valid_rows": valid_rows,
            "trailing_placeholder_count": trailing_placeholder_count,
        }

    def build_payload_struct(
        self,
        extracted: dict[str, Any],
        token_ids: dict[str, list[int]] | None = None,
        meta: dict[str, Any] | None = None,
        speaker: str | None = None,
        language: str | None = None,
    ) -> OmniPayloadStruct:
        """Build OmniPayloadStruct from extracted data."""
        kwargs = {}

        # Build EmbeddingsStruct
        embed_fields = {}
        if "embed" in extracted:
            embed_fields["prefill"] = extracted["embed"]
        for key in ["tts_bos", "tts_eos", "tts_pad", "speech_token", "speech_feat", "embedding"]:
            if key in extracted:
                embed_fields[key] = extracted[key]
        if embed_fields:
            kwargs["embed"] = EmbeddingsStruct(**embed_fields)

        # Build HiddenStatesStruct
        if "hidden" in extracted:
            kwargs["hidden_states"] = HiddenStatesStruct(output=extracted["hidden"])

        # Build IdsStruct
        if token_ids:
            ids_fields = {}
            if "all" in token_ids:
                ids_fields["all"] = token_ids["all"]
            if "prompt" in token_ids:
                ids_fields["prompt"] = token_ids["prompt"]
            if "output" in token_ids:
                ids_fields["output"] = token_ids["output"]
            if ids_fields:
                kwargs["ids"] = IdsStruct(**ids_fields)

        # Build MetaStruct
        if meta:
            kwargs["meta"] = MetaStruct(**meta)

        # Add speaker/language
        if speaker is not None:
            kwargs["speaker"] = speaker
        if language is not None:
            kwargs["language"] = language

        return OmniPayloadStruct(**kwargs)

    def build_full_payload(
        self,
        pooling_output: dict[str, Any],
        request: Any,
        is_finished: bool = True,
    ) -> dict[str, Any] | None:
        """Build full payload for connector path.

        Returns OmniPayload-shaped dict or None if extraction fails.
        """
        if not isinstance(pooling_output, dict):
            logger.warning(
                "StagePayloadBuilder.build_full_payload: pooling_output not a dict (type=%s)",
                type(pooling_output).__name__,
            )
            return None

        # Extract data
        extracted = self.extract_from_pooling_output(pooling_output)

        # Get token IDs from request
        prompt_token_ids = ensure_list(getattr(request, "prompt_token_ids", []) or [])
        output_token_ids = ensure_list(getattr(request, "output_token_ids", []) or [])
        all_token_ids = ensure_list(getattr(request, "all_token_ids", None) or [])
        if not all_token_ids:
            all_token_ids = list(prompt_token_ids) + list(output_token_ids)

        # Trim stop token if configured
        if self.spec.trim_stop_token and is_finished:
            if output_token_ids:
                output_token_ids = output_token_ids[:-1]
            if all_token_ids:
                all_token_ids = all_token_ids[:-1]

        # Build payload
        payload = to_dict(
            self.build_payload_struct(
                extracted,
                token_ids={
                    "all": all_token_ids,
                    "prompt": prompt_token_ids,
                    "output": output_token_ids,
                },
                meta={"finished": torch.tensor(is_finished, dtype=torch.bool)},
            )
        )

        return payload

    def build_token_only(
        self,
        source_outputs: list[Any],
        prompt: Any = None,
        prompt_length_fn: Callable[[dict[str, Any]], int] | None = None,
    ) -> list[Any]:
        """Build token-only placeholder for orchestrator scheduling.

        Returns list of OmniTokensPrompt sized to expected length.
        Actual payload comes via connector path.
        """
        from vllm_omni.inputs.data import OmniTokensPrompt

        token_only_inputs = []

        for i, source_output in enumerate(source_outputs):
            output = source_output.outputs[0]
            mm = getattr(output, "multimodal_output", None)

            # Calculate prompt length
            if prompt_length_fn and mm:
                prompt_len = prompt_length_fn(mm)
            else:
                prompt_len = 1  # Default minimal length

            # Extract small metadata if configured
            additional_info = None
            if self.spec.extract_speaker or self.spec.extract_language:
                additional_info = {}
                if self.spec.extract_speaker:
                    from vllm_omni.model_executor.stage_input_processors.tts_utils import (
                        extract_speaker_from_prompt,
                    )

                    speaker = extract_speaker_from_prompt(prompt, index=i)
                    if speaker is not None:
                        additional_info["speaker"] = speaker
                if self.spec.extract_language:
                    from vllm_omni.model_executor.stage_input_processors.tts_utils import (
                        extract_language_from_prompt,
                    )

                    language = extract_language_from_prompt(prompt, index=i)
                    if language is not None:
                        additional_info["language"] = language

            token_only_inputs.append(
                OmniTokensPrompt(
                    prompt_token_ids=[0] * prompt_len,
                    additional_information=additional_info if additional_info else None,
                    multi_modal_data=None,
                    mm_processor_kwargs=None,
                )
            )

        return token_only_inputs
