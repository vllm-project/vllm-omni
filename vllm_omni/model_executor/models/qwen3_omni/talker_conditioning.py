# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fixed Talker conditioning metadata; no tensors, hashing of weights or owners.

This is an encoding/validation boundary, not a media provenance verifier.
Processed-media records must be minted by the trusted processor path describing
the actual payload. A caller UUID or an arbitrary embedding is not such proof.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict
from hashlib import sha256

import regex as re

from vllm_omni.inputs.processed_media import ProcessedMediaIdentity as TalkerProcessedMediaIdentity

from .talker_input_plan import TalkerPrefillPart


def _token_ids(values: Sequence[int], name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or any(type(value) is not int or value < 0 for value in values):
        raise ValueError(f"{name} must contain non-negative integer token IDs")
    return tuple(values)


def _source_conditioning(
    *,
    prompt_ids: Sequence[int],
    sequence_ids: Sequence[int],
    row_plan: Sequence[TalkerPrefillPart],
    embedding_rows: int,
    hidden_rows: int,
    text_controls: tuple[int, int, int],
    media_token_ids: tuple[int, int, int],
    processed_media: Sequence[TalkerProcessedMediaIdentity],
) -> dict:
    prompt = _token_ids(prompt_ids, "source prompt")
    sequence = _token_ids(sequence_ids, "source sequence")
    if sequence[: len(prompt)] != prompt:
        raise ValueError("Thinker sequence must begin with the complete source prompt")
    for count in (embedding_rows, hidden_rows):
        if type(count) is not int or count < 0 or count > len(sequence):
            raise ValueError("conditioning row count is outside the source sequence")
    for controls, length in ((text_controls, 3), (media_token_ids, 3)):
        if len(_token_ids(controls, "controls")) != length:
            raise ValueError("conditioning control layout has the wrong length")
    rows = []
    for part in row_plan:
        if not isinstance(part, TalkerPrefillPart) or part.kind not in {"user", "assistant"}:
            raise ValueError("conditioning requires the shared Talker row plan")
        available_rows = min(embedding_rows, hidden_rows) if part.kind == "user" else embedding_rows
        if type(part.start) is not int or type(part.end) is not int or not 0 <= part.start < part.end <= available_rows:
            raise ValueError("conditioning row plan is outside the source sequence")
        if part.kind == "assistant" and part.end - part.start < 3:
            raise ValueError("conditioning assistant requires three header rows")
        rows.append((part.kind, part.start, part.end))
    if not rows:
        raise ValueError("conditioning requires at least one retained Talker part")

    if len(set(media_token_ids)) != 3:
        raise ValueError("conditioning media token IDs must distinguish the three modalities")
    token_by_modality = dict(zip(("image", "audio", "video"), media_token_ids, strict=True))
    covered: set[int] = set()
    media = []
    for item in processed_media:
        if not isinstance(item, TalkerProcessedMediaIdentity):
            raise ValueError("raw media labels are not processed-media identity records")
        if item.modality not in token_by_modality:
            raise ValueError("unsupported conditioning media modality")
        if any(
            not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in (item.content_digest, item.preprocessing_digest)
        ):
            raise ValueError("processed-media identity requires content and preprocessing digests")
        if (
            type(item.offset) is not int
            or type(item.length) is not int
            or item.offset < 0
            or item.length <= 0
            or item.offset + item.length > len(prompt)
        ):
            raise ValueError("processed-media position is outside the source prompt")
        mask = item.is_embed
        if mask is not None and (
            not isinstance(mask, tuple) or len(mask) != item.length or any(type(value) is not bool for value in mask)
        ):
            raise ValueError("processed-media embedding mask must align with its position")
        positions = {item.offset + index for index in range(item.length) if mask is None or mask[index]}
        if not positions or any(prompt[index] != token_by_modality[item.modality] for index in positions):
            raise ValueError("processed-media positions do not match the source modality tokens")
        if covered.intersection(positions):
            raise ValueError("multiple processed-media records claim the same embedding row")
        covered.update(positions)
        media.append(asdict(item))
    expected = {index for index, token in enumerate(prompt) if token in media_token_ids}
    if expected != covered:
        raise ValueError("source media tokens lack complete processed-media provenance")
    media.sort(key=lambda item: (item["offset"], item["modality"]))
    return {
        "source_prompt": prompt,
        "source_sequence": sequence,
        "embedding_rows": embedding_rows,
        "hidden_rows": hidden_rows,
        "row_plan": rows,
        "text_controls": text_controls,
        "media_token_ids": media_token_ids,
        "processed_media": media,
    }


def conditioning_metadata_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _namespace(value: str | None, name: str, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str) or not value:
        raise ValueError(f"conditioning namespace {name} must be non-empty")


def thinker_conditioning_digest(
    *,
    prompt_ids: Sequence[int],
    sequence_ids: Sequence[int],
    row_plan: Sequence[TalkerPrefillPart],
    embedding_rows: int,
    hidden_rows: int,
    text_controls: tuple[int, int, int],
    media_token_ids: tuple[int, int, int],
    source_model: str,
    source_adapter: str | None,
    processed_media: Sequence[TalkerProcessedMediaIdentity],
) -> str:
    """Fix source-owned dependencies without guessing the receiving model."""
    _namespace(source_model, "source_model")
    _namespace(source_adapter, "source_adapter", optional=True)
    return conditioning_metadata_digest(
        {
            "schema": "qwen3-omni.thinker-conditioning.v1",
            **_source_conditioning(
                prompt_ids=prompt_ids,
                sequence_ids=sequence_ids,
                row_plan=row_plan,
                embedding_rows=embedding_rows,
                hidden_rows=hidden_rows,
                text_controls=text_controls,
                media_token_ids=media_token_ids,
                processed_media=processed_media,
            ),
            "source_model": source_model,
            "source_adapter": source_adapter,
        }
    )


def complete_talker_conditioning_digest(
    *,
    source_digest: str,
    talker_model: str,
    projection: str,
    talker_adapter: str | None,
    text_controls: tuple[int, int, int],
    codec_controls: tuple[int, int, int, int, int, int],
) -> str:
    """Add the consumer's actual static namespace and effective controls.

    The source digest binds complete source/layout/media metadata. This second
    fixed-input digest deliberately contains no caller salt or request owner.
    """
    if not isinstance(source_digest, str) or re.fullmatch(r"[0-9a-f]{64}", source_digest) is None:
        raise ValueError("source conditioning requires a SHA-256 digest")
    _namespace(talker_model, "talker_model")
    _namespace(projection, "projection")
    _namespace(talker_adapter, "talker_adapter", optional=True)
    for values, length in ((text_controls, 3), (codec_controls, 6)):
        if len(_token_ids(values, "controls")) != length:
            raise ValueError("conditioning control layout has the wrong length")
    return conditioning_metadata_digest(
        {
            "schema": "qwen3-omni.talker-from-source.v1",
            "source_digest": source_digest,
            "talker_model": talker_model,
            "projection": projection,
            "talker_adapter": talker_adapter,
            "text_controls": text_controls,
            "codec_controls": codec_controls,
        }
    )
