# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 SandAI. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Native MAGI-2 prompt conditioning.

Adapted from SandAI's MAGI-2 Preview inference implementation. The model
loading remains in-tree and uses the Transformers Qwen3.5 implementation; no
MAGI-2 reference checkout is imported at runtime.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from tokenizers import Regex, pre_tokenizers
from transformers import AutoTokenizer


def _strip_empty(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            key: value
            for key, value in ((key, _strip_empty(value)) for key, value in obj.items())
            if value is not None and value != [] and value != {}
        }
    if isinstance(obj, list):
        return [_strip_empty(value) for value in obj if value is not None]
    return obj


def _one_line(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return " ".join(value.split())


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _pick(values: dict[str, Any], keys: Iterable[str]) -> list[tuple[str, Any]]:
    return [(key, values[key]) for key in keys if key in values and values[key] not in (None, [], {})]


def json_to_compact_markdown(raw_json: str | dict[str, Any]) -> str:
    """Convert MAGI's structured prompt JSON into its training-time text form."""

    if isinstance(raw_json, str):
        obj = json.loads(raw_json.strip())
    else:
        obj = raw_json

    obj = _strip_empty(obj)
    if not isinstance(obj, dict) or "global_layer" not in obj:
        return raw_json if isinstance(raw_json, str) else json.dumps(raw_json, ensure_ascii=False)

    global_layer = _as_dict(obj.get("global_layer"))
    dynamic_layer = _as_dict(obj.get("dynamic_layer"))
    reference_layer = obj.get("reference_layer", [])

    lines: list[str] = []
    context = _one_line(global_layer.get("context"))
    description = _one_line(global_layer.get("description"))
    if context or description:
        lines.append(f"context: {context}" if context else "context")
        if description:
            lines.append(description)

    aesthetics = _as_dict(global_layer.get("aesthetics"))
    aesthetic_parts = []
    for label, key in (("style", "style"), ("mood", "mood_atmosphere"), ("color", "color_scheme")):
        value = _one_line(aesthetics.get(key))
        if value:
            aesthetic_parts.append(f"{label}={value}")
    if aesthetic_parts:
        lines.append("aesthetics: " + "; ".join(aesthetic_parts))

    audio = _as_dict(global_layer.get("audio_baseline"))
    dialogue = _as_dict(audio.get("dialogue"))
    audio_parts = []
    language = _one_line(dialogue.get("language"))
    speakers = _as_list(dialogue.get("speaker_tags"))
    ambience = _one_line(audio.get("ambience"))
    if language:
        audio_parts.append(f"language={language}")
    if speakers:
        audio_parts.append("speakers=" + ",".join(map(_one_line, speakers)))
    if ambience:
        audio_parts.append(f"ambience={ambience}")
    if audio_parts:
        lines.append("audio: " + "; ".join(audio_parts))

    subjects = global_layer.get("alive_subjects_static")
    if isinstance(subjects, list) and subjects:
        lines.append("subjects:")
        for subject in subjects:
            if not isinstance(subject, dict):
                continue
            sid = _one_line(subject.get("subject_id"))
            desc = _one_line(subject.get("description"))
            position = _one_line(subject.get("position"))
            orientation = _one_line(subject.get("orientation"))
            attrs = []
            for key, value in _pick(
                _as_dict(subject.get("visual_attributes")),
                ["gender", "age_group", "ethnicity", "clothing", "appearance_details"],
            ):
                value = _one_line(value)
                if value:
                    attrs.append(f"{key}={value}")
            row = " - " + " | ".join(part for part in (sid, desc) if part)
            extra = "; ".join(part for part in (position, orientation) if part)
            if extra:
                row += f" ({extra})"
            if attrs:
                row += " :: " + "; ".join(attrs)
            lines.append(row)

    objects = global_layer.get("objects_static")
    if isinstance(objects, list) and objects:
        lines.append("objects:")
        for obj_item in objects:
            if not isinstance(obj_item, dict):
                continue
            oid = _one_line(obj_item.get("object_id"))
            desc = _one_line(obj_item.get("description"))
            shape = _one_line(obj_item.get("shape_and_color"))
            position = _one_line(obj_item.get("position"))
            row = " - " + " | ".join(part for part in (oid, desc) if part)
            details = "; ".join(part for part in (shape, position) if part)
            if details:
                row += " :: " + details
            lines.append(row)

    segments = dynamic_layer.get("timeline_segments")
    if isinstance(segments, list) and segments:
        lines.append("timeline:")
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            basic = _as_dict(segment.get("segment_basic_info"))
            timestamp = _one_line(basic.get("timestamp_range"))
            desc = _one_line(basic.get("segment_description"))
            head = f" - {timestamp}" if timestamp else " -"
            if desc:
                head += f" {desc}"
            lines.append(head.rstrip())

            segment_audio = _as_dict(segment.get("audio"))
            for dialogue_line in segment_audio.get("dialogue_lines", []) or []:
                if not isinstance(dialogue_line, dict):
                    continue
                speaker = _one_line(dialogue_line.get("speaker"))
                text = _one_line(dialogue_line.get("text"))
                timestamp = _one_line(dialogue_line.get("timestamp"))
                if text:
                    line = (f"   - dialogue {speaker}: " if speaker else "   - dialogue: ") + text
                    if timestamp:
                        line += f" ({timestamp})"
                    lines.append(line)

            for alive in segment.get("alive_subjects", []) or []:
                if not isinstance(alive, dict):
                    continue
                sid = _one_line(alive.get("subject_id"))
                action = _as_dict(alive.get("action"))
                parts = [
                    _one_line(action.get("primary_action")),
                    _one_line(action.get("interaction")),
                    _one_line(action.get("facial_expression")),
                ]
                parts = [part for part in parts if part]
                if parts:
                    lines.append(f"   - action {sid}: " + "; ".join(parts))

            for obj_item in segment.get("objects", []) or []:
                if not isinstance(obj_item, dict):
                    continue
                oid = _one_line(obj_item.get("object_id"))
                state = _as_dict(obj_item.get("dynamic_state"))
                parts = [_one_line(state.get("state_change")), _one_line(state.get("motion_detail"))]
                parts = [part for part in parts if part]
                if parts:
                    lines.append(f"   - objects {oid}: " + "; ".join(parts))

    if isinstance(reference_layer, list):
        lines.extend(str(line) for line in reference_layer if str(line).strip())
    return "\n".join(line for line in lines if line.strip())


def normalize_prompt(prompt: str) -> str:
    """Normalize structured prompts while leaving ordinary text byte-for-byte intact."""

    try:
        parsed = json.loads(prompt)
    except (json.JSONDecodeError, TypeError):
        return prompt
    if isinstance(parsed, dict | list):
        return json_to_compact_markdown(prompt)
    return prompt


class Magi2Qwen35TextEncoder(nn.Module):
    """Qwen3.5-27B feature extractor used by MAGI-2 Preview."""

    def __init__(
        self,
        model_path: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        max_length: int = 7000,
        skip_layer: int = 2,
        local_files_only: bool = True,
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.skip_layer = skip_layer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="right",
            local_files_only=local_files_only,
        )
        cjk_split = pre_tokenizers.Split(
            pattern=Regex(
                r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF"
                r"\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF])"
            ),
            behavior="isolated",
        )
        original = self.tokenizer.backend_tokenizer.pre_tokenizer
        self.tokenizer.backend_tokenizer.pre_tokenizer = pre_tokenizers.Sequence([cjk_split, original])

        from transformers import Qwen3_5TextModel

        self.text_model = Qwen3_5TextModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )
        self.text_model.eval().requires_grad_(False)

    def _tokenize(self, prompt: str, *, offsets: bool = False):
        return self.tokenizer(
            [normalize_prompt(prompt)],
            return_tensors="pt",
            padding="longest",
            return_offsets_mapping=offsets,
            max_length=self.max_length,
            truncation=True,
        )

    def get_target_token_indices(self, prompt: str, target: str | None) -> list[int] | None:
        if not target:
            return None
        normalized = normalize_prompt(prompt)
        offsets = self._tokenize(prompt, offsets=True)["offset_mapping"][0]
        start = normalized.find(target)
        if start < 0:
            return None
        end = start + len(target)
        indices = []
        for index, (token_start, token_end) in enumerate(offsets.tolist()):
            if token_start == 0 and token_end == 0 and index != 0:
                continue
            if max(token_start, start) < min(token_end, end):
                indices.append(index)
        return indices

    def pool_figure_tokens(
        self,
        prompt: str,
        targets: list[str],
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """Mean-pool the token features spelling each ``<Figure N>`` marker."""

        embeddings = []
        for target in targets:
            indices = self.get_target_token_indices(prompt, target)
            if indices:
                embeddings.append(text_features[0, indices].mean(dim=0).clone())
            else:
                embeddings.append(text_features.new_zeros(text_features.shape[-1]))
        return torch.stack(embeddings)

    @torch.inference_mode()
    def encode(self, prompt: str) -> torch.Tensor:
        inputs = self._tokenize(prompt)
        device = next(self.text_model.parameters()).device
        inputs = inputs.to(device)
        outputs = self.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )
        if self.skip_layer == 0:
            return outputs.last_hidden_state
        return outputs.hidden_states[-(self.skip_layer + 1)]

    forward = encode


__all__ = [
    "Magi2Qwen35TextEncoder",
    "json_to_compact_markdown",
    "normalize_prompt",
]
