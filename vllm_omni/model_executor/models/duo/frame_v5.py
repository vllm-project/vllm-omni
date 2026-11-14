# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Causal frame-v5 input fragments for the resident Intent producer.

Only tokenization/media preprocessing happens here, never HF model inference.
Each continuation contains NEW media/text plus the previous message delimiter;
no previous video, sampled body, or static prefix is resubmitted.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

INTRO = (
    "<chunk_160> You are the assistant in a real-time full-duplex video call.\n"
    "The assistant's visual appearance and scene are described as follows:\n\""
)
INSTRUCTION = (
    '"\nPredict what the assistant should say in the next chunk. Output <wait> '
    "if the assistant is waiting or continuing an unfinished utterance.\n"
)
SPEECH_CUE = "In this chunk, the assistant should say:"


def build_system_prompt(chunk_token, description, explicit_speak, *, system_prompt=""):
    description = description.strip()
    if not description or any(t in description for t in ("<|", "<wait>", "<blank>", "<chunk_")):
        raise ValueError("Missing/invalid Assistant description")
    if not isinstance(system_prompt, str):
        raise ValueError("system_prompt must be a string")
    instruction = (
        "Begin every speaking chunk with <speak>. Output <wait> "
        if explicit_speak
        else "Predict what the assistant should say in the next chunk. Output <wait> "
    ) + "if the assistant is waiting or continuing an unfinished utterance."
    extra = system_prompt.strip()
    if any(t in extra for t in ("<|", "<speak>", "<wait>", "<blank>", "<chunk_")):
        raise ValueError("System instructions must not contain protocol control tokens")
    prefix = (
        f"{chunk_token} You are the assistant in a real-time full-duplex video call.\n"
        "The assistant's visual appearance and scene are described as follows:\n"
        f'"{description}"\n{instruction}\n'
    )
    return prefix + ("\nAdditional conversation instructions:\n" + extra + "\n" if extra else "") + SPEECH_CUE


def bind_intent_protocol(tokenizer, directory: str, manifest_path: str) -> dict:
    """Bind the operator-selected receipt to its exact tokenizer and adapter."""
    from .intent_adapter import file_sha256

    root = Path(directory)
    manifest = json.loads(Path(manifest_path).read_text())
    for filename, field in (
        ("tokenizer.json", "tokenizer_json_sha256"),
        ("chat_template.jinja", "chat_template_sha256"),
        ("adapter_model.safetensors", "adapter_model_sha256"),
        ("adapter_config.json", "adapter_config_sha256"),
    ):
        if file_sha256(root / filename) != manifest[field]:
            raise ValueError(f"Intent protocol hash mismatch: {filename}")
    chunk = manifest["chunk_tokens"]["708"]
    if tokenizer.encode(chunk["text"], add_special_tokens=False) != [chunk["id"]]:
        raise ValueError("Intent chunk token differs from protocol")
    if manifest["fps"] != 24 or manifest["frames_per_block"] != 17:
        raise ValueError("Intent protocol requires the 17/24 chunk clock")
    tokenizer.duplex_chunk_token = chunk["text"]
    tokenizer.duplex_protocol_manifest = manifest
    return manifest


@dataclass(frozen=True)
class FrameV5Fragment:
    prompt: dict
    expected_token_ids: list[int]
    shared_offsets: tuple[int, ...] | None
    shared_token_ids: tuple[int, ...] | None
    shared_lengths: tuple[int, int]


class NativeFrameV5Serializer:
    def __init__(
        self,
        tokenizer,
        processor,
        *,
        assistant_description,
        total_frames,
        max_pixels=16384,
        fps=24,
        grid_only=False,
        duo_legacy=False,
        system_prompt="",
        system_message=None,
    ):
        if isinstance(assistant_description, str):
            description = assistant_description.strip()
        else:
            fields = ("appearance", "environment", "initial_pose")
            if any(
                not isinstance(assistant_description.get(k), str) or not assistant_description[k].strip()
                for k in fields
            ):
                raise ValueError("Missing static assistant description")
            description = " ".join(assistant_description[k].strip() for k in fields)
        if any(t in description for t in ("<|", "<wait>", "<blank>", "<chunk_")):
            raise ValueError("Assistant description contains protocol controls")
        if type(total_frames) is not int or total_frames < 4 or fps != 24 or max_pixels < 56 * 56:
            raise ValueError("Invalid frame-v5 media geometry/clock")
        self.grid_only = grid_only
        self.tokenizer, self.processor = tokenizer, processor
        self.video_processor = processor.video_processor
        self.total_frames, self.fps = total_frames, fps
        self.size = {"shortest_edge": 1024 if duo_legacy else 56 * 56, "longest_edge": max_pixels}
        self.video_processor.size = dict(self.size)
        manifest = getattr(tokenizer, "duplex_protocol_manifest", {})
        explicit_speak = manifest.get("training_options", {}).get("explicit_speak_token", False)
        intro = INTRO
        instruction = INSTRUCTION
        if explicit_speak:
            intro = intro.replace("visual appearance and scene", "visual appearance, voice and scene")
            instruction = (
                '"\nBegin every speaking chunk with <speak>. Output <wait> '
                "if the assistant is waiting or continuing an unfinished utterance."
            )
        if system_prompt and not duo_legacy:
            instruction += "\nAdditional conversation instructions:\n" + system_prompt.strip() + "\n"
        self.prefix = []
        self.shared_offsets = []
        self.bootstrap_offset = None
        if duo_legacy:
            system = (
                system_message
                if system_message is not None
                else build_system_prompt(
                    tokenizer.duplex_chunk_token, description, explicit_speak, system_prompt=system_prompt
                )
            )
            messages = [dict(role="system", content=system), dict(role="assistant", content="<wait>")]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            self.prefix = self.encode(rendered)
            offsets = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            lengths = []
            for value, start in (
                (description, system.index('"' + description + '"') + 1),
                (SPEECH_CUE, system.rindex(SPEECH_CUE)),
            ):
                start += rendered.index(system)
                rows = [i for i, (a, b) in enumerate(offsets) if a < start + len(value) and b > start]
                self.shared_offsets.extend(rows)
                lengths.append(len(rows))
        elif explicit_speak:
            head = "<|im_start|>system\n" + intro.replace("<chunk_160>", tokenizer.duplex_chunk_token, 1)
            tail = "<|im_end|>\n<|im_start|>assistant\n<wait><|im_end|>\n"
            text = head + description + instruction + tail
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            self.prefix = encoded["input_ids"]
            start, stop = len(head), len(head) + len(description)
            self.shared_offsets = [i for i, (a, b) in enumerate(encoded["offset_mapping"]) if a < stop and b > start]
            lengths = [len(self.shared_offsets), 0]
            self.bootstrap_offset = max(
                i for i, token in enumerate(self.prefix) if token == tokenizer.convert_tokens_to_ids("<wait>")
            )
        else:
            lengths = []
            for text, selected in (
                (
                    "<|im_start|>system\n"
                    + intro.replace("<chunk_160>", getattr(tokenizer, "duplex_chunk_token", "<chunk_160>"), 1),
                    False,
                ),
                (description, True),
                (instruction, False),
                (SPEECH_CUE, True),
                ("<|im_end|>\n<|im_start|>assistant\n<wait><|im_end|>\n", False),
            ):
                ids = self.encode(text)
                if selected:
                    self.shared_offsets.extend(range(len(self.prefix), len(self.prefix) + len(ids)))
                    lengths.append(len(ids))
                self.prefix.extend(ids)
        self.shared_lengths = tuple(lengths)
        self.separator = self.encode("<|im_end|>\n")

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def fragment(self, rgb, *, frame_start, user_token_ids=(), user_emotion=""):
        """Caller supplies only committed user IDs, never future ASR text.

        Pure/idempotent: frame_start selects first/continuation prefix. Producer
        owns block sequencing, so a preprocessing retry cannot advance state.
        """
        if (
            not isinstance(rgb, np.ndarray)
            or rgb.dtype != np.uint8
            or rgb.ndim != 4
            or rgb.shape[-1] != 3
            or len(rgb) not in (4, 5, 17)
            or type(frame_start) is not int
            or frame_start < 0
            or frame_start + len(rgb) > self.total_frames
        ):
            raise ValueError("Expected complete uint8 causal codec observation")
        ids = list(user_token_ids)
        forbidden = set(self.tokenizer.all_special_ids)
        if any(type(i) is not int or i < 0 or i >= len(self.tokenizer) or i in forbidden for i in ids):
            raise ValueError("User speech must contain natural committed token IDs")
        if user_emotion and (
            user_emotion not in self.tokenizer.all_special_tokens or not user_emotion.startswith("<emo_")
        ):
            raise ValueError("Invalid user emotion control")
        body = (ids or self.encode("<blank>")) + self.encode(user_emotion)
        if self.grid_only:
            from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor, smart_resize

            if type(self.video_processor) is not Qwen3VLVideoProcessor or not self.video_processor.do_resize:
                raise ValueError("Grid-only path requires the audited native Qwen3VL processor")
            patch = self.video_processor.patch_size
            temporal = self.video_processor.temporal_patch_size
            if temporal != 2:
                raise ValueError("Frame-v5 requires temporal patch size 2")
            padded_count = len(rgb) + len(rgb) % 2
            height, width = smart_resize(
                padded_count,
                rgb.shape[1],
                rgb.shape[2],
                temporal_factor=temporal,
                factor=patch * self.video_processor.merge_size,
                min_pixels=self.size["shortest_edge"],
                max_pixels=self.size["longest_edge"],
            )
            t, h, w = padded_count // temporal, height // patch, width // patch
        else:
            padded = np.concatenate([rgb, rgb[-1:]], axis=0) if len(rgb) % 2 else rgb
            preprocessed = self.video_processor(videos=[padded], do_sample_frames=False, return_tensors="pt")
            t, h, w = preprocessed["video_grid_thw"][0].tolist()
        merge = int(self.video_processor.merge_size)
        count = h * w // merge**2
        if t != (len(rgb) + 1) // 2 or count < 1 or h * w % merge**2:
            raise ValueError("Invalid independent frame-v5 visual grid")
        video = "".join(
            f"<{min(frame_start + 2 * j + 1, frame_start + len(rgb) - 1) / self.fps:.1f} seconds>"
            + "<|vision_start|>"
            + "<|video_pad|>" * count
            + "<|vision_end|>"
            for j in range(t)
        )
        prefix = self.prefix if frame_start == 0 else self.separator
        suffix = body + self.encode("<|im_end|>\n<|im_start|>assistant\n")
        expected = list(prefix) + self.encode("<|im_start|>user\n" + video) + suffix
        # vLLM's newer processors expand bare video_pad into timestamp/vision
        # triplets; old processors require the surrounding markers.
        from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor

        bare = Qwen3VLMultiModalProcessor._expands_only_video_token(self.processor)
        placeholder = "<|video_pad|>" if bare else "<|vision_start|><|video_pad|><|vision_end|>"
        prompt = dict(
            prompt_token_ids=list(prefix) + self.encode("<|im_start|>user\n" + placeholder) + suffix,
            multi_modal_data={
                "video": (
                    rgb,
                    dict(
                        fps=float(self.fps),
                        total_num_frames=self.total_frames,
                        frames_indices=list(range(frame_start, frame_start + len(rgb))),
                        do_sample_frames=False,
                    ),
                )
            },
            mm_processor_kwargs=dict(do_sample_frames=False, size=dict(self.size)),
        )
        offsets = tuple(self.shared_offsets) if frame_start == 0 else None
        return FrameV5Fragment(
            prompt,
            expected,
            offsets,
            tuple(expected[i] for i in offsets) if offsets is not None else None,
            self.shared_lengths,
        )
