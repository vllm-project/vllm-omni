# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Whisper preprocessing used by Kimi-Audio inputs.

Waveforms arrive as mono 16 kHz audio from the framework's media input path.
The feature extractor is injected; encoder weights and execution stay in the
model worker. GLM encoding consumes the original waveform separately.
"""

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal

import msgspec
import numpy as np
import torch
from transformers import WhisperFeatureExtractor

from .prompt import KimiAudioEncodedAudio, KimiAudioPromptBuilder

if TYPE_CHECKING:
    from vllm_omni.inputs.data import OmniTokensPrompt

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 30 * SAMPLE_RATE
SAMPLES_PER_TOKEN = 1280


@dataclass(frozen=True)
class KimiAudioWhisperInputs:
    input_features: torch.Tensor
    token_lengths: tuple[int, ...]


def prepare_whisper_inputs(
    waveform: np.ndarray,
    feature_extractor: WhisperFeatureExtractor,
    *,
    sampling_rate: int,
) -> KimiAudioWhisperInputs:
    """Split at 30 seconds, pad each segment, and keep its real 12.5 Hz length.

    Passing the full recording to Whisper's usual truncating processor would
    lose later segments. Padding is for Whisper only and must not affect GLM.
    """
    if sampling_rate != SAMPLE_RATE:
        raise ValueError("Kimi-Audio expects audio resampled to 16000 Hz by the input layer")
    if waveform.ndim != 1 or waveform.size == 0 or not np.issubdtype(waveform.dtype, np.floating):
        raise ValueError("Expected a nonempty mono floating point waveform")
    if not np.isfinite(waveform).all():
        raise ValueError("Audio waveform must contain finite samples")
    if (
        feature_extractor.sampling_rate != SAMPLE_RATE
        or feature_extractor.hop_length != 160
        or feature_extractor.n_fft != 400
        or feature_extractor.feature_size != 128
        or feature_extractor.n_samples != CHUNK_SAMPLES
    ):
        raise ValueError("Expected the Kimi-Audio Whisper-large-v3 feature extractor configuration")
    chunks = [waveform[start : start + CHUNK_SAMPLES] for start in range(0, len(waveform), CHUNK_SAMPLES)]
    token_lengths = tuple((len(chunk) - 1) // SAMPLES_PER_TOKEN + 1 for chunk in chunks)
    features = feature_extractor(
        chunks,
        sampling_rate=SAMPLE_RATE,
        padding="max_length",
        max_length=CHUNK_SAMPLES,
        truncation=False,
        do_normalize=False,
        return_tensors="pt",
    )["input_features"]
    return KimiAudioWhisperInputs(features, token_lengths)


def prepare_kimi_audio_inputs(
    messages: Sequence[Mapping[str, object]],
    prompt_builder: KimiAudioPromptBuilder,
    *,
    audio_inputs: Mapping[int, np.ndarray] | None = None,
    sampling_rate: int = SAMPLE_RATE,
    feature_extractor: WhisperFeatureExtractor | None = None,
    output_type: Literal["text", "both"] = "text",
    add_assistant_start_msg: bool = True,
) -> "OmniTokensPrompt":
    """Prepare one engine request from messages and framework-resolved audio.

    Audio is keyed by ORIGINAL message index and must already be mono 16 kHz.
    The CPU builder reserves the exact number of GLM slots from duration; it
    does not execute encoders. The AR worker replaces those slots and fuses
    both streams in ``preprocess``. Resource URLs/paths are never sent there.
    """
    from vllm_omni.data_entry_keys import serialize_payload

    audio_inputs = {} if audio_inputs is None else audio_inputs
    expected = {i for i, message in enumerate(messages) if message.get("message_type") in ("audio", "audio-text")}
    if set(audio_inputs) != expected:
        raise ValueError("audio_inputs must contain exactly the audio/audio-text message indices")
    if sampling_rate != SAMPLE_RATE:
        raise ValueError("Kimi-Audio expects audio resampled to 16000 Hz by the input layer")
    placeholders = {}
    payload = {
        "output_type": output_type,
        "special_tokens": asdict(prompt_builder.tokens),
        "audio_token_offset": prompt_builder.audio_token_offset,
        "audio_vocab_size": prompt_builder.audio_vocab_size,
    }
    for index in sorted(audio_inputs):
        waveform = audio_inputs[index]
        if waveform.ndim != 1 or waveform.size == 0 or not np.issubdtype(waveform.dtype, np.floating):
            raise ValueError("Expected a nonempty mono floating point waveform")
        if not np.isfinite(waveform).all():
            raise ValueError("Audio waveform must contain finite samples")
        # Serialize CPU tensors into owned bytes below, so later caller edits
        # cannot change a queued request or invalidate its cache identity.
        payload[f"waveform_{index}"] = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32))
        num_codes = (waveform.size - 1) // SAMPLES_PER_TOKEN + 1
        features = None
        if messages[index]["message_type"] == "audio":
            if feature_extractor is None:
                raise ValueError("Audio messages require the Whisper feature extractor")
            whisper = prepare_whisper_inputs(waveform, feature_extractor, sampling_rate=sampling_rate)
            payload[f"whisper_features_{index}"] = whisper.input_features
            payload[f"whisper_lengths_{index}"] = list(whisper.token_lengths)
            # Only the shape is needed to apply the existing message rules.
            # These meta features never enter the request or model execution.
            features = torch.empty(num_codes, prompt_builder.continuous_feature_size, device="meta")
        placeholders[index] = KimiAudioEncodedAudio([0] * num_codes, features)

    layout = prompt_builder.build(
        messages,
        audio_inputs=placeholders,
        output_type=output_type,
        add_assistant_start_msg=add_assistant_start_msg,
    )
    payload.update(
        text_token_ids=layout.text_token_ids,
        audio_token_ids=layout.audio_token_ids,
        is_continuous_mask=layout.is_continuous_mask,
        audio_spans=[[index, start, end] for index, (start, end) in layout.audio_spans.items()],
    )
    # The runner buffer crosses a dict[str, Any] IPC boundary. Keep tensors in
    # Omni's explicit bytes/shape/dtype format, not untyped raw tensor objects.
    wire = msgspec.to_builtins(serialize_payload(payload))
    return {
        "prompt_token_ids": layout.audio_token_ids,
        "modalities": ["text", "audio"] if output_type == "both" else ["text"],
        "model_intermediate_buffer": {"kimi_audio_input": wire},
        # Placeholder IDs alone omit text and audio content. Salt the KV cache
        # with the complete conditioning, including resolved waveform bytes.
        "cache_salt": hashlib.sha256(msgspec.msgpack.encode(wire)).hexdigest(),
    }
