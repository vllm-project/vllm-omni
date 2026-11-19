# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processors for a forced-aligner stage.

Bridges a Code2Wav (audio) stage to a pooling forced-aligner stage:

    Talker (LLM_AR) -> Code2Wav (LLM_GENERATION) -> ForcedAligner (LLM_AR + runner=pooling)

``code2wav2aligner`` is a synchronous ``custom_process_input_func`` (the same
shape as ``qwen3_tts.talker2code2wav``). It runs in the orchestrator process,
reads the synthesized waveform from the upstream Code2Wav output and the
ground-truth text from the original request, then builds the aligner prompt
(``<timestamp>`` markers per word) + audio multimodal input. ``asr2aligner``
instead aligns a completed ASR transcript against the original input audio.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
from vllm.logger import init_logger
from vllm.multimodal.audio import MONO_AUDIO_SPEC, get_audio_duration, normalize_audio

from vllm_omni.model_executor.models.interfaces import get_transcription_language
from vllm_omni.model_executor.stage_input_processors.tts_utils import (
    extract_language_from_prompt,
)
from vllm_omni.utils.qwen3_force_align_processor import AUDIO_PLACEHOLDER, build_prompt, segment_words

logger = init_logger(__name__)

# Shared keys / names for the forced-aligner stage payloads, so the producer
# (this processor), the worker-side decode, and the serving consumer don't
# duplicate magic strings.
ALIGNER_WORDS_KEY = "aligner_words"
ALIGNER_DURATION_MS_KEY = "aligner_audio_duration_ms"
TIMESTAMPS_MODALITY = "timestamps"
# Dotted path of the pooling-output decoder hook (read by the AR scheduler for
# a stage whose engine_args set ``pooling_output_decoder``).
POOLING_OUTPUT_DECODER_PATH = "vllm_omni.model_executor.stage_input_processors.forced_aligner.decode_pooling_output"


def decode_pooling_output(logits: Any, request: Any, hf_config: Any) -> Any:
    """Worker-side decoder for the forced-aligner pooling stage.

    Converts the aligner's ``[n_token, classify_num]`` logits into a bare
    ``[n_words, 2]`` int32 tensor of per-word ``[start_ms, end_ms]`` (word
    strings travel separately via ``additional_information``).
    """
    from vllm_omni.engine.serialization import deserialize_additional_information
    from vllm_omni.utils.forced_aligner import _decode_timestamps

    ts_id = int(getattr(hf_config, "timestamp_token_id"))
    thinker_cfg = getattr(hf_config, "thinker_config", hf_config)
    classify_num = int(getattr(thinker_cfg, "classify_num"))
    seg_ms = float(getattr(hf_config, "timestamp_segment_time"))

    prompt_token_ids = list(getattr(request, "prompt_token_ids", []) or [])
    positions = [i for i, t in enumerate(prompt_token_ids) if t == ts_id]

    info = deserialize_additional_information(getattr(request, "additional_information", None)) or {}
    words_field = info.get(ALIGNER_WORDS_KEY)
    words = words_field[0] if isinstance(words_field, list) and words_field else []

    # Real waveform duration (stashed by code2wav2aligner) drives the decode clamp; fall back to the
    # theoretical ceiling (clamp no-op) if absent.
    duration_field = info.get(ALIGNER_DURATION_MS_KEY)
    duration_ms = (
        float(duration_field[0])
        if isinstance(duration_field, list) and duration_field
        else float(classify_num) * seg_ms
    )

    timestamps = _decode_timestamps(
        logits=logits,
        words=list(words),
        timestamp_positions=positions,
        classify_num=classify_num,
        timestamp_segment_time_ms=seg_ms,
        audio_duration_ms=duration_ms,
    )
    # No failure sentinel: an empty [0, 2] tensor decodes downstream to timestamps: null.
    ms = torch.tensor(
        [[t.start_ms, t.end_ms] for t in timestamps],
        dtype=torch.int32,
    ).reshape(-1, 2)
    return ms


def _extract_waveform(multimodal_output: dict[str, Any]) -> tuple[np.ndarray, int]:
    """Pull a mono float32 waveform + sample rate from a Code2Wav output."""
    wav = multimodal_output.get("audio")
    if wav is None:
        wav = multimodal_output.get("model_outputs")
    if isinstance(wav, list):
        wav = wav[0] if wav else None
    if isinstance(wav, torch.Tensor):
        wav_np = wav.detach().cpu().to(torch.float32).flatten().numpy()
    elif isinstance(wav, np.ndarray):
        wav_np = wav.astype(np.float32).flatten()
    else:
        wav_np = np.zeros(0, dtype=np.float32)

    sr_raw = multimodal_output.get("sr")
    if isinstance(sr_raw, list):
        sr_raw = sr_raw[-1] if sr_raw else None
    if isinstance(sr_raw, torch.Tensor):
        sr = int(sr_raw.reshape(-1)[-1].item()) if sr_raw.numel() > 0 else 24000
    elif sr_raw is None:
        sr = 24000
    else:
        sr = int(sr_raw)
    return wav_np, sr


def _extract_text(prompt: Any, index: int = 0) -> str:
    """Read the ground-truth text the aligner needs from the original prompt."""
    if not isinstance(prompt, dict):
        return ""
    text = (prompt.get("additional_information") or {}).get("text")
    if isinstance(text, list):
        return str(text[index]) if index < len(text) else ""
    return str(text) if text else ""


def parse_asr_output(raw_text: str, *, model_config: Any, model_cls: Any | None = None) -> tuple[str, str | None]:
    """Normalize ASR text with its source model's transcription contract.

    A supplied model class takes precedence over config-based resolution.
    Language metadata comes from the model's optional language hook. Missing
    source configuration is an error rather than an implicit model choice.
    """
    if model_config is None:
        raise ValueError("ASR forced alignment requires the source model configuration")
    if model_cls is None:
        from vllm.model_executor.model_loader import get_model_cls

        model_cls = get_model_cls(model_config)
    text = model_cls.post_process_output(raw_text)
    if not isinstance(text, str):
        raise TypeError("ASR post_process_output must return a transcript string")
    return text, get_transcription_language(model_cls, raw_text)


def asr2aligner(
    source_outputs: list[Any],
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
    _streaming_context: Any | None = None,
) -> list[Any]:
    """Align completed ASR text to the original waveform, preserving language.

    A silent transcript still produces a pooling request with no timestamp
    markers, so the downstream stage terminates with an empty alignment.
    """
    from vllm_omni.inputs.data import OmniTextPrompt

    aligner_inputs: list[OmniTextPrompt] = []
    for i, src in enumerate(source_outputs):
        if not getattr(src, "finished", False):
            continue
        source_prompt = prompt[i] if isinstance(prompt, list) else prompt
        if not isinstance(source_prompt, dict):
            raise ValueError("ASR forced alignment requires the original audio prompt")
        audio = (source_prompt.get("multi_modal_data") or {}).get("audio")
        if audio is None and isinstance(source_prompt.get("encoder_prompt"), dict):
            audio = (source_prompt["encoder_prompt"].get("multi_modal_data") or {}).get("audio")
        if isinstance(audio, list) and len(audio) == 1:
            audio = audio[0]
        if not isinstance(audio, tuple) or len(audio) != 2:
            raise ValueError("ASR forced alignment requires decoded audio as (waveform, sample_rate)")
        waveform, sample_rate = audio
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().cpu().to(torch.float32).numpy()
        waveform = np.asarray(waveform, dtype=np.float32)
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise ValueError("ASR forced alignment requires a positive audio sample rate")
        waveform = cast(np.ndarray, normalize_audio(waveform, MONO_AUDIO_SPEC))

        output = src.outputs[0]
        raw_text = getattr(output, "cumulative_text", None) or getattr(output, "text", "")
        if not isinstance(raw_text, str):
            raise TypeError("ASR output must contain a transcript string")
        text, detected_language = parse_asr_output(
            raw_text, model_config=getattr(_streaming_context, "source_model_config", None)
        )
        language_list = extract_language_from_prompt(source_prompt)
        if not language_list and detected_language:
            language_list = [detected_language]
        words = segment_words(text, language_list[0] if language_list else None)
        info: dict[str, Any] = {
            "text": [text],
            ALIGNER_WORDS_KEY: [words],
            ALIGNER_DURATION_MS_KEY: [get_audio_duration(y=waveform, sr=sample_rate) * 1000.0],
        }
        if language_list:
            info["language"] = language_list
        aligner_inputs.append(
            OmniTextPrompt(
                prompt=build_prompt(words) if words else AUDIO_PLACEHOLDER,
                multi_modal_data={"audio": (waveform, sample_rate)},
                additional_information=info,
            )
        )
    return aligner_inputs


def code2wav2aligner(
    source_outputs: list[Any],
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
    _streaming_context: Any | None = None,
) -> list[Any]:
    """Build forced-aligner stage inputs from finished Code2Wav audio.

    Emits a **text prompt** (``prompt`` string + ``multi_modal_data`` audio).
    Tokenization and audio feature extraction are deferred to the aligner
    stage's input preprocessor (run by the orchestrator), so this function
    needs neither the aligner tokenizer nor its mm processor and stays
    unit-testable on CPU.
    """
    from vllm_omni.inputs.data import OmniTextPrompt

    aligner_inputs: list[OmniTextPrompt] = []
    for i, src in enumerate(source_outputs):
        if not getattr(src, "finished", False):
            # Sentence-final alignment: only run once the audio is complete.
            continue
        mm = src.outputs[0].multimodal_output or {}
        wav_np, sample_rate = _extract_waveform(mm)
        if wav_np.size == 0:
            logger.warning("code2wav2aligner: empty waveform for request; skipping alignment.")
            continue

        text = _extract_text(prompt, index=i)
        language_list = extract_language_from_prompt(prompt, index=i)
        language = language_list[0] if language_list else None
        words = segment_words(text, language)
        prompt_str = build_prompt(words)

        additional_information: dict[str, Any] = {
            "text": [text],
            ALIGNER_WORDS_KEY: [words],
            ALIGNER_DURATION_MS_KEY: [float(wav_np.size) / float(sample_rate) * 1000.0],
        }
        if language_list is not None:
            additional_information["language"] = language_list

        aligner_inputs.append(
            OmniTextPrompt(
                prompt=prompt_str,
                multi_modal_data={"audio": (wav_np, sample_rate)},
                additional_information=additional_information,
            )
        )
    return aligner_inputs
