# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared text-to-talker handoff for Ming speech pipelines."""

from __future__ import annotations

from collections.abc import Mapping

import regex as re
import torch
from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.ming_flash_omni.omni_voice import resolve_omni_voice
from vllm_omni.model_executor.models.ming_flash_omni.prompt_utils import create_omni_prompt
from vllm_omni.model_executor.models.ming_flash_omni.reference_audio import REFERENCE_AUDIO_FIELDS

# ASR/QA thinkers in transcribe mode prefix their output with the detected
# language tag ("Chinese\t<transcript>"). The tag is API metadata for
# text-only ASR responses; handing it to the talker makes the language name
# audible at the start of every synthesized utterance. Strip a leading
# ASCII `<lang>\t` tag before the text reaches the talker.
_LANG_TAG_RE = re.compile(r"^\s*[A-Za-z]{1,16}\t")


def strip_language_tag(text: str) -> str:
    """Drop a leading ``<language>\t`` tag from thinker output text."""
    return _LANG_TAG_RE.sub("", text, count=1) if text else text


def build_ming_talker_additional_info(
    text: str,
    prompt: OmniTokensPrompt | TextPrompt | Mapping | None,
) -> dict:
    """Preserve the same voice and generation defaults for sync/async handoffs."""
    if isinstance(prompt, Mapping):
        additional_info = (
            dict(prompt.get("additional_information") or {}) if "additional_information" in prompt else dict(prompt)
        )
    else:
        additional_info = getattr(prompt, "additional_information", None) or {}

    # Speaker embeddings may arrive as nested JSON lists.
    spk_emb = additional_info.get("spk_emb")
    if isinstance(spk_emb, list) and spk_emb and not hasattr(spk_emb[0], "device"):
        # Keep one tensor per speaker; [speakers, 192] is not a batch of
        # projected hidden states and must not become [1, speakers, 192].
        vectors = torch.as_tensor(spk_emb, dtype=torch.float32).reshape(-1, 192)
        spk_emb = [vector.unsqueeze(0) for vector in vectors]

    info = {
        "ming_task": "omni",
        "text": text,
        "spk_emb": spk_emb,
        "voice_name": additional_info.get("voice_name"),
        "prompt_text": additional_info.get("prompt_text"),
        "prompt_wav_lat": additional_info.get("prompt_wav_lat"),
        "prompt_wav_emb": additional_info.get("prompt_wav_emb"),
        "max_text_length": additional_info.get("max_text_length", 50),
    }
    # Preserve caller-supplied controls in both completed-answer and async
    # sentence handoffs. The talker resolves defaults only for absent keys.
    for key in (
        "prompt",
        "instruction",
        "prompt_wav",
        "reference_audio",
        *REFERENCE_AUDIO_FIELDS,
        "prompt_audio_mode",
        "speaker_conditioning",
        "cfg",
        "cfg_strength",
        "sigma",
        "temperature",
        "max_steps",
        "max_decode_steps",
        "min_decode_steps",
        "stop_threshold",
        "use_zero_spk_emb",
        "use_static_cache",
        "normalize_text",
        "scrub_markup",
        "tn_mode",
        "duration_limit",
        "stream_decode",
        "trim_trailing_silence",
        "seed",
    ):
        if key in additional_info:
            info[key] = additional_info[key]
    if not info.get("prompt"):
        info["prompt"] = create_omni_prompt(text)
    info.update(resolve_omni_voice(text, additional_info))
    return info


def build_ming_talker_inputs(
    source_outputs: list,
    prompt: OmniTokensPrompt | TextPrompt | list[OmniTokensPrompt | TextPrompt] | None = None,
) -> list[OmniTokensPrompt]:
    """Convert completed thinker text into self-contained Ming talker inputs."""
    prompts = prompt if isinstance(prompt, list) else [prompt]
    talker_inputs: list[OmniTokensPrompt] = []

    for index, source_output in enumerate(source_outputs):
        generated_text = getattr(source_output.outputs[0], "text", "") or ""
        # ASR transcribe responses carry a `<lang>\t` tag; never read it aloud.
        generated_text = strip_language_tag(generated_text)
        original_prompt = prompts[index] if index < len(prompts) else None
        # A TextPrompt's top-level prompt belongs to the thinker. Only its
        # additional_information may override the talker's synthesis prompt.
        if isinstance(original_prompt, Mapping):
            original_prompt = {"additional_information": original_prompt.get("additional_information") or {}}
        talker_inputs.append(
            OmniTokensPrompt(
                # The talker retokenizes text and builds its own embeddings.
                prompt_token_ids=[0],
                additional_information=build_ming_talker_additional_info(generated_text, original_prompt),
            )
        )

    return talker_inputs


__all__ = ["build_ming_talker_additional_info", "build_ming_talker_inputs", "strip_language_tag"]
