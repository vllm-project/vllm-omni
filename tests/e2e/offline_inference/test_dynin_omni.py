# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E offline smoke tests for Dynin-Omni.

- model: "snu-aidas/Dynin-Omni"
- stage config: tests/e2e/stage_configs/dynin_omni_ci.yaml
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from tests.utils import hardware_test

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DYNIN_CONFIG_PATH: Path | None = None
_DEFAULT_STAGE_CONFIG_PATH = _REPO_ROOT / "tests" / "e2e" / "stage_configs" / "dynin_omni_ci.yaml"

models = ["snu-aidas/Dynin-Omni"]
stage_configs = [str(_DEFAULT_STAGE_CONFIG_PATH)]
test_params = [(model, stage_config) for model in models for stage_config in stage_configs]

DYNIN_CONFIG_PATH = str(_DEFAULT_DYNIN_CONFIG_PATH) if _DEFAULT_DYNIN_CONFIG_PATH is not None else None

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.omni,
    pytest.mark.parametrize("omni_runner", test_params, indirect=True),
]


# prompting util
def _build_mmu_prompt(tokenizer: Any, question: str, dynin_config_path: str | None) -> dict[str, Any]:
    encoded = tokenizer(question, return_tensors="pt", add_special_tokens=True)
    token_ids = [int(v) for v in encoded["input_ids"][0].tolist()]
    attention_mask = [int(v) for v in encoded["attention_mask"][0].tolist()]
    additional_information: dict[str, Any] = {
        "task": ["mmu"],
        "detok_id": [0],
        "prompt_length": [len(token_ids)],
        "attention_mask": [attention_mask],
        "max_new_tokens": [64],
        "steps": [64],
        "block_length": [16],
        "temperature": [0.0],
    }
    if dynin_config_path:
        additional_information["dynin_config_path"] = [str(dynin_config_path)]
    return {
        "prompt_token_ids": token_ids,
        "additional_information": additional_information,
        "modalities": ["text"],
    }


def _build_mmu_multimodal_prompt(
    tokenizer: Any,
    question: str,
    dynin_config_path: str | None,
    *,
    image: Any | None = None,
    audio: tuple[np.ndarray, int] | None = None,
) -> dict[str, Any]:
    if image is None and audio is None:
        raise ValueError("At least one multimodal input (image or audio) must be provided.")

    prefix_chunks: list[str] = []
    mm_data: dict[str, Any] = {}
    if image is not None:
        prefix_chunks.append("<|soi|><|image|><|eoi|>")
        mm_data["image"] = image
    if audio is not None:
        prefix_chunks.append("<|soa|><|audio|><|eoa|>")
        mm_data["audio"] = audio

    prefixed_question = " ".join(prefix_chunks + [question]).strip()
    prompt = _build_mmu_prompt(
        tokenizer=tokenizer,
        question=prefixed_question,
        dynin_config_path=dynin_config_path,
    )
    prompt["multi_modal_data"] = mm_data
    prompt["modalities"] = ["text"]
    return prompt


def _generate_synthetic_image(width: int = 224, height: int = 224) -> np.ndarray:
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    red = np.tile(x, (height, 1))
    green = np.tile(y, (1, width))
    blue = ((red.astype(np.uint16) + green.astype(np.uint16)) // 2).astype(np.uint8)
    return np.stack([red, green, blue], axis=-1)


def _generate_synthetic_audio(duration_s: int = 5, sample_rate: int = 48_000) -> tuple[np.ndarray, int]:
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False, dtype=np.float32)
    waveform = 0.1 * np.sin(2.0 * np.pi * 440.0 * t)
    return waveform.astype(np.float32), sample_rate


# prompting util
def _build_t2s_decode_prompt(dynin_config_path: str | None) -> dict[str, Any]:
    # Bypass stage-0 generation and directly validate token->audio decode path.
    generated_audio_token_ids = [int(v) for v in ([10, 11, 12, 13, 14] * 32)]
    additional_information: dict[str, Any] = {
        "task": ["t2s"],
        "detok_id": [1],
        "generated_token_ids": [generated_audio_token_ids],
        "audio_codebook_size": [4096],
    }
    if dynin_config_path:
        additional_information["dynin_config_path"] = [str(dynin_config_path)]
    return {
        "prompt_token_ids": [0],
        "additional_information": additional_information,
        "modalities": ["audio"],
    }


# prompting util
def _build_t2i_decode_prompt(dynin_config_path: str | None) -> dict[str, Any]:
    # Bypass stage-0 generation and directly validate token->image decode path.
    # MAGVIT decode path expects a square token grid; 1024 tokens -> 32x32.
    generated_image_token_ids = [int(v) for v in ([10, 11, 12, 13, 14, 15, 16, 17] * 128)]
    additional_information: dict[str, Any] = {
        "task": ["t2i"],
        "detok_id": [2],
        "generated_token_ids": [generated_image_token_ids],
        "codebook_size": [8192],
    }
    if dynin_config_path:
        additional_information["dynin_config_path"] = [str(dynin_config_path)]
    return {
        "prompt_token_ids": [0],
        "additional_information": additional_information,
        "modalities": ["image"],
    }


def _configure_dynin_config_env() -> None:
    if DYNIN_CONFIG_PATH:
        os.environ["DYNIN_CONFIG_PATH"] = str(DYNIN_CONFIG_PATH)
    else:
        os.environ.pop("DYNIN_CONFIG_PATH", None)


def _is_finished_request_output(request_output: Any) -> bool:
    if request_output is None:
        return False
    req_list = request_output if isinstance(request_output, list) else [request_output]
    for req in req_list:
        if req is not None and bool(getattr(req, "finished", False)):
            return True
    return False


def _find_stage_output(outputs: list[Any], output_type: str) -> Any | None:
    matched = [
        stage_output for stage_output in outputs if getattr(stage_output, "final_output_type", None) == output_type
    ]
    if not matched:
        return None

    # Prefer the latest finished chunk to avoid picking an intermediate stream output.
    for stage_output in reversed(matched):
        if _is_finished_request_output(getattr(stage_output, "request_output", None)):
            return stage_output
    return matched[-1]


def _to_token_list(value: Any) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "flatten"):
        value = value.flatten().tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for token in value:
        if isinstance(token, bool):
            continue
        try:
            out.append(int(token))
        except Exception:
            continue
    return out


def _extract_text(stage_output: Any, tokenizer: Any | None = None) -> str:
    request_output = getattr(stage_output, "request_output", None)
    if request_output is None:
        return ""
    req_list = request_output if isinstance(request_output, list) else [request_output]
    for req in req_list:
        completions = getattr(req, "outputs", None) or []
        if not completions:
            continue
        completion = completions[0]
        mm_out = (
            getattr(completion, "multimodal_output", None)
            or getattr(req, "multimodal_output", None)
            or getattr(stage_output, "multimodal_output", None)
            or {}
        )
        text = mm_out.get("text")
        if isinstance(text, list) and text:
            text = text[-1]
        if isinstance(text, str) and text.strip():
            return text.strip()
        if tokenizer is not None:
            for key in ("text_tokens", "token_ids"):
                token_ids = _to_token_list(mm_out.get(key))
                if not token_ids:
                    continue
                decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
                if isinstance(decoded, str) and decoded.strip():
                    return decoded.strip()
        fallback = getattr(completion, "text", None)
        if isinstance(fallback, str) and fallback.strip():
            return fallback.strip()
    return ""


def _extract_audio(stage_output: Any) -> Any | None:
    request_output = getattr(stage_output, "request_output", None)
    if request_output is None:
        return None
    req_list = request_output if isinstance(request_output, list) else [request_output]
    for req in req_list:
        completions = getattr(req, "outputs", None) or []
        if not completions:
            continue
        completion = completions[0]
        mm_out = getattr(completion, "multimodal_output", None) or {}
        if "audio" in mm_out:
            return mm_out["audio"]
    return None


def _extract_image(stage_output: Any) -> Any | None:
    request_output = getattr(stage_output, "request_output", None)
    if request_output is None:
        return None
    req_list = request_output if isinstance(request_output, list) else [request_output]
    for req in req_list:
        completions = getattr(req, "outputs", None) or []
        if not completions:
            continue
        completion = completions[0]
        mm_out = getattr(completion, "multimodal_output", None) or {}
        if "image" in mm_out:
            return mm_out["image"]
    return None


def _numel(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, torch.Tensor):
        return int(value.numel())
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            total = 1
            for dim in shape:
                total *= int(dim)
            return int(total)
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return len(value)
    return 0


@hardware_test(res={"cuda": "L4", "rocm": "MI325"})
def test_dynin_t2i_decode_to_image(omni_runner) -> None:
    _configure_dynin_config_env()
    prompt = _build_t2i_decode_prompt(dynin_config_path=DYNIN_CONFIG_PATH)

    outputs = omni_runner.generate([prompt])

    image_output = _find_stage_output(outputs, "image")
    assert image_output is not None
    image_value = _extract_image(image_output)
    assert image_value is not None
    assert _numel(image_value) > 0


@hardware_test(res={"cuda": "L4", "rocm": "MI325"})
def test_dynin_mmu_to_text(omni_runner) -> None:
    _configure_dynin_config_env()
    tokenizer = AutoTokenizer.from_pretrained(omni_runner.model_name, trust_remote_code=True)
    prompt = _build_mmu_prompt(
        tokenizer=tokenizer,
        question="What is 2 + 2? Answer in one short sentence.",
        dynin_config_path=DYNIN_CONFIG_PATH,
    )

    outputs = omni_runner.generate([prompt])

    text_output = _find_stage_output(outputs, "text")
    assert text_output is not None
    text_content = _extract_text(text_output, tokenizer=tokenizer)
    assert text_content


@hardware_test(res={"cuda": "L4", "rocm": "MI325"})
def test_dynin_image_to_text(omni_runner) -> None:
    _configure_dynin_config_env()
    tokenizer = AutoTokenizer.from_pretrained(omni_runner.model_name, trust_remote_code=True)
    prompt = _build_mmu_multimodal_prompt(
        tokenizer=tokenizer,
        question="Describe the image briefly in one sentence.",
        dynin_config_path=DYNIN_CONFIG_PATH,
        image=_generate_synthetic_image(),
    )

    outputs = omni_runner.generate([prompt])

    text_output = _find_stage_output(outputs, "text")
    assert text_output is not None
    text_content = _extract_text(text_output, tokenizer=tokenizer)
    assert text_content


@hardware_test(res={"cuda": "L4", "rocm": "MI325"})
def test_dynin_speech_to_text(omni_runner) -> None:
    _configure_dynin_config_env()
    tokenizer = AutoTokenizer.from_pretrained(omni_runner.model_name, trust_remote_code=True)
    prompt = _build_mmu_multimodal_prompt(
        tokenizer=tokenizer,
        question="Transcribe the audio briefly in one sentence.",
        dynin_config_path=DYNIN_CONFIG_PATH,
        audio=_generate_synthetic_audio(),
    )

    outputs = omni_runner.generate([prompt])

    text_output = _find_stage_output(outputs, "text")
    assert text_output is not None
    text_content = _extract_text(text_output, tokenizer=tokenizer)
    assert text_content


@hardware_test(res={"cuda": "L4", "rocm": "MI325"})
def test_dynin_t2s_decode_to_audio(omni_runner) -> None:
    _configure_dynin_config_env()
    prompt = _build_t2s_decode_prompt(dynin_config_path=DYNIN_CONFIG_PATH)

    outputs = omni_runner.generate([prompt])

    audio_output = _find_stage_output(outputs, "audio")
    assert audio_output is not None
    audio_value = _extract_audio(audio_output)
    assert audio_value is not None
    assert _numel(audio_value) > 0
