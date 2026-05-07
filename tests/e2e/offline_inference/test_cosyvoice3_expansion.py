# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Offline E2E smoke test for CosyVoice3 zero-shot reference inference.

This test uses the official upstream zero-shot prompt text/audio pair and
verifies a stable reference recipe:
- config-derived top_p/top_k and token-length ratios
- model EOS token as the stop token for stage-0 (talker) sampling
- a conservative repetition penalty to avoid degenerate loops
- stage-1 (code2wav) completion reports a normal terminal ``finish_reason``
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from huggingface_hub import snapshot_download
from vllm.sampling_params import SamplingParams

from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.model_executor.models.cosyvoice3.config import CosyVoice3Config
from vllm_omni.model_executor.models.cosyvoice3.tokenizer import get_qwen_tokenizer
from vllm_omni.outputs import OmniRequestOutput

MODEL = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
MODEL_DIR_ENV = "VLLM_OMNI_COSYVOICE3_MODEL_DIR"

# Vendored under tests/assets/cosyvoice3/ so the test does not depend on
# raw.githubusercontent.com being reachable from CI runners.
REFERENCE_PROMPT_WAV_PATH = get_asset_path("cosyvoice3/zero_shot_prompt.wav")
REFERENCE_PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
REFERENCE_SYNTH_TEXT = (
    "CosyVoice is undergoing a comprehensive upgrade, providing more accurate, "
    "stable, faster, and better voice generation capabilities."
)
REFERENCE_STAGE0_TEMPERATURE = 1.0
REFERENCE_STAGE0_REPETITION_PENALTY = 2.0


@functools.lru_cache(maxsize=1)
def _load_reference_prompt_wav() -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(REFERENCE_PROMPT_WAV_PATH), dtype="float32", always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    return np.asarray(audio, dtype=np.float32), int(sr)


@functools.lru_cache(maxsize=1)
def _resolve_model_dir() -> Path:
    override = os.environ.get(MODEL_DIR_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return Path(snapshot_download(MODEL, allow_patterns=["*"]))


def _reference_zero_shot_stage0_sampling(*, text: str) -> SamplingParams:
    config = CosyVoice3Config()
    sampling_cfg = config.llm.get("sampling", {})
    eos_token_id = int(config.llm["eos_token_id"])
    model_dir = _resolve_model_dir()
    tokenizer = get_qwen_tokenizer(
        token_path=str(model_dir / config.qwen_pretrain_path),
        skip_special_tokens=config.skip_special_tokens,
        version=config.version,
    )
    text_len = max(1, len(tokenizer.encode(text, allowed_special=config.allowed_special)))
    return SamplingParams(
        temperature=REFERENCE_STAGE0_TEMPERATURE,
        top_p=float(sampling_cfg.get("top_p", 0.8)),
        top_k=int(sampling_cfg.get("top_k", 25)),
        repetition_penalty=REFERENCE_STAGE0_REPETITION_PENALTY,
        stop_token_ids=[eos_token_id],
        min_tokens=int(text_len * config.min_token_text_ratio),
        max_tokens=int(text_len * config.max_token_text_ratio),
    )


def _concat_audio(audio_val) -> np.ndarray:
    import torch

    if isinstance(audio_val, list):
        tensors = []
        for t in audio_val:
            if t is None:
                continue
            if hasattr(t, "detach"):
                t = t.detach()
            if hasattr(t, "cpu"):
                t = t.cpu()
            if hasattr(t, "float"):
                t = t.float()
            if isinstance(t, torch.Tensor):
                tensors.append(t.reshape(-1))
        if not tensors:
            return np.zeros((0,), dtype=np.float32)
        return torch.cat(tensors, dim=-1).numpy().astype(np.float32, copy=False)

    if hasattr(audio_val, "detach"):
        audio_val = audio_val.detach()
    if hasattr(audio_val, "cpu"):
        audio_val = audio_val.cpu()
    if hasattr(audio_val, "float"):
        audio_val = audio_val.float()
    if hasattr(audio_val, "numpy"):
        audio_val = audio_val.numpy()
    audio_np = np.asarray(audio_val, dtype=np.float32)
    return audio_np.reshape(-1)


def _build_reference_inputs(prompt_audio: tuple[np.ndarray, int]) -> list[dict[str, object]]:
    return [
        {
            "prompt": REFERENCE_SYNTH_TEXT,
            "multi_modal_data": {"audio": prompt_audio},
            "modalities": ["audio"],
            "mm_processor_kwargs": {"prompt_text": REFERENCE_PROMPT_TEXT},
        }
    ]


# (model, stage_configs_path, extra_omni_kwargs) for ``@pytest.mark.parametrize("omni_runner", ..., indirect=True)``
_cosy_deployment = get_deploy_config_path("cosyvoice3.yaml")
_cosy_model_path = str(_resolve_model_dir())
_OMNI_RUNNER_PARAMS = [
    pytest.param(
        (_cosy_model_path, _cosy_deployment, {"async_chunk": False}),
        id="sync",
    ),
    pytest.param(
        (_cosy_model_path, _cosy_deployment, {"async_chunk": True}),
        id="async_chunk",
    ),
]
pytestmark = [
    pytest.mark.full_model,
    pytest.mark.tts,
    pytest.mark.parametrize("omni_runner", _OMNI_RUNNER_PARAMS, indirect=True),
]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_cosyvoice3_offline_reference_zero_shot(omni_runner: OmniRunner) -> None:
    """CosyVoice3 zero-shot reference inference should stop cleanly and produce sane audio."""
    prompt_audio, prompt_sr = _load_reference_prompt_wav()

    sampling_params_list = omni_runner.get_default_sampling_params_list()
    sampling_params_list[0] = _reference_zero_shot_stage0_sampling(text=REFERENCE_SYNTH_TEXT)

    outputs = omni_runner.omni.generate(_build_reference_inputs((prompt_audio, prompt_sr)), sampling_params_list)

    assert outputs, "No outputs returned"
    audio_mm = outputs[0].multimodal_output
    assert "audio" in audio_mm, "No audio output found"

    audio = _concat_audio(audio_mm["audio"])
    assert audio.size > 0, "Generated audio is empty"

    sr_val = audio_mm.get("sr", 24000)
    if isinstance(sr_val, list) and sr_val:
        sr_val = sr_val[-1]
    if hasattr(sr_val, "item"):
        sr_val = sr_val.item()
    sr = int(sr_val)
    assert sr == 24000, f"Unexpected sample_rate={sr}"

    duration_s = audio.size / sr
    assert 2.8 <= duration_s <= 8.8, f"Unexpected duration={duration_s:.3f}s (samples={audio.size}, sr={sr})"

    # Code2wav is ``final_output`` (stage 1); ``Omni.generate`` yields its ``request_output`` / completions.
    pipeline_out = OmniRequestOutput.unwrap_result(outputs[0])
    assert pipeline_out.stage_id == 1, f"expected final stage 1, got {pipeline_out.stage_id}"
    assert pipeline_out.final_output_type == "audio", (
        f"expected audio final_output_type, got {pipeline_out.final_output_type!r}"
    )
    ro = pipeline_out.request_output
    assert ro is not None, "stage 1 should include request_output"
    completions = getattr(ro, "outputs", None) or []
    assert completions, "stage 1 request_output should include at least one completion"
    completion = completions[0]
    finish_reason = getattr(completion, "finish_reason", None)
    assert finish_reason == "stop", f"Stage-1 finish_reason={finish_reason}, expected 'stop'"
    token_ids = getattr(completion, "token_ids", None) or []
    if token_ids:
        assert 1 <= len(token_ids) <= 4096, f"Stage-1 num_tokens={len(token_ids)}, unexpected range"
