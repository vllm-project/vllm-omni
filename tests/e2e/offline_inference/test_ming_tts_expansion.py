# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E expansion tests for Ming-omni-tts offline inference (nightly CI)."""

import asyncio
import uuid
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.inputs import tokens_input

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni import AsyncOmni
from vllm_omni.model_executor.models.ming_tts.config_ming_tts import (
    KEY_MAX_DECODE_STEPS,
    SAMPLE_RATE,
    TEXT_EOS_TOKEN_ID,
)
from vllm_omni.model_executor.models.ming_tts.prompt_assembly import DEFAULT_PROMPT, build_ming_dense_prompt

MODEL = "inclusionAI/Ming-omni-tts-0.5B"
DEPLOY_CONFIG = get_deploy_config_path("ming_tts.yaml")
TEST_TEXT = "我会一直在这里陪着你，直到你慢慢地沉入那个最温柔的梦里。"
TEST_INSTRUCTION = "轻柔的ASMR耳语，慢速，贴近麦克风"
MIN_AUDIO_SAMPLES = 1000

pytestmark = [
    pytest.mark.slow,
    pytest.mark.tts,
    pytest.mark.skip(reason="https://github.com/vllm-project/vllm-omni/issues/4704"),
]


@pytest.fixture(scope="module")
def ming_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)


@pytest.fixture
def ming_engine():
    with OmniRunner(
        MODEL,
        deploy_config=DEPLOY_CONFIG,
        stage_init_timeout=300,
        enforce_eager=True,
    ) as runner:
        yield runner.omni


@pytest.fixture
def async_omni_engine():
    engine = AsyncOmni(
        model=MODEL,
        deploy_config=DEPLOY_CONFIG,
        stage_init_timeout=300,
        enforce_eager=True,
    )
    yield engine
    engine.shutdown()


def _build_prompt(
    tokenizer,
    *,
    text: str = TEST_TEXT,
    instruction=TEST_INSTRUCTION,
    use_zero_spk_emb: bool = True,
) -> dict:
    prompt_dict = build_ming_dense_prompt(
        tokenizer,
        prompt=DEFAULT_PROMPT,
        text=text,
        instruction=instruction,
        runtime_controls={KEY_MAX_DECODE_STEPS: 200},
        use_zero_spk_emb=use_zero_spk_emb,
    )
    prompt = tokens_input(prompt_token_ids=prompt_dict["prompt_token_ids"])
    prompt["prompt"] = prompt_dict["prompt"]
    prompt["text"] = prompt_dict["text"]
    prompt["additional_information"] = prompt_dict["additional_information"]
    return prompt


def _sampling_params_list() -> list[SamplingParams]:
    return [
        SamplingParams(
            temperature=0.0,
            max_tokens=201,
            stop_token_ids=[int(TEXT_EOS_TOKEN_ID)],
        ),
        SamplingParams(temperature=0.0, max_tokens=1),
    ]


def _flatten_audio(audio) -> torch.Tensor:
    if isinstance(audio, list):
        parts = [torch.as_tensor(item, dtype=torch.float32).reshape(-1).cpu() for item in audio]
        parts = [item for item in parts if item.numel() > 0]
        if not parts:
            return torch.zeros((0,), dtype=torch.float32)
        return torch.cat(parts, dim=0)
    return torch.as_tensor(audio, dtype=torch.float32).reshape(-1).cpu()


def _extract_audio(multimodal_output: Mapping[str, Any]) -> torch.Tensor:
    assert isinstance(multimodal_output, Mapping), f"Expected Mapping, got {type(multimodal_output)}"
    audio = multimodal_output.get("audio", multimodal_output.get("model_outputs"))
    assert audio is not None, f"No audio output found, keys={list(multimodal_output.keys())}"
    waveform = _flatten_audio(audio)
    if waveform.numel() == 0:
        raise RuntimeError("Generated audio waveform is empty")
    return waveform


def _extract_sample_rate(multimodal_output: Mapping[str, Any]) -> int:
    sample_rate = multimodal_output.get("sr")
    if sample_rate is None:
        raise RuntimeError("Expected multimodal_output['sr']")
    if isinstance(sample_rate, list):
        sample_rate = sample_rate[-1]
    if hasattr(sample_rate, "item"):
        sample_rate = sample_rate.item()
    return int(sample_rate)


def _extract_final_audio_outputs(outputs):
    final_outputs = []
    for item in outputs:
        if getattr(item, "final_output_type", None) == "audio":
            final_outputs.append(item)
            continue
        request_output = getattr(item, "request_output", None)
        if request_output is None:
            continue
        multimodal_output = getattr(request_output, "multimodal_output", None)
        if isinstance(multimodal_output, Mapping):
            final_outputs.append(item)
            continue
        completions = getattr(request_output, "outputs", None) or []
        if any(isinstance(getattr(completion, "multimodal_output", None), Mapping) for completion in completions):
            final_outputs.append(item)
    return final_outputs


def _extract_multimodal_output(output) -> Mapping[str, Any]:
    multimodal_output = getattr(output, "multimodal_output", None)
    if isinstance(multimodal_output, Mapping):
        return multimodal_output

    request_output = getattr(output, "request_output", None)
    if request_output is not None:
        multimodal_output = getattr(request_output, "multimodal_output", None)
        if isinstance(multimodal_output, Mapping):
            return multimodal_output
        completions = getattr(request_output, "outputs", None) or []
        for completion in completions:
            multimodal_output = getattr(completion, "multimodal_output", None)
            if isinstance(multimodal_output, Mapping):
                return multimodal_output

    raise AssertionError("No multimodal audio output found in Ming generate results")


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_ming_tts_offline_basic(ming_engine, ming_tokenizer) -> None:
    """Test blocking Ming generation through Omni."""
    outputs = ming_engine.generate(
        prompts=[_build_prompt(ming_tokenizer)],
        sampling_params_list=_sampling_params_list(),
        py_generator=False,
    )
    final_outputs = _extract_final_audio_outputs(outputs)
    assert len(final_outputs) == 1, f"Expected one final audio output, got {len(final_outputs)}"
    multimodal_output = _extract_multimodal_output(final_outputs[0])
    waveform = _extract_audio(multimodal_output)
    sample_rate = _extract_sample_rate(multimodal_output)
    assert waveform.ndim == 1
    assert waveform.shape[0] == waveform.numel()
    assert waveform.numel() > MIN_AUDIO_SAMPLES
    assert np.max(np.abs(waveform.numpy())) > 0.01, "Audio appears silent"
    assert sample_rate == SAMPLE_RATE, f"Expected Ming output sample rate {SAMPLE_RATE}, got {sample_rate}"


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_ming_tts_speaker_conditioning_differs(ming_engine, ming_tokenizer) -> None:
    """Test that different Ming speaker controls produce different waveform outputs."""
    style_outputs = ming_engine.generate(
        prompts=[_build_prompt(ming_tokenizer)],
        sampling_params_list=_sampling_params_list(),
        py_generator=False,
    )
    ip_outputs = ming_engine.generate(
        prompts=[_build_prompt(ming_tokenizer, text=TEST_TEXT, instruction={"IP": "灵小甄"}, use_zero_spk_emb=True)],
        sampling_params_list=_sampling_params_list(),
        py_generator=False,
    )

    style_final_outputs = _extract_final_audio_outputs(style_outputs)
    ip_final_outputs = _extract_final_audio_outputs(ip_outputs)
    assert len(style_final_outputs) == 1, "No style audio output produced"
    assert len(ip_final_outputs) == 1, "No IP audio output produced"

    style_waveform = _extract_audio(_extract_multimodal_output(style_final_outputs[0]))
    ip_waveform = _extract_audio(_extract_multimodal_output(ip_final_outputs[0]))
    assert style_waveform.numel() > MIN_AUDIO_SAMPLES
    assert ip_waveform.numel() > MIN_AUDIO_SAMPLES
    assert np.max(np.abs(style_waveform.numpy())) > 0.01, "Style audio appears silent"
    assert np.max(np.abs(ip_waveform.numpy())) > 0.01, "IP audio appears silent"

    overlap = min(int(style_waveform.numel()), int(ip_waveform.numel()))
    mean_abs_diff = torch.mean(torch.abs(style_waveform[:overlap] - ip_waveform[:overlap])).item()
    assert style_waveform.shape != ip_waveform.shape or mean_abs_diff > 1e-4, (
        "Speaker-conditioned outputs should differ, but style and IP waveforms were effectively identical"
    )


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_ming_tts_multiple_prompts_queued(ming_engine, ming_tokenizer) -> None:
    """Regression: supported max_num_seqs=1 config must still drain queued prompts."""
    prompts = [
        _build_prompt(
            ming_tokenizer,
            text="第一条语音用于验证 Ming dense 队列中的长请求可以完成。",
            instruction="平静自然的旁白，语速中等",
        ),
        _build_prompt(
            ming_tokenizer,
            text="第二条短语音也必须生成完成。",
            instruction="清晰明亮的女声",
        ),
    ]
    outputs = ming_engine.generate(
        prompts=prompts,
        sampling_params_list=_sampling_params_list(),
        py_generator=False,
    )
    final_outputs = _extract_final_audio_outputs(outputs)
    assert len(final_outputs) == len(prompts), f"Expected {len(prompts)} audio outputs, got {len(final_outputs)}"
    for i, output in enumerate(final_outputs):
        waveform = _extract_audio(_extract_multimodal_output(output))
        duration_s = waveform.numel() / SAMPLE_RATE
        assert 0.1 < duration_s < 30.0, f"Request {i} audio duration out of range: {duration_s:.2f}s"
        assert np.max(np.abs(waveform.numpy())) > 0.01, f"Request {i} audio appears silent"


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_ming_tts_offline_streaming(async_omni_engine, ming_tokenizer) -> None:
    """Test async_chunk streaming Ming generation through AsyncOmni."""

    async def _run() -> None:
        all_audio_chunks = []
        accumulated_samples = 0
        chunk_idx = 0
        sample_rate = None
        async for stage_output in async_omni_engine.generate(
            prompt=_build_prompt(ming_tokenizer),
            request_id=str(uuid.uuid4()),
            sampling_params_list=_sampling_params_list(),
        ):
            multimodal_output = stage_output.multimodal_output or {}
            audio = multimodal_output.get("audio")
            if "sr" in multimodal_output:
                sample_rate = _extract_sample_rate(multimodal_output)
            if audio is None:
                continue
            finished = stage_output.finished
            if isinstance(audio, torch.Tensor):
                if finished:
                    audio_chunk = audio[accumulated_samples:].float().detach().cpu()
                else:
                    audio_chunk = audio.float().detach().cpu()
            elif isinstance(audio, list):
                audio_chunk = _flatten_audio(audio)
            else:
                audio_chunk = torch.as_tensor(audio, dtype=torch.float32).reshape(-1).cpu()
            accumulated_samples += int(audio_chunk.numel())
            chunk_idx += 1
            if audio_chunk.numel() > 0:
                all_audio_chunks.append(audio_chunk)
        assert all_audio_chunks, "No streaming audio chunks received"
        waveform = torch.cat(all_audio_chunks, dim=0)
        assert waveform.numel() > MIN_AUDIO_SAMPLES
        assert np.max(np.abs(waveform.numpy())) > 0.01, "Audio appears silent"
        assert sample_rate is not None, "Streaming path did not return a sample rate"
        assert sample_rate == SAMPLE_RATE, f"Expected Ming output sample rate {SAMPLE_RATE}, got {sample_rate}"

    asyncio.run(_run())
