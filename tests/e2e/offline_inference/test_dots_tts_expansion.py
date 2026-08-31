# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E tests for dots.tts native AR offline inference."""

from collections.abc import Mapping

import pytest
import torch

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path

DOTS_TTS_MODEL = "rednote-hilab/dots.tts-soar"
DEPLOY_CONFIG = get_deploy_config_path("dots_tts.yaml")
SAMPLE_RATE = 48000

# dots.tts ships a custom tokenizer carrying the added ``<|audio_gen_start|>``
# token, so remote code must be explicitly enabled.
_OMNI_RUNNER_PARAM = (DOTS_TTS_MODEL, DEPLOY_CONFIG, {"trust_remote_code": True})

pytestmark = pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True)


def _build_prompt(omni_runner: OmniRunner, text: str) -> dict:
    """Wrap ``text`` in the prefix scaffold dots.tts prefill expects.

    Unlike most TTS models here, a bare ``{"prompt": ...}`` string does not
    work: prefill needs ``[文本]<text>[文本对应语音]<|audio_gen_start|>`` as
    real Qwen2 token ids.
    """
    from transformers import AutoTokenizer

    from vllm_omni.model_executor.models.dots_tts.dots_tts_prompt import (
        build_dots_tts_prompt,
    )

    tokenizer = AutoTokenizer.from_pretrained(omni_runner.model_name, trust_remote_code=True)
    return build_dots_tts_prompt(tokenizer=tokenizer, text=text)


def _extract_audio(multimodal_output: dict) -> torch.Tensor:
    """Extract the final complete audio tensor from multimodal output."""
    assert isinstance(multimodal_output, (dict, Mapping)), f"Expected dict/Mapping, got {type(multimodal_output)}"

    # The talker emits delta chunks; the output processor accumulates them
    # under "model_outputs" ("audio" kept for backwards compat).
    audio = multimodal_output.get("model_outputs")
    if audio is None:
        audio = multimodal_output.get("audio")
    assert audio is not None, f"No audio key, got {list(multimodal_output.keys())}"

    if isinstance(audio, list):
        valid = [torch.as_tensor(x).float().cpu().reshape(-1) for x in audio if x is not None]
        assert valid, "No valid audio tensors in output list"
        audio = torch.cat(valid, dim=0) if len(valid) > 1 else valid[0]

    assert isinstance(audio, torch.Tensor), f"Expected Tensor, got {type(audio)}"
    return audio.float().cpu().reshape(-1)


@pytest.mark.slow
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_dots_tts_zero_shot_001(omni_runner: OmniRunner) -> None:
    """Zero-shot synthesis produces audio of a plausible duration."""
    prompt = _build_prompt(omni_runner, "Hello, this is a test of dots TTS running on vLLM Omni.")

    outputs = omni_runner.omni.generate([prompt])
    assert len(outputs) == 1

    audio = _extract_audio(outputs[0].outputs[0].multimodal_output)
    duration_s = audio.shape[0] / SAMPLE_RATE
    assert 0.5 < duration_s < 30.0, f"Audio duration out of range: {duration_s:.2f}s"
    assert torch.isfinite(audio).all(), "Audio contains NaN/Inf"
    assert audio.abs().max() > 1e-3, "Audio is silent"


@pytest.mark.slow
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_dots_tts_prefill_decode_mixed_batch_002(omni_runner: OmniRunner) -> None:
    """Mixed prefill+decode batches must keep per-request rows aligned.

    ``compute_logits`` fills ``logits[i]`` positionally from
    ``_results_queue``, so every request in the batch has to push exactly one
    entry per step, prefills included. When the prefill branch pushed nothing,
    a shorter request's stop signal could be read into another request's row.
    Regression guard for that pairing (PR #4765 review).
    """
    long_text = (
        "This is a deliberately long sentence that stays in the decode phase "
        "for many steps, so the shorter requests behind it keep entering "
        "prefill alongside it and reproduce the mixed batch pattern."
    )
    short_texts = ["Hello one.", "Hello two.", "Hello three.", "Hello four."]
    prompts = [_build_prompt(omni_runner, long_text)] + [_build_prompt(omni_runner, t) for t in short_texts]

    outputs = omni_runner.omni.generate(prompts)
    assert len(outputs) == len(prompts)

    for i, out in enumerate(outputs):
        audio = _extract_audio(out.outputs[0].multimodal_output)
        duration_s = audio.shape[0] / SAMPLE_RATE
        assert 0.1 < duration_s < 30.0, f"Request {i} audio duration out of range: {duration_s:.2f}s"


@pytest.mark.slow
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_dots_tts_deterministic_flow_matching_noise_003(omni_runner: OmniRunner) -> None:
    """The same text twice must yield the same audio.

    The DiT flow-matching noise is drawn from a per-request generator seeded
    from ``blake2b(seed:request_key:noise_step)`` rather than the global CUDA
    RNG. Drawing from the global RNG made output non-reproducible and let
    concurrent requests perturb each other's noise streams (PR #4765 review).
    """
    text = "Reading aloud every evening helps children build a lasting love of language."

    first = _extract_audio(
        omni_runner.omni.generate([_build_prompt(omni_runner, text)])[0].outputs[0].multimodal_output
    )
    second = _extract_audio(
        omni_runner.omni.generate([_build_prompt(omni_runner, text)])[0].outputs[0].multimodal_output
    )

    assert first.shape == second.shape, f"Length differs across runs: {first.shape} vs {second.shape}"
    assert torch.allclose(first, second, atol=1e-4), "Audio differs across runs; flow-matching noise is not seeded"
