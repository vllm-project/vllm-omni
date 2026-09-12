# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Live-engine prefix-cache coverage for MammothModa2 AR-to-DiT."""

from __future__ import annotations

import os
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from vllm.sampling_params import SamplingParams

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.model_executor.stage_input_processors import mammoth_moda2 as stage_processor
from vllm_omni.model_extras import build_text_to_image_prompt, get_model_class_name
from vllm_omni.transformers_utils.configs.mammoth_moda2 import Mammothmoda2Config

MODEL = os.environ.get(
    "MAMMOTH_MODA2_TEST_MODEL",
    "bytedance-research/MammothModa2-Preview",
)
DEPLOY_CONFIG = get_deploy_config_path("mammoth_moda2_prefix_cache.yaml")
RUNNER_PARAM = (MODEL, DEPLOY_CONFIG)

IMAGE_SIZE = 256
BLOCK_SIZE = 16
MIN_IMAGE_CONDITION_COSINE = 0.8
PROMPT_CASES = (
    "red",
    "red red red red red",
    "red red red red red red red red red red",
)

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.slow,
    pytest.mark.diffusion,
    pytest.mark.cache,
]


@dataclass(frozen=True)
class AR2DiTCapture:
    cached_tokens: int
    cache_creation_tokens: int
    token_ids: list[int]
    hidden_states: torch.Tensor
    answer_start_index: int


def _build_request(omni: Any, prompt: str) -> dict[str, Any]:
    """Build a request for a cache-miss or cache-hit run."""
    request = build_text_to_image_prompt(
        model_class_name=get_model_class_name(omni),
        prompt={"prompt": prompt, "modalities": ["image"]},
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
    )
    request["additional_information"].update(
        {
            "num_inference_steps": [2],
            "text_guidance_scale": [1.0],
            "cfg_range": [0.0, 1.0],
        }
    )
    return request


def _sampling_params(request: dict[str, Any]) -> list[SamplingParams]:
    info = request["additional_information"]
    ar_width = int(info["ar_width"][0])
    ar_height = int(info["ar_height"][0])
    return [
        SamplingParams(
            temperature=0.0,
            top_k=1,
            max_tokens=ar_height * (ar_width + 1) + 1,
            detokenize=False,
        ),
        SamplingParams(temperature=0.0, max_tokens=1, detokenize=False),
    ]


def _cold_token_request(
    request: dict[str, Any],
    prompt_token_ids: list[int],
) -> dict[str, Any]:
    """Build a token request whose first cache block cannot be reused."""
    cold_request = dict(request)
    cold_request.pop("prompt")
    cold_prompt_ids = list(prompt_token_ids)
    assert len(cold_prompt_ids) >= BLOCK_SIZE
    assert cold_prompt_ids[0] != cold_prompt_ids[1]
    cold_prompt_ids[0], cold_prompt_ids[1] = cold_prompt_ids[1], cold_prompt_ids[0]
    cold_request["prompt_token_ids"] = cold_prompt_ids
    return cold_request


def _install_ar2dit_capture(monkeypatch: pytest.MonkeyPatch) -> list[AR2DiTCapture]:
    """Observe the real AR-to-DiT conversion without changing its result."""
    captures: list[AR2DiTCapture] = []
    original_ar2dit = stage_processor.ar2dit

    def capture_ar2dit(source_outputs, prompts=None, _requires_multimodal_data=False):
        dit_inputs = original_ar2dit(source_outputs, prompts, _requires_multimodal_data)
        for ar_output, dit_input in zip(source_outputs, dit_inputs, strict=True):
            info = dit_input["additional_information"]
            captures.append(
                AR2DiTCapture(
                    cached_tokens=int(ar_output.num_cached_tokens or 0),
                    cache_creation_tokens=int(ar_output.num_cache_creation_tokens or 0),
                    token_ids=list(info["full_token_ids"]),
                    hidden_states=info["full_hidden_states"].detach().cpu().clone(),
                    answer_start_index=int(info["answer_start_index"][0]),
                )
            )
        return dit_inputs

    monkeypatch.setattr(stage_processor, "ar2dit", capture_ar2dit)
    return captures


def _split_conditions(
    capture: AR2DiTCapture,
    config: Mammothmoda2Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror the DiT stage's token masks for numerical comparison."""
    token_ids = torch.tensor(capture.token_ids, dtype=torch.long)
    positions = torch.arange(len(capture.token_ids))
    visual_ids = torch.tensor(
        [
            config.image_token_id,
            config.video_token_id,
            config.vision_start_token_id,
            config.vision_end_token_id,
        ],
        dtype=torch.long,
    )
    questions_mask = positions < capture.answer_start_index
    gen_token_mask = token_ids >= config.llm_config.gen_vocab_start_index
    visual_token_mask = torch.isin(token_ids, visual_ids)
    text_mask = questions_mask & ~(visual_token_mask | gen_token_mask)
    image_mask = ~questions_mask & gen_token_mask
    return capture.hidden_states[text_mask], capture.hidden_states[image_mask]


def _mean_row_cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    assert first.shape == second.shape
    assert first.numel() > 0
    similarity = F.cosine_similarity(first.float(), second.float(), dim=-1, eps=1e-12)
    assert torch.isfinite(similarity).all()
    return float(similarity.mean())


def _assert_miss_hit_pair(
    miss: AR2DiTCapture,
    hit: AR2DiTCapture,
    config: Mammothmoda2Config,
) -> None:
    for capture in (miss, hit):
        assert capture.hidden_states.shape[0] == len(capture.token_ids)
    assert miss.answer_start_index == hit.answer_start_index
    assert miss.token_ids[: miss.answer_start_index] == hit.token_ids[: hit.answer_start_index]

    prompt_tokens = hit.answer_start_index
    expected_cached_tokens = (prompt_tokens - 1) // BLOCK_SIZE * BLOCK_SIZE
    assert hit.cached_tokens == expected_cached_tokens
    assert miss.cached_tokens <= hit.cached_tokens
    assert miss.cached_tokens + miss.cache_creation_tokens >= hit.cached_tokens
    expected_hit_creation = BLOCK_SIZE if prompt_tokens % BLOCK_SIZE == 0 else 0
    assert hit.cache_creation_tokens == expected_hit_creation

    cached_tokens = hit.cached_tokens
    assert torch.equal(
        miss.hidden_states[:cached_tokens],
        hit.hidden_states[:cached_tokens],
    )
    if cached_tokens < prompt_tokens:
        assert (
            _mean_row_cosine(
                miss.hidden_states[cached_tokens:prompt_tokens],
                hit.hidden_states[cached_tokens:prompt_tokens],
            )
            >= 0.999
        )

    miss_text, miss_image = _split_conditions(miss, config)
    hit_text, hit_image = _split_conditions(hit, config)
    assert _mean_row_cosine(miss_text, hit_text) >= 0.999
    assert _mean_row_cosine(miss_image, hit_image) >= MIN_IMAGE_CONDITION_COSINE


@pytest.fixture
def captured_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[tuple[OmniRunner, list[AR2DiTCapture]], None, None]:
    captures = _install_ar2dit_capture(monkeypatch)
    with OmniRunner(MODEL, deploy_config=DEPLOY_CONFIG) as runner:
        yield runner, captures


@hardware_test(res={"cuda": "H100"})
def test_live_engine_exercises_prefix_cache_miss_and_hit(captured_runner):
    """Exercise varied prefix lengths and a mixed hit/miss batch."""
    runner, captures = captured_runner
    config = Mammothmoda2Config.from_pretrained(MODEL)

    # Direct generation is intentional: the test needs raw stage cache
    # accounting and AR-to-DiT conditioning, which OfflineOmniClient hides.
    pairs: list[tuple[AR2DiTCapture, AR2DiTCapture]] = []
    for prompt in PROMPT_CASES:
        request = _build_request(runner.omni, prompt)
        capture_start = len(captures)
        runner.omni.generate(
            request,
            sampling_params_list=_sampling_params(request),
            use_tqdm=False,
        )
        runner.omni.generate(
            request,
            sampling_params_list=_sampling_params(request),
            use_tqdm=False,
        )
        assert len(captures) == capture_start + 2
        pair = captures[capture_start], captures[capture_start + 1]
        _assert_miss_hit_pair(*pair, config)
        pairs.append(pair)

    assert pairs[0][0].cached_tokens == 0
    assert [pair[1].answer_start_index for pair in pairs] == [28, 32, 37]

    warmed_request = _build_request(runner.omni, PROMPT_CASES[2])
    warmed_prompt_ids = pairs[2][1].token_ids[: pairs[2][1].answer_start_index]
    miss_request = _cold_token_request(warmed_request, warmed_prompt_ids)
    capture_start = len(captures)
    runner.omni.generate(
        [warmed_request, miss_request],
        sampling_params_list=_sampling_params(warmed_request),
        use_tqdm=False,
    )
    batch_captures = captures[capture_start:]
    assert len(batch_captures) == 2

    warmed = next(
        capture for capture in batch_captures if capture.token_ids[: capture.answer_start_index] == warmed_prompt_ids
    )
    miss = next(capture for capture in batch_captures if capture is not warmed)
    assert warmed.cached_tokens == (len(warmed_prompt_ids) - 1) // BLOCK_SIZE * BLOCK_SIZE
    assert warmed.cache_creation_tokens == 0
    assert miss.cached_tokens == 0
    assert miss.cache_creation_tokens > 0
