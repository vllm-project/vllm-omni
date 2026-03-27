# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E tests for AsyncOmni Qwen-Image generation flow (no Ray, no HTTP server)."""

from __future__ import annotations

import asyncio
import uuid
from contextlib import ExitStack
from functools import lru_cache

import numpy as np
import pytest
from transformers import AutoTokenizer

from tests.utils import hardware_test
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

CUSTOM_PIPELINE_CLASS = (
    "tests.e2e.offline_inference.custom_pipeline.qwen_image_pipeline_with_logprob.QwenImagePipelineWithLogProbForTest"
)
WORKER_EXTENSION_CLASS = (
    "tests.e2e.offline_inference.custom_pipeline.worker_extension.vLLMOmniColocateWorkerExtensionForTest"
)

# Use your specified HF repo name/path directly here
MODEL = "tiny-random/Qwen-Image"

TOKENIZER_MODEL = "Qwen/Qwen2-1.5B-Instruct"


# ---------------------------------------------------------------------
#                👇 Test Helper Functions & Fixtures 👇
# ---------------------------------------------------------------------

_MIN_PROMPT_TOKENS = 35


def normalize_token_ids(tokenized_output) -> list[int]:
    """Normalize tokenizer outputs into a flat ``list[int]``.

    This handles Transformers 4/5 differences where ``apply_chat_template(tokenize=True)``
    may return either ``list[int]`` or a ``BatchEncoding``/mapping with ``input_ids``.
    """

    token_ids = tokenized_output
    if isinstance(tokenized_output, dict):
        if "input_ids" in tokenized_output:
            token_ids = tokenized_output["input_ids"]
    elif hasattr(tokenized_output, "input_ids"):
        token_ids = tokenized_output.input_ids

    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()

    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)

    if isinstance(token_ids, list) and len(token_ids) == 1 and isinstance(token_ids[0], list | tuple):
        token_ids = list(token_ids[0])

    if not isinstance(token_ids, list):
        raise TypeError(f"token_ids must be list-like token ids, got {type(token_ids).__name__}: {token_ids!r}")

    normalized_ids = []
    for idx, token_id in enumerate(token_ids):
        if hasattr(token_id, "item"):
            token_id = token_id.item()
        try:
            normalized_ids.append(int(token_id))
        except (TypeError, ValueError) as e:
            raise TypeError(f"token_id must be int-convertible, got {type(token_id).__name__}: {token_id!r}") from e
    return normalized_ids


@lru_cache(maxsize=1)
def _tokenize_prompt(text: str) -> list[int]:
    """Tokenize a text prompt into valid token IDs for the model."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, trust_remote_code=True)
    messages = [{"role": "user", "content": text}]
    token_ids = normalize_token_ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
    assert len(token_ids) > _MIN_PROMPT_TOKENS, (
        f"Prompt too short ({len(token_ids)} tokens, need >{_MIN_PROMPT_TOKENS}). "
        f"The pipeline drops the first 34 chat‑template prefix tokens; "
        f"use a longer prompt so content tokens remain after the drop."
    )
    return token_ids


def _sampling_params(*, logprobs: bool = False, seed: int = 42) -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        num_inference_steps=2,
        guidance_scale=0.0,
        height=256,
        width=256,
        seed=seed,
        extra_args={"logprobs": logprobs},
    )


async def _generate_once(
    engine: AsyncOmni,
    prompt: str | list[int] | dict,
    *,
    request_id: str,
    sampling_params: OmniDiffusionSamplingParams,
) -> OmniRequestOutput:
    # Convert text prompt to dict with tokenized prompt_ids
    if isinstance(prompt, str):
        prompt_ids = _tokenize_prompt(prompt)
        prompt = {"prompt_ids": prompt_ids}
    elif isinstance(prompt, list):
        prompt = {"prompt_ids": prompt}
    # else: assume it's already a dict with prompt_ids

    last_output = None
    async for output in engine.generate(
        prompt=prompt,
        request_id=request_id,
        sampling_params_list=[sampling_params],
        output_modalities=["image"],
    ):
        last_output = output

    assert last_output is not None
    assert isinstance(last_output, OmniRequestOutput)
    return last_output


def _assert_valid_image_output(output: OmniRequestOutput) -> None:
    assert output.final_output_type == "image"
    assert output.images, "Expected at least one generated image"

    image = output.images[0]
    arr = np.asarray(image, dtype=np.float32) / 255.0

    assert arr.ndim == 3 and arr.shape[2] == 3, f"Expected HWC RGB image, got shape={arr.shape}"
    assert arr.shape[0] > 0 and arr.shape[1] > 0
    assert 0.0 <= float(arr[0, 0, 0]) <= 1.0


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.asyncio
async def test_async_omni_generate():
    with ExitStack() as after:
        engine = AsyncOmni(
            model=MODEL,
            custom_pipeline_args={"pipeline_class": CUSTOM_PIPELINE_CLASS},
            worker_extension_cls=WORKER_EXTENSION_CLASS,
            enforce_eager=True,
        )
        after.callback(engine.shutdown)

        output = await _generate_once(
            engine,
            "a beautiful sunset over the ocean with vibrant orange and purple clouds "
            "reflecting on the calm water surface near a rocky coastline",
            request_id=f"test_{uuid.uuid4().hex[:8]}",
            sampling_params=_sampling_params(logprobs=False, seed=42),
        )

        _assert_valid_image_output(output)


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.asyncio
async def test_async_omni_generate_with_logprobs():
    with ExitStack() as after:
        engine = AsyncOmni(
            model=MODEL,
            custom_pipeline_args={"pipeline_class": CUSTOM_PIPELINE_CLASS},
            worker_extension_cls=WORKER_EXTENSION_CLASS,
            enforce_eager=True,
        )
        after.callback(engine.shutdown)

        output = await _generate_once(
            engine,
            "a futuristic city at night with neon lights glowing on tall glass "
            "skyscrapers and flying vehicles soaring between the buildings",
            request_id=f"test_lp_{uuid.uuid4().hex[:8]}",
            sampling_params=_sampling_params(logprobs=True, seed=123),
        )

        _assert_valid_image_output(output)

        all_log_probs = output.custom_output.get("all_log_probs")
        assert all_log_probs is not None, "all_log_probs should be present when logprobs=True"
        assert hasattr(all_log_probs, "shape")
        assert all_log_probs.numel() > 0


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.asyncio
async def test_async_omni_generate_concurrent():
    with ExitStack() as after:
        engine = AsyncOmni(
            model=MODEL,
            custom_pipeline_args={"pipeline_class": CUSTOM_PIPELINE_CLASS},
            worker_extension_cls=WORKER_EXTENSION_CLASS,
            enforce_eager=True,
        )
        after.callback(engine.shutdown)

        prompts = [
            "a beautiful sunset over the ocean with vibrant orange and purple clouds "
            "reflecting on the calm water surface near a rocky coastline",
            "a fluffy orange cat sitting on a wooden windowsill looking outside at "
            "a garden full of colorful flowers on a bright sunny afternoon",
            "a majestic mountain landscape covered with fresh white snow under a "
            "clear blue sky with pine trees in the foreground and a frozen lake",
            "a futuristic city at night with neon lights glowing on tall glass "
            "skyscrapers and flying vehicles soaring between the buildings",
        ]

        tasks = [
            _generate_once(
                engine,
                prompt,
                request_id=f"concurrent_{i}_{uuid.uuid4().hex[:8]}",
                sampling_params=_sampling_params(logprobs=False, seed=100 + i),
            )
            for i, prompt in enumerate(prompts)
        ]

        outputs = await asyncio.gather(*tasks)

        assert len(outputs) == len(prompts)
        for output in outputs:
            _assert_valid_image_output(output)
