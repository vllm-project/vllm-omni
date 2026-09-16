# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""E2E test verifying weight transfer actually changes generation output.

Flow:
1. Generate image1 with original weights
2. Randomly perturb a weight → generate image2 (should differ from image1)
3. Restore original weight → generate image3 (should match image1)

This validates that update_weights() truly updates model parameters rather
than just succeeding without effect.
"""

from __future__ import annotations

import asyncio
import copy
import uuid
from contextlib import ExitStack
from functools import lru_cache

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from tests.helpers.mark import hardware_test
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput

MODEL = "tiny-random/Qwen-Image"
TOKENIZER_MODEL = "Qwen/Qwen2-1.5B-Instruct"
_MIN_PROMPT_TOKENS = 35


def normalize_token_ids(tokenized_output) -> list[int]:
    """Normalize tokenizer outputs into a flat list[int]."""
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

    if isinstance(token_ids, list) and len(token_ids) == 1 and isinstance(token_ids[0], (list, tuple)):
        token_ids = list(token_ids[0])

    if not isinstance(token_ids, list):
        raise TypeError(f"token_ids must be list-like, got {type(token_ids).__name__}")

    normalized_ids = []
    for token_id in token_ids:
        if hasattr(token_id, "item"):
            token_id = token_id.item()
        normalized_ids.append(int(token_id))
    return normalized_ids


@lru_cache(maxsize=1)
def _tokenize_prompt(text: str) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, trust_remote_code=True)
    messages = [{"role": "user", "content": text}]
    token_ids = normalize_token_ids(
        tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    )
    assert len(token_ids) > _MIN_PROMPT_TOKENS, (
        f"Prompt too short ({len(token_ids)} tokens, need >{_MIN_PROMPT_TOKENS})"
    )
    return token_ids


def _sampling_params(seed: int = 42) -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        num_inference_steps=2,
        guidance_scale=0.0,
        height=256,
        width=256,
        seed=seed,
    )


async def _generate_once(
    engine: AsyncOmni,
    prompt: str,
    *,
    request_id: str,
    sampling_params: OmniDiffusionSamplingParams,
) -> OmniRequestOutput:
    prompt_ids = _tokenize_prompt(prompt)
    prompt_dict = {"prompt_ids": prompt_ids}

    last_output = None
    async for output in engine.generate(
        prompt=prompt_dict,
        request_id=request_id,
        sampling_params_list=[sampling_params],
        output_modalities=["image"],
    ):
        last_output = output

    assert last_output is not None
    assert isinstance(last_output, OmniRequestOutput)
    assert last_output.images, "Expected at least one generated image"
    return last_output


def _image_to_array(output: OmniRequestOutput) -> np.ndarray:
    """Extract first image as normalized numpy array."""
    image = output.images[0]
    arr = np.asarray(image, dtype=np.float32) / 255.0
    assert arr.ndim == 3 and arr.shape[2] == 3
    return arr


def _images_differ(arr1: np.ndarray, arr2: np.ndarray, threshold: float = 0.01) -> bool:
    """Check if two images differ by more than threshold (mean absolute diff)."""
    diff = np.abs(arr1 - arr2).mean()
    return diff > threshold


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.asyncio
async def test_weight_transfer_changes_generation():
    """Verify that weight updates actually change the generated output."""
    with ExitStack() as after:
        # Initialize engine with weight_transfer_config enabled
        engine = AsyncOmni(
            model=MODEL,
            enforce_eager=True,
            max_num_seqs=1,
            weight_transfer_config={"backend": "ipc"},
        )
        after.callback(engine.shutdown)

        prompt = "a beautiful sunset over the ocean with vibrant orange clouds"
        sampling_params = _sampling_params(seed=42)

        # Step 1: Generate image1 with original weights
        output1 = await _generate_once(
            engine, prompt, request_id=f"original_{uuid.uuid4().hex[:8]}", sampling_params=sampling_params
        )
        image1 = _image_to_array(output1)

        # Step 2: Get a weight tensor from the model and save its original value
        # Access the diffusion worker's model
        stage_client = engine.engine.stage_clients[0]
        worker = stage_client._engine.executor.driver_worker
        model_pipeline = worker.model_runner.pipeline

        # Find a weight to modify (use transformer's first layer)
        target_weight_name = None
        target_weight_original = None

        for name, param in model_pipeline.transformer.named_parameters():
            if param.requires_grad is False and param.numel() > 100:  # Pick a reasonable-sized weight
                target_weight_name = name
                target_weight_original = param.data.clone()
                break

        assert target_weight_name is not None, "Could not find a suitable weight to modify"

        # Step 3: Perturb the weight
        await engine.init_weight_transfer_engine({"backend": "ipc"})
        await engine.start_weight_update()

        # Create a perturbed version (add random noise)
        perturbed_weight = target_weight_original + torch.randn_like(target_weight_original) * 0.1

        await engine.update_weights({
            "names": [f"transformer.{target_weight_name}"],
            "tensors": [perturbed_weight],
        })
        await engine.finish_weight_update()

        # Step 4: Generate image2 with perturbed weights
        output2 = await _generate_once(
            engine, prompt, request_id=f"perturbed_{uuid.uuid4().hex[:8]}", sampling_params=sampling_params
        )
        image2 = _image_to_array(output2)

        # Step 5: Restore original weight
        await engine.start_weight_update()
        await engine.update_weights({
            "names": [f"transformer.{target_weight_name}"],
            "tensors": [target_weight_original],
        })
        await engine.finish_weight_update()

        # Step 6: Generate image3 with restored weights
        output3 = await _generate_once(
            engine, prompt, request_id=f"restored_{uuid.uuid4().hex[:8]}", sampling_params=sampling_params
        )
        image3 = _image_to_array(output3)

        # Assertions
        # image1 and image2 should differ (weight change affected output)
        assert _images_differ(image1, image2), (
            "Image2 should differ from Image1 after weight perturbation, "
            "but they are nearly identical. Weight transfer may not be working."
        )

        # image1 and image3 should be similar (restored weights produce similar output)
        # Note: Due to potential numerical precision or other factors, we use a looser check
        diff_1_3 = np.abs(image1 - image3).mean()
        diff_1_2 = np.abs(image1 - image2).mean()

        assert diff_1_3 < diff_1_2 * 0.5, (
            f"Image3 (restored) should be closer to Image1 (original) than Image2 (perturbed), "
            f"but diff(1,3)={diff_1_3:.6f} is not much smaller than diff(1,2)={diff_1_2:.6f}. "
            "Weight restoration may not be working correctly."
        )

        print(f"\n✓ Weight transfer validation passed:")
        print(f"  - Modified weight: transformer.{target_weight_name}")
        print(f"  - Image1 vs Image2 mean diff: {diff_1_2:.6f} (should be large)")
        print(f"  - Image1 vs Image3 mean diff: {diff_1_3:.6f} (should be small)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
