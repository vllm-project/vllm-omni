# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Real EngineCore cache invalidation through the AsyncOmni control plane."""

import pytest
from PIL import Image
from vllm import SamplingParams

from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config
from vllm_omni.entrypoints.async_omni import AsyncOmni


@pytest.mark.advanced_model
@pytest.mark.omni
@pytest.mark.cuda
@pytest.mark.asyncio
async def test_cache_resets_invalidate_real_engine_and_allow_repeated_multimodal_inputs():
    deploy_config = modify_stage_config(
        get_deploy_config_path("ci/qwen2_5_omni_thinker_only.yaml"),
        updates={
            "stages": {
                0: {
                    "enable_prefix_caching": True,
                    "mm_processor_cache_gb": 0.1,
                    "max_model_len": 2048,
                    "max_num_batched_tokens": 2048,
                    "num_gpu_blocks_override": 256,
                    "enforce_eager": True,
                }
            }
        },
    )
    engine = AsyncOmni(
        model="Qwen/Qwen2.5-Omni-7B",
        deploy_config=deploy_config,
        stage_init_timeout=600,
    )

    async def generate(prompt, request_id):
        final = None
        async for output in engine.generate(
            prompt,
            request_id=request_id,
            sampling_params_list=[SamplingParams(max_tokens=4, temperature=0)],
        ):
            final = output
        assert final is not None and final.finished
        assert final.outputs and final.outputs[0].token_ids
        return final

    try:
        prompt = "Explain how a cache works. " * 40
        cold = await generate(prompt, "cold")
        warm = await generate(prompt, "warm")
        assert cold.num_cached_tokens == 0
        assert warm.num_cached_tokens > 0

        assert await engine.reset_prefix_cache(reset_running_requests=True, reset_connector=True)
        await engine.reset_encoder_cache()
        await engine.reset_mm_cache()
        reset = await generate(prompt, "reset")
        assert reset.num_cached_tokens == 0
        assert reset.outputs[0].token_ids == cold.outputs[0].token_ids

        image_prompt = {
            "prompt": "<|im_start|>user\n<|vision_start|><|IMAGE|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n",
            "multi_modal_data": {"image": Image.new("RGB", (56, 56), "red")},
        }
        before = await generate(image_prompt, "image-before")
        await engine.reset_mm_cache()
        await engine.reset_encoder_cache()
        assert await engine.reset_prefix_cache()
        after = await generate(image_prompt, "image-after")
        assert after.outputs[0].token_ids == before.outputs[0].token_ids
    finally:
        engine.shutdown()
