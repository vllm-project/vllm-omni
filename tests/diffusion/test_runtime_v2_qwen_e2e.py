# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from tests.helpers.mark import hardware_test


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_qwen_image_generates_through_runtime_v2():
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    engine = Omni(
        model="tiny-random/Qwen-Image",
        enable_runtime_v2=True,
        runtime_v2_denoise_chunk_size=2,
        enforce_eager=True,
    )
    try:
        outputs = engine.generate(
            {"prompt": "a red panda walking in snow"},
            sampling_params_list=OmniDiffusionSamplingParams(
                num_inference_steps=2,
                guidance_scale=0.0,
                height=256,
                width=256,
                seed=1234,
            ),
        )
    finally:
        engine.close()

    assert outputs and outputs[0].finished
    assert outputs[0].final_output_type == "image"
    assert outputs[0].images
    image = np.asarray(outputs[0].images[0])
    assert image.ndim == 3 and image.shape[2] == 3
    assert np.isfinite(image).all()
