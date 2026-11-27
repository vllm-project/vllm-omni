# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from tests.helpers.mark import hardware_test
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


@pytest.mark.advanced_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"}, num_cards={"cuda": 4})
def test_physis_wan_same_step_generates_all_chunks():
    model_path = os.getenv("PHYSIS_WAN21_1P3B_MODEL_PATH")
    if not model_path:
        pytest.skip("Set PHYSIS_WAN21_1P3B_MODEL_PATH to the Physis Wan 2.1 1.3B Same-Step snapshot.")

    prompt = {"prompt": "A corgi running along the beach at sunset, waves rolling in, golden light."}
    height, width, frames, chunks = 480, 832, 81, 19
    sampling = OmniDiffusionSamplingParams(
        height=height,
        width=width,
        num_frames=frames,
        fps=16,
        num_inference_steps=4,
        guidance_scale=1.0,
        seed=0,
        output_type="np",
        extra_args={
            "cell_schedule": "same_step",
            "num_denoise_steps": 4,
            "num_chunks": chunks,
            "kv_history_chunks": 18,
            "collect_chunk_timing": True,
        },
    )
    deploy = Path(__file__).resolve().parents[3] / "vllm_omni" / "deploy" / "physis_wan.yaml"
    with closing(Omni(model=model_path, model_class_name="PhysisWanPipeline", deploy_config=str(deploy))) as omni:
        outputs = omni.generate(prompt, sampling, use_tqdm=False)

    assert len(outputs) == 1
    result = outputs[0]
    assert result.finished
    assert result.error is None
    assert result.final_output_type == "video"
    assert len(result.images) == 1
    video = np.asarray(result.images[0])
    assert video.shape == (1, frames, height, width, 3)
    assert np.isfinite(video).all()
    assert not np.array_equal(video[0, 0], video[0, -1])
    assert {name for name in result.stage_durations if name.startswith("physis.chunk_")} == {
        f"physis.chunk_{index:04d}" for index in range(chunks)
    }
