# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os

import pytest
import torch

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniRunner
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.local_model, pytest.mark.diffusion]

MODEL = os.environ.get("HY_NOMAS_MODEL", "tencent/Hy-Nomas")


@pytest.mark.parametrize(
    "omni_runner",
    [
        pytest.param(
            (MODEL, None, {"enforce_eager": True}),
            id="resident",
            marks=hardware_marks(res={"cuda": "H200"}, num_cards=1),
        ),
        pytest.param(
            (MODEL, None, {"enforce_eager": True, "enable_cpu_offload": True}),
            id="cpu_offload",
            marks=hardware_marks(res={"cuda": "H200"}, num_cards=1),
        ),
        pytest.param(
            (MODEL, None, {"enforce_eager": True, "parallel_config": {"ulysses_degree": 2}}),
            id="sp2",
            marks=hardware_marks(res={"cuda": "H200"}, num_cards=2),
        ),
        pytest.param(
            (
                MODEL,
                None,
                {"enforce_eager": True, "parallel_config": {"ulysses_degree": 2}, "enable_cpu_offload": True},
            ),
            id="sp2_cpu_offload",
            marks=hardware_marks(res={"cuda": "H200"}, num_cards=2),
        ),
    ],
    indirect=True,
)
def test_two_segments_then_independent_request(omni_runner: OmniRunner):
    for prompt, extra, steps in (
        ("A forest beside a river. ||| The camera moves along the river.", {"segment_frame_boundaries": [0, 4, 9]}, 4),
        ("A beach at sunrise.", {}, 6),
    ):
        outputs = omni_runner.generate(
            [{"prompt": prompt}],
            [
                OmniDiffusionSamplingParams(
                    height=256,
                    width=256,
                    num_frames=9,
                    fps=24,
                    num_inference_steps=steps,
                    guidance_scale=1.0,
                    seed=42,
                    extra_args=extra,
                    output_type="pt",
                )
            ],
        )
        assert len(outputs) == 1 and len(outputs[0].images) == 1
        frames = outputs[0].images[0]
        assert isinstance(frames, torch.Tensor)
        assert frames.shape == (1, 9, 3, 256, 256)
        assert torch.isfinite(frames).all()
        assert frames.min() >= 0 and frames.max() <= 1
