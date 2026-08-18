# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm_omni.entrypoints.openai.utils import is_video_generation_pipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_video_pipeline_requires_declared_final_video_stage():
    assert is_video_generation_pipeline(
        [
            SimpleNamespace(
                stage_type="llm",
                final_output=False,
                final_output_type=None,
            ),
            SimpleNamespace(
                stage_type="diffusion",
                final_output=True,
                final_output_type="video",
            ),
        ]
    )


@pytest.mark.parametrize(
    "stage_configs",
    [
        [SimpleNamespace(stage_type="diffusion")],
        [
            SimpleNamespace(
                stage_type="diffusion",
                final_output=True,
                final_output_type="image",
            )
        ],
        [
            {
                "stage_type": "diffusion",
                "final_output": False,
                "final_output_type": "video",
            }
        ],
    ],
)
def test_video_pipeline_rejects_non_video_final_outputs(stage_configs):
    assert not is_video_generation_pipeline(stage_configs)
