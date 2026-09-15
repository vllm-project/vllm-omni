# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.io_support import get_diffusion_output_type
from vllm_omni.entrypoints.openai.utils import is_video_generation_pipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    "model_class_name, multiview, controls",
    [
        ("Cosmos3MultiviewPipeline", True, ()),
        ("Cosmos3OmniDiffusersPipeline", False, ("edge", "blur", "depth", "seg", "wsm")),
        ("WanPipeline", False, ()),
        ("UnknownPipeline", False, ()),
    ],
)
def test_multiview_upload_capability_is_explicit(model_class_name, multiview, controls):
    from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata

    metadata = get_diffusion_model_metadata(model_class_name)
    assert metadata.supports_multiview_reference_inputs is multiview
    assert metadata.supported_control_upload_types == controls


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


@pytest.mark.parametrize(
    "model_class_name",
    [
        "LTX2TwoStagePipeline",
        "LTX2DistilledOneStagePipeline",
        "LTX2DistilledTwoStagePipeline",
        "WanDMDPipeline",
        "LingBotWorldCausalDMDPipeline",
        "LongCatVideoAvatarPipeline",
        "Cosmos3MultiviewPipeline",
        "SanaVideoPipeline",
        "SanaImageToVideoPipeline",
        "SanaWmPipeline",
    ],
)
def test_registered_video_aliases_declare_video_output(model_class_name):
    assert get_diffusion_output_type(model_class_name) == "video"


@pytest.mark.parametrize(
    "model_class_name",
    [
        "SanaVideoPipeline",
        "SanaImageToVideoPipeline",
    ],
)
def test_sana_video_pipelines_declare_video_output(model_class_name):
    assert get_diffusion_output_type(model_class_name) == "video"
