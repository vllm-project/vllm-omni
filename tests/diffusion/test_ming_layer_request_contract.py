# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig, resolve_model_class_name
from vllm_omni.diffusion.utils.param_utils import apply_declared_extra_args
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.models.ming_image.pipeline import MING_IMAGE_PIPELINE
from vllm_omni.model_executor.stage_input_processors.ming_image import thinker2image
from vllm_omni.model_extras import get_extra_body_params

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def ming_checkpoint_configs(tmp_path):
    # Shared root metadata from Design-Layer @ff1b22d and Flash-Omni @c1ed556.
    # Their transformer classes, not the Bailing architecture, distinguish them.
    roots = {}
    for variant, transformer in (
        (
            "design",
            {"_class_name": "DiffusionTransformer", "alignment_padding_mode": "learned", "multi_frame_output": True},
        ),
        ("flash", {"_class_name": "ZImageTransformer2DModel"}),
    ):
        root = tmp_path / variant
        (root / "transformer").mkdir(parents=True)
        (root / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["BailingMM2NativeForConditionalGeneration"],
                    "model_type": "bailingmm_moe_v2_lite",
                }
            )
        )
        (root / "transformer/config.json").write_text(json.dumps(transformer))
        roots[variant] = str(root)
    return roots


@dataclass
class CpuCompletion:
    multimodal_output: dict[str, torch.Tensor]


@dataclass
class CpuThinkerOutput:
    prompt_token_ids: list[int]
    outputs: list[CpuCompletion]


def source_output():
    prompt_ids = [11] * 4 + [157158] + [157157] * 256 + [157159]
    hidden = torch.arange(len(prompt_ids) * 2).view(len(prompt_ids), 2)
    payload = {"final_hidden_states": hidden}
    payload.update({f"hidden_states_{layer}": hidden + layer for layer in (5, 12, 20)})
    return CpuThinkerOutput(prompt_ids, [CpuCompletion(payload)])


def test_real_design_config_resolves_layer_pipeline(ming_checkpoint_configs):
    assert resolve_model_class_name(ming_checkpoint_configs["design"]) == "MingImageDiffusionPipeline"


def test_real_flash_config_keeps_zimage_pipeline(ming_checkpoint_configs):
    assert resolve_model_class_name(ming_checkpoint_configs["flash"]) == "MingImagePipeline"


@pytest.mark.parametrize("num_layers", (1, 3, 6))
def test_real_config_preserves_layers_to_stage_bridge(ming_checkpoint_configs, num_layers):
    engine = object.__new__(AsyncOmniEngine)
    engine.model = ming_checkpoint_configs["design"]
    engine.pipeline_config = MING_IMAGE_PIPELINE
    engine._diffusion_od_config_view = None
    od_config = engine.get_diffusion_od_config()
    sampling = OmniDiffusionSamplingParams(extra_args={})
    apply_declared_extra_args(
        sampling,
        get_extra_body_params(od_config.model_class_name),
        {
            "num_layers": num_layers,
            "height": 1024,
            "width": 1024,
            "seed": 42,
        },
    )
    result = thinker2image(
        [source_output()], prompt={"multi_modal_data": {"img2img": "reference"}}, sampling_params=sampling
    )
    assert sampling.extra_args["num_layers"] == num_layers
    assert result[0]["extra"]["num_layers"] == num_layers
    assert result[0]["extra"]["query_hidden_states"].shape == (256, 2)
    assert engine.get_diffusion_od_config() is od_config


def test_real_config_worker_enrichment_preserves_pipeline(ming_checkpoint_configs):
    config = OmniDiffusionConfig(model=ming_checkpoint_configs["design"], model_class_name="MingImageDiffusionPipeline")
    config.enrich_config()
    assert config.model_class_name == "MingImageDiffusionPipeline"
    assert config.tf_model_config.multi_frame_output is True


def test_flash_worker_keeps_zimage_class(ming_checkpoint_configs):
    config = OmniDiffusionConfig(model=ming_checkpoint_configs["flash"])
    config.enrich_config()
    assert config.model_class_name == "MingImagePipeline"


def test_legacy_checkpoint_without_root_keeps_explicit_class(ming_checkpoint_configs):
    root = ming_checkpoint_configs["design"]
    (Path(root) / "config.json").unlink()
    engine = object.__new__(AsyncOmniEngine)
    engine.model = root
    engine.pipeline_config = MING_IMAGE_PIPELINE
    engine._diffusion_od_config_view = None
    assert engine.get_diffusion_od_config().model_class_name == "MingImageDiffusionPipeline"
    config = OmniDiffusionConfig(model=root, model_class_name="MingImageDiffusionPipeline")
    config.enrich_config()
    assert config.model_class_name == "MingImageDiffusionPipeline"
    assert config.tf_model_config.multi_frame_output is True
