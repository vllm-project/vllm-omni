# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Direct Pika model routing, pinned partition resolution, and fixed sampling."""

import json

import pytest

from vllm_omni.diffusion.data import OmniDiffusionConfig, resolve_model_class_name
from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata
from vllm_omni.diffusion.models.minimax_h3 import pipeline_pika_h3 as pika_pipeline
from vllm_omni.diffusion.models.minimax_h3.pika import PIKA_BASE_REVISION, PikaSpec
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest
from vllm_omni.model_extras.pika_h3 import PIKA_MODEL_ID, PIKA_REVISION, is_pika_h3_model
from vllm_omni.model_extras.registry import get_video_generation_defaults, should_preserve_reference_image_size

from .test_pika_h3 import _manifest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _release(tmp_path, task):
    root = tmp_path / "release"
    adapter = root / task
    adapter.mkdir(parents=True)
    (adapter / "manifest.json").write_text(json.dumps(_manifest(task)))
    base = tmp_path / "base" / ("Ref2VA" if task == "ref2va" else "FL2VA")
    base.mkdir(parents=True)
    (base / "model_index.json").write_text("{}")
    return root, adapter, base


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
def test_release_routing_and_local_partition_selection(tmp_path, task):
    root, adapter, base = _release(tmp_path, task)
    assert resolve_model_class_name(PIKA_MODEL_ID) == "PikaH3Pipeline"
    assert resolve_model_class_name(str(root)) == "PikaH3Pipeline"
    assert resolve_model_class_name(str(adapter)) == "PikaH3Pipeline"
    for model, requested_task, base_override in (
        (root, task, base),
        (adapter, "auto", base.parent),
    ):
        resolved_base, resolved_adapter, spec = pika_pipeline.resolve_pika_h3_release(
            str(model), task_type=requested_task, revision=None, base_model_path=str(base_override)
        )
        assert (resolved_base, resolved_adapter, spec.task) == (base, adapter, task)


@pytest.mark.parametrize("manifest", ['{"model":"unrelated"}', "[]", "not JSON"])
def test_unrelated_model_is_not_pika(tmp_path, manifest):
    assert not is_pika_h3_model("someone/pika-other")
    (tmp_path / "manifest.json").write_text(manifest)
    assert not is_pika_h3_model(str(tmp_path))


@pytest.mark.parametrize("task,partition", [("t2va", "FL2VA"), ("ref2va", "Ref2VA")])
@pytest.mark.parametrize("revision", [None, "explicit-release"])
def test_remote_release_pins_base_and_downloads_only_selected_partition(
    tmp_path, monkeypatch, task, partition, revision
):
    adapter_task = "ref2va" if task == "ref2va" else "fl2va"
    root, adapter, base = _release(tmp_path, adapter_task)
    downloads = []

    def download(**kwargs):
        downloads.append(kwargs)
        return str(root if kwargs["model_name_or_path"] == PIKA_MODEL_ID else base.parent)

    monkeypatch.setattr(pika_pipeline, "download_weights_from_hf_specific", download)
    resolved = pika_pipeline.resolve_pika_h3_release(PIKA_MODEL_ID, task_type=task, revision=revision)
    assert resolved[:2] == (base, adapter)
    assert downloads == [
        dict(
            model_name_or_path=PIKA_MODEL_ID,
            cache_dir=None,
            allow_patterns=[f"{adapter_task}/*", "LICENSE", "NOTICE", "licenses/*"],
            revision=revision or PIKA_REVISION,
            require_all=True,
        ),
        dict(
            model_name_or_path="MiniMaxAI/MiniMax-H3",
            cache_dir=None,
            allow_patterns=[f"{partition}/**"],
            revision=PIKA_BASE_REVISION,
            require_all=True,
        ),
    ]


def test_wrong_partition_and_invalid_base_fail_before_loading_weights(tmp_path):
    root, adapter, base = _release(tmp_path, "ref2va")
    with pytest.raises(ValueError, match="matching adapter"):
        pika_pipeline.resolve_pika_h3_release(str(adapter), task_type="fl2va", revision=None)
    with pytest.raises(ValueError, match="one task partition"):
        pika_pipeline.resolve_pika_h3_release(str(root), task_type="combined", revision=None)
    with pytest.raises(ValueError, match="original native"):
        pika_pipeline.resolve_pika_h3_release(
            str(root), task_type="ref2va", revision=None, base_model_path=str(tmp_path / "missing")
        )


def test_pipeline_resolves_shared_loader_config_and_keeps_adapter_fixed(tmp_path, monkeypatch):
    root, adapter, base = _release(tmp_path, "ref2va")
    config = OmniDiffusionConfig(
        model=str(root), task_type="ref2va", additional_config={"pika_base_model_path": str(base)}
    )
    received = []
    monkeypatch.setattr(
        pika_pipeline.MiniMaxH3Pipeline, "__init__", lambda self, **kwargs: received.append(kwargs["od_config"])
    )
    pipeline = pika_pipeline.PikaH3Pipeline(od_config=config)
    assert received == [config]
    assert config.model == str(base)
    assert config.lora_path == str(adapter)
    assert config.task_type == "ref2va"
    assert config.revision is None
    assert pipeline.diffusion_lora_fixed_adapter is True
    assert pipeline._active_pika_spec(OmniDiffusionSamplingParams()) == PikaSpec("ref2va", "0" * 64)
    with pytest.raises(OmniClientError, match="remove the request LoRA override"):
        pipeline._active_pika_spec(OmniDiffusionSamplingParams(lora_scale=0.0))
    with pytest.raises(OmniClientError, match="remove the request LoRA override"):
        pipeline._active_pika_spec(OmniDiffusionSamplingParams(lora_request=LoRARequest("other", 2, "/other")))


@pytest.mark.parametrize("override", [{"lora_path": "/other"}, {"lora_scale": 0.5}])
def test_pipeline_rejects_startup_adapter_override(override):
    with pytest.raises(ValueError, match="remove the startup LoRA override"):
        pika_pipeline.PikaH3Pipeline(od_config=OmniDiffusionConfig(model=PIKA_MODEL_ID, **override))


def test_video_api_defaults_to_six_forwards_and_preserves_reference_geometry():
    defaults = get_video_generation_defaults("PikaH3Pipeline")
    assert defaults.num_inference_steps == 6
    assert (defaults.width, defaults.height, defaults.num_frames, defaults.fps) == (832, 480, 124, 24.0)
    assert defaults.duration_seconds is None
    assert should_preserve_reference_image_size("PikaH3Pipeline", model=PIKA_MODEL_ID)
    metadata = get_diffusion_model_metadata("PikaH3Pipeline")
    assert metadata.final_output_type == "video"
    assert metadata.supports_mixed_reference_inputs
    assert metadata.max_multimodal_image_inputs == 9
    assert not metadata.supports_latent_continuation
    assert not metadata.supports_latent_mask_editing
    assert not metadata.supports_timeline_guides
    assert not metadata.supported_control_upload_types
