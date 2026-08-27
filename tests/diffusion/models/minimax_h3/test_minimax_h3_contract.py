# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import sys
from multiprocessing.reduction import ForkingPickler
from types import SimpleNamespace
from typing import Any, TypeVar
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_ItemT = TypeVar("_ItemT")
_ResultT = TypeVar("_ResultT")


def _append_and_return(items: list[_ItemT], item: _ItemT, result: _ResultT) -> _ResultT:
    items.append(item)
    return result


def test_h3_prepares_resolved_cache_state_immediately_before_denoise():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.partition = "fl2va"
    pipeline.supported_tasks = frozenset({"t2va"})
    pipeline.default_video_shift = 12.0
    pipeline.default_audio_shift = 3.0
    pipeline.device = torch.device("cpu")
    pipeline.od_config = SimpleNamespace()
    cache_spec = object()
    quality_plan = SimpleNamespace(cache_dit=cache_spec)
    pipeline._quality_policy = Mock()
    pipeline._quality_policy.resolve.return_value = quality_plan
    events = []
    pipeline._cache_dit_runtime = SimpleNamespace(prepare=lambda spec: events.append(("prepare", spec)))
    pipeline.encode_prompt = Mock(
        return_value=(
            torch.ones(1, 2),
            torch.ones(1, dtype=torch.long),
        )
    )

    def diffuse(**kwargs):
        events.append(("diffuse", kwargs))
        return torch.zeros(1), torch.zeros(1)

    pipeline.diffuse = diffuse
    pipeline.decode = Mock(return_value=(torch.zeros(1), torch.zeros(1)))
    sampling = OmniDiffusionSamplingParams(
        quality="high",
        width=1344,
        height=768,
        fps=24,
        num_frames=124,
        num_inference_steps=50,
        extra_args={"task": "t2va", "aspect_ratio": "16:9"},
    )
    batch = DiffusionRequestBatch(
        [
            OmniDiffusionRequest(
                prompt="quality boundary",
                sampling_params=sampling,
                request_id="quality-boundary",
            )
        ]
    )

    output = pipeline.forward(batch)

    assert events[0] == ("prepare", cache_spec)
    assert events[1][0] == "diffuse"
    pipeline._quality_policy.resolve.assert_called_once_with(
        quality="high",
        num_inference_steps=50,
        extra_args={"task": "t2va", "aspect_ratio": "16:9"},
    )
    assert output.output == pipeline.decode.return_value


def test_pipeline_import_registry_and_component_discovery():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.registry import (
        _DIFFUSION_MODELS,
        _DIFFUSION_POST_PROCESS_FUNCS,
    )

    assert _DIFFUSION_MODELS["MiniMaxH3Pipeline"] == (
        "minimax_h3",
        "pipeline_minimax_h3",
        "MiniMaxH3Pipeline",
    )
    assert _DIFFUSION_MODELS["MiniMaxH3ModularPipeline"] == _DIFFUSION_MODELS["MiniMaxH3Pipeline"]
    assert _DIFFUSION_POST_PROCESS_FUNCS["MiniMaxH3Pipeline"] == "get_minimax_h3_post_process_func"
    assert (
        _DIFFUSION_POST_PROCESS_FUNCS["MiniMaxH3ModularPipeline"] == _DIFFUSION_POST_PROCESS_FUNCS["MiniMaxH3Pipeline"]
    )
    assert MiniMaxH3Pipeline._dit_modules == ["transformer", "transformers_ref"]
    assert MiniMaxH3Pipeline._encoder_modules == ["text_encoder"]
    assert MiniMaxH3Pipeline._vae_modules == ["video_vae", "audio_vae"]


def _write_partition_index(path, *, partition, tasks):
    path.mkdir(parents=True)
    (path / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "MiniMaxH3Pipeline",
                "_diffusers_version": "0.32.2",
                "_minimax_h3": {
                    "partition": partition,
                    "tasks": tasks,
                    "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
                },
            }
        ),
        encoding="utf-8",
    )


def test_modular_diffusers_index_is_resolved_generically(tmp_path):
    from vllm_omni.diffusion.data import OmniDiffusionConfig, resolve_model_class_name
    from vllm_omni.diffusion.utils.hf_utils import is_diffusion_model
    from vllm_omni.entrypoints.utils import resolve_model_config_path

    (tmp_path / "modular_model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "MiniMaxH3ModularPipeline",
                "_diffusers_version": "0.36.0.dev0",
            }
        ),
        encoding="utf-8",
    )

    assert is_diffusion_model(str(tmp_path))
    assert resolve_model_class_name(str(tmp_path)) == "MiniMaxH3ModularPipeline"
    assert resolve_model_config_path(str(tmp_path)) is None

    config = OmniDiffusionConfig(model=str(tmp_path))
    config.enrich_config()
    assert config.model_class_name == "MiniMaxH3ModularPipeline"
    assert config.supports_multimodal_inputs
    assert config.max_multimodal_image_inputs == 9
    assert config.supports_mixed_reference_inputs


@pytest.mark.parametrize(
    ("task_type", "partition"),
    [
        (None, "combined"),
        ("auto", "combined"),
        ("t2va", "fl2va"),
        ("fl2va", "fl2va"),
        ("ref2va", "ref2va"),
    ],
)
def test_startup_task_selects_weight_partition(task_type, partition):
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _minimax_h3_partition_for_task

    assert _minimax_h3_partition_for_task(task_type) == partition


def test_startup_auto_task_uses_explicit_local_partition(tmp_path):
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _minimax_h3_partition_for_task

    for directory, partition, tasks in (
        ("FL2VA", "fl2va", ["t2va", "fl2va"]),
        ("Ref2VA", "ref2va", ["ref2va"]),
    ):
        model_path = tmp_path / directory
        _write_partition_index(model_path, partition=partition, tasks=tasks)
        assert _minimax_h3_partition_for_task(None, str(model_path)) == partition
        assert _minimax_h3_partition_for_task("auto", str(model_path)) == partition
        assert _minimax_h3_partition_for_task("combined", str(model_path)) == "combined"


def test_startup_task_rejects_unsupported_value():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _minimax_h3_partition_for_task

    with pytest.raises(ValueError, match="task_type must be one of"):
        _minimax_h3_partition_for_task("unsupported")


def test_combined_task_inference_and_transformer_routing():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.errors import OmniClientError

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.partition = "combined"
    pipeline.supported_tasks = frozenset({"t2va", "fl2va", "ref2va"})
    pipeline.transformer = torch.nn.Identity()
    pipeline.transformers_ref = torch.nn.Linear(1, 1)
    assert pipeline._resolve_task(None, {}) == "t2va"
    assert pipeline._resolve_task(None, {"image": object()}) == "fl2va"
    assert pipeline._resolve_task(None, {"audio": object()}) == "ref2va"
    assert pipeline._resolve_task(None, {"video": object()}) == "ref2va"
    assert pipeline._resolve_task("fl2va", {"image": object()}, has_turbo_lora=True) == "fl2va"
    with pytest.raises(OmniClientError, match="supports T2VA/FL2VA requests only"):
        pipeline._resolve_task("ref2va", {}, has_turbo_lora=True)
    pipeline.partition = "ref2va"
    pipeline.supported_tasks = frozenset({"ref2va"})
    assert pipeline._resolve_task(None, {"image": object()}) == "ref2va"

    pipeline.partition = "combined"
    pipeline.supported_tasks = frozenset({"t2va", "fl2va", "ref2va"})
    assert pipeline._transformer_for_task("fl2va") is pipeline.transformer
    assert pipeline._transformer_for_task("ref2va") is pipeline.transformers_ref


@pytest.mark.parametrize(
    (
        "task_type",
        "expected_partition",
        "expected_dits",
        "component_partition",
        "source_partitions",
        "expected_tasks",
    ),
    [
        (None, "combined", 2, "FL2VA", ["FL2VA", "Ref2VA"], {"t2va", "fl2va", "ref2va"}),
        ("fl2va", "fl2va", 1, "FL2VA", ["FL2VA"], {"t2va", "fl2va"}),
        ("ref2va", "ref2va", 1, "Ref2VA", ["Ref2VA"], {"ref2va"}),
    ],
)
def test_pipeline_loads_task_selected_dits_and_shared_components_once(
    monkeypatch,
    tmp_path,
    task_type,
    expected_partition,
    expected_dits,
    component_partition,
    source_partitions,
    expected_tasks,
):
    from vllm_omni.diffusion.data import (
        DiffusionParallelConfig,
        OmniDiffusionConfig,
    )
    from vllm_omni.diffusion.models.minimax_h3 import (
        pipeline_minimax_h3 as pipeline_module,
    )

    _write_partition_index(
        tmp_path / "FL2VA",
        partition="fl2va",
        tasks=["t2va", "fl2va"],
    )
    _write_partition_index(
        tmp_path / "Ref2VA",
        partition="ref2va",
        tasks=["ref2va"],
    )
    for partition_name in ("FL2VA", "Ref2VA"):
        for component in (
            "transformer",
            "tokenizer",
            "processor",
            "text_encoder",
            "video_vae",
            "audio_vae",
        ):
            (tmp_path / partition_name / component).mkdir()

    created: dict[str, list[Any]] = {"dit": [], "text_encoder": [], "video_vae": [], "audio_vae": []}

    class FakeModule(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class FakeDiT(FakeModule):
        def __init__(self, *args, **kwargs):
            super().__init__()
            created["dit"].append((args, kwargs))

    def component_factory(name):
        def create(path, *args, **kwargs):
            created[name].append(str(path))
            return FakeModule()

        return create

    tokenizer_calls: list[Any] = []
    processor_calls: list[Any] = []
    tokenizer_cls = Mock(spec=pipeline_module.Qwen2TokenizerFast)
    tokenizer_cls.from_pretrained.side_effect = lambda *args, **kwargs: _append_and_return(
        tokenizer_calls, (args, kwargs), object()
    )
    processor_cls = Mock(spec=pipeline_module.Qwen3VLProcessor)
    processor_cls.from_pretrained.side_effect = lambda *args, **kwargs: _append_and_return(
        processor_calls, (args, kwargs), object()
    )
    monkeypatch.setattr(pipeline_module, "MiniMaxH3DiTModel", FakeDiT)
    monkeypatch.setattr(
        pipeline_module,
        "MiniMaxH3Qwen3VLEncoder",
        component_factory("text_encoder"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "MiniMaxH3VideoVAE",
        component_factory("video_vae"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "MiniMaxH3AudioVAE",
        component_factory("audio_vae"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "Qwen2TokenizerFast",
        tokenizer_cls,
    )
    monkeypatch.setattr(
        pipeline_module,
        "Qwen3VLProcessor",
        processor_cls,
    )
    monkeypatch.setattr(
        pipeline_module,
        "_dit_rank_world",
        lambda: (None, 0, 1),
    )
    monkeypatch.setattr(
        pipeline_module,
        "get_local_device",
        lambda: torch.device("cpu"),
    )
    download_calls = []

    def fake_download(**kwargs):
        download_calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(
        pipeline_module,
        "download_weights_from_hf_specific",
        fake_download,
    )

    od_config = OmniDiffusionConfig(
        model="MiniMaxAI/MiniMax-H3",
        revision=None,
        task_type=task_type,
        quantization_config=None,
        parallel_config=DiffusionParallelConfig(
            cfg_parallel_size=1,
            text_encoder_tp_size=1,
        ),
    )
    pipeline = pipeline_module.MiniMaxH3Pipeline(od_config=od_config)

    assert pipeline.partition == expected_partition
    assert pipeline.supported_tasks == expected_tasks
    assert len(created["dit"]) == expected_dits
    component_path = tmp_path / component_partition
    assert created["text_encoder"] == [str(component_path / "text_encoder")]
    assert created["video_vae"] == [str(component_path / "video_vae")]
    assert created["audio_vae"] == [str(component_path / "audio_vae")]
    assert len(tokenizer_calls) == 1
    assert len(processor_calls) == 1
    expected_dit_modules = ["transformer", "transformers_ref"] if expected_dits == 2 else ["transformer"]
    assert pipeline._dit_modules == expected_dit_modules
    assert hasattr(pipeline, "transformers_ref") is (expected_dits == 2)
    if expected_partition == "ref2va":
        assert pipeline._transformer_for_task("ref2va") is pipeline.transformer
    assert [source.model_or_path for source in pipeline.weights_sources] == [
        *(str(tmp_path / partition_name) for partition_name in source_partitions),
        str(component_path),
    ]
    assert [source.subfolder for source in pipeline.weights_sources] == [
        *("transformer" for _ in source_partitions),
        "text_encoder",
    ]
    expected_prefixes = ["transformer."]
    if expected_dits == 2:
        expected_prefixes.append("transformers_ref.")
    expected_prefixes.append("text_encoder.")
    assert [source.prefix for source in pipeline.weights_sources] == expected_prefixes
    expected_patterns = (
        pipeline_module.MINIMAX_H3_DOWNLOAD_PATTERNS
        if expected_partition == "combined"
        else pipeline_module.MINIMAX_H3_TASK_DOWNLOAD_PATTERNS[expected_partition]
    )
    assert download_calls == [
        {
            "model_name_or_path": "MiniMaxAI/MiniMax-H3",
            "cache_dir": None,
            "allow_patterns": expected_patterns,
            "revision": None,
            "require_all": True,
        }
    ]


def test_combined_weight_loader_routes_each_contiguous_partition():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    class FakeTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.loaded = []
            self.post_load_calls = 0

        def load_weights(self, weights):
            self.loaded = [name for name, _ in weights]
            return set(self.loaded)

        def post_load_weights(self):
            self.post_load_calls += 1

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.transformer = FakeTransformer()
    pipeline.transformers_ref = FakeTransformer()
    pipeline.text_encoder = torch.nn.Identity()
    pipeline.video_vae = torch.nn.Identity()
    pipeline.audio_vae = torch.nn.Identity()
    loaded = pipeline.load_weights(
        iter(
            [
                ("transformer.a", torch.ones(1)),
                ("transformer.b", torch.ones(1)),
                ("transformers_ref.a", torch.ones(1)),
            ]
        )
    )

    assert pipeline.transformer.loaded == ["a", "b"]
    assert pipeline.transformers_ref.loaded == ["a"]
    assert pipeline.transformer.post_load_calls == 1
    assert pipeline.transformers_ref.post_load_calls == 1
    assert loaded == {"transformer.a", "transformer.b", "transformers_ref.a"}


def test_dlo_offload_plan_includes_token_refiner():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    assert MiniMaxH3Pipeline._offload_plan.offload_submodules == {"token_refiner": "blocks"}
    assert MiniMaxH3Pipeline._offload_plan.resident_dit_paths == frozenset({"transformer"})


def test_joint_postprocess_is_multiprocessing_picklable():
    from vllm_omni.diffusion.models.minimax_h3 import (
        get_minimax_h3_post_process_func,
    )

    postprocess = get_minimax_h3_post_process_func(SimpleNamespace())
    postprocess = ForkingPickler.loads(ForkingPickler.dumps(postprocess))
    video = torch.linspace(0, 1, 2 * 3 * 2 * 4 * 5).reshape(2, 3, 2, 4, 5)
    audio = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6)

    result = postprocess((video, audio), output_type="np")

    assert isinstance(result["video"], list)
    assert result["video"][0].shape == (2, 4, 5, 3)
    np.testing.assert_array_equal(result["audio"], audio.numpy())
    assert result["audio_sample_rate"] == 32000
    assert result["fps"] == 24


def test_cfg_parallel_is_rejected_for_distilled_checkpoint():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    od_config = SimpleNamespace(
        parallel_config=SimpleNamespace(cfg_parallel_size=2),
    )
    with pytest.raises(ValueError, match="CFG-distilled"):
        MiniMaxH3Pipeline(od_config=od_config)


def test_shape_contract_matches_three_reference_tasks():
    from vllm_omni.diffusion.models.minimax_h3.time_request import (
        MINIMAX_H3_SHAPE_PLANNER,
        minimax_h3_align_frame_count,
    )

    assert minimax_h3_align_frame_count(round(8.7 * 24)) == 209
    assert MINIMAX_H3_SHAPE_PLANNER.video_latent_t(209) == 62
    assert MINIMAX_H3_SHAPE_PLANNER.audio_latent_t(8.7) == 348
    assert minimax_h3_align_frame_count(round(5.0 * 24)) == 124
    assert MINIMAX_H3_SHAPE_PLANNER.video_latent_t(124) == 37
    assert MINIMAX_H3_SHAPE_PLANNER.audio_latent_t(5.0) == 200
    assert MINIMAX_H3_SHAPE_PLANNER.frame_count_from_video_latent_t(62) == 209


def test_shifted_sigma_schedule_matches_reference_values():
    from vllm_omni.diffusion.models.minimax_h3.time_request import (
        minimax_h3_time_shift_sigmas,
    )

    sigmas = minimax_h3_time_shift_sigmas(num_steps=5, shift_scale=12.0)

    assert sigmas == pytest.approx(
        [1.0, 0.9729729891, 0.9230769277, 0.8000000119, 0.0],
        abs=1e-7,
    )


def test_base_schedule_overrides_the_uniform_sigma_positions():
    from vllm_omni.diffusion.models.minimax_h3.time_request import (
        minimax_h3_time_shift_sigmas,
    )

    base_schedule = [1.0, 0.7, 0.4, 0.15, 0.0]

    video = minimax_h3_time_shift_sigmas(
        num_steps=len(base_schedule),
        shift_scale=12.0,
        base_schedule=base_schedule,
    )
    audio = minimax_h3_time_shift_sigmas(
        num_steps=len(base_schedule),
        shift_scale=3.0,
        base_schedule=base_schedule,
    )

    assert video == pytest.approx([1.0, 0.9655172, 0.8888889, 0.6792453, 0.0], abs=1e-6)
    assert audio == pytest.approx([1.0, 0.875, 0.6666667, 0.3461539, 0.0], abs=1e-6)
    assert video != pytest.approx(
        minimax_h3_time_shift_sigmas(num_steps=len(base_schedule), shift_scale=12.0),
        abs=1e-6,
    )


@pytest.mark.parametrize(
    "base_schedule",
    [
        [0.9, 0.5, 0.0],
        [1.0, 0.5, 0.1],
        [1.0, 0.5, 0.5, 0.0],
        [1.0],
        [],
        [1.0, float("nan"), 0.4, 0.0],
        [1.0, float("inf"), 0.4, 0.0],
    ],
)
def test_base_schedule_rejects_malformed_positions(base_schedule):
    from vllm_omni.diffusion.models.minimax_h3.time_request import (
        minimax_h3_time_shift_sigmas,
    )

    with pytest.raises(ValueError):
        minimax_h3_time_shift_sigmas(
            num_steps=max(len(base_schedule), 1),
            shift_scale=12.0,
            base_schedule=base_schedule,
        )


def _distilled_pipeline(diffuse_calls, base_schedule_by_partition):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.sched import DMD2SigmaSchedule

    schedules = {
        partition: None if positions is None else DMD2SigmaSchedule.from_positions(positions)
        for partition, positions in base_schedule_by_partition.items()
    }
    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.partition = "combined"
    pipeline.supported_tasks = frozenset({"t2va", "fl2va", "ref2va"})
    pipeline.default_video_shift = 12.0
    pipeline.default_audio_shift = 3.0
    pipeline.device = torch.device("cpu")
    pipeline.od_config = SimpleNamespace()
    pipeline._base_schedule_by_partition = schedules
    pipeline._quality_policy = Mock()
    pipeline._quality_policy.resolve.return_value = SimpleNamespace(cache_dit=None)
    pipeline._cache_dit_runtime = SimpleNamespace(prepare=lambda spec: None)
    pipeline.encode_prompt = Mock(
        return_value=(
            torch.ones(1, 2),
            torch.ones(1, dtype=torch.long),
        )
    )

    def diffuse(**kwargs):
        diffuse_calls.append(kwargs)
        return torch.zeros(1), torch.zeros(1)

    pipeline.diffuse = diffuse
    pipeline.decode = Mock(return_value=(torch.zeros(1), torch.zeros(1)))
    return pipeline


def _t2va_batch(num_inference_steps=None):
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    sampling = OmniDiffusionSamplingParams(
        quality="lossless",
        width=1344,
        height=768,
        fps=24,
        num_frames=124,
        num_inference_steps=num_inference_steps,
        extra_args={"task": "t2va", "aspect_ratio": "16:9"},
    )
    return DiffusionRequestBatch(
        [
            OmniDiffusionRequest(
                prompt="distilled schedule",
                sampling_params=sampling,
                request_id="distilled",
            )
        ]
    )


def test_base_schedule_is_scoped_to_the_serving_partition():
    """A distilled FL2VA must not drag a regular Ref2VA onto its step count."""
    distilled = [1.0, 0.7, 0.4, 0.15, 0.0]
    pipeline = _distilled_pipeline([], {"fl2va": distilled, "ref2va": None})

    assert pipeline._base_schedule_for_task("t2va").base_schedule == tuple(distilled)
    assert pipeline._base_schedule_for_task("fl2va").base_schedule == tuple(distilled)
    assert pipeline._base_schedule_for_task("ref2va") is None


def test_partially_constructed_pipeline_falls_back_to_the_uniform_schedule():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)

    assert pipeline._base_schedule_for_task("t2va") is None


def test_distilled_forward_reports_denoising_steps_not_sigma_boundaries():
    diffuse_calls: list[dict[str, Any]] = []
    base_schedule = [1.0, 0.7, 0.4, 0.15, 0.0]
    pipeline = _distilled_pipeline(diffuse_calls, {"fl2va": base_schedule, "ref2va": None})

    pipeline.forward(_t2va_batch())

    assert diffuse_calls[0]["base_schedule"] == tuple(base_schedule)
    # Five boundaries describe four denoising steps.
    assert diffuse_calls[0]["num_steps"] == 4
    pipeline._quality_policy.resolve.assert_called_once_with(
        quality="lossless",
        num_inference_steps=4,
        extra_args={"task": "t2va", "aspect_ratio": "16:9"},
    )


def test_distilled_forward_accepts_the_matching_explicit_step_count():
    diffuse_calls: list[dict[str, Any]] = []
    pipeline = _distilled_pipeline(diffuse_calls, {"fl2va": [1.0, 0.7, 0.4, 0.15, 0.0], "ref2va": None})

    pipeline.forward(_t2va_batch(num_inference_steps=4))

    assert diffuse_calls[0]["num_steps"] == 4


def test_distilled_forward_rejects_a_mismatched_explicit_step_count():
    from vllm_omni.errors import OmniClientError

    pipeline = _distilled_pipeline([], {"fl2va": [1.0, 0.7, 0.4, 0.15, 0.0], "ref2va": None})

    with pytest.raises(OmniClientError, match="must be 4 or omitted"):
        pipeline.forward(_t2va_batch(num_inference_steps=50))


def test_absent_base_schedule_key_differs_from_an_empty_list():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _read_base_schedule,
    )

    assert _read_base_schedule({}) is None
    assert _read_base_schedule({"base_schedule": [1.0, 0.5, 0.0]}).base_schedule == (1.0, 0.5, 0.0)
    with pytest.raises(ValueError):
        _read_base_schedule({"base_schedule": []})


def test_cudnn_packed_attention_uses_python_length_without_padding_mask():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3Attention,
    )

    class FakeBackend:
        supports_prefix_kv_slicing = True

        @classmethod
        def supports_packed_mask_free(cls) -> bool:
            return False

    class FakeAttention(torch.nn.Module):
        attn_backend = FakeBackend

        def __init__(self):
            super().__init__()
            self.metadata = None

        def forward(self, query, key, value, metadata):
            self.metadata = metadata
            return query

    attention = object.__new__(MiniMaxH3Attention)
    torch.nn.Module.__init__(attention)
    attention.attention = FakeAttention()
    # Model-side Ulysses shards rows before Attention gathers them again.
    # The Python packed_total must therefore remain global even though q is
    # local here.
    q = torch.randn(4, 2, 4)
    cu_seqlens = torch.tensor([0, 5, 8], dtype=torch.int32)

    output = attention._run_packed_attention(
        q,
        q,
        q,
        cu_seqlens=cu_seqlens,
        max_seqlen=5,
        packed_total=8,
    )

    assert output.shape == q.shape
    metadata = attention.attention.metadata
    assert metadata is not None
    assert metadata.attn_mask is None
    assert metadata.extra["valid_kv_length"] == 5


def test_packed_attention_skips_mask_for_packed_mask_free_backend():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3Attention,
    )

    class FakeBackend:
        supports_prefix_kv_slicing = False

        @classmethod
        def supports_packed_mask_free(cls) -> bool:
            return True

    class FakeAttention(torch.nn.Module):
        attn_backend = FakeBackend

        def __init__(self):
            super().__init__()
            self.metadata = None

        def forward(self, query, key, value, metadata):
            self.metadata = metadata
            return query

    attention = object.__new__(MiniMaxH3Attention)
    torch.nn.Module.__init__(attention)
    attention.attention = FakeAttention()
    q = torch.randn(8, 2, 4)

    attention._run_packed_attention(
        q,
        q,
        q,
        cu_seqlens=torch.tensor([0, 5, 8], dtype=torch.int32),
        max_seqlen=5,
        packed_total=8,
    )

    metadata = attention.attention.metadata
    assert metadata is not None
    assert metadata.attn_mask is None
    assert metadata.extra["valid_kv_length"] == 5
    assert metadata.extra["npu_attn_varlen"] is True
    packed_padding = metadata.packed_padding
    assert packed_padding is not None
    assert packed_padding.q_length == 5
    assert packed_padding.kv_length == 5
    assert packed_padding.cu_seqlens_q.tolist() == [0, 5]
    assert packed_padding.cu_seqlens_k.tolist() == [0, 5]


def test_packed_attention_keeps_padding_mask_for_other_backends():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3Attention,
    )

    class FakeBackend:
        supports_prefix_kv_slicing = False

        @classmethod
        def supports_packed_mask_free(cls) -> bool:
            return False

    class FakeAttention(torch.nn.Module):
        attn_backend = FakeBackend

        def __init__(self):
            super().__init__()
            self.metadata = None

        def forward(self, query, key, value, metadata):
            self.metadata = metadata
            return query

    attention = object.__new__(MiniMaxH3Attention)
    torch.nn.Module.__init__(attention)
    attention.attention = FakeAttention()
    q = torch.randn(8, 2, 4)

    attention._run_packed_attention(
        q,
        q,
        q,
        cu_seqlens=torch.tensor([0, 5, 8], dtype=torch.int32),
        max_seqlen=5,
        packed_total=8,
    )

    metadata = attention.attention.metadata
    assert metadata is not None
    assert torch.equal(
        metadata.attn_mask,
        torch.tensor([[True, True, True, True, True, False, False, False]]),
    )


def _fake_packed_attention(
    backend_name: str,
    *,
    prefix_kv_slicing: bool = False,
    supports_multi_doc: bool = False,
):
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
        MiniMaxH3Attention,
    )

    class FakeBackend:
        supports_prefix_kv_slicing = prefix_kv_slicing

        @staticmethod
        def get_name() -> str:
            return backend_name

        @classmethod
        def supports_multi_doc_packed_varlen(cls) -> bool:
            return supports_multi_doc

    class FakeAttention(torch.nn.Module):
        attn_backend = FakeBackend

        def __init__(self):
            super().__init__()
            self.metadata = None

        def forward(self, query, key, value, metadata):
            self.metadata = metadata
            return query

    attention = object.__new__(MiniMaxH3Attention)
    torch.nn.Module.__init__(attention)
    attention.attention = FakeAttention()
    return attention


def test_packed_attention_drops_prefix_semantics_for_a_co_batched_layout():
    """A batch's valid rows are block-diagonal, so no prefix can describe them."""
    attention = _fake_packed_attention("FLASH_ATTN", supports_multi_doc=True)
    q = torch.randn(12, 2, 4)

    attention._run_packed_attention(
        q,
        q,
        q,
        # Two requests: rows + padding tail each.
        cu_seqlens=torch.tensor([0, 5, 6, 11, 12], dtype=torch.int32),
        max_seqlen=5,
        packed_total=12,
        num_requests=2,
    )

    assert attention.attention.metadata.attn_mask is None
    assert attention.attention.metadata.extra["valid_kv_length"] == 12


@pytest.mark.parametrize("backend_name", ["TORCH_SDPA", "FLASH_ATTN"])
def test_packed_attention_rejects_backends_without_multi_doc_capability(backend_name):
    """Backend names alone must not allow cross-request attention."""
    attention = _fake_packed_attention(backend_name, supports_multi_doc=False)
    q = torch.randn(12, 2, 4)

    with pytest.raises(ValueError, match=backend_name):
        attention._run_packed_attention(
            q,
            q,
            q,
            cu_seqlens=torch.tensor([0, 5, 6, 11, 12], dtype=torch.int32),
            max_seqlen=5,
            packed_total=12,
            num_requests=2,
        )


def test_reference_image_resize_contract():
    from PIL import Image

    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _reference_image_shape,
    )

    assert _reference_image_shape(Image.new("RGB", (1080, 1440))) == (
        2048,
        2720,
    )
    with pytest.raises(ValueError, match="aspect ratio"):
        _reference_image_shape(Image.new("RGB", (100, 501)))


def test_fl2va_supports_first_last_and_explicit_frame_index_contracts():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _resolve_fl2va_keyframe_indices,
    )

    assert _resolve_fl2va_keyframe_indices({}, 1) == [0]
    assert _resolve_fl2va_keyframe_indices({}, 2) == [0, -1]
    assert _resolve_fl2va_keyframe_indices({"frame_index": -1}, 1) == [-1]
    assert _resolve_fl2va_keyframe_indices({"frame_indices": [0, -1]}, 2) == [0, -1]
    with pytest.raises(ValueError, match="frame_indices"):
        _resolve_fl2va_keyframe_indices({"frame_indices": [0, 1]}, 2)


def test_minimax_h3_uses_the_official_output_canvas_policy():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _resolve_output_canvas,
    )

    assert _resolve_output_canvas(21 / 9, 768) == (672, 1536)
    assert _resolve_output_canvas(16 / 9, 768) == (768, 1344)
    assert _resolve_output_canvas(9 / 16, 768) == (1344, 768)
    with pytest.raises(ValueError, match="short_edge"):
        _resolve_output_canvas(16 / 9, 720)


def test_minimax_h3_accepts_sglang_auto_aspect_ratio_alias():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    sampling = SimpleNamespace(
        fps=24,
        num_frames=1,
        height=None,
        width=None,
        extra_args={"duration": 5.0, "aspect_ratio": "auto"},
    )
    height, width, *_ = pipeline._resolve_shape("ref2va", sampling, None)
    assert (height, width) == (768, 1344)


def test_minimax_h3_advertises_the_official_ref2va_image_limit():
    from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata

    assert get_diffusion_model_metadata("MiniMaxH3Pipeline").max_multimodal_image_inputs == 9


def test_text_attention_routes_local_gqa_heads_through_sdpa_helper(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import encoder as encoder_module

    class FakeEncoderGroup:
        rank_in_group = 0
        world_size = 2

        def all_reduce(self, tensor):
            del tensor

    config = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=2,
        rms_norm_eps=1e-6,
    )
    with (
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=0),
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=1),
    ):
        attention = encoder_module.MiniMaxH3Qwen3VLTextAttention(FakeEncoderGroup(), config, torch.float32)
    # This CPU contract test only needs projection shapes and head reshaping.
    # Avoid dispatching platform-specific norm and linear kernels (for example,
    # rocm_unquantized_gemm when the test suite is collected on ROCm).
    attention.q_norm = nn.Identity()
    attention.k_norm = nn.Identity()
    with torch.no_grad():
        attention.qkv_proj.weight.zero_()
    attention.o_proj = nn.Linear(
        attention.qkv_proj.local_num_heads * attention.head_dim,
        config.hidden_size,
        bias=False,
    )
    attn_call = {}

    def fake_attention(query, key, value):
        attn_call.update(query_shape=query.shape, key_shape=key.shape, value_shape=value.shape)
        return query

    monkeypatch.setattr(encoder_module, "_scaled_dot_product_attention", fake_attention)
    hidden_states = torch.randn(1, 3, config.hidden_size)
    cos = torch.ones(1, 3, config.head_dim)
    sin = torch.zeros_like(cos)

    output = attention(hidden_states, (cos, sin))

    assert attn_call["query_shape"] == (1, config.num_attention_heads // 2, 3, config.head_dim)
    assert attn_call["key_shape"] == (1, config.num_key_value_heads // 2, 3, config.head_dim)
    assert attn_call["value_shape"] == (1, config.num_key_value_heads // 2, 3, config.head_dim)
    assert output.shape == hidden_states.shape


def test_text_attention_sdpa_helper_preserves_expanded_kv_fallback(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import encoder as encoder_module

    sdpa_call = {}

    def fake_sdpa(query, key, value, **kwargs):
        sdpa_call.update(query_shape=query.shape, key_shape=key.shape, value_shape=value.shape, kwargs=kwargs)
        return query

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)
    output = encoder_module._scaled_dot_product_attention(
        torch.randn(1, 4, 3, 8),
        torch.randn(1, 2, 3, 8),
        torch.randn(1, 2, 3, 8),
    )

    assert sdpa_call["query_shape"] == sdpa_call["key_shape"] == sdpa_call["value_shape"] == (1, 4, 3, 8)
    assert sdpa_call["kwargs"] == {"dropout_p": 0.0, "is_causal": True}
    assert output.shape == (1, 4, 3, 8)


def test_encoder_forward_uses_hook_compatible_encode_entrypoint():
    from vllm_omni.diffusion.models.minimax_h3.encoder import (
        MiniMaxH3Qwen3VLEncoder,
    )

    encoder = object.__new__(MiniMaxH3Qwen3VLEncoder)
    torch.nn.Module.__init__(encoder)
    expected = torch.ones(2, 3)
    encoder.encode_ids = Mock(return_value=expected)
    input_ids = torch.tensor([1, 2])
    pixel_values = torch.ones(1, 4)
    image_grid_thw = torch.tensor([[1, 1, 1]])

    actual = encoder(
        input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )

    assert actual is expected
    encoder.encode_ids.assert_called_once_with(
        input_ids,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )


def test_encoder_forward_forwards_video_inputs():
    from vllm_omni.diffusion.models.minimax_h3.encoder import (
        MiniMaxH3Qwen3VLEncoder,
    )

    encoder = object.__new__(MiniMaxH3Qwen3VLEncoder)
    torch.nn.Module.__init__(encoder)
    expected = torch.ones(2, 3)
    encoder.encode_ids = Mock(return_value=expected)
    input_ids = torch.tensor([1, 2])
    pixel_values_videos = torch.ones(1, 4)
    video_grid_thw = torch.tensor([[2, 1, 1]])

    actual = encoder(
        input_ids,
        pixel_values_videos=pixel_values_videos,
        video_grid_thw=video_grid_thw,
    )

    assert actual is expected
    encoder.encode_ids.assert_called_once_with(
        input_ids,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=pixel_values_videos,
        video_grid_thw=video_grid_thw,
    )


def test_reference_video_shape_uses_h3_adapt_shape_policy():
    from vllm_omni.diffusion.models.minimax_h3.reference_video import (
        _reference_video_shape,
    )

    assert _reference_video_shape(1280, 720) == (1344, 768)
    assert _reference_video_shape(3844, 2160) == (1344, 768)


def test_text_encoder_stub_constructs_without_group_or_weights():
    from vllm_omni.diffusion.models.minimax_h3.encoder import (
        MiniMaxH3Qwen3VLEncoder,
    )

    encoder = MiniMaxH3Qwen3VLEncoder(
        "/nonexistent/text_encoder",
        device=torch.device("cpu"),
        load_model=False,
        encoder_group=None,
    )
    assert not encoder.is_loaded
    assert encoder.tp_size == 1
    # The stub has no parameters, so it never contributes to the runner's
    # strict missing-parameter check on non-encoder ranks.
    assert list(encoder.named_parameters()) == []


def test_global_quant_config_is_shared_by_dit_and_encoder():
    from vllm_omni.quantization import build_quant_config, resolve_component_quant_config

    global_config = build_quant_config("fp8")

    assert resolve_component_quant_config(global_config, "transformer") is global_config
    assert resolve_component_quant_config(global_config, "text_encoder") is global_config


def test_minimax_h3_fp8_component_selection():
    from vllm_omni.quantization import build_quant_config, resolve_component_quant_config

    dit_only = build_quant_config({"transformer": {"method": "fp8"}})
    encoder_only = build_quant_config({"text_encoder": {"method": "fp8"}})

    assert resolve_component_quant_config(dit_only, "transformer").get_name() == "fp8"
    assert resolve_component_quant_config(dit_only, "text_encoder") is None
    assert resolve_component_quant_config(encoder_only, "transformer") is None
    assert resolve_component_quant_config(encoder_only, "text_encoder").get_name() == "fp8"


def test_text_encoder_rejects_serialized_fp8():
    from vllm_omni.diffusion.models.minimax_h3.encoder import _validate_encoder_quant_config

    config = SimpleNamespace(
        get_name=lambda: "fp8",
        is_checkpoint_fp8_serialized=True,
    )
    with pytest.raises(ValueError, match="online FP8 only"):
        _validate_encoder_quant_config(config)


def test_text_encoder_linear_delegates_quantization_to_vllm_factory():
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    import vllm_omni.diffusion.models.minimax_h3.encoder as encoder_module

    group = SimpleNamespace(rank_in_group=0, world_size=1)
    method = UnquantizedLinearMethod()
    quant_config = Mock()
    quant_config.get_quant_method.return_value = method
    prefix = "text_encoder.text_model.layers.0.mlp.gate_up_proj"

    with (
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=0),
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=1),
    ):
        linear = encoder_module.MiniMaxH3Qwen3VLMergedColumnParallelLinear(
            group,
            input_size=4,
            intermediate_size=8,
            dtype=torch.bfloat16,
            quant_config=quant_config,
            prefix=prefix,
        )

    assert linear.quant_method is method
    quant_config.get_quant_method.assert_called_once_with(linear, prefix=prefix)


def test_no_offload_keeps_text_encoder_resident():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = SimpleNamespace(
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
    )
    pipeline.text_encoder = Mock()
    expected = torch.ones(2, 3)
    pipeline.text_encoder.encode_ids.return_value = expected
    input_ids = torch.tensor([1, 2])

    actual = pipeline._encode_text_hidden(input_ids, {})

    assert actual is expected
    pipeline.text_encoder.load_to_device.assert_called_once_with()
    pipeline.text_encoder.offload_to_cpu.assert_not_called()


def test_model_offload_uses_hooked_text_encoder_call():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = SimpleNamespace(
        enable_cpu_offload=True,
        enable_layerwise_offload=False,
    )
    expected = torch.ones(2, 3)
    pipeline.text_encoder = Mock(return_value=expected)
    pipeline._model_cpu_offload_modules = [pipeline.text_encoder]
    input_ids = torch.tensor([1, 2])
    vision_kwargs = {"pixel_values": torch.ones(1, 4)}

    actual = pipeline._encode_text_hidden(input_ids, vision_kwargs)

    assert actual is expected
    pipeline.text_encoder.assert_called_once_with(input_ids, **vision_kwargs)
    pipeline.text_encoder.load_to_device.assert_not_called()


@pytest.mark.parametrize(
    "offload_flag",
    ["enable_layerwise_offload", "enable_distributed_layerwise_offload"],
)
def test_layerwise_offload_releases_text_encoder(offload_flag):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = SimpleNamespace(
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
        enable_distributed_layerwise_offload=False,
    )
    setattr(pipeline.od_config, offload_flag, True)
    pipeline.text_encoder = Mock()
    expected = torch.ones(2, 3)
    pipeline.text_encoder.encode_ids.return_value = expected

    actual = pipeline._encode_text_hidden(torch.tensor([1, 2]), {})

    assert actual is expected
    pipeline.text_encoder.load_to_device.assert_called_once_with()
    pipeline.text_encoder.offload_to_cpu.assert_called_once_with()


def test_distributed_layerwise_offload_releases_text_encoder():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = SimpleNamespace(
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
        enable_distributed_layerwise_offload=True,
    )
    pipeline.text_encoder = Mock()
    expected = torch.ones(2, 3)
    pipeline.text_encoder.encode_ids.return_value = expected

    actual = pipeline._encode_text_hidden(torch.tensor([1, 2]), {})

    assert actual is expected
    pipeline.text_encoder.load_to_device.assert_called_once_with()
    pipeline.text_encoder.offload_to_cpu.assert_called_once_with()


def test_distributed_layerwise_offload_stages_vae_component():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = SimpleNamespace(
        enable_layerwise_offload=False,
        enable_distributed_layerwise_offload=True,
    )
    component = Mock()

    with pipeline._component_on_device(component):
        component.load_to_device.assert_called_once_with()
        component.offload_to_cpu.assert_not_called()

    component.offload_to_cpu.assert_called_once_with()


def test_distributed_layerwise_resident_blocks_are_stage_scoped():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    controller = Mock()
    pipeline._dlo_residency_controller = controller

    with pipeline._resident_dit_layers_on_device():
        controller.load_resident_layers.assert_called_once_with()
        controller.offload_resident_layers.assert_not_called()

    controller.offload_resident_layers.assert_called_once_with()


def test_distributed_layerwise_resident_blocks_release_on_failure():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    controller = Mock()
    pipeline._dlo_residency_controller = controller

    with pytest.raises(RuntimeError, match="denoise failed"):
        with pipeline._resident_dit_layers_on_device():
            raise RuntimeError("denoise failed")

    controller.load_resident_layers.assert_called_once_with()
    controller.offload_resident_layers.assert_called_once_with()


def test_distributed_layerwise_resident_blocks_can_be_skipped():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    controller = Mock()
    pipeline._dlo_residency_controller = controller

    with pipeline._resident_dit_layers_on_device(enabled=False):
        pass

    controller.load_resident_layers.assert_not_called()
    controller.offload_resident_layers.assert_not_called()


def test_encoder_layerwise_offload_keeps_tp_blocks_rank_local(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3.encoder import (
        MiniMaxH3Qwen3VLEncoder,
    )

    class Stack(torch.nn.Module):
        def __init__(self, count):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(count)])

    class TextStack(torch.nn.Module):
        def __init__(self, count):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(count)])

    hooks = []

    def fake_apply(block, next_block, device, stream, pin_memory):
        hook = SimpleNamespace(
            block=block,
            next_block=next_block,
            device=device,
            stream=stream,
            pin_memory=pin_memory,
            _prev_hook=None,
            offload_layer=Mock(),
        )
        hooks.append(hook)
        return hook

    monkeypatch.setattr(
        "vllm_omni.diffusion.offloader.layerwise_backend.apply_block_hook",
        fake_apply,
    )
    monkeypatch.setattr(
        "vllm_omni.platforms.current_omni_platform.Stream",
        Mock(return_value="copy-stream"),
    )
    encoder = object.__new__(MiniMaxH3Qwen3VLEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.device_target = torch.device("cpu")
    encoder.vision = Stack(2)
    encoder.text_model = TextStack(3)

    encoder.enable_omni_layerwise_offload(pin_memory=False)

    assert len(hooks) == 5
    assert all(hook._prev_hook is not None for hook in hooks)
    assert all(hook.device == torch.device("cpu") for hook in hooks)


def test_video_vae_keeps_reference_fp32_weights(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import vae as vae_module

    class FakeRemote(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Linear(1, 1).half()

    monkeypatch.setattr(
        vae_module,
        "_load_component_config",
        lambda _path: {
            "latent_channels": 1,
            "latents_mean": [0.0],
            "latents_std": [1.0],
        },
    )
    monkeypatch.setattr(
        vae_module,
        "_load_remote_component",
        lambda _path, _config: FakeRemote(),
    )

    video_vae = vae_module.MiniMaxH3VideoVAE(
        "unused",
        device=torch.device("cpu"),
    )

    assert next(video_vae.parameters()).dtype == torch.float32


def test_video_vae_can_load_on_cpu_for_staged_gpu_residency(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import vae as vae_module

    class FakeRemote(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Linear(1, 1)

    monkeypatch.setattr(
        vae_module,
        "_load_component_config",
        lambda _path: {
            "latent_channels": 1,
            "latents_mean": [0.0],
            "latents_std": [1.0],
        },
    )
    monkeypatch.setattr(
        vae_module,
        "_load_remote_component",
        lambda _path, _config: FakeRemote(),
    )

    video_vae = vae_module.MiniMaxH3VideoVAE(
        "unused",
        device=torch.device("meta"),
        load_device=torch.device("cpu"),
    )

    assert next(video_vae.parameters()).device.type == "cpu"
    assert video_vae._device_target.type == "meta"


def test_video_vae_uses_snapshot_stager_for_accelerator_target(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import vae as vae_module

    class FakeRemote(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Linear(1, 1)

    stager = Mock()
    stager_cls = Mock(return_value=stager)
    monkeypatch.setattr(
        vae_module,
        "_load_component_config",
        lambda _path: {
            "latent_channels": 1,
            "latents_mean": [0.0],
            "latents_std": [1.0],
        },
    )
    monkeypatch.setattr(
        vae_module,
        "_load_remote_component",
        lambda _path, _config: FakeRemote(),
    )
    monkeypatch.setattr(vae_module, "PinnedModuleStager", stager_cls)

    video_vae = vae_module.MiniMaxH3VideoVAE(
        "unused",
        device=torch.device("cuda"),
        load_device=torch.device("cpu"),
    )
    video_vae.load_to_device()
    video_vae.offload_to_cpu()

    stager_cls.assert_called_once_with(
        video_vae.remote,
        torch.device("cuda"),
        pin_memory=True,
    )
    stager.load.assert_called_once_with()
    stager.offload.assert_called_once_with()


def test_video_vae_encode_uses_configured_parallel_tiling():
    from vllm_omni.diffusion.models.minimax_h3.vae import (
        MiniMaxH3VideoVAE,
    )

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.parallel_tiling = True
            self.encode_calls = []

        def encode_videos(self, frames, *, use_fp16_latent):
            assert self.parallel_tiling
            self.encode_calls.append((frames, use_fp16_latent))
            return [torch.ones(1, 1, 2, 2, 2)]

    video_vae = object.__new__(MiniMaxH3VideoVAE)
    torch.nn.Module.__init__(video_vae)
    video_vae.model = FakeModel()
    video_vae.config_dict = {
        "latent_channels": 1,
        "latents_mean": [0.0],
        "latents_std": [1.0],
    }

    rows, shape = video_vae.encode_video("frames")

    assert video_vae.model.parallel_tiling
    assert video_vae.model.encode_calls == [("frames", True)]
    assert rows.shape == (2, 4)
    assert shape == (2, 2, 2)


def test_distributed_video_vae_encodes_references_sequentially(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3 import (
        pipeline_minimax_h3 as pipeline_module,
    )

    prepared = [
        {"prepared_path": "video-1.mp4"},
        {"prepared_path": "video-2.mp4"},
    ]

    class FakeVideoVAE:
        def __init__(self):
            self.calls = []

        def is_distributed_enabled(self):
            return True

        def encode_video(self, frames):
            self.calls.append(frames)
            index = len(self.calls)
            return torch.full((1, 2), index, dtype=torch.float32), (index, 2, 3)

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.video_vae = FakeVideoVAE()

    monkeypatch.setattr(
        pipeline_module,
        "_dit_rank_world",
        lambda: ("dit-group", 1, 4),
    )

    def fake_broadcast_object_list(values, *, src, group, device):
        assert values == [None]
        assert (src, group, device) == (0, "dit-group", torch.device("cpu"))
        values[0] = prepared

    monkeypatch.setattr(
        pipeline_module.dist,
        "broadcast_object_list",
        fake_broadcast_object_list,
    )
    monkeypatch.setattr(
        pipeline_module,
        "load_video_frames",
        lambda path: f"frames:{path}",
    )

    rows, shapes = pipeline._encode_video_conditions_resident(None, count=2)

    assert pipeline.video_vae.calls == [
        "frames:video-1.mp4",
        "frames:video-2.mp4",
    ]
    torch.testing.assert_close(
        rows,
        torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
    )
    assert shapes == [(1, 2, 3), (2, 2, 3)]


@pytest.mark.parametrize(
    ("case", "extra", "image_count", "expected"),
    [
        ("F1", {"frame_index": -1}, 1, [-1]),
        ("F2", {"frame_indices": [0, -1]}, 2, [0, -1]),
    ],
)
def test_f1_f2_official_fl2va_keyframe_matrix(case, extra, image_count, expected):
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _resolve_fl2va_keyframe_indices,
    )

    assert case in {"F1", "F2"}
    assert _resolve_fl2va_keyframe_indices(extra, image_count) == expected


@pytest.mark.parametrize(
    ("case", "counts"),
    [
        ("R1", (3, 0, 0)),
        ("R2", (1, 0, 0)),
        ("R3", (1, 1, 0)),
        ("R4", (0, 1, 1)),
        ("R5", (1, 1, 1)),
        ("R6", (1, 0, 2)),
    ],
)
def test_r1_r6_ref2va_reference_count_matrix(case, counts):
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _validate_ref2va_reference_counts,
    )

    assert case.startswith("R")
    _validate_ref2va_reference_counts(*counts)


def test_ref2va_reference_count_validation_preserves_client_error_metadata():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _validate_ref2va_reference_counts,
    )
    from vllm_omni.errors import OmniClientError

    with pytest.raises(OmniClientError, match="at least one image or video"):
        _validate_ref2va_reference_counts(0, 0, 0)


@pytest.mark.parametrize(
    ("case", "start_time", "expected_duration"),
    [
        ("R7", None, 10.0),
        ("R8", 4.0, 6.0),
    ],
)
def test_r7_r8_ref2va_video_segment_matrix(monkeypatch, tmp_path, case, start_time, expected_duration):
    from vllm_omni.diffusion.models.minimax_h3 import reference_video as reference_video_module

    source = tmp_path / f"{case}.mp4"
    source.touch()
    metadata = {
        "width": 1280,
        "height": 720,
        "fps": 24.0,
        "frame_count": 240,
        "duration": 10.0,
        "format_names": ("mp4",),
        "video_codec": "h264",
        "audio_codecs": (),
        "file_size": 1024,
    }
    transcode_calls: list[Any] = []
    monkeypatch.setattr(reference_video_module, "_probe_video", lambda _path: metadata)
    monkeypatch.setattr(
        reference_video_module,
        "_transcode_reference_video",
        lambda source, **kwargs: _append_and_return(transcode_calls, (source, kwargs), "prepared.mp4"),
    )

    prepared = reference_video_module.prepare_reference_videos(
        [str(source)],
        target_frame_count=209,
        workdir=str(tmp_path / "work"),
        start_time_seconds=start_time,
    )

    assert prepared[0]["duration_seconds"] == pytest.approx(expected_duration)
    assert prepared[0]["audio_duration_seconds"] == pytest.approx(min(expected_duration, 209 / 24.0))
    assert prepared[0]["start_time_seconds"] == pytest.approx(start_time or 0.0)
    assert transcode_calls[0][1]["duration_seconds"] == pytest.approx(expected_duration)
    assert transcode_calls[0][1]["target_frame_count"] == 209


def test_ref2va_qwen_sampling_uses_one_selective_decode(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import reference_video as reference_video_module

    decoded = np.arange(3 * 2 * 2 * 3, dtype=np.uint8).reshape(3, 2, 2, 3)
    decode_calls: list[Any] = []
    monkeypatch.setattr(
        reference_video_module,
        "_probe_video",
        lambda _path: {
            "frame_count": 25,
            "width": 2,
            "height": 2,
        },
    )
    monkeypatch.setattr(
        reference_video_module,
        "_decode_video_frames_ffmpeg",
        lambda path, **kwargs: _append_and_return(decode_calls, (path, kwargs), decoded),
    )

    sampled = reference_video_module.sample_reference_video_frames("prepared.mp4")

    assert decode_calls == [
        (
            "prepared.mp4",
            {
                "frame_count": 25,
                "indices": [0, 12, 24],
                "width": 2,
                "height": 2,
            },
        )
    ]
    for frame, expected in zip(sampled["frames"], decoded, strict=True):
        np.testing.assert_array_equal(frame, expected)
    assert sampled["block_timestamps"] == [0.25, 1.0]


@pytest.mark.parametrize(
    ("indices", "frame_count"),
    [([0, 12], 25), (None, 2)],
)
def test_ref2va_ffmpeg_decode_reads_exact_rgb_frames(monkeypatch, indices, frame_count):
    from vllm_omni.diffusion.models.minimax_h3 import reference_video as reference_video_module

    expected = np.arange(2 * 2 * 3 * 3, dtype=np.uint8).reshape(2, 2, 3, 3)
    observed = []

    def fake_run(command, **kwargs):
        observed.append((command, kwargs))
        return SimpleNamespace(stdout=expected.tobytes())

    monkeypatch.setattr(reference_video_module.subprocess, "run", fake_run)

    actual = reference_video_module._decode_video_frames_ffmpeg(
        "prepared.mp4",
        frame_count=frame_count,
        indices=indices,
        width=3,
        height=2,
    )

    np.testing.assert_array_equal(actual, expected)
    command, kwargs = observed[0]
    assert command[0] == "ffmpeg"
    assert command[command.index("-frames:v") + 1] == "2"
    if indices is None:
        assert "-vf" not in command
    else:
        assert command[command.index("-vf") + 1] == "select=eq(n\\,0)+eq(n\\,12)"
    assert command[-1] == "pipe:1"
    assert kwargs == {"check": True, "capture_output": True}


def test_ref2va_load_video_frames_uses_ffmpeg_without_decord(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import reference_video as reference_video_module

    expected = np.zeros((2, 2, 3, 3), dtype=np.uint8)
    decode_calls: list[Any] = []
    monkeypatch.setitem(sys.modules, "decord", None)
    monkeypatch.setattr(
        reference_video_module,
        "_probe_video",
        lambda _path: {
            "frame_count": 2,
            "width": 3,
            "height": 2,
        },
    )
    monkeypatch.setattr(
        reference_video_module,
        "_decode_video_frames_ffmpeg",
        lambda path, **kwargs: _append_and_return(decode_calls, (path, kwargs), expected),
    )

    actual = reference_video_module.load_video_frames("prepared.mp4")

    assert actual is expected
    assert decode_calls == [
        (
            "prepared.mp4",
            {
                "frame_count": 2,
                "width": 3,
                "height": 2,
            },
        )
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"frame_count": 0, "width": 3, "height": 2}, "video has no frames"),
        ({"frame_count": 1, "width": 0, "height": 2}, "invalid dimensions"),
        (
            {"frame_count": 1, "indices": [], "width": 3, "height": 2},
            "invalid frame indices",
        ),
    ],
)
def test_ref2va_ffmpeg_decode_rejects_invalid_metadata(kwargs, message):
    from vllm_omni.diffusion.models.minimax_h3 import reference_video as reference_video_module
    from vllm_omni.errors import OmniClientError

    with pytest.raises(OmniClientError, match=message):
        reference_video_module._decode_video_frames_ffmpeg("prepared.mp4", **kwargs)


def test_ref2va_probe_uses_container_frame_count_without_scan(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import reference_video as reference_video_module

    calls: list[Any] = []
    probe = {
        "streams": [
            {
                "codec_type": "video",
                "width": 4,
                "height": 2,
                "r_frame_rate": "24/1",
                "nb_frames": "3",
                "duration": "0.125",
                "sample_aspect_ratio": "1:1",
            }
        ],
        "format": {"duration": "0.125", "size": "123"},
    }
    monkeypatch.setattr(
        reference_video_module.subprocess,
        "run",
        lambda command, **kwargs: _append_and_return(
            calls,
            (command, kwargs),
            SimpleNamespace(stdout=json.dumps(probe)),
        ),
    )

    metadata = reference_video_module._probe_video("prepared.mp4")

    assert metadata["frame_count"] == 3
    assert len(calls) == 1
    assert "-count_frames" not in calls[0][0]


def test_ref2va_probe_counts_frames_only_when_metadata_is_missing(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import reference_video as reference_video_module

    calls: list[Any] = []
    stream = {
        "codec_type": "video",
        "width": 4,
        "height": 2,
        "r_frame_rate": "24/1",
        "duration": "0.125",
        "sample_aspect_ratio": "1:1",
    }
    first_probe = {
        "streams": [stream],
        "format": {"duration": "0.125", "size": "123"},
    }
    second_probe = {
        "streams": [{**stream, "nb_read_frames": "3"}],
        "format": {"duration": "0.125", "size": "123"},
    }
    monkeypatch.setattr(
        reference_video_module.subprocess,
        "run",
        lambda command, **kwargs: _append_and_return(
            calls,
            (command, kwargs),
            SimpleNamespace(stdout=json.dumps(first_probe if not calls else second_probe)),
        ),
    )

    metadata = reference_video_module._probe_video("prepared.mp4")

    assert metadata["frame_count"] == 3
    assert len(calls) == 2
    assert "-count_frames" in calls[1][0]


def test_ref2va_ffmpeg_decode_rejects_incomplete_output(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import reference_video as reference_video_module
    from vllm_omni.errors import OmniClientError

    monkeypatch.setattr(
        reference_video_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=b"short"),
    )

    with pytest.raises(OmniClientError, match="decoded 5 bytes.*expected 18"):
        reference_video_module._decode_video_frames_ffmpeg(
            "prepared.mp4",
            frame_count=1,
            width=3,
            height=2,
        )


def test_ref2va_two_video_recipe_tolerates_container_rounding(monkeypatch, tmp_path):
    from vllm_omni.diffusion.models.minimax_h3 import reference_video as reference_video_module

    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.touch()
    second.touch()
    metadata = {
        str(first): {
            "width": 1280,
            "height": 720,
            "fps": 24.0,
            "frame_count": 241,
            "duration": 10.041667,
            "format_names": ("mp4",),
            "video_codec": "h264",
            "audio_codecs": (),
            "file_size": 1024,
        },
        str(second): {
            "width": 1280,
            "height": 720,
            "fps": 24.0,
            "frame_count": 120,
            "duration": 4.966667,
            "format_names": ("mp4",),
            "video_codec": "h264",
            "audio_codecs": (),
            "file_size": 1024,
        },
    }
    transcode_calls: list[Any] = []
    monkeypatch.setattr(reference_video_module, "_probe_video", lambda path: metadata[str(path)])
    monkeypatch.setattr(
        reference_video_module,
        "_transcode_reference_video",
        lambda source, **kwargs: _append_and_return(
            transcode_calls,
            (source, kwargs),
            f"{source}.prepared.mp4",
        ),
    )

    prepared = reference_video_module.prepare_reference_videos(
        [str(first), str(second)],
        target_frame_count=124,
        workdir=str(tmp_path / "work"),
    )

    assert [item["duration_seconds"] for item in prepared] == pytest.approx([10.041667, 4.958333])
    assert sum(item["duration_seconds"] for item in prepared) == pytest.approx(15.0)
    assert transcode_calls[1][1]["duration_seconds"] == pytest.approx(4.958333)


def test_ref2va_two_video_recipe_rejects_real_duration_overflow(monkeypatch, tmp_path):
    from vllm_omni.diffusion.models.minimax_h3 import reference_video as reference_video_module
    from vllm_omni.errors import OmniClientError

    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.touch()
    second.touch()
    metadata = {
        str(first): {
            "width": 1280,
            "height": 720,
            "fps": 24.0,
            "frame_count": 240,
            "duration": 10.0,
            "format_names": ("mp4",),
            "video_codec": "h264",
            "audio_codecs": (),
            "file_size": 1024,
        },
        str(second): {
            "width": 1280,
            "height": 720,
            "fps": 24.0,
            "frame_count": 120,
            "duration": 5.02,
            "format_names": ("mp4",),
            "video_codec": "h264",
            "audio_codecs": (),
            "file_size": 1024,
        },
    }
    monkeypatch.setattr(reference_video_module, "_probe_video", lambda path: metadata[str(path)])
    monkeypatch.setattr(reference_video_module, "_transcode_reference_video", lambda source, **kwargs: "prepared.mp4")

    with pytest.raises(OmniClientError, match="15 seconds"):
        reference_video_module.prepare_reference_videos(
            [str(first), str(second)],
            target_frame_count=124,
            workdir=str(tmp_path / "work"),
        )


@pytest.mark.parametrize("duration", [4.0, 15.0])
def test_g2_output_duration_accepts_official_boundaries(duration):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    sampling = SimpleNamespace(
        fps=24,
        num_frames=1,
        height=None,
        width=None,
        extra_args={"duration": duration},
    )

    height, width, *_ = pipeline._resolve_shape("ref2va", sampling, None)
    assert height > 0 and width > 0


@pytest.mark.parametrize("duration", [3.99, 15.01, float("nan"), "not-a-duration"])
def test_g2_output_duration_rejects_out_of_contract_values(duration):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    sampling = SimpleNamespace(
        fps=24,
        num_frames=1,
        height=None,
        width=None,
        extra_args={"duration": duration},
    )
    with pytest.raises(ValueError, match="duration"):
        pipeline._resolve_shape("ref2va", sampling, None)


def test_g1_fanout_uses_incrementing_output_seeds():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _minimax_h3_output_seeds,
        _resolve_minimax_h3_num_outputs,
    )

    assert _resolve_minimax_h3_num_outputs(3) == 3
    assert _minimax_h3_output_seeds(100, 3) == [100, 101, 102]
    with pytest.raises(ValueError, match="num_outputs_per_prompt"):
        _resolve_minimax_h3_num_outputs(11)


def test_g3_task_specific_aspect_ratio_policy():
    from PIL import Image

    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _resolve_minimax_h3_aspect_ratio,
    )

    image = Image.new("RGB", (1280, 720))
    assert _resolve_minimax_h3_aspect_ratio("fl2va", "9:16", image) == pytest.approx(16 / 9)
    assert _resolve_minimax_h3_aspect_ratio("ref2va", None, None) == pytest.approx(16 / 9)
    assert _resolve_minimax_h3_aspect_ratio("ref2va", "auto", None) == pytest.approx(16 / 9)
    with pytest.raises(ValueError, match="requires an explicit"):
        _resolve_minimax_h3_aspect_ratio("t2va", None, None)
    with pytest.raises(ValueError, match="one of"):
        _resolve_minimax_h3_aspect_ratio("t2va", "2:1", None)


def test_g3_t2va_shape_requires_a_named_ratio_and_fl2va_ignores_override():
    from PIL import Image

    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    t2va_sampling = SimpleNamespace(
        fps=24,
        num_frames=1,
        height=None,
        width=None,
        extra_args={"duration": 5.0},
    )
    with pytest.raises(ValueError, match="requires an explicit aspect_ratio"):
        pipeline._resolve_shape("t2va", t2va_sampling, None)

    fl2va_sampling = SimpleNamespace(
        fps=24,
        num_frames=1,
        height=None,
        width=None,
        extra_args={"duration": 5.0, "aspect_ratio": "9:16"},
    )
    height, width, *_ = pipeline._resolve_shape("fl2va", fl2va_sampling, Image.new("RGB", (1280, 720)))
    assert (height, width) == (768, 1344)


def test_g4_reference_image_boundaries_and_aspect_ratio():
    from PIL import Image

    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _validate_reference_image,
    )

    _validate_reference_image(Image.new("RGB", (256, 256)))
    _validate_reference_image(Image.new("RGB", (5760, 2304)))
    with pytest.raises(ValueError, match="dimensions"):
        _validate_reference_image(Image.new("RGB", (255, 255)))
    with pytest.raises(ValueError, match="aspect ratio"):
        _validate_reference_image(Image.new("RGB", (256, 641)))


def test_g4_reference_image_file_format_and_size_contract(tmp_path):
    from PIL import Image

    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
        _load_image,
    )

    valid_path = tmp_path / "reference.png"
    Image.new("RGB", (256, 256)).save(valid_path)
    assert _load_image(valid_path).size == (256, 256)

    invalid_path = tmp_path / "reference.bmp"
    Image.new("RGB", (256, 256)).save(invalid_path)
    with pytest.raises(ValueError, match="must use"):
        _load_image(invalid_path)


def test_g4_standalone_audio_duration_and_total_duration_contract():
    from vllm_omni.diffusion.models.minimax_h3.reference_video import (
        validate_reference_audio_waveforms,
    )

    sample_rate = 16000
    validate_reference_audio_waveforms(
        [
            (torch.zeros(1, 2 * sample_rate), sample_rate),
            (torch.zeros(1, 13 * sample_rate), sample_rate),
        ]
    )
    with pytest.raises(ValueError, match="duration"):
        validate_reference_audio_waveforms([(torch.zeros(1, int(1.9 * sample_rate)), sample_rate)])
    with pytest.raises(ValueError, match="at most 15 seconds in total"):
        validate_reference_audio_waveforms(
            [
                (torch.zeros(1, 8 * sample_rate), sample_rate),
                (torch.zeros(1, 8 * sample_rate), sample_rate),
            ]
        )


def test_ref2va_audio_duration_validation_precedes_rank_branch(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as pipeline_module

    pipeline = object.__new__(pipeline_module.MiniMaxH3Pipeline)
    pipeline.device = torch.device("cpu")
    monkeypatch.setattr(pipeline_module, "_dit_rank_world", lambda: (None, 1, 2))

    waveform = torch.zeros(1, 10)
    with pytest.raises(ValueError, match="max_duration_seconds must be positive"):
        pipeline._encode_audio_conditions_resident(
            [(waveform, 10)],
            max_duration_seconds=0,
        )


def test_ref2va_standalone_audio_condition_is_bounded_to_output_duration():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    pipeline.device = torch.device("cpu")
    observed = []

    class FakeAudioVAE:
        def encode_waveform(self, waveform, sample_rate):
            observed.append((waveform.clone(), sample_rate))
            return waveform, waveform.shape[-1]

    pipeline.audio_vae = FakeAudioVAE()
    waveform = torch.arange(40, dtype=torch.float32).reshape(1, 40)

    encoded, lengths = pipeline._encode_audio_conditions_resident(
        [(waveform, 10)],
        max_duration_seconds=2.5,
    )

    assert observed[0][0].shape == (1, 25)
    assert observed[0][1] == 10
    assert encoded.shape == (1, 25)
    assert lengths == [25]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("fps", 23.0, "FPS"),
        ("duration", 1.0, "duration"),
        ("format_names", ("avi",), "container"),
        ("video_codec", "vp9", "H.264"),
        ("audio_codecs", ("opus",), "AAC"),
        ("file_size", 50 * 1024 * 1024 + 1, "size"),
    ],
)
def test_g4_reference_video_metadata_validation(field, value, message, tmp_path):
    from vllm_omni.diffusion.models.minimax_h3.reference_video import (
        _validate_reference_video_metadata,
    )

    metadata = {
        "width": 1280,
        "height": 720,
        "fps": 24.0,
        "duration": 10.0,
        "format_names": ("mp4",),
        "video_codec": "h264",
        "audio_codecs": ("aac",),
        "file_size": 1024,
    }
    metadata[field] = value
    with pytest.raises(ValueError, match=message):
        _validate_reference_video_metadata(metadata, index=0, source=str(tmp_path / "reference.mp4"))


_ENCODER_HIDDEN = 4
_ENCODER_HEAD_DIM = 2
_ENCODER_NUM_HEADS = 2
_ENCODER_NUM_KV_HEADS = 1
_ENCODER_INTERMEDIATE = 4

# One distinct value per checkpoint tensor, none of them a parameter initializer
# (fused weights start as `torch.empty`, norms as ones). A uniform fill would
# prove only that every row was written, not that each source landed in its own
# row range, which is the invariant the fused loaders have to get right.
_SOURCE_FILL = {
    "self_attn.q_proj.weight": 3.0,
    "self_attn.k_proj.weight": 4.0,
    "self_attn.v_proj.weight": 5.0,
    "mlp.gate_proj.weight": 6.0,
    "mlp.up_proj.weight": 7.0,
    "input_layernorm.weight": 8.0,
}

_QKV_TARGET = "text_model.layers.0.self_attn.qkv_proj.weight"
_GATE_UP_TARGET = "text_model.layers.0.mlp.gate_up_proj.weight"
_NORM_TARGET = "text_model.layers.0.input_layernorm.weight"


def _one_layer_text_encoder():
    """Encoder stub holding both fused weights plus one plain parameter."""
    from vllm_omni.diffusion.models.minimax_h3.encoder import (
        MiniMaxH3Qwen3VLEncoder,
        MiniMaxH3Qwen3VLMergedColumnParallelLinear,
        MiniMaxH3Qwen3VLQKVParallelLinear,
        MiniMaxH3Qwen3VLRMSNorm,
    )
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _SingleRankEncoderGroup

    group = _SingleRankEncoderGroup(0)
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.mlp = nn.Module()
    with (
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=0),
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=1),
    ):
        layer.self_attn.qkv_proj = MiniMaxH3Qwen3VLQKVParallelLinear(
            group,
            hidden_size=_ENCODER_HIDDEN,
            num_heads=_ENCODER_NUM_HEADS,
            num_kv_heads=_ENCODER_NUM_KV_HEADS,
            head_dim=_ENCODER_HEAD_DIM,
            dtype=torch.float32,
        )
        layer.mlp.gate_up_proj = MiniMaxH3Qwen3VLMergedColumnParallelLinear(
            group,
            input_size=_ENCODER_HIDDEN,
            intermediate_size=_ENCODER_INTERMEDIATE,
            dtype=torch.float32,
        )
    layer.input_layernorm = MiniMaxH3Qwen3VLRMSNorm(_ENCODER_HIDDEN)

    encoder = object.__new__(MiniMaxH3Qwen3VLEncoder)
    nn.Module.__init__(encoder)
    encoder.text_model = nn.Module()
    encoder.text_model.layers = nn.ModuleList([layer])
    return encoder


def _write_text_encoder_checkpoint(path, *, omit=()):
    """Write a one-layer Qwen3-VL-named checkpoint, one fill value per source."""
    import json

    import safetensors.torch

    prefix = "model.language_model.layers.0"
    q_rows = _ENCODER_NUM_HEADS * _ENCODER_HEAD_DIM
    kv_rows = _ENCODER_NUM_KV_HEADS * _ENCODER_HEAD_DIM
    shapes = {
        "self_attn.q_proj.weight": (q_rows, _ENCODER_HIDDEN),
        "self_attn.k_proj.weight": (kv_rows, _ENCODER_HIDDEN),
        "self_attn.v_proj.weight": (kv_rows, _ENCODER_HIDDEN),
        "mlp.gate_proj.weight": (_ENCODER_INTERMEDIATE, _ENCODER_HIDDEN),
        "mlp.up_proj.weight": (_ENCODER_INTERMEDIATE, _ENCODER_HIDDEN),
        "input_layernorm.weight": (_ENCODER_HIDDEN,),
    }
    tensors = {f"{prefix}.{source}": torch.full(shape, _SOURCE_FILL[source]) for source, shape in shapes.items()}
    for source in omit:
        del tensors[f"{prefix}.{source}"]
    safetensors.torch.save_file(tensors, str(path / "model.safetensors"))
    index = {"weight_map": dict.fromkeys(tensors, "model.safetensors")}
    (path / "model.safetensors.index.json").write_text(json.dumps(index))
    return tensors


@pytest.mark.parametrize(
    ("omitted_sources", "expected_detail", "unreported_target"),
    [
        (("self_attn.q_proj.weight",), f"{_QKV_TARGET}: ['q']", _GATE_UP_TARGET),
        (("self_attn.k_proj.weight",), f"{_QKV_TARGET}: ['k']", _GATE_UP_TARGET),
        (("self_attn.v_proj.weight",), f"{_QKV_TARGET}: ['v']", _GATE_UP_TARGET),
        (("mlp.gate_proj.weight",), f"{_GATE_UP_TARGET}: ['gate']", _QKV_TARGET),
        (("mlp.up_proj.weight",), f"{_GATE_UP_TARGET}: ['up']", _QKV_TARGET),
        (
            ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"),
            f"{_QKV_TARGET}: ['q', 'k', 'v']",
            _GATE_UP_TARGET,
        ),
    ],
)
def test_encoder_load_reports_missing_fused_source_shards(
    tmp_path, omitted_sources, expected_detail, unreported_target
):
    encoder = _one_layer_text_encoder()
    weights = _write_text_encoder_checkpoint(tmp_path, omit=omitted_sources)

    with pytest.raises(RuntimeError) as excinfo:
        encoder.load_weights(weights.items())

    message = str(excinfo.value)
    assert expected_detail in message
    assert unreported_target not in message


def test_encoder_load_rejects_unloaded_plain_param(tmp_path):
    encoder = _one_layer_text_encoder()
    weights = _write_text_encoder_checkpoint(tmp_path, omit=("input_layernorm.weight",))

    with pytest.raises(RuntimeError, match=r"weights not loaded.*input_layernorm"):
        encoder.load_weights(weights.items())


def test_encoder_load_places_every_source_in_its_own_rows(tmp_path):
    encoder = _one_layer_text_encoder()
    weights = _write_text_encoder_checkpoint(tmp_path)

    encoder.load_weights(weights.items())

    params = dict(encoder.named_parameters())
    q_rows = _ENCODER_NUM_HEADS * _ENCODER_HEAD_DIM
    kv_rows = _ENCODER_NUM_KV_HEADS * _ENCODER_HEAD_DIM
    qkv = params[_QKV_TARGET]
    gate_up = params[_GATE_UP_TARGET]
    slices = {
        "self_attn.q_proj.weight": qkv[:q_rows],
        "self_attn.k_proj.weight": qkv[q_rows : q_rows + kv_rows],
        "self_attn.v_proj.weight": qkv[q_rows + kv_rows :],
        "mlp.gate_proj.weight": gate_up[:_ENCODER_INTERMEDIATE],
        "mlp.up_proj.weight": gate_up[_ENCODER_INTERMEDIATE:],
        "input_layernorm.weight": params[_NORM_TARGET],
    }
    for source, rows in slices.items():
        assert torch.equal(rows, torch.full_like(rows, _SOURCE_FILL[source])), source
