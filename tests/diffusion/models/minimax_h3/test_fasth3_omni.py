# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from PIL import Image

from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, OmniDiffusionConfig
from vllm_omni.diffusion.models.minimax_h3.fasth3_checkpoint import FastH3CheckpointSpec, FastH3OmniCheckpointSpec
from vllm_omni.diffusion.models.minimax_h3.lora import TurboSpec
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest
from vllm_omni.model_executor.models.minimax_h3.checkpoint import (
    minimax_h3_transformer_subfolder,
    resolve_minimax_h3_encoder_model_root,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def omni_metadata():
    return {
        "schema_version": "fasth3-inference-contract-v1",
        "model_type": "ref2va",
        "transformer_component": "transformer_ref",
        "conditioning": "fixed_ordered_references_target_only_flow",
        "base_model_revision": "hf://MiniMaxAI/MiniMax-H3@9bfb6693f2cf6de171db46d1aa586f67d773a1da",
        "pdd_steps": 32,
        "pdd_step_indices": list(range(0, 33, 4)),
        "grid_max_t": 0.999,
        "num_inference_steps": 8,
        "transformer_forwards": 8,
        "video_scheduler_shift": 12.0,
        "audio_scheduler_shift": 3.0,
        "guidance_scale": 1.0,
        "attention_backend": "VIDEO_SPARSE_ATTN_H3",
        "vsa_sparsity": 0.9,
        "vsa_tile_size": 128,
        "vsa_ref_keep_rate": 0.1,
        "vsa_ref_policy": "p2_multi_region",
    }


def sampling(**kwargs):
    defaults = dict(
        num_inference_steps=8, guidance_scale=1.0, extra_args={}, lora_request=None, timesteps=None, sigmas=None
    )
    return OmniDiffusionSamplingParams(**(defaults | kwargs))


def test_omni_contract_selects_ref_transformer():
    spec = FastH3CheckpointSpec.from_metadata(omni_metadata())
    assert isinstance(spec, FastH3OmniCheckpointSpec)
    assert spec.transformer_component == "transformer_ref"
    assert spec.release_metadata()["tasks"] == ["ref2va"]
    spec.check_request(sampling())


@pytest.mark.parametrize("field", list(omni_metadata()))
def test_omni_rejects_missing_contract_fields(field):
    metadata = omni_metadata()
    del metadata[field]
    with pytest.raises(ValueError):
        FastH3CheckpointSpec.from_metadata(metadata)


@pytest.mark.parametrize(
    "field,value",
    [
        ("pdd_steps", 16),
        ("pdd_step_indices", [0, 8, 16, 24, 32]),
        ("vsa_tile_size", 64),
        ("vsa_ref_policy", "prefix_dense"),
        ("video_scheduler_shift", 10.0),
        ("base_model_revision", "hf://other/model@main"),
    ],
)
def test_omni_rejects_incompatible_metadata(field, value):
    metadata = omni_metadata() | {field: value}
    with pytest.raises(ValueError, match=field):
        FastH3CheckpointSpec.from_metadata(metadata)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_inference_steps": 4},
        {"guidance_scale": 2.0},
        {"sigmas": [1, 0]},
        {"extra_args": {"flow_shift": 10}},
        {"extra_args": {"task": "t2va"}},
        {"extra_args": {"pdd_steps": 16}},
        {"extra_args": {"latent_refine": {"steps": 2}}},
    ],
)
def test_omni_rejects_incompatible_requests(kwargs):
    with pytest.raises(OmniClientError):
        FastH3CheckpointSpec.from_metadata(omni_metadata()).check_request(sampling(**kwargs))


def test_local_export_uses_its_ref_config_and_pinned_native_encoder(tmp_path, monkeypatch):
    (tmp_path / "fastvideo_inference.json").write_text(json.dumps(omni_metadata()))
    assert minimax_h3_transformer_subfolder(model=str(tmp_path)) == "transformer_ref"
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return "/base/snapshot"

    monkeypatch.setattr(
        "vllm_omni.model_executor.models.minimax_h3.checkpoint.download_weights_from_hf_specific", download
    )
    assert resolve_minimax_h3_encoder_model_root(str(tmp_path), None, "auto") == "/base/snapshot/Ref2VA/text_encoder"
    assert calls[0]["model_name_or_path"] == "MiniMaxAI/MiniMax-H3"
    assert calls[0]["revision"] == "9bfb6693f2cf6de171db46d1aa586f67d773a1da"
    assert all("transformer" not in pattern for pattern in calls[0]["allow_patterns"])


def test_omni_base_components_do_not_download_either_base_transformer(monkeypatch):
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return "/base/snapshot"

    monkeypatch.setattr(
        "vllm_omni.diffusion.models.minimax_h3.fasth3_checkpoint.download_weights_from_hf_specific", download
    )
    spec = FastH3CheckpointSpec.from_metadata(omni_metadata())
    assert str(spec.resolve_base_components(load_text_encoder=True)) == "/base/snapshot/Ref2VA"
    assert calls[0]["revision"] == spec.base_revision
    assert set(calls[0]["allow_patterns"]) == {
        "Ref2VA/video_vae/**",
        "Ref2VA/audio_vae/**",
        "Ref2VA/text_encoder/**",
        "Ref2VA/tokenizer/**",
        "Ref2VA/processor/**",
    }


@pytest.mark.parametrize(
    "device,capability,missing_module,message",
    [
        ("cpu", None, None, "CUDA"),
        ("cuda:3", (9, 0), None, "SM100"),
        ("cuda:3", (10, 1), None, "SM100"),
        ("cuda:3", (10, 0), "fastvideo_kernel", "fastvideo-kernel"),
        ("cuda:3", (10, 3), "fastvideo_kernel.block_sparse_attn", "fastvideo-kernel"),
        ("cuda:3", (12, 0), "flashinfer.cute_dsl.sparse", "bsa_attn_sm120_blk64_fwd"),
        ("cuda:3", (12, 1), "flashinfer.cute_dsl.sparse", "bsa_attn_sm120_blk64_fwd"),
    ],
)
def test_omni_serving_rejects_unavailable_tile128_provider(monkeypatch, device, capability, missing_module, message):
    from vllm_omni.diffusion.attention.ops import block_sparse

    monkeypatch.setattr(
        "vllm_omni.diffusion.models.minimax_h3.fasth3_checkpoint.get_local_device", lambda: torch.device(device)
    )
    monkeypatch.setattr(block_sparse.current_omni_platform, "get_device_capability", lambda index: capability)
    if missing_module:
        monkeypatch.setitem(sys.modules, missing_module, None)
    config = OmniDiffusionConfig(
        diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="FASTVIDEO_VSA"))
    )
    with pytest.raises((ValueError, ImportError), match=message):
        FastH3OmniCheckpointSpec().check_serving_contract(partition="ref2va", od_config=config)


def test_omni_serving_resolves_its_base_with_bf16_whatever_the_dtype(mocker):
    from vllm_omni.diffusion.models.minimax_h3 import fasth3_checkpoint

    resolve = mocker.patch.object(fasth3_checkpoint, "resolve_block_sparse_provider", return_value="flashinfer")
    config = OmniDiffusionConfig(
        diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="FASTVIDEO_VSA")),
        dtype=torch.float16,
    )
    FastH3OmniCheckpointSpec().check_serving_contract(partition="ref2va", od_config=config)
    # H3's attention operands are BF16 whatever --dtype is, and tile128 requires BF16.
    assert resolve.call_args.args[3] == torch.bfloat16
    assert resolve.call_args.kwargs["block_size"] == 128


def test_omni_serving_checks_flashinfer_on_the_worker_device(monkeypatch, mocker):
    from vllm_omni.diffusion.attention.ops import block_sparse, flashinfer_block_sparse

    device = torch.device("cuda:3")
    monkeypatch.setattr("vllm_omni.diffusion.models.minimax_h3.fasth3_checkpoint.get_local_device", lambda: device)
    capability = mocker.patch.object(block_sparse.current_omni_platform, "get_device_capability", return_value=(12, 0))
    require_kernel = mocker.patch.object(flashinfer_block_sparse, "require_flashinfer_sparse")
    config = OmniDiffusionConfig(
        diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="FASTVIDEO_VSA"))
    )
    FastH3OmniCheckpointSpec().check_serving_contract(partition="ref2va", od_config=config)
    capability.assert_called_once_with(3)
    require_kernel.assert_called_once_with("bf16", device)


def test_omni_accepts_one_ordinary_lora():
    request = LoRARequest("ordinary", 1, "/models/ordinary.safetensors")
    spec = FastH3CheckpointSpec.from_metadata(omni_metadata())
    spec.check_request(sampling(lora_request=request, lora_scale=0.85))


def test_omni_accepts_lora_composition():
    requests = [LoRARequest("one", 1, "/models/one"), LoRARequest("two", 2, "/models/two")]
    spec = FastH3CheckpointSpec.from_metadata(omni_metadata())
    spec.check_request(sampling(lora_request=requests, lora_scale=[1.0, 0.5]))


@pytest.fixture
def ref2va_turbo_spec() -> TurboSpec:
    return TurboSpec(
        filename="minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors",
        task_family="ref2v",
        denoise_steps=8,
        video_shift=6.0,
        audio_shift=3.0,
        rank=128,
        alpha=8.0,
    )


@pytest.mark.parametrize("adapter_kind", ["turbo", "distilled"])
def test_omni_rejects_distilled_adapter_schedule(adapter_kind, ref2va_turbo_spec):
    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = SimpleNamespace()
    pipeline._fasth3_checkpoint = FastH3OmniCheckpointSpec()
    pipeline._turbo_lora_specs = {1: ref2va_turbo_spec} if adapter_kind == "turbo" else {}
    pipeline._native_distilled_lora_adapter_ids = {1} if adapter_kind == "distilled" else set()
    conditioning = SimpleNamespace(task="ref2va", num_frames=56, video_edit_clean_rows=None, audio_edit_clean_rows=None)
    request = LoRARequest(adapter_kind, 1, f"/models/{adapter_kind}")
    with pytest.raises(OmniClientError, match="not Turbo or distilled adapters"):
        pipeline._prepare_encoder_conditioning_inputs(conditioning, sampling(lora_request=request))


@pytest.mark.parametrize("variant", ["omni", "ordinary", "turbo_first", "turbo_last"])
def test_local_encoder_accepts_lora_composition(variant, ref2va_turbo_spec):
    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline._fasth3_checkpoint = FastH3OmniCheckpointSpec() if variant == "omni" else None
    pipeline.partition = "ref2va"
    pipeline.supported_tasks = frozenset({"ref2va"})
    pipeline.device = torch.device("cpu")
    pipeline.load_text_encoder = pipeline.load_vae_encoder = True
    pipeline._turbo_lora_specs = {2: ref2va_turbo_spec} if variant.startswith("turbo") else {}
    pipeline.default_video_shift = 12.0
    pipeline.default_audio_shift = 3.0
    pipeline._native_distilled_lora_adapter_ids = set()
    pipeline.encode_prompt = Mock(return_value=(torch.ones(2, 5120, dtype=torch.bfloat16), torch.tensor([0, 1])))
    pipeline.video_vae = Mock()
    pipeline.audio_vae = Mock()
    pipeline.video_vae.is_distributed_enabled.return_value = False
    pipeline.video_vae.encode_image.return_value = torch.ones(64, 96)
    requests = [LoRARequest("one", 1, "/models/one"), LoRARequest("two", 2, "/models/two")]
    if variant == "turbo_first":
        requests.reverse()
    params = sampling(
        height=32,
        width=32,
        num_frames=96,
        lora_request=requests,
        lora_scale=[0.8, 0.3],
        extra_args={"task": "ref2va", "flow_shift": 6.0 if variant.startswith("turbo") else 12.0},
    )
    conditioning, window_text = pipeline._prepare_local_conditioning(
        {"prompt": "Replace the person.", "multi_modal_data": {"image": Image.new("RGB", (256, 256))}}, params
    )
    assert conditioning.task == "ref2va"
    assert conditioning.visual_condition.shape == (64, 96)
    assert window_text is None
    pipeline.encode_prompt.assert_called_once()


@pytest.mark.parametrize("kind", ["two_turbo", "distilled_with_swap"])
def test_lora_composition_rejects_conflicting_schedules(kind, ref2va_turbo_spec):
    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline._turbo_lora_specs = {1: ref2va_turbo_spec, 2: ref2va_turbo_spec} if kind == "two_turbo" else {}
    pipeline._native_distilled_lora_adapter_ids = {1} if kind == "distilled_with_swap" else set()
    requests = [LoRARequest("one", 1, "/models/one"), LoRARequest("two", 2, "/models/two")]
    message = "at most one Turbo" if kind == "two_turbo" else "cannot be composed"
    with pytest.raises(OmniClientError, match=message):
        pipeline._active_turbo_spec(sampling(lora_request=requests, lora_scale=[1.0, 0.8]))
