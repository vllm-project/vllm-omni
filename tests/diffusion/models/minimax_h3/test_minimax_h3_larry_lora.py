# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import regex as re
import torch
from safetensors.torch import save_file
from vllm.lora.layers.column_parallel_linear import MergedColumnParallelLinearWithLoRA
from vllm.lora.layers.row_parallel_linear import RowParallelLinearWithLoRA
from vllm.lora.lora_weights import PackedLoRALayerWeights

from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
from vllm_omni.diffusion.models.minimax_h3 import larry_lora as larry
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def tiny_targets(monkeypatch):
    dimensions = {
        "attn.qkv_proj": (8, 24),
        "attn.out_proj": (8, 8),
        "mlp.fc1": (8, 32),
        "mlp.fc2": (16, 8),
        "adaln_proj.linear": (4, 16),
    }
    targets = {
        name: next(shape for suffix, shape in dimensions.items() if name.endswith(suffix))
        for name in larry.larry_targets()
    }
    targets["final_layer.adaln_proj.linear"] = (4, 8)
    monkeypatch.setattr(larry, "larry_targets", lambda: targets)
    return targets


@pytest.fixture
def pipeline():
    pipe = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipe)
    pipe.partition = "ref2va"
    pipe._turbo_lora_specs = {}
    pipe._native_lora_adapter_ids = set()
    pipe._lora_sigma_schedules = {}
    pipe._larry_lora_ids = {7}
    pipe._base_schedule_by_partition = {"fl2va": None, "ref2va": None}
    pipe.default_video_shift = 12.0
    pipe.default_audio_shift = 3.0
    return pipe


def _request(path: str | Path) -> LoRARequest:
    return LoRARequest("larry", 7, str(path))


def _sampling() -> OmniDiffusionSamplingParams:
    return OmniDiffusionSamplingParams(
        lora_request=_request("larry"), lora_scale=1.0, num_inference_steps=8, extra_args={}
    )


def _write_larry(path: Path) -> dict[str, torch.Tensor]:
    tensors = {}
    for index, (name, (width, height)) in enumerate(larry.larry_targets().items()):
        rank = index % 3 + 1
        tensors[f"{name}.lora_A.weight"] = (
            torch.arange(rank * width).reshape(rank, width).remainder(17) / 100
        ).bfloat16()
        tensors[f"{name}.lora_B.weight"] = (
            torch.arange(height * rank).reshape(height, rank).remainder(13) / 100
        ).bfloat16()
    save_file(tensors, str(path))
    return tensors


def _load(path: Path, partition: str = "ref2va"):
    loaded = larry.load_minimax_h3_larry_lora(
        partition=partition, lora_request=_request(path), lora_path=path, dtype=torch.float32
    )
    assert loaded is not None
    return loaded


def test_original_h3_target_contract():
    targets = larry.larry_targets()
    assert len(targets) == 259
    assert targets["blocks.0.attn.qkv_proj"] == (5376, 21504)
    assert targets["blocks.49.adaln_proj.linear"] == (2688, 96768)
    assert targets["final_layer.adaln_proj.linear"] == (2688, 10752)
    assert all(re.fullmatch(larry.LARRY_TARGET_PATTERN, "transformer." + name) for name in targets)
    assert re.fullmatch(larry.LARRY_TARGET_PATTERN, "transformer.token_refiner.blocks.0.adaln_proj.linear") is None
    assert re.fullmatch(larry.LARRY_TARGET_PATTERN, "transformers_ref.blocks.0.attn.qkv_proj") is None


@pytest.mark.parametrize("partition", ["fl2va", "ref2va", "combined"])
def test_load_preserves_every_mixed_rank_delta(tmp_path, tiny_targets, partition):
    path = tmp_path / larry.LARRY_FILENAME
    tensors = _write_larry(path)
    model, helper = _load(path, partition)
    assert model.rank == helper.r == 3
    assert set(model.loras) == {"transformer." + name for name in tiny_targets}
    assert {layer.rank for layer in model.loras.values()} == {1, 2, 3}

    for name, weights in model.loras.items():
        raw = name.removeprefix("transformer.")
        a, b = tensors[raw + ".lora_A.weight"].float(), tensors[raw + ".lora_B.weight"].float()
        x = torch.arange(2 * a.shape[1]).reshape(2, -1).float() / 20
        expected = (x @ a.T) @ b.T
        if isinstance(weights, PackedLoRALayerWeights):
            assert weights.scaling == [1.0] * len(weights.lora_b)
            actual = torch.cat([(x @ aa.T) @ bb.T for aa, bb in zip(weights.lora_a, weights.lora_b)], dim=-1)
        else:
            assert weights.scaling == 1.0
            actual = (x @ weights.lora_a.T) @ weights.lora_b.T
        torch.testing.assert_close(actual, expected)
    assert larry.select_larry_file(tmp_path) == path


@pytest.mark.parametrize("defect", ["missing", "unexpected", "shape", "dtype"])
def test_invalid_checkpoint_is_rejected(tmp_path, tiny_targets, defect):
    path = tmp_path / larry.LARRY_FILENAME
    tensors = _write_larry(path)
    name = next(iter(tensors))
    if defect == "missing":
        del tensors[name]
    elif defect == "unexpected":
        tensors["unknown.lora_A.weight"] = torch.ones(1, 1, dtype=torch.bfloat16)
    elif defect == "shape":
        tensors[name] = tensors[name][:, :-1].contiguous()
    else:
        tensors[name] = tensors[name].float()
    save_file(tensors, str(path))
    with pytest.raises(ValueError, match="mismatch|BF16"):
        _load(path)


def test_other_artifact_is_left_to_existing_loaders(tmp_path):
    path = tmp_path / "other.safetensors"
    path.write_bytes(b"not a Larry file")
    assert (
        larry.load_minimax_h3_larry_lora(
            partition="ref2va", lora_request=_request(path), lora_path=path, dtype=torch.float32
        )
        is None
    )


def test_standard_offload_is_rejected(tmp_path, tiny_targets):
    path = tmp_path / larry.LARRY_FILENAME
    _write_larry(path)
    with pytest.raises(ValueError, match="does not support"):
        larry.load_minimax_h3_larry_lora(
            partition="ref2va",
            lora_request=_request(path),
            lora_path=path,
            dtype=torch.float32,
            unsupported_offload_mode="layerwise offload",
        )


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("suffix", ["attn.qkv_proj", "mlp.fc1", "adaln_proj.linear", "attn.out_proj", "mlp.fc2"])
def test_tp_slicing_reconstructs_global_delta(tp, suffix, tiny_targets):
    width, height = tiny_targets["blocks.0." + suffix]
    a = torch.arange(3 * width).reshape(3, width).float() / 100
    b = torch.arange(height * 3).reshape(height, 3).float() / 100
    x = torch.arange(2 * width).reshape(2, width).float() / 20
    expected = (x @ a.T) @ b.T
    weights = larry.pack_larry_layer("transformer.blocks.0." + suffix, a, b)
    shards = []
    for rank in range(tp):
        if isinstance(weights, PackedLoRALayerWeights):
            layer = object.__new__(MergedColumnParallelLinearWithLoRA)
            torch.nn.Module.__init__(layer)
            layer.n_slices = len(weights.lora_b)
            layer.output_slices = tuple(bb.shape[0] // tp for bb in weights.lora_b)
            layer.output_ids = (rank,) * layer.n_slices
            local_b = layer.slice_lora_b(weights.lora_b)
            shards.append([(x @ aa.T) @ bb.T for aa, bb in zip(weights.lora_a, local_b)])
        elif suffix in {"attn.out_proj", "mlp.fc2"}:
            layer = object.__new__(RowParallelLinearWithLoRA)
            torch.nn.Module.__init__(layer)
            layer.tp_rank, layer.input_size = rank, width // tp
            shards.append((x.chunk(tp, dim=-1)[rank] @ layer.slice_lora_a(a).T) @ b.T)
        else:
            shards.append((x @ a.T) @ b.chunk(tp, dim=0)[rank].T)
    if isinstance(weights, PackedLoRALayerWeights):
        actual = torch.cat(
            [torch.cat([shard[i] for shard in shards], dim=-1) for i in range(len(weights.lora_b))], dim=-1
        )
    elif suffix in {"attn.out_proj", "mlp.fc2"}:
        actual = torch.stack(shards).sum(0)
    else:
        actual = torch.cat(shards, dim=-1)
    torch.testing.assert_close(actual, expected)


@dataclass
class _RecordingLayer:
    n_slices: int
    a: list | torch.Tensor | None = None
    b: list | torch.Tensor | None = None

    def set_lora(self, index, lora_a, lora_b):
        self.a, self.b = lora_a, lora_b

    def reset_lora(self, index):
        self.a = self.b = None


@pytest.mark.parametrize("scale", [0.5, 1.0])
def test_manager_binds_all_targets_and_scales_once(tmp_path, tiny_targets, pipeline, scale):
    path = tmp_path / larry.LARRY_FILENAME
    _write_larry(path)
    model, _ = _load(path)
    manager = object.__new__(DiffusionLoRAManager)
    manager.pipeline = pipeline
    manager._expected_lora_modules = {"qkv_proj", "out_proj", "fc1", "fc2", "linear"}
    manager._lora_modules = {
        name: _RecordingLayer(len(weights.lora_b) if weights.is_packed else 1) for name, weights in model.loras.items()
    }
    manager._bind_adapter_weights(model, scale)
    for name, layer in manager._lora_modules.items():
        weights = model.loras[name]
        if weights.is_packed:
            for actual, original in zip(layer.b, weights.lora_b):
                torch.testing.assert_close(actual, original * scale)
        else:
            torch.testing.assert_close(layer.b, weights.lora_b * scale)
    with pytest.raises(ValueError, match="Larry LoRA binding is incomplete"):
        pipeline._validate_diffusion_lora_binding(lora_model=model, bound_lora_names=frozenset())


def test_pipeline_reload_clears_larry_classification(tmp_path, tiny_targets, pipeline, mocker):
    path = tmp_path / larry.LARRY_FILENAME
    _write_larry(path)
    clear = mocker.spy(pipeline, "_clear_adaln_caches")
    loaded = pipeline._load_diffusion_lora_adapter(lora_request=_request(path), lora_path=path, dtype=torch.float32)
    assert loaded is not None and pipeline._larry_lora_ids == {7}
    other = tmp_path / "generic-adapter"
    assert (
        pipeline._load_diffusion_lora_adapter(lora_request=_request(other), lora_path=other, dtype=torch.float32)
        is None
    )
    assert not pipeline._larry_lora_ids
    assert clear.call_count == 2


@pytest.mark.parametrize("steps", [4, 5, 6, 7, 8])
@pytest.mark.parametrize("sampler", [None, "euler"])
def test_larry_uses_requested_update_count(pipeline, steps, sampler):
    sampling = _sampling()
    sampling.num_inference_steps = steps
    sampling.extra_args = {"sampler": sampler}
    positions, updates = pipeline._resolve_sigma_positions("ref2va", sampling)
    assert positions is None and updates == steps


@pytest.mark.parametrize("steps", [None, 3, 9, 50, True, 6.0])
def test_larry_rejects_out_of_contract_steps(pipeline, steps):
    sampling = _sampling()
    sampling.num_inference_steps = steps
    with pytest.raises(OmniClientError, match="4..8"):
        pipeline._resolve_sigma_positions("ref2va", sampling)


@pytest.mark.parametrize("extra", [{"flow_shift": 6}, {"audio_flow_shift": 12}, {"sampler": "res_multistep"}])
def test_larry_rejects_wrong_clock_or_sampler(pipeline, extra):
    sampling = _sampling()
    sampling.extra_args = extra
    with pytest.raises(OmniClientError):
        pipeline._resolve_sigma_positions("ref2va", sampling)


def test_ref2va_cannot_bind_combined_transformer(pipeline):
    pipeline.partition = "combined"
    with pytest.raises(OmniClientError, match="--task-type ref2va"):
        pipeline._resolve_sigma_positions("ref2va", _sampling())


def test_larry_rejects_checkpoint_pinned_schedule(pipeline, mocker):
    mocker.patch.object(pipeline, "_base_schedule_for_task", return_value=object())
    with pytest.raises(OmniClientError, match="pinned distilled base_schedule"):
        pipeline._resolve_sigma_positions("ref2va", _sampling())


def test_scale_zero_leaves_base_sampling_unchanged(pipeline):
    sampling = _sampling()
    sampling.lora_scale, sampling.num_inference_steps = 0, 50
    assert pipeline._resolve_sigma_positions("ref2va", sampling) == (None, 50)


def test_adapter_scale_change_and_disable_clear_adaln_cache(pipeline, mocker):
    clear = mocker.spy(pipeline, "_clear_adaln_caches")
    sampling = _sampling()
    pipeline._prepare_adaln_adapter(sampling)
    pipeline._prepare_adaln_adapter(sampling)
    sampling.lora_scale = 0.5
    pipeline._prepare_adaln_adapter(sampling)
    sampling.lora_request = None
    pipeline._prepare_adaln_adapter(sampling)
    assert clear.call_count == 3
