# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from vllm.lora.lora_weights import PackedLoRALayerWeights

from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.models.minimax_h3 import lora as lora_module
from vllm_omni.diffusion.models.minimax_h3.lora import load_minimax_h3_native_lora
from vllm_omni.diffusion.sched.sigma_schedule import DMD2SigmaSchedule
from vllm_omni.errors import OmniClientError
from vllm_omni.lora.request import LoRARequest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_TINY_HIDDEN = 2
_TINY_INNER = 3
_TINY_FFN = 2
_TINY_TIME = 2
_TINY_BLOCK_ADALN = 4
_TINY_FINAL_ADALN = 3
_TINY_RANK = 4
_TINY_NUM_QUERY_GROUPS = 2
_TINY_HEADS_PER_GROUP = 1
_TINY_HEAD_DIM = 1
_TINY_QKV_SLICE = _TINY_NUM_QUERY_GROUPS * _TINY_HEADS_PER_GROUP * _TINY_HEAD_DIM


@pytest.fixture(autouse=True)
def _use_tiny_native_dimensions(monkeypatch):
    monkeypatch.setattr(lora_module, "_NATIVE_RANK", _TINY_RANK)
    monkeypatch.setattr(lora_module, "_NATIVE_ALPHA", _TINY_RANK)
    monkeypatch.setattr(lora_module, "_NATIVE_HIDDEN_SIZE", _TINY_HIDDEN)
    monkeypatch.setattr(lora_module, "_NATIVE_ATTENTION_INNER_SIZE", _TINY_INNER)
    monkeypatch.setattr(lora_module, "_NATIVE_FFN_HIDDEN_SIZE", _TINY_FFN)
    monkeypatch.setattr(lora_module, "_NATIVE_TIME_EMBED_DIM", _TINY_TIME)
    monkeypatch.setattr(lora_module, "_NATIVE_BLOCK_ADALN_OUT", _TINY_BLOCK_ADALN)
    monkeypatch.setattr(lora_module, "_NATIVE_FINAL_ADALN_OUT", _TINY_FINAL_ADALN)
    monkeypatch.setattr(lora_module, "_NATIVE_NUM_QUERY_GROUPS", _TINY_NUM_QUERY_GROUPS)
    monkeypatch.setattr(lora_module, "_NATIVE_HEADS_PER_GROUP", _TINY_HEADS_PER_GROUP)
    monkeypatch.setattr(lora_module, "_NATIVE_HEAD_DIM", _TINY_HEAD_DIM)
    monkeypatch.setattr(lora_module, "_NATIVE_QKV_SLICE", _TINY_QKV_SLICE)
    monkeypatch.setattr(
        lora_module,
        "_NATIVE_TARGET_DIMS",
        {
            "attn.qkv_proj": (_TINY_HIDDEN, 3 * _TINY_INNER),
            "attn.out_proj": (_TINY_INNER, _TINY_HIDDEN),
            "mlp.fc1": (_TINY_HIDDEN, 2 * _TINY_FFN),
            "mlp.fc2": (_TINY_FFN, _TINY_HIDDEN),
            "adaln_proj.linear": (_TINY_TIME, _TINY_BLOCK_ADALN),
        },
    )
    monkeypatch.setattr(lora_module, "_NATIVE_FINAL_ADALN_DIMS", (_TINY_TIME, _TINY_FINAL_ADALN))
    monkeypatch.setattr(
        lora_module,
        "_NATIVE_EXPECTED_TARGETS",
        frozenset(
            [
                *(
                    f"blocks.{block_index}.{suffix}"
                    for block_index in range(50)
                    for suffix in lora_module._NATIVE_TARGET_SUFFIXES
                )
            ]
            + [
                *(
                    f"token_refiner.blocks.{block_index}.{suffix}"
                    for block_index in range(2)
                    for suffix in lora_module._NATIVE_TOKEN_REFINER_SUFFIXES
                )
            ]
            + ["final_layer.adaln_proj.linear"]
        ),
    )


def _request(path) -> LoRARequest:
    return LoRARequest(
        lora_name="flashgen",
        lora_int_id=7,
        lora_path=str(path),
    )


def _write_tiny_native(
    path,
    *,
    omit_target: str | None = None,
    shape_overrides: dict[str, tuple[int, int]] | None = None,
    metadata: dict[str, str] | None = None,
) -> None:
    rank = _TINY_RANK
    tensors = {}
    overrides = shape_overrides or {}
    for target in sorted(lora_module._NATIVE_EXPECTED_TARGETS):
        if target == omit_target:
            continue
        input_dim, output_dim = lora_module._native_target_dims(target)
        raw_target = f"transformer.{target}"
        a_name = f"{raw_target}.lora_A.default.weight"
        b_name = f"{raw_target}.lora_B.default.weight"
        tensors[a_name] = torch.ones(overrides.get(a_name, (rank, input_dim)))
        if target.endswith(".attn.qkv_proj"):
            grouped_rows = []
            for group in range(_TINY_NUM_QUERY_GROUPS):
                grouped_rows.extend(
                    [
                        torch.full((1, rank), float(group * 10 + 1)),
                        torch.full((1, rank), float(group * 10 + 2)),
                        torch.full((1, rank), float(group * 10 + 3)),
                    ]
                )
            tensors[b_name] = torch.cat(grouped_rows, dim=0)
        elif target.endswith(".mlp.fc1"):
            tensors[b_name] = torch.cat(
                (
                    torch.full((output_dim // 2, rank), 2.0),
                    torch.full((output_dim // 2, rank), 1.0),
                ),
                dim=0,
            )
        else:
            tensors[b_name] = torch.ones(overrides.get(b_name, (output_dim, rank)))
    save_file(
        tensors,
        str(path),
        metadata=metadata
        or {
            "format": "pt",
            "key_format": "minimax-h3-native",
            "qkv_layout": "grouped",
            "lora_rank": str(rank),
            "lora_alpha": str(rank),
            "base_schedule": "1.0,0.7,0.4,0.15,0.0",
            "tasks": "t2va",
        },
    )


def test_h3_native_loads_and_packs_qkv_and_fc1(tmp_path):
    path = tmp_path / "minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors"
    _write_tiny_native(path)

    loaded = load_minimax_h3_native_lora(
        partition="fl2va",
        lora_request=_request(path),
        lora_path=path,
        dtype=torch.float32,
    )

    assert loaded is not None
    lora_model, peft_helper, schedule = loaded
    assert peft_helper.r == _TINY_RANK
    assert schedule.num_inference_steps == 4
    assert len(lora_model.loras) == 259
    assert "blocks.0.adaln_proj.linear" in lora_model.loras
    assert "final_layer.adaln_proj.linear" in lora_model.loras

    qkv = lora_model.get_lora("blocks.0.attn.qkv_proj")
    assert isinstance(qkv, PackedLoRALayerWeights)
    torch.testing.assert_close(qkv.lora_b[0], torch.tensor([[10.0], [11.0]]))
    torch.testing.assert_close(qkv.lora_b[1], torch.tensor([[12.0], [22.0]]))
    torch.testing.assert_close(qkv.lora_b[2], torch.tensor([[13.0], [23.0]]))

    fc1 = lora_model.get_lora("blocks.0.mlp.fc1")
    assert isinstance(fc1, PackedLoRALayerWeights)
    torch.testing.assert_close(fc1.lora_b[0], torch.full((1, _TINY_RANK), 2.0))
    torch.testing.assert_close(fc1.lora_b[1], torch.full((1, _TINY_RANK), 1.0))


def test_h3_native_rejects_bad_metadata_and_ref2va(tmp_path):
    path = tmp_path / "minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors"
    _write_tiny_native(path, metadata={"key_format": "minimax-h3-native", "qkv_layout": "runtime"})
    with pytest.raises(ValueError, match="qkv_layout='grouped'"):
        load_minimax_h3_native_lora(
            partition="fl2va",
            lora_request=_request(path),
            lora_path=path,
            dtype=torch.float32,
        )

    valid = tmp_path / "valid.safetensors"
    _write_tiny_native(valid)
    with pytest.raises(ValueError, match="supports T2VA only"):
        load_minimax_h3_native_lora(
            partition="ref2va",
            lora_request=_request(valid),
            lora_path=valid,
            dtype=torch.float32,
        )


def test_h3_native_rejects_truncated_artifact(tmp_path):
    path = tmp_path / "minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors"
    _write_tiny_native(path, omit_target="blocks.49.mlp.fc2")
    with pytest.raises(ValueError, match="target set does not match"):
        load_minimax_h3_native_lora(
            partition="fl2va",
            lora_request=_request(path),
            lora_path=path,
            dtype=torch.float32,
        )


def test_h3_native_declared_file_fails_closed_on_invalid_metadata(tmp_path):
    path = tmp_path / "minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors"
    _write_tiny_native(path, metadata={"key_format": "other"})
    with pytest.raises(ValueError, match="requires safetensors metadata"):
        load_minimax_h3_native_lora(
            partition="fl2va",
            lora_request=_request(path),
            lora_path=path,
            dtype=torch.float32,
        )


def test_pipeline_native_schedule_and_task_validation(monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.partition = "fl2va"
    pipeline.supported_tasks = frozenset({"t2va", "fl2va", "ref2va"})
    pipeline._turbo_lora_adapter_ids = set()
    pipeline._native_lora_adapter_ids = {7}
    pipeline._lora_sigma_schedules = {7: DMD2SigmaSchedule.from_positions([1.0, 0.7, 0.4, 0.15, 0.0])}
    pipeline._base_schedule_by_partition = {"fl2va": None}

    sampling = SimpleNamespace(
        lora_request=LoRARequest("flashgen", 7, "/tmp/native.safetensors"),
        lora_scale=1.0,
        num_inference_steps=4,
        extra_args={},
    )
    assert pipeline._sigma_schedule_for_request(sampling, "t2va").num_inference_steps == 4
    with pytest.raises(OmniClientError, match="num_inference_steps must be 4"):
        pipeline._validate_native_sampling(
            SimpleNamespace(lora_request=sampling.lora_request, num_inference_steps=5),
            task="t2va",
        )
    with pytest.raises(OmniClientError, match="supports T2VA requests only"):
        pipeline._resolve_task(
            "fl2va",
            {},
            has_turbo_lora=False,
            has_native_lora=True,
        )

    pipeline._base_schedule_by_partition = {"fl2va": DMD2SigmaSchedule.from_positions([1.0, 0.5, 0.0])}
    with pytest.raises(OmniClientError, match="already pins base_schedule"):
        pipeline._sigma_schedule_for_request(sampling, "t2va")


def test_pipeline_replaces_native_classification_after_reload(monkeypatch, tmp_path):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as pipeline_module

    path = tmp_path / "minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors"
    _write_tiny_native(path)
    request = _request(path)

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.partition = "fl2va"
    pipeline.od_config = SimpleNamespace(
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
        enable_distributed_layerwise_offload=False,
    )
    pipeline._turbo_lora_adapter_ids = set()
    pipeline._native_lora_adapter_ids = set()
    pipeline._lora_sigma_schedules = {}

    loaded = pipeline._load_diffusion_lora_adapter(
        lora_request=request,
        lora_path=path,
        dtype=torch.float32,
    )
    assert loaded is not None
    assert request.lora_int_id in pipeline._native_lora_adapter_ids
    assert request.lora_int_id in pipeline._lora_sigma_schedules

    monkeypatch.setattr(pipeline_module, "load_minimax_h3_turbo_lora", lambda **_: None)
    monkeypatch.setattr(pipeline_module, "load_minimax_h3_native_lora", lambda **_: None)
    assert pipeline._load_diffusion_lora_adapter(lora_request=request, lora_path=path, dtype=torch.float32) is None
    assert request.lora_int_id not in pipeline._native_lora_adapter_ids
    assert request.lora_int_id not in pipeline._lora_sigma_schedules


@pytest.mark.parametrize(
    "offload_mode",
    [
        "model-level CPU offload (--enable-cpu-offload)",
        "layerwise offload (--enable-layerwise-offload)",
    ],
)
def test_h3_native_rejects_offload_modes(tmp_path, offload_mode):
    path = tmp_path / "minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors"
    _write_tiny_native(path)

    with pytest.raises(ValueError, match="does not support"):
        load_minimax_h3_native_lora(
            partition="fl2va",
            lora_request=_request(path),
            lora_path=path,
            dtype=torch.float32,
            unsupported_offload_mode=offload_mode,
        )


def test_h3_native_allows_distributed_layerwise_offload(monkeypatch):
    """DLO keeps LoRA A/B buffers resident, so native must not fail closed."""
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as pipeline_module

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.partition = "fl2va"
    pipeline.od_config = SimpleNamespace(
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
        enable_distributed_layerwise_offload=True,
    )
    pipeline._turbo_lora_adapter_ids = set()
    pipeline._native_lora_adapter_ids = set()
    pipeline._lora_sigma_schedules = {}
    captured: dict[str, object] = {}
    schedule = DMD2SigmaSchedule.from_positions([1.0, 0.7, 0.4, 0.15, 0.0])

    def load_native(**kwargs):
        captured.update(kwargs)
        return object(), object(), schedule

    monkeypatch.setattr(pipeline_module, "load_minimax_h3_turbo_lora", lambda **_: None)
    monkeypatch.setattr(pipeline_module, "load_minimax_h3_native_lora", load_native)

    request = _request("flashgen")
    loaded = pipeline._load_diffusion_lora_adapter(
        lora_request=request,
        lora_path="flashgen",
        dtype=torch.bfloat16,
    )

    assert loaded is not None
    assert captured["unsupported_offload_mode"] is None
    assert request.lora_int_id in pipeline._native_lora_adapter_ids
    assert pipeline._lora_sigma_schedules[request.lora_int_id] is schedule


def test_h3_native_qkv_reorder_matches_base_loader_contract():
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import _reorder_grouped_qkv_to_qkv

    num_query_groups = 56
    heads_per_group = 1
    head_dim = 128
    grouped = torch.arange(num_query_groups * (heads_per_group + 2) * head_dim, dtype=torch.float32)
    grouped = grouped.reshape(num_query_groups, (heads_per_group + 2) * head_dim)
    reordered = _reorder_grouped_qkv_to_qkv(
        grouped.reshape(-1, 1),
        num_query_groups=num_query_groups,
        heads_per_group=heads_per_group,
        head_dim=head_dim,
    ).reshape(-1)
    q_size = num_query_groups * heads_per_group * head_dim
    k_size = num_query_groups * head_dim
    q, k, v = torch.split(reordered, [q_size, k_size, k_size])
    assert q.reshape(num_query_groups, head_dim)[0, 0] == grouped[0, 0]
    assert k.reshape(num_query_groups, head_dim)[0, 0] == grouped[0, head_dim]
    assert v.reshape(num_query_groups, head_dim)[0, 0] == grouped[0, 2 * head_dim]


def test_legacy_manager_uses_native_loader(tmp_path):
    path = tmp_path / "minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors"
    _write_tiny_native(path)

    class _Pipeline:
        def _load_diffusion_lora_adapter(self, **kwargs):
            from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
                MiniMaxH3Pipeline,
            )

            pipeline = object.__new__(MiniMaxH3Pipeline)
            pipeline.partition = "fl2va"
            pipeline.od_config = SimpleNamespace(
                enable_cpu_offload=False,
                enable_layerwise_offload=False,
                enable_distributed_layerwise_offload=False,
            )
            pipeline._turbo_lora_adapter_ids = set()
            pipeline._native_lora_adapter_ids = set()
            pipeline._lora_sigma_schedules = {}
            return pipeline._load_diffusion_lora_adapter(**kwargs)

    manager = object.__new__(DiffusionLoRAManager)
    manager.pipeline = _Pipeline()
    manager.dtype = torch.float32
    manager._expected_lora_modules = {"qkv_proj", "fc1", "adaln_proj.linear"}

    lora_model, peft_helper = manager._load_adapter(_request(path))
    assert lora_model.id == 7
    assert peft_helper.lora_alpha == _TINY_RANK
    assert len(lora_model.loras) == 259


def test_pipeline_schedule_inactive_when_scale_zero():
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.partition = "fl2va"
    pipeline._turbo_lora_adapter_ids = set()
    pipeline._native_lora_adapter_ids = {7}
    pipeline._lora_sigma_schedules = {7: DMD2SigmaSchedule.from_positions([1.0, 0.7, 0.4, 0.15, 0.0])}
    pipeline._base_schedule_by_partition = {"fl2va": None}

    sampling = SimpleNamespace(
        lora_request=LoRARequest("flashgen", 7, "/tmp/native.safetensors"),
        lora_scale=0.0,
        num_inference_steps=4,
    )
    assert pipeline._sigma_schedule_for_request(sampling, "t2va") is None
    assert not pipeline._has_active_native_lora(sampling)


def test_pipeline_schedule_falls_back_after_eviction(monkeypatch, tmp_path):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as pipeline_module

    path = tmp_path / "minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors"
    _write_tiny_native(path)
    request = _request(path)

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.partition = "fl2va"
    pipeline.od_config = SimpleNamespace(
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
        enable_distributed_layerwise_offload=False,
    )
    pipeline._turbo_lora_adapter_ids = set()
    pipeline._native_lora_adapter_ids = set()
    pipeline._lora_sigma_schedules = {}
    pipeline._base_schedule_by_partition = {"fl2va": None}

    loaded = pipeline._load_diffusion_lora_adapter(
        lora_request=request,
        lora_path=path,
        dtype=torch.float32,
    )
    assert loaded is not None
    assert request.lora_int_id in pipeline._lora_sigma_schedules

    monkeypatch.setattr(pipeline_module, "load_minimax_h3_turbo_lora", lambda **_: None)
    monkeypatch.setattr(pipeline_module, "load_minimax_h3_native_lora", lambda **_: None)
    assert pipeline._load_diffusion_lora_adapter(lora_request=request, lora_path=path, dtype=torch.float32) is None
    assert request.lora_int_id not in pipeline._lora_sigma_schedules

    sampling = SimpleNamespace(
        lora_request=request,
        lora_scale=1.0,
        num_inference_steps=4,
    )
    assert pipeline._sigma_schedule_for_request(sampling, "t2va") is None


def test_native_packed_qkv_slices_are_tp2_divisible():
    assert lora_module._NATIVE_QKV_SLICE % 2 == 0
    assert lora_module._NATIVE_FFN_HIDDEN_SIZE % 2 == 0


def test_lora_manager_activates_native_packed_qkv(tmp_path):
    path = tmp_path / "minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors"
    _write_tiny_native(path)
    loaded = load_minimax_h3_native_lora(
        partition="fl2va",
        lora_request=_request(path),
        lora_path=path,
        dtype=torch.float32,
    )
    assert loaded is not None
    lora_model, _, _ = loaded
    packed = lora_model.get_lora("blocks.0.attn.qkv_proj")
    assert isinstance(packed, PackedLoRALayerWeights)
    assert len(packed.lora_b) == 3
    assert all(b.shape[0] == _TINY_QKV_SLICE for b in packed.lora_b)

    class _DummyLoRALayer:
        def __init__(self):
            self.n_slices = 3
            self.output_slices = (_TINY_QKV_SLICE, _TINY_QKV_SLICE, _TINY_QKV_SLICE)
            self.set_calls: list[tuple[list, list]] = []
            self.reset_calls = 0

        def reset_lora(self, index: int):
            self.reset_calls += 1

        def set_lora(self, index: int, lora_a, lora_b):
            self.set_calls.append((lora_a, lora_b))

    layer = _DummyLoRALayer()
    manager = object.__new__(DiffusionLoRAManager)
    manager.pipeline = torch.nn.Module()
    manager._lora_modules = {"blocks.0.attn.qkv_proj": layer}
    manager._registered_adapters = {
        lora_model.id: lora_model,
    }
    manager._activate_adapter(lora_model.id, scale=0.5)

    assert len(layer.set_calls) == 1
    lora_a_list, lora_b_list = layer.set_calls[0]
    assert len(lora_a_list) == 3
    assert len(lora_b_list) == 3
    torch.testing.assert_close(lora_b_list[0], packed.lora_b[0] * 0.5)
    torch.testing.assert_close(lora_b_list[1], packed.lora_b[1] * 0.5)
    torch.testing.assert_close(lora_b_list[2], packed.lora_b[2] * 0.5)


def test_lora_manager_tp2_splits_native_packed_qkv_per_rank():
    """Each Q/K/V slice must be divisible by TP2 so rank-local layers can shard rows."""

    tp_size = 2
    slice_rows = lora_module._NATIVE_QKV_SLICE
    assert slice_rows % tp_size == 0
    local_rows = slice_rows // tp_size

    class _TpQkvLoRALayer:
        def __init__(self, tp_rank: int):
            self.tp_rank = tp_rank
            self.n_slices = 3
            self.output_slices = (local_rows, local_rows, local_rows)
            self.set_calls: list[tuple[list, list]] = []

        def reset_lora(self, index: int):
            return

        def set_lora(self, index: int, lora_a, lora_b):
            self.set_calls.append((lora_a, lora_b))

    for tp_rank in range(tp_size):
        layer = _TpQkvLoRALayer(tp_rank)
        start = tp_rank * local_rows
        end = start + local_rows
        full_packed = PackedLoRALayerWeights(
            module_name="blocks.0.attn.qkv_proj",
            rank=lora_module._NATIVE_RANK,
            lora_alphas=[64, 64, 64],
            lora_a=[torch.ones(lora_module._NATIVE_RANK, lora_module._NATIVE_HIDDEN_SIZE)] * 3,
            lora_b=[torch.full((slice_rows, lora_module._NATIVE_RANK), float(i + 1)) for i in range(3)],
            scaling=[1.0, 1.0, 1.0],
        )
        local_b = [b[start:end].contiguous() for b in full_packed.lora_b]
        layer.set_lora(0, full_packed.lora_a, local_b)
        assert len(layer.set_calls) == 1
        _, lora_b_list = layer.set_calls[0]
        assert all(b.shape[0] == local_rows for b in lora_b_list)
        assert lora_b_list[0][0, 0].item() == 1.0
        assert lora_b_list[1][0, 0].item() == 2.0
        assert lora_b_list[2][0, 0].item() == 3.0
