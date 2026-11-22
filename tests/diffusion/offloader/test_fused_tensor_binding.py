# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Fused many-to-one checkpoint bindings on the checkpoint-mmap path.

A text encoder fuses ``q/k/v`` into ``qkv_proj`` and ``gate/up`` into
``gate_up_proj`` by row concatenation, so one runtime tensor is backed by
several checkpoint entries. These tests pin the plan shape, the preflight
validation, and the staged values against the concatenation the ordinary
loader would have produced.
"""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.model_loader.checkpoint_adapters.direct_mmap import (
    DirectMmapTensorPolicy,
    _MiniMaxH3DirectMmapAdapter,
)
from vllm_omni.diffusion.model_loader.host_weight_plan import (
    FusedTensorBinding,
    build_checkpoint_mmap_plan,
)
from vllm_omni.diffusion.offloader.distributed_layerwise_backend import (
    DistributedLayerwiseOffloadHook,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]

CUDA = hardware_marks(res={"cuda": "H100"}, num_cards=1)

_HIDDEN = 8
_Q_ROWS = 4
_KV_ROWS = 2
_UP_ROWS = 6


class _FusedEncoder(nn.Module):
    """Minimal encoder shaped like the MiniMax-H3 text encoder blocks."""

    def __init__(self) -> None:
        super().__init__()
        self.text_model = nn.Module()
        self.text_model.layers = nn.ModuleList([_FusedEncoderLayer()])

    def map_checkpoint_weight_name(self, name: str) -> tuple[str, str | int | None] | None:
        prefix = "model.language_model."
        if not name.startswith(prefix):
            return None
        rest = "text_model." + name[len(prefix) :]
        if rest.endswith(".self_attn.q_proj.weight"):
            return (rest.replace("q_proj", "qkv_proj"), "q")
        if rest.endswith(".self_attn.k_proj.weight"):
            return (rest.replace("k_proj", "qkv_proj"), "k")
        if rest.endswith(".self_attn.v_proj.weight"):
            return (rest.replace("v_proj", "qkv_proj"), "v")
        if rest.endswith(".mlp.gate_proj.weight"):
            return (rest.replace("gate_proj", "gate_up_proj"), 0)
        if rest.endswith(".mlp.up_proj.weight"):
            return (rest.replace("up_proj", "gate_up_proj"), 1)
        return (rest, None)


class _FusedEncoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.qkv_proj = nn.Linear(_HIDDEN, _Q_ROWS + 2 * _KV_ROWS, bias=False)
        self.mlp = nn.Module()
        self.mlp.gate_up_proj = nn.Linear(_HIDDEN, 2 * _UP_ROWS, bias=False)


def _write_checkpoint(tmp_path) -> tuple[object, dict[str, torch.Tensor]]:
    """Write one encoder checkpoint file and return its loader source."""
    slices = {
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(_Q_ROWS, _HIDDEN),
        "model.language_model.layers.0.self_attn.k_proj.weight": torch.randn(_KV_ROWS, _HIDDEN),
        "model.language_model.layers.0.self_attn.v_proj.weight": torch.randn(_KV_ROWS, _HIDDEN),
        "model.language_model.layers.0.mlp.gate_proj.weight": torch.randn(_UP_ROWS, _HIDDEN),
        "model.language_model.layers.0.mlp.up_proj.weight": torch.randn(_UP_ROWS, _HIDDEN),
    }
    checkpoint_dir = tmp_path / "text_encoder"
    checkpoint_dir.mkdir(exist_ok=True)
    save_file(slices, str(checkpoint_dir / "model.safetensors"))
    source = SimpleNamespace(
        model_or_path=str(tmp_path),
        subfolder="text_encoder",
        prefix="text_encoder.",
    )
    return source, slices


def _build_plan(tmp_path, monkeypatch):
    import vllm_omni.diffusion.model_loader.host_weight_plan as plan_module

    encoder = _FusedEncoder()
    pipeline = nn.Module()
    pipeline.text_encoder = encoder
    adapter = _MiniMaxH3DirectMmapAdapter(pipeline)
    monkeypatch.setattr(plan_module, "get_direct_mmap_adapter", lambda _pipeline: adapter)
    source, slices = _write_checkpoint(tmp_path)
    result = build_checkpoint_mmap_plan(
        pipeline,
        dit_modules=(),
        encoder_modules=(("text_encoder", encoder),),
        sources=(source,),
        model_path=None,
        tensor_parallel_size=1,
        use_hsdp=False,
        online_quantization=False,
    )
    return result, slices


def test_encoder_fused_policy_declares_row_slices() -> None:
    adapter = _MiniMaxH3DirectMmapAdapter(nn.Identity())

    qkv = adapter.policy_for("text_encoder.text_model.layers.0.self_attn.qkv_proj.weight", torch.empty(0))
    gate_up = adapter.policy_for("text_encoder.text_model.layers.0.mlp.gate_up_proj.weight", torch.empty(0))

    assert qkv.fused_sources == ("q_proj", "k_proj", "v_proj")
    assert qkv.transform is None
    assert gate_up.fused_sources == ("gate_proj", "up_proj")
    assert gate_up.transform is None
    fused = qkv.fuse([torch.zeros(2, 3), torch.ones(1, 3)])
    assert fused.shape == (3, 3)
    assert torch.equal(fused[2], torch.ones(3))


def test_plan_builds_fused_bindings_in_loader_order(tmp_path, monkeypatch) -> None:
    result, _ = _build_plan(tmp_path, monkeypatch)

    assert result.fallback_reason is None
    bindings = result.plan.bindings
    qkv = bindings["text_encoder.text_model.layers.0.self_attn.qkv_proj.weight"]
    gate_up = bindings["text_encoder.text_model.layers.0.mlp.gate_up_proj.weight"]
    assert isinstance(qkv, FusedTensorBinding)
    assert isinstance(gate_up, FusedTensorBinding)
    assert qkv.checkpoint_keys == (
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.self_attn.k_proj.weight",
        "model.language_model.layers.0.self_attn.v_proj.weight",
    )
    assert gate_up.checkpoint_keys == (
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.0.mlp.up_proj.weight",
    )
    assert qkv.file_paths == tuple(
        str(tmp_path / "text_encoder" / "model.safetensors") for _ in range(len(qkv.checkpoint_keys))
    )
    assert len(gate_up.file_paths) == 2


def test_plan_fails_closed_without_a_fusion_policy(tmp_path, monkeypatch) -> None:
    import vllm_omni.diffusion.model_loader.host_weight_plan as plan_module

    class _PlainAdapter:
        def policy_for(self, runtime_name, target):
            del runtime_name, target
            return DirectMmapTensorPolicy(allow_custom_loader=True)

    encoder = _FusedEncoder()
    pipeline = nn.Module()
    pipeline.text_encoder = encoder
    source, _ = _write_checkpoint(tmp_path)
    monkeypatch.setattr(plan_module, "get_direct_mmap_adapter", lambda _pipeline: _PlainAdapter())

    result = build_checkpoint_mmap_plan(
        pipeline,
        dit_modules=(),
        encoder_modules=(("text_encoder", encoder),),
        sources=(source,),
        model_path=None,
        tensor_parallel_size=1,
        use_hsdp=False,
        online_quantization=False,
    )

    assert result.plan is None
    assert "multiple checkpoint entries map to runtime tensor" in result.fallback_reason


def test_plan_fails_closed_without_an_encoder_mapping(tmp_path, monkeypatch) -> None:
    import vllm_omni.diffusion.model_loader.host_weight_plan as plan_module

    encoder = _FusedEncoder()
    setattr(encoder, "map_checkpoint_weight_name", None)
    pipeline = nn.Module()
    pipeline.text_encoder = encoder
    source, _ = _write_checkpoint(tmp_path)
    monkeypatch.setattr(
        plan_module,
        "get_direct_mmap_adapter",
        lambda _pipeline: _MiniMaxH3DirectMmapAdapter(pipeline),
    )

    result = build_checkpoint_mmap_plan(
        pipeline,
        dit_modules=(),
        encoder_modules=(("text_encoder", encoder),),
        sources=(source,),
        model_path=None,
        tensor_parallel_size=1,
        use_hsdp=False,
        online_quantization=False,
    )

    assert result.plan is None
    assert "without a checkpoint-name mapping" in result.fallback_reason


def test_plan_fails_closed_on_fused_shape_mismatch(tmp_path, monkeypatch) -> None:
    result, _ = _build_plan(tmp_path, monkeypatch)
    assert result.plan is not None

    encoder = _FusedEncoder()
    encoder.text_model.layers[0].self_attn.qkv_proj = nn.Linear(_HIDDEN, _Q_ROWS + 2 * _KV_ROWS + 1, bias=False)
    pipeline = nn.Module()
    pipeline.text_encoder = encoder
    import vllm_omni.diffusion.model_loader.host_weight_plan as plan_module

    monkeypatch.setattr(
        plan_module,
        "get_direct_mmap_adapter",
        lambda _pipeline: _MiniMaxH3DirectMmapAdapter(pipeline),
    )
    source, _ = _write_checkpoint(tmp_path)

    mismatch = build_checkpoint_mmap_plan(
        pipeline,
        dit_modules=(),
        encoder_modules=(("text_encoder", encoder),),
        sources=(source,),
        model_path=None,
        tensor_parallel_size=1,
        use_hsdp=False,
        online_quantization=False,
    )

    assert mismatch.plan is None
    assert "shape mismatch for fused tensor" in mismatch.fallback_reason


def test_staged_fused_values_match_row_concatenation(tmp_path, monkeypatch) -> None:
    result, slices = _build_plan(tmp_path, monkeypatch)
    assert result.plan is not None
    plan = result.plan
    qkv_name = "text_encoder.text_model.layers.0.self_attn.qkv_proj.weight"
    gate_up_name = "text_encoder.text_model.layers.0.mlp.gate_up_proj.weight"

    # Placeholder parameters mirror what the mmap realization leaves behind: a
    # value-less tensor whose shape and dtype the staging path validates.
    qkv_param = nn.Parameter(torch.empty(_Q_ROWS + 2 * _KV_ROWS, _HIDDEN))
    gate_up_param = nn.Parameter(torch.empty(2 * _UP_ROWS, _HIDDEN))
    fused_sources = {
        id(qkv_param): _fused_source(plan.bindings[qkv_name], tmp_path),
        id(gate_up_param): _fused_source(plan.bindings[gate_up_name], tmp_path),
    }

    cpu_sources, metadata = DistributedLayerwiseOffloadHook._collect_mmap_sources(
        {"qkv": qkv_param, "gate_up": gate_up_param},
        {},
        None,
        fused_sources,
    )

    dtype = qkv_param.dtype
    total_numel = sum(meta["numel"] for meta in metadata[dtype])
    slot = torch.zeros(total_numel, dtype=dtype)
    staged = DistributedLayerwiseOffloadHook._pack_mmap_sources(cpu_sources, metadata, {dtype: slot})[dtype]

    staged_qkv = staged[: metadata[dtype][0]["numel"]].view_as(qkv_param)
    expected_qkv = torch.cat(
        [
            slices["model.language_model.layers.0.self_attn.q_proj.weight"],
            slices["model.language_model.layers.0.self_attn.k_proj.weight"],
            slices["model.language_model.layers.0.self_attn.v_proj.weight"],
        ],
        dim=0,
    )
    assert torch.equal(staged_qkv, expected_qkv)

    gate_up_offset = metadata[dtype][1]["offset"]
    staged_gate_up = staged[gate_up_offset : gate_up_offset + gate_up_param.numel()].view_as(gate_up_param)
    expected_gate_up = torch.cat(
        [
            slices["model.language_model.layers.0.mlp.gate_proj.weight"],
            slices["model.language_model.layers.0.mlp.up_proj.weight"],
        ],
        dim=0,
    )
    assert torch.equal(staged_gate_up, expected_gate_up)


def _fused_source(binding: FusedTensorBinding, _tmp_path):
    from vllm_omni.diffusion.offloader.distributed_layerwise_backend import _FusedMmapSource

    tensors = []
    for checkpoint_key, file_path in zip(binding.checkpoint_keys, binding.file_paths, strict=True):
        from safetensors import safe_open

        with safe_open(file_path, framework="pt", device="cpu") as handle:
            tensors.append(handle.get_tensor(checkpoint_key))
    return _FusedMmapSource.build(slices=tuple(tensors), fuse=binding.fuse)
