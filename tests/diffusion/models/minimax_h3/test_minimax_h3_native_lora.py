# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from vllm.lora.lora_weights import PackedLoRALayerWeights

from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.models.minimax_h3.npu import lora as lora_module
from vllm_omni.diffusion.models.minimax_h3.npu.lora import load_minimax_h3_native_lora
from vllm_omni.diffusion.sched.sigma_schedule import DMD2SigmaSchedule
from vllm_omni.errors import OmniClientError
from vllm_omni.lora.request import LoRARequest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_TINY_HIDDEN = 2
_TINY_FFN = 2
_TINY_TIME = 2
_TINY_BLOCK_ADALN = 4
_TINY_FINAL_ADALN = 3
_TINY_RANK = 4
_TINY_NUM_QUERY_GROUPS = 2
_TINY_HEADS_PER_GROUP = 1
_TINY_HEAD_DIM = 1
_TINY_QKV_SLICE = _TINY_NUM_QUERY_GROUPS * _TINY_HEADS_PER_GROUP * _TINY_HEAD_DIM
# Grouped qkv stores (heads_per_group + 2) * head_dim rows per query group, so
# the attention inner size follows from the head layout and cannot be picked
# independently: 3 * inner must equal the grouped row count.
_TINY_INNER = _TINY_QKV_SLICE


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

    # The writer tags group g with q=g*10+1, k=g*10+2, v=g*10+3, so a correct
    # grouped -> q/k/v permutation yields slices that are uniform per role.
    qkv = lora_model.get_lora("blocks.0.attn.qkv_proj")
    assert isinstance(qkv, PackedLoRALayerWeights)
    torch.testing.assert_close(qkv.lora_b[0], torch.tensor([[1.0] * _TINY_RANK, [11.0] * _TINY_RANK]))
    torch.testing.assert_close(qkv.lora_b[1], torch.tensor([[2.0] * _TINY_RANK, [12.0] * _TINY_RANK]))
    torch.testing.assert_close(qkv.lora_b[2], torch.tensor([[3.0] * _TINY_RANK, [13.0] * _TINY_RANK]))

    fc1 = lora_model.get_lora("blocks.0.mlp.fc1")
    assert isinstance(fc1, PackedLoRALayerWeights)
    torch.testing.assert_close(fc1.lora_b[0], torch.full((_TINY_FFN, _TINY_RANK), 2.0))
    torch.testing.assert_close(fc1.lora_b[1], torch.full((_TINY_FFN, _TINY_RANK), 1.0))


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

    # The exact-filename guard runs before the partition check, so the artifact
    # has to keep the published name for this to reach the Ref2VA rejection.
    valid_dir = tmp_path / "valid"
    valid_dir.mkdir()
    valid = valid_dir / "minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors"
    _write_tiny_native(valid)
    with pytest.raises(ValueError, match="supports T2VA only"):
        load_minimax_h3_native_lora(
            partition="ref2va",
            lora_request=_request(valid),
            lora_path=valid,
            dtype=torch.float32,
        )

    renamed = tmp_path / "renamed.safetensors"
    _write_tiny_native(renamed)
    with pytest.raises(ValueError, match="supports only"):
        load_minimax_h3_native_lora(
            partition="fl2va",
            lora_request=_request(renamed),
            lora_path=renamed,
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


def test_h3_native_rejects_schedule_with_wrong_interval_count(tmp_path):
    """A mislabeled schedule must fail closed instead of silently changing steps."""
    path = tmp_path / "minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors"
    _write_tiny_native(
        path,
        metadata={
            "format": "pt",
            "key_format": "minimax-h3-native",
            "qkv_layout": "grouped",
            "lora_rank": str(_TINY_RANK),
            "lora_alpha": str(_TINY_RANK),
            "base_schedule": "1.0,0.5,0.0",
            "tasks": "t2va",
        },
    )
    with pytest.raises(ValueError, match="4-interval base_schedule"):
        load_minimax_h3_native_lora(
            partition="fl2va",
            lora_request=_request(path),
            lora_path=path,
            dtype=torch.float32,
        )


@pytest.mark.parametrize(
    ("raw_schedule", "message"),
    [
        (None, "requires safetensors metadata base_schedule"),
        ("", "must not be empty"),
        (",,", "must not be empty"),
        ("1.0,oops,0.15,0.0", "is malformed"),
        ("0.9,0.7,0.4,0.15,0.0", "is malformed"),
    ],
)
def test_h3_native_rejects_malformed_schedule_metadata(tmp_path, raw_schedule, message):
    """safetensors metadata is string-only, so the loader owns the parse contract."""
    metadata = {
        "format": "pt",
        "key_format": "minimax-h3-native",
        "qkv_layout": "grouped",
        "lora_rank": str(_TINY_RANK),
        "lora_alpha": str(_TINY_RANK),
        "tasks": "t2va",
    }
    if raw_schedule is not None:
        metadata["base_schedule"] = raw_schedule
    path = tmp_path / "minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors"
    _write_tiny_native(path, metadata=metadata)

    with pytest.raises(ValueError, match=message):
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


def test_pipeline_native_step_count_follows_adapter_schedule():
    """The accepted step count is the adapter's interval count, never a literal."""
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.partition = "fl2va"
    pipeline._native_lora_adapter_ids = {7}
    pipeline._lora_sigma_schedules = {7: DMD2SigmaSchedule.from_positions([1.0, 0.5, 0.0])}
    request = LoRARequest("flashgen", 7, "/tmp/native.safetensors")

    pipeline._validate_native_sampling(
        SimpleNamespace(lora_request=request, num_inference_steps=2),
        task="t2va",
    )
    with pytest.raises(OmniClientError, match="num_inference_steps must be 2 or omitted, not 3"):
        pipeline._validate_native_sampling(
            SimpleNamespace(lora_request=request, num_inference_steps=3),
            task="t2va",
        )
    with pytest.raises(OmniClientError, match="requires num_inference_steps=2"):
        pipeline._validate_native_sampling(
            SimpleNamespace(lora_request=request, num_inference_steps=4),
            task="t2va",
        )


@pytest.mark.parametrize("step_execution", [False, True])
def test_pipeline_native_omitted_step_count_follows_execution_mode(step_execution):
    """Only request mode can resolve an omitted count from the adapter schedule.

    ``StepScheduler`` reads ``num_inference_steps`` off the request at admission,
    before this pipeline sees it, so under step execution an omitted value can
    only disagree with the denoise loop. The advertised contract has to match.
    """
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.partition = "fl2va"
    pipeline._native_lora_adapter_ids = {7}
    pipeline._lora_sigma_schedules = {7: DMD2SigmaSchedule.from_positions([1.0, 0.7, 0.4, 0.15, 0.0])}
    pipeline.od_config = SimpleNamespace(step_execution=step_execution)
    sampling = SimpleNamespace(
        lora_request=LoRARequest("flashgen", 7, "/tmp/native.safetensors"),
        num_inference_steps=None,
    )

    if not step_execution:
        # Request mode defaults to the adapter's own interval count.
        pipeline._validate_native_sampling(sampling, task="t2va")
        return

    with pytest.raises(OmniClientError, match="explicit num_inference_steps=4 under step execution"):
        pipeline._validate_native_sampling(sampling, task="t2va")
    # The step-mode message must not offer omission it cannot accept.
    with pytest.raises(OmniClientError, match=r"must be 4, not 5$"):
        pipeline._validate_native_sampling(
            SimpleNamespace(lora_request=sampling.lora_request, num_inference_steps=5),
            task="t2va",
        )


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


def test_native_target_pattern_matches_component_qualified_names():
    """The manager matches "<component>.<module>", so the pattern needs the prefix."""
    import regex as re

    pattern = lora_module._NATIVE_TARGET_PATTERN
    matched = [
        target
        for target in lora_module._NATIVE_EXPECTED_TARGETS
        if re.search(pattern, f"transformer.{target}") is not None
    ]
    assert len(matched) == 259
    assert re.search(pattern, "transformer.blocks.0.attn.q_proj") is None
    assert re.search(pattern, "transformer.token_refiner.blocks.0.adaln_proj.linear") is None


def test_replace_layers_binds_every_native_target(monkeypatch):
    """Regression for bound=0/259: layer replacement must reach all 259 modules."""
    from vllm.lora.layers import BaseLayerWithLoRA
    from vllm.lora.peft_helper import PEFTHelper

    from vllm_omni.diffusion.lora import manager as manager_module

    class _StubLoRALayer(BaseLayerWithLoRA):
        def __init__(self, base_layer):
            super().__init__()
            self.base_layer = base_layer

    def _linear() -> torch.nn.Module:
        return torch.nn.Linear(_TINY_HIDDEN, _TINY_HIDDEN)

    def _block(*, with_adaln: bool) -> torch.nn.Module:
        block = torch.nn.Module()
        block.attn = torch.nn.Module()
        block.attn.qkv_proj = _linear()
        block.attn.out_proj = _linear()
        block.mlp = torch.nn.Module()
        block.mlp.fc1 = _linear()
        block.mlp.fc2 = _linear()
        if with_adaln:
            block.adaln_proj = torch.nn.Module()
            block.adaln_proj.linear = _linear()
        return block

    transformer = torch.nn.Module()
    transformer.blocks = torch.nn.ModuleList([_block(with_adaln=True) for _ in range(50)])
    transformer.token_refiner = torch.nn.Module()
    transformer.token_refiner.blocks = torch.nn.ModuleList([_block(with_adaln=False) for _ in range(2)])
    transformer.final_layer = torch.nn.Module()
    transformer.final_layer.adaln_proj = torch.nn.Module()
    transformer.final_layer.adaln_proj.linear = _linear()
    pipeline = torch.nn.Module()
    pipeline.transformer = transformer

    monkeypatch.setattr(manager_module, "from_layer_diffusion", lambda *, layer, **_: _StubLoRALayer(layer))

    manager = object.__new__(DiffusionLoRAManager)
    manager.pipeline = pipeline
    manager.dtype = torch.float32
    manager.max_cached_adapters = 1
    manager._lora_modules = {}
    manager._max_lora_rank = 64
    manager._resident_lora_device = None

    peft_helper = PEFTHelper.from_dict(
        {
            "r": _TINY_RANK,
            "lora_alpha": _TINY_RANK,
            "target_modules": lora_module._NATIVE_TARGET_PATTERN,
        }
    )
    manager._replace_layers_with_lora(peft_helper)

    assert set(manager._lora_modules) == {f"transformer.{target}" for target in lora_module._NATIVE_EXPECTED_TARGETS}


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
    manager._active_adapter_id = None
    manager._adapter_scales = {}
    manager._activate_adapter(lora_model.id, scale=0.5)

    assert manager._active_adapter_id == lora_model.id

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


# 259 modules, of which 52 qkv expand to 3 slices and 52 fc1 expand to 2.
_EXPECTED_NATIVE_SLICE_PAIRS = 415


def _iter_native_slice_pairs(lora_model):
    """Yield ``(module_name, lora_a, lora_b)`` per slice, flattening packed weights."""
    for module_name, weights in lora_model.loras.items():
        lora_a = weights.lora_a if isinstance(weights.lora_a, list) else [weights.lora_a]
        lora_b = weights.lora_b if isinstance(weights.lora_b, list) else [weights.lora_b]
        for a, b in zip(lora_a, lora_b, strict=True):
            yield module_name, a, b


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_native_lora_load_is_platform_agnostic(tmp_path, dtype):
    """The artifact is trained on Ascend, but loading it is not Ascend-only.

    This whole module runs in the CPU job with no ``torch_npu`` present, so a
    passing run is itself the evidence. The assertions pin the property that
    makes it portable: tensors land on CPU in exactly the requested dtype and
    the loader never picks a device, leaving placement to the pipeline.
    """
    path = tmp_path / "minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors"
    _write_tiny_native(path)

    loaded = load_minimax_h3_native_lora(
        partition="fl2va",
        lora_request=_request(path),
        lora_path=path,
        dtype=dtype,
    )

    assert loaded is not None
    lora_model, _, _ = loaded
    tensors = {
        f"{module_name}.{side}": tensor
        for module_name, a, b in _iter_native_slice_pairs(lora_model)
        for side, tensor in (("lora_a", a), ("lora_b", b))
    }
    assert sum(1 for _ in _iter_native_slice_pairs(lora_model)) == _EXPECTED_NATIVE_SLICE_PAIRS
    assert {name: tensor.device.type for name, tensor in tensors.items() if tensor.device.type != "cpu"} == {}
    assert {name: tensor.dtype for name, tensor in tensors.items() if tensor.dtype != dtype} == {}


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_native_lora_delta_matches_between_cpu_and_cuda(tmp_path):
    """The packed adapter produces the same delta weight on CUDA as on CPU.

    Covers the GPU side of the same artifact: every ``lora_b @ lora_a`` product,
    including the reordered q/k/v slices, is recomputed on CUDA and compared
    against the CPU result, so a GPU deployment cannot silently diverge.
    """
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
    checked = 0
    for module_name, a, b in _iter_native_slice_pairs(lora_model):
        cpu_delta = b @ a
        cuda_delta = b.cuda() @ a.cuda()
        assert cuda_delta.device.type == "cuda"
        torch.testing.assert_close(cuda_delta.cpu(), cpu_delta, msg=f"delta mismatch for {module_name}")
        checked += 1
    assert checked == _EXPECTED_NATIVE_SLICE_PAIRS
