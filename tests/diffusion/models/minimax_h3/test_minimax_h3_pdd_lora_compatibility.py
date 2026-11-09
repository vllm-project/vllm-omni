# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reject competing final-head ownership before mutating live adapters."""

import gc
import weakref
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from vllm.lora.layers import BaseLayerWithLoRA
from vllm.lora.lora_model import LoRAModel
from vllm.lora.lora_weights import LoRALayerWeights

from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.models.minimax_h3 import pdd_artifact_lifecycle as lifecycle
from vllm_omni.diffusion.models.minimax_h3.pdd_artifact import PDDAdapter, PDDConfig, PDDParallelHead
from vllm_omni.diffusion.models.minimax_h3.pdd_artifact_lifecycle import MiniMaxH3PDDLifecycleMixin
from vllm_omni.errors import OmniClientError, client_error_metadata

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_GENERIC_ID = 10
_PDD_ID = 20


class _GenericHead(BaseLayerWithLoRA):
    def __init__(self, base_layer):
        nn.Module.__init__(self)
        self.base_layer = base_layer

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x):
        return self.base_layer(x) + 1, None


def _dit():
    model = nn.Module()
    model.arch = SimpleNamespace(pdd_steps=None)
    model.final_layer = nn.Module()
    for name in ("video_out", "audio_out"):
        head = nn.Linear(2, 1)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        setattr(model.final_layer, name, head)
    return model


class _Pipeline(MiniMaxH3PDDLifecycleMixin, nn.Module):
    def __init__(self):
        super().__init__()
        self.partition = "combined"
        self.transformer = _dit()
        self.transformers_ref = _dit()
        self._ensure_pdd_bookkeeping()
        self._native_lora_adapter_ids = set()
        self._turbo_lora_specs = {}


def _install_pdd(pipeline, component="transformer"):
    cfg = PDDConfig(dit_component=component)
    adapter = PDDAdapter(cfg, _PDD_ID, *cfg.plans())
    dit = getattr(pipeline, component)
    adapter.install_heads(dit)
    adapter.load_head_bank(
        dit,
        {name: torch.full((32, 1, 2), float(i + 1)) for i, name in enumerate(("video_out", "audio_out"))},
        {name: torch.zeros(32, 1) for name in ("video_out", "audio_out")},
    )
    return adapter


def _snapshot(dit):
    x = torch.ones(1, 2)
    result = []
    for name in ("video_out", "audio_out"):
        head = getattr(dit.final_layer, name)
        base = head(x)
        base = base[0] if isinstance(base, tuple) else base
        pdd = head(x, 0, _PDD_ID)[0] if isinstance(head, PDDParallelHead) else None
        result.append((head, base.detach().clone(), None if pdd is None else pdd.detach().clone()))
    return result


def _assert_unchanged(dit, before):
    for name, (head, base, pdd) in zip(("video_out", "audio_out"), before):
        assert getattr(dit.final_layer, name) is head
        actual = head(torch.ones(1, 2))
        torch.testing.assert_close(actual[0] if isinstance(actual, tuple) else actual, base)
        if pdd is not None:
            torch.testing.assert_close(head(torch.ones(1, 2), 0, _PDD_ID)[0], pdd)


def _manager(pipeline, target, monkeypatch):
    weights = LoRALayerWeights(target, 1, 1, torch.ones(1, 2), torch.ones(1, 1))
    manager = object.__new__(DiffusionLoRAManager)
    manager.pipeline = pipeline
    manager._registered_adapters = {_GENERIC_ID: LoRAModel(_GENERIC_ID, 1, {target: weights})}
    manager._active_adapter_id = _PDD_ID
    manager._suspended_adapter_id = None
    manager._active_scale = 1.0
    manager._adapter_scales = {_GENERIC_ID: 1.0, _PDD_ID: 1.0}
    manager._merged_adapters = {}
    manager._lora_modules = {}
    calls = {}
    for name in (
        "_bind_adapter_weights",
        "_reset_lora_layers",
        "_publish_moe_lora_context",
        "_deactivate_all_adapters",
    ):
        calls[name] = Mock()
        monkeypatch.setattr(manager, name, calls[name])
    return manager, calls


@pytest.mark.parametrize("existing_head", ["video_out", "audio_out", "checkpoint"])
def test_pdd_install_rejects_competing_heads_before_replacing_either_head(existing_head):
    pipeline = _Pipeline()
    layer = pipeline.transformer.final_layer
    message = "generic PEFT LoRA"
    if existing_head == "checkpoint":
        pipeline.transformer.arch.pdd_steps = 32
        message = "checkpoint heads"
    else:
        setattr(layer, existing_head, _GenericHead(getattr(layer, existing_head)))
    before = _snapshot(pipeline.transformer)
    cfg = PDDConfig()

    with pytest.raises(OmniClientError, match=message) as exc:
        PDDAdapter(cfg, _PDD_ID, *cfg.plans()).install_heads(pipeline.transformer)

    assert client_error_metadata(exc.value) == (400, "BadRequestError")
    _assert_unchanged(pipeline.transformer, before)


@pytest.mark.parametrize(
    "target,active,suspended",
    [
        ("transformer.final_layer.video_out", _PDD_ID, None),
        ("final_layer.audio_out", _GENERIC_ID, None),
        ("video_out", None, _GENERIC_ID),
    ],
)
def test_generic_head_activation_cannot_bypass_pdd_guard_or_reset_current_state(target, active, suspended, monkeypatch):
    pipeline = _Pipeline()
    _install_pdd(pipeline)
    before = _snapshot(pipeline.transformer)
    manager, calls = _manager(pipeline, target, monkeypatch)
    manager._active_adapter_id, manager._suspended_adapter_id = active, suspended
    scales = dict(manager._adapter_scales)

    with pytest.raises(OmniClientError, match="Generic PEFT LoRA") as exc:
        manager._activate_adapter(_GENERIC_ID, 1.0)

    assert client_error_metadata(exc.value) == (400, "BadRequestError")
    assert (manager._active_adapter_id, manager._suspended_adapter_id) == (active, suspended)
    assert manager._active_scale == 1.0
    assert manager._adapter_scales == scales
    for call in calls.values():
        call.assert_not_called()
    _assert_unchanged(pipeline.transformer, before)


@pytest.mark.parametrize(
    "pdd_component,target,kind",
    [
        ("transformers_ref", "transformer.final_layer.video_out", "generic"),
        ("transformer", "transformer.blocks.0.attn.qkv_proj", "native"),
        ("transformer", "transformer.blocks.0.mlp.fc1", "turbo"),
    ],
)
def test_other_dit_head_and_non_head_adapters_still_reach_binding(pdd_component, target, kind, monkeypatch):
    pipeline = _Pipeline()
    _install_pdd(pipeline, pdd_component)
    before = _snapshot(getattr(pipeline, pdd_component))
    if kind == "native":
        pipeline._native_lora_adapter_ids.add(_GENERIC_ID)
    elif kind == "turbo":
        pipeline._turbo_lora_specs[_GENERIC_ID] = SimpleNamespace()
    manager, calls = _manager(pipeline, target, monkeypatch)

    manager._activate_adapter(_GENERIC_ID, 1.0)

    calls["_bind_adapter_weights"].assert_called_once_with(manager._registered_adapters[_GENERIC_ID], 1.0)
    calls["_reset_lora_layers"].assert_not_called()
    assert manager._active_adapter_id == _GENERIC_ID
    _assert_unchanged(getattr(pipeline, pdd_component), before)


@pytest.mark.parametrize("already_registered", [False, True])
def test_rejected_pdd_load_preserves_head_objects_and_metadata(already_registered, monkeypatch):
    pipeline = _Pipeline()
    layer = pipeline.transformer.final_layer
    layer.audio_out = _GenericHead(layer.audio_out)
    cfg = PDDConfig()
    if already_registered:
        pipeline._pdd_adapters[_PDD_ID] = {"cfg": cfg, "marker": "preserved"}
        pipeline._pdd_adapter_ids.add(_PDD_ID)
        pipeline._pdd_active_adapters[_PDD_ID] = PDDAdapter(cfg, _PDD_ID, *cfg.plans())
    metadata = dict(pipeline._pdd_adapters), set(pipeline._pdd_adapter_ids), dict(pipeline._pdd_active_adapters)
    before = _snapshot(pipeline.transformer)
    monkeypatch.setattr(lifecycle, "load_minimax_h3_pdd_lora", lambda **kwargs: (object(), object(), cfg, {}, {}))

    with pytest.raises(OmniClientError, match="generic PEFT LoRA") as exc:
        pipeline._load_pdd_lora_adapter(
            lora_request=SimpleNamespace(lora_int_id=_PDD_ID),
            lora_path="unused-synthetic-artifact",
            dtype=torch.float32,
            unsupported_offload_mode=None,
        )

    assert client_error_metadata(exc.value) == (400, "BadRequestError")
    assert pipeline._pdd_adapters == metadata[0]
    assert pipeline._pdd_adapter_ids == metadata[1]
    assert pipeline._pdd_active_adapters == metadata[2]
    _assert_unchanged(pipeline.transformer, before)


def _stage_pdd(pipeline, adapter_id):
    """Tiny CPU artifacts at the boundary after parsing and registration."""
    pipeline._pdd_adapters[adapter_id] = {
        "cfg": PDDConfig(),
        "head_weights": {name: torch.ones(32, 1, 2) for name in ("video_out", "audio_out")},
        "head_biases": {name: torch.zeros(32, 1) for name in ("video_out", "audio_out")},
    }
    pipeline._pdd_adapter_ids.add(adapter_id)


def _cache_manager(pipeline, adapter_ids):
    manager = object.__new__(DiffusionLoRAManager)
    manager.pipeline, manager._closed = pipeline, False
    manager.max_cached_adapters = len(adapter_ids)
    manager._registered_adapters = {aid: LoRAModel(aid, 1, {}) for aid in adapter_ids}
    manager._active_adapter_id = manager._suspended_adapter_id = None
    manager._fixed_adapter_request = None
    manager._merged_adapters, manager._lora_modules = {}, {}
    manager._adapter_scales = dict.fromkeys(adapter_ids, 1.0)
    manager._adapter_access_order = OrderedDict((aid, 0) for aid in adapter_ids)
    manager._pinned_adapters = set()
    return manager


@pytest.mark.parametrize("existing_adapter", [False, True])
@pytest.mark.parametrize("cleanup", ["remove", "evict", "close"])
def test_failed_pdd_bank_install_restores_heads_and_can_retry(existing_adapter, cleanup, monkeypatch):
    pipeline = _Pipeline()
    if existing_adapter:
        _stage_pdd(pipeline, _PDD_ID)
        pipeline._ensure_pdd_heads(_PDD_ID)
    before = _snapshot(pipeline.transformer)
    other_dit = _snapshot(pipeline.transformers_ref)
    adapters_before = dict(pipeline._pdd_active_adapters)
    attempt_id = _PDD_ID + 1
    _stage_pdd(pipeline, attempt_id)
    metadata = pipeline._pdd_adapters[attempt_id]
    manager = _cache_manager(pipeline, [attempt_id, *adapters_before])
    failed_banks = []
    load_bank = PDDParallelHead.load_bank

    def fail_audio(head, adapter_id, weight, bias, plans):
        if head is pipeline.transformer.final_layer.audio_out:
            raise torch.OutOfMemoryError("injected audio bank allocation failure")
        load_bank(head, adapter_id, weight, bias, plans)
        failed_banks.append(weakref.ref(head.banks[str(adapter_id)]))

    with monkeypatch.context() as patch:
        patch.setattr(PDDParallelHead, "load_bank", fail_audio)
        with pytest.raises(torch.OutOfMemoryError, match="audio bank"):
            pipeline._ensure_pdd_heads(attempt_id)

    gc.collect()
    assert failed_banks and all(ref() is None for ref in failed_banks)
    assert pipeline._pdd_active_adapters == adapters_before
    assert pipeline._pdd_adapters[attempt_id] is metadata
    _assert_unchanged(pipeline.transformer, before)
    _assert_unchanged(pipeline.transformers_ref, other_dit)
    for name in ("video_out", "audio_out"):
        head = getattr(pipeline.transformer.final_layer, name)
        if isinstance(head, PDDParallelHead):
            assert str(attempt_id) not in head.banks

    adapter = pipeline._ensure_pdd_heads(attempt_id)
    assert pipeline._ensure_pdd_heads(attempt_id) is adapter
    live_banks = []
    for name, (_, base, _) in zip(("video_out", "audio_out"), before):
        head = getattr(pipeline.transformer.final_layer, name)
        torch.testing.assert_close(head(torch.ones(1, 2))[0], base)
        torch.testing.assert_close(head(torch.ones(1, 2), 0, attempt_id)[0], torch.full((1, 1), 2.0))
        live_banks.append(weakref.ref(head.banks[str(attempt_id)]))

    if cleanup == "close":
        manager.close()
    elif cleanup == "evict":
        manager._evict_for_new_adapter()
    else:
        assert manager.remove_adapter(attempt_id)
    gc.collect()
    assert all(ref() is None for ref in live_banks)
    assert attempt_id not in pipeline._pdd_adapter_ids
    assert attempt_id not in pipeline._pdd_adapters
    assert attempt_id not in pipeline._pdd_active_adapters
    if existing_adapter and cleanup != "close":
        _assert_unchanged(pipeline.transformer, before)
    for name, (_, base, _) in zip(("video_out", "audio_out"), before):
        torch.testing.assert_close(getattr(pipeline.transformer.final_layer, name)(torch.ones(1, 2))[0], base)


def test_failed_pdd_head_construction_restores_original_projections(monkeypatch):
    pipeline = _Pipeline()
    _stage_pdd(pipeline, _PDD_ID)
    before = _snapshot(pipeline.transformer)
    init_head = PDDParallelHead.__init__
    constructed = []

    def fail_audio(head, source, num_steps):
        if source is before[1][0]:
            raise torch.OutOfMemoryError("injected audio head allocation failure")
        init_head(head, source, num_steps)
        constructed.append(weakref.ref(head))

    with monkeypatch.context() as patch:
        patch.setattr(PDDParallelHead, "__init__", fail_audio)
        with pytest.raises(torch.OutOfMemoryError, match="audio head"):
            pipeline._ensure_pdd_heads(_PDD_ID)

    gc.collect()
    assert constructed and all(ref() is None for ref in constructed)
    _assert_unchanged(pipeline.transformer, before)
    assert not pipeline._pdd_active_adapters
    pipeline._ensure_pdd_heads(_PDD_ID)


def test_failed_pdd_reinstall_preserves_preexisting_same_id_banks(monkeypatch):
    pipeline = _Pipeline()
    _stage_pdd(pipeline, _PDD_ID)
    pipeline._ensure_pdd_heads(_PDD_ID)
    # Exercise recovery when the model already owns this ID's banks.
    pipeline._pdd_active_adapters.pop(_PDD_ID)
    before = _snapshot(pipeline.transformer)
    layer = pipeline.transformer.final_layer
    original_banks = [getattr(layer, name).banks[str(_PDD_ID)] for name in ("video_out", "audio_out")]
    load_bank = PDDParallelHead.load_bank

    def fail_audio(head, adapter_id, weight, bias, plans):
        if head is layer.audio_out:
            raise torch.OutOfMemoryError("injected audio bank allocation failure")
        load_bank(head, adapter_id, weight, bias, plans)

    monkeypatch.setattr(PDDParallelHead, "load_bank", fail_audio)
    with pytest.raises(torch.OutOfMemoryError, match="audio bank"):
        pipeline._ensure_pdd_heads(_PDD_ID)

    _assert_unchanged(pipeline.transformer, before)
    for name, bank in zip(("video_out", "audio_out"), original_banks):
        assert getattr(layer, name).banks[str(_PDD_ID)] is bank
