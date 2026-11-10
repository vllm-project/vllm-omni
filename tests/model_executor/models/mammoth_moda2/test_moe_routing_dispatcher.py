# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for mask-keyed MoE CUDA-graph descriptor expansion (#9865)."""

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_omni.cudagraph.moe_routing_batch_descriptor import (
    BatchDescriptor,
    patch_batch_descriptor_moe_routing_key,
)
from vllm_omni.cudagraph.moe_routing_dispatcher import (
    apply_moe_routing_key,
    expand_batch_descriptors_for_moe_routing,
    with_moe_routing_key,
)
from vllm_omni.model_executor.models.mammoth_moda2.moe_cudagraph import (
    MoERoutingKey,
    moe_routing_key_from_input_ids,
    routing_ids_for_replay,
)

patch_batch_descriptor_moe_routing_key()

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_expand_batch_descriptors_disabled_keeps_blend_only() -> None:
    base = BatchDescriptor(num_tokens=8, num_reqs=8, uniform=True)
    out = expand_batch_descriptors_for_moe_routing([base], enabled=False)
    assert len(out) == 1
    assert out[0].moe_routing_key == int(MoERoutingKey.BLEND)


def test_expand_batch_descriptors_enabled_triples_keys() -> None:
    base = BatchDescriptor(num_tokens=8, num_reqs=8, uniform=True, num_active_loras=0)
    out = expand_batch_descriptors_for_moe_routing([base], enabled=True)
    keys = sorted(d.moe_routing_key for d in out)
    assert keys == [
        int(MoERoutingKey.BLEND),
        int(MoERoutingKey.ALL_UND),
        int(MoERoutingKey.ALL_GEN),
    ]
    assert all(d.num_tokens == 8 for d in out)


def test_with_moe_routing_key_replace() -> None:
    desc = BatchDescriptor(num_tokens=4)
    keyed = with_moe_routing_key(desc, MoERoutingKey.ALL_UND)
    assert keyed.moe_routing_key == int(MoERoutingKey.ALL_UND)
    assert keyed.num_tokens == 4
    # Distinct graph keys.
    assert keyed != replace(desc, moe_routing_key=int(MoERoutingKey.BLEND))


def _captured(*descriptors: BatchDescriptor) -> set[BatchDescriptor]:
    return set(descriptors)


def test_apply_moe_routing_key_selects_captured_specialty() -> None:
    """Replay must leave the default BLEND descriptor when the mask is uniform."""
    base = BatchDescriptor(num_tokens=1, num_reqs=1, uniform=True)
    captured = _captured(
        *(replace(base, moe_routing_key=int(key)) for key in MoERoutingKey),
    )
    # Upstream dispatch omits the key, so the incoming descriptor is BLEND.
    assert base.moe_routing_key == int(MoERoutingKey.BLEND)

    und = apply_moe_routing_key(CUDAGraphMode.FULL, base, MoERoutingKey.ALL_UND, captured, enabled=True)
    gen = apply_moe_routing_key(CUDAGraphMode.FULL, base, MoERoutingKey.ALL_GEN, captured, enabled=True)
    mixed = apply_moe_routing_key(CUDAGraphMode.FULL, base, MoERoutingKey.BLEND, captured, enabled=True)
    assert und.moe_routing_key == int(MoERoutingKey.ALL_UND)
    assert gen.moe_routing_key == int(MoERoutingKey.ALL_GEN)
    assert mixed.moe_routing_key == int(MoERoutingKey.BLEND)
    assert und.num_tokens == base.num_tokens
    assert und.uniform is True


def test_apply_moe_routing_key_falls_back_to_blend_then_original() -> None:
    base = BatchDescriptor(num_tokens=4, num_reqs=4, uniform=True, moe_routing_key=int(MoERoutingKey.BLEND))
    blend_only = {replace(base, moe_routing_key=int(MoERoutingKey.BLEND))}
    selected = apply_moe_routing_key(CUDAGraphMode.FULL, base, MoERoutingKey.ALL_GEN, blend_only, enabled=True)
    assert selected.moe_routing_key == int(MoERoutingKey.BLEND)

    untouched = apply_moe_routing_key(CUDAGraphMode.FULL, base, MoERoutingKey.ALL_UND, set(), enabled=True)
    assert untouched == base

    eager = apply_moe_routing_key(CUDAGraphMode.NONE, base, MoERoutingKey.ALL_GEN, blend_only, enabled=True)
    assert eager.moe_routing_key == int(MoERoutingKey.BLEND)

    disabled = apply_moe_routing_key(
        CUDAGraphMode.FULL,
        replace(base, moe_routing_key=int(MoERoutingKey.ALL_GEN)),
        MoERoutingKey.ALL_GEN,
        blend_only,
        enabled=False,
    )
    assert disabled.moe_routing_key == int(MoERoutingKey.BLEND)


def test_routing_ids_ignore_padding_and_stale_cpu_under_async() -> None:
    gen_start = 100
    # Tail zeros are graph padding, not scheduled understanding tokens.
    padded = torch.tensor([150, 0, 0, 0])
    scheduled = routing_ids_for_replay(padded, padded, num_tokens=1, prev_sampled_token_ids=None)
    assert moe_routing_key_from_input_ids(scheduled, gen_start) is MoERoutingKey.ALL_GEN
    assert moe_routing_key_from_input_ids(padded, gen_start) is MoERoutingKey.BLEND

    cpu = torch.tensor([3])
    gpu = torch.tensor([180])
    async_ids = routing_ids_for_replay(cpu, gpu, num_tokens=1, prev_sampled_token_ids=gpu)
    assert moe_routing_key_from_input_ids(async_ids, gen_start) is MoERoutingKey.ALL_GEN
    sync_ids = routing_ids_for_replay(cpu, gpu, num_tokens=1, prev_sampled_token_ids=None)
    assert moe_routing_key_from_input_ids(sync_ids, gen_start) is MoERoutingKey.ALL_UND


def test_select_moe_replay_batch_descriptor_uses_unpadded_ids() -> None:
    from vllm_omni.model_executor.models.mammoth_moda2.gpu_ar_model_runner import MammothGPUARModelRunner
    from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

    base = BatchDescriptor(num_tokens=8, num_reqs=1, uniform=True)
    captured = {replace(base, moe_routing_key=int(key)) for key in MoERoutingKey}
    runner = object.__new__(MammothGPUARModelRunner)
    runner.compilation_config = SimpleNamespace(cudagraph_specialize_moe_routing=True)
    runner.model = SimpleNamespace(gen_vocab_start_index=100)
    # Real decode token is gen; the padded tail and the GPU buffer are not.
    runner.input_ids = SimpleNamespace(
        cpu=torch.tensor([150, 0, 0, 0, 0, 0, 0, 0]),
        gpu=torch.tensor([1, 1, 1, 1, 1, 1, 1, 1]),
    )
    runner.input_batch = SimpleNamespace(prev_sampled_token_ids=None)
    runner.cudagraph_dispatcher = SimpleNamespace(cudagraph_keys={CUDAGraphMode.FULL: captured})

    selected = MammothGPUARModelRunner._select_moe_replay_batch_descriptor(runner, CUDAGraphMode.FULL, base, 1)
    assert selected.moe_routing_key == int(MoERoutingKey.ALL_GEN)
    assert runner._last_moe_routing_replay_key == int(MoERoutingKey.ALL_GEN)

    runner.input_batch = SimpleNamespace(prev_sampled_token_ids=torch.tensor([180]))
    runner.input_ids.cpu = torch.tensor([3, 0, 0, 0, 0, 0, 0, 0])
    runner.input_ids.gpu = torch.zeros(8, dtype=torch.int64)
    runner.input_ids.gpu[0] = 180
    selected = MammothGPUARModelRunner._select_moe_replay_batch_descriptor(runner, CUDAGraphMode.FULL, base, 1)
    assert selected.moe_routing_key == int(MoERoutingKey.ALL_GEN)

    runner.compilation_config.cudagraph_specialize_moe_routing = False
    unchanged = MammothGPUARModelRunner._select_moe_replay_batch_descriptor(runner, CUDAGraphMode.FULL, base, 1)
    assert unchanged == base

    # Shared base hook stays a no-op when specialize is off (fail-fast when on
    # is covered in tests/worker/test_gpu_ar_worker_mammoth_runner.py).
    shared = object.__new__(OmniGPUModelRunner)
    shared.compilation_config = SimpleNamespace(cudagraph_specialize_moe_routing=False)
    assert OmniGPUModelRunner._select_moe_replay_batch_descriptor(shared, CUDAGraphMode.FULL, base, 1) is base


def test_ar_execute_model_retargets_replay_descriptor() -> None:
    """The retargeted descriptor must reach ``set_forward_context``.

    Parses the AST instead of matching source text so a reformat or a local
    rename does not break the test, while a real break in the dataflow --
    dropping the retarget, or passing a descriptor other than the retargeted
    ``batch_desc`` to ``set_forward_context`` -- still fails.
    """
    root = Path(__file__).resolve().parents[4]
    tree = ast.parse((root / "vllm_omni/worker/gpu_ar_model_runner.py").read_text(encoding="utf-8"))

    execute_models = [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "execute_model"
    ]
    assert execute_models, "gpu_ar_model_runner.execute_model not found"

    def called_name(call: ast.Call) -> str | None:
        # ``set_forward_context(...)`` and ``x.set_forward_context(...)`` alike.
        if isinstance(call.func, ast.Name):
            return call.func.id
        if isinstance(call.func, ast.Attribute):
            return call.func.attr
        return None

    def sets_batch_desc_from(fn: ast.FunctionDef, name: str) -> bool:
        for node in ast.walk(fn):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            if called_name(node.value) != name:
                continue
            if any(isinstance(t, ast.Name) and t.id == "batch_desc" for t in node.targets):
                return True
        return False

    def forward_context_calls(fn: ast.FunctionDef) -> list[ast.Call]:
        return [
            call for call in ast.walk(fn) if isinstance(call, ast.Call) and called_name(call) == "set_forward_context"
        ]

    def passes_batch_desc(call: ast.Call) -> bool:
        for kw in call.keywords:
            if kw.arg != "batch_descriptor":
                continue
            value = kw.value
            if isinstance(value, ast.Name) and value.id == "batch_desc":
                return True
            if isinstance(value, ast.Attribute) and value.attr == "batch_desc":
                return True
        return False

    # execute_model is long; the retarget and the forward-context call sit in
    # different branches, so check both anywhere in the function.
    retargeting = [fn for fn in execute_models if sets_batch_desc_from(fn, "_select_moe_replay_batch_descriptor")]
    assert retargeting, "execute_model does not retarget batch_desc for MoE replay"
    fwd_calls = forward_context_calls(retargeting[0])
    assert fwd_calls, "execute_model does not call set_forward_context"
    assert any(passes_batch_desc(call) for call in fwd_calls), (
        "execute_model does not pass the retargeted batch_desc to set_forward_context"
    )
