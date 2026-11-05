# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for mask-keyed MoE CUDA-graph descriptor expansion (#9865)."""

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
    from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

    base = BatchDescriptor(num_tokens=8, num_reqs=1, uniform=True)
    captured = {replace(base, moe_routing_key=int(key)) for key in MoERoutingKey}
    runner = object.__new__(OmniGPUModelRunner)
    runner.compilation_config = SimpleNamespace(cudagraph_specialize_moe_routing=True)
    runner.model = SimpleNamespace(gen_vocab_start_index=100)
    # Real decode token is gen; the padded tail and the GPU buffer are not.
    runner.input_ids = SimpleNamespace(
        cpu=torch.tensor([150, 0, 0, 0, 0, 0, 0, 0]),
        gpu=torch.tensor([1, 1, 1, 1, 1, 1, 1, 1]),
    )
    runner.input_batch = SimpleNamespace(prev_sampled_token_ids=None)
    runner.cudagraph_dispatcher = SimpleNamespace(cudagraph_keys={CUDAGraphMode.FULL: captured})

    selected = OmniGPUModelRunner._select_moe_replay_batch_descriptor(runner, CUDAGraphMode.FULL, base, 1)
    assert selected.moe_routing_key == int(MoERoutingKey.ALL_GEN)
    assert runner._last_moe_routing_replay_key == int(MoERoutingKey.ALL_GEN)

    runner.input_batch = SimpleNamespace(prev_sampled_token_ids=torch.tensor([180]))
    runner.input_ids.cpu = torch.tensor([3, 0, 0, 0, 0, 0, 0, 0])
    runner.input_ids.gpu = torch.zeros(8, dtype=torch.int64)
    runner.input_ids.gpu[0] = 180
    selected = OmniGPUModelRunner._select_moe_replay_batch_descriptor(runner, CUDAGraphMode.FULL, base, 1)
    assert selected.moe_routing_key == int(MoERoutingKey.ALL_GEN)

    runner.compilation_config.cudagraph_specialize_moe_routing = False
    unchanged = OmniGPUModelRunner._select_moe_replay_batch_descriptor(runner, CUDAGraphMode.FULL, base, 1)
    assert unchanged == base


def test_ar_execute_model_retargets_replay_descriptor() -> None:
    """The live AR forward must pass the unpadded count into replay selection."""
    root = Path(__file__).resolve().parents[4]
    source = (root / "vllm_omni/worker/gpu_ar_model_runner.py").read_text(encoding="utf-8")
    needle = (
        "batch_desc = self._select_moe_replay_batch_descriptor(\n"
        "                cudagraph_mode,\n"
        "                batch_desc,\n"
        "                num_tokens_unpadded,\n"
        "            )"
    )
    assert needle in source
