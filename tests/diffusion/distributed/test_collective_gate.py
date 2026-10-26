# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unit tests for the DLO/QKV-AllToAll ordering gate.

Covers the barrier rendezvous and the hook's deferred-AllGather state machine
that keeps the background weight gather from taking the links ahead of the
block's foreground AllToAll.
"""

from unittest.mock import Mock

import pytest
import torch
from torch import nn

import vllm_omni.diffusion.offloader.distributed_layerwise_backend as dist_backend_module
from tests.diffusion.offloader.helpers import DummyStream, patch_offload_runtime
from vllm_omni.diffusion.distributed.collective_gate import QKVAllToAllBarrier
from vllm_omni.diffusion.offloader.distributed_layerwise_backend import (
    DistributedLayerwiseOffloadHook,
    attach_collective_gate,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


@pytest.fixture
def patched_offload_runtime(monkeypatch):
    patch_offload_runtime(monkeypatch, dist_backend_module.current_omni_platform, synchronize=True)


class TestQKVAllToAllBarrier:
    def test_signal_without_arm_is_a_noop(self):
        on_signal = Mock()
        barrier = QKVAllToAllBarrier(on_signal)

        barrier.signal()

        on_signal.assert_not_called()
        assert not barrier.armed

    def test_arm_then_signal_fires_once(self):
        on_signal = Mock()
        barrier = QKVAllToAllBarrier(on_signal)

        barrier.arm()
        assert barrier.armed
        barrier.signal()

        on_signal.assert_called_once_with()
        assert not barrier.armed

        # A second signal in the same forward must not re-issue the gather.
        barrier.signal()
        on_signal.assert_called_once_with()

    def test_disarm_suppresses_a_pending_signal(self):
        on_signal = Mock()
        barrier = QKVAllToAllBarrier(on_signal)

        barrier.arm()
        barrier.disarm()
        barrier.signal()

        on_signal.assert_not_called()


def _gated_hook() -> DistributedLayerwiseOffloadHook:
    """A hook wired for the gated path, with hand-set shard buffers."""
    hook = DistributedLayerwiseOffloadHook(
        next_block=nn.Linear(2, 2, bias=False),
        device=torch.device("cpu"),
        dp_group=object(),
        dp_size=2,
        rank=0,
        copy_stream=DummyStream(),
        comm_stream=DummyStream(),
        pin_memory=False,
        order_allgather_after_attn_a2a=True,
    )
    device_buf = {torch.float32: torch.zeros(4)}
    shard_buf = {torch.float32: torch.zeros(2)}
    hook.gpu_buffers = [dict(device_buf), dict(device_buf)]
    hook.gpu_shard_buffers = [dict(shard_buf), dict(shard_buf)]
    hook.cpu_shards = {torch.float32: torch.zeros(2)}
    hook._ag_output_sizes = {torch.float32: 4}
    hook.metadata = {}
    hook._cached_repoint = ()
    hook.rank_local_mmap = False
    return hook


class TestDeferredAllGather:
    def test_prefetch_defers_the_gather_until_the_barrier_signals(self, patched_offload_runtime, monkeypatch):
        hook = _gated_hook()
        gather = Mock()
        monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", gather)

        hook.prefetch_layer(0, non_blocking=True)

        # H2D ran; the collective is held back for the block's AllToAll.
        gather.assert_not_called()
        assert hook._a2a_barrier.armed
        assert hook._deferred_allgather is not None
        # The ready event must still be published for the consumer.
        assert hook.ready_events[0] is hook._prefetch_done

        hook._a2a_barrier.signal()

        gather.assert_called_once()
        assert hook._deferred_allgather is None
        assert not hook._a2a_barrier.armed

    def test_post_forward_flush_issues_an_unsignalled_gather(self, patched_offload_runtime, monkeypatch):
        """A block whose attention never ran must still get its weights."""
        hook = _gated_hook()
        gather = Mock()
        monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", gather)

        hook.prefetch_layer(0, non_blocking=True)
        gather.assert_not_called()

        # pre_forward ran, attention did not: the pairing flush must fire.
        hook.post_forward(hook.next_block, None)

        gather.assert_called_once()
        assert hook._deferred_allgather is None
        assert not hook._a2a_barrier.armed

    def test_sync_prefetch_issues_immediately_without_arming(self, patched_offload_runtime, monkeypatch):
        """The cache-dit sync fallback has no attention signal to wait for."""
        hook = _gated_hook()
        gather = Mock()
        monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", gather)

        hook.prefetch_layer(0, non_blocking=False)

        gather.assert_called_once()
        assert not hook._a2a_barrier.armed
        assert hook._deferred_allgather is None

    def test_disabled_by_default(self, patched_offload_runtime, monkeypatch):
        hook = DistributedLayerwiseOffloadHook(
            next_block=nn.Linear(2, 2, bias=False),
            device=torch.device("cpu"),
            dp_group=object(),
            dp_size=2,
            rank=0,
            copy_stream=DummyStream(),
            comm_stream=DummyStream(),
            pin_memory=False,
        )
        hook.gpu_buffers = [{torch.float32: torch.zeros(4)}] * 2
        hook.gpu_shard_buffers = [{torch.float32: torch.zeros(2)}] * 2
        hook.cpu_shards = {torch.float32: torch.zeros(2)}
        hook._ag_output_sizes = {torch.float32: 4}
        hook.metadata = {}
        hook._cached_repoint = ()
        gather = Mock()
        monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", gather)

        hook.prefetch_layer(0, non_blocking=True)

        assert hook._a2a_barrier is None
        gather.assert_called_once()


class TestAttachCollectiveGate:
    def test_block_without_attention_reports_zero(self):
        hook = _gated_hook()

        assert attach_collective_gate(nn.Linear(2, 2), hook) == 0
