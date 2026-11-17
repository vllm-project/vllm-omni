# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections.abc import Callable
from contextlib import nullcontext

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention.backends import ring_flash_attn
from vllm_omni.diffusion.attention.backends.ring import ring_utils
from vllm_omni.diffusion.attention.backends.ring.fused_merge import try_fused_ring_merge
from vllm_omni.diffusion.attention.backends.ring.ring_selector import AttnType

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None, reason="requires NVIDIA CUDA"
)


class _FakeComm:
    """Simulate fresh receives for native Ring merge dispatch checks."""

    def __init__(self, rank: int, shards: tuple[list[torch.Tensor], list[torch.Tensor]]):
        self.rank = rank
        self.world_size = len(shards[0])
        self.shards = shards
        self.step = 0
        self.channel = 0
        self.merge_flags: list[bool] = []
        self.fused_results: list[bool] = []

    def send_recv(self, tensor: torch.Tensor) -> torch.Tensor:
        shards = self.shards[self.channel]
        torch.testing.assert_close(tensor, shards[(self.rank - self.step) % self.world_size], rtol=0, atol=0)
        self.channel += 1
        return shards[(self.rank - self.step - 1) % self.world_size].clone()

    def commit(self) -> None:
        assert self.channel == 2
        self.step += 1

    def wait(self) -> None:
        self.channel = 0


def _install_fake_ring(
    monkeypatch: pytest.MonkeyPatch, device: str, world: int, rank: int
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], list[_FakeComm], list[int]]:
    shape = (1, 2, 2, 8)
    keys = [torch.full(shape, index, dtype=torch.bfloat16, device=device) for index in range(world)]
    values = [torch.full(shape, 2 * index + 1, dtype=torch.bfloat16, device=device) for index in range(world)]
    instances: list[_FakeComm] = []
    attended: list[int] = []

    def make_comm(_group: object) -> _FakeComm:
        comm = _FakeComm(rank, (keys, values))
        instances.append(comm)
        return comm

    def attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        softmax_scale: float,
        causal: bool,
        **_unused: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attended.append(int(k[0, 0, 0, 0].item()))
        scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * softmax_scale
        if causal:
            mask = torch.ones(q.shape[1], k.shape[1], device=q.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(mask, -torch.inf)
        out = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v.float())
        return out, scores.logsumexp(-1)

    def merge(out, lse, block_out, block_lse, *, lse_layout, use_fused_merge=False):
        instances[-1].merge_flags.append(use_fused_merge)
        return ring_utils.update_out_and_lse(
            out, lse, block_out, block_lse, lse_layout=lse_layout, use_fused_merge=use_fused_merge
        )

    def try_merge(*args):
        result = try_fused_ring_merge(*args)
        instances[-1].fused_results.append(result is not None)
        return result

    monkeypatch.setattr(ring_flash_attn, "RingComm", make_comm)
    monkeypatch.setattr(ring_flash_attn, "update_out_and_lse", merge)
    monkeypatch.setattr(ring_utils, "try_fused_ring_merge", try_merge)
    monkeypatch.setattr(ring_flash_attn, "select_flash_attn_impl", lambda *_args, **_kwargs: attention)
    q = torch.zeros(shape, dtype=torch.bfloat16, device=device)
    return (q, keys[rank], values[rank]), instances, attended


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@requires_cuda
@pytest.mark.parametrize(
    "world,rank,causal,backend,hip,opt_in",
    [
        (1, 0, False, AttnType.FA3, False, True),
        (2, 1, False, AttnType.FA3, False, True),
        (8, 7, False, AttnType.FA3, False, True),
        (8, 0, True, AttnType.FA3, False, True),
        (8, 3, True, AttnType.FA3, False, True),
        (2, 1, False, AttnType.FA, False, True),
        (8, 7, False, AttnType.FA4, False, False),
        (8, 7, False, AttnType.TORCH, False, False),
        (8, 7, False, AttnType.FA3, True, True),
        (8, 7, False, AttnType.AITER, True, False),
    ],
)
@torch.inference_mode()
def test_native_ring_merge_dispatch(monkeypatch, world, rank, causal, backend, hip, opt_in):
    # Mocked communication checks dispatch; it does not qualify NCCL ordering.
    # Mocking HIP checks dispatch only, not execution on ROCm hardware.
    if hip:
        monkeypatch.setattr(torch.version, "hip", "test-rocm")
    inputs, instances, attended = _install_fake_ring(monkeypatch, "cuda", world, rank)
    snapshots = [tensor.clone() for tensor in inputs]
    expected_visits = [(rank - step) % world for step in range(rank + 1 if causal else world)]
    options = dict(softmax_scale=8**-0.5, causal=causal, attn_type=backend)

    with torch.enable_grad():
        reference = ring_flash_attn.ring_flash_attn_forward(None, *inputs, **options)
    actual = ring_flash_attn.ring_flash_attn_forward(None, *inputs, **options)
    for tensor, expected in zip(actual, reference):
        torch.testing.assert_close(tensor, expected, rtol=2e-6, atol=2e-6)
    for tensor, snapshot in zip(inputs, snapshots):
        torch.testing.assert_close(tensor, snapshot, rtol=0, atol=0)
    assert attended == expected_visits * 2
    assert instances[0].merge_flags == instances[1].merge_flags == [opt_in] * len(expected_visits)
    merge_count = len(expected_visits) - 1 if opt_in else 0
    assert instances[0].fused_results == [False] * merge_count
    assert instances[1].fused_results == [not hip] * merge_count


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@requires_cuda
@pytest.mark.parametrize("fallback", ["grad", "compile", "device"])
@torch.inference_mode()
def test_native_ring_merge_fallbacks(monkeypatch, fallback):
    inputs, instances, _ = _install_fake_ring(monkeypatch, "cuda", world=5, rank=4)
    device_index = inputs[0].device.index
    assert device_index is not None
    overrides: dict[str, tuple[object, str, Callable[[], object]]] = {
        "compile": (torch.compiler, "is_compiling", lambda: True),
        "device": (torch.accelerator, "current_device_index", lambda: device_index + 1),
    }
    if fallback in overrides:
        target, name, value = overrides[fallback]
        monkeypatch.setattr(target, name, value)
    with torch.enable_grad() if fallback == "grad" else nullcontext():
        output, lse = ring_flash_attn.ring_flash_attn_forward(
            None, *inputs, softmax_scale=8**-0.5, causal=False, attn_type=AttnType.FA3
        )
    assert torch.isfinite(output).all() and torch.isfinite(lse).all()
    assert instances[0].merge_flags == [True] * 5
    assert instances[0].fused_results == [False] * 4


@pytest.mark.cpu
@torch.inference_mode()
def test_native_ring_cpu_skips_fused_merge(monkeypatch):
    inputs, instances, _ = _install_fake_ring(monkeypatch, "cpu", world=5, rank=4)

    def unexpected_cuda_query() -> bool:
        raise AssertionError("CPU Ring must not query CUDA capture state")

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", unexpected_cuda_query)
    output, lse = ring_flash_attn.ring_flash_attn_forward(
        None, *inputs, softmax_scale=8**-0.5, causal=False, attn_type=AttnType.FA3
    )
    assert torch.isfinite(output).all() and torch.isfinite(lse).all()
    assert instances[0].merge_flags == [True] * 5
    assert instances[0].fused_results == [False] * 4
