# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for plan-driven Hugging Face text encoder tensor parallelism."""

from __future__ import annotations

import copy
import threading
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from vllm_omni.diffusion.distributed import text_encoder_tp
from vllm_omni.diffusion.distributed.text_encoder_tp import (
    RowShardedLinear,
    maybe_shard_text_encoder,
    shard_text_encoder,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.parallel, pytest.mark.core_model, pytest.mark.cpu]


class _ThreadGroup:
    """In-process stand-in for a TP GroupCoordinator; one thread per rank."""

    def __init__(self, world_size: int, rank: int, shared: dict[str, Any]) -> None:
        self.world_size = world_size
        self.rank_in_group = rank
        self.ranks = list(range(world_size))
        self._shared = shared

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        barrier, slots = self._shared["barrier"], self._shared["slots"]
        slots[self.rank_in_group] = input_
        barrier.wait()
        output = torch.stack(slots).sum(dim=0)
        barrier.wait()
        return output


def _tiny_qwen25_vl() -> Qwen2_5_VLForConditionalGeneration:
    config = Qwen2_5_VLConfig(
        text_config=dict(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]},
        ),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64),
    )
    torch.manual_seed(0)
    model = Qwen2_5_VLForConditionalGeneration(config).eval()
    del model.model.visual
    return model


def _last_hidden(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), output_hidden_states=True)
    return out.hidden_states[-1]


@pytest.mark.parametrize("tp_size", [2])
def test_sharded_encoder_matches_unsharded(tp_size):
    base = _tiny_qwen25_vl()
    input_ids = torch.randint(0, 128, (2, 9))
    expected = _last_hidden(base, input_ids)

    barrier = threading.Barrier(tp_size)
    shared: dict[str, Any] = {"barrier": barrier, "slots": [None] * tp_size}
    results: list[torch.Tensor | None] = [None] * tp_size
    errors: list[BaseException] = []

    def run(rank: int) -> None:
        try:
            model = shard_text_encoder(copy.deepcopy(base), _ThreadGroup(tp_size, rank, shared))
            results[rank] = _last_hidden(model, input_ids)
        except BaseException as exc:  # surface thread failures in the test
            errors.append(exc)
            barrier.abort()

    threads = [threading.Thread(target=run, args=(rank,)) for rank in range(tp_size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert not errors, errors

    for rank in range(tp_size):
        torch.testing.assert_close(results[rank], expected, rtol=1e-4, atol=1e-4)


def test_shard_layout():
    model = shard_text_encoder(_tiny_qwen25_vl(), _ThreadGroup(2, 1, {}))
    layer = model.model.language_model.layers[0]
    assert layer.self_attn.q_proj.weight.shape == (32, 64)
    assert layer.self_attn.q_proj.bias.shape == (32,)
    assert layer.self_attn.k_proj.weight.shape == (16, 64)
    assert isinstance(layer.self_attn.o_proj, RowShardedLinear)
    assert layer.self_attn.o_proj.weight.shape == (64, 32)
    assert layer.mlp.gate_proj.weight.shape == (64, 64)
    assert isinstance(layer.mlp.down_proj, RowShardedLinear)
    assert layer.mlp.down_proj.weight.shape == (64, 64)
    # Embeddings and norms stay replicated.
    assert model.model.language_model.embed_tokens.weight.shape == (128, 64)


def test_tp_size_must_divide_kv_heads():
    with pytest.raises(ValueError, match="num_key_value_heads"):
        shard_text_encoder(_tiny_qwen25_vl(), _ThreadGroup(4, 0, {}))


def test_model_without_tp_plan_is_rejected():
    model = nn.Sequential(nn.Linear(4, 4))
    model.config = SimpleNamespace()
    with pytest.raises(ValueError, match="base_model_tp_plan"):
        shard_text_encoder(model, _ThreadGroup(2, 0, {}))


def test_maybe_shard_is_noop_for_tp1(monkeypatch):
    def fail(_tp_size):
        raise AssertionError("group must not be created for text_encoder_tp_size=1")

    monkeypatch.setattr(text_encoder_tp, "get_text_encoder_tp_group", fail)
    model = nn.Linear(4, 4)
    od_config = SimpleNamespace(parallel_config=SimpleNamespace(text_encoder_tp_size=1))
    assert maybe_shard_text_encoder(model, od_config) is model


def _fake_parallel_state(monkeypatch, world_size: int, dp_ranks: list[int] | None, rank: int = 0) -> dict[str, Any]:
    """Patch parallel_state with a fake world (and optional DP group) for group construction."""
    from vllm_omni.diffusion.distributed import parallel_state

    world = SimpleNamespace(
        world_size=world_size, ranks=list(range(world_size)), rank_in_group=rank, local_rank=0, device_group=None
    )
    dp = None if dp_ranks is None else SimpleNamespace(world_size=len(dp_ranks), ranks=dp_ranks)
    captured: dict[str, Any] = {}

    def fake_init(group_ranks, local_rank, backend, parallel_mode):
        captured.update(group_ranks=group_ranks, parallel_mode=parallel_mode)
        return SimpleNamespace(world_size=len(group_ranks[0]))

    monkeypatch.setattr(parallel_state, "_TEXT_ENCODER_TP", None)
    monkeypatch.setattr(parallel_state, "_DP", dp)
    monkeypatch.setattr(parallel_state, "get_world_group", lambda: world)
    monkeypatch.setattr(parallel_state, "init_model_parallel_group", fake_init)
    monkeypatch.setattr(parallel_state.torch.distributed, "get_backend", lambda _group: "gloo")
    return captured


@pytest.mark.parametrize(
    ("world_size", "dp_ranks", "tp_size", "expected"),
    [
        (4, None, 4, [[0, 1, 2, 3]]),
        (4, None, 2, [[0, 1], [2, 3]]),
        (2, None, 1, [[0], [1]]),
        # DP=2 x 2 ranks per replica: groups stay inside each replica.
        (4, [0, 2], 2, [[0, 1], [2, 3]]),
    ],
)
def test_text_encoder_tp_groups_tile_replicas(monkeypatch, world_size, dp_ranks, tp_size, expected):
    from vllm_omni.diffusion.distributed import parallel_state

    captured = _fake_parallel_state(monkeypatch, world_size, dp_ranks)
    group = parallel_state.get_text_encoder_tp_group(tp_size)
    assert captured == {"group_ranks": expected, "parallel_mode": "tensor"}
    # Cached on reuse; a different size is rejected.
    assert parallel_state.get_text_encoder_tp_group(tp_size) is group
    with pytest.raises(ValueError, match="already initialized"):
        parallel_state.get_text_encoder_tp_group(tp_size + 1)


def test_text_encoder_tp_size_must_divide_world(monkeypatch):
    from vllm_omni.diffusion.distributed import parallel_state

    _fake_parallel_state(monkeypatch, 4, None)
    with pytest.raises(ValueError, match="evenly divide"):
        parallel_state.get_text_encoder_tp_group(3)


def test_text_encoder_tp_group_cannot_span_dp_replicas(monkeypatch):
    """DP=2, TP=2: an encoder group of 4 would all-reduce two different requests."""
    from vllm_omni.diffusion.distributed import parallel_state

    captured = _fake_parallel_state(monkeypatch, 4, [0, 2])
    with pytest.raises(ValueError, match="data-parallel replica"):
        parallel_state.get_text_encoder_tp_group(4)
    assert captured == {}


def test_text_encoder_tp_rejects_unexpected_dp_layout(monkeypatch):
    from vllm_omni.diffusion.distributed import parallel_state

    _fake_parallel_state(monkeypatch, 4, [0, 1])  # DP-innermost layout is not supported
    with pytest.raises(RuntimeError, match="DP-outermost"):
        parallel_state.get_text_encoder_tp_group(2)


def _gloo_worker(rank: int, world_size: int, dp_size: int, tp_size: int, init_file: str) -> None:
    from datetime import timedelta

    import torch.distributed as dist

    from vllm_omni.diffusion.distributed import parallel_state

    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size, timeout=timedelta(seconds=60)
    )
    try:
        parallel_state._WORLD = parallel_state.init_world_group(list(range(world_size)), 0, "gloo")
        replica_size = world_size // dp_size
        if dp_size > 1:
            dp_groups = [list(range(i, world_size, replica_size)) for i in range(replica_size)]
            parallel_state._DP = parallel_state.init_model_parallel_group(dp_groups, 0, "gloo", parallel_mode="data")
        group = parallel_state.get_text_encoder_tp_group(tp_size)
        assert group.ranks == list(range(rank // tp_size * tp_size, (rank // tp_size + 1) * tp_size))

        model = _tiny_qwen25_vl()
        # Each DP replica serves a different request (different tokens and length).
        replica = rank // replica_size
        generator = torch.Generator().manual_seed(100 + replica)
        input_ids = torch.randint(0, 128, (2, 7 + 3 * replica), generator=generator)
        expected = _last_hidden(model, input_ids)

        shard_text_encoder(model, group)
        assert isinstance(model.model.language_model.layers[0].self_attn.o_proj, RowShardedLinear)
        torch.testing.assert_close(_last_hidden(model, input_ids), expected, rtol=1e-4, atol=1e-4)
    finally:
        parallel_state.destroy_model_parallel()
        parallel_state._WORLD = None
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("world_size", "dp_size", "tp_size"),
    [
        (2, 1, 2),  # encoder TP over the whole world
        (4, 1, 2),  # two encoder groups tiling one replica
        (4, 2, 2),  # one encoder group per DP replica, different requests per replica
    ],
)
def test_text_encoder_tp_real_process_groups(tmp_path, world_size, dp_size, tp_size):
    """Real gloo process groups and collectives, one process per rank."""
    torch.multiprocessing.spawn(
        _gloo_worker,
        args=(world_size, dp_size, tp_size, str(tmp_path / "rendezvous")),
        nprocs=world_size,
        join=True,
    )
