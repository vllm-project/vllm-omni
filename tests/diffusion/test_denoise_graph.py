# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Denoise-graph policy (``vllm_omni/diffusion/graph.py``) on CPU.

The capture/replay machinery itself is ``GraphedForward``'s and is covered by
``test_xpu_graph.py``; here a CPU subclass "captures" by running the body and
"replays" by running it again into the captured output.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import vllm_omni.diffusion.cuda_graph as cuda_graph_mod
import vllm_omni.diffusion.distributed.utils as dist_utils
import vllm_omni.diffusion.graph as graph_mod
from vllm_omni.diffusion.cuda_graph import GraphedForward, copy_into
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, SkipSoftmaxSpec
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.graph import (
    DENOISE_GRAPH_MAX_ENTRIES,
    build_denoise_graph_runner,
    invalidate_denoise_graphs,
    write_through,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _CPUGraphedForward(GraphedForward):
    graph_kind = "fake"
    _sync_probe = False

    def _graphs_supported(self) -> bool:
        return True

    @contextmanager
    def _warmup_stream(self) -> Iterator[None]:
        yield

    def _capture_graph(self, body: Callable[[], Any]) -> tuple[Any, Any]:
        outputs = body()
        return SimpleNamespace(replay=lambda: copy_into(outputs, body())), outputs


@pytest.fixture
def platform(monkeypatch) -> SimpleNamespace:
    fake = SimpleNamespace(device_type="cpu", outer_capture=False, graph_cls=_CPUGraphedForward)
    fake.is_graph_capturing = lambda: fake.outer_capture
    fake.get_graphed_forward_cls = lambda: fake.graph_cls
    monkeypatch.setattr(graph_mod, "current_omni_platform", fake)
    monkeypatch.setattr(cuda_graph_mod, "current_omni_platform", fake)
    monkeypatch.setattr(dist_utils, "get_local_device", lambda: torch.device("cpu"))
    return fake


def _od_config(**overrides):
    fields = dict(
        diffusion_use_graph=True,
        enforce_eager=False,
        diffusion_kv_mode=DiffusionKVCacheMode.DENSE_LEGACY,
        cache_backend="none",
        enable_cpu_offload=False,
        enable_layerwise_offload=False,
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _affine(*, x: torch.Tensor, scale: float = 2.0) -> torch.Tensor:
    return x * scale + 1


def test_runner_replays_fresh_inputs_into_fresh_outputs(platform):
    runner = build_denoise_graph_runner(_od_config(), _affine, "T")
    assert isinstance(runner, _CPUGraphedForward)
    assert (runner.name, runner.max_entries) == ("T", DENOISE_GRAPH_MAX_ENTRIES)

    first = runner(x=torch.ones(3))
    second = runner(x=torch.full((3,), 2.0))

    torch.testing.assert_close(first, torch.full((3,), 3.0))
    torch.testing.assert_close(second, torch.full((3,), 5.0))
    assert runner.stats()["captures"] == 1
    assert runner.stats()["replays"] == 2


def test_a_changed_flag_captures_its_own_graph(platform):
    runner = build_denoise_graph_runner(_od_config(), _affine, "T")

    torch.testing.assert_close(runner(x=torch.ones(2), scale=2.0), torch.full((2,), 3.0))
    torch.testing.assert_close(runner(x=torch.ones(2), scale=3.0), torch.full((2,), 4.0))

    assert runner.stats()["captures"] == 2


def test_list_inputs_and_outputs_replay_per_item(platform):
    # Z-Image's step takes and returns per-sample lists, which may differ in length.
    runner = build_denoise_graph_runner(_od_config(), lambda *, xs: [x * 2 for x in xs], "T")

    runner(xs=[torch.ones(2), torch.ones(3)])
    out = runner(xs=[torch.full((2,), 2.0), torch.full((3,), 3.0)])

    assert isinstance(out, list)
    torch.testing.assert_close(out[0], torch.full((2,), 4.0))
    torch.testing.assert_close(out[1], torch.full((3,), 6.0))
    assert runner.stats()["captures"] == 1


def test_an_opaque_argument_turns_graphs_off(platform):
    # Pipelines must pass names, not modules (see Wan's ``component``): a module
    # would keep its capture-time identity on replay.
    runner = build_denoise_graph_runner(_od_config(), lambda *, model, x: model(x), "T")

    out = runner(model=torch.nn.Identity(), x=torch.ones(2))

    torch.testing.assert_close(out, torch.ones(2))
    assert not runner.enabled


def test_invalidate_denoise_graphs_drops_every_runner_and_keeps_capturing(platform):
    runners = [build_denoise_graph_runner(_od_config(), _affine, name) for name in ("A", "B")]
    for runner in runners:
        runner(x=torch.ones(2))

    invalidate_denoise_graphs("adapter changed")

    assert [runner.num_graphs for runner in runners] == [0, 0]
    torch.testing.assert_close(runners[0](x=torch.ones(2)), torch.full((2,), 3.0))
    assert runners[0].stats()["captures"] == 2
    assert runners[0].enabled


def test_platform_without_graphs_stays_eager(platform):
    platform.graph_cls = GraphedForward  # CUDA-only, so disabled on CPU

    assert build_denoise_graph_runner(_od_config(), _affine, "T") is None


_GATED_SKIP_SOFTMAX = AttentionSpec(
    backend="TRTLLM_ATTN", skip_softmax=SkipSoftmaxSpec(threshold=0.1, disabled_until_timestep=0.5)
)


@pytest.mark.parametrize(
    ("overrides", "refusal"),
    [
        ({"diffusion_use_graph": False}, None),
        ({"enforce_eager": True}, None),
        ({"diffusion_kv_mode": DiffusionKVCacheMode.PAGED_SCHEDULER}, None),
        ({"diffusion_kv_mode": "paged_scheduler"}, None),
        ({"cache_backend": "cache_dit"}, None),
        ({"enable_cpu_offload": True}, None),
        ({"enable_layerwise_offload": True}, None),
        ({"enable_distributed_layerwise_offload": True}, None),
        ({"diffusion_offload_config": {"mode": "layer", "components": ["dit"]}}, None),
        ({"parallel_config": SimpleNamespace(pipeline_parallel_size=2)}, None),
        ({"diffusion_kv_cache_dtype": "fp8", "diffusion_kv_cache_skip_step_indices": (1,)}, None),
        ({"diffusion_attention_config": AttentionConfig(default=_GATED_SKIP_SOFTMAX)}, None),
        ({"diffusion_attention_config": AttentionConfig(per_role={"self": _GATED_SKIP_SOFTMAX})}, None),
        ({}, "pipeline-specific reason"),
    ],
)
def test_build_runner_stays_eager_when_unsafe(platform, overrides, refusal):
    assert build_denoise_graph_runner(_od_config(**overrides), _affine, "T", refusal=refusal) is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"diffusion_kv_cache_dtype": "fp8", "diffusion_kv_cache_skip_step_indices": ()},
        {"diffusion_kv_cache_skip_step_indices": (1,)},
        {"diffusion_kv_cache_dtype": "auto", "diffusion_kv_cache_skip_step_indices": (1,)},
        {
            "diffusion_attention_config": AttentionConfig(
                default=AttentionSpec(backend="TRTLLM_ATTN", skip_softmax=SkipSoftmaxSpec(threshold=0.1))
            )
        },
    ],
    ids=["no-skip-steps", "skip-steps-without-kv-quant", "skip-steps-with-auto-kv", "ungated-skip-softmax"],
)
def test_build_runner_allows_step_independent_attention(platform, overrides):
    assert build_denoise_graph_runner(_od_config(**overrides), _affine, "T") is not None


def test_write_through_reuses_storage_only_when_layout_matches():
    dst = torch.zeros(2, 3)
    src = torch.ones(2, 3)
    assert write_through(dst, src) is dst
    torch.testing.assert_close(dst, src)

    for other in (torch.ones(3, 2), torch.ones(2, 3, dtype=torch.float64)):
        assert write_through(dst, other) is other
    assert write_through(None, src) is src


def test_is_capturing_follows_the_platform_on_its_device(platform):
    assert graph_mod.is_capturing(torch.device("cpu")) is False
    platform.outer_capture = True
    assert graph_mod.is_capturing(torch.device("cpu")) is True
    assert graph_mod.is_capturing(torch.device("meta")) is False


@pytest.mark.parametrize(
    ("use_graph", "has_runner", "warns"),
    [(True, False, True), (True, True, False), (False, False, False)],
)
def test_warn_if_denoise_graph_unsupported(monkeypatch, use_graph, has_runner, warns):
    warnings = []
    monkeypatch.setattr(graph_mod.logger, "warning", lambda *args, **kwargs: warnings.append(args))
    pipeline = SimpleNamespace(_denoise_graph_runner=None) if has_runner else SimpleNamespace()

    graph_mod.warn_if_denoise_graph_unsupported(SimpleNamespace(diffusion_use_graph=use_graph), pipeline)

    assert bool(warnings) is warns
