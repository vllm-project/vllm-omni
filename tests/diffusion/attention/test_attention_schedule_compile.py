# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Scheduled attention under a real torch.compile.

A two-block toy transformer is compiled with the production ``regionally_compile`` helper and a
Dynamo backend that keeps every captured graph and counts its executions. Each block runs a linear
projection and ``torch.sin`` before a real ``Attention`` layer and a linear projection and
``torch.cos`` after it, so the two sides of the attention call can be told apart in the graphs.
The baseline and the prepared candidates compute scaled dot-product attention on CPU. The
TRTLLM_ATTN candidates subclass the production implementation, so ``__init__``, the skip config and
``_resolve_skip_factor`` (including the timestep gate) are production code. ``forward`` is replaced
by the SDPA math and passes ``key.shape[1]``, which equals the ``max_kv_len`` of ``forward_cuda`` for
the unpacked inputs used here. ``forward_cuda`` and the kernel are not exercised here.

Compilation on the target backend and Inductor is not covered here.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch._dynamo.utils import counters as dynamo_counters

import vllm_omni.diffusion.attention.layer as layer_mod
from vllm_omni.diffusion.attention.backends.trtllm_attn import TrtllmAttentionBackend, TrtllmAttentionImpl
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
from vllm_omni.diffusion.attention.schedule import (
    AttentionScheduleRange,
    parse_attention_sigma_schedule,
    select_attention_profile_by_sigma,
)
from vllm_omni.diffusion.compile import regionally_compile
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import (
    AttentionConfig,
    AttentionScheduleConfig,
    AttentionSpec,
    OmniDiffusionConfig,
    SkipSoftmaxSpec,
)
from vllm_omni.diffusion.forward_context import (
    DenoiseProgressMixin,
    begin_scheduled_denoise,
    bind_attention_schedule,
    bind_attention_sigma_schedule,
    set_forward_context,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]
_BLOCKS = 2
_HEADS = 2
_HEAD_SIZE = 16
_SEQ = 8
_HIDDEN = _HEADS * _HEAD_SIZE


def _timesteps(count):
    """``count`` evenly spaced timesteps on a 1000-step training scale, from 1000 down."""
    return [1000.0 * (count - index) / count for index in range(count)]


_TIMESTEPS = _timesteps(8)
_WEIGHT_SEED = 0
_LATENT_SEED = 1
_TRACE: list[tuple[str, Any, Any]] = []
_DENSE = ("SDPA", None, None)
_APPROX = ("TRTLLM_ATTN", 0.5, 4.0)


def _attention_math(query, key, value, scale, gain=None):
    """Scaled dot-product attention on [batch, seq, heads, head_size] tensors, times ``gain`` if given."""
    out = F.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), scale=scale)
    if gain is not None:
        out = out * gain
    return out.transpose(1, 2)


class _DenseImpl:
    """The baseline. It records a call only when it runs outside a traced graph."""

    def __init__(self, *, softmax_scale, **kwargs):
        self.softmax_scale = softmax_scale

    def forward(self, query, key, value, attn_metadata=None):
        out = _attention_math(query, key, value, self.softmax_scale)
        if not torch.compiler.is_compiling():
            _TRACE.append(_DENSE)
        return out


class _DenseBackend:
    supports_paged_kv = False
    supports_piecewise_spans = False

    @staticmethod
    def get_name() -> str:
        return "SDPA"

    @staticmethod
    def get_impl_cls():
        return _DenseImpl

    @staticmethod
    def supports_attention_mask(spec) -> bool:
        return True


class _RecordingTrtllmImpl(TrtllmAttentionImpl):
    """The production TRTLLM_ATTN implementation with ``forward`` replaced by the SDPA math."""

    def forward(self, query, key, value, attn_metadata=None):
        factor = self._resolve_skip_factor(key.shape[1])
        _TRACE.append(("TRTLLM_ATTN", self.skip.threshold, factor))
        gain = None if factor is None else 1.0 - self.skip.threshold
        out = _attention_math(query, key, value, self.softmax_scale, gain)
        return out


class _RecordingTrtllmBackend(TrtllmAttentionBackend):
    @staticmethod
    def get_impl_cls():
        return _RecordingTrtllmImpl


def _resolve(*, role, head_size, attention_config=None, role_category=None, allow_trtllm_default=True):
    """Stand-in for the platform selector; the platform default is _DenseBackend."""
    spec = None
    if attention_config is not None:
        spec, _source = attention_config.resolve_with_source(role=role, role_category=role_category)
    if spec is None:
        return _DenseBackend, None
    assert spec.backend.upper() == "TRTLLM_ATTN", spec.backend
    return _RecordingTrtllmBackend, spec


def _config(schedule: AttentionScheduleConfig | None = None) -> OmniDiffusionConfig:
    return OmniDiffusionConfig(diffusion_attention_schedule=schedule)


def _trtllm(threshold, disabled_until_timestep=0.0):
    skip = SkipSoftmaxSpec(threshold=threshold, disabled_until_timestep=disabled_until_timestep)
    return AttentionConfig(default=AttentionSpec(backend="TRTLLM_ATTN", skip_softmax=skip))


def _ranges(*entries):
    return tuple(AttentionScheduleRange(start=start, end=end, profile=profile) for start, end, profile in entries)


class _Block(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.to_qkv = nn.Linear(_HIDDEN, 3 * _HIDDEN)
        self.to_out = nn.Linear(_HIDDEN, _HIDDEN)
        self.attn = Attention(
            num_heads=_HEADS,
            head_size=_HEAD_SIZE,
            causal=False,
            softmax_scale=_HEAD_SIZE**-0.5,
            prefix=f"blocks.{index}.attn",
        )

    def forward(self, hidden_states):
        batch, seq, _ = hidden_states.shape
        qkv = torch.sin(self.to_qkv(hidden_states)).view(batch, seq, 3, _HEADS, _HEAD_SIZE)
        query, key, value = qkv.unbind(2)
        out = self.attn(query, key, value)
        return hidden_states + torch.cos(self.to_out(out.reshape(batch, seq, _HIDDEN)))


class _Model(nn.Module):
    _repeated_blocks = ["_Block"]

    def __init__(self, blocks=_BLOCKS):
        super().__init__()
        self.blocks = nn.ModuleList(_Block(index) for index in range(blocks))

    def forward(self, hidden_states):
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return hidden_states


def _call_name(node) -> str:
    return getattr(node.target, "__name__", str(node.target))


class _CountingBackend:
    """A Dynamo backend that keeps the calls in each captured graph and counts the graph's executions."""

    def __init__(self):
        self.graphs: list[set[str]] = []
        self.executions: list[int] = []

    def __call__(self, gm, example_inputs):
        index = len(self.graphs)
        self.graphs.append({_call_name(node) for node in gm.graph.nodes if node.op.startswith("call_")})
        self.executions.append(0)

        def run(*args):
            self.executions[index] += 1
            return gm.forward(*args)

        return run


def _frame_compiles() -> int:
    """Calls of Dynamo's fullgraph=False frame converter in this process."""
    return dynamo_counters["frames"]["total"]


@dataclass
class _Step:
    trace: list[tuple[str, Any, Any]]
    executions: list[int]
    graphs: int
    compiles: int | None
    latents: torch.Tensor


class _Run(list[_Step]):
    """The steps of one request."""


class _ToyPipeline(DenoiseProgressMixin):
    """A denoise loop around the model: it publishes each step, then runs the model once."""

    def __init__(self, model, config, counter, *, fullgraph=False):
        self.model = model
        self.config = config
        self.counter = counter
        self.fullgraph = fullgraph
        self.scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))

    def run(
        self,
        schedule,
        *,
        grad_enabled=False,
        inference_mode=False,
        timesteps=_TIMESTEPS,
        sigma_schedule=None,
        sigmas=None,
    ):
        """One request with the given request schedule; returns what each step ran."""
        assert not (grad_enabled and inference_mode)
        steps = _Run()
        with (
            torch.inference_mode() if inference_mode else torch.set_grad_enabled(grad_enabled),
            set_forward_context(omni_diffusion_config=self.config),
            bind_attention_schedule(schedule),
            bind_attention_sigma_schedule(sigma_schedule),
        ):
            latents = torch.randn(1, _SEQ, _HIDDEN, generator=torch.Generator().manual_seed(_LATENT_SEED))
            total = begin_scheduled_denoise(len(timesteps))
            for step_idx, timestep in enumerate(timesteps):
                self.record_denoise_step(
                    step_idx,
                    timestep,
                    total_steps=total,
                    normalized_sigma=None if sigmas is None else sigmas[step_idx],
                )
                trace_start = len(_TRACE)
                before = list(self.counter.executions)
                compiles_before = _frame_compiles()
                latents = latents - 0.1 * self.model(latents)
                compiles = None if self.fullgraph else _frame_compiles() - compiles_before
                executions = [
                    count - (before[index] if index < len(before) else 0)
                    for index, count in enumerate(self.counter.executions)
                ]
                steps.append(
                    _Step(
                        trace=_TRACE[trace_start:],
                        executions=executions,
                        graphs=len(self.counter.graphs),
                        compiles=compiles,
                        latents=latents,
                    )
                )
            self.record_denoise_step(None)
        return steps


@pytest.fixture
def compile_env(monkeypatch):
    monkeypatch.setattr(layer_mod.SDPABackend, "get_impl_cls", staticmethod(lambda: _DenseImpl))
    monkeypatch.setattr(layer_mod, "build_parallel_attention_strategy", lambda **kwargs: NoParallelAttention())
    monkeypatch.setattr(layer_mod, "get_attn_backend_for_role", lambda **kwargs: _resolve(**kwargs))
    _TRACE.clear()
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()
    _TRACE.clear()


def _pipeline(schedule_config, *, compiled=True, fullgraph=False, backend=None, dynamic=False, blocks=_BLOCKS):
    config = _config(schedule_config)
    with set_current_diffusion_config(config):
        torch.manual_seed(_WEIGHT_SEED)
        model = _Model(blocks)
    counter = _CountingBackend() if backend is None else backend
    if compiled:
        regionally_compile(model, backend=counter, fullgraph=fullgraph, dynamic=dynamic)
    return _ToyPipeline(model, config, counter, fullgraph=fullgraph)


def _require_compiled_steps(counter, steps, blocks=_BLOCKS):
    """Each step ran compiled graphs on both sides of every attention call."""
    assert counter.graphs, "no compiled graph was captured; the model ran eagerly"
    for index, step in enumerate(steps):
        runs = list(zip(counter.graphs, step.executions))
        before = sum(count for calls, count in runs if "sin" in calls)
        after = sum(count for calls, count in runs if "cos" in calls)
        assert (before, after) == (blocks, blocks), f"step {index}: compiled executions {before=} {after=}"


def _require_no_compile_after_first_step(requests, also_allowed=()):
    """Dynamo's frame converter is called only during the first step of the first request."""
    assert requests[0][0].compiles, (
        "the first step compiled no frame, so the frame counter did not change and cannot show a later compile"
    )
    for request, steps in enumerate(requests):
        for index, step in enumerate(steps):
            if (request, index) != (0, 0) and (request, index) not in also_allowed:
                assert step.compiles == 0, (
                    f"request {request} step {index}: {step.compiles} frame compile attempt(s) after the first step"
                )


def _require_attention_outside_graphs(counter, graphs=2):
    assert len(counter.graphs) == graphs, counter.graphs
    for calls in counter.graphs:
        assert "scaled_dot_product_attention" not in calls, calls
        assert not {"sin", "cos"} <= calls, calls


def _expected(labels, entries, blocks=_BLOCKS):
    """Per-step traces: each letter in ``labels`` is one step, with one entry per block."""
    return [[entries[label]] * blocks for label in labels]


def test_unseen_boundaries_across_requests_add_no_graph(compile_env):
    pipeline = _pipeline(AttentionScheduleConfig(profiles={"approx": _trtllm(0.5)}))
    entries = {"D": _DENSE, "A": _APPROX}
    requests = [
        (_ranges((2, 5, "approx")), "DDAAADDD"),
        (_ranges((0, 1, "approx")), "ADDDDDDD"),
        (_ranges((3, None, "approx")), "DDDAAAAA"),
        (_ranges((1, 2, "approx"), (6, 7, "approx")), "DADDDDAD"),
        (_ranges((3, None, "approx")), "DDDAAA"),
        ((), "DDDDDDDD"),
        (None, "DDDDDDDD"),
    ]

    results: list[tuple[Any, list[_Step]]] = []
    first = pipeline.run(requests[0][0])
    results.append((requests[0][0], first))
    graphs = first[0].graphs
    with torch._dynamo.config.patch(error_on_recompile=True):
        for schedule, labels in requests[1:]:
            results.append((schedule, pipeline.run(schedule, timesteps=_timesteps(len(labels)))))

    for (schedule, labels), (_schedule, steps) in zip(requests, results):
        assert [step.trace for step in steps] == _expected(labels, entries), schedule
        assert [step.graphs for step in steps] == [graphs] * len(labels), schedule
        _require_compiled_steps(pipeline.counter, steps)
    _require_no_compile_after_first_step([steps for _schedule, steps in results])
    _require_attention_outside_graphs(pipeline.counter)


def test_dense_approximate_dense_keeps_the_prefix_and_routes_back(compile_env):
    pipeline = _pipeline(AttentionScheduleConfig(profiles={"approx": _trtllm(0.5)}))
    schedule = _ranges((2, 5, "approx"))

    scheduled = pipeline.run(schedule)
    dense = pipeline.run(())
    eager = _pipeline(None, compiled=False).run(None)

    assert [step.trace for step in scheduled] == _expected("DDAAADDD", {"D": _DENSE, "A": _APPROX})
    for index in (0, 1):
        assert torch.equal(scheduled[index].latents, dense[index].latents), index
    assert not torch.equal(scheduled[2].latents, dense[2].latents)
    for compiled_step, eager_step in zip(dense, eager):
        torch.testing.assert_close(compiled_step.latents, eager_step.latents)
    assert [step.graphs for step in scheduled + dense] == [scheduled[0].graphs] * (2 * len(_TIMESTEPS))
    _require_compiled_steps(pipeline.counter, scheduled + dense)
    _require_no_compile_after_first_step([scheduled, dense])
    _require_attention_outside_graphs(pipeline.counter)


def test_same_backend_candidates_run_their_own_parameters(compile_env):
    pipeline = _pipeline(AttentionScheduleConfig(profiles={"skip_low": _trtllm(0.25), "skip_high": _trtllm(0.75)}))
    schedule = _ranges((1, 3, "skip_low"), (5, 7, "skip_high"))

    steps = pipeline.run(schedule)

    for block in pipeline.model.blocks:
        candidates = block.attn._schedule_candidates
        assert candidates["skip_low"].impl is not candidates["skip_high"].impl
    entries = {"D": _DENSE, "L": ("TRTLLM_ATTN", 0.25, 2.0), "H": ("TRTLLM_ATTN", 0.75, 6.0)}
    assert [step.trace for step in steps] == _expected("DLLDDHHD", entries)
    assert [step.graphs for step in steps] == [steps[0].graphs] * len(_TIMESTEPS)
    _require_compiled_steps(pipeline.counter, steps)
    _require_no_compile_after_first_step([steps])
    _require_attention_outside_graphs(pipeline.counter)


def test_fullgraph_compile_cannot_contain_the_schedule_boundary(compile_env):
    pipeline = _pipeline(None, fullgraph=True)

    steps = pipeline.run(None)

    assert len(pipeline.counter.graphs) == 1, pipeline.counter.graphs
    assert {"sin", "scaled_dot_product_attention", "cos"} <= pipeline.counter.graphs[0]
    assert [step.trace for step in steps] == [[]] * len(_TIMESTEPS)
    _require_compiled_steps(pipeline.counter, steps)
    assert [step.graphs for step in steps] == [1] * len(_TIMESTEPS)

    torch._dynamo.reset()
    scheduled = _pipeline(AttentionScheduleConfig(profiles={"approx": _trtllm(0.5)}), fullgraph=True)
    schedule = _ranges((2, 5, "approx"))
    with pytest.raises(torch._dynamo.exc.Unsupported, match="disable"):
        scheduled.run(schedule)

    assert scheduled.counter.graphs == []
    assert _TRACE == []


@pytest.mark.parametrize(
    ("disabled_until_timestep", "labels"),
    [(0.5, "DDGGFFDD")],
    ids=["gate-on"],
)
def test_private_timestep_gate_runs_inside_the_scheduled_range(compile_env, disabled_until_timestep, labels):
    pipeline = _pipeline(AttentionScheduleConfig(profiles={"skip": _trtllm(0.5, disabled_until_timestep)}))
    schedule = _ranges((2, 6, "skip"))

    steps = pipeline.run(schedule)

    entries = {"D": _DENSE, "F": ("TRTLLM_ATTN", 0.5, 4.0), "G": ("TRTLLM_ATTN", 0.5, None)}
    assert [step.trace for step in steps] == _expected(labels, entries)
    assert [step.graphs for step in steps] == [steps[0].graphs] * len(_TIMESTEPS)
    _require_compiled_steps(pipeline.counter, steps)
    _require_no_compile_after_first_step([steps])
    _require_attention_outside_graphs(pipeline.counter)


def _equivalent_step_schedule(windows, sigmas):
    """Represent exactly the same profile choices as integer step ranges."""
    names = [select_attention_profile_by_sigma(windows, sigma) for sigma in sigmas]
    entries = []
    start = 0
    for end in range(1, len(names) + 1):
        if end == len(names) or names[end] != names[start]:
            if names[start] is not None:
                entries.append((start, end, names[start]))
            start = end
    return _ranges(*entries)


@pytest.mark.parametrize("dynamic,inference_mode", [(False, False), (True, True)])
def test_sigma_values_windows_counts_and_flow_shifts_do_not_recompile(compile_env, dynamic, inference_mode):
    # Both profiles use TRT, with distinct prepared configs; one keeps the real
    # private timestep gate. This is CPU math, not a TRT kernel/capture test.
    config = AttentionScheduleConfig(profiles={"approx": _trtllm(0.5), "gated": _trtllm(0.25, 0.6)})
    pipeline = _pipeline(config, dynamic=dynamic)
    requests = []
    window_sets = [
        [(0.0, 0.2, "gated"), (0.4, 0.8, "approx")],
        [(0.1, 0.6, "approx"), (0.7, 1.0, "gated")],
        [(0.0, 0.35, "approx"), (0.35, 1.0, "gated")],
    ]
    for count, shift, entries in zip((37, 21, 43), (1.0, 3.0, 7.0), window_sets):
        windows = parse_attention_sigma_schedule(
            [{"low": low, "high": high, "profile": name} for low, high, name in entries]
        )
        noise = [1.0 - i / (count - 1) for i in range(count)]
        sigmas = [shift * s / (1.0 + (shift - 1.0) * s) for s in noise]
        timesteps = _timesteps(count)
        steps = pipeline.run(
            None, sigma_schedule=windows, sigmas=sigmas, timesteps=timesteps, inference_mode=inference_mode
        )
        requests.append(steps)
        _require_compiled_steps(pipeline.counter, steps)

        # Real tensor trajectories, not only selection traces, agree with the
        # equivalent step-index schedule and with an eager sigma execution.
        eager = _pipeline(config, compiled=False)
        by_step = eager.run(
            _equivalent_step_schedule(windows, sigmas), timesteps=timesteps, inference_mode=inference_mode
        )
        by_sigma = eager.run(
            None, sigma_schedule=windows, sigmas=sigmas, timesteps=timesteps, inference_mode=inference_mode
        )
        for compiled_step, step_step, sigma_step in zip(steps, by_step, by_sigma):
            torch.testing.assert_close(compiled_step.latents, step_step.latents)
            torch.testing.assert_close(compiled_step.latents, sigma_step.latents)
            assert compiled_step.trace == step_step.trace == sigma_step.trace
    _require_no_compile_after_first_step(requests)
    _require_attention_outside_graphs(pipeline.counter)
