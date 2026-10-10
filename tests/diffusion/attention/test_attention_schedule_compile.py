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
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
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
    """The eager boundary: each block splits around attention and no graph holds the kernel."""
    assert len(counter.graphs) == graphs, counter.graphs
    for calls in counter.graphs:
        assert "scaled_dot_product_attention" not in calls, calls
        assert "scheduled_attention" not in calls, calls
        assert not {"sin", "cos"} <= calls, calls


def _require_one_graph_with_scheduled_op(counter):
    """Every block shares one graph that holds the opaque op, so the selection still runs outside Dynamo."""
    assert len(counter.graphs) == 1, counter.graphs
    calls = counter.graphs[0]
    assert {"sin", "scheduled_attention", "cos"} <= calls, calls
    assert "scaled_dot_product_attention" not in calls, calls


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
    _require_one_graph_with_scheduled_op(pipeline.counter)


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
    _require_one_graph_with_scheduled_op(pipeline.counter)


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
    _require_one_graph_with_scheduled_op(pipeline.counter)


def test_fullgraph_compile_contains_the_scheduled_op(compile_env):
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
    steps = scheduled.run(schedule)

    assert [step.trace for step in steps] == _expected("DDAAADDD", {"D": _DENSE, "A": _APPROX})
    _require_compiled_steps(scheduled.counter, steps)
    _require_one_graph_with_scheduled_op(scheduled.counter)


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
    _require_one_graph_with_scheduled_op(pipeline.counter)


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
    _require_one_graph_with_scheduled_op(pipeline.counter)


class _MetadataBlock(_Block):
    """A block whose attention call carries metadata the scheduled-attention op cannot represent."""

    def forward(self, hidden_states):
        batch, seq, _ = hidden_states.shape
        qkv = torch.sin(self.to_qkv(hidden_states)).view(batch, seq, 3, _HEADS, _HEAD_SIZE)
        query, key, value = qkv.unbind(2)
        out = self.attn(query, key, value, AttentionMetadata(extra={"backend_private": torch.ones(1)}))
        return hidden_states + torch.cos(self.to_out(out.reshape(batch, seq, _HIDDEN)))


class _MetadataModel(_Model):
    _repeated_blocks = ["_MetadataBlock"]

    def __init__(self, blocks=_BLOCKS):
        nn.Module.__init__(self)
        self.blocks = nn.ModuleList(_MetadataBlock(index) for index in range(blocks))


class _InductorCounting(_CountingBackend):
    """``_CountingBackend`` that compiles each graph with Inductor."""

    def __call__(self, gm, example_inputs):
        from torch._inductor.compile_fx import compile_fx

        index = len(self.graphs)
        self.graphs.append({_call_name(node) for node in gm.graph.nodes if node.op.startswith("call_")})
        self.executions.append(0)
        compiled = compile_fx(gm, example_inputs)

        def run(*args):
            self.executions[index] += 1
            return compiled(*args)

        return run


def test_metadata_the_op_cannot_carry_keeps_the_eager_boundary(compile_env):
    config = _config(AttentionScheduleConfig(profiles={"approx": _trtllm(0.5)}))
    with set_current_diffusion_config(config):
        torch.manual_seed(_WEIGHT_SEED)
        model = _MetadataModel()
    counter = _CountingBackend()
    regionally_compile(model, backend=counter)
    pipeline = _ToyPipeline(model, config, counter)

    steps = pipeline.run(_ranges((2, 5, "approx")))

    assert [step.trace for step in steps] == _expected("DDAAADDD", {"D": _DENSE, "A": _APPROX})
    _require_compiled_steps(counter, steps)
    _require_attention_outside_graphs(counter)


def test_autograd_keeps_the_eager_boundary(compile_env):
    pipeline = _pipeline(AttentionScheduleConfig(profiles={"approx": _trtllm(0.5)}))

    steps = pipeline.run(_ranges((2, 5, "approx")), grad_enabled=True)

    assert [step.trace for step in steps] == _expected("DDAAADDD", {"D": _DENSE, "A": _APPROX})
    _require_compiled_steps(pipeline.counter, steps)
    # Step 0 compiles for a leaf latent; later steps see latents that require grad and compile once more.
    _require_attention_outside_graphs(pipeline.counter, graphs=4)


def test_repeated_blocks_share_one_graph_under_dynamic_shapes(compile_env):
    pipeline = _pipeline(AttentionScheduleConfig(profiles={"approx": _trtllm(0.5)}), dynamic=True, blocks=6)

    first = pipeline.run(_ranges((2, 5, "approx")))
    with torch._dynamo.config.patch(error_on_recompile=True):
        second = pipeline.run(_ranges((0, 3, "approx")))

    assert [step.trace for step in first] == _expected("DDAAADDD", {"D": _DENSE, "A": _APPROX}, blocks=6)
    assert [step.trace for step in second] == _expected("AAADDDDD", {"D": _DENSE, "A": _APPROX}, blocks=6)
    _require_compiled_steps(pipeline.counter, first + second, blocks=6)
    _require_no_compile_after_first_step([first, second])
    _require_one_graph_with_scheduled_op(pipeline.counter)


def test_dense_request_matches_a_service_without_a_schedule_bit_for_bit(compile_env):
    plain = _pipeline(None, backend=_InductorCounting(), dynamic=True)
    plain_steps = plain.run(None)
    scheduled = _pipeline(
        AttentionScheduleConfig(profiles={"approx": _trtllm(0.5)}), backend=_InductorCounting(), dynamic=True
    )
    dense_steps = scheduled.run(())

    for index, (expected, actual) in enumerate(zip(plain_steps, dense_steps)):
        assert torch.equal(expected.latents, actual.latents), index
    _require_one_graph_with_scheduled_op(scheduled.counter)
    assert "scaled_dot_product_attention" in plain.counter.graphs[0]


def test_scheduled_op_id_is_a_cpu_attribute_outside_the_state_dict(compile_env):
    pipeline = _pipeline(AttentionScheduleConfig(profiles={"approx": _trtllm(0.5)}), compiled=False)
    model = pipeline.model.to(torch.float64)
    ids = set()
    for block in model.blocks:
        op_id = block.attn._schedule_op_id
        assert op_id.device.type == "cpu" and op_id.dtype == torch.int64
        assert layer_mod._SCHEDULED_ATTENTION_LAYERS[int(op_id)] is block.attn
        ids.add(int(op_id))
    assert len(ids) == len(model.blocks)
    assert not any("schedule_op_id" in name for name in model.state_dict())
    assert _pipeline(None, compiled=False).model.blocks[0].attn._schedule_op_id is None


def test_scheduled_op_returns_a_fresh_contiguous_tensor_and_rejects_a_wrong_shape(compile_env, monkeypatch):
    pipeline = _pipeline(AttentionScheduleConfig(profiles={"approx": _trtllm(0.5)}), compiled=False)
    attn = pipeline.model.blocks[0].attn
    query = torch.randn(1, _SEQ, _HEADS, _HEAD_SIZE)

    monkeypatch.setattr(attn, "_forward_impl", lambda q, k, v, md=None: q)
    out = layer_mod._scheduled_attention_op(query, query, query, None, attn._schedule_op_id, "-", [])
    assert torch.equal(out, query) and out.untyped_storage().data_ptr() != query.untyped_storage().data_ptr()

    monkeypatch.setattr(attn, "_forward_impl", lambda q, k, v, md=None: q.transpose(1, 2))
    with pytest.raises(RuntimeError, match="compiled graph expects"):
        layer_mod._scheduled_attention_op(query, query, query, None, attn._schedule_op_id, "-", [])


def test_op_metadata_rule_covers_every_attention_metadata_field():
    from dataclasses import fields

    carried_or_checked = {"attn_mask", "extra", "joint_strategy", "video_layout"}
    assert {f.name for f in fields(AttentionMetadata)} - carried_or_checked == set(layer_mod._SCHEDULED_OP_NONE_FIELDS)


def test_op_metadata_round_trips_layouts_and_scalar_extras():
    from vllm_omni.diffusion.attention.backends.abstract import VideoTokenLayout

    mask = torch.ones(1, 4, dtype=torch.bool)
    metadata = AttentionMetadata(
        attn_mask=mask,
        extra={"vsa_dit_seq_shape": (3, 8, 8), "preserve_vsa_all_blocks": True, "valid_kv_length": 5, "ids": [1, 2]},
        video_layout=VideoTokenLayout(prefix_len=0, latent_grid=(3, 8, 8), used_len=192),
    )
    spec, ints = layer_mod._encode_scheduled_op_metadata(metadata)
    assert spec == "M;Liig;Xids=l2;Xpreserve_vsa_all_blocks=b1;Xvalid_kv_length=i;Xvsa_dit_seq_shape=t3"
    assert ints == [0, 192, 3, 8, 8, 1, 2, 5, 3, 8, 8]
    decoded = layer_mod._decode_scheduled_op_metadata(mask, spec, ints)
    assert decoded == metadata and decoded.extra is not metadata.extra
    assert layer_mod._encode_scheduled_op_metadata(None) == ("-", [])
    assert layer_mod._decode_scheduled_op_metadata(None, "-", []) is None
    empty = layer_mod._encode_scheduled_op_metadata(AttentionMetadata())
    assert layer_mod._decode_scheduled_op_metadata(None, *empty) == AttentionMetadata()


@pytest.mark.parametrize(
    "metadata",
    [
        AttentionMetadata(extra={"gate_compress": torch.ones(1)}),
        AttentionMetadata(extra={"scale": 0.5}),
        AttentionMetadata(extra={"kv_cache_dtype": "fp8"}),
        AttentionMetadata(joint_query=torch.ones(1)),
        AttentionMetadata(joint_strategy="rear"),
    ],
    ids=["tensor", "float", "string", "joint", "rear"],
)
def test_op_metadata_it_cannot_carry_returns_none(metadata):
    assert layer_mod._encode_scheduled_op_metadata(metadata) is None


def test_op_metadata_with_video_spans_keeps_the_boundary():
    from vllm_omni.diffusion.attention.backends.abstract import VideoTokenLayout, VideoTokenSpan

    span = VideoTokenSpan(start=0, latent_grid=(1, 2, 2), role="target")
    metadata = AttentionMetadata(video_layout=VideoTokenLayout(used_len=4, video_spans=(span,)))
    assert layer_mod._encode_scheduled_op_metadata(metadata) is None
