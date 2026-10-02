# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Talker K-step frame graphs (``talker_kstep_graph``) vs. the eager ``decode_frames``.

On CPU the captured tail is emulated: either the tail functions run eagerly
(no graph factory), or a fake factory records each tail and replays it by
calling it again, which is exactly what a CUDA graph does with static buffers.
Both must give ``decode_frames``'s ids, emit flags, final generator states and
final ``inputs_embeds`` bit for bit. The real capture is
``test_talker_kstep_graph_cuda.py``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.logits_processor.builtin import LogitBiasLogitsProcessor, MinPLogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.minicpmo_4_5 import talker_kstep_graph as kg
from vllm_omni.model_executor.models.minicpmo_4_5.talker_frame_plan import CodecFramePlan
from vllm_omni.worker import gpu_talker_multiframe as mf

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

VOCAB, HIDDEN, EOS, WINDOW = 16, 8, 15, 16
PAD_ROWS = 3  # the talker graph's padded token count exceeds B * K


def native_sampler() -> Sampler:
    """vLLM's sampler on its PyTorch top-k/top-p path (what CUDA runs with seeded rows)."""
    sampler = Sampler()
    sampler.topk_topp_sampler.forward = sampler.topk_topp_sampler.forward_native
    return sampler


class Toy:
    """Codec embedding/head plus a causal 'talker graph' with static buffers.

    ``replay()`` recomputes every row of every request span from
    ``inputs_embeds`` and writes the static output in place, like a FULL-graph
    replay; ``fresh_at`` makes the given replays return a new tensor instead.
    """

    def __init__(self, batch: int, frames: int, seed: int = 0):
        gen = torch.Generator().manual_seed(seed)
        self.batch, self.frames = batch, frames
        self.emb = nn.Embedding(VOCAB, HIDDEN).requires_grad_(False)
        self.head = nn.Linear(HIDDEN, VOCAB).requires_grad_(False)
        with torch.no_grad():
            self.emb.weight.copy_(torch.randn(VOCAB, HIDDEN, generator=gen))
            self.head.weight.copy_(torch.randn(VOCAB, HIDDEN, generator=gen))
            self.head.bias.copy_(torch.randn(VOCAB, generator=gen))
            self.head.bias[EOS] = 1.5
        self.proj = torch.randn(HIDDEN, HIDDEN, generator=gen)
        self.prefix = torch.randn(batch, HIDDEN, generator=gen)
        self.inputs_embeds = torch.randn(batch * frames + PAD_ROWS, HIDDEN, generator=gen)
        self.out = torch.zeros(batch * frames + PAD_ROWS, HIDDEN)
        self.replays = 0
        self.fresh_at: set[int] = set()

    def replay(self) -> torch.Tensor:
        self.replays += 1
        hidden = torch.zeros_like(self.out)
        for i in range(self.batch):
            state = self.prefix[i]
            for j in range(self.frames):
                row = i * self.frames + j
                state = 0.7 * state + self.inputs_embeds[row]
                hidden[row] = torch.tanh(state @ self.proj) * 3.0
        if self.replays in self.fresh_at:
            return hidden
        self.out.copy_(hidden)
        return self.out

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.emb(ids)

    def codec_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(hidden).float()


def metadata(batch: int, mode: str) -> SamplingMetadata:
    """Frame metadata (``frame_sampling_metadata`` shape: no penalties, no min_tokens)."""
    greedy = mode == "greedy"
    if mode == "seeded":
        generators = {i: torch.Generator().manual_seed(100 + i) for i in range(batch)}
    elif mode == "mixed":
        generators = {i: torch.Generator().manual_seed(100 + i) for i in range(0, batch, 2)}
    else:
        generators = {}
    return SamplingMetadata(
        temperature=None if greedy else torch.full((batch,), 0.8),
        all_greedy=greedy,
        all_random=not greedy,
        top_p=None if greedy else torch.full((batch,), 0.85),
        top_k=None if greedy else torch.full((batch,), 6, dtype=torch.int32),
        generators=generators,
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(batch),
        presence_penalties=torch.zeros(batch),
        repetition_penalties=torch.ones(batch),
        output_token_ids=[],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def controls_for(batch: int, frames: int, step: int, scenario: str):
    """One step's ``build_controls`` output; ``full`` exercises every control."""
    gen = torch.Generator().manual_seed(1000 + step)
    force = [[False] * batch for _ in range(frames)]
    mask = [[False] * batch for _ in range(frames)]
    budgets = [frames + 5] * batch
    min_state: dict[int, tuple] = {}
    stop_ids = [{EOS} for _ in range(batch)]
    penalties = None
    if scenario == "full":
        for k in range(1, frames):
            mask[k][0] = k < frames - 2  # the Talker's own min_tokens holds EOS back
            if batch > 1 and k == frames - 1 - (step % 2):
                force[k][1] = True  # chunk limit: forced EOS mid-loop
        budgets[-1] = max(1, frames - 2 - step % 3)  # vLLM max_tokens truncates mid-loop
        min_state = {0: (frames + 3, [1, 2], {EOS, 4}, False)}
        stop_ids[0] = {EOS, 7}
        penalties = torch.full((batch,), 1.3)
    recent = [torch.randint(0, VOCAB - 1, (int(n),), generator=gen).tolist() for n in (0, 3, 20, 9)[:batch]]
    recent += [[] for _ in range(batch - len(recent))]
    plan = CodecFramePlan(
        request_ids=[f"r{i}" for i in range(batch)],
        eos_token_id=EOS,
        force_eos=force,
        mask_eos=mask,
        recent_codes=recent,
    )
    rows_host = torch.tensor([[i * frames + k for i in range(batch)] for k in range(frames)], dtype=torch.long)
    controls, (rows,) = mf.build_controls(
        plan,
        frames=frames,
        vocab_size=VOCAB,
        budgets=budgets,
        stop_ids=stop_ids,
        min_tokens_state=min_state,
        penalties=penalties,
        device=torch.device("cpu"),
        extra=[rows_host],
    )
    first = torch.randint(0, VOCAB - 1, (batch,), generator=gen)
    return controls, rows, first


def eager_frames(toy: Toy, first, frames, controls, rows, sampler, md):
    """``maybe_run``'s eager loop, verbatim."""

    def forward(k: int, embeds: torch.Tensor) -> torch.Tensor:
        toy.inputs_embeds.index_copy_(0, rows[k], embeds.to(toy.inputs_embeds.dtype))
        return toy.replay().index_select(0, rows[k])

    return mf.decode_frames(
        first,
        frames,
        controls,
        embed=toy.embed,
        forward=forward,
        codec_logits=toy.codec_logits,
        sample=lambda logits: sampler(logits, md).sampled_token_ids[:, 0],
    )


def graph_frames(graphs, toy: Toy, first, frames, controls, rows, sampler, md):
    return graphs(
        first=first,
        frames=frames,
        controls=controls,
        rows=rows,
        inputs_embeds=toy.inputs_embeds,
        hidden=toy.out,
        replay=toy.replay,
        embed=toy.embed,
        codec_logits=toy.codec_logits,
        sampler=sampler,
        sampling_metadata=md,
    )


class FakeGraph:
    """Records a tail; ``replay()`` reruns it, as a CUDA graph reruns its kernels."""

    made: list[FakeGraph] = []

    def __init__(self, fn, generators):
        self.fn, self.generators, self.replays = fn, list(generators), 0
        FakeGraph.made.append(self)

    def replay(self) -> None:
        self.replays += 1
        self.fn()


@pytest.fixture(autouse=True)
def _fresh_fake_graphs():
    FakeGraph.made = []
    yield


def _states(md: SamplingMetadata) -> dict[int, torch.Tensor]:
    return {i: g.get_state() for i, g in md.generators.items()}


def _run_pair(batch, frames, mode, scenario, steps, factory, fresh_at=()):
    """Drive both paths over ``steps`` steps from identical state; assert equality."""
    ref, new = Toy(batch, frames), Toy(batch, frames)
    new.fresh_at = set(fresh_at)
    md_ref, md_new = metadata(batch, mode), metadata(batch, mode)
    sampler = native_sampler()
    graphs = kg.TalkerKStepFrameGraphs(graph_factory=factory)
    results = []
    for step in range(steps):
        controls, rows, first = controls_for(batch, frames, step, scenario)
        torch.manual_seed(5000 + step)
        want = eager_frames(ref, first, frames, controls, rows, sampler, md_ref)
        rng_ref = torch.get_rng_state()
        torch.manual_seed(5000 + step)
        got = graph_frames(graphs, new, first, frames, controls, rows, sampler, md_new)
        results.append(got)
        if got is None:
            continue
        # Unseeded rows draw from the default generator: same draws, same end state.
        assert torch.equal(torch.get_rng_state(), rng_ref), f"step {step}: default generator state"
        assert torch.equal(got[0], want[0]), f"step {step}: sampled ids differ"
        assert torch.equal(got[1], want[1]), f"step {step}: emit flags differ"
        assert torch.equal(new.inputs_embeds, ref.inputs_embeds), f"step {step}: inputs_embeds differ"
        assert torch.equal(new.out, ref.out)
        for i, state in _states(md_ref).items():
            assert torch.equal(md_new.generators[i].get_state(), state), f"step {step}: generator {i} state"
    return graphs, results, (ref, new)


@pytest.mark.parametrize("frames", [2, 3, 8])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("mode", ["greedy", "seeded", "mixed", "unseeded"])
@pytest.mark.parametrize("scenario", ["plain", "full"])
def test_eager_tail_matches_decode_frames(scenario, mode, batch, frames):
    """No graph factory on CPU: the tail functions run eagerly, every step."""
    graphs, results, _ = _run_pair(batch, frames, mode, scenario, steps=3, factory=None)
    assert all(r is not None for r in results)
    assert graphs.stats.eager_steps == 3 and graphs.stats.graph_steps == 0
    assert not FakeGraph.made


@pytest.mark.parametrize("frames", [2, 3, 8])
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("mode", ["greedy", "seeded", "mixed"])
def test_replayed_tails_match_decode_frames(mode, batch, frames):
    """First step eager, then one capture (first/mid/last), then replays only."""
    graphs, results, (_, new) = _run_pair(batch, frames, mode, "full", steps=4, factory=FakeGraph)
    assert all(r is not None for r in results)
    assert graphs.stats.eager_steps == 1 and graphs.stats.graph_steps == 3
    assert graphs.stats.captures == 1
    names = 3 if frames > 2 else 2
    assert len(FakeGraph.made) == names
    first, *rest = FakeGraph.made
    assert first.replays == 3 and not first.generators
    # mid replays frames - 2 times a step, last once; both own the seeded rows' slots.
    assert sum(g.replays for g in rest) == 3 * (frames - 1)
    seeded = len(metadata(batch, mode).generators)
    assert all(len(g.generators) == seeded for g in rest)
    assert new.replays == 4 * (frames - 1)


def test_a_moved_forward_buffer_finishes_eagerly_then_declines():
    """A replay that returns a new tensor (not the captured output): that step
    finishes on what it returned, identically; the shape then stays eager."""
    batch, frames = 2, 8
    # Step 0 eager + capture; step 1 replays; the 3rd forward of step 2 moves.
    fresh = {2 * (frames - 1) + 3}
    graphs, results, _ = _run_pair(batch, frames, "seeded", "full", steps=4, factory=FakeGraph, fresh_at=fresh)
    assert [r is None for r in results] == [False, False, False, True]
    assert graphs.stats.unstable == 1


def test_capture_failure_keeps_the_eager_tail():
    def broken(fn, generators):
        raise RuntimeError("operation not permitted when stream is capturing")

    graphs, results, _ = _run_pair(1, 4, "seeded", "full", steps=3, factory=broken)
    assert results[0] is not None and results[1] is None and results[2] is None
    assert graphs.stats.captures == 0


def test_capture_does_not_move_generators():
    """Capturing records the RNG ops without running them; a capture that does
    consume randoms must still leave every slot where eager sampling expects it."""

    def draws_while_capturing(fn, generators):
        for gen in generators:
            torch.rand(4, generator=gen)
        torch.rand(4)  # the default generator, which unseeded rows draw from
        return FakeGraph(fn, generators)

    graphs, results, _ = _run_pair(3, 4, "mixed", "plain", steps=3, factory=draws_while_capturing)
    assert all(r is not None for r in results) and graphs.stats.graph_steps == 2


def test_shapes_are_keyed_by_rows_and_frames():
    graphs = kg.TalkerKStepFrameGraphs(graph_factory=FakeGraph)
    sampler = native_sampler()
    for batch in (1, 2, 1, 2):
        toy = Toy(batch, 4)
        md = metadata(batch, "seeded")
        controls, rows, first = controls_for(batch, 4, 0, "plain")
        assert graph_frames(graphs, toy, first, 4, controls, rows, sampler, md) is not None
    # Each toy has its own buffers, so every call is a new shape: eager + capture.
    assert graphs.stats.captures == 4 and graphs.stats.graph_steps == 0


# ---------------------------------------------------------------------------
# Declines: what a graph cannot reproduce goes back to decode_frames.
# ---------------------------------------------------------------------------


def _controls(width: int = 1):
    return SimpleNamespace(stop_ids=torch.zeros((1, width), dtype=torch.long))


def test_flashinfer_sampling_without_generators_declines():
    md = metadata(2, "unseeded")
    flashinfer = SimpleNamespace(topk_topp_sampler=SimpleNamespace(forward=lambda: None, use_fp64_gumbel=False))
    flashinfer.topk_topp_sampler.forward.__name__ = "forward_cuda"
    assert "FlashInfer" in kg.graph_decline(flashinfer, md, _controls())
    # forward_cuda takes forward_native with per-request generators, or with no top-k/top-p.
    assert kg.graph_decline(flashinfer, metadata(2, "mixed"), _controls()) is None
    assert kg.graph_decline(flashinfer, replace(md, top_k=None, top_p=None), _controls()) is None
    assert kg.graph_decline(flashinfer, metadata(2, "greedy"), _controls()) is None
    assert kg.graph_decline(native_sampler(), md, _controls()) is None
    cpu = Sampler()
    if cpu.topk_topp_sampler.forward.__name__ == "forward_cpu":
        assert kg.graph_decline(cpu, md, _controls()) is not None


def test_active_min_p_or_logit_bias_and_wide_stop_sets_decline():
    md = metadata(2, "seeded")
    min_p = MinPLogitsProcessor.__new__(MinPLogitsProcessor)
    min_p.min_p_count = 1
    bias = LogitBiasLogitsProcessor.__new__(LogitBiasLogitsProcessor)
    bias.biases = {0: {3: 1.0}}
    assert "MinPLogitsProcessor" in kg.graph_decline(
        native_sampler(), replace(md, logitsprocs=LogitsProcessors([min_p])), _controls()
    )
    assert "LogitBias" in kg.graph_decline(
        native_sampler(), replace(md, logitsprocs=LogitsProcessors([bias])), _controls()
    )
    min_p.min_p_count = 0
    bias.biases = {}
    idle = replace(md, logitsprocs=LogitsProcessors([min_p, bias]))
    assert kg.graph_decline(native_sampler(), idle, _controls()) is None
    assert "stop ids" in kg.graph_decline(native_sampler(), md, _controls(kg.STOP_WIDTH + 1))
    assert kg.graph_decline(native_sampler(), md, _controls(kg.STOP_WIDTH)) is None


def test_a_non_full_talker_forward_declines(monkeypatch):
    monkeypatch.setattr(kg, "_forward_graph_mode", lambda: "PIECEWISE")
    toy, md = Toy(1, 4), metadata(1, "seeded")
    controls, rows, first = controls_for(1, 4, 0, "plain")
    graphs = kg.TalkerKStepFrameGraphs(graph_factory=FakeGraph)
    assert graph_frames(graphs, toy, first, 4, controls, rows, native_sampler(), md) is None
    assert toy.replays == 0 and "PIECEWISE" in next(iter(graphs.stats.declined))


# ---------------------------------------------------------------------------
# Wiring: the MiniCPM-o wrapper hook, maybe_run's opt-in hook, the overlay.
# ---------------------------------------------------------------------------


def test_wrapper_hook_is_off_by_default():
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    wrapper = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    nn.Module.__init__(wrapper)
    assert wrapper.kstep_decode_frames(first=None) is None
    calls = []
    wrapper._kstep_frame_graphs = lambda **kw: calls.append(kw) or "ran"
    assert wrapper.kstep_decode_frames(first=1) == "ran" and calls == [{"first": 1}]


@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "seeded"])
def test_maybe_run_with_the_hook_matches_single_frame(monkeypatch, greedy: bool):
    """The real maybe_run / scheduler emulation of test_gpu_talker_multiframe with the
    hook installed (the harness's forward allocates a new output every call, so the
    hook finishes each step eagerly and then declines; the stable case is above)."""
    from tests.worker import test_gpu_talker_multiframe as base

    monkeypatch.setattr(mf, "_SAMPLER", native_sampler())
    results: list[bool] = []

    class Hooked(base._Model):
        def __init__(self, talker):
            super().__init__(talker)
            self.graphs = kg.TalkerKStepFrameGraphs(graph_factory=FakeGraph)

        def kstep_decode_frames(self, **kwargs):
            out = self.graphs(**kwargs)
            results.append(out is not None)
            return out

    def prepare(runner):
        runner.model = Hooked(runner.model.talker)

    ref = base._single_frame(greedy)
    got, widths = base._engine(greedy, 4, batch_vocab=base.VOCAB, prepare=prepare)
    base._assert_same(got, ref)
    assert set(widths[1:]) == {4}
    assert results and results[0]
    # Each step's new output buffer is a new shape that finishes eagerly; after
    # MAX_UNSTABLE_PER_SHAPE of them the (rows, frames) size stays on decode_frames.
    assert sum(results) <= kg.MAX_UNSTABLE_PER_SHAPE and not any(results[kg.MAX_UNSTABLE_PER_SHAPE :])


def test_maybe_run_hook_returning_none_keeps_decode_frames(monkeypatch):
    from tests.worker import test_gpu_talker_multiframe as base

    seen: list[dict] = []

    class Declines(base._Model):
        def kstep_decode_frames(self, **kwargs):
            seen.append(kwargs)
            return None

    ref = base._single_frame(False)
    got, _ = base._engine(
        False, 4, batch_vocab=base.VOCAB, prepare=lambda r: setattr(r, "model", Declines(r.model.talker))
    )
    base._assert_same(got, ref)
    assert seen and set(seen[0]) == {
        "first",
        "frames",
        "controls",
        "rows",
        "inputs_embeds",
        "hidden",
        "replay",
        "embed",
        "codec_logits",
        "sampler",
        "sampling_metadata",
    }
    assert seen[0]["sampler"] is mf._codec_sampler()


def test_s1graph_overlay_adds_only_the_stage1_switch():
    import vllm_omni
    from vllm_omni.config.stage_config import resolve_deploy_yaml

    deploy = Path(vllm_omni.__file__).parent / "deploy"
    raw = resolve_deploy_yaml(deploy / "minicpmo_4_5_dxsched_on_s1graph.yaml")
    base = resolve_deploy_yaml(deploy / "minicpmo_4_5_dxsched_on.yaml")
    stage1 = next(s for s in raw["stages"] if s["stage_id"] == 1)
    base1 = next(s for s in base["stages"] if s["stage_id"] == 1)
    assert stage1["hf_overrides"] == {**base1.get("hf_overrides", {}), "talker_kstep_graph_sampling": True}
    assert "talker_kstep_graph_sampling" not in base1.get("hf_overrides", {})
    strip = {**stage1}
    strip.pop("hf_overrides")
    base_strip = {**base1}
    base_strip.pop("hf_overrides", None)
    assert strip == base_strip
    assert [s for s in raw["stages"] if s["stage_id"] != 1] == [s for s in base["stages"] if s["stage_id"] != 1]
    assert {k: v for k, v in raw.items() if k not in ("stages", "base_config")} == {
        k: v for k, v in base.items() if k not in ("stages", "base_config")
    }
