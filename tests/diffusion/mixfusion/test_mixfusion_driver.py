# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Framework-level tests for the model-agnostic MixFusion driver (CPU).

Everything here runs against a fake transformer and a fake adapter with no
model imports, pinning the three responsibilities the driver owns:

- dispatch: dense batches go to ``dense_step`` unchanged, interrupted
  pipelines return ``None`` and ragged batches take the packed path only when
  admitted;
- admission: every rejection reason in ``MixFusionDriver.admit`` is enforced
  (missing adapter/hooks, attention kwargs, SP, true CFG, image latents,
  foreign hooks, chunk policy) and the per-request chunk policy is aggregated
  with the strictest request's limits;
- lifecycle: ``armed`` publishes the step the hooks read, restores any
  previous step (CFG calls the transformer twice) and survives exceptions;
  ``serial_ragged`` rebuilds one single-request dense batch per request.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.forward_context import (
    ForwardContext,
    MixFusionStep,
    get_forward_context,
    override_forward_context,
)
from vllm_omni.diffusion.hooks.mixfusion import apply_mixfusion
from vllm_omni.diffusion.mixfusion.driver import (
    DEFAULT_MAX_CHUNKS,
    DEFAULT_MIN_CHUNK_TOKENS,
    MixFusionDriver,
)
from vllm_omni.diffusion.mixfusion.plan import MixFusionRoot
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.utils import StepRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

PACKED_OUTPUT = "packed-output"
_DEFAULT = object()


class _FakeTransformer(nn.Module):
    """Minimal stand-in for a DiT: records the packed call and the armed step."""

    def __init__(self, *, plan: dict | None = None, hooks_applied: bool = True) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([nn.Identity()])
        # Mirrors the shape of a real plan: a wildcard picks modules out of a
        # ModuleList, which is where a foreign layerwise/FSDP hook would sit.
        self._mixfusion_plan = {"transformer_blocks.*": None} if plan is None else plan
        self._mixfusion_hooks_applied = hooks_applied
        self.calls: list[dict] = []
        self.armed_at_call: MixFusionStep | None = None

    def forward(self, hidden_states, **kwargs):
        # ``hidden_states`` is explicit, like a real model forward: the adapter
        # preflight reads it off ``BoundArguments.arguments``.
        self.armed_at_call = get_forward_context().mixfusion_step
        self.calls.append({"hidden_states": hidden_states, **kwargs})
        return PACKED_OUTPUT


class _FakeAdapter:
    """Packs by handing the per-request latents straight to the fake forward."""

    def __init__(self, *, can_pack: bool = True, refuse_reason: str = "refused") -> None:
        self._can_pack = can_pack
        self._refuse_reason = refuse_reason

    def build_packed_call(self, input_batch) -> dict:
        return {"hidden_states": input_batch.latents}

    def can_pack(self, module, bound) -> tuple[bool, str]:
        if "hidden_states" not in bound.arguments:
            raise TypeError("MixFusion packing requires a `hidden_states` argument.")
        return self._can_pack, "" if self._can_pack else self._refuse_reason


class _DriftingAdapter(_FakeAdapter):
    """Admits the admission preflight, then declines the armed hook's call."""

    def __init__(self) -> None:
        super().__init__()
        self.can_pack_calls = 0

    def can_pack(self, module, bound) -> tuple[bool, str]:
        self.can_pack_calls += 1
        if self.can_pack_calls == 1:
            return True, ""
        return False, "answer drifted after admission"


def _pipeline(*, interrupt: bool = False, attention_kwargs: dict | None = None, sp_size: int = 1):
    return SimpleNamespace(
        interrupt=interrupt,
        attention_kwargs=attention_kwargs,
        parallel_config=SimpleNamespace(sequence_parallel_size=sp_size),
    )


def _driver(
    transformer: _FakeTransformer | None = None,
    adapter: object = _DEFAULT,
    *,
    pipeline=None,
) -> tuple[MixFusionDriver, _FakeTransformer]:
    transformer = transformer if transformer is not None else _FakeTransformer()
    if adapter is _DEFAULT:
        adapter = _FakeAdapter()
    return MixFusionDriver(pipeline if pipeline is not None else _pipeline(), transformer, adapter), transformer


def _ragged_batch(
    *,
    lengths: tuple[int, ...] = (8, 12),
    extras: list[dict] | None = None,
    do_true_cfg: bool = False,
    image_latents: torch.Tensor | None = None,
) -> SimpleNamespace:
    if extras is None:
        extras = [{"mixfusion_min_chunk_tokens": 1} for _ in lengths]
    return SimpleNamespace(
        latents=[torch.zeros(1, length, 4) for length in lengths],
        timesteps=torch.tensor([500.0]),
        request_extras=extras,
        do_true_cfg=do_true_cfg,
        image_latents=image_latents,
    )


def _dense_batch() -> SimpleNamespace:
    return SimpleNamespace(latents=torch.zeros(2, 8, 4), timesteps=torch.tensor([500.0, 500.0]))


def _record_dense(calls: list, result: str = "dense"):
    def dense_step(batch, **kwargs):
        calls.append((batch, kwargs))
        return result

    return dense_step


def _state(request_id: str, latents: torch.Tensor) -> StepRequestState:
    """One ragged request state accepted by ``InputBatch.make_batch``."""
    return StepRequestState(
        request_id=request_id,
        sampling=SimpleNamespace(true_cfg_scale=None, cfg_normalize=False),
        latents=latents,
        timesteps=torch.tensor([500.0]),
        extra={"use_ragged_latents": True},
    )


# ── dispatch ──


def test_dense_batch_goes_straight_to_dense_step():
    driver, transformer = _driver()
    batch = _dense_batch()
    calls: list = []

    result = driver.run(batch, states=("s0", "s1"), dense_step=_record_dense(calls))

    assert result == "dense"
    assert calls == [(batch, {"states": ("s0", "s1")})]
    assert transformer.calls == []


def test_interrupted_pipeline_returns_none_without_denoising():
    driver, transformer = _driver(pipeline=_pipeline(interrupt=True))
    calls: list = []

    assert driver.run(_ragged_batch(), states=(), dense_step=_record_dense(calls)) is None
    assert calls == []
    assert transformer.calls == []


def test_run_packs_an_admissible_ragged_batch():
    driver, transformer = _driver()
    batch = _ragged_batch()
    ctx = ForwardContext()

    with override_forward_context(ctx):
        result = driver.run(batch, states=(), dense_step=_record_dense([]))

        assert result == PACKED_OUTPUT
        assert transformer.calls == [{"hidden_states": batch.latents}]
        step = transformer.armed_at_call
        assert isinstance(step, MixFusionStep)
        assert step.adapter is driver.adapter
        assert (step.min_chunk_tokens, step.max_chunks) == (1, DEFAULT_MAX_CHUNKS)
        # The packed branch bypasses the dense wrapper that publishes it.
        assert driver.pipeline._current_timestep is batch.timesteps
        # The armed step covers exactly the packed call.
        assert ctx.mixfusion_step is None


def test_run_falls_back_to_serial_ragged_when_admission_rejects():
    transformer = _FakeTransformer(hooks_applied=False)
    driver, _ = _driver(transformer)
    batch = _ragged_batch()
    serial_batches: list = []
    driver.serial_ragged = lambda input_batch, states, dense_step: serial_batches.append(input_batch) or "serial"

    with override_forward_context(ForwardContext()):
        result = driver.run(batch, states=(), dense_step=_record_dense([]))

    assert result == "serial"
    assert serial_batches == [batch]
    assert transformer.calls == []
    # Published even on the fallback path: the serial dense calls read it.
    assert driver.pipeline._current_timestep is batch.timesteps


def test_run_falls_back_to_serial_ragged_when_the_armed_hook_declines():
    # Admission accepted this call; the armed hook then asks again and gets a
    # different answer. The sentinel must become the serial fallback instead of
    # handing list latents to the model forward.
    transformer = _FakeTransformer(plan={"": MixFusionRoot()}, hooks_applied=False)
    apply_mixfusion(transformer, {"": MixFusionRoot()})
    adapter = _DriftingAdapter()
    driver, _ = _driver(transformer, adapter)
    batch = _ragged_batch()
    serial_batches: list = []
    driver.serial_ragged = lambda input_batch, states, dense_step: serial_batches.append(input_batch) or "serial"

    with override_forward_context(ForwardContext()):
        result = driver.run(batch, states=(), dense_step=_record_dense([]))

    assert result == "serial"
    assert serial_batches == [batch]
    assert transformer.calls == []
    assert adapter.can_pack_calls == 2


# ── admission ──


def test_admission_accepts_a_plain_ragged_batch():
    driver, _ = _driver()

    accepted, reason = driver.admit(_ragged_batch(), min_chunk_tokens=1, max_chunks=DEFAULT_MAX_CHUNKS)

    assert (accepted, reason) == (True, "ok")


def test_admission_rejects_when_the_adapter_declines_packing():
    driver, _ = _driver(adapter=_FakeAdapter(can_pack=False, refuse_reason="shape not supported"))

    accepted, reason = driver.admit(_ragged_batch(), min_chunk_tokens=1, max_chunks=DEFAULT_MAX_CHUNKS)

    assert not accepted
    assert reason == "adapter declined packing: shape not supported"


def test_admission_rejects_a_packed_call_the_adapter_cannot_bind():
    # A packed call without ``hidden_states`` makes ``can_pack`` raise
    # ``TypeError``; admission must reject rather than crash.
    class _IncompleteAdapter(_FakeAdapter):
        def build_packed_call(self, input_batch) -> dict:
            return {}

    driver, _ = _driver(adapter=_IncompleteAdapter())

    accepted, reason = driver.admit(_ragged_batch(), min_chunk_tokens=1, max_chunks=DEFAULT_MAX_CHUNKS)

    assert not accepted
    assert reason.startswith("adapter cannot bind the packed call: ")


def test_admission_rejects_a_transformer_without_an_adapter():
    driver, _ = _driver(adapter=None)

    accepted, reason = driver.admit(_ragged_batch(), min_chunk_tokens=1, max_chunks=DEFAULT_MAX_CHUNKS)

    assert not accepted
    assert "_FakeTransformer declares no MixFusion adapter" in reason


def test_admission_rejects_missing_hooks():
    driver, _ = _driver(_FakeTransformer(hooks_applied=False))

    accepted, reason = driver.admit(_ragged_batch(), min_chunk_tokens=1, max_chunks=DEFAULT_MAX_CHUNKS)

    assert not accepted
    assert reason == "mixfusion hooks not applied"


def test_admission_rejects_attention_kwargs():
    driver, _ = _driver(pipeline=_pipeline(attention_kwargs={"scale": 0.5}))

    accepted, reason = driver.admit(_ragged_batch(), min_chunk_tokens=1, max_chunks=DEFAULT_MAX_CHUNKS)

    assert not accepted
    assert reason == "attention_kwargs is set"


def test_admission_accepts_empty_attention_kwargs():
    driver, _ = _driver(pipeline=_pipeline(attention_kwargs={}))

    accepted, _ = driver.admit(_ragged_batch(), min_chunk_tokens=1, max_chunks=DEFAULT_MAX_CHUNKS)

    assert accepted


def test_admission_rejects_sequence_parallelism():
    driver, _ = _driver(pipeline=_pipeline(sp_size=2))

    accepted, reason = driver.admit(_ragged_batch(), min_chunk_tokens=1, max_chunks=DEFAULT_MAX_CHUNKS)

    assert not accepted
    assert reason == "sequence parallelism is enabled"


def test_admission_rejects_true_cfg():
    driver, _ = _driver()

    accepted, reason = driver.admit(
        _ragged_batch(do_true_cfg=True),
        min_chunk_tokens=1,
        max_chunks=DEFAULT_MAX_CHUNKS,
    )

    assert not accepted
    assert reason == "do_true_cfg is enabled"


def test_admission_rejects_image_latents():
    driver, _ = _driver()

    accepted, reason = driver.admit(
        _ragged_batch(image_latents=torch.zeros(1, 4, 4)),
        min_chunk_tokens=1,
        max_chunks=DEFAULT_MAX_CHUNKS,
    )

    assert not accepted
    assert reason == "image_latents is set"


def test_admission_rejects_foreign_hooks_on_planned_modules():
    transformer = _FakeTransformer()
    transformer.transformer_blocks[0].register_forward_pre_hook(lambda module, args: None)
    driver, _ = _driver(transformer)

    accepted, reason = driver.admit(_ragged_batch(), min_chunk_tokens=1, max_chunks=DEFAULT_MAX_CHUNKS)

    assert not accepted
    assert reason == "foreign hooks on the packed path (transformer_blocks.*:torch-native)"


def test_admission_rejects_foreign_hooks_on_the_root():
    transformer = _FakeTransformer(plan={})
    transformer.register_forward_pre_hook(lambda module, args: None)
    driver, _ = _driver(transformer)

    accepted, reason = driver.admit(_ragged_batch(), min_chunk_tokens=1, max_chunks=DEFAULT_MAX_CHUNKS)

    assert not accepted
    assert reason == "foreign hooks on the packed path (<root>:torch-native)"


def test_admission_rejects_chunks_below_the_minimum_token_count():
    driver, _ = _driver()

    accepted, reason = driver.admit(_ragged_batch(lengths=(4,)), min_chunk_tokens=8, max_chunks=DEFAULT_MAX_CHUNKS)

    assert not accepted
    assert reason == "chunk_size=4 < min_chunk_tokens=8"


def test_admission_rejects_more_chunks_than_allowed():
    driver, _ = _driver()

    accepted, reason = driver.admit(_ragged_batch(), min_chunk_tokens=1, max_chunks=4)

    assert not accepted
    assert reason == "chunk_count=5 > max_chunks=4"


# ── chunk policy ──


def test_chunk_policy_defaults_without_request_extras():
    driver, _ = _driver()

    assert MixFusionDriver._chunk_policy(SimpleNamespace(request_extras=None)) == (
        DEFAULT_MIN_CHUNK_TOKENS,
        DEFAULT_MAX_CHUNKS,
    )


def test_chunk_policy_aggregates_the_strictest_request():
    driver, _ = _driver()
    batch = SimpleNamespace(
        request_extras=[
            {"mixfusion_min_chunk_tokens": 64, "mixfusion_max_chunks": 8},
            {"mixfusion_min_chunk_tokens": 128},
        ]
    )

    assert MixFusionDriver._chunk_policy(batch) == (128, 8)


def test_chunk_policy_fills_missing_extras_with_defaults():
    driver, _ = _driver()
    batch = SimpleNamespace(request_extras=[{"mixfusion_min_chunk_tokens": 32}, {}])

    assert MixFusionDriver._chunk_policy(batch) == (DEFAULT_MIN_CHUNK_TOKENS, DEFAULT_MAX_CHUNKS)


# ── armed lifecycle ──


def test_armed_publishes_the_step_and_restores_the_previous_one():
    driver, _ = _driver()
    previous = MixFusionStep(min_chunk_tokens=7, max_chunks=3)
    ctx = ForwardContext(mixfusion_step=previous)

    with override_forward_context(ctx):
        with driver.armed(min_chunk_tokens=64, max_chunks=9):
            step = get_forward_context().mixfusion_step
            assert step is not None
            assert (step.min_chunk_tokens, step.max_chunks) == (64, 9)
            assert step.adapter is driver.adapter
        assert ctx.mixfusion_step is previous


def test_armed_restores_the_previous_step_after_an_exception():
    driver, _ = _driver()
    previous = MixFusionStep(min_chunk_tokens=7, max_chunks=3)
    ctx = ForwardContext(mixfusion_step=previous)

    with override_forward_context(ctx):
        with pytest.raises(RuntimeError, match="boom"):
            with driver.armed(min_chunk_tokens=64, max_chunks=9):
                raise RuntimeError("boom")
        assert ctx.mixfusion_step is previous


# ── serial ragged path ──


def test_serial_ragged_builds_one_dense_batch_per_request():
    driver, transformer = _driver()
    first = torch.full((1, 8, 4), 1.0)
    second = torch.full((2, 8, 4), 2.0)
    states = [_state("req-0", first), _state("req-1", second)]
    input_batch = SimpleNamespace(latents=[first, second])
    seen: list = []

    def dense_step(batch, **kwargs):
        seen.append((batch, kwargs))
        return f"dense:{batch.request_ids[0]}"

    outputs = driver.serial_ragged(input_batch, states, dense_step)

    assert outputs == ["dense:req-0", "dense:req-1"]
    assert transformer.calls == []
    assert [batch.request_ids for batch, _ in seen] == [["req-0"], ["req-1"]]
    # A single ragged request still reports list latents; the driver flattens
    # it so the dense path receives the request's [num_images, S, C] tensor.
    assert isinstance(seen[0][0].latents, torch.Tensor)
    assert torch.equal(seen[0][0].latents, first)
    assert torch.equal(seen[1][0].latents, second)
    assert seen[0][0].num_reqs == 1
    assert seen[1][0].num_reqs == 1
    assert seen[0][1]["states"] is seen[0][0].states


def test_serial_ragged_reuses_the_input_batch_type():
    driver, _ = _driver()
    state = _state("req-0", torch.zeros(1, 8, 4))
    batches: list[InputBatch] = []

    driver.serial_ragged(
        SimpleNamespace(latents=[state.latents]),
        [state],
        lambda batch, **kwargs: batches.append(batch),
    )

    assert isinstance(batches[0], InputBatch)
    assert batches[0].states == (state,)
