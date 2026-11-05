# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Framework-level tests for the model-agnostic MixFusion driver (CPU).

Everything here runs against a fake transformer and a fake adapter with no
model imports, pinning the three responsibilities the driver owns:

- dispatch: dense batches go to ``dense_step`` unchanged, interrupted
  pipelines return ``None`` and ragged batches take the packed path only when
  admitted;
- admission: every rejection reason in ``MixFusionDriver.admit`` is enforced
  and the per-request chunk policy uses the strictest request's limits;
- lifecycle: ``armed`` publishes a step and restores the previous one, even
  on exceptions; ``serial_ragged`` rebuilds one dense batch per compatible
  group of requests.
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
from vllm_omni.diffusion.mixfusion.driver import (
    DEFAULT_MAX_CHUNKS,
    DEFAULT_MIN_CHUNK_TOKENS,
    MixFusionDriver,
)
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
    """Admission-only adapter: packing itself is the hooks' concern."""

    def __init__(self, *, refuse_reason: str | None = None) -> None:
        self._refuse_reason = refuse_reason

    def can_pack(self, module, input_batch) -> tuple[bool, str]:
        return self._refuse_reason is None, self._refuse_reason or ""


def _pipeline(
    *,
    interrupt: bool = False,
    attention_kwargs: dict | None = None,
    sp_size: int = 1,
    cfg_size: int = 1,
):
    return SimpleNamespace(
        interrupt=interrupt,
        attention_kwargs=attention_kwargs,
        parallel_config=SimpleNamespace(sequence_parallel_size=sp_size, cfg_parallel_size=cfg_size),
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


def _packed_dense_step(transformer: _FakeTransformer, calls: list | None = None):
    """Stand-in for the pipeline's dense step: forwards the packed list."""

    def dense_step(batch, **kwargs):
        if calls is not None:
            calls.append((batch, kwargs))
        return transformer(hidden_states=batch.latents)

    return dense_step


def _state(
    request_id: str,
    latents: torch.Tensor,
    *,
    do_true_cfg: bool = False,
    true_cfg_scale: float | None = None,
    cfg_normalize: bool = False,
    guidance: torch.Tensor | None = None,
    img_shapes: list | None = None,
    height: int | None = None,
) -> StepRequestState:
    """One ragged request state accepted by ``InputBatch.make_batch``."""
    return StepRequestState(
        request_id=request_id,
        sampling=SimpleNamespace(true_cfg_scale=true_cfg_scale, cfg_normalize=cfg_normalize, height=height),
        latents=latents,
        timesteps=torch.tensor([500.0]),
        do_true_cfg=do_true_cfg,
        guidance=guidance,
        img_shapes=img_shapes,
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
    dense_calls: list = []

    with override_forward_context(ctx):
        # The packed branch delegates to the pipeline's dense step inside the
        # armed context; the fake step forwards the list to the model.
        result = driver.run(batch, states=("s0", "s1"), dense_step=_packed_dense_step(transformer, dense_calls))

        assert result == PACKED_OUTPUT
        assert dense_calls == [(batch, {"states": ("s0", "s1")})]
        assert transformer.calls == [{"hidden_states": batch.latents}]
        assert isinstance(transformer.armed_at_call, MixFusionStep)
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


# ── admission ──


def _hooked_blocks() -> _FakeTransformer:
    transformer = _FakeTransformer()
    transformer.transformer_blocks[0].register_forward_pre_hook(lambda module, args: None)
    return transformer


def _hooked_root() -> _FakeTransformer:
    transformer = _FakeTransformer(plan={})
    transformer.register_forward_pre_hook(lambda module, args: None)
    return transformer


@pytest.mark.parametrize(
    ("driver_kwargs", "batch_kwargs", "admit_kwargs", "reason"),
    [
        ({}, {}, {}, "ok"),
        ({"pipeline": _pipeline(attention_kwargs={})}, {}, {}, "ok"),
        # Sequential true-CFG packs both branches in one armed step.
        ({}, {"do_true_cfg": True}, {}, "ok"),
        ({"adapter": None}, {}, {}, "_FakeTransformer declares no MixFusion adapter"),
        ({"transformer": lambda: _FakeTransformer(hooks_applied=False)}, {}, {}, "mixfusion hooks not applied"),
        ({"pipeline": _pipeline(attention_kwargs={"scale": 0.5})}, {}, {}, "attention_kwargs is set"),
        ({"pipeline": _pipeline(sp_size=2)}, {}, {}, "sequence parallelism is enabled"),
        ({"pipeline": _pipeline(cfg_size=2)}, {"do_true_cfg": True}, {}, "CFG parallel is enabled"),
        (
            {"transformer": _hooked_blocks},
            {},
            {},
            "foreign hooks on the packed path (transformer_blocks.*:torch-native)",
        ),
        ({"transformer": _hooked_root}, {}, {}, "foreign hooks on the packed path (<root>:torch-native)"),
        (
            {"adapter": _FakeAdapter(refuse_reason="shape not supported")},
            {},
            {},
            "adapter declined packing: shape not supported",
        ),
        ({}, {"lengths": (4,)}, {"min_chunk_tokens": 8}, "chunk_size=4 < min_chunk_tokens=8"),
        ({}, {}, {"max_chunks": 4}, "chunk_count=5 > max_chunks=4"),
    ],
)
def test_admission(driver_kwargs, batch_kwargs, admit_kwargs, reason):
    # Transformers are built per case: hooks registered on a shared module
    # would leak into the other cases.
    driver_kwargs = {k: v() if k == "transformer" else v for k, v in driver_kwargs.items()}
    driver, _ = _driver(**driver_kwargs)
    policy = {"min_chunk_tokens": 1, "max_chunks": DEFAULT_MAX_CHUNKS, **admit_kwargs}

    assert driver.admit(_ragged_batch(**batch_kwargs), **policy) == (reason == "ok", reason)


# ── chunk policy ──


@pytest.mark.parametrize(
    ("extras", "expected"),
    [
        (None, (DEFAULT_MIN_CHUNK_TOKENS, DEFAULT_MAX_CHUNKS)),
        (
            [{"mixfusion_min_chunk_tokens": 64, "mixfusion_max_chunks": 8}, {"mixfusion_min_chunk_tokens": 128}],
            (128, 8),
        ),
        # A request without extras contributes the defaults, which then win.
        ([{"mixfusion_min_chunk_tokens": 32}, {}], (DEFAULT_MIN_CHUNK_TOKENS, DEFAULT_MAX_CHUNKS)),
    ],
)
def test_chunk_policy_uses_the_strictest_request(extras, expected):
    assert MixFusionDriver._chunk_policy(SimpleNamespace(request_extras=extras)) == expected


# ── armed lifecycle ──


@pytest.mark.parametrize("raises", [False, True])
def test_armed_publishes_a_step_and_restores_the_previous_one(raises):
    driver, _ = _driver()
    previous = MixFusionStep()
    ctx = ForwardContext(mixfusion_step=previous)

    with override_forward_context(ctx):
        try:
            with driver.armed():
                step = get_forward_context().mixfusion_step
                assert isinstance(step, MixFusionStep) and step is not previous
                if raises:
                    raise RuntimeError("boom")
        except RuntimeError:
            assert raises
        assert ctx.mixfusion_step is previous


# ── serial ragged path ──


def test_serial_ragged_batches_requests_with_the_same_latent_shape():
    driver, transformer = _driver()
    first = torch.full((1, 8, 4), 1.0)
    second = torch.full((2, 8, 4), 2.0)
    states = [_state("req-0", first), _state("req-1", second)]
    seen: list = []

    def dense_step(batch, **kwargs):
        seen.append((batch, kwargs))
        return batch.latents + 100

    outputs = driver.serial_ragged(SimpleNamespace(latents=[first, second]), states, dense_step)

    assert transformer.calls == []
    # Different image counts share a batch: the batch dim is per-image, and
    # S/C match, so ``make_batch`` can express both requests together.
    assert len(seen) == 1
    batch, kwargs = seen[0]
    assert batch.request_ids == ["req-0", "req-1"]
    assert batch.num_reqs == 2
    assert isinstance(batch.latents, torch.Tensor)
    assert torch.equal(batch.latents, torch.cat([first, second], dim=0))
    assert kwargs["states"] is batch.states
    # Outputs are sliced per request, in request order.
    assert len(outputs) == 2
    assert torch.equal(outputs[0], first + 100)
    assert torch.equal(outputs[1], second + 100)


def test_serial_ragged_splits_groups_by_latent_shape():
    driver, _ = _driver()
    latents = [torch.full((1, length, 4), float(length)) for length in (8, 12, 8, 12)]
    states = [_state(f"req-{idx}", lat) for idx, lat in enumerate(latents)]
    seen: list[InputBatch] = []

    def dense_step(batch, **kwargs):
        seen.append(batch)
        return batch.latents + 1

    outputs = driver.serial_ragged(SimpleNamespace(latents=latents), states, dense_step)

    assert [batch.request_ids for batch in seen] == [["req-0", "req-2"], ["req-1", "req-3"]]
    for output, lat in zip(outputs, latents):
        assert torch.equal(output, lat + 1)


@pytest.mark.parametrize(
    "geometry",
    [
        # Equal area, different aspect ratio: same S, different RoPE grid.
        [{"img_shapes": [[(1, 2, 4)]]}, {"img_shapes": [[(1, 4, 2)]]}, {"img_shapes": [[(1, 2, 4)]]}],
        # A relaxed step-key field alone also separates requests.
        [{"height": 512}, {"height": 1024}, {"height": 512}],
    ],
)
def test_serial_ragged_splits_groups_by_image_geometry(geometry):
    driver, _ = _driver()
    latents = [torch.full((1, 8, 4), float(idx)) for idx in range(3)]
    states = [_state(f"req-{idx}", lat, **geometry[idx]) for idx, lat in enumerate(latents)]
    seen: list[InputBatch] = []

    def dense_step(batch, **kwargs):
        seen.append(batch)
        return batch.latents + 1

    outputs = driver.serial_ragged(SimpleNamespace(latents=latents), states, dense_step)

    assert [batch.request_ids for batch in seen] == [["req-0", "req-2"], ["req-1"]]
    for output, lat in zip(outputs, latents):
        assert torch.equal(output, lat + 1)


def test_serial_ragged_splits_groups_by_cfg_and_guidance_settings():
    # ``InputBatch`` raises on mixed CFG or guidance settings, so requests that
    # differ in either must land in different dense batches.
    driver, _ = _driver()
    latents = [torch.full((1, 8, 4), float(idx)) for idx in range(3)]
    states = [
        _state("req-0", latents[0], do_true_cfg=True),
        _state("req-1", latents[1]),
        _state("req-2", latents[2], guidance=torch.zeros(())),
    ]
    seen: list[InputBatch] = []

    def dense_step(batch, **kwargs):
        seen.append(batch)
        return batch.latents + 1

    outputs = driver.serial_ragged(SimpleNamespace(latents=latents), states, dense_step)

    assert [batch.request_ids for batch in seen] == [["req-0"], ["req-1"], ["req-2"]]
    for output, lat in zip(outputs, latents):
        assert torch.equal(output, lat + 1)


def test_serial_ragged_maps_list_outputs_to_requests():
    driver, _ = _driver()
    latents = [torch.zeros(1, 8, 4), torch.ones(1, 8, 4)]
    states = [_state("req-0", latents[0]), _state("req-1", latents[1])]
    per_request = [torch.full_like(latents[0], 7.0), torch.full_like(latents[1], 9.0)]

    outputs = driver.serial_ragged(
        SimpleNamespace(latents=latents),
        states,
        lambda batch, **kwargs: per_request,
    )

    assert outputs == per_request
