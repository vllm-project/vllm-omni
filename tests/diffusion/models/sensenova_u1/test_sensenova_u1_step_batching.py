# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Step-wise batching tests for the SenseNova-U1.5 pipeline.

A step-mode wave carries several in-flight requests through one
``denoise_step`` call. The pipeline forwards each request on its own state
(flash KV caches, geometry, CFG branches) and concatenates the velocity rows,
so the runner's per-request row slices keep every request evolving exactly as
it would alone. These tests pin that multi-request contract:

- a batched pair reproduces each request's solo denoise call sequence and
  final pixels bit-for-bit;
- heterogeneous waves stay safe: mixed t2i/it2i, unequal step counts, and a
  mid-flight admission that must not disturb the request already running;
- driving two requests through the real runner batches them into shared waves.

The driver below mirrors ``DiffusionModelRunner._execute_stepwise_core``: one
``denoise_step`` per wave over the still-active requests, then a
``step_scheduler`` update per request on its own ``latents.shape[0]`` row
slice, with ``post_decode`` firing per request as soon as its own schedule is
exhausted.
"""

import types

import numpy as np
import pytest
import torch

import vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 as pipe_mod
from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import SenseNovaU1Pipeline
from vllm_omni.diffusion.worker.utils import StepRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

PATCH = 2
MERGE = 2
H = W = 16
GRID = H // PATCH
DIM = 16
BATCH = 2
STEPS = 3


class _StubEmbedder:
    """Deterministic stand-in for timestep/noise-scale embedders: [N] -> [N, DIM]."""

    def __init__(self, dim: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        self.w = torch.randn(dim, generator=g)
        self.b = torch.randn(dim, generator=g)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.w + self.b


class _Recorder:
    """Records every model call and answers deterministically.

    Both stubs are pure functions of their inputs: a request must see the
    same response for the same image state no matter how waves interleave it
    with peers, which is exactly what the solo-vs-batched comparison pins.
    (Random replay would couple calls through one generator and cannot be
    aligned across different wave interleavings.)
    """

    def __init__(self):
        self.extract_calls: list[torch.Tensor] = []
        self.denoise_calls: list[dict] = []
        self.cleaned: list[object] = []

    def extract_feature(self, image_input, gen_model, grid_hw):
        self.extract_calls.append(image_input.clone())
        pooled = image_input.float().mean(dim=-1, keepdim=True)
        embeds = pooled.expand(*image_input.shape[:-1], DIM)
        return (embeds * 7.0).to(image_input.dtype)

    def denoise(self, image_prediction, ns, t, z, image_embeds, caches, p, step_i, is_it2i):
        self.denoise_calls.append(
            {
                "image_prediction": image_prediction.clone(),
                "t": t.clone(),
                "z": z.clone(),
                "image_embeds": image_embeds.clone(),
                "step_i": step_i,
                "is_it2i": is_it2i,
            }
        )
        return torch.tanh(z) * (1.0 - t) + 0.25


def _make_ns(steps: int = STEPS, seed: int = 1234):
    """Build one request's denoising namespace the way ``_init_noise_and_schedule`` does."""
    g = torch.Generator().manual_seed(seed)
    return types.SimpleNamespace(
        image_prediction=torch.randn(BATCH, 3, H, W, generator=g),
        timesteps=1.0 - torch.arange(steps + 1, dtype=torch.float32) / steps,
        grid_h=GRID,
        grid_w=GRID,
        grid_hw=torch.tensor([[GRID, GRID]]),
        token_h=GRID,
        token_w=GRID,
        noise_scale=0.5,
    )


def _make_caches():
    return {
        "cond": object(),
        "uncond": object(),
        "img_cond": {"nested": True},  # dict entries must be skipped by cleanup
    }


def _make_setup():
    pipe = object.__new__(SenseNovaU1Pipeline)
    pipe.patch_size = PATCH
    pipe.merge_size = MERGE
    pipe.fm_modules = {
        "timestep_embedder": _StubEmbedder(DIM, 1),
        "noise_scale_embedder": _StubEmbedder(DIM, 2),
    }
    pipe.model_cfg = types.SimpleNamespace(
        add_noise_scale_embedding=True,
        noise_scale_max_value=2.0,
    )
    return pipe


def _install_recorder(pipe):
    recorder = _Recorder()
    pipe._extract_feature = recorder.extract_feature
    pipe._denoise = recorder.denoise
    return recorder


def _track_cleanup(monkeypatch, recorder):
    monkeypatch.setattr(pipe_mod, "clear_flash_kv_cache", lambda cache: recorder.cleaned.append(cache))


def _build_step_request(ns, caches, request_id="req-1", think_text="", is_it2i=False, num_steps=STEPS):
    """Assemble the StepRequestState exactly as prepare_encode lays it out."""
    req = StepRequestState(request_id=request_id, sampling=types.SimpleNamespace(), prompt="a cat")
    p = types.SimpleNamespace(batch_size=BATCH, num_steps=num_steps, image_size=[H, W])
    step = types.SimpleNamespace(
        mode="it2i" if is_it2i else "t2i",
        p=p,
        ns=ns,
        prefix=None,
        caches=caches,
        think_text=think_text,
        output=None,
        z=None,
        cursor=None,
        queued=False,
        input_images=None,
    )
    req.latents = ns.image_prediction
    req.timesteps = ns.timesteps[:-1]
    req.step_index = 0
    req.extra[pipe_mod.SenseNovaU1Pipeline._STEP_KEY] = step
    return req


def _drive_wave(pipe, reqs):
    """Advance one scheduler wave the way the runner does.

    A single ``denoise_step`` covers every still-active request; each request
    then consumes its own ``latents.shape[0]`` row slice of the concatenated
    velocity, and the slices must cover it exactly.
    """
    v_pred = pipe.denoise_step(None, states=list(reqs))
    offset = 0
    for req in reqs:
        rows = req.latents.shape[0]
        pipe.step_scheduler(req, v_pred[offset : offset + rows])
        offset += rows
    assert offset == v_pred.shape[0]


def _run_to_completion(pipe, reqs):
    """Drive requests to completion, returning outputs and post_decode order.

    Mirrors the runner loop: each wave covers the still-active requests only,
    and a request is decoded as soon as its own schedule is exhausted, while
    its peers keep stepping.
    """
    outputs = {}
    decode_order = []
    while True:
        active = [req for req in reqs if not req.denoise_completed]
        if not active:
            break
        _drive_wave(pipe, active)
        for req in active:
            if req.denoise_completed:
                outputs[req.request_id] = pipe.post_decode(req)
                decode_order.append(req.request_id)
    return outputs, decode_order


def _assert_matching_calls(calls, expected):
    assert len(calls) == len(expected)
    for got, want in zip(calls, expected):
        assert got["step_i"] == want["step_i"]
        assert got["is_it2i"] == want["is_it2i"]
        assert torch.equal(got["image_prediction"], want["image_prediction"])
        assert torch.equal(got["t"], want["t"])
        assert torch.equal(got["z"], want["z"])
        assert torch.equal(got["image_embeds"], want["image_embeds"])


def _solo_run(monkeypatch, seed, request_id):
    """Run one request alone on its own pipeline; return calls, output, caches."""
    pipe = _make_setup()
    recorder = _install_recorder(pipe)
    _track_cleanup(monkeypatch, recorder)
    ns, caches = _make_ns(seed=seed), _make_caches()
    req = _build_step_request(ns, caches, request_id=request_id)
    outputs, _ = _run_to_completion(pipe, [req])
    return recorder.denoise_calls, outputs[request_id], caches


def test_pipeline_declares_step_execution():
    from vllm_omni.diffusion.models.interface import SupportsStepExecution

    assert SenseNovaU1Pipeline.supports_step_execution is True
    assert isinstance(SenseNovaU1Pipeline, SupportsStepExecution)


@pytest.mark.parametrize("think_text,is_it2i", [("", False), ("reasoning...", False), ("it2i", True)])
def test_step_execution_matches_full_request_path(think_text, is_it2i, monkeypatch):
    """Driving the four step methods must reproduce the request path bit-for-bit."""
    pipe_full = _make_setup()
    rec_full = _install_recorder(pipe_full)
    _track_cleanup(monkeypatch, rec_full)
    ns_full, caches_full = _make_ns(), _make_caches()
    full_out = pipe_full._run_denoising_loop(ns_full, caches_full, _ns_params(), think_text, is_it2i)

    pipe_step = _make_setup()
    rec_step = _install_recorder(pipe_step)
    _track_cleanup(monkeypatch, rec_step)
    ns_step, caches_step = _make_ns(), _make_caches()
    req = _build_step_request(ns_step, caches_step, think_text=think_text, is_it2i=is_it2i)
    # Cache references must be captured before post_decode clears the dicts.
    cond_step, uncond_step = caches_step["cond"], caches_step["uncond"]
    input_batch = types.SimpleNamespace(states=(req,))

    while not req.denoise_completed:
        v_pred = pipe_step.denoise_step(input_batch)
        pipe_step.step_scheduler(req, v_pred)
    step_out = pipe_step.post_decode(req)

    # Every denoise forward saw bit-identical inputs, in the same order.
    assert len(rec_step.denoise_calls) == len(rec_full.denoise_calls) == STEPS
    for step_call, full_call in zip(rec_step.denoise_calls, rec_full.denoise_calls):
        assert step_call["step_i"] == full_call["step_i"]
        assert step_call["is_it2i"] == full_call["is_it2i"]
        assert torch.equal(step_call["t"], full_call["t"])
        assert torch.equal(step_call["image_prediction"], full_call["image_prediction"])
        assert torch.equal(step_call["z"], full_call["z"])
        assert torch.equal(step_call["image_embeds"], full_call["image_embeds"])

    # Feature extraction saw bit-identical inputs, in the same order.
    assert len(rec_step.extract_calls) == len(rec_full.extract_calls) == STEPS
    for step_in, full_in in zip(rec_step.extract_calls, rec_full.extract_calls):
        assert torch.equal(step_in, full_in)

    # Final pixels are bit-identical.
    step_img = step_out.output["payload"]["image"]
    full_img = full_out.output["payload"]["image"]
    assert np.array_equal(np.asarray(step_img), np.asarray(full_img))

    # think_text reaches the metadata unchanged.
    metadata = step_out.output["metadata"]
    if think_text:
        assert metadata["text"] == {"think_text": think_text}
    else:
        assert metadata == {}

    # The request retired after exactly num_steps scheduler updates.
    assert req.step_index == STEPS
    assert req.denoise_completed

    # post_decode released exactly the non-dict cache entries, like the loop.
    assert rec_step.cleaned == [cond_step, uncond_step]


def _ns_params():
    return types.SimpleNamespace(batch_size=BATCH, num_steps=STEPS, image_size=[H, W])


def test_step_scheduler_update_matches_reference_euler_step():
    pipe = _make_setup()
    ns, caches = _make_ns(), _make_caches()
    req = _build_step_request(ns, caches)
    tok = (H // (PATCH * MERGE)) * (W // (PATCH * MERGE))
    ch = 3 * (PATCH * MERGE) ** 2
    v_pred = torch.full((BATCH, tok, ch), 0.5)

    latents_before = req.latents.clone()
    t, t_next = ns.timesteps[0], ns.timesteps[1]
    expected_z = pipe_mod._patchify(latents_before, PATCH * MERGE) + (t_next - t) * v_pred
    expected = pipe_mod._unpatchify(expected_z, PATCH * MERGE, H, W)

    # denoise_step parks the patchified latents on the step context; the
    # scheduler update consumes that copy rather than re-patchifying.
    step = req.extra[SenseNovaU1Pipeline._STEP_KEY]
    step.z = pipe_mod._patchify(latents_before, PATCH * MERGE)
    pipe.step_scheduler(req, v_pred)

    assert req.step_index == 1
    assert torch.equal(req.latents, expected)
    # The original noise tensor is never mutated in place.
    assert req.latents is not latents_before
    assert torch.equal(ns.image_prediction, latents_before)


def test_denoise_step_concatenates_and_slices_per_request():
    """Two requests at different steps: rows concatenate, and runner-style
    per-request slices drive each scheduler update with its own t."""
    pipe = _make_setup()
    recorder = _install_recorder(pipe)

    req_a = _build_step_request(_make_ns(seed=1234), _make_caches(), request_id="req-a", is_it2i=False)
    req_b = _build_step_request(_make_ns(seed=5678), _make_caches(), request_id="req-b", is_it2i=True)
    req_b.step_index = 1  # mid-flight request joins request a at step 0

    v_pred = pipe.denoise_step(None, states=[req_a, req_b])

    total_rows = req_a.latents.shape[0] + req_b.latents.shape[0]
    assert v_pred.shape[0] == total_rows
    assert [call["step_i"] for call in recorder.denoise_calls] == [0, 1]
    assert [call["is_it2i"] for call in recorder.denoise_calls] == [False, True]

    # Runner contract: slice noise_pred by each request's latents rows.
    offset = 0
    for req in (req_a, req_b):
        rows = req.latents.shape[0]
        pipe.step_scheduler(req, v_pred[offset : offset + rows].clone())
        offset += rows
    assert offset == v_pred.shape[0]
    assert req_a.step_index == 1
    assert req_b.step_index == 2


def test_step_batch_matches_sequential_execution(monkeypatch):
    """Batching two requests into shared waves must change nothing per request.

    Every denoise forward, and the final pixels, must match a run where each
    request drove the step methods alone. The batch recorder sees the wave
    order (req-a then req-b per wave), so its call log is the interleaving of
    the two solo logs.
    """
    solo_a_calls, solo_a_out, _ = _solo_run(monkeypatch, seed=1234, request_id="req-a")
    solo_b_calls, solo_b_out, _ = _solo_run(monkeypatch, seed=5678, request_id="req-b")

    pipe = _make_setup()
    recorder = _install_recorder(pipe)
    _track_cleanup(monkeypatch, recorder)
    caches_a, caches_b = _make_caches(), _make_caches()
    # Cache references must be captured before any release clears the dicts.
    cond_a, uncond_a = caches_a["cond"], caches_a["uncond"]
    cond_b, uncond_b = caches_b["cond"], caches_b["uncond"]
    req_a = _build_step_request(_make_ns(seed=1234), caches_a, request_id="req-a")
    req_b = _build_step_request(_make_ns(seed=5678), caches_b, request_id="req-b")

    outputs, decode_order = _run_to_completion(pipe, [req_a, req_b])

    assert decode_order == ["req-a", "req-b"]
    interleaved = [call for pair in zip(solo_a_calls, solo_b_calls) for call in pair]
    _assert_matching_calls(recorder.denoise_calls, interleaved)

    for request_id, solo_out in (("req-a", solo_a_out), ("req-b", solo_b_out)):
        solo_img = solo_out.output["payload"]["image"]
        batch_img = outputs[request_id].output["payload"]["image"]
        assert np.array_equal(np.asarray(batch_img), np.asarray(solo_img))

    # Each request's caches were released exactly once, in decode order.
    assert recorder.cleaned == [cond_a, uncond_a, cond_b, uncond_b]


def test_step_batch_mixed_t2i_and_it2i_requests(monkeypatch):
    """A wave may mix t2i and it2i requests: each forward uses its own branch
    layout and CFG structure, and the concatenated rows stay per-request."""
    pipe = _make_setup()
    recorder = _install_recorder(pipe)
    _track_cleanup(monkeypatch, recorder)
    caches_t2i, caches_it2i = _make_caches(), _make_caches()
    cond_t2i = caches_t2i["cond"]
    cond_it2i = caches_it2i["cond"]
    req_t2i = _build_step_request(_make_ns(), caches_t2i, request_id="req-t2i", is_it2i=False)
    req_it2i = _build_step_request(_make_ns(seed=5678), caches_it2i, request_id="req-it2i", is_it2i=True)

    outputs, decode_order = _run_to_completion(pipe, [req_t2i, req_it2i])

    assert decode_order == ["req-t2i", "req-it2i"]
    # One forward per request per wave, in states order.
    assert [call["is_it2i"] for call in recorder.denoise_calls] == [False, True] * STEPS
    assert req_t2i.step_index == req_it2i.step_index == STEPS
    assert outputs["req-t2i"].output["payload"]["image"] is not None
    assert outputs["req-it2i"].output["payload"]["image"] is not None
    # Both requests released their non-dict caches (img_cond dict is skipped).
    assert recorder.cleaned.count(cond_t2i) == 1
    assert recorder.cleaned.count(cond_it2i) == 1


def test_step_batch_unequal_step_counts(monkeypatch):
    """A shorter request retires mid-wave while its longer peer keeps going."""
    pipe = _make_setup()
    recorder = _install_recorder(pipe)
    _track_cleanup(monkeypatch, recorder)
    req_long = _build_step_request(_make_ns(steps=3, seed=1234), _make_caches(), request_id="req-long", num_steps=3)
    req_short = _build_step_request(_make_ns(steps=2, seed=5678), _make_caches(), request_id="req-short", num_steps=2)

    outputs, decode_order = _run_to_completion(pipe, [req_long, req_short])

    # The 2-step request finishes first; the final wave carries req-long alone.
    assert decode_order == ["req-short", "req-long"]
    assert [call["step_i"] for call in recorder.denoise_calls] == [0, 0, 1, 1, 2]
    assert req_short.step_index == 2
    assert req_long.step_index == 3
    assert outputs["req-short"].output["payload"]["image"] is not None
    assert outputs["req-long"].output["payload"]["image"] is not None


def _admit_request(pipe, request_id, seed=5678):
    """Admit a new image request the way the runner does: through the real
    ``prepare_encode`` with the prefix stubbed out (think off)."""
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    pipe._extract_input_images = lambda prompt: None
    pipe._init_noise_and_schedule = lambda p: _make_ns(seed=seed)
    pipe._t2i_prefix = lambda p, ns: types.SimpleNamespace(cursor=None, past_kv_cond="kv")
    pipe._t2i_caches = lambda p, ns, ctx: (_make_caches(), "")
    req = StepRequestState(
        request_id=request_id,
        sampling=OmniDiffusionSamplingParams(height=H, width=W, num_inference_steps=STEPS, seed=9),
        prompt="a boat",
    )
    returned = pipe.prepare_encode(req)
    assert returned is req
    return req


def test_mid_flight_admission_continues_older_request(monkeypatch):
    """A request admitted mid-denoise joins the wave without disturbing it.

    req-a steps alone first; req-b is then prepared the way the runner
    prepares a newly admitted request (prepare_encode on a fresh state) and
    joins from its own step 0. req-a's forwards must stay bit-identical to a
    solo run -- including the step it already took.
    """
    solo_calls, solo_out, _ = _solo_run(monkeypatch, seed=1234, request_id="req-a")

    pipe = _make_setup()
    recorder = _install_recorder(pipe)
    _track_cleanup(monkeypatch, recorder)
    caches_a = _make_caches()
    req_a = _build_step_request(_make_ns(seed=1234), caches_a, request_id="req-a")
    _drive_wave(pipe, [req_a])
    assert req_a.step_index == 1

    # Admission: the runner builds the new state and calls prepare_encode,
    # which parks the request-local denoising state on it.
    req_b = _admit_request(pipe, "req-b")

    outputs, decode_order = _run_to_completion(pipe, [req_a, req_b])

    assert decode_order == ["req-a", "req-b"]
    # Wave log: [a0] then [a1, b0], [a2, b1], [b2]. req-a owns indices 0, 1, 3.
    _assert_matching_calls(
        [recorder.denoise_calls[i] for i in (0, 1, 3)],
        solo_calls,
    )
    solo_img = solo_out.output["payload"]["image"]
    assert np.array_equal(np.asarray(outputs["req-a"].output["payload"]["image"]), np.asarray(solo_img))
    # The admitted request completed its own schedule.
    assert req_b.step_index == STEPS
    assert outputs["req-b"].output["payload"]["image"] is not None


def test_step_batch_noise_is_per_request():
    """Denoising noise is seeded per request, so concurrent requests stay
    independent: same seed reproduces, different seeds differ."""
    assert torch.equal(_make_ns(seed=1234).image_prediction, _make_ns(seed=1234).image_prediction)
    assert not torch.equal(_make_ns(seed=1234).image_prediction, _make_ns(seed=5678).image_prediction)


# ---------------------------------------------------------------------------
# Through the real runner
# ---------------------------------------------------------------------------


def _noop_forward_context(*args, **kwargs):
    del args, kwargs
    from contextlib import nullcontext

    return nullcontext()


class TestThroughTheRunner:
    """Two requests driven by the runner that will call the pipeline."""

    @staticmethod
    def _runner(pipeline):
        import torch as _torch

        from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
        from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner

        runner = object.__new__(DiffusionModelRunner)
        runner.vllm_config = types.SimpleNamespace(
            kernel_config=types.SimpleNamespace(ir_op_priority=types.SimpleNamespace(set_priority=lambda: None)),
            compilation_config=types.SimpleNamespace(ir_enable_torch_wrap=True),
        )
        runner.od_config = types.SimpleNamespace(
            cache_backend=None,
            diffusion_kv_mode=DiffusionKVCacheMode.DENSE_LEGACY,
            parallel_config=types.SimpleNamespace(use_hsdp=False),
            streaming_output=False,
            step_execution=True,
            max_num_seqs=2,
        )
        runner.device = _torch.device("cpu")
        runner.pipeline = pipeline
        runner.cache_backend = None
        runner.offload_backend = None
        runner.state_cache = {}
        runner.input_batch = None
        runner.kv_transfer_manager = types.SimpleNamespace(
            receive_multi_kv_cache_distributed=lambda req, cfg_kv_collect_func=None, target_device=None: None
        )
        return runner

    def test_two_requests_share_one_denoise_wave(self, monkeypatch: pytest.MonkeyPatch):
        """Two scheduled requests reach one ``denoise_step`` call per tick.

        Only the transformer forward (``_denoise_one``) is stubbed; the batch
        assembly, the row slicing, and the scheduler updates are the
        production ones, so this pins the whole step-wise batching contract
        end to end: both requests coexist in one step batch (max_num_seqs=2),
        advance together, and retire independently.
        """
        import vllm_omni.diffusion.worker.diffusion_model_runner as runner_module
        from vllm_omni.diffusion.request import OmniDiffusionRequest
        from vllm_omni.diffusion.sched.interface import (
            CachedRequestData,
            DiffusionSchedulerOutput,
            NewRequestData,
        )
        from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        monkeypatch.setattr(runner_module, "set_forward_context", _noop_forward_context)
        pipe = _make_setup()
        # 32x32 at patch 2 x merge 2 is an 8x8 patch grid; latents carry one
        # row per request in the [N, tokens, channels] patchified layout.
        pipe.patch_size, pipe.merge_size = 2, 2
        schedule = types.SimpleNamespace(
            image_prediction=torch.zeros(1, 3, 32, 32),
            timesteps=torch.tensor([0.0, 0.25, 0.75, 1.0]),
        )
        pipe._init_noise_and_schedule = lambda p: schedule
        pipe._t2i_prefix = lambda p, ns: types.SimpleNamespace(cursor=None, past_kv_cond="kv")
        pipe._t2i_caches = lambda p, ns, ctx: ({"cond": {}}, "")
        seen_steps: list[int] = []
        v_pred = torch.full((1, 64, 3 * 16), 2.0)

        def _denoise_one(image_prediction, ns, caches, p, step_i, is_edit):
            seen_steps.append(step_i)
            return torch.zeros_like(v_pred), v_pred

        pipe._denoise_one = _denoise_one
        wave_sizes: list[int] = []
        real_denoise_step = SenseNovaU1Pipeline.denoise_step

        def _spy_denoise_step(input_batch, *, states=None, **kwargs):
            wave_sizes.append(len(tuple(states if states is not None else input_batch.states)))
            return real_denoise_step(pipe, input_batch, states=states, **kwargs)

        pipe.denoise_step = _spy_denoise_step

        runner = self._runner(pipe)

        def _request(request_id: str) -> OmniDiffusionRequest:
            sampling = OmniDiffusionSamplingParams(num_inference_steps=3, seed=42, width=32, height=32)
            return OmniDiffusionRequest(
                prompt={"prompt": "draw the sky", "modalities": ["image"]},
                request_id=request_id,
                sampling_params=sampling,
            )

        scheduled = DiffusionSchedulerOutput(
            step_id=0,
            scheduled_new_reqs=[NewRequestData(request_id="req-a", req=_request("req-a"))]
            + [NewRequestData(request_id="req-b", req=_request("req-b"))],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            finished_req_ids=set(),
            num_running_reqs=2,
            num_waiting_reqs=0,
        )
        output = DiffusionModelRunner.execute_stepwise(runner, scheduled)

        for step_id in range(1, 6):
            live = [rid for rid in ("req-a", "req-b") if rid in runner.state_cache]
            outputs = [output.get_request_output(rid) for rid in ("req-a", "req-b")]
            if all(o is not None and o.finished for o in outputs):
                break
            cached = DiffusionSchedulerOutput(
                step_id=step_id,
                scheduled_new_reqs=[],
                scheduled_cached_reqs=CachedRequestData(request_ids=live),
                finished_req_ids=set(),
                num_running_reqs=len(live),
                num_waiting_reqs=0,
            )
            output = DiffusionModelRunner.execute_stepwise(runner, cached)

        # Every tick that denoised at all carried both requests in one call.
        assert wave_sizes == [2, 2, 2]
        # One forward per request per wave, at ascending step indices.
        assert seen_steps == [0, 0, 1, 1, 2, 2]
        for rid in ("req-a", "req-b"):
            request_output = output.get_request_output(rid)
            assert request_output is not None and request_output.finished is True
            assert request_output.result.output["payload"]["image"] is not None
            assert rid not in runner.state_cache
