# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CausVid Wan unit checks (layer split + FlowEuler + WanCell)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_executor import (
    ARDiffusionCellContext,
    CellRunSpec,
    CellTopology,
)
from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_plan import Ordering
from vllm_omni.diffusion.models.causvid_wan.pipeline_causvid_wan import (
    CAUSVID_DEFAULT_DENOISE_STEPS,
    CAUSVID_DEFAULT_SHIFT,
    CausVidWanPipeline,
)
from vllm_omni.diffusion.models.causvid_wan.transformer import StageWanTransformer, stage_layer_range
from vllm_omni.diffusion.models.causvid_wan.wan_cell import WanCell
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.experimental.ar_diffusion.flow_euler import FlowEuler
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import ARDiffusionNoisyKVSpec, NoisyKVCache, NoisyKVState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _bare_pipeline(*, step_replicas: int = 1, history: int = 1, layer_groups: int = 1) -> CausVidWanPipeline:
    """``__new__`` stub with the fields ``_plan_for`` reads."""
    pipeline = CausVidWanPipeline.__new__(CausVidWanPipeline)
    pipeline.step_replicas = step_replicas
    pipeline.layer_groups = layer_groups
    pipeline.max_history_chunks = history
    pipeline.transformer = SimpleNamespace(num_layers=1, local_num_layers=1)
    pipeline._default_shift = CAUSVID_DEFAULT_SHIFT
    pipeline._default_denoise = CAUSVID_DEFAULT_DENOISE_STEPS
    return pipeline


def test_stage_layer_range_covers_all_layers():
    from vllm.distributed.utils import get_pp_indices

    covered: list[int] = []
    for g in range(2):
        start, end = stage_layer_range(30, g, 2)
        covered.extend(range(start, end))
    assert covered == list(range(30))
    assert stage_layer_range(30, 0, 2) == (0, 15)
    assert stage_layer_range(30, 1, 2) == (15, 30)
    # G=4 must match Wan/vLLM get_pp_indices (not the floor split).
    for g in range(4):
        assert stage_layer_range(30, g, 4) == get_pp_indices(30, g, 4)
    assert stage_layer_range(30, 3, 4) == (23, 30)


def test_flow_euler_matches_diffusers_shifted_endpoint():
    """Match FlowMatchEulerDiscreteScheduler: linspace ends at warp(1/1000), then warp again."""
    shift = 5.0
    steps = 4

    def warp(sigma: torch.Tensor) -> torch.Tensor:
        return shift * sigma / (1 + (shift - 1) * sigma)

    sampler = FlowEuler(steps, shift=shift)
    expected = warp(torch.linspace(1.0, float(warp(torch.tensor(1.0 / 1000))), steps, dtype=torch.float64))
    assert sampler.sigmas == [*expected.float().tolist(), 0.0]
    # Diffusers' own endpoint double-warp yields a larger last step than warp(1/1000).
    last_delta = abs(sampler.sigmas[-1] - sampler.sigmas[-2])
    assert last_delta > 0.01


def test_causvid_wan_chunk_noise_matches_reference_seed():
    shape = (1, 16, 1, 2, 2)
    seed, chunk = 7, 2
    adapter = WanCell(
        SimpleNamespace(prepare_cell=lambda *_a, **_k: None),  # type: ignore[arg-type]
        sampler=FlowEuler(3, shift=5.0),
        prompt_embeds=torch.empty(1, 1, 32),
        latent_shape=shape,
        seed=seed,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    reference = torch.randn(shape, generator=torch.Generator().manual_seed(seed * 1_000_003 + chunk * 4096))
    torch.testing.assert_close(adapter._init_noise(chunk), reference)
    torch.testing.assert_close(adapter._init_noise(chunk), reference)


def test_resolve_member_input_reuses_live_tensor_without_bool_check():
    """Executor may clear pending input; live latent must use ``is None``, not ``or``."""
    shape = (1, 4, 2, 2, 2)
    adapter = WanCell(
        SimpleNamespace(prepare_cell=lambda *_a, **_k: None),  # type: ignore[arg-type]
        sampler=FlowEuler(2, shift=1.0),
        prompt_embeds=torch.empty(1, 1, 8),
        latent_shape=shape,
        seed=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    live = {0: torch.ones(shape)}
    latent, tokens = adapter._resolve_member_input(req="r0", chunk=0, member_input=None, live=live)
    assert tokens is None
    assert latent is live[0]
    torch.testing.assert_close(latent, torch.ones(shape))


def test_prompt_text_extracts_omni_text_prompt_dict():
    assert CausVidWanPipeline._prompt_text("hello") == "hello"
    assert CausVidWanPipeline._prompt_text({"prompt": "from dict", "negative_prompt": "x"}) == "from dict"
    with pytest.raises(ValueError, match="prompt"):
        CausVidWanPipeline._prompt_text({"negative_prompt": "x"})
    with pytest.raises(TypeError, match="Unsupported prompt"):
        CausVidWanPipeline._prompt_text(123)  # type: ignore[arg-type]


def test_rotary_emb_applies_chunk_temporal_offset():
    """Chunk k must use absolute temporal positions ``k * T``, not restart at 0."""
    head_dim = 24  # divisible for t/h/w split used by WanRotaryPosEmbed
    max_seq = 16
    # Fake freqs: identity encoding of position index on even dims.
    freqs = torch.arange(max_seq, dtype=torch.float32).unsqueeze(1).expand(max_seq, head_dim).contiguous()
    rope = SimpleNamespace(
        patch_size=(1, 1, 1),
        max_seq_len=max_seq,
        attention_head_dim=head_dim,
        freqs_cos=freqs,
        freqs_sin=freqs,
    )
    tf = StageWanTransformer.__new__(StageWanTransformer)
    torch.nn.Module.__init__(tf)
    tf._wan = SimpleNamespace(rope=rope)
    tf._rotary = {}
    latent = torch.zeros(1, 4, 3, 2, 2)  # T=3
    cos0, _ = tf._rotary_emb(0, latent)
    cos1, _ = tf._rotary_emb(1, latent)
    assert cos0.shape == cos1.shape
    assert not torch.equal(cos0, cos1)
    swapped, _ = tf._rotary_emb(1, torch.zeros(1, 4, 3, 4, 1))
    assert not torch.equal(swapped, cos1)
    assert len(tf._rotary) == 1
    # After even-dim slice (Wan rotary packs cos[..., 0::2]), temporal axis still differs.
    assert cos1.mean().item() > cos0.mean().item()


def test_cell_schedule_rejects_unknown_values():
    pipeline = _bare_pipeline()
    req = OmniDiffusionRequest(
        prompt="x",
        request_id="sched-0",
        sampling_params=OmniDiffusionSamplingParams(extra_args={"cell_schedule": "unsupported"}),
    )
    with pytest.raises(ValueError, match="cell_schedule"):
        pipeline._plan_for(req)


@pytest.mark.parametrize(("name", "step_replicas"), [("self_forcing", 1), ("latest_kv", 4)])
def test_cell_schedule_accepts_known_values(name, step_replicas):
    from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_plan import Ordering

    pipeline = _bare_pipeline(step_replicas=step_replicas, history=6)
    req = OmniDiffusionRequest(
        prompt="x",
        request_id=f"sched-{name}",
        sampling_params=OmniDiffusionSamplingParams(
            extra_args={"cell_schedule": name, "num_denoise_steps": 3, "num_chunks": 2}
        ),
    )
    plan, _, _ = pipeline._plan_for(req)
    assert plan.schedule.ordering is Ordering(name)
    assert plan.schedule.ordering.value == name


@pytest.mark.parametrize("name", ["serial", "latest", "interleaved", "vertical", "same_step", "same_step_kv"])
def test_cell_schedule_rejects_removed_aliases(name):
    pipeline = _bare_pipeline()
    req = OmniDiffusionRequest(
        prompt="x",
        request_id=f"sched-alias-{name}",
        sampling_params=OmniDiffusionSamplingParams(extra_args={"cell_schedule": name}),
    )
    with pytest.raises(ValueError, match="cell_schedule|same_step is not supported"):
        pipeline._plan_for(req)


@pytest.mark.parametrize("source", ["num_inference_steps", "num_denoise_steps", "sampling_params"])
@pytest.mark.parametrize("steps", [0, -1])
def test_cell_schedule_rejects_nonpositive_steps(source, steps):
    pipeline = _bare_pipeline()
    extra = {} if source == "sampling_params" else {source: steps}
    req = OmniDiffusionRequest(
        prompt="x",
        request_id="invalid-steps",
        sampling_params=OmniDiffusionSamplingParams(
            num_inference_steps=steps if source == "sampling_params" else 4, extra_args=extra
        ),
    )
    with pytest.raises(ValueError, match="num_denoise_steps must be positive"):
        pipeline._plan_for(req)


@pytest.mark.parametrize(
    ("stages", "schedule", "expected_steps"),
    [
        (1, "self_forcing", CAUSVID_DEFAULT_DENOISE_STEPS),  # S=1 keeps model default
        (3, "latest_kv", 2),  # S≠1 and S≠D+1 (default D=3) → derive D=S-1 → S=D+1
        (4, "latest_kv", 3),  # S=D+1 with D=default
        (5, "latest_kv", 4),  # S≠1 and S≠D+1 → derive D=S-1
    ],
)
def test_cell_schedule_unset_steps_keep_topology_default(stages, schedule, expected_steps):
    pipeline = _bare_pipeline(step_replicas=stages, history=6)
    req = OmniDiffusionRequest(
        prompt="x",
        request_id="default-steps",
        sampling_params=OmniDiffusionSamplingParams(extra_args={"cell_schedule": schedule, "num_chunks": 2}),
    )
    plan, _, steps = pipeline._plan_for(req)
    assert steps == plan.schedule.num_denoise_steps == expected_steps
    assert plan.schedule.step_replicas in (1, steps + 1)


@pytest.mark.parametrize(
    ("stages", "expected_ordering"),
    [
        (1, Ordering.SELF_FORCING),  # S=1 layer PP defaults to Self-Forcing
        (4, Ordering.LATEST_KV),  # S>1 vertical defaults to Latest-KV
    ],
)
def test_cell_schedule_default_derives_from_topology(stages, expected_ordering):
    """cell_schedule absent: the deploy topology fixes the ordering."""
    pipeline = _bare_pipeline(step_replicas=stages, history=6)
    req = OmniDiffusionRequest(
        prompt="x",
        request_id="default-schedule",
        sampling_params=OmniDiffusionSamplingParams(extra_args={"num_chunks": 2}),
    )
    plan, _, _ = pipeline._plan_for(req)
    assert plan.schedule.ordering is expected_ordering


@pytest.mark.parametrize(("layer_groups", "expect_reject"), [(1, False), (2, True)])
def test_causvid_s1_stream_decode_single_rank_only(layer_groups, expect_reject):
    """S=1 keeps local streaming on a single rank (G=1); G>1 has no VAE on the finish rank."""
    pipeline = _bare_pipeline(step_replicas=1, history=6, layer_groups=layer_groups)
    req = OmniDiffusionRequest(
        prompt="x",
        request_id="s1-stream",
        sampling_params=OmniDiffusionSamplingParams(
            extra_args={
                "cell_schedule": "self_forcing",
                "num_denoise_steps": 3,
                "num_chunks": 2,
                "stream_decode": True,
            }
        ),
    )
    if expect_reject:
        with pytest.raises(ValueError, match="no VAE on its finish rank"):
            pipeline._plan_for(req)
    else:
        plan, _, _ = pipeline._plan_for(req)
        assert plan.schedule.step_replicas == 1


def test_causvid_rejects_explicit_s_eq_d():
    """Latest-KV does not accept the S=D Same-Step topology."""
    pipeline = _bare_pipeline(step_replicas=3, history=6)
    req = OmniDiffusionRequest(
        prompt="x",
        request_id="s-eq-d",
        sampling_params=OmniDiffusionSamplingParams(
            extra_args={"cell_schedule": "latest_kv", "num_denoise_steps": 3, "num_chunks": 2}
        ),
    )
    with pytest.raises(ValueError, match=r"step_replicas must be 1 or D\+1"):
        pipeline._plan_for(req)


def test_causvid_rejects_vertical_self_forcing():
    """Vertical (S>1) step replicas are Latest-KV only; Self-Forcing is S=1 layer PP."""
    pipeline = _bare_pipeline(step_replicas=4, history=6)
    req = OmniDiffusionRequest(
        prompt="x",
        request_id="vertical-sf",
        sampling_params=OmniDiffusionSamplingParams(
            extra_args={"cell_schedule": "self_forcing", "num_denoise_steps": 3, "num_chunks": 2}
        ),
    )
    with pytest.raises(ValueError, match="self_forcing requires step_replicas=1"):
        pipeline._plan_for(req)


def test_causvid_rejects_s1_latest_kv():
    """Latest-KV is the vertical S=D+1 clean topology; S=1 layer PP is self_forcing."""
    pipeline = _bare_pipeline(step_replicas=1, history=6)
    req = OmniDiffusionRequest(
        prompt="x",
        request_id="s1-lkv",
        sampling_params=OmniDiffusionSamplingParams(
            extra_args={"cell_schedule": "latest_kv", "num_denoise_steps": 3, "num_chunks": 2}
        ),
    )
    with pytest.raises(ValueError, match="latest_kv requires step_replicas=D\\+1"):
        pipeline._plan_for(req)


def test_cell_schedule_s1_steps_independent_of_history():
    """S=1 self_forcing always runs the clean step (ordering property)."""
    pipeline = _bare_pipeline(step_replicas=1, history=6)
    sf = OmniDiffusionRequest(
        prompt="x",
        request_id="s1-sf",
        sampling_params=OmniDiffusionSamplingParams(
            extra_args={"cell_schedule": "self_forcing", "num_denoise_steps": 3, "num_chunks": 2}
        ),
    )
    plan_sf, _, _ = pipeline._plan_for(sf)
    assert plan_sf.schedule.num_schedule_steps == 4


@pytest.mark.parametrize("shape", [(1, 16, 1, 4, 8), (1, 16, 1, 8, 16)])
def test_pipeline_rejects_token_count_change_before_admission(shape):
    pipeline = _bare_pipeline()
    torch.nn.Module.__init__(pipeline)
    transformer = StageWanTransformer.__new__(StageWanTransformer)
    torch.nn.Module.__init__(transformer)
    transformer.patch_size = (1, 2, 2)
    transformer.register_parameter("test_device", torch.nn.Parameter(torch.zeros(1)))
    pipeline.transformer = transformer
    pipeline.step_replicas = pipeline.layer_groups = pipeline.max_history_chunks = 1
    pipeline.block_size = pipeline.max_chunk_tokens = 16
    pipeline._chunk_ctx = None
    pipeline._default_shift = CAUSVID_DEFAULT_SHIFT
    pipeline._default_denoise = CAUSVID_DEFAULT_DENOISE_STEPS
    cache = NoisyKVCache(
        ARDiffusionNoisyKVSpec(1, 1, 1, 16, 16, 1),
        dtype=torch.float32,
        device=torch.device("cpu"),
        layer_groups=1,
        max_batch_size=1,
    )
    ctx = ARDiffusionCellContext(CellRunSpec(CellTopology(1, 1)), NoisyKVState(cache))
    req = OmniDiffusionRequest(
        prompt="x",
        request_id="different-shape",
        sampling_params=OmniDiffusionSamplingParams(extra_args={"latent_shape": shape, "num_denoise_steps": 1}),
    )
    with pipeline.bind_ar_diffusion_noisy_kv(ctx.kv):
        with pytest.raises(ValueError, match="does not match preallocated NoisyKV"):
            pipeline.forward(req)
    assert pipeline.block_size == pipeline.max_chunk_tokens == cache.spec.max_chunk_tokens == 16
    assert not ctx.pending and not ctx.inflight and not cache.pool.keys


def test_latent_adapter_g_gt1_packs_hidden_and_advances_on_last_only():
    """Non-last groups emit tokens; only stage-last runs FlowEuler."""
    shape = (1, 4, 1, 2, 2)
    device = torch.device("cpu")
    dtype = torch.float32
    prompt = torch.zeros(1, 8, 16, device=device, dtype=dtype)

    class _FakeTransformer:
        def __init__(self, *, is_layer_group_first: bool, is_layer_group_last: bool) -> None:
            self.is_layer_group_first = is_layer_group_first
            self.is_layer_group_last = is_layer_group_last
            self.start_layer = 0
            self.end_layer = 1
            self.dim = 8

        def prepare_cell(self, *_a, **_k) -> None:
            return None

        def seq_len_for_latent(self, _shape) -> int:
            return 3

        def embed(self, latent):
            return torch.ones(latent.shape[0], 3, 8, device=latent.device, dtype=latent.dtype)

        def block(self, _index, tokens, **_kw):
            return tokens

        def unembed(self, _tokens, *, latent, **_kw):
            return torch.zeros_like(latent)

    sampler = FlowEuler(2, shift=1.0)
    mid = WanCell(
        _FakeTransformer(is_layer_group_first=True, is_layer_group_last=False),  # type: ignore[arg-type]
        sampler=sampler,
        prompt_embeds=prompt,
        latent_shape=shape,
        seed=0,
        device=device,
        dtype=dtype,
    )
    last = WanCell(
        _FakeTransformer(is_layer_group_first=False, is_layer_group_last=True),  # type: ignore[arg-type]
        sampler=sampler,
        prompt_embeds=prompt,
        latent_shape=shape,
        seed=0,
        device=device,
        dtype=dtype,
    )

    from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.batch import (
        ComputeBatch,
        MemberRef,
    )

    def _compute(adapter, req, cell, member_input=None):
        chunk, step, block = cell
        batch = ComputeBatch(
            members=(MemberRef(req, chunk=chunk, step=step, block=block, task_index=0),),
            kv_views=(None,),
        )
        return adapter.forward(batch, [adapter], member_inputs=[member_input])

    mid_out = _compute(mid, "r0", (0, 0, 0))
    assert {"latent", "hidden_states"} <= set(mid_out)
    packed = mid.pack_activation(mid_out)
    assert "latent" in packed and "hidden_states" in packed

    last_in = last.unpack_activation(packed)
    last_out = _compute(last, "r0", (0, 0, 0), member_input=last_in)
    assert "latent" in last_out
    assert 0 not in last.finished.get("r0", {})
    # Next denoise step on a non-first group still needs peer hidden_states.
    last_out2 = _compute(
        last,
        "r0",
        (0, 1, 0),
        member_input={"latent": last_out["latent"], "hidden_states": packed["hidden_states"]},
    )
    assert 0 in last.finished["r0"]
    assert last_out2["latent"].shape == shape


def test_pipeline_forward_passes_members_map_once(monkeypatch):
    pipeline = CausVidWanPipeline.__new__(CausVidWanPipeline)
    pipeline._chunk_ctx = SimpleNamespace()
    calls: list[dict] = []
    tags: dict[str, int] = {}
    cells = {"A": WanCell.__new__(WanCell), "B": WanCell.__new__(WanCell)}

    def _admit(req: OmniDiffusionRequest, *, decode_tag=0):
        tags[req.request_id] = decode_tag
        return cells[req.request_id], req.request_id, False, None

    monkeypatch.setattr(pipeline, "_execution_context", lambda: pipeline._chunk_ctx)
    monkeypatch.setattr(pipeline, "admit_cell_request", _admit)
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.causvid_wan.pipeline_causvid_wan.run_cell_pipeline",
        lambda **kwargs: calls.append(kwargs),
    )
    emitted: list[str] = []

    def _emit(adapter, kv_id, stream_job=None):
        del adapter, stream_job
        emitted.append(kv_id)
        return DiffusionOutput(finished=True)

    monkeypatch.setattr(pipeline, "emit_cell_output", _emit)
    batch = DiffusionRequestBatch(
        [
            OmniDiffusionRequest(prompt="a", request_id="A", sampling_params=OmniDiffusionSamplingParams()),
            OmniDiffusionRequest(prompt="b", request_id="B", sampling_params=OmniDiffusionSamplingParams()),
        ]
    )
    outs = pipeline.forward(batch)
    assert len(calls) == 1
    assert set(calls[0]["members"]) == {"A", "B"}
    assert calls[0]["members"]["A"] is cells["A"]
    assert calls[0]["members"]["B"] is cells["B"]
    assert calls[0]["ctx"] is pipeline._chunk_ctx
    assert tags == {"A": 0, "B": 1}
    assert emitted == ["A", "B"]
    assert len(outs) == 2


@pytest.mark.parametrize("failure_stage", ["admit", "compute", "output"])
def test_pipeline_forward_cancels_stream_jobs_on_failure(monkeypatch, failure_stage):
    pipeline = CausVidWanPipeline.__new__(CausVidWanPipeline)
    pipeline._chunk_ctx = SimpleNamespace()
    cancelled: list[str] = []

    class _Job:
        def __init__(self, rid: str) -> None:
            self.rid = rid

        def cancel(self) -> None:
            cancelled.append(self.rid)

    cells = {rid: WanCell.__new__(WanCell) for rid in ("A", "B")}

    def _admit(req: OmniDiffusionRequest, *, decode_tag=0):
        if req.request_id == "B" and failure_stage == "admit":
            raise ValueError("decode wave failed")
        return cells[req.request_id], req.request_id, False, _Job(req.request_id)

    monkeypatch.setattr(pipeline, "_execution_context", lambda: pipeline._chunk_ctx)
    monkeypatch.setattr(pipeline, "admit_cell_request", _admit)

    def run(**kwargs):
        if failure_stage == "compute":
            raise ValueError("decode wave failed")

    def emit(*args):
        raise ValueError("decode wave failed")

    monkeypatch.setattr("vllm_omni.diffusion.models.causvid_wan.pipeline_causvid_wan.run_cell_pipeline", run)
    monkeypatch.setattr(pipeline, "emit_cell_output", emit)
    batch = DiffusionRequestBatch(
        [
            OmniDiffusionRequest(prompt="a", request_id="A", sampling_params=OmniDiffusionSamplingParams()),
            OmniDiffusionRequest(prompt="b", request_id="B", sampling_params=OmniDiffusionSamplingParams()),
        ]
    )
    with pytest.raises(ValueError, match="decode wave failed"):
        pipeline.forward(batch)
    assert cancelled == (["A"] if failure_stage == "admit" else ["A", "B"])


def test_conditioning_cache_reuses_blocks_and_invalidates_member_order():
    """Exercise real StageWanTransformer caching, not the WanCell test double."""
    tf = StageWanTransformer.__new__(StageWanTransformer)
    torch.nn.Module.__init__(tf)
    calls = []

    def condition(timestep, text, *_args, **_kwargs):
        calls.append((timestep.clone(), text.clone()))
        value = text.mean(dim=1) + timestep[:, None]
        return value, value, text, None

    tf._wan = SimpleNamespace(condition_embedder=condition, timestep_proj_prepare=lambda value, _: value)
    a, b = torch.ones(1, 3, 4), torch.full((1, 3, 4), 7.0)
    timestep = torch.tensor([800.0, 500.0])
    tf.prepare_cell((a, b))
    first = tf._embedding((0, 1), timestep, batch_size=2, device=a.device, dtype=a.dtype)
    tf._rotary[0] = (a, b)
    tf.prepare_cell((a, b))
    second = tf._embedding((0, 1), timestep, batch_size=2, device=a.device, dtype=a.dtype)
    assert second is first
    assert len(calls) == 1
    assert 0 in tf._rotary
    tf.prepare_cell((b, a))
    assert not tf._rotary
    third = tf._embedding((0, 1), timestep, batch_size=2, device=a.device, dtype=a.dtype)
    assert len(calls) == 2
    torch.testing.assert_close(third[2], torch.cat([b, a]))
    assert not torch.equal(third[2], first[2])
    tf._embedding((0, 1), timestep + 1, batch_size=2, device=a.device, dtype=a.dtype)
    assert len(calls) == 3


@pytest.mark.parametrize("texts", [(torch.zeros(2, 3, 4),), (torch.zeros(1, 3, 4), torch.zeros(1, 5, 4))])
def test_conditioning_rejects_invalid_member_shapes(texts):
    tf = StageWanTransformer.__new__(StageWanTransformer)
    torch.nn.Module.__init__(tf)
    with pytest.raises(ValueError, match="conditioning requires"):
        tf.prepare_cell(texts)
