# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contract tests for the TaoMate-H3 streaming port."""

from __future__ import annotations

import json
from fractions import Fraction

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from vllm_omni.diffusion.models.taomate_h3 import geometry as geo
from vllm_omni.diffusion.models.taomate_h3.kv_cache import (
    AUDIO_TOKEN_TAG,
    TEXT_TOKEN_TAG,
    VIDEO_TOKEN_TAG,
    CleanAVKVCache,
    KVContract,
)
from vllm_omni.diffusion.models.taomate_h3.lora import TaoMateLoRAAdapter, is_taomate_lora_dir, taomate_lora_targets
from vllm_omni.diffusion.models.taomate_h3.packed import (
    prompt_rope_start,
    taomate_audio_only_frozen_prefix_packed_layout,
    taomate_audio_only_packed_layout,
    taomate_phase_packed_layout,
)
from vllm_omni.diffusion.models.taomate_h3.schedule import (
    euler_eta0_update_,
    student_sigmas,
    teacher_sigmas,
    time_shift_sigmas,
)

# ----------------------------------------------------------------------------
# geometry


def test_direct_plan_matches_the_official_request_geometry() -> None:
    plan = geo.direct_5s_plan()
    assert plan.native_frame_count == 124
    assert [p.frame_count for p in plan.phases] == [39, 34, 34, 17]
    assert [p.video_latent_count for p in plan.phases] == [12, 10, 10, 5]
    assert plan.video_latent_count == 37
    assert plan.audio_latent_count == 207
    assert [p.audio_latent_start for p in plan.phases] == [0, 65, 122, 178]


def test_continuation_plan_has_steady_geometry_and_rounded_audio() -> None:
    first = geo.request_plan(1)
    assert first.native_frame_count == 119
    assert [p.video_latent_count for p in first.phases] == [10, 10, 10, 5]
    assert first.audio_latent_count in (198, 199)
    # Audio boundaries are rounded on the global timeline: request k starts at
    # frame 124 + 119 (k - 1).
    for k in (1, 2, 5, 12):
        plan = geo.request_plan(k)
        start_frame = 124 + 119 * (k - 1)
        expected = geo.audio_latent_boundary(start_frame + 119) - geo.audio_latent_boundary(start_frame)
        assert plan.audio_latent_count == expected
        assert plan.phases[0].audio_latent_start == 0


def test_video_temporal_positions_follow_the_release_spacing() -> None:
    assert geo.video_temporal_position(0) == 0
    assert geo.video_temporal_position(1) == Fraction(5, 3)
    assert geo.video_temporal_position(2) == Fraction(5, 3) * 5
    # Five latents span 17 frames * 5/3 = 85/3 RoPE units.
    assert geo.video_temporal_position(5) == Fraction(85, 3)


def test_canvas_geometry_and_frame_budget() -> None:
    canvas = geo.CanvasGeometry(height=864, width=480)
    assert (canvas.latent_h, canvas.latent_w, canvas.frame_rows) == (54, 30, 405)
    with pytest.raises(ValueError):
        geo.CanvasGeometry(height=720, width=1280)
    assert geo.phases_for_frames(1) == 4
    assert geo.phases_for_frames(124) == 4
    assert geo.phases_for_frames(125) == 8
    assert geo.phases_for_frames(124 + 119 * 3) == 16


# ----------------------------------------------------------------------------
# schedules


def test_student_and_teacher_sigma_ladders() -> None:
    video, audio = student_sigmas()
    assert len(video) == len(audio) == 4
    assert video[0] == pytest.approx(1.0) and video[-1] == pytest.approx(0.0)
    full = time_shift_sigmas(num_steps=50, shift_scale=12.0)
    assert video == [full[0], full[16], full[33], full[49]]
    t_video, t_audio = teacher_sigmas()
    assert len(t_video) == len(t_audio) == 10
    assert t_video[3] > t_audio[3]  # video shift 12 keeps more noise than audio shift 3


def test_euler_eta0_update_matches_reference_formula() -> None:
    state = torch.randn(7, 4)
    velocity = torch.randn(7, 4)
    sigma, sigma_next = 0.7, 0.3
    expected_denoised = state + sigma * velocity
    expected = (sigma_next / sigma) * state + (1 - sigma_next / sigma) * expected_denoised
    out = euler_eta0_update_(state.clone(), velocity, sigma_curr=sigma, sigma_next=sigma_next)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


# ----------------------------------------------------------------------------
# packed layouts


def test_phase_layout_times_sit_on_the_global_timeline() -> None:
    plan = geo.request_plan(1)
    phase = plan.phases[2]
    text_len = 11
    origin = 17  # first request's text length
    video_offset, audio_offset = 37, 207
    packed = taomate_phase_packed_layout(
        text_len=text_len,
        phase=phase,
        latent_h=54,
        latent_w=30,
        media_time_origin=origin,
        video_latent_offset=video_offset,
        audio_latent_offset=audio_offset,
    )
    seq_len = int(packed["seq_len"])
    used = text_len + 2 * phase.audio_latent_count + phase.video_latent_count * 405
    assert seq_len % 64 == 0 and seq_len >= used
    assert int(packed["cu_seqlens"][1]) == used
    grid = packed["img_position_ids"]
    text_pos = packed["text_pos"].view(-1)
    start = prompt_rope_start(text_len=text_len, media_time_origin=origin, video_latent_offset=video_offset)
    assert grid[text_pos[0], 0].item() == pytest.approx(start)
    assert grid[text_pos[-1], 0].item() == pytest.approx(start + text_len - 1)
    first_video_row = packed["img_pos"].view(-1)[0]
    expected_video_time = float(origin + geo.video_temporal_position(video_offset + phase.video_latent_start))
    assert grid[first_video_row, 0].item() == pytest.approx(expected_video_time)
    audio_pos = packed["audio_pos"].view(-1).view(2, phase.audio_latent_count)
    assert grid[audio_pos[0, 0], 0].item() == pytest.approx(origin + audio_offset + phase.audio_latent_start)
    assert grid[audio_pos[1, -1], 0].item() == pytest.approx(origin + audio_offset + phase.audio_latent_stop - 1)
    tags = packed["token_tags"]
    assert (tags[text_pos] == TEXT_TOKEN_TAG).all()
    assert (tags[packed["audio_pos"].view(-1)] == AUDIO_TOKEN_TAG).all()
    assert (tags[packed["img_pos"].view(-1)] == VIDEO_TOKEN_TAG).all()
    assert (tags[used:] == -1).all()


def test_audio_only_layouts() -> None:
    plain = taomate_audio_only_packed_layout(text_len=9, audio_t=207, latent_h=54, latent_w=30)
    assert plain["img_pos"].numel() == 0 and plain["update_mask"].numel() == 0
    assert plain["audio_pos"].numel() == 2 * 207
    assert plain["audio_update_mask"].all()
    assert int(plain["cu_seqlens"][1]) == 9 + 414
    frozen = taomate_audio_only_frozen_prefix_packed_layout(
        text_len=9,
        ref_audio_t=40,
        audio_t=198,
        latent_h=54,
        latent_w=30,
        reference_time_start=9 + 207 - 40,
        target_time_start=9 + 207,
    )
    mask = frozen["audio_update_mask"]
    assert mask.shape[0] == 2 * (40 + 198)
    assert not mask[:80].any() and mask[80:].all()
    grid = frozen["img_position_ids"]
    audio_pos = frozen["audio_pos"].view(-1)
    assert grid[audio_pos[0], 0].item() == pytest.approx(9 + 207 - 40)
    assert grid[audio_pos[80], 0].item() == pytest.approx(9 + 207)


# ----------------------------------------------------------------------------
# KV cache


def _fake_kv(rows: int, contract: KVContract) -> tuple[torch.Tensor, torch.Tensor]:
    key = torch.randn(rows, contract.local_heads, contract.head_dim).to(contract.dtype)
    value = torch.randn(rows, contract.local_heads, contract.head_dim).to(contract.dtype)
    return key, value


def _stage_chunk(
    cache: CleanAVKVCache, contract: KVContract, *, video_rows: int, audio_rows: int, text_rows: int
) -> None:
    seq = text_rows + audio_rows + video_rows + 3
    tags = torch.full((seq,), -1, dtype=torch.long)
    tags[:text_rows] = TEXT_TOKEN_TAG
    tags[text_rows : text_rows + audio_rows] = AUDIO_TOKEN_TAG
    tags[text_rows + audio_rows : text_rows + audio_rows + video_rows] = VIDEO_TOKEN_TAG
    mask = (tags == AUDIO_TOKEN_TAG) | (tags == VIDEO_TOKEN_TAG)
    cache.begin_clean_commit(cache.committed_blocks)
    for name in contract.layer_names:
        key, value = _fake_kv(seq, contract)
        cache.stage(name, key, value, tags, mask)
    cache.commit()


def test_kv_cache_retention_policy() -> None:
    contract = KVContract(num_layers=3, local_heads=2, head_dim=4, dtype=torch.float32)
    cache = CleanAVKVCache(contract)
    _stage_chunk(cache, contract, video_rows=12, audio_rows=6, text_rows=5)
    cache.retain_sink_and_recent_commits()
    assert cache.history_tokens == 18 and cache.committed_blocks == 1
    _stage_chunk(cache, contract, video_rows=10, audio_rows=4, text_rows=5)
    cache.retain_sink_and_recent_commits()
    assert cache.history_tokens == 18 + 14
    _stage_chunk(cache, contract, video_rows=10, audio_rows=4, text_rows=5)
    cache.retain_sink_and_recent_commits()
    # Chunk 0 aged to a video-only sink (12 rows), chunks 1 and 2 complete.
    assert cache.history_video_tokens == 12 + 10 + 10
    assert cache.history_audio_tokens == 4 + 4
    _stage_chunk(cache, contract, video_rows=5, audio_rows=2, text_rows=5)
    cache.retain_sink_and_recent_commits()
    assert cache.history_video_tokens == 12 + 10 + 5
    assert cache.history_audio_tokens == 4 + 2
    dropped = cache.drop_audio_history()
    assert dropped == 6 and cache.history_audio_tokens == 0 and cache.history_video_tokens == 27
    for name in contract.layer_names:
        assert cache.history(name).key.shape[0] == 27


def test_kv_cache_rollback_and_ordering() -> None:
    contract = KVContract(num_layers=2, local_heads=1, head_dim=2, dtype=torch.float32)
    cache = CleanAVKVCache(contract)
    cache.begin_clean_commit(0)
    with pytest.raises(RuntimeError):
        cache.commit()  # missing layers
    cache.rollback()
    assert not cache.clean_commit_active
    with pytest.raises(RuntimeError):
        cache.begin_clean_commit(1)


# ----------------------------------------------------------------------------
# LoRA adapter


class _TinyArch:
    num_layers = 1
    token_refiner_num_layers = 1
    num_attention_heads = 2
    attention_head_dim = 4
    hidden_size = 8
    ffn_hidden_size = 6


class _TupleLinear(nn.Linear):
    """A linear returning ``(out, bias)`` like vLLM's parallel linears."""

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        return super().forward(x), None


def _tiny_transformer() -> nn.Module:
    arch = _TinyArch()
    inner = arch.num_attention_heads * arch.attention_head_dim
    model = nn.Module()
    model.arch = arch
    for block in ("token_refiner.blocks.0", "blocks.0"):
        parent = model
        for part in block.split("."):
            child = getattr(parent, part, None)
            if child is None:
                child = nn.Module()
                parent.add_module(part, child)
            parent = child
        attn = nn.Module()
        attn.qkv_proj = _TupleLinear(arch.hidden_size, 3 * inner, bias=False)
        attn.out_proj = _TupleLinear(inner, arch.hidden_size, bias=False)
        mlp = nn.Module()
        mlp.fc1 = _TupleLinear(arch.hidden_size, 2 * arch.ffn_hidden_size, bias=False)
        mlp.fc2 = _TupleLinear(arch.ffn_hidden_size, arch.hidden_size, bias=False)
        parent.add_module("attn", attn)
        parent.add_module("mlp", mlp)
    return model


def _write_adapter(tmp_path, model: nn.Module, rank: int = 3, alpha: float = 6.0) -> str:
    tensors = {}
    modules = dict(model.named_modules())
    for target in taomate_lora_targets(num_blocks=1, num_refiner_blocks=1):
        weight = modules[target].weight
        out_features, in_features = weight.shape
        tensors[f"{target}.lora_a"] = torch.randn(rank, in_features)
        tensors[f"{target}.lora_b"] = torch.randn(out_features, rank)
    save_file(tensors, str(tmp_path / "adapter_model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps({"rank": rank, "alpha": alpha}))
    return str(tmp_path)


def _adapter_tensor(tmp_path, name: str) -> torch.Tensor:
    from safetensors import safe_open

    with safe_open(str(tmp_path / "adapter_model.safetensors"), framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def test_lora_adapter_hooks_add_scaled_delta_and_can_be_disabled(tmp_path) -> None:
    torch.manual_seed(0)
    model = _tiny_transformer()
    adapter_dir = _write_adapter(tmp_path, model)
    assert is_taomate_lora_dir(adapter_dir)
    adapter = TaoMateLoRAAdapter.load(adapter_dir, transformer=model, device=torch.device("cpu"), dtype=torch.float32)
    assert adapter.scale == pytest.approx(2.0)
    assert len(adapter.bound_targets) == 8
    x = torch.randn(5, 8)
    fc2_in = torch.randn(5, 6)
    with adapter.disabled():
        base_out, _ = model.get_submodule("blocks.0.mlp.fc2")(fc2_in)
    lora_out, _ = model.get_submodule("blocks.0.mlp.fc2")(fc2_in)
    a = adapter._lora_a["blocks.0.mlp.fc2"]
    b = adapter._lora_b["blocks.0.mlp.fc2"]
    expected = base_out + (fc2_in @ a.t()) @ b.t() * adapter.scale
    torch.testing.assert_close(lora_out, expected, rtol=1e-5, atol=1e-5)
    # The qkv B rows are consumed in the adapter's merged [Q; K; V] order.
    qkv_a = adapter._lora_a["blocks.0.attn.qkv_proj"]
    qkv_b = adapter._lora_b["blocks.0.attn.qkv_proj"]
    assert qkv_b.shape == (3 * 8, 3)
    torch.testing.assert_close(qkv_b, _adapter_tensor(tmp_path, "blocks.0.attn.qkv_proj.lora_b"))
    with adapter.disabled():
        base_qkv, _ = model.get_submodule("blocks.0.attn.qkv_proj")(x)
    lora_qkv, _ = model.get_submodule("blocks.0.attn.qkv_proj")(x)
    torch.testing.assert_close(lora_qkv - base_qkv, (x @ qkv_a.t()) @ qkv_b.t() * adapter.scale, rtol=1e-5, atol=1e-5)
    adapter.unbind()
    plain, _ = model.get_submodule("blocks.0.mlp.fc2")(fc2_in)
    torch.testing.assert_close(plain, base_out)


def test_lora_adapter_rejects_incomplete_inventory(tmp_path) -> None:
    model = _tiny_transformer()
    _write_adapter(tmp_path, model)
    tensors = {}
    from safetensors import safe_open

    with safe_open(str(tmp_path / "adapter_model.safetensors"), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if not key.startswith("blocks.0.mlp.fc2"):
                tensors[key] = handle.get_tensor(key)
    save_file(tensors, str(tmp_path / "adapter_model.safetensors"))
    with pytest.raises(Exception, match="inventory"):
        TaoMateLoRAAdapter.load(str(tmp_path), transformer=model, device=torch.device("cpu"), dtype=torch.float32)


# ----------------------------------------------------------------------------
# streaming audio decoder tail handling


class _FakeAudioVAE:
    sample_rate = 32000

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        # 800 samples per latent, value = latent index of the window position.
        steps = int(latent.shape[-1])
        wave = latent[0, 0].repeat_interleave(800).view(1, 1, steps * 800).expand(1, 2, -1)
        return wave.float()


def test_audio_decoder_pads_the_rounded_tail_but_rejects_missing_audio() -> None:
    from vllm_omni.diffusion.models.taomate_h3.stream_decode import StreamingAudioDecoder

    decoder = StreamingAudioDecoder(_FakeAudioVAE(), device=torch.device("cpu"))
    latents = torch.zeros(2, 32, 603)
    latents[0, 0] = torch.arange(603, dtype=torch.float32)
    decoder.append(latents)
    # 362 frames at 24 fps need 482666 samples; 603 latents give 482400.
    samples = decoder.decode_range(453333, 482666)
    assert samples.shape == (482666 - 453333, 2)
    assert samples[-1, 0] == 0.0 and samples[-267, 0] == 602.0
    with pytest.raises(RuntimeError, match="do not cover"):
        decoder.decode_range(482400, 482400 + 2000)


# ----------------------------------------------------------------------------
# pinned packed lengths


def test_pinned_lengths_cover_every_phase_and_teacher_document() -> None:
    from vllm_omni.diffusion.models.taomate_h3 import pipeline as pipeline_module
    from vllm_omni.diffusion.models.taomate_h3.audio_teacher import ROLLOVER_LATENTS_PER_CHANNEL
    from vllm_omni.diffusion.models.taomate_h3.kv_cache import KVContract

    canvas = geo.CanvasGeometry(height=864, width=480)
    session = pipeline_module._Session(
        session_id="s",
        canvas=canvas,
        seed=1,
        contract=KVContract(num_layers=1, local_heads=1, head_dim=1),
        video_decoder=None,  # type: ignore[arg-type]
        audio_decoder=None,  # type: ignore[arg-type]
        audio_kv_reset_requests=12,
        pad_text_tokens=128,
    )
    seen: set[tuple[int, int]] = set()
    for request_index in range(6):
        plan = geo.request_plan(request_index)
        for phase in plan.phases:
            pinned = session.pinned_phase_seq_len(phase, text_len=40)
            assert pinned is not None and pinned % 64 == 0
            used = 40 + 2 * phase.audio_latent_count + phase.video_latent_count * canvas.frame_rows
            assert used <= pinned < used + 64 + 64  # one 64-token bucket of text padding plus the 64-row alignment
            seen.add((phase.index, pinned))
    # Two shapes for phase 0 (12 latents once, then 10) and one per later phase.
    assert len(seen) == 5
    teacher_first = session.pinned_teacher_seq_len(40, with_reference=False)
    teacher_next = session.pinned_teacher_seq_len(40, with_reference=True)
    # A 40-token prompt reserves one 64-token bucket, not the whole 128-token budget.
    assert teacher_first == -(-(64 + 2 * 207) // 64) * 64
    assert teacher_next == -(-(64 + 2 * (207 + ROLLOVER_LATENTS_PER_CHANNEL)) // 64) * 64
    assert session.text_budget(40) == 64 and session.text_budget(65) == 128 and session.text_budget(300) == 128
    assert session.pinned_phase_seq_len(plan.phases[0], text_len=200) is None


# ----------------------------------------------------------------------------
# just-in-time prompt hold (model_config.taomate_h3_hold_for_prompt)


_NUM_PHASES = 4  # 34/34/34/17-frame phases per five-second request


class _RequestStartedError(Exception):
    pass


def _hold_pipeline(monkeypatch):
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.taomate_h3.pipeline import TaoMateH3Pipeline

    pipe = object.__new__(TaoMateH3Pipeline)  # no weights: only the boundary logic runs
    pipe._tm_hold_for_prompt = True
    pipe._tm_hold_poll_seconds = 0.0
    pipe._tm_hold_max_seconds = 0.0
    pipe._tm_hold_fallback_prompt = None
    pipe._tm_hold_fallback_encoded = None
    monkeypatch.setattr(TaoMateH3Pipeline, "device", torch.device("cpu"), raising=False)

    def begin_request(**kwargs):
        raise _RequestStartedError

    session = SimpleNamespace(begin_request=begin_request)
    monkeypatch.setattr(pipe, "_session", lambda state: session, raising=False)
    monkeypatch.setattr(pipe, "_require_bound_ar_state", lambda: None, raising=False)
    return pipe


def _hold_state(chunk_index: int, applied_version: int):
    from types import SimpleNamespace

    from vllm_omni.diffusion.worker.utils import StepRequestState
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    state = StepRequestState(request_id="r", sampling=OmniDiffusionSamplingParams())
    state.chunk_index = chunk_index
    state.total_chunks = 400
    state.chunk_num_steps = 3
    state.step_in_chunk = 3
    state.step_index = 3  # end of the previous phase, as the runner leaves it
    state.timesteps = torch.tensor([1.0, 0.6, 0.3])
    state.prompt_embeds = torch.zeros(4, 8)
    state.extra = {"taomate_prompt_version": applied_version, "text_tags": torch.ones(4, dtype=torch.long)}
    state.interaction_sessions["prompt"] = SimpleNamespace(version=applied_version)
    return state


def test_hold_idles_at_a_request_boundary_until_a_prompt_update(monkeypatch) -> None:
    pipe = _hold_pipeline(monkeypatch)
    state = _hold_state(chunk_index=_NUM_PHASES, applied_version=1)  # request 1, no new prompt yet
    pipe.prepare_next_chunk(state)
    assert state.extra["taomate_held"] is True
    assert state.step_in_chunk == 0  # the runner must not decode the held boundary again
    assert state.current_timestep is not None  # the runner's InputBatch still batches the held state
    assert pipe.supports_chunk_step_grouping is False  # each idle step goes back to the scheduler

    applies = []
    monkeypatch.setattr(pipe, "apply_interaction_at_chunk_boundary", lambda s: applies.append(1), raising=False)
    assert pipe.denoise_step(None, states=[state]) is None  # idle step
    pipe.step_scheduler(state, None)  # no-op while held
    assert state.step_in_chunk == 0 and applies == [1]

    def apply_prompt(s):
        s.interaction_sessions["prompt"].version += 1

    monkeypatch.setattr(pipe, "apply_interaction_at_chunk_boundary", apply_prompt, raising=False)
    with pytest.raises(_RequestStartedError):  # the new prompt releases the hold and starts request 1
        pipe.denoise_step(None, states=[state])
    assert state.extra["taomate_held"] is False
    assert state.extra["taomate_prompt_version"] == 2
    assert pipe.supports_chunk_step_grouping is True  # the started request's steps group again


def test_hold_never_blocks_the_first_request_or_request_mode(monkeypatch) -> None:
    pipe = _hold_pipeline(monkeypatch)
    with pytest.raises(_RequestStartedError):  # request 0 uses the session.start prompt
        pipe.prepare_next_chunk(_hold_state(chunk_index=0, applied_version=0))
    state = _hold_state(chunk_index=_NUM_PHASES, applied_version=1)
    state.extra["taomate_no_hold"] = True
    with pytest.raises(_RequestStartedError):
        pipe.prepare_next_chunk(state)
    pipe._tm_hold_for_prompt = False
    with pytest.raises(_RequestStartedError):  # default: free-running stream keeps the last prompt
        pipe.prepare_next_chunk(_hold_state(chunk_index=_NUM_PHASES, applied_version=1))


# ----------------------------------------------------------------------------
# load-time warmup (model_config.taomate_h3_warmup_requests)


def test_warmup_session_visits_every_teacher_shape():
    """Four warmup requests cover the first request and the three-request audio latent cycle."""
    from vllm_omni.diffusion.models.taomate_h3.geometry import (
        REQUEST_NATIVE_FRAMES,
        STEADY_NATIVE_FRAMES,
        phases_for_frames,
        request_plan,
    )
    from vllm_omni.diffusion.models.taomate_h3.pipeline import TaoMateH3Pipeline

    pipe = object.__new__(TaoMateH3Pipeline)
    pipe._tm_warmup_requests = 4
    pipe._tm_warmup_session_ids = set()
    pipe._tm_default_height = 864
    pipe._tm_default_width = 480
    pipe._tm_default_seed = 8301
    requests = list(pipe.ar_diffusion_warmup_requests("warmup"))
    assert len(requests) == 1
    num_frames = requests[0].sampling_params.num_frames
    assert num_frames == REQUEST_NATIVE_FRAMES + 3 * STEADY_NATIVE_FRAMES
    assert phases_for_frames(num_frames) == 4 * _NUM_PHASES
    assert requests[0].sampling_params.extra_args["session_id"] == "warmup"
    assert pipe._tm_warmup_session_ids == {"warmup"}  # the load-time graph pre-capture keys on it
    # The teacher document shape is set by the request's audio latent count;
    # every count of the steady-state cycle appears within the first four requests.
    counts = {request_plan(index).audio_latent_count for index in range(4)}
    assert counts == {request_plan(index).audio_latent_count for index in range(40)}


# ----------------------------------------------------------------------------
# prompt updates keep the runner's recorded prompt length and the token tags in step


def _prompt_state(text_len: int):
    from vllm_omni.diffusion.worker.utils import StepRequestState
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    state = StepRequestState(request_id="r", sampling=OmniDiffusionSamplingParams())
    state.prompt_embeds = torch.zeros(text_len, 8)
    state.txt_seq_lens = [text_len]
    state.extra = {"text_tags": torch.ones(text_len, dtype=torch.long)}
    # Enough denoise state for InputBatch.make_batch.
    state.latents = torch.zeros(2, 3)
    state.timesteps = torch.tensor([1.0, 0.5])
    state.step_index = 0
    return state


@pytest.mark.parametrize("new_len", [3, 7])
def test_prompt_update_records_the_new_length_and_tags(new_len: int) -> None:
    """A shorter or longer prompt must not be padded or truncated to the first prompt's length."""
    from vllm_omni.diffusion.interaction.modality_handlers.taomate_h3_prompt import (
        TaoMateH3PromptInteractionHandler,
        TaoMateQueuedPromptEvent,
    )
    from vllm_omni.diffusion.worker.input_batch import InputBatch

    calls: list[str] = []

    def encode(*, prompt: str, **kwargs):
        calls.append(prompt)
        rows = len(prompt.split())
        tags = torch.ones(rows, dtype=torch.long)
        tags[0] = 0  # the encoder marks template tokens with tag 0
        return torch.full((rows, 8), float(rows)), tags

    handler = TaoMateH3PromptInteractionHandler(encode_prompt=encode, device=torch.device("cpu"), dtype=torch.bfloat16)
    state = _prompt_state(text_len=4)
    prompt = " ".join(["word"] * new_len)
    handler.enqueue(state, event_id="e1", received_at=0.0, payload={"prompt": prompt}, transition_chunks=5)
    event = state.interaction_sessions["prompt"].pending_event
    assert isinstance(event, TaoMateQueuedPromptEvent) and event.transition_chunks == 0  # always a hard switch
    assert calls == [prompt]

    metadata = handler.apply_at_chunk_boundary(state, boundary_at=1.0)
    assert metadata is not None and metadata.completed_event_ids == ["e1"]
    assert state.prompt_embeds is not None and tuple(state.prompt_embeds.shape) == (new_len, 8)
    assert state.txt_seq_lens == [new_len] and state.prompt_embeds_mask is None
    assert torch.equal(state.extra["text_tags"], event.target_text_tags)
    assert int(state.extra["text_tags"][0]) == 0

    # The step runner's batch now pads to the new length, i.e. not at all.
    batch = InputBatch.make_batch([state])
    assert batch.prompt_embeds is not None and tuple(batch.prompt_embeds.shape) == (1, new_len, 8)
    assert torch.equal(batch.prompt_embeds[0], torch.full((new_len, 8), float(new_len)))
    assert tuple(state.prompt_embeds.shape[-2:]) == (new_len, 8)

    # A second boundary without a new prompt changes nothing.
    handler.apply_at_chunk_boundary(state, boundary_at=2.0)
    assert state.txt_seq_lens == [new_len]


def test_teacher_warm_shapes_cover_the_request_cycle(monkeypatch) -> None:
    """Session-start warming runs one forward per distinct teacher document shape (three)."""
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.taomate_h3.audio_teacher import TaoMateAudioTeacher
    from vllm_omni.diffusion.models.taomate_h3.cuda_graph import kwargs_signature
    from vllm_omni.diffusion.models.taomate_h3.geometry import CanvasGeometry

    teacher = TaoMateAudioTeacher(SimpleNamespace(), lora=None, device=torch.device("cpu"))
    assert (
        teacher.warm_shapes(
            text_embeddings=torch.zeros(5, 5120),
            text_tags=torch.ones(5, dtype=torch.long),
            canvas=CanvasGeometry(height=864, width=480),
            seq_len_for=lambda with_reference: None,
        )
        == 0
    )  # no graph: nothing to warm

    class _Graph:
        enabled = True
        captures = 0

    signatures: list = []

    def fake_forward(forward_kwargs, *, plan, pin=False):
        signatures.append(kwargs_signature(forward_kwargs))
        _Graph.captures += 1
        return None, None

    teacher.graph = _Graph()
    monkeypatch.setattr(teacher, "_forward", fake_forward)
    captured = teacher.warm_shapes(
        text_embeddings=torch.zeros(5, 5120),
        text_tags=torch.ones(5, dtype=torch.long),
        canvas=CanvasGeometry(height=864, width=480),
        seq_len_for=lambda with_reference: 640 if with_reference else 576,
    )
    assert captured == 3 and len(signatures) == 3 and len(set(signatures)) == 3
    audio_rows = sorted(dict(sig[1])["audio_x"][1][1] for sig in signatures)
    assert audio_rows == [576, 640, 640]  # x/audio_x carry the pinned document length


def test_teacher_graph_text_length_range_parses() -> None:
    from vllm_omni.diffusion.models.taomate_h3.pipeline import parse_text_length_range

    assert parse_text_length_range(None) is None and parse_text_length_range("") is None
    assert list(parse_text_length_range("8-10")) == [8, 9, 10]
    assert list(parse_text_length_range([3, 4])) == [3, 4]
    assert list(parse_text_length_range(7)) == [7]
    for bad in ("10-8", "0-4", "a-b", "1-2-3", 0):
        with pytest.raises(ValueError):
            parse_text_length_range(bad)


# ----------------------------------------------------------------------------
# host preparation of the next phase overlapped with the device decode


def _ahead_session(pad_text_tokens: int = 128):
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.taomate_h3.pipeline import _Session

    canvas = geo.CanvasGeometry(height=864, width=480)
    session = _Session(
        session_id="s",
        canvas=canvas,
        seed=7,
        contract=KVContract(num_layers=1, local_heads=1, head_dim=8),
        video_decoder=None,  # type: ignore[arg-type]
        audio_decoder=None,  # type: ignore[arg-type]
        audio_kv_reset_requests=12,
        pad_text_tokens=pad_text_tokens,
    )
    return session, SimpleNamespace()


def test_prepare_ahead_layout_matches_a_fresh_one_and_noise_is_reused() -> None:
    from vllm_omni.diffusion.models.taomate_h3 import geometry as geo
    from vllm_omni.diffusion.models.taomate_h3.packed import taomate_phase_packed_layout

    session, transformer = _ahead_session()
    device = torch.device("cpu")
    session.begin_request(
        text_embeddings=torch.zeros(40, 5120), text_tags=torch.ones(40, dtype=torch.long), device=device
    )
    session.begin_phase(0, transformer=transformer, device=device)
    session.prepare_ahead(completed_phase_index=0)
    assert session.layout_ahead is not None and session.noise_ahead is None
    ahead_packed = session.layout_ahead[1]
    fresh = taomate_phase_packed_layout(
        text_len=40,
        phase=geo.request_plan(0).phases[1],
        latent_h=session.canvas.latent_h,
        latent_w=session.canvas.latent_w,
        media_time_origin=session.media_time_origin,
        video_latent_offset=0,
        audio_latent_offset=0,
        seq_len=session.pinned_phase_seq_len(geo.request_plan(0).phases[1], 40),
    )
    for key in ("img_position_ids", "img_pos", "audio_pos", "token_tags", "cu_seqlens"):
        assert torch.equal(ahead_packed[key], fresh[key]), key
    phase_state = session.begin_phase(1, transformer=transformer, device=device)
    assert session.layout_ahead is None and phase_state.seq_len == int(fresh["seq_len"])
    # Across the request boundary: the noise draws of request 1 are made ahead and consumed.
    session.begin_phase(3, transformer=transformer, device=device)
    session.prepare_ahead(completed_phase_index=3)
    assert session.noise_ahead is not None and session.noise_ahead[0] == 1
    expected_audio, expected_video = session._request_noise(1)
    session.finish_request()
    request = session.begin_request(
        text_embeddings=torch.zeros(40, 5120), text_tags=torch.ones(40, dtype=torch.long), device=device
    )
    assert session.noise_ahead is None and request.index == 1
    fresh_request_rows = expected_audio.view(2, geo.REQUEST_AUDIO_LATENTS, 32)[
        :, geo.REQUEST_AUDIO_LATENTS - request.plan.audio_latent_count :
    ]
    assert torch.equal(request.audio_rows.view(2, -1, 32), fresh_request_rows)
    # A prompt of another length at the boundary makes begin_phase rebuild the layout.
    session.finish_request()
    session.begin_request(
        text_embeddings=torch.zeros(40, 5120), text_tags=torch.ones(40, dtype=torch.long), device=device
    )
    session.begin_phase(3, transformer=transformer, device=device)
    session.prepare_ahead(completed_phase_index=3)
    session.finish_request()
    session.begin_request(
        text_embeddings=torch.zeros(50, 5120), text_tags=torch.ones(50, dtype=torch.long), device=device
    )
    phase_state = session.begin_phase(0, transformer=transformer, device=device)
    assert phase_state.branch.text_len == 50 and session.layout_ahead is None


def test_lora_merge_builds_student_shadows_and_swaps_parameter_sets(tmp_path) -> None:
    """The merged student weight is W + scale * B @ A; modes swap the target's parameters, not its hooks."""
    model = _tiny_transformer()
    adapter_dir = _write_adapter(tmp_path, model)
    adapter = TaoMateLoRAAdapter.load(adapter_dir, transformer=model, device=torch.device("cpu"), dtype=torch.float32)
    fc2 = model.get_submodule("blocks.0.mlp.fc2")
    base_weight = fc2.weight.detach().clone()
    a = _adapter_tensor(tmp_path, "blocks.0.mlp.fc2.lora_a")
    b = _adapter_tensor(tmp_path, "blocks.0.mlp.fc2.lora_b")
    fc2_in = torch.randn(5, 6)
    with adapter.disabled():
        base_out, _ = fc2(fc2_in)
    hooked_out, _ = fc2(fc2_in)

    merged = adapter.build_merged_student(model)
    assert merged == len(adapter.targets) and adapter.merged and adapter.enabled is False
    shadow = fc2.taomate_student
    torch.testing.assert_close(shadow.weight, base_weight + b @ a * adapter.scale, rtol=1e-5, atol=1e-5)
    assert not shadow._forward_hooks and torch.equal(fc2.weight, base_weight)  # base untouched until installed

    adapter.ensure_student()
    assert fc2.weight is shadow.weight
    student_out, _ = fc2(fc2_in)
    torch.testing.assert_close(student_out, hooked_out, rtol=1e-4, atol=1e-4)  # same math as the hooks
    with adapter.disabled():
        assert torch.equal(fc2.weight, base_weight)
        teacher_out, _ = fc2(fc2_in)
        torch.testing.assert_close(teacher_out, base_out)
    assert fc2.weight is shadow.weight  # student weights back after the teacher
    again, _ = fc2(fc2_in)
    torch.testing.assert_close(again, student_out)


def test_hold_is_released_after_its_time_bound(monkeypatch) -> None:
    """taomate_h3_hold_max_seconds: a slow prompt decision stalls the stream at most that long."""
    pipe = _hold_pipeline(monkeypatch)
    pipe._tm_hold_max_seconds = 0.05
    state = _hold_state(chunk_index=_NUM_PHASES, applied_version=1)  # request 1, no new prompt
    pipe.prepare_next_chunk(state)
    assert state.extra["taomate_held"] is True and "taomate_hold_since" in state.extra
    pipe.prepare_next_chunk(state)  # still within the bound
    assert state.extra["taomate_held"] is True
    state.extra["taomate_hold_since"] -= 1.0  # the bound has passed
    with pytest.raises(_RequestStartedError):  # the request starts with the previous prompt
        pipe.prepare_next_chunk(state)
    assert state.extra["taomate_held"] is False and "taomate_hold_since" not in state.extra
    assert state.extra["taomate_prompt_version"] == 1  # a late update still applies at the next boundary


def test_hold_bound_can_fall_back_to_a_neutral_prompt(monkeypatch) -> None:
    """With taomate_h3_hold_fallback_prompt the expired hold runs a neutral prompt, not the previous line."""
    pipe = _hold_pipeline(monkeypatch)
    pipe._tm_hold_max_seconds = 0.05
    pipe._tm_hold_fallback_prompt = "She listens quietly."
    calls: list[str] = []

    def encode_prompt(prompt: str):
        calls.append(prompt)
        tags = torch.ones(6, dtype=torch.long)
        tags[0] = 0
        return torch.full((6, 8), 7.0), tags

    monkeypatch.setattr(pipe, "encode_prompt", encode_prompt, raising=False)
    state = _hold_state(chunk_index=_NUM_PHASES, applied_version=1)
    pipe.prepare_next_chunk(state)
    state.extra["taomate_hold_since"] -= 1.0
    with pytest.raises(_RequestStartedError):
        pipe.prepare_next_chunk(state)
    assert calls == ["She listens quietly."]
    assert tuple(state.prompt_embeds.shape) == (6, 8) and state.txt_seq_lens == [6]
    assert int(state.extra["text_tags"][0]) == 0
    # The encoding is cached: a second expiry does not re-encode.
    state2 = _hold_state(chunk_index=2 * _NUM_PHASES, applied_version=1)
    pipe.prepare_next_chunk(state2)
    state2.extra["taomate_hold_since"] -= 1.0
    with pytest.raises(_RequestStartedError):
        pipe.prepare_next_chunk(state2)
    assert calls == ["She listens quietly."]


def test_fp8_activation_quant_is_pointed_at_the_cuda_kernel() -> None:
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.taomate_h3.pipeline import use_cuda_fp8_activation_quant

    class _Quant:
        def __init__(self) -> None:
            self._forward_method = self.forward_native

        def forward_native(self, x):
            return "native"

        def forward_cuda(self, x):
            return "cuda"

    quant = _Quant()
    linear = torch.nn.Linear(2, 2)
    linear.quant_method = SimpleNamespace(fp8_linear=SimpleNamespace(quant_fp8=quant))  # type: ignore[attr-defined]
    plain = torch.nn.Linear(2, 2)  # no quant method: untouched
    root = torch.nn.Sequential(linear, plain)
    assert use_cuda_fp8_activation_quant(root) == 1
    assert quant._forward_method("x") == "cuda"
    assert use_cuda_fp8_activation_quant(root) == 0  # idempotent


# ----------------------------------------------------------------------------
# buffered K/V cache: in-place assembly, staging and compaction match the reference semantics


def _reference_history(commits: list[tuple[torch.Tensor, torch.Tensor, list[int]]], selection):
    """History rows after ``selection`` of (block, video_only) over committed (key, value, tags) blocks."""
    keys, values = [], []
    for block, video_only in selection:
        key, value, tags = commits[block]
        keep = [i for i, tag in enumerate(tags) if not video_only or tag == VIDEO_TOKEN_TAG]
        idx = torch.tensor(keep)
        keys.append(key.index_select(0, idx))
        values.append(value.index_select(0, idx))
    return torch.cat(keys), torch.cat(values)


def test_kv_cache_assemble_and_in_place_staging_match_concatenation() -> None:
    torch.manual_seed(1)
    contract = KVContract(num_layers=2, local_heads=2, head_dim=4)
    cache = CleanAVKVCache(contract, capacity_rows=8)  # small: forces a buffer growth below
    text, audio, video = 3, 2, 4
    seq = text + audio + video
    tags = torch.tensor([1] * text + [AUDIO_TOKEN_TAG] * audio + [VIDEO_TOKEN_TAG] * video)
    mask = torch.tensor([False] * text + [True] * (audio + video))
    commits: list[tuple[torch.Tensor, torch.Tensor, list[int]]] = []
    for block in range(4):
        cache.begin_clean_commit(block)
        for layer in contract.layer_names:
            k = torch.randn(seq, 2, 4, dtype=torch.bfloat16)
            v = torch.randn(seq, 2, 4, dtype=torch.bfloat16)
            history = cache.history(layer)
            keys, values = cache.assemble(layer, k[:text], v[:text], k[text:], v[text:])
            expected_k = torch.cat(([history.key] if history is not None else []) + [k[text:], k[:text]])
            expected_v = torch.cat(([history.value] if history is not None else []) + [v[text:], v[:text]])
            assert torch.equal(keys, expected_k) and torch.equal(values, expected_v)
            cache.stage_in_place(layer, audio + video, tags, mask)
            if layer == contract.layer_names[0]:
                commits.append((k[text:].clone(), v[text:].clone(), tags[text:].tolist()))
        cache.commit()
        cache.retain_sink_and_recent_commits()
    # After four commits the policy keeps block 0 video-only and blocks 2 and 3 whole.
    assert cache.committed_blocks == 4 and cache.history_tokens == video + 2 * (audio + video)
    ref_k, ref_v = _reference_history(commits, [(0, True), (2, False), (3, False)])
    history = cache.history(contract.layer_names[0])
    assert torch.equal(history.key, ref_k) and torch.equal(history.value, ref_v)
    # Dropping the audio history compacts to the video rows of every kept block.
    removed = cache.drop_audio_history()
    assert removed == 2 * audio and cache.history_tokens == 3 * video
    ref_k, ref_v = _reference_history(commits, [(0, True), (2, True), (3, True)])
    history = cache.history(contract.layer_names[0])
    assert torch.equal(history.key, ref_k) and torch.equal(history.value, ref_v)


def test_kv_cache_stage_paths_agree() -> None:
    """stage() (full-sequence rows) and assemble()+stage_in_place() leave the same history."""
    contract = KVContract(num_layers=1, local_heads=1, head_dim=2)
    layer = contract.layer_names[0]
    seq, text = 6, 2
    tags = torch.tensor([1, 1, AUDIO_TOKEN_TAG, AUDIO_TOKEN_TAG, VIDEO_TOKEN_TAG, VIDEO_TOKEN_TAG])
    mask = torch.tensor([False, False, True, True, True, True])
    k = torch.randn(seq, 1, 2, dtype=torch.bfloat16)
    v = torch.randn(seq, 1, 2, dtype=torch.bfloat16)
    a = CleanAVKVCache(contract)
    a.begin_clean_commit(0)
    a.stage(layer, k, v, tags, mask)
    a.commit()
    b = CleanAVKVCache(contract)
    b.begin_clean_commit(0)
    b.assemble(layer, k[:text], v[:text], k[text:], v[text:])
    b.stage_in_place(layer, seq - text, tags, mask)
    b.commit()
    assert torch.equal(a.history(layer).key, b.history(layer).key)
    assert torch.equal(a.history(layer).value, b.history(layer).value)
    with pytest.raises(RuntimeError):
        b.begin_clean_commit(1)
        b.stage_in_place(layer, seq - text - 1, tags, mask)  # row count must match the mask


def test_contiguous_span_detection() -> None:
    from vllm_omni.diffusion.models.taomate_h3.pipeline import _contiguous_span

    assert _contiguous_span(torch.arange(5, 12)) == (5, 7)
    assert _contiguous_span(torch.tensor([3])) == (3, 1)
    assert _contiguous_span(torch.tensor([1, 2, 4])) is None
    assert _contiguous_span(torch.tensor([], dtype=torch.long)) is None
