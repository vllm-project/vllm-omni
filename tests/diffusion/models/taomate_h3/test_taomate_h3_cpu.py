# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contract tests for the TaoMate-H3 streaming port."""

from __future__ import annotations

import json
from contextlib import contextmanager
from fractions import Fraction

import pytest
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

from vllm_omni.diffusion.models.taomate_h3 import geometry as geo
from vllm_omni.diffusion.models.taomate_h3.kv_cache import (
    AUDIO_TOKEN_TAG,
    TEXT_TOKEN_TAG,
    VIDEO_TOKEN_TAG,
    TaoMateKVHistory,
)
from vllm_omni.diffusion.models.taomate_h3.lora import (
    TaoMateLoRAError,
    prepare_taomate_lora,
    taomate_lora_targets,
)
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

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

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


@pytest.mark.parametrize("text_len", [9, 34])
def test_audio_only_layouts(text_len) -> None:
    plain = taomate_audio_only_packed_layout(text_len=text_len, audio_t=207, latent_h=54, latent_w=30)
    assert plain["img_pos"].numel() == 0 and plain["update_mask"].numel() == 0
    assert plain["audio_pos"].numel() == 2 * 207
    assert plain["audio_update_mask"].all()
    assert int(plain["cu_seqlens"][1]) == text_len + 414
    frozen = taomate_audio_only_frozen_prefix_packed_layout(
        text_len=text_len,
        ref_audio_t=40,
        audio_t=198,
        latent_h=54,
        latent_w=30,
        reference_time_start=text_len + 207 - 40,
        target_time_start=text_len + 207,
    )
    mask = frozen["audio_update_mask"]
    assert mask.shape[0] == 2 * (40 + 198)
    assert not mask[:80].any() and mask[80:].all()
    grid = frozen["img_position_ids"]
    audio_pos = frozen["audio_pos"].view(-1)
    assert grid[audio_pos[0], 0].item() == pytest.approx(text_len + 207 - 40)
    assert grid[audio_pos[80], 0].item() == pytest.approx(text_len + 207)


def test_audio_only_shared_builder_rejects_visual_prefixes() -> None:
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence_ref2va_blocks

    with pytest.raises(ValueError, match="audio-only"):
        minimax_h3_packed_sequence_ref2va_blocks(
            text_len=9,
            latent_t=0,
            latent_h=54,
            latent_w=30,
            audio_t=198,
            ref_blocks=({"kind": "image", "latent_h": 54, "latent_w": 30},),
        )


# ----------------------------------------------------------------------------
# KV cache


def _segmented_cache(*, num_layers=2, heads=2, head_dim=4, capacity=80):
    from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionKVBranchSpec, ARDiffusionSegmentedKVSpec
    from vllm_omni.experimental.ar_diffusion.kv_cache import ARDiffusionKVCache, ARDiffusionKVConfig
    from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState

    pool = ARDiffusionKVCache(
        ARDiffusionKVConfig(enable=True, chunk_size=16, window_chunks=1, gpu_memory_fraction=0.01),
        num_layers=num_layers,
        num_kv_heads=heads,
        head_size=head_dim,
        dtype=torch.bfloat16,
        block_size=16,
        max_model_len=100000,
        available_bytes=128 * 1024**2,
        kv_branches=(ARDiffusionKVBranchSpec("main", 0),),
        session_capacity=1,
        device=torch.device("cpu"),
        segmented=ARDiffusionSegmentedKVSpec(
            block_size=16,
            max_retained_tokens=capacity,
            max_retained_segments=5,
            max_pending_tokens=capacity,
            max_pending_segments=2,
            initial_staging_tokens=capacity * 2 + 8,
        ),
    )
    state = ARDiffusionKVState(pool, "test", {"main": pool.begin_request("test")}, num_layers=num_layers)
    return state, state.segmented("main")


def test_segmented_retention_matches_dense_history_over_thirteen_windows():
    torch.manual_seed(1)
    state, cache = _segmented_cache()
    ledger = TaoMateKVHistory(lambda: cache)
    reference: list[tuple[torch.Tensor, torch.Tensor]] = []
    try:
        for window in range(13):
            if window == 12:
                expected_drop = sum(pair[0].shape[0] for pair in reference[-2:])
                assert ledger.drop_audio_history() == expected_drop
                reference = [(a[:0], v) for a, v in reference]
            for phase in range(4):
                audio, video, text = (
                    3 + ((window + phase) % 3),
                    (12 if not reference else 10) if phase == 0 else (10, 10, 5)[phase - 1],
                    3,
                )
                cache.prepare_layer_views(media_tokens=audio + video, condition_tokens=text)
                if len(reference) <= 2:
                    retained = [tensor for pair in reference for tensor in pair]
                else:
                    retained = [reference[0][1]] + [tensor for pair in reference[-2:] for tensor in pair]
                expected_history = torch.cat(retained) if retained else torch.empty(0, 2, 2, 4, dtype=torch.bfloat16)
                assert cache.history_tokens == expected_history.shape[0]
                media = torch.randn(audio + video, 2, 2, 4, dtype=torch.bfloat16)
                condition = torch.randn(text, 2, 2, 4, dtype=torch.bfloat16)
                for noisy in range(3):
                    for layer in range(2):
                        k, v = cache.assemble(layer, condition[:, 0], condition[:, 1], media[:, 0], media[:, 1])
                        torch.testing.assert_close(k[: cache.history_tokens], expected_history[:, 0], rtol=0, atol=0)
                        torch.testing.assert_close(v[: cache.history_tokens], expected_history[:, 1], rtol=0, atol=0)
                        torch.testing.assert_close(
                            k[cache.history_tokens :], torch.cat((media[:, 0], condition[:, 0])), rtol=0, atol=0
                        )
                    assert ledger.committed_phases == window * 4 + phase
                ledger.begin_commit(audio_tokens=audio, video_tokens=video)
                for layer in range(2):
                    cache.assemble(layer, condition[:, 0], condition[:, 1], media[:, 0], media[:, 1])
                    cache.stage(layer)
                ledger.commit()
                reference.append((media[:audio].clone(), media[audio:].clone()))
        assert ledger.committed_phases == 52
    finally:
        state.close()


def test_segmented_failed_commit_rolls_back_without_advancing_model_history():
    state, cache = _segmented_cache()
    ledger = TaoMateKVHistory(lambda: cache)
    try:
        cache.prepare_layer_views(media_tokens=5, condition_tokens=1)
        ledger.begin_commit(audio_tokens=2, video_tokens=3)
        x = torch.ones(5, 2, 4, dtype=torch.bfloat16)
        cache.assemble(0, x[:1], x[:1], x, x)
        cache.stage(0)
        with pytest.raises(RuntimeError):
            ledger.commit()
        assert ledger.committed_phases == 0 and cache.history_tokens == 0
        ledger.rollback()
        ledger.begin_commit(audio_tokens=2, video_tokens=3)
        for layer in range(2):
            cache.assemble(layer, x[:1], x[:1], x, x)
            cache.stage(layer)
        ledger.commit()
        assert ledger.committed_phases == 1 and cache.history_tokens == 5
    finally:
        state.close()


# ----------------------------------------------------------------------------
# attention compatibility


@pytest.fixture
def vdn_window():
    from vllm_omni.diffusion.attention.backends.vdnh3_attn import VDNLayout

    return VDNLayout(
        used=2,
        text_len=1,
        video_start=1,
        num_frames=1,
        frame_height=1,
        frame_width=1,
        chunk=1,
        radius=0,
    )


@pytest.mark.parametrize("refiner", [False, True], ids=["teacher", "refiner"])
@pytest.mark.parametrize("with_vdn", [False, True])
def test_attention_preserves_upstream_vdn_argument_on_base_path(monkeypatch, vdn_window, refiner, with_vdn):
    from contextlib import nullcontext
    from unittest.mock import Mock

    from vllm_omni.diffusion.models.taomate_h3.attention import (
        MiniMaxH3Attention,
        StreamContext,
        TaoMateH3StreamingAttention,
        stream_context,
    )

    attention = object.__new__(TaoMateH3StreamingAttention)
    nn.Module.__init__(attention)
    attention.layer_name = None if refiner else "blocks.0.attn"
    hidden = torch.zeros(2, 4)
    base_forward = Mock(return_value=hidden)
    monkeypatch.setattr(MiniMaxH3Attention, "forward", base_forward)
    window = vdn_window if with_vdn else None
    # The teacher has no streaming context; a refiner must use base attention
    # even when the surrounding student forward has installed one.
    context = stream_context(Mock(spec=StreamContext)) if refiner else nullcontext()
    with context:
        output = attention(
            hidden,
            rope_table=None,
            cu_seqlens=torch.tensor([0, 2]),
            max_seqlen=2,
            vdn_window=window,
        )
    assert output is hidden
    assert base_forward.call_args.kwargs["vdn_window"] is window


def test_streaming_attention_rejects_vdn_before_running_kernels(vdn_window):
    from unittest.mock import Mock

    from vllm_omni.diffusion.models.taomate_h3.attention import (
        StreamContext,
        TaoMateH3StreamingAttention,
        stream_context,
    )

    attention = object.__new__(TaoMateH3StreamingAttention)
    nn.Module.__init__(attention)
    attention.layer_name = "blocks.0.attn"
    with stream_context(Mock(spec=StreamContext)), pytest.raises(ValueError, match="does not support VDN windows"):
        attention(
            torch.zeros(2, 4),
            rope_table=None,
            cu_seqlens=torch.tensor([0, 2]),
            max_seqlen=2,
            vdn_window=vdn_window,
        )


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


@pytest.mark.parametrize("scale", [0.5, 1.0, 1.5])
def test_native_lora_preserves_fused_rows_and_unscaled_bf16_factors(tmp_path, scale) -> None:
    torch.manual_seed(0)
    model = _tiny_transformer()
    adapter_dir = _write_adapter(tmp_path, model, rank=3, alpha=3 * scale)
    lora, peft, bank = prepare_taomate_lora(adapter_dir, transformer=model, device=torch.device("cpu"))
    assert peft.r == 3 and peft.lora_alpha == 3 * scale
    assert len(lora.loras) == 8 and bank is None
    assert set(peft.target_modules) == set(lora.loras)
    assert lora.clone(42).id == 42
    artifact = load_file(str(tmp_path / "adapter_model.safetensors"))
    for name, weights in lora.loras.items():
        target = name.removeprefix("transformer.")
        assert weights.scaling == scale
        assert weights.lora_a.dtype == weights.lora_b.dtype == torch.bfloat16
        # Neither QKV/fc1 row order nor B's numerical values may change here.
        assert torch.equal(weights.lora_a, artifact[f"{target}.lora_a"].bfloat16())
        assert torch.equal(weights.lora_b, artifact[f"{target}.lora_b"].bfloat16())
        assert not model.get_submodule(target)._forward_hooks


@pytest.mark.parametrize("artifact", ["peft_keys", "incomplete_inventory", "corrupt_file", "missing_file"])
def test_native_lora_rejects_unrelated_or_unreadable_artifacts(tmp_path, artifact) -> None:
    model = _tiny_transformer()
    _write_adapter(tmp_path, model)
    path = tmp_path / "adapter_model.safetensors"
    if artifact == "peft_keys":
        save_file({"blocks.0.attn.qkv_proj.lora_A.weight": torch.zeros(3, 8)}, str(path))
    elif artifact == "incomplete_inventory":
        tensors = {key: value for key, value in load_file(str(path)).items() if not key.startswith("blocks.0.mlp.fc2")}
        save_file(tensors, str(path))
    elif artifact == "corrupt_file":
        path.write_bytes(b"not a safetensors file")
    else:
        path.unlink()
    match = "inventory" if artifact in {"peft_keys", "incomplete_inventory"} else "cannot read"
    with pytest.raises(TaoMateLoRAError, match=match):
        prepare_taomate_lora(tmp_path, transformer=model, device=torch.device("cpu"))


@contextmanager
def _native_manager_case(tmp_path, monkeypatch, *, merge, scale, device="cpu"):
    """Exercise a native artifact and production bindings with tiny real linears."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear

    from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
    from vllm_omni.diffusion.models.taomate_h3.audio_teacher import TaoMateAudioTeacher
    from vllm_omni.diffusion.models.taomate_h3.pipeline import TaoMateH3Pipeline

    device = torch.device(device)
    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
    if device.type == "cpu":
        monkeypatch.setattr(
            linear,
            "dispatch_unquantized_gemm",
            lambda *_args, **_kwargs: lambda _layer, x, weight, bias=None: torch.nn.functional.linear(x, weight, bias),
        )
    with set_current_vllm_config(VllmConfig()):
        torch.manual_seed(17)
        transformer = _tiny_transformer()
        originals = {}
        for target in taomate_lora_targets(num_blocks=1, num_refiner_blocks=1):
            parent_name, _, leaf = target.rpartition(".")
            parent = transformer.get_submodule(parent_name)
            old = getattr(parent, leaf)
            if leaf == "qkv_proj":
                layer = linear.QKVParallelLinear(8, 4, 2, bias=False, params_dtype=torch.bfloat16)
            elif leaf == "fc1":
                layer = linear.MergedColumnParallelLinear(8, [6, 6], bias=False, params_dtype=torch.bfloat16)
            else:
                layer = linear.RowParallelLinear(
                    old.in_features, old.out_features, bias=False, params_dtype=torch.bfloat16
                )
            layer.to(device=device)
            layer.weight.data.copy_(old.weight)
            setattr(parent, leaf, layer)
            originals[f"transformer.{target}"] = layer
        adapter_dir = _write_adapter(tmp_path, transformer, rank=8, alpha=8 * scale)
        prepared = prepare_taomate_lora(adapter_dir, transformer=transformer, device=device, merge=merge)
        pipeline = TaoMateH3Pipeline.__new__(TaoMateH3Pipeline)
        nn.Module.__init__(pipeline)
        pipeline.transformer = transformer
        pipeline._tm_prepared_lora = prepared
        pipeline._tm_lora_manager = None
        pipeline._tm_teacher_graph = pipeline._tm_text_graph = None
        pipeline._tm_teacher = TaoMateAudioTeacher(transformer, manager=None, device=device)
        pipeline._turbo_lora_specs = {}
        pipeline._native_distilled_lora_adapter_ids = set()
        base = {name: layer.weight for name, layer in originals.items()}
        manager = DiffusionLoRAManager(pipeline, device, torch.bfloat16, lora_path=adapter_dir, lora_scale=0.0)
        try:
            yield pipeline, manager, prepared, base, originals
        finally:
            manager.close()


@pytest.mark.parametrize("merge", [False, True])
@pytest.mark.parametrize("scale", [0.5, 1.0, 1.5])
def test_native_artifact_uses_real_manager_and_preserves_teacher(tmp_path, monkeypatch, merge, scale) -> None:
    from torch.utils._python_dispatch import TorchDispatchMode

    class MatmulCount(TorchDispatchMode):
        def __init__(self):
            self.mm = self.addmm = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.mm += func._schema.name in {"aten::linear", "aten::matmul", "aten::mm"}
            self.addmm += func._schema.name in {"aten::addmm_", "aten::addmm"}
            return func(*args, **(kwargs or {}))

    with _native_manager_case(tmp_path, monkeypatch, merge=merge, scale=scale) as case:
        pipeline, manager, prepared, base, originals = case
        assert pipeline._tm_teacher.manager is manager and pipeline._tm_prepared_lora is None

        def unexpected_reload(*args, **kwargs):
            pytest.fail("teacher/request transition reloaded or rebound the fixed adapter")

        monkeypatch.setattr(manager, "_load_adapter", unexpected_reload)
        monkeypatch.setattr(manager, "_bind_adapter_weights", unexpected_reload)
        manager.set_request_adapter(None, 0.0)
        with torch.inference_mode():
            for name, original in originals.items():
                layer = pipeline.get_submodule(name)
                assert layer.base_layer is original and layer.n_slices == 1
                weights = prepared[0].loras[name]
                assert weights.scaling == scale and weights.lora_a.device.type == "cpu"
                x = torch.randn(5, weights.lora_a.shape[1]).bfloat16()
                if merge:
                    merged = (base[name].float() + weights.lora_b.float() @ weights.lora_a.float() * scale).bfloat16()
                    expected = torch.nn.functional.linear(x, merged)
                else:
                    expected = torch.nn.functional.linear(x, base[name])
                    expected.addmm_(x @ weights.lora_a.T, weights.lora_b.T, alpha=scale)
                pointers = (layer.lora_a_stacked[0].data_ptr(), layer.lora_b_stacked[0].data_ptr())
                with MatmulCount() as count:
                    actual = layer(x)[0]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                # One base GEMM, plus one shrink and one fused expansion. QKV
                # and fc1 must not repeat shrink for each logical slice.
                assert (count.mm, count.addmm) == ((1, 0) if merge else (2, 1))
                with pipeline._tm_teacher.manager.suspended():
                    assert layer.base_layer.weight is base[name]
                    torch.testing.assert_close(layer(x)[0], torch.nn.functional.linear(x, base[name]), rtol=0, atol=0)
                torch.testing.assert_close(layer(x)[0], expected, rtol=0, atol=0)
                assert pointers == (layer.lora_a_stacked[0].data_ptr(), layer.lora_b_stacked[0].data_ptr())
        manager.close()
        assert pipeline._tm_teacher.manager is None
        assert all(pipeline.get_submodule(name) is original for name, original in originals.items())
        assert all(original.weight is base[name] for name, original in originals.items())


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


def _session(*, cache=None, pad_text_tokens=128):
    from vllm_omni.diffusion.models.taomate_h3.pipeline import _Session

    return _Session(
        session_id="s",
        canvas=geo.CanvasGeometry(height=864, width=480),
        seed=7,
        cache=cache,
        video_decoder=None,  # type: ignore[arg-type]
        audio_decoder=None,  # type: ignore[arg-type]
        audio_kv_reset_requests=12,
        pad_text_tokens=pad_text_tokens,
    )


def test_pinned_lengths_cover_every_phase_and_teacher_document() -> None:
    from vllm_omni.diffusion.models.taomate_h3.audio_teacher import ROLLOVER_LATENTS_PER_CHANNEL

    session = _session()
    seen: set[tuple[int, int]] = set()
    for request_index in range(6):
        plan = geo.request_plan(request_index)
        for phase in plan.phases:
            pinned = session.pinned_phase_seq_len(phase, text_len=40)
            assert pinned is not None and pinned % 64 == 0
            used = 40 + 2 * phase.audio_latent_count + phase.video_latent_count * session.canvas.frame_rows
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

    state = _prompt_state(text_len=4)
    state.chunk_index = chunk_index
    state.total_chunks = 400
    state.chunk_num_steps = 3
    state.step_in_chunk = 3
    state.step_index = 3  # end of the previous phase, as the runner leaves it
    state.timesteps = torch.tensor([1.0, 0.6, 0.3])
    state.extra["taomate_prompt_version"] = applied_version
    state.interaction_sessions["prompt"] = SimpleNamespace(version=applied_version)
    return state


def test_hold_idles_at_a_request_boundary_until_a_prompt_update(monkeypatch) -> None:
    pipe = _hold_pipeline(monkeypatch)
    state = _hold_state(chunk_index=_NUM_PHASES, applied_version=1)  # request 1, no new prompt yet
    pipe.prepare_next_chunk(state)
    assert state.extra["taomate_held"] is True
    assert state.step_in_chunk == 3  # no fake step or timestep reset while waiting
    assert pipe.supports_chunk_step_grouping is True
    assert pipe.ar_diffusion_step_wait(state).timeout_seconds is None
    with pytest.raises(RuntimeError, match="waiting requests"):
        pipe.denoise_step(None, states=[state])

    def apply_prompt(s):
        s.interaction_sessions["prompt"].version += 1

    monkeypatch.setattr(pipe, "apply_interaction_at_chunk_boundary", apply_prompt, raising=False)
    with pytest.raises(_RequestStartedError):
        pipe.ar_diffusion_resume_step(state, "interaction")
    assert state.extra["taomate_held"] is False
    assert state.extra["taomate_prompt_version"] == 2
    assert pipe.ar_diffusion_step_wait(state) is None


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
# request validation


@pytest.fixture
def prepare_encode_pipeline(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from vllm_omni.diffusion.models.taomate_h3 import pipeline as pipeline_module
    from vllm_omni.diffusion.worker.utils import StepRequestState
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    pipe = object.__new__(pipeline_module.TaoMateH3Pipeline)
    monkeypatch.setattr(pipeline_module.TaoMateH3Pipeline, "device", torch.device("cpu"), raising=False)
    monkeypatch.setattr(pipeline_module, "StreamingVideoDecoder", Mock())
    monkeypatch.setattr(pipeline_module, "StreamingAudioDecoder", Mock())
    pipe._ar_diffusion_kv_state = SimpleNamespace(session_id="session", segmented=lambda branch: Mock(), reset=Mock())
    pipe._tm_teacher = object()
    pipe._tm_default_height = 864
    pipe._tm_default_width = 480
    pipe._tm_default_canvas = geo.CanvasGeometry(height=864, width=480)
    pipe._tm_default_seed = 8301
    pipe._tm_output_rank = True
    pipe._tm_audio_kv_reset_requests = 12
    pipe._tm_pad_text_tokens = 0
    pipe.transformer = SimpleNamespace(
        arch=SimpleNamespace(num_layers=1, attention_head_dim=4), streaming_local_heads=1
    )
    pipe.video_vae = object()
    pipe.audio_vae = object()
    pipe.encode_prompt = Mock(return_value=(torch.zeros(2, 8), torch.ones(2, dtype=torch.long)))
    previous = Mock()
    pipe._tm_sessions = {"session": previous}
    state = StepRequestState(
        request_id="request",
        sampling=OmniDiffusionSamplingParams(num_frames=124),
        prompt="A presenter greets the audience.",
    )
    return pipe, state, previous


@pytest.mark.parametrize(
    ("fps", "frame_rate", "modality"),
    [
        (30, None, None),
        (None, 30, None),
        (30, 24, None),
        (24, 30, None),
        (0, None, None),
        (float("nan"), None, None),
        (None, float("inf"), None),
        (None, None, "image"),
        (None, None, "video"),
        (None, None, "audio"),
    ],
)
def test_prepare_encode_rejects_invalid_requests_before_encoding(prepare_encode_pipeline, fps, frame_rate, modality):
    from vllm_omni.errors import OmniClientError

    pipe, state, previous = prepare_encode_pipeline
    state.sampling.fps = fps
    state.sampling.frame_rate = frame_rate
    if modality:
        state.prompt = {"prompt": state.prompt, "multi_modal_data": {modality: object()}}
    with pytest.raises(OmniClientError, match="text prompts only" if modality else "24 fps"):
        pipe.prepare_encode(state)
    pipe.encode_prompt.assert_not_called()
    previous.close.assert_not_called()
    assert pipe._tm_sessions == {"session": previous}
    assert state.prompt_embeds is None and state.extra == {}


@pytest.mark.parametrize(("fps", "frame_rate"), [(None, None), (24, None), (None, 24), (24.0, 24.0)])
@pytest.mark.parametrize("structured_prompt", [False, True])
def test_prepare_encode_accepts_text_at_native_or_omitted_rates(
    prepare_encode_pipeline, fps, frame_rate, structured_prompt
):
    pipe, state, previous = prepare_encode_pipeline
    state.sampling.fps = fps
    state.sampling.frame_rate = frame_rate
    prompt = state.prompt
    if structured_prompt:
        state.prompt = {"prompt": prompt, "multi_modal_data": {}}
    assert pipe.prepare_encode(state) is state
    pipe.encode_prompt.assert_called_once_with(prompt=prompt)
    previous.close.assert_called_once_with()
    assert pipe._tm_sessions["session"] is not previous
    assert state.prompt_embeds is pipe.encode_prompt.return_value[0]
    assert state.extra["taomate_session_id"] == "session"


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
def test_prompt_update_records_the_new_length_and_tags(new_len: int, monkeypatch, ahead_session) -> None:
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
    session, transformer = ahead_session
    previous = session.begin_request(
        text_embeddings=state.prompt_embeds, text_tags=state.extra["text_tags"], device=torch.device("cpu")
    )
    prompt = " ".join(["word"] * new_len)
    handler.enqueue(state, event_id="e1", received_at=0.0, payload={"prompt": prompt}, transition_chunks=5)
    event = state.interaction_sessions["prompt"].pending_event
    assert isinstance(event, TaoMateQueuedPromptEvent) and event.transition_chunks == 0  # always a hard switch
    assert calls == [prompt]

    metadata = handler.apply_at_chunk_boundary(state, boundary_at=1.0)
    assert metadata is not None
    assert metadata.started_event_ids == metadata.active_event_ids == metadata.completed_event_ids == ["e1"]
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

    # Acknowledgement precedes conditioning when installed inside an existing window.
    pipe = _hold_pipeline(monkeypatch)
    pipe._tm_hold_for_prompt = False
    pipe.transformer = transformer
    monkeypatch.setattr(pipe, "_session", lambda state: session)
    state.total_chunks, state.chunk_num_steps, state.chunk_index = 8, 3, 3
    pipe.prepare_next_chunk(state)
    assert session.request is previous and session.phase.branch.text_len == 4
    assert torch.equal(session.phase.branch.text_embeddings_dev, previous.text_embeddings)
    session.finish_request()
    state.chunk_index = 4
    pipe.prepare_next_chunk(state)
    assert session.request is not previous and session.phase.branch.text_len == new_len
    assert torch.equal(session.phase.branch.text_embeddings_dev, event.target_prompt_embeds)


def test_teacher_warm_shapes_cover_the_request_cycle(monkeypatch) -> None:
    """Session-start warming runs one forward per distinct teacher document shape (three)."""
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.taomate_h3.audio_teacher import TaoMateAudioTeacher
    from vllm_omni.diffusion.models.taomate_h3.cuda_graph import kwargs_signature
    from vllm_omni.diffusion.models.taomate_h3.geometry import CanvasGeometry

    teacher = TaoMateAudioTeacher(SimpleNamespace(), manager=None, device=torch.device("cpu"))
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


@pytest.fixture
def ahead_session():
    from types import SimpleNamespace

    state, cache = _segmented_cache(num_layers=1, heads=1, head_dim=8, capacity=20000)
    try:
        yield _session(cache=lambda: cache), SimpleNamespace()
    finally:
        state.close()


def test_prepare_ahead_layout_matches_a_fresh_one_and_noise_is_reused(ahead_session) -> None:
    from vllm_omni.diffusion.models.taomate_h3 import geometry as geo
    from vllm_omni.diffusion.models.taomate_h3.packed import taomate_phase_packed_layout

    session, transformer = ahead_session
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


@pytest.mark.parametrize("scale", [0.5, 1.0, 1.5])
def test_merged_student_preserves_bf16_rounding_and_base_storage(tmp_path, scale) -> None:
    from types import SimpleNamespace

    torch.manual_seed(7)
    model = _tiny_transformer().bfloat16()
    fc2 = model.get_submodule("blocks.0.mlp.fc2")
    base_weight = fc2.weight
    base_values = base_weight.detach().clone()
    adapter_dir = _write_adapter(tmp_path, model, rank=3, alpha=3 * scale)
    lora, _, bank = prepare_taomate_lora(adapter_dir, transformer=model, device=torch.device("cpu"), merge=True)
    assert bank is not None
    weights = lora.loras["transformer.blocks.0.mlp.fc2"]
    expected = (base_values.float() + weights.lora_b.float() @ weights.lora_a.float() * scale).bfloat16()
    shadow = fc2.taomate_student
    assert torch.equal(shadow.weight, expected)
    assert fc2.weight is base_weight
    assert not shadow._forward_hooks
    with pytest.raises(TaoMateLoRAError, match="bound after quantization"):
        bank.activate()

    wrappers = {
        name: SimpleNamespace(base_layer=model.get_submodule(name.removeprefix("transformer."))) for name in lora.loras
    }
    bank.bind(wrappers)
    bank.activate()
    student_weight = fc2.weight
    assert student_weight is shadow.weight
    x = torch.randn(5, 6).bfloat16()
    student_output, _ = fc2(x)
    assert torch.equal(student_output, torch.nn.functional.linear(x, expected))
    bank.suspend()
    assert fc2.weight is base_weight and torch.equal(fc2.weight, base_values)
    bank.activate()
    assert fc2.weight is student_weight
    bank.close()
    bank.close()
    assert fc2.weight is base_weight and not hasattr(fc2, "taomate_student")


def test_merged_student_snapshots_parameters_only_after_final_quantization(tmp_path) -> None:
    model = _tiny_transformer()
    adapter_dir = _write_adapter(tmp_path, model)
    lora, _, bank = prepare_taomate_lora(adapter_dir, transformer=model, device=torch.device("cpu"), merge=True)
    assert bank is not None
    modules = {name: model.get_submodule(name.removeprefix("transformer.")) for name in lora.loras}
    # Model loading can replace both parameter objects and quantization buffers
    # between preparation and binding. The bank must retain those final objects.
    for module in modules.values():
        module.weight = nn.Parameter(module.weight.bfloat16(), requires_grad=False)
        module.register_buffer("weight_scale", torch.tensor(2.0))
        module.taomate_student.weight = nn.Parameter(module.taomate_student.weight.bfloat16(), requires_grad=False)
        module.taomate_student.register_buffer("weight_scale", torch.tensor(3.0))
    fc2 = modules["transformer.blocks.0.mlp.fc2"]
    base_weight, base_scale = fc2.weight, fc2.weight_scale
    student_weight, student_scale = fc2.taomate_student.weight, fc2.taomate_student.weight_scale
    bank.bind(modules)
    bank.activate()
    assert fc2.weight is student_weight and fc2.weight_scale is student_scale
    bank.suspend()
    assert fc2.weight is base_weight and fc2.weight_scale is base_scale
    bank.close()


def test_merged_student_preparation_and_binding_failures_preserve_base(tmp_path, monkeypatch) -> None:
    from vllm_omni.diffusion.models.taomate_h3 import lora as lora_module

    model = _tiny_transformer()
    adapter_dir = _write_adapter(tmp_path, model)
    original_weights = {name: module.weight for name, module in model.named_modules() if hasattr(module, "weight")}
    original_shadow = lora_module._shadow_linear
    calls = 0

    def fail_after_first_shadow(module, weight):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected shadow failure")
        return original_shadow(module, weight)

    with monkeypatch.context() as patch:
        patch.setattr(lora_module, "_shadow_linear", fail_after_first_shadow)
        with pytest.raises(RuntimeError, match="injected shadow failure"):
            prepare_taomate_lora(adapter_dir, transformer=model, device=torch.device("cpu"), merge=True)
    for name, weight in original_weights.items():
        assert model.get_submodule(name).weight is weight
        assert not hasattr(model.get_submodule(name), "taomate_student")
    lora, _, bank = prepare_taomate_lora(adapter_dir, transformer=model, device=torch.device("cpu"), merge=True)
    assert bank is not None
    with pytest.raises(TaoMateLoRAError, match="original linear"):
        bank.bind({})
    bank.close()
    for name, weight in original_weights.items():
        assert model.get_submodule(name).weight is weight
        assert not hasattr(model.get_submodule(name), "taomate_student")


@pytest.mark.parametrize("transpose", [False, True])
def test_merged_student_requantizes_already_fp8_base_and_restores_scales(tmp_path, transpose) -> None:
    model = _tiny_transformer()
    adapter_dir = _write_adapter(tmp_path, model, rank=3, alpha=1.5)

    class FP8Method:
        def process_weights_after_loading(self, module):
            value = module.weight.detach().float()
            scale = value.abs().amax(dim=1, keepdim=True).clamp_min(1e-5) / 240
            quantized = (value / scale).to(torch.float8_e4m3fn)
            module.weight = nn.Parameter(quantized.t().contiguous() if transpose else quantized, requires_grad=False)
            module.weight_scale = nn.Parameter(scale, requires_grad=False)

    targets = {
        target: model.get_submodule(target) for target in taomate_lora_targets(num_blocks=1, num_refiner_blocks=1)
    }
    for module in targets.values():
        module.output_size, module.input_size = module.weight.shape
        module.quant_method = FP8Method()
        module.quant_method.process_weights_after_loading(module)
    fc2 = targets["blocks.0.mlp.fc2"]
    base_weight, base_scale = fc2.weight, fc2.weight_scale
    base = base_weight.t().float() if transpose else base_weight.float()
    base = base * base_scale
    lora, _, bank = prepare_taomate_lora(adapter_dir, transformer=model, device=torch.device("cpu"), merge=True)
    assert bank is not None
    pair = lora.loras["transformer.blocks.0.mlp.fc2"]
    merged = (base + pair.lora_b.float() @ pair.lora_a.float() * 0.5).bfloat16().float()
    expected_scale = merged.abs().amax(dim=1, keepdim=True).clamp_min(1e-5) / 240
    expected_weight = (merged / expected_scale).to(torch.float8_e4m3fn)
    if transpose:
        expected_weight = expected_weight.t().contiguous()
    bank.bind({f"transformer.{target}": module for target, module in targets.items()})
    bank.activate()
    assert torch.equal(fc2.weight.float(), expected_weight.float())
    assert torch.equal(fc2.weight_scale, expected_scale)
    bank.suspend()
    assert fc2.weight is base_weight and fc2.weight_scale is base_scale
    bank.close()


def test_hold_is_released_after_its_time_bound(monkeypatch) -> None:
    """taomate_h3_hold_max_seconds: a slow prompt decision stalls the stream at most that long."""
    pipe = _hold_pipeline(monkeypatch)
    pipe._tm_hold_max_seconds = 0.05
    state = _hold_state(chunk_index=_NUM_PHASES, applied_version=1)  # request 1, no new prompt
    pipe.prepare_next_chunk(state)
    assert pipe.ar_diffusion_step_wait(state).timeout_seconds == 0.05
    pipe.prepare_next_chunk(state)
    assert state.extra["taomate_held"] is True
    with pytest.raises(_RequestStartedError):
        pipe.ar_diffusion_resume_step(state, "timeout")
    assert state.extra["taomate_held"] is False
    assert state.extra["taomate_prompt_version"] == 1


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
    with pytest.raises(_RequestStartedError):
        pipe.ar_diffusion_resume_step(state, "timeout")
    assert calls == ["She listens quietly."]
    assert tuple(state.prompt_embeds.shape) == (6, 8) and state.txt_seq_lens == [6]
    assert int(state.extra["text_tags"][0]) == 0
    # The encoding is cached: a second expiry does not re-encode.
    state2 = _hold_state(chunk_index=2 * _NUM_PHASES, applied_version=1)
    pipe.prepare_next_chunk(state2)
    with pytest.raises(_RequestStartedError):
        pipe.ar_diffusion_resume_step(state2, "timeout")
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


def test_contiguous_span_detection() -> None:
    from vllm_omni.diffusion.models.taomate_h3.pipeline import _contiguous_span

    assert _contiguous_span(torch.arange(5, 12)) == (5, 7)
    assert _contiguous_span(torch.tensor([3])) == (3, 1)
    assert _contiguous_span(torch.tensor([1, 2, 4])) is None
    assert _contiguous_span(torch.tensor([], dtype=torch.long)) is None


def test_request_mode_reusing_session_starts_with_empty_engine_kv(prepare_encode_pipeline):
    pipe, request, _ = prepare_encode_pipeline
    bound, _ = _segmented_cache()
    pipe._ar_diffusion_kv_state = bound
    pipe._tm_sessions = {}
    try:
        pipe.prepare_encode(request)
        first = pipe._tm_sessions[bound.session_id]
        first.cache.prepare_layer_views(media_tokens=5, condition_tokens=1)
        first.history.begin_commit(audio_tokens=2, video_tokens=3)
        x = torch.ones(5, 2, 4, dtype=torch.bfloat16)
        for layer in range(2):
            first.cache.assemble(layer, x[:1], x[:1], x, x)
            first.cache.stage(layer)
        first.history.commit()
        first_cache = first.cache
        assert first_cache.history_tokens == 5
        pipe.prepare_encode(request)
        second = pipe._tm_sessions[bound.session_id]
        assert second.cache is not first_cache
        assert second.cache.history_tokens == 0
        assert second.history.committed_phases == 0 and second.media_time_origin is None
        with pytest.raises(RuntimeError, match="closed"):
            first_cache.prepare_layer_views(media_tokens=5, condition_tokens=1)
    finally:
        bound.close()


def test_model_kv_requires_current_matching_ar_binding(prepare_encode_pipeline):
    pipe, request, _ = prepare_encode_pipeline
    pipe.prepare_encode(request)
    session = pipe._tm_sessions["session"]
    pipe._ar_diffusion_kv_state.session_id = "another-session"
    with pytest.raises(RuntimeError, match="outside its bound session"):
        _ = session.cache
    pipe._ar_diffusion_kv_state = None
    with pytest.raises(RuntimeError, match="requires the AR-Diffusion engine"):
        _ = session.cache
