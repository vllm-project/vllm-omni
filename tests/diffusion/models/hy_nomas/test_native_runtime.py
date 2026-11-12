# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import ast
import copy
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm_omni.diffusion.compile import regionally_compile
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.hy_nomas import pipeline_hy_nomas
from vllm_omni.diffusion.models.hy_nomas.camera import prepare_camera_inputs
from vllm_omni.diffusion.models.hy_nomas.hy_nomas_transformer import ActionModule, HyNomasTransformer
from vllm_omni.diffusion.models.hy_nomas.pixelumm import PixelUMMDCAELatentDecoder
from vllm_omni.diffusion.models.hy_nomas.pixelumm_model import DCAEVPixelUMMDecoder, GenExpertArch
from vllm_omni.diffusion.models.hy_nomas.processing_hy_nomas import prepare_camera
from vllm_omni.diffusion.models.hy_nomas.scheduler import EulerSolver

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


_SMALL_TRANSFORMER_CONFIG = {
    "in_channels": 4,
    "out_channels": 4,
    "num_layers": 1,
    "num_attention_heads": 4,
    "attention_head_dim": 8,
    "text_embed_dim": 6,
    "rope_axes_dim": [2, 2, 4],
    "camera_embed_dim": 8,
    "camera_attention_dim": 32,
    "camera_num_attention_heads": 4,
    "camera_rope_axes_dim": [2, 2, 4],
    "mlp_ratio": 4.0,
    "rope_theta": 256.0,
}


def _small_transformer(sp_size=1, num_layers=1):
    config = {**_SMALL_TRANSFORMER_CONFIG, "num_layers": num_layers}
    return HyNomasTransformer(config, sp_size=sp_size, device="cpu", dtype=torch.float32).eval().requires_grad_(False)


@torch.no_grad()
@pytest.mark.parametrize("boundaries", [None, [0, 24, 49]])
def test_camera_preset_preprocessing_accepts_extrinsics_only(boundaries):
    model = _small_transformer()
    config = pipeline_hy_nomas.build_model_config()
    camera = prepare_camera_inputs(
        "pan_left",
        num_frames=49,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        normalize_relative_to_first_frame=True,
        normalize_framewise=True,
        normalize_displacement=True,
        normalize_displacement_per_video=True,
        segment_boundaries_pixel=boundaries,
        translation_scale=1.0,
        rotation_scale=2.5,
    )
    assert camera["norm_intrinsics"] is None and camera["action_onehot"] is None
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        expected = model._preprocess_camera(
            norm_extrinsics=camera["norm_extrinsics"],
            norm_intrinsics=None,
            action_onehot=None,
            oh=12,
            ow=20,
            img_dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
    actual = prepare_camera(
        model,
        config,
        "pan_left",
        num_frames=49,
        latent_h=12,
        latent_w=20,
        boundaries=boundaries,
        device=torch.device("cpu"),
    )
    assert actual.shape == (1, 13, 64) and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_production_package_has_no_reference_imports():
    directory = Path(pipeline_hy_nomas.__file__).parent
    for path in directory.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            assert not any(
                name.split(".")[0] in ("hymm", "configs", "flash_attn", "flash_attn_interface") for name in names
            ), path


def test_import_and_real_small_modules_with_reference_imports_blocked():
    code = f"""
import importlib.abc
import runpy
import sys
class BlockReference(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('hymm', 'configs'):
            raise AssertionError('Runtime imported reference code: ' + fullname)
sys.meta_path.insert(0, BlockReference())
checks = runpy.run_path({str(Path(__file__).resolve())!r})
checks['test_native_dit_cache_and_camera']()
checks['test_real_pixelumm_stream_matches_full_decode']()
assert not any(name.split('.')[0] in ('hymm', 'configs') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@torch.no_grad()
def test_native_dit_cache_and_camera():
    torch.manual_seed(17)
    model = _small_transformer()
    cache = [{"k": torch.zeros(1, 20, 4, 8), "v": torch.zeros(1, 20, 4, 8)}]
    mask = torch.zeros(1, 20, dtype=torch.bool)
    inputs = dict(
        hidden_states=torch.randn(1, 4, 2, 1, 1),
        timestep=torch.tensor([300.0]),
        text_states=torch.randn(1, 3, 6),
        text_mask=torch.tensor([[1, 1, 0]]),
        kv_cache_list=cache,
        cache_mask_list=[mask],
    )
    original = cache[0]["k"].clone()
    first, written = model.forward_with_cache(**inputs)
    assert written == 0 and torch.equal(cache[0]["k"][:, 5:], original[:, 5:])
    camera = torch.randn(1, 2, 64)
    camera_output, written = model.forward_with_cache(**inputs, camera_embedding=camera)
    assert not torch.equal(first, camera_output)
    disabled, _ = model.forward_with_cache(**inputs, camera_embedding=camera, camera_has_mask=torch.tensor([False]))
    torch.testing.assert_close(disabled, first, rtol=0, atol=0)
    committed, written = model.forward_with_cache(**inputs, update_cache=True)
    assert written == 5 and mask[0, :5].tolist() == [True, True, True, True, False]
    torch.testing.assert_close(committed, first, rtol=0, atol=0)
    history = cache[0]["k"][:, :5].clone()
    next_output, _ = model.forward_with_cache(**inputs, cache_valid_len=5, global_frame_counter=2, cache_txt=False)
    assert torch.isfinite(next_output).all() and torch.equal(cache[0]["k"][:, :5], history)
    assert len(model.attention.masks) <= 32
    model.clear_attention_cache()
    assert not model.attention.masks and not model.camera_attention.masks


def _step_kw(f=2, h=1, w=1, txt_mask=(1, 1, 0), offs=None, wins=None, **extra):
    t_len = len(txt_mask)
    base = {"hidden_states": torch.randn(1, 4, f, h, w), "timestep": torch.tensor([300.0])}
    act = {"intra_seg_txt_offsets": offs, "intra_seg_vid_windows": wins}
    return {
        **base,
        "text_states": torch.randn(1, t_len, 6),
        "text_mask": torch.tensor([list(txt_mask)]),
        **act,
        **extra,
    }


@torch.no_grad()
def test_transformer_hoists_step_masks_and_camera_and_hits_warm_fast_path(mocker):
    from vllm_omni.diffusion.models.hy_nomas import hy_nomas_transformer, layers

    torch.manual_seed(29)
    model = _small_transformer(num_layers=3)
    cache = [{"k": torch.zeros(1, 32, 4, 8), "v": torch.zeros(1, 32, 4, 8)} for _ in range(3)]
    shared_mask = torch.zeros(1, 32, dtype=torch.bool)
    shared_mask[:, :4] = torch.tensor([[True, False, True, True]])
    inputs = _step_kw(
        offs=[[1, 2]],
        wins=[(0, 1, 0), (1, 2, 1)],
        kv_cache_list=cache,
        cache_valid_len=4,
        global_frame_counter=2,
        cache_mask_list=[shared_mask] * 3,
        camera_embedding=torch.randn(1, 2, 64),
    )
    sr, sa = mocker.spy(hy_nomas_transformer, "rotary_tables"), mocker.spy(hy_nomas_transformer, "action_ranges")
    sm, sc = mocker.spy(model.attention, "resolve_masks"), mocker.spy(model.camera_attention, "resolve_masks")

    cold_out, cold_written = model.forward_with_cache(**inputs)
    assert (cold_written, sr.call_count, sa.call_count, sm.call_count, sc.call_count) == (0, 1, 1, 1, 1)
    cached_rope_gt0 = next(iter(model._step_rope_cache.values()))

    warm_out, warm_written = model.forward_with_cache(**inputs)
    assert (warm_written, sr.call_count, sa.call_count, sm.call_count, sc.call_count) == (0, 1, 1, 1, 1)
    assert next(iter(model._step_rope_cache.values()))[0] is cached_rope_gt0[0]
    torch.testing.assert_close(warm_out, cold_out, rtol=0, atol=0)

    gfc0_cold, _ = model.forward_with_cache(**{**inputs, "global_frame_counter": 0})
    assert sr.call_count == 2
    gfc0_warm, _ = model.forward_with_cache(**{**inputs, "global_frame_counter": 0})
    assert sr.call_count == 2
    torch.testing.assert_close(gfc0_warm, gfc0_cold, rtol=0, atol=0)

    commit_out, commit_written = model.forward_with_cache(**{**inputs, "update_cache": True})
    assert (commit_written, sm.call_count, sr.call_count) == (5, 1, 2)
    torch.testing.assert_close(commit_out, cold_out, rtol=0, atol=0)
    assert shared_mask[0, 4:9].tolist() == [True, True, True, True, False]

    post_commit_warm_out, _ = model.forward_with_cache(**inputs)
    torch.testing.assert_close(post_commit_warm_out, cold_out, rtol=0, atol=0)
    assert (sm.call_count, sr.call_count) == (1, 2)

    shared_mask[0, 0] = False
    mutated_out, _ = model.forward_with_cache(**inputs)
    assert sm.call_count == 2 and not torch.equal(mutated_out, cold_out)

    model.clear_attention_cache()
    assert not model._step_attention_plan_cache and not model._step_camera_plan_cache and not model._step_rope_cache
    assert not hasattr(layers, "FlashAttentionImpl") and isinstance(
        layers.TokenRefinerBlock(32, 4).attn, layers.SDPAImpl
    )
    assert layers.TimestepEmbedder(32, device="meta")(torch.empty(1, device="meta")).device.type == "meta"

    m0, m1 = torch.ones(1, 32, dtype=torch.bool), torch.ones(1, 32, dtype=torch.bool)
    m1[0, 0] = False
    hetero_out, _ = model.forward_with_cache(**{**inputs, "cache_mask_list": [m0, m1, m0]})
    homo_out, _ = model.forward_with_cache(**{**inputs, "cache_mask_list": [m0] * 3})
    assert not torch.equal(hetero_out, homo_out)

    for valid_len in range(1, 11):
        hs = torch.randn(1, 4, 2, 1, (valid_len % 3) + 1)
        model.forward_with_cache(
            **{**inputs, "cache_valid_len": valid_len, "global_frame_counter": valid_len, "hidden_states": hs}
        )
    assert (
        max(len(c) for c in (model._step_attention_plan_cache, model._step_camera_plan_cache, model._step_rope_cache))
        <= 8
    )


@torch.no_grad()
def test_regional_compile_wraps_blocks_matches_eager_and_avoids_recompile_explosion():
    from tests.diffusion.ar_diffusion.test_segmented_kv import make_cache, make_state

    assert HyNomasTransformer._repeated_blocks == ["MMDoubleStreamBlock", "ActionModule"]
    act_meta = ActionModule(_SMALL_TRANSFORMER_CONFIG, 32, device="meta")
    assert act_meta.freqs_cos.device.type == "meta" and act_meta.freqs_sin.device.type == "meta"

    torch.manual_seed(41)
    eager = _small_transformer(num_layers=2)
    compiled = copy.deepcopy(eager)
    prev_fail = torch._dynamo.config.fail_on_recompile_limit_hit
    torch._dynamo.reset()
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    owners = [make_state(make_cache(num_layers=2, num_kv_heads=4, head_size=8)) for _ in range(2)]
    try:
        regionally_compile(compiled, backend="aot_eager", dynamic=True)
        wrapped = {n for n, m in compiled.named_modules() if "forward" in m.__dict__}
        assert wrapped == {"double_blocks.0", "double_blocks.1", "action_modules.0", "action_modules.1"}

        x_act, cam_act, cmask = torch.randn(1, 4, 32), torch.randn(1, 2, 64), torch.tensor([True])
        torch.testing.assert_close(
            compiled.action_modules[0](x_act, 2, 1, 2, cam_act, compiled.camera_attention, 1, cmask),
            eager.action_modules[0](x_act, 2, 1, 2, cam_act, eager.camera_attention, 1, cmask),
            rtol=0,
            atol=0,
        )
        tb = eager._resolve_step_rope_tables(2, 1, 1, 0, 0, x_act.device, x_act.device)
        ie, te, vc, tm_b = torch.randn(1, 2, 32), torch.randn(1, 3, 32), torch.randn(1, 32), torch.tensor([[1, 1, 0]])
        fb_m = [torch.zeros(1, 16, dtype=torch.bool) for _ in range(2)]
        fb_kv = [{"k": torch.zeros(1, 16, 4, 8), "v": torch.zeros(1, 16, 4, 8)} for _ in range(2)]
        re_i, re_t = eager.double_blocks[0](
            ie, te, vc, tb, tm_b, fb_kv[0], 0, True, True, fb_m[0], 0, eager.attention, 1, [[1]], [(0, 2, 0)], 1
        )
        rc_i, rc_t = compiled.double_blocks[0](
            ie, te, vc, tb, tm_b, fb_kv[1], 0, True, True, fb_m[1], 0, compiled.attention, 1, [[1]], [(0, 2, 0)], 1
        )
        assert torch.equal(re_i, rc_i) and torch.equal(re_t, rc_t) and torch.equal(fb_m[0], fb_m[1])

        kv_e = [{"k": torch.zeros(1, 96, 4, 8), "v": torch.zeros(1, 96, 4, 8)} for _ in range(2)]
        kv_c = [{"k": torch.zeros(1, 96, 4, 8), "v": torch.zeros(1, 96, 4, 8)} for _ in range(2)]
        masks_d = [torch.zeros(1, 96, dtype=torch.bool) for _ in range(2)]
        masks_s = [torch.zeros(1, 96, dtype=torch.bool) for _ in range(2)]
        seg_e, seg_c, hist_d = owners[0].segmented("main"), owners[1].segmented("main"), 0
        rounds = [
            (1, 1, 1, 0, True, False, [[1]], [(0, 1, 0)], True, False, False),
            (2, 1, 1, 0, True, True, [[1, 2]], [(0, 1, 0), (1, 2, 1)], True, False, False),
            (2, 1, 2, 2, True, False, [[0, 1, 2]], [(0, 1, 0), (1, 2, 1)], False, False, False),
            (3, 1, 1, 2, True, False, [[1, 2, 3]], [(0, 1, 0), (1, 2, 1), (2, 3, 2)], True, False, True),
            (5, 1, 1, 4, False, True, None, None, True, False, False),
            (2, 2, 1, 6, True, False, [None], [(0, 2, 0)], True, False, False),
            (3, 1, 2, 6, True, False, [[1]], [(0, 2, 0), (2, 3, 0)], True, False, True),
            (1, 1, 2, 8, False, False, [[1, 3]], [(0, 1, 0)], False, False, False),
            (2, 1, 1, 0, True, True, [[1, 2]], [(0, 1, 0), (1, 2, 1)], True, True, False),
            (2, 1, 2, 2, False, True, None, None, True, True, False),
        ]
        for idx, (f, h, w, gfc, c_txt, upd, offs, wins, use_cam, use_seg, hetero) in enumerate(rounds):
            t_len, cam = 3 + idx % 4, (torch.randn(1, f, 64) if use_cam else None)
            cm, tm = (torch.tensor([idx % 3 != 0]) if use_cam else None), (1,) * (t_len - 1) + (idx % 2,)
            ex = {"timestep": torch.tensor([300.0 - idx * 10]), "update_cache": upd, "global_frame_counter": gfc}
            common = _step_kw(f, h, w, tm, offs, wins, **ex, cache_txt=c_txt, camera_embedding=cam, camera_has_mask=cm)
            if use_seg:
                media = f * h * w + (t_len if c_txt else 0)
                pend = [
                    (s.prepare_layer_views(media, 0 if c_txt else t_len), s.begin_commit((media,)))[1]
                    for s in (seg_e, seg_c)
                ]
                exp, ew = eager.forward_with_cache(
                    **common,
                    cache_valid_len=seg_e.history_tokens,
                    kv_cache_state=seg_e,
                    cache_mask_list=[masks_s[0]] * 2,
                )
                act, aw = compiled.forward_with_cache(
                    **common,
                    cache_valid_len=seg_c.history_tokens,
                    kv_cache_state=seg_c,
                    cache_mask_list=[masks_s[1]] * 2,
                )
                for s, p in zip((seg_e, seg_c), pend):
                    s.stage(0), s.stage(1), s.commit(retain=(*s.handles, *p))
            else:
                if hetero:
                    m_e = [masks_d[0].clone(), masks_d[0].clone()]
                    m_e[1][0, 0] = not m_e[0][0, 0]
                    m_c = [m.clone() for m in m_e]
                else:
                    m_e, m_c = [masks_d[0]] * 2, [masks_d[1]] * 2
                exp, ew = eager.forward_with_cache(
                    **common, cache_valid_len=hist_d, kv_cache_list=kv_e, cache_mask_list=m_e
                )
                act, aw = compiled.forward_with_cache(
                    **common, cache_valid_len=hist_d, kv_cache_list=kv_c, cache_mask_list=m_c
                )
                hist_d += ew
            assert aw == ew
            torch.testing.assert_close(act, exp, rtol=0, atol=0)
        assert torch.equal(masks_d[0], masks_d[1]) and torch.equal(masks_s[0], masks_s[1])
    finally:
        for owner in owners:
            owner.close()
        torch._dynamo.config.fail_on_recompile_limit_hit = prev_fail
        torch._dynamo.reset()


@torch.no_grad()
def test_native_segmented_kv_matches_contiguous_storage():
    from tests.diffusion.ar_diffusion.test_segmented_kv import make_cache, make_state

    torch.manual_seed(23)
    model = _small_transformer()
    owner = make_state(make_cache(num_layers=1, num_kv_heads=4, head_size=8))
    state = owner.segmented("main")
    contiguous = [{"k": torch.zeros(1, 32, 4, 8), "v": torch.zeros(1, 32, 4, 8)}]
    masks = [torch.zeros(1, 32, dtype=torch.bool) for _ in range(2)]
    try:
        for turn in range(2):
            inputs = dict(
                hidden_states=torch.randn(1, 4, 2, 1, 1),
                timestep=torch.tensor([300.0]),
                text_states=torch.randn(1, 3, 6),
                text_mask=torch.tensor([[1, 1, 0]]),
                cache_txt=turn == 0,
                update_cache=True,
                cache_valid_len=state.history_tokens,
                global_frame_counter=turn * 2,
                camera_embedding=torch.randn(1, 2, 64),
            )
            media = 5 if turn == 0 else 2
            state.prepare_layer_views(media, 0 if turn == 0 else 3)
            pending = state.begin_commit((media,))
            expected, written = model.forward_with_cache(**inputs, kv_cache_list=contiguous, cache_mask_list=[masks[0]])
            actual, actual_written = model.forward_with_cache(
                **inputs, kv_cache_state=state, cache_mask_list=[masks[1]]
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert actual_written == written and torch.equal(masks[0], masks[1])
            state.stage(0)
            state.commit(retain=(*state.handles, *pending))
    finally:
        owner.close()


@pytest.mark.parametrize(
    "update",
    [
        {"num_layers": 0},
        {"num_attention_heads": 0},
        {"rope_axes_dim": [16, 56, 55]},
        {"camera_rope_axes_dim": [8, 28, 30]},
    ],
)
def test_invalid_model_geometry_fails_before_allocation(update):
    with pytest.raises(ValueError, match="dimensions|RoPE"):
        HyNomasTransformer({**_SMALL_TRANSFORMER_CONFIG, **update}, device="meta")


@torch.no_grad()
def test_real_pixelumm_stream_matches_full_decode():
    torch.manual_seed(11)
    arch = GenExpertArch(hidden_size=16, intermediate_size=32, num_layers=1, num_heads=2, num_kv_heads=1, head_dim=8)
    model = PixelUMMDCAELatentDecoder(
        DCAEVPixelUMMDecoder(arch), seed=7, temporal_window_latents=4, temporal_overlap_latents=2, decoder_tile_pixels=0
    ).eval()
    latent = torch.randn(1, 32, 7, 1, 1)
    full = model.decode(latent).sample
    session = model.create_streaming_decode_session()
    parts = []
    for start, end in ((0, 2), (2, 4), (4, 7)):
        session.final_segment = end == 7
        part = session.push(latent[:, :, start:end], pixel_frames=4 * (end - start))
        if part.numel():
            parts.append(part)
    parts.append(session.finish())
    torch.testing.assert_close(torch.cat(parts, dim=2), full, rtol=0, atol=0)
    assert session.buffer.numel() == 0 and session.pending is None


@pytest.mark.parametrize("steps", [4, 6, 8])
def test_euler_matches_explicit_released_schedule(steps):
    solver = EulerSolver()
    solver.set_timesteps(steps)
    expected = torch.linspace(1, 0, steps + 1)
    expected = 5 * expected / (1 + 4 * expected)
    assert torch.equal(solver.timesteps, expected[:-1] * 1000)
    sample = torch.tensor([1.25], dtype=torch.bfloat16)
    for index, timestep in enumerate(solver.timesteps):
        velocity = torch.tensor([0.5], dtype=torch.bfloat16)
        result = solver.step(velocity, timestep, sample).prev_sample
        reference = sample.float() + velocity.float() * (expected[index + 1] - expected[index])
        assert torch.equal(result, reference)
        sample = result


def _native_ulysses_worker(rank, rendezvous):
    from vllm_omni.diffusion.distributed import parallel_state
    from vllm_omni.diffusion.distributed.group_coordinator import SequenceParallelGroupCoordinator

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=90))
    group = None
    try:
        ulysses = dist.new_group([0, 1], backend="gloo")
        singletons = [dist.new_group([index], backend="gloo") for index in range(2)]
        group = SequenceParallelGroupCoordinator(
            [[0, 1]], rank, "gloo", ulysses_group=ulysses, ring_group=singletons[rank], allgather_group=singletons[rank]
        )
        configs = [
            OmniDiffusionConfig(dtype=torch.float32, parallel_config=DiffusionParallelConfig(ulysses_degree=size))
            for size in (1, 2)
        ]
        with pytest.MonkeyPatch.context() as patch, torch.no_grad():
            patch.setattr(parallel_state, "_SP", group)
            pipeline_hy_nomas.validate_sequence_parallelism(2)
            torch.manual_seed(17)
            with set_forward_context(omni_diffusion_config=configs[1]), set_current_diffusion_config(configs[1]):
                model = _small_transformer(2, num_layers=2)
                compiled_model = regionally_compile(copy.deepcopy(model), backend="aot_eager", dynamic=True)
                assert hasattr(compiled_model.double_blocks[0].forward, "_torchdynamo_orig_callable")
            reference = [{"k": torch.zeros(1, 32, 4, 8), "v": torch.zeros(1, 32, 4, 8)} for _ in range(2)]
            candidate = [{"k": torch.zeros(1, 32, 2, 8), "v": torch.zeros(1, 32, 2, 8)} for _ in range(2)]
            compiled_kv = [{"k": torch.zeros(1, 32, 2, 8), "v": torch.zeros(1, 32, 2, 8)} for _ in range(2)]
            masks = [torch.zeros(1, 32, dtype=torch.bool) for _ in range(3)]
            history = 0
            for turn in range(2):
                inputs = dict(
                    hidden_states=torch.randn(1, 4, 2, 1, 2),
                    timestep=torch.tensor([300.0]),
                    text_states=torch.randn(1, 3, 6),
                    text_mask=torch.tensor([[1, 1, 0]]),
                    update_cache=True,
                    cache_txt=turn == 0,
                    cache_valid_len=history,
                    global_frame_counter=turn * 2,
                    camera_embedding=torch.randn(1, 2, 64),
                )
                model.sp_size = 1
                with set_forward_context(omni_diffusion_config=configs[0]):
                    expected, written = model.forward_with_cache(
                        **inputs, kv_cache_list=reference, cache_mask_list=[masks[0]] * 2
                    )
                model.sp_size = 2
                with set_forward_context(omni_diffusion_config=configs[1]):
                    actual, actual_written = model.forward_with_cache(
                        **inputs, kv_cache_list=candidate, cache_mask_list=[masks[1]] * 2
                    )
                    compiled_out, compiled_written = compiled_model.forward_with_cache(
                        **inputs, kv_cache_list=compiled_kv, cache_mask_list=[masks[2]] * 2
                    )
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(compiled_out, expected, rtol=1e-5, atol=1e-6)
                assert actual_written == written and compiled_written == written
                assert torch.equal(masks[0], masks[1]) and torch.equal(masks[0], masks[2])
                history += written
                for layer in range(2):
                    torch.testing.assert_close(
                        candidate[layer]["k"][:, :history],
                        reference[layer]["k"][:, :history, rank * 2 : rank * 2 + 2],
                        rtol=1e-5,
                        atol=1e-6,
                    )
            assert dist.is_initialized() and dist.get_world_size() == 2
    finally:
        if group is not None:
            group.destroy()
        dist.destroy_process_group()


@pytest.mark.parallel
def test_native_model_uses_public_ulysses_with_real_gloo(tmp_path):
    mp.spawn(_native_ulysses_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)
