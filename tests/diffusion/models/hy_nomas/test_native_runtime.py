# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import ast
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.hy_nomas import pipeline_hy_nomas
from vllm_omni.diffusion.models.hy_nomas.camera import prepare_camera_inputs
from vllm_omni.diffusion.models.hy_nomas.hy_nomas_transformer import HyNomasTransformer
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
            reference = [{"k": torch.zeros(1, 32, 4, 8), "v": torch.zeros(1, 32, 4, 8)} for _ in range(2)]
            candidate = [{"k": torch.zeros(1, 32, 2, 8), "v": torch.zeros(1, 32, 2, 8)} for _ in range(2)]
            masks = [torch.zeros(1, 32, dtype=torch.bool) for _ in range(2)]
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
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                assert actual_written == written and torch.equal(masks[0], masks[1])
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
