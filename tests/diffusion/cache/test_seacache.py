# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the SeaCache backend: SEA filter math, hook schedule, and
per-model extractor wiring (FLUX.1, Qwen-Image, FLUX.2-klein)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_omni.diffusion.cache.seacache.backend import SeaCacheBackend
from vllm_omni.diffusion.cache.seacache.config import SeaCacheConfig
from vllm_omni.diffusion.cache.seacache.filter import (
    ab_from_sigma,
    apply_sea_filter,
    rel_l1,
    sea_filter_response,
)
from vllm_omni.diffusion.cache.seacache.hook import SeaCacheHook, apply_seacache_hook
from vllm_omni.diffusion.cache.teacache.extractors import (
    extract_flux2_klein_context,
    extract_flux_context,
    extract_qwen_context,
)
from vllm_omni.diffusion.data import DiffusionCacheConfig
from vllm_omni.diffusion.models.flux.flux_transformer import FluxTransformer2DModel

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Descending sigmas of a 6-step trajectory.
_SIGMAS = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]


@pytest.fixture(scope="function", autouse=True)
def _cpu_runtime():
    """Single-process distributed env and CPU dispatch for tiny transformers."""
    import os

    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29503")
    init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://")
    initialize_model_parallel()
    with (
        patch(
            "vllm_omni.diffusion.cache.seacache.hook.get_classifier_free_guidance_world_size",
            return_value=1,
        ),
        patch(
            "vllm_omni.diffusion.cache.seacache.hook.get_classifier_free_guidance_rank",
            return_value=0,
        ),
        set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))),
        patch(
            "vllm_omni.diffusion.attention.layer.get_attn_backend_for_role",
            return_value=(SDPABackend, None),
        ),
    ):
        yield
    cleanup_dist_env_and_memory()


def _make_flux_module():
    torch.manual_seed(1234)
    module = FluxTransformer2DModel(
        num_layers=2,
        num_single_layers=2,
        num_attention_heads=2,
        attention_head_dim=16,
        joint_attention_dim=32,
        pooled_projection_dim=16,
        axes_dims_rope=(4, 4, 8),
    )
    # vLLM linears allocate with torch.empty (filled at checkpoint load) and
    # CustomOp bakes forward_cuda at construction; initialize the weights and
    # force native RoPE so the forward is finite on CPU.
    from vllm_omni.diffusion.layers.rope import RotaryEmbedding

    for param in module.parameters():
        if param.dim() >= 2:
            torch.nn.init.normal_(param, std=0.02)
        else:
            torch.nn.init.zeros_(param)
    for sub in module.modules():
        if isinstance(sub, RotaryEmbedding):
            sub._forward_method = sub.forward_native
    return module


def _make_inputs(grid=(4, 4), txt_len=8, seed=0, sigma=1.0):
    g = torch.Generator().manual_seed(seed)
    h, w = grid
    ids = torch.zeros(h, w, 3)
    ids[..., 1] += torch.arange(h)[:, None]
    ids[..., 2] += torch.arange(w)[None, :]
    return {
        "hidden_states": torch.randn(1, h * w, 64, generator=g),
        "encoder_hidden_states": torch.randn(1, txt_len, 32, generator=g),
        "pooled_projections": torch.randn(1, 16, generator=g),
        "timestep": torch.tensor([sigma]),
        "img_ids": ids.reshape(h * w, 3),
        "txt_ids": torch.zeros(txt_len, 3),
        "guidance": torch.tensor([3.5]),
        "return_dict": False,
    }


def _count_block_runs(module):
    runs = {"n": 0}
    orig = module.transformer_blocks[0].forward

    def counted(*args, **kwargs):
        runs["n"] += 1
        return orig(*args, **kwargs)

    module.transformer_blocks[0].forward = counted
    return runs


def _get_hook(module):
    return module._hook_registry.get_hook(SeaCacheHook._HOOK_NAME)


def _get_state(hook, branch="positive"):
    hook.state_manager.set_context(f"seacache_{branch}")
    return hook.state_manager.get_state()


def _hooked_module(thresh, num_steps=6):
    module = _make_flux_module()
    apply_seacache_hook(module, SeaCacheConfig(sea_thresh=thresh))
    _get_hook(module).num_inference_steps = num_steps
    return module


def _response(shape, a, b, norm_mode="mean", dims=(-2, -3)):
    return sea_filter_response(
        shape=shape,
        dims=dims,
        a=a,
        b=b,
        power_exp=2.0,
        norm_mode=norm_mode,
        device=torch.device("cpu"),
    )


def _mock_pipeline(transformer_cls="FluxTransformer2DModel", pipeline_cls="FluxPipeline"):
    transformer = type(transformer_cls, (), {})()
    return type(pipeline_cls, (), {"transformer": transformer})()


class TestSeaFilter:
    @pytest.mark.parametrize("sigma", [0.9, 0.5, 0.1])
    def test_normalization_pins_the_filter_gain(self, sigma):
        a, b = ab_from_sigma(sigma)
        for norm_mode, stat in (("mean", "mean"), ("peak", "amax")):
            response = _response((1, 16, 16, 8), a, b, norm_mode=norm_mode)
            assert getattr(response, stat)().item() == pytest.approx(1.0, abs=1e-5)

    def test_passband_widens_as_sigma_falls(self):
        """Guards against an inverted filter, e.g. swapped a/b."""
        gains = []
        for sigma in [0.95, 0.75, 0.55, 0.35, 0.15, 0.05]:
            a, b = ab_from_sigma(sigma)
            # fftfreq(16) reaches its maximum |f| = 0.5 at index 8.
            gains.append(_response((16, 16), a, b, dims=(-2, -1))[8, 8].item())
        assert all(x < y for x, y in zip(gains, gains[1:]))

    def test_terminal_sigma_and_rel_l1(self):
        # FLUX's first sigma is exactly 1.0; the clamp keeps the filter finite.
        a, b = ab_from_sigma(1.0)
        assert a > 0 and b < 1.0
        assert torch.isfinite(apply_sea_filter(torch.randn(1, 8, 8, 4), a=a, b=b)).all()
        x = torch.randn(1, 8, 8, 4, dtype=torch.bfloat16)
        assert rel_l1(x, x) == 0.0
        assert rel_l1(2 * x, x) == pytest.approx(1.0, rel=1e-3)


class TestSeaCacheHook:
    """Schedule and residual semantics on a tiny FLUX transformer."""

    def test_zero_threshold_is_bit_identical(self):
        """sea_thresh=0 is the uncached control: every step runs and the output
        matches the un-hooked forward bit for bit."""
        reference = _make_flux_module()
        hooked = _hooked_module(0.0)
        runs = _count_block_runs(hooked)
        for step, sigma in enumerate(_SIGMAS):
            inputs = _make_inputs(seed=100 + step, sigma=sigma)
            assert torch.equal(reference(**inputs)[0], hooked(**inputs)[0])
        assert runs["n"] == len(_SIGMAS)

    def test_large_threshold_runs_first_and_last_only(self):
        module = _hooked_module(1e9)
        runs = _count_block_runs(module)
        per_step = []
        for step, sigma in enumerate(_SIGMAS):
            before = runs["n"]
            module(**_make_inputs(seed=200 + step, sigma=sigma))
            per_step.append(runs["n"] - before)
        assert per_step == [1, 0, 0, 0, 0, 1]

    def test_skipped_step_reuses_the_cached_residual(self):
        module = _hooked_module(1e9)
        module(**_make_inputs(seed=500, sigma=1.0))
        module(**_make_inputs(seed=501, sigma=0.8))
        state = _get_state(_get_hook(module))
        assert state.previous_residual is not None

        skip_inputs = _make_inputs(seed=502, sigma=0.6)
        actual = module(**skip_inputs)
        ctx = extract_flux_context(module, **skip_inputs)
        expected = ctx.postprocess(ctx.hidden_states + state.previous_residual)
        assert torch.equal(expected[0], actual[0])

    def test_cfg_branches_accumulate_independently(self):
        module = _hooked_module(1e9, num_steps=4)
        module.do_true_cfg = True
        runs = _count_block_runs(module)
        hook = _get_hook(module)
        for step in range(4):
            sigma = 1.0 - 0.25 * step
            # Positive then negative, mirroring sequential CFG.
            module(**_make_inputs(seed=800 + 2 * step, sigma=sigma))
            module(**_make_inputs(seed=801 + 2 * step, sigma=sigma))
        assert runs["n"] == 4
        for branch in ("positive", "negative"):
            state = _get_state(hook, branch)
            assert state.real_steps == 2
            assert state.skipped_steps == 2


class TestSeaCacheExtractors:
    def test_flux_extractor_provides_sigma_and_grid(self):
        ctx = extract_flux_context(_make_flux_module(), **_make_inputs(sigma=0.37))
        assert ctx.sigma == pytest.approx(0.37)
        assert ctx.grid_hw == (4, 4)

    def test_qwen_extractor_slices_to_the_noise_segment(self):
        """Edit pipelines concatenate [noise tokens; condition tokens]; the
        extractor must expose the noise grid so the SEA filter never sees the
        step-constant condition segment."""
        from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
            QwenImageTransformer2DModel,
        )

        od_config = MagicMock()
        od_config.parallel_config.sequence_parallel_size = 1
        torch.manual_seed(4321)
        module = QwenImageTransformer2DModel(
            od_config,
            num_layers=2,
            num_attention_heads=2,
            attention_head_dim=16,
            joint_attention_dim=32,
            zero_cond_t=True,
        )
        g = torch.Generator().manual_seed(7)
        # 16 noise tokens (4x4 grid) + 8 condition tokens; zero_cond_t doubles
        # the timestep to [t, 0], broadcasting the modulation to batch 2.
        ctx = extract_qwen_context(
            module,
            hidden_states=torch.randn(1, 24, 64, generator=g),
            encoder_hidden_states=torch.randn(1, 8, 32, generator=g),
            encoder_hidden_states_mask=torch.ones(1, 8, dtype=torch.bool),
            timestep=torch.tensor([0.37]),
            img_shapes=[[(1, 4, 4), (1, 2, 4)]],
            txt_seq_lens=[8],
        )
        assert ctx.sigma == pytest.approx(0.37)
        assert ctx.grid_hw == (4, 4)
        assert ctx.grid_seq_len == 16
        assert tuple(SeaCacheHook._decision_feature(ctx).shape) == (2, 16, 32)

    def test_klein_extractor_excludes_condition_rows(self):
        """Klein appends condition rows with T >= 10 after the noise grid
        (the rows with T == 0)."""
        from vllm_omni.diffusion.models.flux2_klein.flux2_klein_transformer import (
            Flux2Transformer2DModel,
        )

        torch.manual_seed(4321)
        module = Flux2Transformer2DModel(
            num_layers=2,
            num_single_layers=2,
            num_attention_heads=2,
            attention_head_dim=16,
            joint_attention_dim=32,
            axes_dims_rope=(4, 4, 4, 4),
            guidance_embeds=False,
        )

        def rows(t_val, count):
            return torch.cartesian_prod(
                torch.full((1,), t_val, dtype=torch.int64),
                torch.arange(4),
                torch.arange(4),
                torch.zeros(1, dtype=torch.int64),
            )[:count]

        g = torch.Generator().manual_seed(13)
        ctx = extract_flux2_klein_context(
            module,
            hidden_states=torch.randn(1, 24, 128, generator=g),
            encoder_hidden_states=torch.randn(1, 8, 32, generator=g),
            timestep=torch.tensor([0.42]),
            img_ids=torch.cat([rows(0, 16), rows(10, 8)]),
            txt_ids=torch.zeros(8, 4),
            guidance=None,
        )
        assert ctx.sigma == pytest.approx(0.42)
        assert ctx.grid_hw == (4, 4)
        assert ctx.grid_seq_len == 16
        assert SeaCacheHook._decision_feature(ctx).shape[1] == 16

    def test_run_full_stack_prefers_the_flux2_full_runner(self):
        """Klein's run_transformer_blocks covers only the dual-stream blocks."""
        hidden, encoder = torch.zeros(1, 4, 8), torch.zeros(1, 2, 8)
        calls = []

        def run_blocks():
            calls.append("blocks")
            return hidden, encoder

        def run_full(h, c):
            calls.append("full")
            return h + 1.0, c

        ctx = SimpleNamespace(
            hidden_states=hidden,
            encoder_hidden_states=encoder,
            run_transformer_blocks=run_blocks,
            extra_states={"run_flux2_full_transformer_with_single": run_full},
        )
        SeaCacheHook._run_full_stack(ctx)
        assert calls == ["full"]
        assert torch.equal(ctx.hidden_states, hidden + 1.0)


class TestSeaCacheBackend:
    @pytest.mark.parametrize(
        ("kwargs", "transformer_cls", "match"),
        [
            ({}, "ZImageTransformer2DModel", "does not support transformer"),
            ({"sea_thresh": -0.1}, "FluxTransformer2DModel", "non-negative"),
            ({"sea_thresh": float("nan")}, "FluxTransformer2DModel", "finite"),
            ({"sea_norm_mode": "lowpass"}, "FluxTransformer2DModel", "sea_norm_mode"),
        ],
    )
    @patch("vllm_omni.diffusion.cache.seacache.backend.apply_seacache_hook")
    def test_enable_rejects_invalid_config_or_transformer(self, mock_apply_hook, kwargs, transformer_cls, match):
        backend = SeaCacheBackend(DiffusionCacheConfig(**kwargs))
        with pytest.raises(ValueError, match=match):
            backend.enable(_mock_pipeline(transformer_cls))
        mock_apply_hook.assert_not_called()

    def test_refresh_resets_state_and_keeps_num_steps(self):
        module = _make_flux_module()
        pipeline = SimpleNamespace(transformer=module)
        backend = SeaCacheBackend(DiffusionCacheConfig(sea_thresh=0.3))
        backend.enable(pipeline)
        hook = _get_hook(module)
        hook.num_inference_steps = 28
        _get_state(hook).skipped_steps = 5

        backend.refresh(pipeline, num_inference_steps=50)

        assert hook.num_inference_steps == 50
        assert _get_state(hook).skipped_steps == 0

    @patch("vllm_omni.diffusion.cache.seacache.backend.apply_seacache_hook")
    def test_enable_dispatches_klein_pipeline_by_alias(self, mock_apply_hook):
        """Klein shares the Flux2Transformer2DModel class name with full Flux2,
        so it dispatches by pipeline class to the extractor alias."""
        backend = SeaCacheBackend(DiffusionCacheConfig(sea_thresh=0.3))
        backend.enable(_mock_pipeline("Flux2Transformer2DModel", pipeline_cls="Flux2KleinPipeline"))
        mock_apply_hook.assert_called_once()
        assert mock_apply_hook.call_args.args[1].transformer_type == "Flux2Klein"
