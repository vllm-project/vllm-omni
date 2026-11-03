# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Model specific tests for CacheDiT enablement.
"""

import ast
import logging
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from cache_dit.caching.cache_blocks.pattern_0_1_2 import CachedBlocks_Pattern_0_1_2
from vllm.distributed import parallel_state

import vllm_omni.diffusion.cache.cachedit as cd_backend
import vllm_omni.diffusion.cache.cachedit.model_specific as cd_model_specific
from vllm_omni.diffusion.cache.cachedit import CacheDiTAdapterConfig, CacheDiTBackend, cache_summary
from vllm_omni.diffusion.cache.cachedit.config import (
    VALIDATED_TAYLORSEER_MAX_CONTINUOUS_CACHED_STEPS,
)
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import AttentionConfig, DiffusionCacheConfig
from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3VFMTransformer
from vllm_omni.diffusion.models.helios.helios_transformer import HeliosTransformer3DModel
from vllm_omni.diffusion.models.longcat_image.longcat_image_transformer import LongCatImageTransformer2DModel
from vllm_omni.diffusion.models.ltx2.ltx2_transformer import LTX2VideoTransformer3DModel
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import build_wan_scheduler
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SEPARATE_CFG_TRANSFORMERS = [
    HeliosTransformer3DModel,
    LongCatImageTransformer2DModel,
    Cosmos3VFMTransformer,
]

SAMPLE_CACHE_CONFIG = DiffusionCacheConfig()


@contextmanager
def _force_torch_sdpa():
    """Pin TORCH_SDPA so CPU shape tests do not pick CUDA-only backends (FA3)."""
    od_config = SimpleNamespace(
        diffusion_attention_config=AttentionConfig(default="TORCH_SDPA"),
        parallel_config=SimpleNamespace(ring_degree=1),
    )
    with set_current_diffusion_config(od_config):
        yield


def test_custom_cache_dit_enablers_are_registered_explicitly():
    expected_enablers = {
        "Wan22Pipeline": cd_model_specific.enable_cache_for_wan22,
        "Wan22I2VPipeline": cd_model_specific.enable_cache_for_wan22,
        "Wan22TI2VPipeline": cd_model_specific.enable_cache_for_wan22,
        "Wan22VACEPipeline": cd_model_specific.enable_cache_for_wan22,
        "Wan22S2VPipeline": cd_model_specific.enable_cache_for_wan22_s2v,
        "Cosmos3OmniDiffusersPipeline": cd_model_specific.enable_cache_for_cosmos3,
        "Cosmos3OmniPipeline": cd_model_specific.enable_cache_for_cosmos3,
        "Krea2Pipeline": cd_model_specific.enable_cache_for_krea2,
        "Kandinsky6TI2VAPipeline": cd_model_specific.enable_cache_for_kandinsky6,
        "Magi2Pipeline": cd_model_specific.enable_cache_for_magi2,
        "MammothModa2DiTPipeline": cd_model_specific.enable_cache_for_mammothmoda2,
    }

    with patch.dict(cd_backend.CUSTOM_DIT_ENABLERS, {}, clear=True):
        cd_model_specific.register_custom_dit_enablers()
        assert cd_backend.CUSTOM_DIT_ENABLERS == expected_enablers


@pytest.fixture()
def init_fake_tp_group(mocker):
    """Provide a fake TP group so vLLM linear layers can be instantiated."""
    mock_tp = mocker.MagicMock()
    mock_tp.world_size = 1
    mock_tp.rank_in_group = 0
    old = parallel_state._TP
    parallel_state._TP = mock_tp
    yield
    parallel_state._TP = old


def test_wan22_vace_uses_wan22_custom_cache_dit_enabler():
    assert cd_backend.CUSTOM_DIT_ENABLERS["Wan22VACEPipeline"] is cd_model_specific.enable_cache_for_wan22


@pytest.mark.parametrize(
    "pipeline_name",
    ["Cosmos3OmniDiffusersPipeline", "Cosmos3OmniPipeline"],
)
def test_cosmos3_aliases_use_cosmos3_custom_cache_dit_enabler(pipeline_name: str):
    assert cd_backend.CUSTOM_DIT_ENABLERS[pipeline_name] is cd_model_specific.enable_cache_for_cosmos3


@patch("vllm_omni.diffusion.cache.cachedit.model_specific.enable_cache_for_dit")
@patch("vllm_omni.diffusion.cache.cachedit.model_specific.BlockAdapter")
def test_magi2_cache_dit_targets_only_nested_repeated_layers(mock_block_adapter, mock_enable_cache):
    pipeline = Mock()
    transformer_block = pipeline.transformer.block
    layers = torch.nn.ModuleList([torch.nn.Identity()])
    transformer_block.layers = layers
    adapter = mock_block_adapter.return_value
    refresh = Mock()
    mock_enable_cache.return_value = refresh

    result = cd_model_specific.enable_cache_for_magi2(pipeline, SAMPLE_CACHE_CONFIG)

    mock_block_adapter.assert_called_once()
    adapter_kwargs = mock_block_adapter.call_args.kwargs
    assert adapter_kwargs["transformer"] is transformer_block
    assert adapter_kwargs["blocks"] == [layers]
    assert adapter_kwargs["has_separate_cfg"] is False
    assert adapter_kwargs["check_forward_pattern"] is True
    assert result.refresh is refresh
    assert result.targets == (adapter,)
    get_transformer = mock_enable_cache.call_args.kwargs["get_pipeline_transformer"]
    assert get_transformer(pipeline) is transformer_block


@patch("vllm_omni.diffusion.cache.cachedit.model_specific.enable_cache_for_dit")
@patch("vllm_omni.diffusion.cache.cachedit.model_specific.BlockAdapter")
def test_mammothmoda2_cache_dit_targets_only_main_layers(mock_block_adapter, mock_enable_cache):
    pipeline = Mock()
    transformer = pipeline.gen_transformer
    layers = torch.nn.ModuleList([torch.nn.Identity()])
    transformer.layers = layers
    adapter = mock_block_adapter.return_value
    refresh = Mock()
    mock_enable_cache.return_value = refresh

    result = cd_model_specific.enable_cache_for_mammothmoda2(pipeline, SAMPLE_CACHE_CONFIG)

    mock_block_adapter.assert_called_once()
    adapter_kwargs = mock_block_adapter.call_args.kwargs
    assert adapter_kwargs["transformer"] is transformer
    assert adapter_kwargs["blocks"] == [layers]
    assert adapter_kwargs["has_separate_cfg"] is True
    assert adapter_kwargs["check_forward_pattern"] is True
    assert result.refresh is refresh
    assert result.targets == (adapter,)
    get_transformer = mock_enable_cache.call_args.kwargs["get_pipeline_transformer"]
    assert get_transformer(pipeline) is transformer
    # Sequential-CFG parity is mandatory while cache-dit is installed.
    assert pipeline._cache_dit_requires_paired_cfg is True


@patch("vllm_omni.diffusion.cache.cachedit.backend.cache_dit.summary")
@patch("vllm_omni.diffusion.cache.cachedit.backend.BlockAdapter.is_cached", return_value=True)
def test_cache_summary_uses_custom_nested_targets(mock_is_cached, mock_summary):
    target = object()
    pipeline = Mock(_cache_dit_targets=(target,))

    cd_backend.cache_summary(pipeline, details=True)

    mock_is_cached.assert_called_once_with(target)
    mock_summary.assert_called_once_with(target, details=True)


def test_cachedit_public_api_is_explicit():
    assert set(cd_backend.__all__) == {
        "BagelCachedAdapter",
        "CUSTOM_DIT_ENABLERS",
        "CacheDiTAdapterConfig",
        "CacheDiTBackend",
        "CacheDiTEnableResult",
        "CacheDiTConfig",
        "CacheDiTRequestSpec",
        "RequestScopedCacheDiTRuntime",
        "SensenovaCachedAdapter",
        "cache_summary",
        "enable_cache_for_dit",
    }
    assert not hasattr(cd_backend, "enable_cache_for_wan22")
    assert not hasattr(cd_backend, "enable_cache_for_wan22_s2v")


def test_cachedit_consumers_use_package_api():
    cache_dir = Path(cd_backend.__file__).resolve().parents[1]
    package_root = cache_dir.parents[1]
    legacy_module = "vllm_omni.diffusion.cache.cache_dit_backend"
    internal_prefix = "vllm_omni.diffusion.cache.cachedit."

    invalid_imports = []
    for source_path in package_root.rglob("*.py"):
        if source_path.is_relative_to(cache_dir):
            continue
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                modules.append(node.module)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)

            for module in modules:
                if module == legacy_module or module.startswith(internal_prefix):
                    invalid_imports.append(
                        f"{source_path.relative_to(package_root)}:{getattr(node, 'lineno', '?')}: {module}"
                    )

    assert not invalid_imports, "Cache-DiT consumers must use the package API:\n" + "\n".join(invalid_imports)


@pytest.mark.parametrize("transformer_model", SEPARATE_CFG_TRANSFORMERS)
def test_cache_dit_configs_have_separate_cfg(transformer_model):
    """Check models with separate CFG set has_separate_cfg=True in their configs."""
    assert hasattr(transformer_model, "_cache_dit_adapter_config")
    assert isinstance(transformer_model._cache_dit_adapter_config, CacheDiTAdapterConfig)
    assert transformer_model._cache_dit_adapter_config.has_separate_cfg is True


def test_ltx2_cache_dit_uses_one_forward_per_denoise_step():
    """LTX batches all guidance passes into one Transformer invocation."""
    adapter_config = LTX2VideoTransformer3DModel._cache_dit_adapter_config

    assert adapter_config.has_separate_cfg is False


@patch("vllm_omni.diffusion.cache.cachedit.model_specific.BlockAdapter")
@patch("vllm_omni.diffusion.cache.cachedit.model_specific.cache_dit")
def test_separate_wan22_custom_enabler_has_separate_cfg(mock_cache_dit, mock_block_adapter):
    """Ensure that Wan22, which has a custom enabler, setts custom CFG correctly."""
    mock_pipeline = Mock()
    cd_model_specific.enable_cache_for_wan22(mock_pipeline, SAMPLE_CACHE_CONFIG)

    mock_cache_dit.enable_cache.assert_called_once()
    adapter_kwargs = mock_block_adapter.call_args.kwargs
    assert adapter_kwargs["has_separate_cfg"] is True


@contextmanager
def _cfg_parallel(world_size: int | None):
    """Pretend the CFG-parallel group has ``world_size`` ranks (``None``: parallel state not initialized)."""
    with (
        patch(
            "vllm_omni.diffusion.distributed.parallel_state.model_parallel_is_initialized",
            return_value=world_size is not None,
        ),
        patch(
            "vllm_omni.diffusion.distributed.parallel_state.get_classifier_free_guidance_world_size",
            return_value=world_size or 1,
        ),
    ):
        yield


@pytest.mark.parametrize(
    ("declared", "cfg_world_size", "expected"),
    [
        (True, None, True),  # no parallel state (single process): keep the declared value
        (True, 1, True),  # both CFG branches run on this rank
        (True, 2, False),  # CFG parallel: one branch, so one transformer call, per rank and step
        (False, 1, False),
        (False, 2, False),
    ],
)
def test_resolve_has_separate_cfg(declared, cfg_world_size, expected):
    with _cfg_parallel(cfg_world_size):
        assert cd_backend.backend.resolve_has_separate_cfg(declared) is expected


@pytest.mark.parametrize("has_transformer_2", [False, True])
@patch("vllm_omni.diffusion.cache.cachedit.model_specific.BlockAdapter")
@patch("vllm_omni.diffusion.cache.cachedit.model_specific.cache_dit")
def test_wan22_custom_enabler_drops_separate_cfg_under_cfg_parallel(
    mock_cache_dit,
    mock_block_adapter,
    has_transformer_2,
):
    """Under CFG parallel each rank calls the transformer once per step, so Wan2.2 must not pair calls."""
    mock_pipeline = Mock()
    if not has_transformer_2:
        mock_pipeline.transformer_2 = None
    with _cfg_parallel(2):
        cd_model_specific.enable_cache_for_wan22(mock_pipeline, SAMPLE_CACHE_CONFIG)

    mock_cache_dit.enable_cache.assert_called_once()
    assert mock_block_adapter.call_args.kwargs["has_separate_cfg"] is False


@patch("vllm_omni.diffusion.cache.cachedit.backend.cache_dit")
def test_enable_cache_for_dit_drops_separate_cfg_under_cfg_parallel(mock_cache_dit):
    """The generic path (model-declared adapters, Krea2, MammothModa2, Cosmos3) resolves the flag too."""
    block_adapter = SimpleNamespace(has_separate_cfg=True)
    with _cfg_parallel(2):
        cd_backend.enable_cache_for_dit(
            Mock(),
            SAMPLE_CACHE_CONFIG,
            block_adapter=block_adapter,
            get_pipeline_transformer=lambda pipeline: Mock(),
        )

    assert block_adapter.has_separate_cfg is False
    assert mock_cache_dit.enable_cache.call_args.args[0] is block_adapter


@pytest.mark.parametrize(("cfg_world_size", "expected"), [(None, None), (1, None), (2, False)])
@patch("vllm_omni.diffusion.cache.cachedit.backend.cache_dit")
def test_enable_cache_for_dit_builtin_adapter_drops_separate_cfg_under_cfg_parallel(
    mock_cache_dit,
    cfg_world_size,
    expected,
):
    """Without a declared adapter, Cache-DiT's registry adapter (e.g. QwenImage) would pair calls."""
    transformer = Mock()
    with _cfg_parallel(cfg_world_size):
        cd_backend.enable_cache_for_dit(
            Mock(), SAMPLE_CACHE_CONFIG, get_pipeline_transformer=lambda pipeline: transformer
        )

    assert mock_cache_dit.enable_cache.call_args.args[0] is transformer
    assert mock_cache_dit.enable_cache.call_args.kwargs["cache_config"].enable_separate_cfg is expected


@pytest.mark.parametrize(("cfg_world_size", "expected"), [(None, False), (1, False), (2, True)])
@patch("vllm_omni.diffusion.cache.cachedit.backend.cache_dit")
def test_backend_marks_pipeline_one_call_per_cfg_rank(mock_cache_dit, cfg_world_size, expected):
    """CFGParallelMixin reads the marker to reject multi-branch topologies with two calls on one rank."""
    pipeline_cls = type("FakeCfgParallelPipeline", (), {})
    pipeline = pipeline_cls()
    pipeline.transformer = Mock()
    backend = CacheDiTBackend(DiffusionCacheConfig())
    with (
        patch.dict(cd_backend.backend.CUSTOM_DIT_ENABLERS, {pipeline_cls.__name__: lambda *_: Mock()}),
        _cfg_parallel(cfg_world_size),
    ):
        backend.enable(pipeline)
    assert pipeline._cache_dit_one_call_per_cfg_rank is expected

    backend.disable(pipeline)
    assert pipeline._cache_dit_one_call_per_cfg_rank is False


@pytest.mark.parametrize("has_transformer_2", [False, True])
@patch("vllm_omni.diffusion.cache.cachedit.model_specific.BlockAdapter")
@patch("vllm_omni.diffusion.cache.cachedit.model_specific.cache_dit")
def test_wan22_custom_enabler_passes_taylorseer_calibrator(
    mock_cache_dit,
    mock_block_adapter,
    has_transformer_2,
):
    mock_pipeline = Mock()
    mock_pipeline.transformer.blocks = [Mock()]
    if has_transformer_2:
        mock_pipeline.transformer_2.blocks = [Mock()]
    else:
        mock_pipeline.transformer_2 = None
    cache_config = DiffusionCacheConfig(enable_taylorseer=True, taylorseer_order=1)

    cd_model_specific.enable_cache_for_wan22(mock_pipeline, cache_config)

    enable_cache_kwargs = mock_cache_dit.enable_cache.call_args.kwargs
    calibrator_config = enable_cache_kwargs["calibrator_config"]
    assert calibrator_config is not None
    assert calibrator_config.taylorseer_order == 1

    adapter_kwargs = mock_block_adapter.call_args.kwargs
    for modifier in adapter_kwargs["params_modifiers"]:
        assert modifier._context_kwargs["calibrator_config"] is calibrator_config


@patch("vllm_omni.diffusion.cache.cachedit.backend.cache_dit")
@patch("vllm_omni.diffusion.cache.cachedit.model_specific.BlockAdapter")
@patch("vllm_omni.diffusion.cache.cachedit.model_specific.cache_dit")
def test_wan22_custom_enabler_low_noise_expert_only(mock_cache_dit, mock_block_adapter, mock_backend_cache_dit):
    """boundary_ratio=1.0 loads only transformer_2; cache and refresh must target it."""
    transformer_2 = Mock()
    transformer_2.blocks = [Mock()]
    pipeline = SimpleNamespace(transformer=None, transformer_2=transformer_2)

    refresh = cd_model_specific.enable_cache_for_wan22(pipeline, SAMPLE_CACHE_CONFIG)

    mock_cache_dit.enable_cache.assert_called_once()
    adapter_kwargs = mock_block_adapter.call_args.kwargs
    assert adapter_kwargs["transformer"] is transformer_2
    assert adapter_kwargs["blocks"] == [transformer_2.blocks]

    refresh(pipeline, 40, False)
    mock_backend_cache_dit.refresh_context.assert_called_once()
    assert mock_backend_cache_dit.refresh_context.call_args.args[0] is transformer_2
    assert mock_backend_cache_dit.refresh_context.call_args.kwargs["num_inference_steps"] == 40

    # forward()'s per-schedule refresh must hit the same expert.
    mock_backend_cache_dit.refresh_context.reset_mock()
    pipeline._cache_dit_refresh_for_schedule(pipeline, torch.arange(30), None, False)
    mock_backend_cache_dit.refresh_context.assert_called_once()
    assert mock_backend_cache_dit.refresh_context.call_args.args[0] is transformer_2
    assert mock_backend_cache_dit.refresh_context.call_args.kwargs["num_inference_steps"] == 30


def test_wan22_custom_enabler_without_any_transformer_raises():
    pipeline = SimpleNamespace(transformer=None, transformer_2=None)

    with pytest.raises(ValueError, match="no loaded transformer"):
        cd_model_specific.enable_cache_for_wan22(pipeline, SAMPLE_CACHE_CONFIG)


@pytest.mark.parametrize(
    ("engine_boundary", "request_boundary", "expected_boundary"),
    [
        (None, None, 0.875),  # omitted everywhere: default 0.875
        (None, 0.9, 0.9),  # request override when the engine has none
        (0.875, 0.9, 0.875),  # engine override wins over the request
    ],
)
def test_resolve_wan_boundary_ratio(engine_boundary, request_boundary, expected_boundary):
    from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import resolve_wan_boundary_ratio

    sampling_params = SimpleNamespace(boundary_ratio=request_boundary)
    assert resolve_wan_boundary_ratio(engine_boundary, sampling_params) == expected_boundary


@pytest.mark.parametrize("boundary_ratio", [None, 0.875])
def test_wan22_step_split_matches_shifted_denoise_schedule(boundary_ratio):
    """The per-expert Cache-DiT step counts must follow the schedule forward() runs.

    forward() applies the flow shift in set_timesteps and falls back to boundary
    0.875 when the server pins none; an unshifted probe or a missing fallback
    sizes the experts' contexts for the wrong number of steps.
    """
    pipeline = SimpleNamespace(
        boundary_ratio=boundary_ratio,
        scheduler=build_wan_scheduler("unipc", 5.0),
        transformer=torch.nn.Linear(1, 1),
        _sample_solver="unipc",
        _flow_shift=5.0,
    )

    num_high_noise_steps, num_low_noise_steps = cd_model_specific._split_wan22_inference_steps(pipeline, 40)

    reference = build_wan_scheduler("unipc", 5.0)
    reference.set_timesteps(40, device="cpu", shift=5.0)
    expected_high_noise_steps = int((reference.timesteps >= 0.875 * 1000).sum())
    assert (num_high_noise_steps, num_low_noise_steps) == (expected_high_noise_steps, 40 - expected_high_noise_steps)
    assert num_low_noise_steps > 0


@patch("vllm_omni.diffusion.cache.cachedit.backend.BlockAdapter")
@patch("vllm_omni.diffusion.cache.cachedit.backend.cache_dit")
def test_cosmos3_cache_dit_wraps_gen_layers(mock_cache_dit, mock_block_adapter):
    """Cosmos3 should cache only the repeated GEN pathway blocks."""
    mock_pipeline = Mock()
    gen_layers = object()
    mock_pipeline.transformer.gen_layers = gen_layers
    mock_pipeline.transformer._cache_dit_adapter_config = Cosmos3VFMTransformer._cache_dit_adapter_config

    cd_model_specific.enable_cache_for_cosmos3(mock_pipeline, SAMPLE_CACHE_CONFIG)

    mock_cache_dit.enable_cache.assert_called_once()
    adapter_kwargs = mock_block_adapter.call_args.kwargs
    assert adapter_kwargs["transformer"] is mock_pipeline.transformer
    assert adapter_kwargs["blocks"] == [gen_layers]
    assert adapter_kwargs["has_separate_cfg"] is True
    assert adapter_kwargs["check_forward_pattern"] is False


# This test is skipped on ROCm since rocm_unquantized_gemm doesn't support CPU backend
@pytest.mark.skipif(
    current_omni_platform.is_rocm(),
    reason="vLLM ROCm custom ops lack CPU fallback",
)
def test_ltx2_cache_dit_receives_audio_as_encoder(init_fake_tp_group, request: pytest.FixtureRequest):
    """CacheDiT Pattern_0 treats the second positional arg as encoder_hidden_states,
    which is a collision for one of the kwargs in LTX2 since we treat the audio
    hidden states as encoder_hidden_states.

    This test ensures that a tiny LTX2 transformer can be initialized and run
    through the cache DiT backend without hitting a collision on the kwargs.
    """
    seq_len = 4
    video_in = torch.full((1, seq_len, 16), 1.0)
    audio_in = torch.full((1, seq_len, 16), 2.0)
    text_in = torch.full((1, seq_len, 16), 3.0)
    audio_text_in = torch.full((1, seq_len, 16), 4.0)

    with _force_torch_sdpa():
        model = LTX2VideoTransformer3DModel(
            in_channels=16,
            out_channels=16,
            patch_size=1,
            patch_size_t=1,
            num_attention_heads=2,
            attention_head_dim=8,
            cross_attention_dim=16,
            audio_in_channels=16,
            audio_out_channels=16,
            audio_num_attention_heads=2,
            audio_attention_head_dim=8,
            audio_cross_attention_dim=16,
            num_layers=2,
            caption_channels=16,
        )

    # NOTE: This is currently using the LTX2 custom enabler, but the custom
    # enablers will be consolidated after
    # https://github.com/vllm-project/vllm-omni/pull/2527 lands.
    LTX2Pipeline = type("LTX2Pipeline", (), {})
    pipeline = LTX2Pipeline()
    pipeline.transformer = model
    backend = CacheDiTBackend(DiffusionCacheConfig())
    backend.enable(pipeline)
    request.addfinalizer(lambda: backend.disable(pipeline))
    backend.refresh(pipeline, num_inference_steps=5)

    # Wrap call_Fn_blocks in CacheDiT so that we can verify the
    # hidden/encoder states are what we expect them to be
    captured = {}
    original = CachedBlocks_Pattern_0_1_2.call_Fn_blocks

    def call_Fn_blocks_and_capture(self, hidden_states, encoder_hidden_states, *a, **kw):
        captured["hidden_states"] = hidden_states
        captured["encoder_hidden_states"] = encoder_hidden_states
        return original(self, hidden_states, encoder_hidden_states, *a, **kw)

    # Also, map projections to identity so that we can just check
    # the captured tensors directly instead of having to reproject
    identity = torch.nn.Identity()
    with (
        patch.object(model, "proj_in", identity),
        patch.object(model, "audio_proj_in", identity),
        patch.object(CachedBlocks_Pattern_0_1_2, "call_Fn_blocks", call_Fn_blocks_and_capture),
        torch.no_grad(),
    ):
        model(
            hidden_states=video_in,
            audio_hidden_states=audio_in,
            encoder_hidden_states=text_in,
            audio_encoder_hidden_states=audio_text_in,
            timestep=torch.tensor([[1000.0] * seq_len]),
            num_frames=1,
            height=2,
            width=2,
            audio_num_frames=seq_len,
        )

    # Pattern_0 maps (hidden_states, encoder_hidden_states) to (video, audio)
    assert torch.equal(captured["hidden_states"], video_in)
    assert torch.equal(captured["encoder_hidden_states"], audio_in)


def test_summary_with_no_transformer_is_nonfatal():
    """Regression test for https://github.com/vllm-project/vllm-omni/issues/4325."""

    class FakePipeline:
        pass

    cache_summary(pipeline=FakePipeline())


# This test is skipped on ROCm since rocm_unquantized_gemm doesn't support CPU backend
@pytest.mark.skipif(
    current_omni_platform.is_rocm(),
    reason="vLLM ROCm custom ops lack CPU fallback",
)
def test_mammothmoda2_cache_dit_runs_end_to_end_on_tiny_model(init_fake_tp_group, request: pytest.FixtureRequest):
    """A tiny MammothModa2 DiT runs through Cache-DiT with paired CFG forwards.

    Exercises the Pattern_3 contract end to end: after enabling the custom
    enabler, blocks must be callable as ``block(hidden_states, **kwargs)``
    and the pipeline's two-forwards-per-step CFG cadence must drive the
    cache context without shape or parity errors.
    """
    from cache_dit import BlockAdapter

    from tests.diffusion.models.mammoth_moda2.helpers import initialize_block_weights
    from vllm_omni.diffusion.models.mammoth_moda2.mammothmoda2_dit_model import Transformer2DModel
    from vllm_omni.diffusion.models.mammoth_moda2.rope_real import RotaryPosEmbedReal

    torch.manual_seed(0)

    # MammothModa2 attention resolves through the shared Omni attention layer,
    # which picks the platform-default backend at construction time. Pin
    # TORCH_SDPA so a CUDA-visible host does not select FA3 for CPU tensors.
    with _force_torch_sdpa():
        model = Transformer2DModel(
            patch_size=2,
            in_channels=4,
            hidden_size=96,
            num_layers=4,
            num_refiner_layers=1,
            num_attention_heads=2,
            num_kv_heads=2,
            multiple_of=8,
            axes_dim_rope=(16, 16, 16),
            axes_lens=(300, 512, 512),
            text_feat_dim=16,
        )
    initialize_block_weights(model)
    model.eval()

    MammothModa2Pipeline = type("MammothModa2DiTPipeline", (), {})
    pipeline = MammothModa2Pipeline()
    pipeline.gen_transformer = model
    backend = CacheDiTBackend(DiffusionCacheConfig())
    backend.enable(pipeline)
    request.addfinalizer(lambda: backend.disable(pipeline))
    assert BlockAdapter.is_cached(model)

    num_inference_steps = 8
    backend.refresh(pipeline, num_inference_steps=num_inference_steps)

    freqs_cis = RotaryPosEmbedReal.get_freqs_real((16, 16, 16), (300, 512, 512), theta=10000)
    latents = torch.randn(1, 4, 8, 8)
    text = torch.randn(1, 5, 16)
    negative = torch.randn(1, 3, 16)
    text_mask = torch.ones(1, 5, dtype=torch.bool)
    negative_mask = torch.ones(1, 3, dtype=torch.bool)

    with torch.no_grad():
        for step in range(num_inference_steps):
            timestep = torch.full((1,), 1.0 - step / num_inference_steps)
            # Conditional pass ...
            cond = model(
                hidden_states=latents,
                timestep=timestep,
                text_hidden_states=text,
                freqs_cis=freqs_cis,
                text_attention_mask=text_mask,
            )
            # ... always followed by the unconditional pass (parity).
            uncond = model(
                hidden_states=latents,
                timestep=timestep,
                text_hidden_states=negative,
                freqs_cis=freqs_cis,
                text_attention_mask=negative_mask,
            )
            latents = uncond + 4.0 * (cond - uncond)

    assert latents.shape == (1, 4, 8, 8)
    assert torch.isfinite(latents).all()


def _taylorseer_warning_logged(caplog) -> bool:
    return any(
        "validated cache window" in record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING
    )


def test_taylorseer_order_two_long_window_warns(caplog):
    config = cd_backend.CacheDiTConfig(
        enable_taylorseer=True,
        taylorseer_order=2,
        max_continuous_cached_steps=VALIDATED_TAYLORSEER_MAX_CONTINUOUS_CACHED_STEPS + 2,
    )
    with caplog.at_level("WARNING"):
        calibrator = config.to_calibrator_config()

    assert calibrator is not None
    assert _taylorseer_warning_logged(caplog)


def test_taylorseer_order_two_validated_window_does_not_warn(caplog):
    config = cd_backend.CacheDiTConfig(
        enable_taylorseer=True,
        taylorseer_order=2,
        max_continuous_cached_steps=VALIDATED_TAYLORSEER_MAX_CONTINUOUS_CACHED_STEPS,
    )
    with caplog.at_level("WARNING"):
        calibrator = config.to_calibrator_config()

    assert calibrator is not None
    assert not _taylorseer_warning_logged(caplog)


def test_taylorseer_order_above_two_warns_without_validated_window(caplog):
    config = cd_backend.CacheDiTConfig(
        enable_taylorseer=True,
        taylorseer_order=3,
        max_continuous_cached_steps=VALIDATED_TAYLORSEER_MAX_CONTINUOUS_CACHED_STEPS,
    )
    with caplog.at_level("WARNING"):
        calibrator = config.to_calibrator_config()

    assert calibrator is not None
    assert _taylorseer_warning_logged(caplog)


def test_taylorseer_order_one_does_not_warn(caplog):
    config = cd_backend.CacheDiTConfig(enable_taylorseer=True, taylorseer_order=1)
    with caplog.at_level("WARNING"):
        config.to_calibrator_config()

    assert not _taylorseer_warning_logged(caplog)
