# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Contract tests for the Self-Forcing causal DMD port."""

import os

import pytest
import torch

from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

_TINY = {
    "patch_size": (1, 2, 2),
    "num_attention_heads": 2,
    "attention_head_dim": 64,
    "in_channels": 4,
    "out_channels": 4,
    "text_dim": 8,
    "freq_dim": 16,
    "ffn_dim": 32,
    "num_layers": 2,
    "rope_max_seq_len": 32,
    "num_frames_per_block": 1,
    "window_frames": 3,
}
# 2 x 3 post-patch tokens per frame: not a whole FlashAttention page.
_LATENT_HEIGHT, _LATENT_WIDTH = 4, 6


class _NativePlatform:
    """Answers no to every accelerator query so vLLM-Omni custom ops pick ``forward_native``."""

    def __getattr__(self, name: str):
        if name.startswith("is_"):
            return lambda: False
        raise AttributeError(name)


@pytest.fixture(autouse=True)
def _runtime(monkeypatch):
    """Single-process TP group without Ulysses, SDPA cross-attention, default GEMM dispatch."""
    from vllm.config import VllmConfig
    from vllm.config.vllm import set_current_vllm_config
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.layers.utils import default_unquantized_gemm

    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, OmniDiffusionConfig

    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.dispatch_unquantized_gemm",
        lambda *_args, **_kwargs: default_unquantized_gemm,
    )
    monkeypatch.setattr("vllm_omni.diffusion.models.lingbot_world.transformer._ulysses_state", lambda: (1, 0, None))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29514")
    init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method="env://", backend="gloo")
    initialize_model_parallel()
    diffusion_config = OmniDiffusionConfig(
        model="self-forcing-test",
        diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA")),
    )
    with set_current_vllm_config(VllmConfig()), set_current_diffusion_config(diffusion_config), torch.inference_mode():
        yield
    cleanup_dist_env_and_memory()


@pytest.fixture
def native_ops(monkeypatch):
    monkeypatch.setattr("vllm_omni.diffusion.layers.custom_op.current_omni_platform", _NativePlatform())


def _tiny_model(device: str = "cpu", dtype: torch.dtype = torch.float32, **overrides):
    from vllm_omni.diffusion.models.self_forcing.transformer import SelfForcingTransformer3DModel

    torch.manual_seed(0)
    model = SelfForcingTransformer3DModel(**{**_TINY, **overrides})
    for parameter in model.parameters():
        parameter.normal_(0, 0.2)
    return model.to(device=device, dtype=dtype).eval()


def _block(model, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    shape = (1, model.config.in_channels, model.config.num_frames_per_block, _LATENT_HEIGHT, _LATENT_WIDTH)
    return torch.randn(shape, generator=generator).to(device=model.patch_embedding.weight.device, dtype=model.dtype)


def _cache(model):
    return model.allocate_cache(
        latent_height=_LATENT_HEIGHT,
        latent_width=_LATENT_WIDTH,
        device=model.patch_embedding.weight.device,
        dtype=model.dtype,
    )


def _text(model) -> torch.Tensor:
    return torch.randn(1, 5, model.config.text_dim).to(device=model.patch_embedding.weight.device, dtype=model.dtype)


def _timestep(model, value: float) -> torch.Tensor:
    return torch.tensor([value], device=model.patch_embedding.weight.device)


@pytest.mark.cpu
def test_dmd_schedule_matches_the_reference_shifted_table() -> None:
    """The closed form must equal Self-Forcing's lookup into the shifted 1000-step flow table."""
    from vllm_omni.diffusion.models.self_forcing.pipeline_self_forcing import (
        SELF_FORCING_DMD_TIMESTEPS,
        self_forcing_dmd_schedule,
    )

    sigmas = torch.linspace(1.0, 0.0, 1001)[:-1]
    sigmas = 5.0 * sigmas / (1 + 4.0 * sigmas)
    table = torch.cat((sigmas * 1000, torch.zeros(1)))
    expected = [table[1000 - raw].item() for raw in SELF_FORCING_DMD_TIMESTEPS]
    schedule = self_forcing_dmd_schedule()
    assert [timestep for timestep, _ in schedule] == pytest.approx(expected, abs=1e-3)
    assert [sigma for _, sigma in schedule] == pytest.approx([value / 1000 for value in expected], abs=1e-6)


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"num_layers": 29}, "num_layers"),
        ({"in_channels": 36}, "in_channels"),
        ({"_class_name": "WanTransformer3DModel"}, "_class_name"),
    ],
)
def test_from_config_rejects_checkpoints_of_another_geometry(config, message) -> None:
    from vllm_omni.diffusion.models.self_forcing.transformer import SelfForcingTransformer3DModel

    with pytest.raises(ValueError, match=message):
        SelfForcingTransformer3DModel.from_config(config)


@pytest.mark.cpu
@pytest.mark.usefixtures("native_ops")
def test_load_weights_packs_wan_qkv_and_rejects_unknown_names() -> None:
    model = _tiny_model()
    checkpoint = {}
    for name, parameter in model.named_parameters():
        if ".self_attn.qkv." in name:
            for projection in ("q", "k", "v"):
                shard = torch.randn(parameter.shape[0] // 3, *parameter.shape[1:])
                checkpoint[name.replace(".qkv.", f".{projection}.")] = shard
        else:
            checkpoint[name] = torch.randn_like(parameter)
    loaded = model.load_weights(checkpoint.items())

    assert set(dict(model.named_parameters())) <= loaded
    packed = dict(model.named_parameters())["blocks.0.self_attn.qkv.weight"]
    expected = torch.cat([checkpoint[f"blocks.0.self_attn.{p}.weight"] for p in ("q", "k", "v")])
    torch.testing.assert_close(packed, expected)
    with pytest.raises(KeyError, match="camera"):
        model.load_weights([("blocks.0.camera_scale.weight", torch.zeros(1))])


@pytest.mark.cpu
@pytest.mark.usefixtures("native_ops")
def test_forward_rejects_a_partial_block() -> None:
    model = _tiny_model(num_frames_per_block=2, window_frames=4)
    with pytest.raises(ValueError, match="exactly 2 latent frames"):
        model(
            torch.randn(1, 4, 1, _LATENT_HEIGHT, _LATENT_WIDTH),
            torch.zeros(1),
            _text(model),
            cache=_cache(model),
            start_frame=0,
            update_cache=False,
        )


@hardware_test(res={"cuda": "L4"})
def test_probes_leave_the_cache_untouched_and_commits_extend_it() -> None:
    """DMD probes must not persist K/V; only the clean commit advances the causal history."""
    model = _tiny_model(device="cuda", dtype=torch.bfloat16)
    cache = _cache(model)
    text = _text(model)

    first = model(_block(model, 1), _timestep(model, 937.5), text, cache=cache, start_frame=0, update_cache=False)
    again = model(_block(model, 1), _timestep(model, 937.5), text, cache=cache, start_frame=0, update_cache=False)
    torch.testing.assert_close(first, again, rtol=0, atol=0)
    assert cache.self_attention[0].end == 0

    model(_block(model, 2), _timestep(model, 0.0), text, cache=cache, start_frame=0, update_cache=True)
    assert cache.self_attention[0].end == model.tokens_per_frame(_LATENT_HEIGHT, _LATENT_WIDTH)

    history = model(_block(model, 3), _timestep(model, 937.5), text, cache=cache, start_frame=1, update_cache=False)
    no_history = model(
        _block(model, 3), _timestep(model, 937.5), text, cache=_cache(model), start_frame=1, update_cache=False
    )
    assert not torch.allclose(history, no_history)


@hardware_test(res={"cuda": "L4"})
def test_window_keeps_only_the_most_recent_frames() -> None:
    model = _tiny_model(device="cuda", dtype=torch.bfloat16, window_frames=3)
    cache = _cache(model)
    text = _text(model)
    for frame in range(5):
        model(_block(model, frame), _timestep(model, 0.0), text, cache=cache, start_frame=frame, update_cache=True)
    tokens_per_frame = model.tokens_per_frame(_LATENT_HEIGHT, _LATENT_WIDTH)
    assert cache.self_attention[0].end == 3 * tokens_per_frame
    assert cache.self_attention[0].absolute_end == 5 * tokens_per_frame


def _pipeline():
    """The pipeline around the tiny transformer and a tiny Wan VAE, without loading a checkpoint."""
    from diffusers import AutoencoderKLWan

    from vllm_omni.diffusion.models.self_forcing.pipeline_self_forcing import SelfForcingPipeline

    pipeline = SelfForcingPipeline.__new__(SelfForcingPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.transformer = _tiny_model(num_frames_per_block=3, window_frames=21)
    pipeline.vae = AutoencoderKLWan(
        base_dim=8, z_dim=4, dim_mult=[1, 2], num_res_blocks=1, temperal_downsample=[True], attn_scales=[]
    ).to(torch.bfloat16)
    # Request geometry follows the released Wan2.1 VAE, not the tiny one.
    pipeline.vae_scale_factor_spatial, pipeline.vae_scale_factor_temporal = 8, 4
    pipeline._ar_height, pipeline._ar_width = 480, 832
    pipeline._streaming_decoder_instance = None
    return pipeline


@pytest.mark.cpu
@pytest.mark.usefixtures("native_ops")
def test_kv_cache_spec_describes_the_transformer_and_its_decoder_state() -> None:
    spec = _pipeline().ar_diffusion_kv_cache_spec()
    assert (spec.num_layers, spec.num_kv_heads, spec.head_size) == (2, 2, 64)
    assert (spec.tokens_per_frame, spec.frames_per_block, spec.window_frames, spec.sink_frames) == (1560, 3, 21, 0)
    assert [branch.name for branch in spec.kv_branches] == ["main"]
    assert spec.cross_attention_lengths == {"text": 512}
    # The streaming VAE cache measured at 37832 KiB per 64x64 fp32 output, at 832x480 in bf16.
    assert spec.model_owned_state_bytes_per_session == 37832 * 1024 * (480 * 832) // (64 * 64) // 2


@pytest.mark.cpu
def test_stepwise_deploy_streams_the_default_geometry_on_the_ar_diffusion_engine() -> None:
    """Without the AR-Diffusion engine no session is bound and every streamed request fails."""
    from tests.helpers.stage_config import get_deploy_config_path
    from vllm_omni.config.stage_config import load_deploy_config
    from vllm_omni.diffusion.models.self_forcing.pipeline_self_forcing import (
        SELF_FORCING_DEFAULT_HEIGHT,
        SELF_FORCING_DEFAULT_WIDTH,
    )

    (stage,) = load_deploy_config(get_deploy_config_path("self_forcing_stepwise.yaml")).stages
    assert stage.model_class_name == "SelfForcingPipeline"
    assert stage.engine_extras["engine_backend"].endswith("ar_diffusion.engine.ARDiffusionEngine")
    assert (stage.step_execution, stage.engine_extras["streaming_output"], stage.max_num_seqs) == (True, True, 1)
    model_config = stage.engine_extras["model_config"]
    assert (model_config["ar_diffusion_height"], model_config["ar_diffusion_width"]) == (
        SELF_FORCING_DEFAULT_HEIGHT,
        SELF_FORCING_DEFAULT_WIDTH,
    )


def _request(**sampling_overrides):
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    prompt = sampling_overrides.pop("prompt", "a red fox running through snow")
    sampling = OmniDiffusionSamplingParams(generator=torch.Generator(), **sampling_overrides)
    return DiffusionRequestBatch(
        requests=[OmniDiffusionRequest(prompt=prompt, sampling_params=sampling, request_id="r")]
    )


@pytest.mark.cpu
@pytest.mark.usefixtures("native_ops")
def test_request_defaults_to_the_distilled_81_frame_clip() -> None:
    inputs = _pipeline()._parse_request(_request())
    assert (inputs.height, inputs.width, inputs.num_frames, inputs.num_latent_frames) == (480, 832, 81, 21)


@pytest.mark.cpu
@pytest.mark.usefixtures("native_ops")
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"num_frames": 80}, "num_frames"),
        ({"num_frames": 85}, "3-frame AR blocks"),
        ({"guidance_scale": 5.0}, "classifier-free guidance"),
        ({"num_inference_steps": 8}, "num_inference_steps"),
        ({"height": 488}, "multiples of 16"),
        ({"prompt": {"prompt": "a fox", "negative_prompt": "blurry"}}, "negative prompt"),
    ],
)
def test_request_rejects_settings_the_dmd_generator_cannot_honor(overrides, message) -> None:
    pipeline = _pipeline()
    with pytest.raises(ValueError, match=message):
        pipeline._parse_request(_request(**overrides))
