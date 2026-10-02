# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
from pathlib import Path
from shutil import rmtree
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from tests.model_tests.diffusion.anima_builder import CHECKPOINT_FILENAME, tiny_anima_builder
from vllm_omni.diffusion.attention import selector
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_qwenimage import DistributedAutoencoderKLQwenImage
from vllm_omni.diffusion.models.anima import pipeline_anima
from vllm_omni.diffusion.models.anima.anima_text_conditioner import AnimaTextConditioner
from vllm_omni.diffusion.models.anima.anima_transformer import AnimaTransformer3DModel
from vllm_omni.diffusion.models.anima.pipeline_anima import AnimaPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize(
    "offload_args",
    [
        {"enable_cpu_offload": True},
        {"enable_layerwise_offload": True},
        {"enable_distributed_layerwise_offload": True},
        {"diffusion_offload_config": {"mode": "module", "components": ["dit"]}},
        {
            "diffusion_offload_config": {
                "mode": "layer",
                "components": ["dit"],
                "layer_options": {"dit": {"weight_transfer": "rank-local"}},
            }
        },
    ],
)
def test_native_anima_rejects_offload(offload_args):
    config = OmniDiffusionConfig(model="anima.safetensors", **offload_args)
    with pytest.raises(NotImplementedError, match="offload"):
        AnimaPipeline(od_config=config, device=torch.device("cpu"))


def test_native_anima_initializes_without_offload():
    config = OmniDiffusionConfig(model="anima.safetensors")
    AnimaPipeline(od_config=config, device=torch.device("cpu"))


def test_anima_registration() -> None:
    """The native Anima pipeline is registered and importable."""
    from vllm_omni.diffusion.registry import DiffusionModelRegistry

    assert DiffusionModelRegistry._try_load_model_cls("AnimaPipeline") is not None


@pytest.mark.parametrize("model_class_name", ["AnimaPipeline", "AnimaModularPipeline"])
def test_enrich_config_native_anima_checkpoint(tmp_path: Path, model_class_name: str) -> None:
    """Native and reference Anima class names resolve to AnimaPipeline."""
    dummy_checkpoint = tmp_path / "model.safetensors"
    dummy_checkpoint.write_text("dummy")

    config = OmniDiffusionConfig(
        model=str(dummy_checkpoint),
        model_class_name=model_class_name,
    )
    config.enrich_config()

    assert config.diffusion_load_format == "default"
    assert config.model_class_name == "AnimaPipeline"
    assert config.diffusers_pipeline_cls is None


def test_native_anima_component_paths_use_custom_pipeline_args() -> None:
    """Anima resolves shared and component-specific paths from native args."""
    pipeline = AnimaPipeline.__new__(AnimaPipeline)
    pipeline.od_config = SimpleNamespace(
        custom_pipeline_args={
            "components_path": "/tmp/anima-components",
            "vae_path": "/tmp/anima-vae",
            "text_encoder_model": "/tmp/anima-text-encoder",
        }
    )

    assert pipeline._component_path("vae", "/tmp/default") == "/tmp/anima-vae"
    assert pipeline._component_path("text_encoder", "/tmp/default") == "/tmp/anima-text-encoder"
    assert pipeline._component_path("tokenizer", "/tmp/default") == "/tmp/default"


def test_native_anima_converts_original_cosmos_transformer_keys() -> None:
    """Original Cosmos checkpoint names map to native module names."""
    converted = AnimaPipeline._convert_original_transformer_state_dict(
        {
            "net.x_embedder.proj.1.weight": "patch",
            "net.blocks.0.self_attn.q_proj.weight": "q",
            "net.blocks.0.self_attn.q_norm.weight": "q_norm",
            "net.blocks.0.mlp.layer1.weight": "mlp",
            "net.final_layer.linear.weight": "out",
            "net.accum_iteration": "drop",
        }
    )

    assert converted == {
        "patch_embed.proj.weight": "patch",
        "transformer_blocks.0.attn1.to_q.weight": "q",
        "transformer_blocks.0.attn1.norm_q.weight": "q_norm",
        "transformer_blocks.0.ff.net.0.proj.weight": "mlp",
        "proj_out.weight": "out",
    }


def test_native_anima_resolves_vae_scale_factor_from_loaded_vae() -> None:
    """The loaded VAE determines the spatial scale factor."""
    from vllm_omni.diffusion.models.anima.pipeline_anima import _anima_vae_scale_factor_from_vae

    vae = SimpleNamespace(config=SimpleNamespace(spatial_compression_ratio=16))

    assert _anima_vae_scale_factor_from_vae(vae) == 16


@pytest.mark.parametrize("use_config_file", [False, True])
def test_native_anima_loads_synthetic_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_config_file: bool
) -> None:
    """Native modules load exact weights from an Anima safetensors checkpoint."""
    tiny_transformer_config = {
        "in_channels": 1,
        "out_channels": 1,
        "num_attention_heads": 1,
        "attention_head_dim": 12,
        "num_layers": 1,
        "mlp_ratio": 1.0,
        "text_embed_dim": 4,
        "adaln_lora_dim": 3,
        "max_size": (1, 2, 2),
        "patch_size": (1, 1, 1),
        "rope_scale": (1.0, 1.0, 1.0),
        "concat_padding_mask": True,
        "extra_pos_embed_type": None,
    }
    tiny_text_conditioner_config = {
        "source_dim": 4,
        "target_dim": 4,
        "model_dim": 4,
        "num_layers": 1,
        "num_attention_heads": 1,
        "target_vocab_size": 8,
        "min_sequence_length": 4,
    }
    if use_config_file:
        (tmp_path / "anima.json").write_text(
            json.dumps({"transformer": tiny_transformer_config, "text_conditioner": tiny_text_conditioner_config})
        )
    else:
        monkeypatch.setattr(pipeline_anima, "ANIMA_TRANSFORMER_CONFIG", tiny_transformer_config)
        monkeypatch.setattr(pipeline_anima, "ANIMA_TEXT_CONDITIONER_CONFIG", tiny_text_conditioner_config)

    monkeypatch.setattr(selector, "_cached_get_backend_cls", lambda *_args, **_kwargs: SDPABackend)
    transformer = AnimaTransformer3DModel(**tiny_transformer_config)
    text_conditioner = AnimaTextConditioner(**tiny_text_conditioner_config)
    transformer_state = {name: tensor.detach().clone() for name, tensor in transformer.state_dict().items()}
    text_conditioner_state = {name: tensor.detach().clone() for name, tensor in text_conditioner.state_dict().items()}
    checkpoint_state = {
        **{f"transformer.{name}": tensor for name, tensor in transformer_state.items()},
        **{f"text_conditioner.{name}": tensor for name, tensor in text_conditioner_state.items()},
    }

    checkpoint_path = tmp_path / "anima.safetensors"
    save_file(checkpoint_state, str(checkpoint_path))

    pipeline = AnimaPipeline.__new__(AnimaPipeline)
    pipeline.od_config = SimpleNamespace(model=str(checkpoint_path), dtype=torch.float32)
    pipeline.device = torch.device("cpu")

    def assert_loaded(loaded_transformer, loaded_text_conditioner):
        for name, tensor in transformer_state.items():
            assert torch.equal(loaded_transformer.state_dict()[name], tensor)
        for name, tensor in text_conditioner_state.items():
            assert torch.equal(loaded_text_conditioner.state_dict()[name], tensor)

    loaded_transformer, loaded_text_conditioner = pipeline._load_native_denoiser_components(dict(checkpoint_state))
    assert_loaded(loaded_transformer, loaded_text_conditioner)

    loaded_transformer, loaded_text_conditioner = pipeline._load_native_denoiser_components()
    assert_loaded(loaded_transformer, loaded_text_conditioner)

    if use_config_file:
        tiny_transformer_config["num_layers"] = 2
        (tmp_path / "anima.json").write_text(
            json.dumps({"transformer": tiny_transformer_config, "text_conditioner": tiny_text_conditioner_config})
        )
        with pytest.raises(RuntimeError, match="Missing key"):
            pipeline._load_native_denoiser_components()


def test_tiny_anima_generates_images(monkeypatch: pytest.MonkeyPatch) -> None:
    """Load every saved component and exercise CFG, decoding, and multiple outputs on CPU."""
    monkeypatch.setattr(selector, "_cached_get_backend_cls", lambda *_args, **_kwargs: SDPABackend)
    monkeypatch.setattr(current_omni_platform, "is_available", lambda: False)
    monkeypatch.setattr(
        DistributedAutoencoderKLQwenImage,
        "init_distributed",
        lambda self: setattr(self, "distributed_executor", SimpleNamespace(parallel_size=1)),
    )
    model_dir = tiny_anima_builder()
    try:
        config = OmniDiffusionConfig(
            model=str(Path(model_dir) / CHECKPOINT_FILENAME), model_class_name="AnimaPipeline", dtype=torch.float32
        )
        pipeline = AnimaPipeline(od_config=config, device=torch.device("cpu"))
        pipeline.load_weights()
        pipeline.eval()
        outputs = []
        with torch.inference_mode():
            for _ in range(2):
                request = OmniDiffusionRequest(
                    prompt="Dummy prompt",
                    request_id="tiny-anima",
                    sampling_params=OmniDiffusionSamplingParams(
                        height=32,
                        width=32,
                        num_inference_steps=2,
                        guidance_scale=4.0,
                        num_outputs_per_prompt=2,
                        generator=torch.Generator(device="cpu").manual_seed(42),
                    ),
                )
                outputs.append(pipeline(DiffusionRequestBatch([request]))[0].output)
        assert outputs[0].shape == (2, 3, 32, 32)
        assert torch.isfinite(outputs[0]).all()
        assert torch.equal(outputs[0], outputs[1])
        assert not torch.equal(outputs[0][0], outputs[0][1])
    finally:
        rmtree(model_dir)


def test_native_anima_profiler_setup_is_deferred_and_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Profiler targets are wrapped once, after deferred components are loaded."""
    pipeline = AnimaPipeline.__new__(AnimaPipeline)
    pipeline.od_config = SimpleNamespace(enable_diffusion_pipeline_profiler=True)
    pipeline._profiler_initialized = False
    pipeline.vae = SimpleNamespace(decode=lambda: "decoded")
    pipeline.text_encoder = SimpleNamespace(forward=lambda: "encoded")

    monkeypatch.setattr(current_omni_platform, "is_available", lambda: False)
    pipeline._setup_profiler()
    wrapped_decode = pipeline.vae.decode
    wrapped_text_encoder = pipeline.text_encoder.forward
    pipeline._setup_profiler()

    assert pipeline.vae.decode is wrapped_decode
    assert pipeline.text_encoder.forward is wrapped_text_encoder
    assert pipeline.vae.decode() == "decoded"
    assert pipeline.text_encoder.forward() == "encoded"
    assert "AnimaPipeline.vae.decode" in pipeline.stage_durations
    assert "AnimaPipeline.text_encoder.forward" in pipeline.stage_durations


def _make_anima_forward_probe():
    pipeline = AnimaPipeline.__new__(AnimaPipeline)
    pipeline.device = torch.device("cpu")
    pipeline.vae_scale_factor = 8
    pipeline.transformer = SimpleNamespace(dtype=torch.float32)
    pipeline.text_encoder = SimpleNamespace(dtype=torch.float32)
    pipeline._current_timestep = None
    pipeline._num_timesteps = 0
    pipeline._guidance_scale = 0.0
    captured = {}

    def encode_prompt(**_kwargs):
        return {
            "qwen_prompt_embeds": torch.zeros(1, 2, 4),
            "qwen_attention_mask": torch.ones(1, 2),
            "t5_input_ids": torch.ones(1, 2, dtype=torch.long),
            "t5_attention_mask": torch.ones(1, 2),
            "negative_qwen_prompt_embeds": torch.zeros(1, 2, 4),
            "negative_qwen_attention_mask": torch.ones(1, 2),
            "negative_t5_input_ids": torch.ones(1, 2, dtype=torch.long),
            "negative_t5_attention_mask": torch.ones(1, 2),
        }

    def condition_prompt_embeds(**_kwargs):
        return torch.zeros(1, 2, 4)

    def prepare_latents(**kwargs):
        captured["prepare_latents"] = kwargs
        return torch.zeros(1, 16, 1, kwargs["height"] // 8, kwargs["width"] // 8)

    def diffuse(**kwargs):
        captured["diffuse"] = kwargs
        return kwargs["latents"]

    pipeline.encode_prompt = encode_prompt
    pipeline.condition_prompt_embeds = condition_prompt_embeds
    pipeline.prepare_latents = prepare_latents
    pipeline.prepare_timesteps = lambda **_kwargs: (torch.ones(1), 1)
    pipeline.diffuse = diffuse

    def decode_latents(latents, output_type="pil"):
        captured["output_type"] = output_type
        return DiffusionOutput(output=latents)

    pipeline.decode_latents = decode_latents
    return pipeline, captured


def test_native_anima_forward_uses_official_default_resolution() -> None:
    """Forward uses Anima's reference resolution when none is requested."""
    pipeline, captured = _make_anima_forward_probe()
    req = OmniDiffusionRequest(
        prompt="a red cube",
        sampling_params=OmniDiffusionSamplingParams(),
        request_id="anima-defaults",
    )

    pipeline.forward(DiffusionRequestBatch([req]))

    assert captured["prepare_latents"]["height"] == 1024
    assert captured["prepare_latents"]["width"] == 1024


def test_native_anima_forward_honors_sampling_output_type() -> None:
    """Request-level output type controls Anima decoding."""
    pipeline, captured = _make_anima_forward_probe()
    req = OmniDiffusionRequest(
        prompt="a red cube",
        sampling_params=OmniDiffusionSamplingParams(output_type="latent"),
        request_id="anima-output-type",
    )

    outputs = pipeline.forward(DiffusionRequestBatch([req]))

    assert len(outputs) == 1
    assert isinstance(outputs[0], DiffusionOutput)
    assert outputs[0].output.shape == (1, 16, 1, 128, 128)
    assert captured["output_type"] == "latent"


@pytest.mark.parametrize("request_count", [0, 2])
def test_native_anima_forward_rejects_non_single_request_batches(request_count: int) -> None:
    pipeline, captured = _make_anima_forward_probe()
    requests = [
        OmniDiffusionRequest(
            prompt="a red cube",
            sampling_params=OmniDiffusionSamplingParams(),
            request_id=f"anima-{index}",
        )
        for index in range(request_count)
    ]

    with pytest.raises(ValueError, match="supports one request per forward"):
        pipeline.forward(DiffusionRequestBatch(requests))

    assert captured == {}


def test_native_anima_explicit_guidance_scale_drives_cfg_multiplier() -> None:
    """An explicit guidance scale controls native classifier-free guidance."""
    pipeline, captured = _make_anima_forward_probe()
    req = OmniDiffusionRequest(
        prompt="a red cube",
        sampling_params=OmniDiffusionSamplingParams(guidance_scale=5.0),
        request_id="anima-guidance",
    )

    pipeline.forward(DiffusionRequestBatch([req]))

    assert captured["diffuse"]["do_true_cfg"] is True
    assert captured["diffuse"]["true_cfg_scale"] == 5.0


def test_native_anima_true_cfg_scale_overrides_guidance_multiplier() -> None:
    """An explicit true-CFG scale overrides the guidance-scale multiplier."""
    pipeline, captured = _make_anima_forward_probe()
    req = OmniDiffusionRequest(
        prompt="a red cube",
        sampling_params=OmniDiffusionSamplingParams(guidance_scale=5.0, true_cfg_scale=3.0),
        request_id="anima-true-cfg",
    )

    pipeline.forward(DiffusionRequestBatch([req]))

    assert captured["diffuse"]["do_true_cfg"] is True
    assert captured["diffuse"]["true_cfg_scale"] == 3.0


def test_native_anima_cfg_equation() -> None:
    """The denoising loop applies the standard true-CFG equation."""
    pipeline = AnimaPipeline.__new__(AnimaPipeline)
    pipeline.device = torch.device("cpu")
    pipeline._interrupt = False

    class MockProgressBar:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def update(self):
            pass

    pipeline.progress_bar = lambda **_kwargs: MockProgressBar()

    class MockTransformer:
        dtype = torch.float32

        def __call__(self, hidden_states, timestep, encoder_hidden_states, padding_mask, return_dict=False):
            # Return double the encoder_hidden_states to simulate transformer output
            return (encoder_hidden_states * 2.0,)

    pipeline.transformer = MockTransformer()

    class MockScheduler:
        def set_begin_index(self, index):
            pass

        def step(self, noise_pred, t, latents, return_dict=False):
            # Return the noise_pred itself to inspect it
            return (noise_pred,)

    pipeline.scheduler = MockScheduler()

    prompt_embeds = torch.tensor([[[2.0, 3.0]]])
    negative_prompt_embeds = torch.tensor([[[1.0, 2.0]]])
    latents = torch.zeros(1, 16, 1, 16, 16)
    padding_mask = torch.zeros(1, 1, 16, 16)
    timesteps = torch.tensor([500.0])

    # Cond = prompt_embeds * 2.0 = [4.0, 6.0]
    # Uncond = negative_prompt_embeds * 2.0 = [2.0, 4.0]
    # Expected noise_pred = uncond + true_cfg_scale * (cond - uncond)
    #                     = [2.0, 4.0] + 4.0 * ([4.0, 6.0] - [2.0, 4.0])
    #                     = [2.0, 4.0] + 4.0 * [2.0, 2.0]
    #                     = [2.0, 4.0] + [8.0, 8.0] = [10.0, 12.0]
    out_latents = pipeline.diffuse(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        latents=latents,
        padding_mask=padding_mask,
        timesteps=timesteps,
        do_true_cfg=True,
        true_cfg_scale=4.0,
    )

    expected = torch.tensor([[[10.0, 12.0]]])
    assert torch.allclose(out_latents, expected)
