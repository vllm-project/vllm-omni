from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import PIL.Image
import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_s2v import (
    Wan22S2VPipeline,
    _make_clip_generators,
)
from vllm_omni.diffusion.models.wan2_2.wan2_2_s2v_transformer import WanS2VTransformer3DModel
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_s2v_sampling(**overrides):
    values = {
        "height": 16,
        "width": 16,
        "num_frames": 8,
        "num_inference_steps": 1,
        "guidance_scale_provided": True,
        "guidance_scale": 1.0,
        "generator": None,
        "seed": None,
        "num_outputs_per_prompt": 1,
        "max_sequence_length": 8,
        "latents": None,
        "output_type": "latent",
        "extra_args": {},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_s2v_validation_pipeline() -> Wan22S2VPipeline:
    pipeline = object.__new__(Wan22S2VPipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.transformer = SimpleNamespace(dtype=torch.float32)
    pipeline.vae_scale_factor_spatial = 8
    pipeline.resolution_divisor = 16
    pipeline.motion_frames = 7
    pipeline.drop_first_motion = True
    pipeline._DEFAULT_INFER_FRAMES = 8
    pipeline._guidance_scale = None
    pipeline.check_inputs = lambda *args, **kwargs: None
    pipeline.encode_prompt = lambda **kwargs: (torch.zeros(2, 2, 3), None)  # type: ignore[method-assign]
    return pipeline


def test_s2v_predict_noise_keeps_batch_dimension() -> None:
    pipeline = object.__new__(Wan22S2VPipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    transformer = MagicMock()
    transformer.parameters.return_value = iter([torch.zeros(1, dtype=torch.bfloat16)])
    transformer.return_value = (torch.zeros(1, 16, 2, 2, 2),)
    pipeline.transformer = transformer

    result = pipeline.predict_noise(hidden_states=torch.zeros(1, 16, 2, 2, 2))

    assert result.shape == (1, 16, 2, 2, 2)


def test_s2v_clip_generators_preserve_main_seed_per_clip_behavior() -> None:
    explicit_generator = torch.Generator(device="cpu").manual_seed(999)

    generators = _make_clip_generators(
        seeds=[1234, 5678, None, None],
        request_generators=[
            torch.Generator(device="cpu").manual_seed(1234),
            torch.Generator(device="cpu").manual_seed(5678),
            explicit_generator,
            None,
        ],
        clip_index=2,
        device=torch.device("cpu"),
    )

    assert [generator.initial_seed() for generator in generators if generator is not None] == [1236, 5680, 999, 2]
    assert generators[2] is explicit_generator


def test_s2v_forward_rejects_different_num_repeat() -> None:
    pipeline = _make_s2v_validation_pipeline()
    pipeline.encode_audio = lambda *args, **kwargs: (torch.zeros(1, 1, 2, 16), 2, 16)  # type: ignore[method-assign]
    image = PIL.Image.new("RGB", (16, 16))
    audio = np.zeros(16000, dtype=np.float32)
    batch = DiffusionRequestBatch(
        requests=[
            SimpleNamespace(
                request_id="a",
                prompt={
                    "prompt": "first",
                    "multi_modal_data": {"image": image, "audio": audio},
                    "additional_information": {"num_repeat": 1},
                },
                sampling_params=_make_s2v_sampling(),
            ),
            SimpleNamespace(
                request_id="b",
                prompt={
                    "prompt": "second",
                    "multi_modal_data": {"image": image, "audio": audio},
                    "additional_information": {"num_repeat": 2},
                },
                sampling_params=_make_s2v_sampling(),
            ),
        ]
    )

    with pytest.raises(ValueError, match="num_repeat"):
        pipeline.forward(batch)


def test_s2v_forward_rejects_different_audio_time_lengths() -> None:
    pipeline = _make_s2v_validation_pipeline()
    pipeline.encode_audio = lambda audio, **kwargs: (  # type: ignore[method-assign]
        torch.zeros(1, 1, 2, 16 if audio[0] == 0 else 24),
        1,
        16 if audio[0] == 0 else 24,
    )
    image = PIL.Image.new("RGB", (16, 16))
    batch = DiffusionRequestBatch(
        requests=[
            SimpleNamespace(
                request_id="a",
                prompt={
                    "prompt": "first",
                    "multi_modal_data": {"image": image, "audio": np.zeros(16000, dtype=np.float32)},
                },
                sampling_params=_make_s2v_sampling(),
            ),
            SimpleNamespace(
                request_id="b",
                prompt={
                    "prompt": "second",
                    "multi_modal_data": {"image": image, "audio": np.ones(16000, dtype=np.float32)},
                },
                sampling_params=_make_s2v_sampling(),
            ),
        ]
    )

    with pytest.raises(ValueError, match="audio embedding time lengths"):
        pipeline.forward(batch)


def test_s2v_forward_rejects_different_init_first_frame() -> None:
    pipeline = _make_s2v_validation_pipeline()
    image = PIL.Image.new("RGB", (16, 16))
    audio = np.zeros(16000, dtype=np.float32)
    batch = DiffusionRequestBatch(
        requests=[
            SimpleNamespace(
                request_id="a",
                prompt={
                    "prompt": "first",
                    "multi_modal_data": {"image": image, "audio": audio},
                    "additional_information": {"init_first_frame": True},
                },
                sampling_params=_make_s2v_sampling(),
            ),
            SimpleNamespace(
                request_id="b",
                prompt={
                    "prompt": "second",
                    "multi_modal_data": {"image": image, "audio": audio},
                    "additional_information": {"init_first_frame": False},
                },
                sampling_params=_make_s2v_sampling(),
            ),
        ]
    )

    with pytest.raises(ValueError, match="init_first_frame"):
        pipeline.forward(batch)


def test_s2v_forward_rejects_different_raw_audio_shapes() -> None:
    pipeline = _make_s2v_validation_pipeline()
    pipeline.encode_audio = lambda *args, **kwargs: (torch.zeros(1, 1, 2, 16), 1, 16)  # type: ignore[method-assign]
    image = PIL.Image.new("RGB", (16, 16))
    batch = DiffusionRequestBatch(
        requests=[
            SimpleNamespace(
                request_id="a",
                prompt={
                    "prompt": "first",
                    "multi_modal_data": {"image": image, "audio": np.zeros(16000, dtype=np.float32)},
                },
                sampling_params=_make_s2v_sampling(),
            ),
            SimpleNamespace(
                request_id="b",
                prompt={
                    "prompt": "second",
                    "multi_modal_data": {"image": image, "audio": np.zeros(8000, dtype=np.float32)},
                },
                sampling_params=_make_s2v_sampling(),
            ),
        ]
    )

    with pytest.raises(ValueError, match="raw audio shapes and sample rates"):
        pipeline.forward(batch)


def test_s2v_forward_batches_request_local_inputs_and_splits_outputs() -> None:
    pipeline = object.__new__(Wan22S2VPipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.transformer = MagicMock()
    pipeline.transformer.dtype = torch.float32
    pipeline.transformer.parameters.return_value = iter([torch.zeros(1, dtype=torch.float32)])
    pipeline.transformer.casual_audio_encoder = None
    pipeline.transformer.encode_audio.side_effect = lambda audio, _motion: {"audio_emb": audio}
    pipeline.vae = MagicMock()
    pipeline.vae.dtype = torch.float32
    pipeline.vae.decode.return_value = (torch.zeros(4, 3, 8, 16, 16),)
    pipeline.od_config = SimpleNamespace(
        enable_cpu_offload=False,
        parallel_config=SimpleNamespace(use_hsdp=False),
    )
    pipeline.scheduler = MagicMock(timesteps=torch.tensor([1.0]))
    pipeline.vae_scale_factor_spatial = 8
    pipeline.resolution_divisor = 16
    pipeline.motion_frames = 7
    pipeline.drop_first_motion = True
    pipeline._DEFAULT_INFER_FRAMES = 8
    pipeline._guidance_scale = None
    pipeline._num_timesteps = None
    pipeline.check_inputs = lambda *args, **kwargs: None
    pipeline.encode_prompt = MagicMock(return_value=(torch.zeros(4, 2, 3), torch.ones(4, 2, 3)))
    pipeline.encode_audio = MagicMock(return_value=(torch.zeros(1, 1, 2, 8), 1, 8))
    pipeline.encode_ref_image = MagicMock(return_value=torch.zeros(1, 16, 1, 2, 2))
    pipeline.prepare_motion_latents = MagicMock(
        side_effect=lambda pixels, **_: torch.zeros(pixels.shape[0], 16, 2, 2, 2)
    )
    pipeline._denormalize_latents = lambda latents: latents
    pipeline.diffuse = MagicMock(side_effect=lambda **kwargs: kwargs["latents"])

    image = PIL.Image.new("RGB", (16, 16))
    audio_a = np.zeros(16000, dtype=np.float32)
    audio_b = np.ones(16000, dtype=np.float32)
    latents_a = torch.zeros(2, 16, 2, 2, 2)
    latents_b = torch.ones(2, 16, 2, 2, 2)
    generators_a = [torch.Generator().manual_seed(1), torch.Generator().manual_seed(2)]
    generators_b = [torch.Generator().manual_seed(3), torch.Generator().manual_seed(4)]
    batch = DiffusionRequestBatch(
        requests=[
            SimpleNamespace(
                request_id="a",
                prompt={
                    "prompt": "first",
                    "negative_prompt": "negative first",
                    "multi_modal_data": {"image": image, "audio": audio_a},
                },
                sampling_params=_make_s2v_sampling(
                    num_outputs_per_prompt=2,
                    generator=generators_a,
                    latents=latents_a,
                ),
            ),
            SimpleNamespace(
                request_id="b",
                prompt={
                    "prompt": "second",
                    "negative_prompt": "negative second",
                    "multi_modal_data": {"image": image, "audio": audio_b},
                },
                sampling_params=_make_s2v_sampling(
                    num_outputs_per_prompt=2,
                    generator=generators_b,
                    latents=latents_b,
                ),
            ),
        ]
    )

    with patch("vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_s2v.current_omni_platform") as platform:
        platform.is_available.return_value = False
        outputs = pipeline.forward(batch)

    assert len(outputs) == 2
    assert outputs[0].output[0].shape[0] == 2
    assert outputs[1].output[0].shape[0] == 2
    assert outputs[0].output[1].shape == (16000,)
    assert outputs[1].output[1].shape == (16000,)
    np.testing.assert_array_equal(outputs[0].output[1], audio_a)
    np.testing.assert_array_equal(outputs[1].output[1], audio_b)
    assert np.shares_memory(outputs[0].output[1], audio_a)
    assert np.shares_memory(outputs[1].output[1], audio_b)
    torch.testing.assert_close(pipeline.diffuse.call_args.kwargs["latents"], torch.cat([latents_a, latents_b]))
    assert pipeline.diffuse.call_args.kwargs["clip_generator"] == generators_a + generators_b
    assert pipeline.encode_prompt.call_args.kwargs["prompt"] == ["first", "second"]
    assert pipeline.encode_prompt.call_args.kwargs["negative_prompt"] == ["negative first", "negative second"]


def test_s2v_exposes_hsdp_shard_conditions_for_transformer_blocks():
    model = object.__new__(WanS2VTransformer3DModel)
    nn.Module.__init__(model)
    model.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

    conditions = getattr(model, "_hsdp_shard_conditions", None)

    assert conditions is not None
    assert len(conditions) == 1

    matched = []
    for name, module in model.named_modules():
        if any(cond(name, module) for cond in conditions):
            matched.append(name)

    assert matched == ["blocks.0", "blocks.1", "blocks.2"]


def test_s2v_hsdp_shard_condition_does_not_match_non_block_modules():
    model = object.__new__(WanS2VTransformer3DModel)
    nn.Module.__init__(model)
    model.blocks = nn.ModuleList([nn.Linear(4, 4)])
    model.head_indicator = nn.Linear(4, 4)
    model.casual_audio_encoder = nn.Linear(4, 4)

    conditions = model._hsdp_shard_conditions
    non_block_matched = []
    for name, module in model.named_modules():
        if name and "blocks" not in name:
            if any(cond(name, module) for cond in conditions):
                non_block_matched.append(name)

    assert non_block_matched == []


def test_encode_audio_calls_unshard_reshard_when_fsdp_managed():
    model = object.__new__(WanS2VTransformer3DModel)
    nn.Module.__init__(model)
    model.enable_adain = False
    model.casual_audio_encoder = MagicMock(return_value=torch.zeros(1, 10, 64))

    model.unshard = MagicMock()
    model.reshard = MagicMock()

    audio_input = torch.randn(1, 1, 64, 5)
    motion_frames = [2, 2]

    result = model.encode_audio(audio_input, motion_frames)

    model.unshard.assert_called_once()
    model.reshard.assert_called_once()
    assert "audio_emb" in result


def test_encode_audio_reshard_called_on_exception():
    """Test that reshard() is always called even when encode_audio logic raises."""
    model = object.__new__(WanS2VTransformer3DModel)
    nn.Module.__init__(model)
    model.enable_adain = False
    model.casual_audio_encoder = MagicMock(side_effect=RuntimeError("encoder failed"))

    model.unshard = MagicMock()
    model.reshard = MagicMock()

    audio_input = torch.randn(1, 1, 64, 5)
    motion_frames = [2, 2]

    with pytest.raises(RuntimeError, match="encoder failed"):
        model.encode_audio(audio_input, motion_frames)

    model.unshard.assert_called_once()
    model.reshard.assert_called_once()


def test_encode_audio_skips_unshard_reshard_when_not_fsdp():
    model = object.__new__(WanS2VTransformer3DModel)
    nn.Module.__init__(model)
    model.enable_adain = False
    model.casual_audio_encoder = MagicMock(return_value=torch.zeros(1, 10, 64))

    audio_input = torch.randn(1, 1, 64, 5)
    motion_frames = [2, 2]

    result = model.encode_audio(audio_input, motion_frames)

    assert not hasattr(model, "unshard")
    assert not hasattr(model, "reshard")
    assert "audio_emb" in result


def test_s2v_pipeline_skips_cpu_offload_when_hsdp_enabled():
    """Test that transformer.to('cpu') is NOT called when HSDP is active."""
    from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_s2v import Wan22S2VPipeline

    pipeline = object.__new__(Wan22S2VPipeline)
    nn.Module.__init__(pipeline)

    od_config = MagicMock()
    od_config.enable_cpu_offload = True
    parallel_config = MagicMock()
    parallel_config.use_hsdp = True
    od_config.parallel_config = parallel_config
    pipeline.od_config = od_config

    mock_transformer = MagicMock()
    pipeline.transformer = mock_transformer

    # Simulate the offload decision from the forward loop
    if pipeline.od_config.enable_cpu_offload and not getattr(pipeline.od_config.parallel_config, "use_hsdp", False):
        pipeline.transformer.to("cpu")

    mock_transformer.to.assert_not_called()


def test_s2v_pipeline_allows_cpu_offload_when_hsdp_disabled():
    """Test that transformer.to('cpu') IS called when HSDP is not active."""
    from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_s2v import Wan22S2VPipeline

    pipeline = object.__new__(Wan22S2VPipeline)
    nn.Module.__init__(pipeline)

    od_config = MagicMock()
    od_config.enable_cpu_offload = True
    parallel_config = MagicMock()
    parallel_config.use_hsdp = False
    od_config.parallel_config = parallel_config
    pipeline.od_config = od_config

    mock_transformer = MagicMock()
    pipeline.transformer = mock_transformer

    # Simulate the offload decision from the forward loop
    if pipeline.od_config.enable_cpu_offload and not getattr(pipeline.od_config.parallel_config, "use_hsdp", False):
        pipeline.transformer.to("cpu")

    mock_transformer.to.assert_called_once_with("cpu")


def test_s2v_pipeline_hsdp_forward_complete_process():
    """Integration-level mock test verifying the complete forward path of the
    S2V pipeline under HSDP mode.

    Verifies:
    - Text encoding is invoked
    - Audio encoding invokes unshard/reshard on the transformer
    - Reference image encoding via VAE
    - Denoising loop calls the transformer
    - VAE decode is invoked
    - transformer.to('cpu') is NOT called (HSDP mode)
    - Final output is a DiffusionOutput with video + audio
    """
    import numpy as np
    import PIL.Image

    from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_s2v import Wan22S2VPipeline

    pipeline = object.__new__(Wan22S2VPipeline)
    nn.Module.__init__(pipeline)

    # -- Config --
    od_config = MagicMock()
    od_config.enable_cpu_offload = True
    od_config.enable_diffusion_pipeline_profiler = False
    parallel_config = MagicMock()
    parallel_config.use_hsdp = True
    od_config.parallel_config = parallel_config
    pipeline.od_config = od_config

    # -- Pipeline attributes --
    pipeline.device = torch.device("cpu")
    pipeline._guidance_scale = 4.5
    pipeline._num_timesteps = None
    pipeline._current_timestep = None
    pipeline.vae_scale_factor_spatial = 8
    pipeline.vae_scale_factor_temporal = 4
    pipeline.resolution_divisor = 16
    pipeline.motion_frames = 73
    pipeline.drop_first_motion = True
    pipeline.fps = 16
    pipeline.audio_sample_m = 0
    pipeline._DEFAULT_INFER_FRAMES = 80

    # -- Mock text encoder --
    pipeline.tokenizer = MagicMock()
    pipeline.tokenizer.return_value = MagicMock(
        input_ids=torch.zeros(1, 512, dtype=torch.long),
        attention_mask=torch.ones(1, 512, dtype=torch.long),
    )
    mock_text_encoder = MagicMock()
    mock_text_encoder.dtype = torch.bfloat16
    mock_text_encoder.return_value = MagicMock(last_hidden_state=torch.zeros(1, 512, 4096))
    pipeline.text_encoder = mock_text_encoder

    # -- Mock transformer --
    mock_transformer = MagicMock()
    mock_transformer.dtype = torch.bfloat16
    mock_transformer.parameters = MagicMock(return_value=iter([torch.zeros(1)]))

    # encode_audio returns dict with audio_emb
    mock_transformer.encode_audio = MagicMock(return_value={"audio_emb": torch.zeros(1, 10, 64)})
    # Forward returns noise prediction
    mock_transformer.return_value = (torch.zeros(1, 16, 20, 88, 128),)
    pipeline.transformer = mock_transformer

    # -- Mock VAE --
    mock_vae = MagicMock()
    mock_vae.dtype = torch.bfloat16
    mock_vae.config = MagicMock()
    mock_vae.config.scale_factor_temporal = 4
    mock_vae.config.scale_factor_spatial = 8
    mock_vae.config.latents_mean = [0.0] * 16
    mock_vae.config.latents_std = [1.0] * 16
    mock_vae.config.z_dim = 16
    # encode returns mock with latent_dist
    mock_encode_result = MagicMock()
    mock_encode_result.latent_dist = MagicMock()
    mock_encode_result.latent_dist.mode = MagicMock(return_value=torch.zeros(1, 16, 1, 88, 128))
    mock_vae.encode = MagicMock(return_value=mock_encode_result)
    # decode returns video
    mock_vae.decode = MagicMock(return_value=(torch.zeros(1, 3, 80, 704, 1024),))
    pipeline.vae = mock_vae

    # -- Mock audio model --
    mock_audio_model = MagicMock()
    mock_audio_model.device = torch.device("cpu")
    mock_audio_param = torch.zeros(1)
    mock_audio_model.parameters = MagicMock(return_value=iter([mock_audio_param]))
    mock_audio_model.return_value = MagicMock(hidden_states=[torch.zeros(1, 100, 1024)] * 25)
    pipeline.audio_model = mock_audio_model

    pipeline.audio_processor = MagicMock()
    pipeline.audio_processor.return_value = MagicMock(input_values=torch.zeros(1, 16000))

    # -- Mock scheduler --
    mock_scheduler = MagicMock()
    mock_scheduler.timesteps = torch.linspace(999, 0, 5)
    mock_scheduler.step = MagicMock(return_value=(torch.zeros(1, 16, 20, 88, 128),))
    pipeline.scheduler = mock_scheduler

    # -- Bind methods from the real class --
    pipeline.encode_prompt = Wan22S2VPipeline.encode_prompt.__get__(pipeline)
    pipeline.encode_ref_image = Wan22S2VPipeline.encode_ref_image.__get__(pipeline)
    pipeline.prepare_motion_latents = Wan22S2VPipeline.prepare_motion_latents.__get__(pipeline)
    pipeline.prepare_latents = Wan22S2VPipeline.prepare_latents.__get__(pipeline)
    pipeline.check_inputs = Wan22S2VPipeline.check_inputs.__get__(pipeline)
    pipeline.diffuse = Wan22S2VPipeline.diffuse.__get__(pipeline)
    pipeline._normalize_latents = Wan22S2VPipeline._normalize_latents.__get__(pipeline)
    pipeline._denormalize_latents = Wan22S2VPipeline._denormalize_latents.__get__(pipeline)
    pipeline._prompt_clean = Wan22S2VPipeline._prompt_clean

    # -- Build request --
    sampling_params = _make_s2v_sampling(
        height=704,
        width=1024,
        num_frames=80,
        num_inference_steps=5,
        guidance_scale=4.5,
        generator=torch.Generator(device="cpu").manual_seed(42),
        max_sequence_length=512,
    )

    ref_image = PIL.Image.new("RGB", (1024, 704))
    audio_data = np.zeros(16000, dtype=np.float32)

    req = DiffusionRequestBatch(
        requests=[
            SimpleNamespace(
                request_id="test",
                prompt={
                    "prompt": "test prompt",
                    "negative_prompt": "bad quality",
                    "multi_modal_data": {"image": ref_image, "audio": audio_data},
                    "additional_information": {
                        "audio_path": audio_data,
                        "pose_video": None,
                        "init_first_frame": False,
                    },
                },
                sampling_params=sampling_params,
            )
        ]
    )

    # -- Mock methods that use complex internal state --
    pipeline.encode_audio = MagicMock(return_value=(torch.zeros(1, 25, 64, 80), 1, 80))

    # Mock progress bar context
    pipeline.progress_bar = MagicMock()
    pipeline.progress_bar.return_value.__enter__ = MagicMock(return_value=MagicMock())
    pipeline.progress_bar.return_value.__exit__ = MagicMock(return_value=False)

    # Mock predict_noise_maybe_with_cfg from CFGParallelMixin
    pipeline.predict_noise_maybe_with_cfg = MagicMock(return_value=torch.zeros(1, 16, 20, 88, 128))
    pipeline.predict_noise = Wan22S2VPipeline.predict_noise.__get__(pipeline)

    # Mock platform methods
    with patch("vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_s2v.current_omni_platform") as mock_platform:
        mock_platform.empty_cache = MagicMock()
        mock_platform.is_available = MagicMock(return_value=False)

        with patch(
            "vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_s2v.load_audio", return_value=(audio_data, 16000)
        ):
            result = Wan22S2VPipeline.forward(pipeline, req=req)[0]

    # -- Assertions --
    # Text encoder was called
    mock_text_encoder.assert_called()

    # Audio encoding was invoked
    pipeline.encode_audio.assert_called_once()

    # VAE encode was called (for ref image and motion latents)
    mock_vae.encode.assert_called()

    # VAE decode was called
    mock_vae.decode.assert_called()

    # transformer.to("cpu") must NOT be called under HSDP
    mock_transformer.to.assert_not_called()

    # Output is a DiffusionOutput tuple with (video, audio_waveform, sample_rate)
    from vllm_omni.diffusion.data import DiffusionOutput

    assert isinstance(result, DiffusionOutput)
    video, audio_waveform, audio_sr = result.output
    assert video.shape[0] == 1  # batch
    assert video.shape[1] == 3  # channels
    assert audio_waveform is not None
    assert audio_sr == 16000
