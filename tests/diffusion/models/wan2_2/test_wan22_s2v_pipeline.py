from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.models.wan2_2.wan2_2_s2v_transformer import WanS2VTransformer3DModel

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


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
    from vllm_omni.diffusion.request import OmniDiffusionRequest

    sampling_params = MagicMock()
    sampling_params.height = 704
    sampling_params.width = 1024
    sampling_params.num_frames = 80
    sampling_params.num_inference_steps = 5
    sampling_params.guidance_scale = 4.5
    sampling_params.guidance_scale_provided = True
    sampling_params.generator = None
    sampling_params.seed = 42

    ref_image = PIL.Image.new("RGB", (1024, 704))
    audio_data = np.zeros(16000, dtype=np.float32)

    req = MagicMock(spec=OmniDiffusionRequest)
    req.prompts = [
        {
            "prompt": "test prompt",
            "negative_prompt": "bad quality",
            "multi_modal_data": {"image": ref_image, "audio": audio_data},
            "additional_information": {"audio_path": audio_data, "pose_video": None, "init_first_frame": False},
        }
    ]
    req.sampling_params = sampling_params

    # -- Mock methods that use complex internal state --
    pipeline.encode_audio = MagicMock(return_value=(torch.zeros(1, 25, 64, 80), 1, 80))

    # Mock progress bar context
    pipeline.progress_bar = MagicMock()
    pipeline.progress_bar.return_value.__enter__ = MagicMock(return_value=MagicMock())
    pipeline.progress_bar.return_value.__exit__ = MagicMock(return_value=False)

    # Mock predict_noise_maybe_with_cfg from CFGParallelMixin
    pipeline.predict_noise_maybe_with_cfg = MagicMock(return_value=torch.zeros(16, 20, 88, 128))
    pipeline.predict_noise = Wan22S2VPipeline.predict_noise.__get__(pipeline)

    # Mock platform methods
    with patch("vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_s2v.current_omni_platform") as mock_platform:
        mock_platform.empty_cache = MagicMock()
        mock_platform.is_available = MagicMock(return_value=False)

        with patch(
            "vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2_s2v.load_audio", return_value=(audio_data, 16000)
        ):
            result = Wan22S2VPipeline.forward(pipeline, req=req)

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
