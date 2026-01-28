"""
Tests for Ovis Image model pipeline.

Strategy:
1. `mock_dependencies` fixture mocks heavy external components (VAE, Scheduler, TextEncoder)
   to allow fast testing of the pipeline logic without downloading weights.
   - Mocks are configured to return tensors on the correct device.
   - Transformer is mocked dynamically to return random noise of correct shape.

2. `test_real_transformer_init_and_forward` tests the actual `OvisImageTransformer2DModel`
   initialization and forward pass with a small configuration to ensure code coverage
   and correctness of the model definition itself, independent of the pipeline mocks.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig

# Mock the OvisImageTransformer2DModel to avoid complex init if needed,
# or let it run if it's lightweight. It's likely not lightweight.
# Better to mock the transformer forwarding to return random noise.
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.ovis_image.pipeline_ovis_image import OvisImagePipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


@pytest.fixture
def mock_dependencies(monkeypatch):
    """
    Mock external dependencies to avoid loading real models.
    """
    device = get_local_device()

    # Mock Tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = MagicMock(
        input_ids=torch.zeros((1, 50), dtype=torch.long, device=device),
        attention_mask=torch.ones((1, 50), dtype=torch.long, device=device),
    )
    mock_tokenizer.apply_chat_template.return_value = "dummy prompt"
    mock_tokenizer.model_max_length = 1024

    # Mock Text Encoder
    mock_text_encoder = MagicMock()
    mock_text_encoder.dtype = torch.float32
    # Output of text encoder must be on the same device as inputs (which are moved to execution_device)
    mock_text_encoder.return_value.last_hidden_state = torch.randn(1, 50, 32, device=device)

    # Mock VAE
    mock_vae = MagicMock()
    mock_vae.config.block_out_channels = [128, 256, 512, 512]  # Scale factor 8
    mock_vae.config.scale_factor_temporal = 1
    mock_vae.config.scale_factor_spatial = 8
    mock_vae.config.scaling_factor = 0.18215
    mock_vae.config.shift_factor = 0.0
    # Decode return value
    mock_vae.decode.return_value = [torch.randn(1, 3, 128, 128, device=device)]
    # Ensure .to() returns self so configuration persists
    mock_vae.to.return_value = mock_vae

    # Mock Scheduler
    mock_scheduler = MagicMock()
    mock_scheduler.config = MagicMock()
    # Timesteps on device to match latents during denoising loop interaction if needed
    mock_scheduler.timesteps = torch.tensor([1.0, 0.5, 0.0], device=device)
    mock_scheduler.set_timesteps.return_value = None

    # Make step return dynamic based on input sample shape
    def mock_scheduler_step(model_output, timestep, sample, **kwargs):
        # sample is the latents, should be preserved
        return (torch.randn_like(sample),)

    mock_scheduler.step.side_effect = mock_scheduler_step

    module_path = "vllm_omni.diffusion.models.ovis_image.pipeline_ovis_image"

    monkeypatch.setattr(f"{module_path}.Qwen2TokenizerFast.from_pretrained", lambda *a, **k: mock_tokenizer)
    monkeypatch.setattr(f"{module_path}.Qwen3Model.from_pretrained", lambda *a, **k: mock_text_encoder)
    monkeypatch.setattr(f"{module_path}.AutoencoderKL.from_pretrained", lambda *a, **k: mock_vae)
    monkeypatch.setattr(
        f"{module_path}.FlowMatchEulerDiscreteScheduler.from_pretrained", lambda *a, **k: mock_scheduler
    )

    return {
        "tokenizer": mock_tokenizer,
        "text_encoder": mock_text_encoder,
        "vae": mock_vae,
        "scheduler": mock_scheduler,
        "device": device,
    }


@pytest.fixture
def ovis_pipeline(mock_dependencies, monkeypatch):
    """
    Creates an OvisImagePipeline instance with mocked components.
    """
    # Create config
    tf_config = TransformerConfig(
        params={
            "in_channels": 4,
            "out_channels": 4,
            "sample_size": 32,
            "patch_size": 2,
            "num_attention_heads": 4,
            "attention_head_dim": 8,
            "num_layers": 1,
            "caption_channels": 32,
        }
    )

    od_config = OmniDiffusionConfig(
        model="dummy-ovis",
        tf_model_config=tf_config,
        dtype=torch.float32,
        num_gpus=1,
    )

    # Mock Transformer Layer separately to avoid full init
    # We patch OvisImageTransformer2DModel class in the module
    mock_transformer_cls = MagicMock()
    mock_transformer_instance = MagicMock()
    mock_transformer_instance.dtype = torch.float32
    mock_transformer_instance.in_channels = 16  # Must be 16 so num_channel_latents=4, packed=16
    # Forward return: noise prediction

    def mock_forward(hidden_states, *args, **kwargs):
        # hidden_states shape: (B, SeqLen, Channels)
        return (torch.randn_like(hidden_states),)

    mock_transformer_instance.forward.side_effect = mock_forward
    # Also make the instance itself callable to mimic __call__
    mock_transformer_instance.side_effect = mock_forward

    mock_transformer_cls.return_value = mock_transformer_instance

    monkeypatch.setattr(
        "vllm_omni.diffusion.models.ovis_image.pipeline_ovis_image.OvisImageTransformer2DModel", mock_transformer_cls
    )

    # Initialize pipeline
    # We use a dummy model path check override
    with patch("os.path.exists", return_value=True):
        pipeline = OvisImagePipeline(od_config=od_config)

    return pipeline


def test_interface_compliance(ovis_pipeline):
    """Verify methods required by vllm-omni framework."""
    assert hasattr(ovis_pipeline, "load_weights")
    assert hasattr(ovis_pipeline, "scheduler")
    assert hasattr(ovis_pipeline, "transformer")
    assert hasattr(ovis_pipeline, "text_encoder")
    # assert hasattr(ovis_pipeline, "vae") # Ovis uses VAE


def test_basic_generation(ovis_pipeline):
    """Test the forward pass logic."""
    # Setup request
    req = OmniDiffusionRequest(
        prompts=["A photo of a cat"],
        sampling_params=OmniDiffusionSamplingParams(
            height=256,
            width=256,
            num_inference_steps=2,
            guidance_scale=1.0,
        ),
    )

    output = ovis_pipeline(req)

    assert output is not None
    assert output.output is not None
    # Output should be a tensor from mocked VAE decode [torch.randn(1, 3, 128, 128)]
    assert isinstance(output.output, torch.Tensor)
    assert output.output.shape == (1, 3, 128, 128)

    # Check that transformer was called
    assert ovis_pipeline.transformer.call_count > 0


def test_guidance_scale(ovis_pipeline):
    """Test that classifier-free guidance path is taken when scale > 1.0."""
    req = OmniDiffusionRequest(
        prompts=[
            {
                "prompt": "A photo of a cat",
                "negative_prompt": "bad quality",
            }
        ],
        sampling_params=OmniDiffusionSamplingParams(
            height=256,
            width=256,
            num_inference_steps=1,
            guidance_scale=2.0,  # Trigger CFG
        ),
    )

    ovis_pipeline(req)
    assert ovis_pipeline.transformer.call_count >= 2


def test_resolution_check(ovis_pipeline):
    """Test resolution divisible validation logic if present."""
    # Pass odd resolution
    req = OmniDiffusionRequest(
        prompts=["test"],
        sampling_params=OmniDiffusionSamplingParams(
            height=250,  # Not divisible by 16 (8*2)
            width=250,
        ),
    )

    # Should warn but proceed (as per code I read earlier) or resize?
    # The code had `logger.warning(...)`

    output = ovis_pipeline(req)
    assert output is not None


def test_real_transformer_init_and_forward():
    """Test the real OvisImageTransformer2DModel initialization and forward pass for coverage."""
    from unittest.mock import patch

    from vllm_omni.diffusion.models.ovis_image.ovis_image_transformer import OvisImageTransformer2DModel

    device = get_local_device()
    tf_config = TransformerConfig(
        params={
            "patch_size": 2,
            "in_channels": 16,
            "out_channels": 16,
            "num_layers": 1,
            "num_single_layers": 1,
            "attention_head_dim": 8,
            "num_attention_heads": 2,
            "joint_attention_dim": 32,
            "axes_dims_rope": (4, 4, 4),
        }
    )

    od_config = OmniDiffusionConfig(model="dummy-ovis", tf_model_config=tf_config, dtype=torch.bfloat16, num_gpus=1)
    torch.set_default_dtype(torch.bfloat16)

    # Mock distributed state for QKVParallelLinear initialization
    # We patch get_tp_group because get_tensor_model_parallel_rank calls it and asserts _TP is not None
    mock_group = MagicMock()
    mock_group.rank_in_group = 0
    mock_group.world_size = 1

    with patch("vllm.distributed.parallel_state.get_tp_group", return_value=mock_group):
        # Initialize real model
        model = OvisImageTransformer2DModel(
            od_config=od_config,
            patch_size=1,
            in_channels=16,
            out_channels=16,
            num_single_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=32,
            axes_dims_rope=(2, 2, 4),
        ).to(device)

        # Create dummy inputs
        B, Seq, C = 1, 16, 16
        hidden_states = torch.randn(B, Seq, C, device=device)
        encoder_hidden_states = torch.randn(B, 10, 32, device=device)  # joint_attention_dim=32
        timestep = torch.tensor([1], device=device)
        img_ids = torch.zeros(Seq, 3, device=device)
        txt_ids = torch.zeros(10, 3, device=device)

        # Run forward
        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            return_dict=False,
        )

        assert output is not None
        assert isinstance(output, tuple)
        assert output[0].shape == hidden_states.shape
