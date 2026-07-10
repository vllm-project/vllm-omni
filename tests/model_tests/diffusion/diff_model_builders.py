import os
import tempfile

import torch
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

TINY_MODEL_DIR = os.path.join(tempfile.gettempdir(), "vllm-omni-tiny-models")


def _get_tiny_model_path(name: str) -> str:
    path = os.path.join(TINY_MODEL_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path


def tiny_flux2_klein_builder() -> str:
    """Build a tiny Flux2Klein model."""
    model_dir = _get_tiny_model_path("Flux2KleinPipeline")

    pipe = Flux2KleinPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(),
        vae=AutoencoderKLFlux2(
            down_block_types=("DownEncoderBlock2D",),
            up_block_types=("UpDecoderBlock2D",),
            block_out_channels=(32,),
            layers_per_block=1,
            latent_channels=16,
            norm_num_groups=16,
        ),
        # NOTE: For now we need 28 layers because of hardcoded stuff in the model :(
        text_encoder=Qwen3ForCausalLM(
            Qwen3Config(
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=28,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=16,
                vocab_size=151936,
                max_position_embeddings=512,
            )
        ),
        tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B"),
        # NOTE: For now we need at least 2 layers for the transformer
        # due to hardcoded hacks in CacheDiT for Flux2Klein specifically.
        transformer=Flux2Transformer2DModel(
            in_channels=64,
            num_layers=2,
            num_single_layers=2,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=96,
            timestep_guidance_channels=32,
            axes_dims_rope=(4, 4, 4, 4),
        ),
    )
    # Need dtypes to be consistent; for now we just put it on bfloat16
    pipe.to(torch.bfloat16).save_pretrained(model_dir)
    return model_dir
