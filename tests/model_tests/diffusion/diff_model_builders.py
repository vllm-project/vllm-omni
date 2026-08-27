from functools import partial
from pathlib import Path

import torch
from diffusers import (
    AutoencoderKLWan,
    DPMSolverMultistepScheduler,
    SanaVideoPipeline,
    SanaVideoTransformer3DModel,
)
from transformers import Gemma2Config, Gemma2Model, GemmaTokenizerFast

from tests.helpers.tiny_model import _get_tiny_model_path, build_tiny_from_configs

TINY_CONFIGS_DIR = Path(__file__).parent / "tiny_configs"


def _shrink_dit_rope_config(
    config: dict,
    num_layers: int = 2,
    num_single_layers: int | None = None,
    target_head_dim: int = 32,
    target_heads: int = 4,
    default_axes_dims_rope: list[int] | None = None,
    joint_attention_dim: int | None = None,
) -> dict:
    config["num_layers"] = num_layers
    if num_single_layers is not None:
        config["num_single_layers"] = num_single_layers
    if joint_attention_dim is not None:
        config["joint_attention_dim"] = joint_attention_dim
    axes_dims_rope = config.get("axes_dims_rope", default_axes_dims_rope)
    factor = config["attention_head_dim"] / target_head_dim
    config["attention_head_dim"] = target_head_dim
    config["num_attention_heads"] = target_heads
    config["axes_dims_rope"] = [int(d / factor) for d in axes_dims_rope]
    return config


def tiny_flux2_klein_builder() -> str:
    """Build a tiny Flux2Klein model from vendored configs."""
    return build_tiny_from_configs(
        "Flux2KleinPipeline", "black-forest-labs/FLUX.2-klein-4B", TINY_CONFIGS_DIR / "Flux2KleinPipeline"
    )


def tiny_ltx2_builder() -> str:
    """Build a tiny LTX2 model from vendored configs."""
    return build_tiny_from_configs("LTX2Pipeline", "Lightricks/LTX-2", TINY_CONFIGS_DIR / "LTX2Pipeline")


def tiny_sana_video_builder() -> str:
    """Build a tiny 480p SANA-Video model without downloading model weights.

    The tokenizer is the only component loaded from the upstream repository.
    All modules with weights are initialized locally from intentionally small
    configs, and the scheduler is constructed from its weight-free config.
    """
    model_id = "Efficient-Large-Model/SANA-Video_2B_480p_diffusers"
    model_dir = _get_tiny_model_path("SanaVideoPipeline")

    tokenizer = GemmaTokenizerFast.from_pretrained(model_id, subfolder="tokenizer")
    scheduler = DPMSolverMultistepScheduler(
        algorithm_type="dpmsolver++",
        flow_shift=8.0,
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
    )
    text_encoder = Gemma2Model(
        Gemma2Config(
            vocab_size=256000,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=512,
            sliding_window=256,
            layer_types=["sliding_attention"],
        )
    )
    transformer = SanaVideoTransformer3DModel(
        in_channels=16,
        out_channels=16,
        num_attention_heads=2,
        attention_head_dim=16,
        num_layers=1,
        num_cross_attention_heads=2,
        cross_attention_head_dim=16,
        cross_attention_dim=32,
        caption_channels=32,
        mlp_ratio=2.0,
        sample_size=30,
        patch_size=(1, 2, 2),
    )
    vae = AutoencoderKLWan(
        base_dim=8,
        decoder_base_dim=8,
        z_dim=16,
        dim_mult=[1, 1, 1, 1],
        num_res_blocks=1,
        temperal_downsample=[False, True, True],
        latents_mean=[0.0] * 16,
        latents_std=[1.0] * 16,
    )
    pipeline = SanaVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
    )
    pipeline.to(dtype=torch.bfloat16).save_pretrained(model_dir)
    return model_dir


def _shrink_flux_clip_text_encoder(config: dict) -> dict:
    config["num_hidden_layers"] = 2
    return config


def _shrink_flux_t5_text_encoder(config: dict) -> dict:
    config["num_layers"] = 2
    config["num_decoder_layers"] = 2
    config["d_ff"] = 64
    config["num_heads"] = 4
    return config


def _shrink_qwen_text_encoder_config(config: dict, hidden_size: int = 64) -> dict:
    """Shrink a Qwen2.5-VL text encoder (nested Qwen-Image or flat LongCat layout)."""
    # Nested: text fields live under text_config, with top-level mirrors.
    # Flat: text fields live at the top level.
    text_config = config["text_config"] if "text_config" in config else config
    old_head_dim = text_config["hidden_size"] / text_config["num_attention_heads"]
    text_config["num_hidden_layers"] = 2
    text_config["intermediate_size"] = 64
    text_config["hidden_size"] = hidden_size
    text_config["num_attention_heads"] = 2
    text_config["num_key_value_heads"] = 2
    if "layer_types" in text_config:
        text_config["layer_types"] = text_config["layer_types"][:2]
    factor = old_head_dim / (hidden_size / 2)
    mrope_section = text_config["rope_scaling"]["mrope_section"]
    text_config["rope_scaling"]["mrope_section"] = [round(d / factor) for d in mrope_section]
    config["num_hidden_layers"] = 2
    config["intermediate_size"] = 64
    config["hidden_size"] = hidden_size
    config["num_attention_heads"] = 2
    config["num_key_value_heads"] = 2
    config["vision_config"]["depth"] = 2
    config["vision_config"]["intermediate_size"] = 64
    config["vision_config"]["fullatt_block_indexes"] = [0, 1]
    config["vision_config"]["out_hidden_size"] = hidden_size
    return config


def _shrink_qwen_transformer_config(config: dict, joint_attention_dim: int = 64) -> dict:
    return _shrink_dit_rope_config(config, joint_attention_dim=joint_attention_dim)


def tiny_qwen_image_builder() -> str:
    return build_tiny_from_configs(
        "QwenImagePipeline",
        "Qwen/Qwen-Image",
        transform={"text_encoder": _shrink_qwen_text_encoder_config, "transformer": _shrink_qwen_transformer_config},
    )


def tiny_qwen_image_edit_builder() -> str:
    return build_tiny_from_configs(
        "QwenImageEditPipeline",
        "Qwen/Qwen-Image-Edit",
        transform={"text_encoder": _shrink_qwen_text_encoder_config, "transformer": _shrink_qwen_transformer_config},
    )


def tiny_qwen_image_edit_plus_builder() -> str:
    return build_tiny_from_configs(
        "QwenImageEditPlusPipeline",
        "Qwen/Qwen-Image-Edit-2511",
        transform={"text_encoder": _shrink_qwen_text_encoder_config, "transformer": _shrink_qwen_transformer_config},
    )


def tiny_longcat_image_builder() -> str:
    return build_tiny_from_configs(
        "LongCatImagePipeline",
        "meituan-longcat/LongCat-Image",
        transform={
            "text_encoder": _shrink_qwen_text_encoder_config,
            "transformer": partial(
                _shrink_dit_rope_config,
                num_single_layers=2,
                default_axes_dims_rope=[16, 56, 56],
                joint_attention_dim=64,
            ),
        },
    )


def tiny_longcat_image_edit_builder() -> str:
    return build_tiny_from_configs(
        "LongCatImageEditPipeline",
        "meituan-longcat/LongCat-Image-Edit",
        transform={
            "text_encoder": _shrink_qwen_text_encoder_config,
            "transformer": partial(
                _shrink_dit_rope_config,
                num_single_layers=2,
                default_axes_dims_rope=[16, 56, 56],
                joint_attention_dim=64,
            ),
        },
    )


def tiny_flux_builder() -> str:
    return build_tiny_from_configs(
        "FluxPipeline",
        "black-forest-labs/FLUX.1-schnell",
        transform={
            "text_encoder": _shrink_flux_clip_text_encoder,
            "text_encoder_2": _shrink_flux_t5_text_encoder,
            "transformer": partial(_shrink_dit_rope_config, num_single_layers=2, default_axes_dims_rope=[16, 56, 56]),
        },
    )


def tiny_flux_kontext_builder() -> str:
    return build_tiny_from_configs(
        "FluxKontextPipeline",
        "black-forest-labs/FLUX.1-Kontext-dev",
        transform={
            "text_encoder": _shrink_flux_clip_text_encoder,
            "text_encoder_2": _shrink_flux_t5_text_encoder,
            "transformer": partial(_shrink_dit_rope_config, num_single_layers=2, default_axes_dims_rope=[16, 56, 56]),
        },
    )


def tiny_flux2_builder() -> str:
    def shrink_text_encoder(config: dict) -> dict:
        # The real checkpoint ships language_model.lm_head.weight as its own
        # tensor even though tie_word_embeddings defaults to True; keep it
        # untied here too so transformers doesn't drop it on save.
        config["tie_word_embeddings"] = False
        config["text_config"]["num_hidden_layers"] = 31
        config["text_config"]["intermediate_size"] = 64
        config["text_config"]["num_attention_heads"] = 4
        config["text_config"]["head_dim"] = 32
        config["text_config"]["num_key_value_heads"] = 2
        config["vision_config"]["num_hidden_layers"] = 2
        config["vision_config"]["intermediate_size"] = 64
        config["vision_config"]["num_attention_heads"] = 4
        config["vision_config"]["head_dim"] = 32
        return config

    return build_tiny_from_configs(
        "Flux2Pipeline",
        "black-forest-labs/FLUX.2-dev",
        transform={
            "text_encoder": shrink_text_encoder,
            "transformer": partial(_shrink_dit_rope_config, num_single_layers=2),
        },
    )
