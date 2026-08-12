from functools import partial
from pathlib import Path

from tests.helpers.tiny_model import build_tiny_from_configs

TINY_CONFIGS_DIR = Path(__file__).parent / "tiny_configs"


def _shrink_dit_rope_config(
    config: dict,
    num_layers: int = 2,
    num_single_layers: int | None = None,
    target_head_dim: int = 32,
    target_heads: int = 4,
    default_axes_dims_rope: list[int] | None = None,
) -> dict:
    config["num_layers"] = num_layers
    if num_single_layers is not None:
        config["num_single_layers"] = num_single_layers
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


def _shrink_flux_clip_text_encoder(config: dict) -> dict:
    config["num_hidden_layers"] = 2
    return config


def _shrink_flux_t5_text_encoder(config: dict) -> dict:
    config["num_layers"] = 2
    config["num_decoder_layers"] = 2
    config["d_ff"] = 64
    config["num_heads"] = 4
    return config


def tiny_qwen_image_builder() -> str:
    def shrink_text_encoder(config: dict) -> dict:
        config["num_hidden_layers"] = 2
        config["intermediate_size"] = 64
        config["text_config"]["num_hidden_layers"] = 2
        config["text_config"]["intermediate_size"] = 64
        config["text_config"]["layer_types"] = config["text_config"]["layer_types"][:2]
        config["vision_config"]["depth"] = 2
        config["vision_config"]["intermediate_size"] = 64
        config["vision_config"]["fullatt_block_indexes"] = [0, 1]
        return config

    return build_tiny_from_configs(
        "QwenImagePipeline",
        "Qwen/Qwen-Image",
        transform={"text_encoder": shrink_text_encoder, "transformer": _shrink_dit_rope_config},
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
