# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transformers compatibility helpers for Omni-Diffusion's Dream model."""

from __future__ import annotations

import inspect
import sys
from collections.abc import Mapping
from typing import Any

import torch
import transformers
from transformers import GenerationConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from vllm.logger import init_logger

from vllm_omni.model_executor.models.omni_diffusion.utils import (
    OMNI_DIFFUSION_DEFAULT_PARTIAL_ROTARY_FACTOR,
    OMNI_DIFFUSION_DEFAULT_ROPE_THETA,
    OMNI_DIFFUSION_DEFAULT_ROPE_TYPE,
)

logger = init_logger(__name__)

_DREAM_GENERATION_CONFIG_DEFAULTS: Mapping[str, Any] = {
    "eps": 1e-3,
    "steps": 512,
    "alg": "origin",
    "alg_temp": None,
    "num_return_sequences": 1,
    "return_dict_in_generate": False,
    "output_history": False,
}
_DREAM_GENERATION_CONFIG_NONE_DEFAULT_FIELDS = frozenset({"alg_temp"})
_DREAM_GENERATION_CONFIG_TOKEN_FIELDS = ("bos_token_id", "eos_token_id", "pad_token_id", "mask_token_id")


def ensure_dream_rope_parameters(config: Any) -> None:
    """Migrate legacy Dream RoPE fields to Transformers v5 format."""
    model_type = getattr(config, "model_type", None)
    if model_type != "Dream":
        raise ValueError(f"Omni-Diffusion RoPE migration expected Dream model config, got model_type={model_type!r}.")

    rope_parameters = getattr(config, "rope_parameters", None)
    if rope_parameters is not None:
        logger.debug("Dream RoPE parameters already exist, skipping migration: %r", rope_parameters)
        return

    logger.warning(
        "need to modify the config.json to be compatibility with transformers v5. "
        "huggingface model transformers version is 4.51.3 and now transformers version is %s",
        transformers.__version__,
    )

    rope_scaling = getattr(config, "rope_scaling", None)
    rope_theta = getattr(config, "rope_theta", None)
    logger.info("Dream config RoPE fields: rope_scaling=%r rope_theta=%r", rope_scaling, rope_theta)

    rope_parameters = {
        "rope_type": OMNI_DIFFUSION_DEFAULT_ROPE_TYPE,
        "rope_theta": rope_theta if rope_theta is not None else OMNI_DIFFUSION_DEFAULT_ROPE_THETA,
    }

    if rope_scaling is not None:
        if not isinstance(rope_scaling, Mapping):
            raise TypeError(f"Dream config.rope_scaling must be a mapping or None, got {type(rope_scaling)!r}.")
        # Transformers v5 expects all RoPE configuration under
        # config.rope_parameters. This preserves the legacy Dream semantics.
        rope_parameters.update({k: v for k, v in rope_scaling.items() if k != "type"})

        if rope_scaling.get("rope_type") is not None and str(rope_scaling.get("rope_type")).strip():
            rope_parameters["rope_type"] = str(rope_scaling.get("rope_type"))
        elif rope_scaling.get("type") is not None and str(rope_scaling.get("type")).strip():
            rope_parameters["rope_type"] = str(rope_scaling.get("type"))
        else:
            rope_parameters["rope_type"] = OMNI_DIFFUSION_DEFAULT_ROPE_TYPE

    logger.info("Dream RoPE parameters for Transformers v5: %r", rope_parameters)
    config.rope_parameters = rope_parameters


def _get_dream_rope_type(config: Any) -> str | None:
    ensure_dream_rope_parameters(config)
    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, Mapping):
        return str(rope_parameters.get("rope_type", "default"))
    return None


def _compute_default_dream_rope_parameters(
    config: Any,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, float]:
    ensure_dream_rope_parameters(config)
    rope_parameters = getattr(config, "rope_parameters", {})

    if rope_parameters.get("rope_theta") is not None and isinstance(rope_parameters.get("rope_theta"), (float, int)):
        rope_theta = float(rope_parameters.get("rope_theta"))
    elif getattr(config, "rope_theta", None) is not None and isinstance(
        getattr(config, "rope_theta", None), (float, int)
    ):
        rope_theta = float(getattr(config, "rope_theta"))
    else:
        rope_theta = OMNI_DIFFUSION_DEFAULT_ROPE_THETA
    logger.debug("compute_default_dream_rope_parameters, rope_theta = %f", rope_theta)

    if rope_parameters.get("partial_rotary_factor") is not None and isinstance(
        rope_parameters.get("partial_rotary_factor"), (float, int)
    ):
        partial_rotary_factor = float(rope_parameters.get("partial_rotary_factor"))
    elif getattr(config, "partial_rotary_factor", None) is not None and isinstance(
        getattr(config, "partial_rotary_factor", None), (float, int)
    ):
        partial_rotary_factor = float(getattr(config, "partial_rotary_factor"))
    else:
        partial_rotary_factor = OMNI_DIFFUSION_DEFAULT_PARTIAL_ROTARY_FACTOR
    logger.debug("compute_default_dream_rope_parameters, partial_rotary_factor = %f", partial_rotary_factor)

    hidden_size = int(config.hidden_size)
    num_attention_heads = int(config.num_attention_heads)
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "hidden_size % num_attention_heads != 0, "
            f"hidden_size={hidden_size}, num_attention_heads={num_attention_heads}"
        )

    head_dim = hidden_size // num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    logger.debug("compute_default_dream_rope_parameters, head_dim=%d, dim=%d", head_dim, dim)

    # Match the legacy Dream/Transformers v4 default RoPE calculation.
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float32) / dim)
    )
    return inv_freq, 1.0


def ensure_default_rope_init_function() -> None:
    """Register Dream's legacy default RoPE initializer when v5 omits it."""
    if "default" in ROPE_INIT_FUNCTIONS:
        return

    def _compute_default_rope_parameters(
        config: Any,
        device: torch.device | None = None,
        seq_len: int | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, float]:
        del seq_len
        return _compute_default_dream_rope_parameters(config, device)

    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters
    logger.warning(
        "Registered the default RoPE initializer required by Omni-Diffusion "
        "for compatibility with this transformers version. "
        "huggingface model transformers version is 4.51.3 and now transformers version is %s",
        transformers.__version__,
    )


def repair_default_dream_rope_buffers(model: Any) -> None:
    """Replace Transformers v5 Dream RoPE buffers with v4-equivalent values."""
    model_type = getattr(model.config, "model_type", None)
    if model_type != "Dream":
        raise ValueError(f"Omni-Diffusion RoPE migration expected Dream model config, got model_type={model_type!r}.")
    rope_type = _get_dream_rope_type(model.config)
    if rope_type != "default":
        logger.info("Omni-Diffusion RoPE compatibility repair skipped for rope_type=%r.", rope_type)
        return

    repaired = 0
    rope_parameters = getattr(model.config, "rope_parameters", {})
    for module in model.modules():
        current_inv_freq = getattr(module, "inv_freq", None)
        if not isinstance(current_inv_freq, torch.Tensor):
            continue
        inv_freq, attention_scaling = _compute_default_dream_rope_parameters(
            model.config,
            current_inv_freq.device,
        )
        module.register_buffer("inv_freq", inv_freq, persistent=False)
        module.original_inv_freq = module.inv_freq
        if hasattr(module, "attention_scaling"):
            module.attention_scaling = attention_scaling
        repaired += 1
    logger.debug(
        "Omni-Diffusion RoPE repair summary: rope_type=%r rope_theta=%r partial_rotary_factor=%r repaired=%d",
        rope_type,
        rope_parameters.get("rope_theta", getattr(model.config, "rope_theta", None)),
        rope_parameters.get("partial_rotary_factor", getattr(model.config, "partial_rotary_factor", None)),
        repaired,
    )
    logger.info("Repaired %d Omni-Diffusion default RoPE buffer(s) for transformers v5 compatibility.", repaired)


def _patch_legacy_dream_generation_config_validate_class(cls: type[Any]) -> None:
    if cls.__name__ != "DreamGenerationConfig":
        raise ValueError(f"Expected DreamGenerationConfig, got cls.__name__={cls.__name__!r}.")
    if getattr(cls, "_vllm_omni_validate_patched", False):
        return

    validate = getattr(cls, "validate", None)
    if validate is None:
        raise TypeError("DreamGenerationConfig has no function validate.")
    signature = inspect.signature(validate)
    if "user_set_attributes" in signature.parameters and "strict" in signature.parameters:
        cls._vllm_omni_validate_patched = True
        return

    logger.info("Patching Omni-Diffusion DreamGenerationConfig.validate for transformers v5 compatibility.")

    def _validate_v5_compatible(self: Any, *args: Any, **kwargs: Any) -> Any:
        # The remote config uses the v4 validate(is_init=False) signature.
        kwargs.pop("strict", None)
        kwargs.pop("user_set_attributes", None)
        return validate(self, *args, **kwargs)

    cls.validate = _validate_v5_compatible
    cls._vllm_omni_validate_patched = True
    logger.info(
        "Patched Omni-Diffusion generation config validate for transformers v5: %s.%s",
        cls.__module__,
        cls.__name__,
    )


def patch_remote_dream_generation_config_validate(
    model_path: str,
    trust_remote_code: bool | None,
) -> None:
    """Patch Dream's dynamically loaded generation config class for v5."""
    if not trust_remote_code:
        logger.warning(
            "Skipping Omni-Diffusion DreamGenerationConfig.validate patch because "
            "trust_remote_code is not enabled. Omni-Diffusion normally requires "
            "trust_remote_code=True to load modeling_dream.py."
        )
        return
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    model_cls = get_class_from_dynamic_module(
        "modeling_dream.DreamModel",
        model_path,
        trust_remote_code=trust_remote_code,
    )
    modeling_module = sys.modules.get(model_cls.__module__)
    cls = getattr(modeling_module, "DreamGenerationConfig", None)
    if cls is None or not isinstance(cls, type):
        raise TypeError(f"Could not load DreamGenerationConfig from modeling_dream module, got {cls!r}.")
    _patch_legacy_dream_generation_config_validate_class(cls)


def patch_legacy_dream_generation_config_validate(generation_config: Any) -> None:
    """Patch an already loaded Dream generation config instance for v5."""
    if generation_config is not None:
        _patch_legacy_dream_generation_config_validate_class(type(generation_config))


def ensure_dream_generation_config_fields(
    generation_config: Any,
    model_config: Any | None,
    tokenizer: Any | None,
) -> None:
    """Restore Dream-specific generation fields omitted by generic configs."""
    if generation_config is None:
        return

    for name, default_value in _DREAM_GENERATION_CONFIG_DEFAULTS.items():
        if hasattr(generation_config, name):
            value = getattr(generation_config, name)
            if value is not None or name in _DREAM_GENERATION_CONFIG_NONE_DEFAULT_FIELDS:
                continue
        value = getattr(model_config, name, default_value) if model_config is not None else default_value
        setattr(generation_config, name, value)

    for token_name in _DREAM_GENERATION_CONFIG_TOKEN_FIELDS:
        if getattr(generation_config, token_name, None) is not None:
            continue
        token_id = getattr(model_config, token_name, None) if model_config is not None else None
        if token_id is None and tokenizer is not None:
            token_id = getattr(tokenizer, token_name, None)
        if token_id is not None:
            setattr(generation_config, token_name, token_id)


def initialize_dream_generation_config(
    model: Any,
    tokenizer: Any,
    model_path: str,
    trust_remote_code: bool | None,
    top_k: int | None,
) -> Any:
    """Load and fill the Dream generation config used by Omni-Diffusion."""
    generation_config = getattr(model, "generation_config", None)
    patch_legacy_dream_generation_config_validate(generation_config)
    if generation_config is None:
        patch_remote_dream_generation_config_validate(
            model_path,
            trust_remote_code,
        )
        generation_config = GenerationConfig.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
        patch_legacy_dream_generation_config_validate(generation_config)
        model.generation_config = generation_config

    ensure_dream_generation_config_fields(generation_config, model.config, tokenizer)
    generation_config.max_new_tokens = 8192
    generation_config.chat_format = "chatml"
    generation_config.max_window_size = 8192
    generation_config.use_cache = True
    generation_config.do_sample = False
    generation_config.temperature = 1.0
    generation_config.top_k = top_k
    generation_config.top_p = 1.0
    generation_config.num_beams = 1
    generation_config.pad_token_id = tokenizer.pad_token_id
    logger.info("Omni-Diffusion generation config: data:%s", generation_config.__dict__)
    return generation_config
