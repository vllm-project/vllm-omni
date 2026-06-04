import os
from collections.abc import Mapping
from functools import lru_cache

from vllm.logger import init_logger
from vllm.transformers_utils.config import get_hf_file_to_dict

logger = init_logger(__name__)


def load_diffusers_config(model_name) -> dict:
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline

    config = DiffusionPipeline.load_config(model_name)
    return config


def _looks_like_bagel(model_name: str) -> bool:
    """Best-effort detection for Bagel (non-diffusers) diffusion models."""
    try:
        cfg = get_hf_file_to_dict("config.json", model_name)
        model_type = cfg.get("model_type")
        if model_type == "bagel":
            return True
        architectures = cfg.get("architectures") or []
        return "BagelForConditionalGeneration" in architectures
    except Exception:
        return False


def _looks_like_dreamzero(model_name: str) -> bool:
    """Best-effort detection for DreamZero-style VLA diffusion checkpoints."""
    try:
        cfg = get_hf_file_to_dict("config.json", model_name)
        if cfg.get("model_type") != "vla":
            return False
        action_head_cfg = cfg.get("action_head_cfg") or {}
        if not isinstance(action_head_cfg, Mapping):
            return False
        action_head_config = action_head_cfg.get("config") or {}
        if not isinstance(action_head_config, Mapping):
            return False
        diffusion_model_cfg = action_head_config.get("diffusion_model_cfg") or {}
        if not isinstance(diffusion_model_cfg, Mapping):
            return False
        return (
            action_head_cfg.get("_target_")
            == "groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf.WANPolicyHead"
            and diffusion_model_cfg.get("_target_")
            == ("groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk.CausalWanModel")
        )
    except Exception:
        return False


@lru_cache
def is_diffusion_model(model_name: str) -> bool:
    """Check if a model is a diffusion model.

    Uses multiple fallback strategies to detect diffusion models:
    1. Check local file system for model_index.json (fastest, no imports)
    2. Check using vllm's get_hf_file_to_dict utility
    3. Try the standard diffusers approach (may fail due to import issues)
    """
    # Strategy 1: Check local file system first (fastest, avoids import issues)
    if os.path.isdir(model_name):
        model_index_path = os.path.join(model_name, "model_index.json")
        if os.path.exists(model_index_path):
            try:
                import json

                with open(model_index_path) as f:
                    config_dict = json.load(f)
                if config_dict.get("_class_name") and config_dict.get("_diffusers_version"):
                    logger.debug("Detected diffusion model via local model_index.json")
                    return True
            except Exception as e:
                logger.debug("Failed to read local model_index.json: %s", e)

    # Strategy 2: Check using vllm's utility (works for both local and remote models)
    try:
        config_dict = get_hf_file_to_dict("model_index.json", model_name)
        if config_dict is not None and config_dict.get("_class_name") and config_dict.get("_diffusers_version"):
            logger.debug("Detected diffusion model via model_index.json")
            return True
    except Exception as e:
        logger.debug("Failed to check model_index.json via get_hf_file_to_dict: %s", e)

    # Strategy 3: Try the standard diffusers approach (may fail due to import issues)
    # This is last because it requires importing diffusers/xformers/flash_attn
    # which may have compatibility issues
    try:
        load_diffusers_config(model_name)
        return True
    except (ImportError, ModuleNotFoundError) as e:
        logger.debug("Failed to import diffusers dependencies: %s", e)
        logger.debug("This may be due to flash_attn/PyTorch version mismatch")
    except Exception as e:
        logger.debug("Failed to load diffusers config via DiffusionPipeline: %s", e)

        # Bagel and DreamZero are not diffusers pipelines (no model_index.json),
        # but are still diffusion-style models in vllm-omni. Detect them via
        # config.json.
    return _looks_like_bagel(model_name) or _looks_like_dreamzero(model_name)
