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
    except Exception:
        return False
    model_type = cfg.get("model_type")
    if model_type == "bagel":
        return True
    architectures = cfg.get("architectures") or []
    return "BagelForConditionalGeneration" in architectures


@lru_cache
def is_diffusion_model(model_name: str) -> bool:
    try:
        load_diffusers_config(model_name)
        return True
    except Exception:
        # Bagel is not a diffusers pipeline (no model_index.json), but is still a
        # diffusion-style model in vllm-omni. Detect it via config.json.
        return _looks_like_bagel(model_name)
