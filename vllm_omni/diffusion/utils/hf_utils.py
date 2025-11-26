from functools import lru_cache

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from vllm.logger import init_logger

logger = init_logger(__name__)


def load_diffusers_config(model_name) -> dict:
    config = DiffusionPipeline.load_config(model_name)
    return config


@lru_cache
def is_diffusion_model(model_name: str) -> bool:
    try:
        load_diffusers_config(model_name)
        return True
    except Exception:
        return False
