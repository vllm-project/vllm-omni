from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from functools import lru_cache


def load_diffusers_config(model_name) -> dict:
    try:
        config = DiffusionPipeline.load_config(model_name)
        return config
    except Exception as e:
        print(f"Error loading config for model {model_name}: {e}")
        return {}


@lru_cache
def is_diffusion_model(model_name: str) -> bool:
    try:
        config = load_diffusers_config(model_name)
        return True
    except Exception:
        return False
