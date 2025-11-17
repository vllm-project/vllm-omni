from diffusers.pipelines.pipeline_utils import DiffusionPipeline

def load_diffusers_config(model_name) -> dict:
    try:
        config = DiffusionPipeline.load_config(model_name)
        return config
    except Exception as e:
        print(f"Error loading config for model {model_name}: {e}")
        return {}