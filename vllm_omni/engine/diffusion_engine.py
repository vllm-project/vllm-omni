
from typing import Any, Optional
from diffusers import DiffusionPipeline

from ..config import DiTConfig
from ..worker.gpu_diffusion_worker import (
    DiffusionGPUWorker,
    DiffusionRunnerOutput,
)


class DiffusionEngine:
    "Legacy placeholder for DiffusionEngine"

    def __init__(
        self,
        dit_config: DiTConfig,
        log_stats: bool = False,
        multiprocess_mode: bool = False
    ) -> None:
        self.dit_config = dit_config
        self.model_config = dit_config.model_config
        self.cache_config = dit_config.dit_cache_config

        self.height = dit_config.height
        self.width = dit_config.width
        self.num_inference_steps = dit_config.num_inference_steps
        self.guidance_scale = dit_config.guidance_scale
        
        self.log_stats = log_stats
        self.multiprocess_mode = multiprocess_mode

    def load_model(self) -> DiffusionPipeline:
        """Load the diffusion model pipeline."""
        # Placeholder implementation
        # In a full implementation, this would load the actual model
        return DiffusionPipeline.from_pretrained(self.model_config.model_name)
    
    def generate(self, *args, **kwargs):
        """Generate images based on the input prompt."""
        # Placeholder implementation
        # In a full implementation, this would use the loaded model to generate images
        raise NotImplementedError("This method should be implemented in subclasses.")


class DiffusersPipelineEngine(DiffusionEngine):
    """Thin executor that delegates to a diffusion worker."""

    # crucial: must match signature of parent class

    def __init__(
        self,
        dit_config: DiTConfig,
        log_stats: bool = False,
        multiprocess_mode: bool = False
    ) -> None:
        super().__init__(dit_config, log_stats, multiprocess_mode)

        self.worker = DiffusionGPUWorker(
            dit_config.model_config.model_path,
            pipeline_name=dit_config.diffusers_pipeline,
            device=dit_config.device_config.device if dit_config.device_config else None,
            dtype=dit_config.model_config.dtype if dit_config.model_config else None
        )

    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        image: Optional[Any] = None,
    ) -> DiffusionRunnerOutput:
        
        return self.worker.generate(
            prompt=prompt,
            height=self.height,
            width=self.width,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            negative_prompt=negative_prompt,
            seed=seed,
            image=image,
        )
