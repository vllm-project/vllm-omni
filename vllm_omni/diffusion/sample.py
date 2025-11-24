from dataclasses import dataclass


@dataclass
class DiffusionSamplingParams:
    # TODO
    _ = 0  # Placeholder for base class


@dataclass
class QwenImageSamplingParams(DiffusionSamplingParams):
    # Video parameters
    # height: int = 1024
    # width: int = 1024
    negative_prompt: str = " "
    num_frames: int = 1
    # Denoising stage
    guidance_scale: float = 4.0
    num_inference_steps: int = 50
