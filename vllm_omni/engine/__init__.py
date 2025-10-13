"""
Engine components for vLLM-omni.
"""
from vllm.v1.engine import EngineCoreOutput
from typing import Optional
import torch
from .output_processor import MultimodalOutputProcessor


class OmniEngineCoreOutput(EngineCoreOutput):
    #multimodal outputs
    multimodal_outputs: Optional[dict[str, torch.Tensor]] = None

    # Output data type hint (e.g., "text", "image", "text+image", "latent").
    output_type: Optional[str] = None
    