"""
OmniRequest: Extended request class for vLLM-omni multimodal processing.

This class extends vLLM's Request to support multimodal and non-autoregressive
processing with additional fields and methods specific to vLLM-omni.
"""

import enum
import time
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from vllm.v1.request import Request as vLLMRequest


class RequestType(enum.Enum):
    """Types of requests supported by vLLM-omni."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"
    DIFFUSION = "diffusion"


class ProcessingStage(enum.Enum):
    """Processing stages for multi-stage models."""
    PREPROCESSING = "preprocessing"
    AR_GENERATION = "ar_generation"
    DIFFUSION_GENERATION = "diffusion_generation"
    POSTPROCESSING = "postprocessing"
    COMPLETED = "completed"


@dataclass
class MultimodalData:
    """Container for multimodal input data."""
    data_type: str  # "image", "audio", "video", etc.
    data: Any  # The actual data (numpy array, bytes, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)
    processed: bool = False


@dataclass
class DiffusionParams:
    """Parameters specific to diffusion model generation."""
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    scheduler: str = "ddpm"
    strength: float = 1.0
    eta: float = 0.0


class OmniRequest(vLLMRequest):
    """
    Extended request class for vLLM-omni multimodal and non-autoregressive processing.
    
    This class extends vLLM's Request with additional fields and methods to support:
    - Multimodal input processing
    - Non-autoregressive generation (diffusion models)
    - Multi-stage processing pipelines
    - Enhanced caching for different model types
    """
    
    def __init__(
        self,
        request_id: str,
        prompt: Optional[str] = None,
        prompt_token_ids: Optional[List[int]] = None,
        sampling_params: Optional[Any] = None,
        arrival_time: Optional[float] = None,
        lora_request: Optional[Any] = None,
        multi_modal_data: Optional[Dict[str, Any]] = None,
        multi_modal_placeholders: Optional[Dict[str, str]] = None,
        priority: int = 0,
        # vLLM-omni specific parameters
        request_type: RequestType = RequestType.TEXT,
        processing_stage: ProcessingStage = ProcessingStage.PREPROCESSING,
        multimodal_inputs: Optional[List[MultimodalData]] = None,
        diffusion_params: Optional[DiffusionParams] = None,
        output_format: str = "text",
        cache_key: Optional[str] = None,
        stage_configs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        # Initialize parent class
        super().__init__(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            arrival_time=arrival_time or time.time(),
            lora_request=lora_request,
            multi_modal_data=multi_modal_data,
            multi_modal_placeholders=multi_modal_placeholders,
            priority=priority,
            **kwargs
        )
        
        # vLLM-omni specific attributes
        self.request_type = request_type
        self.processing_stage = processing_stage
        self.multimodal_inputs = multimodal_inputs or []
        self.diffusion_params = diffusion_params or DiffusionParams()
        self.output_format = output_format
        self.cache_key = cache_key
        self.stage_configs = stage_configs or {}
        
        # Processing state
        self.current_stage = 0
        self.stage_results = {}
        self.hidden_states = None
        self.intermediate_outputs = []
        
        # Timing and metrics
        self.stage_timings = {}
        self.total_processing_time = 0.0
        
        # Error handling
        self.errors = []
        self.retry_count = 0
        self.max_retries = 3
    
    def add_multimodal_input(self, data: MultimodalData) -> None:
        """Add a multimodal input to the request."""
        self.multimodal_inputs.append(data)
    
    def get_multimodal_inputs_by_type(self, data_type: str) -> List[MultimodalData]:
        """Get all multimodal inputs of a specific type."""
        return [inp for inp in self.multimodal_inputs if inp.data_type == data_type]
    
    def update_processing_stage(self, stage: ProcessingStage) -> None:
        """Update the current processing stage."""
        self.processing_stage = stage
        self.stage_timings[stage.value] = time.time()
    
    def add_stage_result(self, stage: str, result: Any) -> None:
        """Add a result from a processing stage."""
        self.stage_results[stage] = result
        self.intermediate_outputs.append({
            'stage': stage,
            'result': result,
            'timestamp': time.time()
        })
    
    def get_stage_result(self, stage: str) -> Any:
        """Get a result from a specific processing stage."""
        return self.stage_results.get(stage)
    
    def set_hidden_states(self, hidden_states: Any) -> None:
        """Set hidden states for the request."""
        self.hidden_states = hidden_states
    
    def get_hidden_states(self) -> Any:
        """Get hidden states for the request."""
        return self.hidden_states
    
    def add_error(self, error: str) -> None:
        """Add an error to the request."""
        self.errors.append({
            'error': error,
            'timestamp': time.time(),
            'stage': self.processing_stage.value
        })
    
    def has_errors(self) -> bool:
        """Check if the request has any errors."""
        return len(self.errors) > 0
    
    def can_retry(self) -> bool:
        """Check if the request can be retried."""
        return self.retry_count < self.max_retries
    
    def increment_retry(self) -> None:
        """Increment the retry count."""
        self.retry_count += 1
    
    def generate_cache_key(self) -> str:
        """Generate a cache key for the request."""
        if self.cache_key:
            return self.cache_key
        
        # Generate cache key based on request content
        key_parts = [
            self.request_id,
            str(self.request_type.value),
            str(self.prompt_token_ids) if self.prompt_token_ids else str(self.prompt),
            str(self.diffusion_params.num_inference_steps) if self.diffusion_params else "0",
            str(len(self.multimodal_inputs))
        ]
        return "_".join(key_parts)
    
    def get_processing_time(self) -> float:
        """Get the total processing time for the request."""
        if self.stage_timings:
            return time.time() - min(self.stage_timings.values())
        return 0.0
    
    def is_completed(self) -> bool:
        """Check if the request is completed."""
        return self.processing_stage == ProcessingStage.COMPLETED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the request to a dictionary for serialization."""
        return {
            'request_id': self.request_id,
            'request_type': self.request_type.value,
            'processing_stage': self.processing_stage.value,
            'prompt': self.prompt,
            'prompt_token_ids': self.prompt_token_ids,
            'output_format': self.output_format,
            'multimodal_inputs_count': len(self.multimodal_inputs),
            'diffusion_params': self.diffusion_params.__dict__ if self.diffusion_params else None,
            'stage_results': list(self.stage_results.keys()),
            'has_errors': self.has_errors(),
            'retry_count': self.retry_count,
            'processing_time': self.get_processing_time()
        }
    
    def __repr__(self) -> str:
        """String representation of the request."""
        return (f"OmniRequest(id={self.request_id}, "
                f"type={self.request_type.value}, "
                f"stage={self.processing_stage.value}, "
                f"multimodal_inputs={len(self.multimodal_inputs)})")


# Factory functions for creating different types of requests
def create_text_request(
    request_id: str,
    prompt: str,
    sampling_params: Optional[Any] = None,
    **kwargs
) -> OmniRequest:
    """Create a text-only request."""
    return OmniRequest(
        request_id=request_id,
        prompt=prompt,
        request_type=RequestType.TEXT,
        sampling_params=sampling_params,
        **kwargs
    )


def create_image_request(
    request_id: str,
    prompt: str,
    image_data: Any,
    diffusion_params: Optional[DiffusionParams] = None,
    **kwargs
) -> OmniRequest:
    """Create an image generation request."""
    multimodal_input = MultimodalData(
        data_type="image",
        data=image_data,
        metadata={"is_input": True}
    )
    
    return OmniRequest(
        request_id=request_id,
        prompt=prompt,
        request_type=RequestType.IMAGE,
        multimodal_inputs=[multimodal_input],
        diffusion_params=diffusion_params,
        output_format="image",
        **kwargs
    )


def create_multimodal_request(
    request_id: str,
    prompt: str,
    multimodal_inputs: List[MultimodalData],
    **kwargs
) -> OmniRequest:
    """Create a multimodal request."""
    return OmniRequest(
        request_id=request_id,
        prompt=prompt,
        request_type=RequestType.MULTIMODAL,
        multimodal_inputs=multimodal_inputs,
        **kwargs
    )


def create_diffusion_request(
    request_id: str,
    prompt: str,
    diffusion_params: DiffusionParams,
    **kwargs
) -> OmniRequest:
    """Create a diffusion model request."""
    return OmniRequest(
        request_id=request_id,
        prompt=prompt,
        request_type=RequestType.DIFFUSION,
        diffusion_params=diffusion_params,
        **kwargs
    )