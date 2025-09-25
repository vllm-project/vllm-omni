"""
Output processing for multimodal outputs in vLLM-omni.
"""

from typing import List, Dict, Any, Optional, Callable, Union
from vllm.outputs import RequestOutput, CompletionOutput
from vllm.v1.outputs import ModelRunnerOutput as EngineCoreOutput


class MultimodalOutputProcessor:
    """Handles multimodal output processing for vLLM-omni."""
    
    def __init__(self):
        self.output_handlers: Dict[str, Callable] = {
            "image": self._process_image_output,
            "text+image": self._process_text_image_output,
            "latents": self._process_latents_output,
            "text": self._process_text_output,
            "pooling": self._process_pooling_output,
        }
    
    def process_output(self, engine_core_output: Any) -> List[RequestOutput]:
        """Process engine core output and return formatted RequestOutput."""
        if engine_core_output is None:
            return []
        
        # If it's already a RequestOutput, return as is
        if isinstance(engine_core_output, RequestOutput):
            return [engine_core_output]
        
        # If it's a list of RequestOutputs, return as is
        if isinstance(engine_core_output, list):
            return engine_core_output
        
        # Otherwise, process based on output type
        output_type = self._detect_output_type(engine_core_output)
        handler = self.output_handlers.get(output_type, self._process_pooling_output)
        
        return handler(engine_core_output)
    
    def _detect_output_type(self, output: Any) -> str:
        """Detect the type of output based on its content."""
        if hasattr(output, 'output_type'):
            return output.output_type
        
        # Check for image-related attributes
        if hasattr(output, 'image') or hasattr(output, 'images'):
            if hasattr(output, 'text') or hasattr(output, 'texts'):
                return "text+image"
            else:
                return "image"
        
        # Check for latent-related attributes
        if hasattr(output, 'latents') or hasattr(output, 'latent_representation'):
            return "latents"
        
        # Check for pooling output
        if hasattr(output, 'pooler_output') and output.pooler_output is not None:
            return "pooling"
        
        # Default to text
        return "text"
    
    def _process_text_output(self, output: Any) -> List[RequestOutput]:
        """Process text output."""
        if isinstance(output, RequestOutput):
            return [output]
        
        # Create a mock RequestOutput for text
        completion_output = CompletionOutput(
            index=0,
            text=getattr(output, 'text', ''),
            token_ids=getattr(output, 'token_ids', []),
            cumulative_logprob=getattr(output, 'cumulative_logprob', 0.0),
            logprobs=getattr(output, 'logprobs', None),
            finish_reason=getattr(output, 'finish_reason', 'length')
        )
        
        request_output = RequestOutput(
            request_id=getattr(output, 'request_id', 'unknown'),
            prompt=getattr(output, 'prompt', ''),
            prompt_token_ids=getattr(output, 'prompt_token_ids', []),
            outputs=[completion_output],
            finished=getattr(output, 'finished', True)
        )
        
        return [request_output]
    
    def _process_image_output(self, output: Any) -> List[RequestOutput]:
        """Process image output."""
        # For image outputs, we need to create a special RequestOutput
        # that can handle image data
        
        # Extract image data
        image_data = getattr(output, 'image', None)
        if image_data is None:
            image_data = getattr(output, 'images', [None])[0]
        
        # Create a completion output with image data
        completion_output = CompletionOutput(
            index=0,
            text="",  # No text for pure image output
            token_ids=[],
            cumulative_logprob=0.0,
            logprobs=None,
            finish_reason="stop"
        )
        
        # Add image data to the completion output
        completion_output.image = image_data
        
        request_output = RequestOutput(
            request_id=getattr(output, 'request_id', 'unknown'),
            prompt=getattr(output, 'prompt', ''),
            prompt_token_ids=getattr(output, 'prompt_token_ids', []),
            outputs=[completion_output],
            finished=getattr(output, 'finished', True)
        )
        
        return [request_output]
    
    def _process_text_image_output(self, output: Any) -> List[RequestOutput]:
        """Process combined text and image output."""
        # Extract text and image data
        text_data = getattr(output, 'text', '')
        image_data = getattr(output, 'image', None)
        
        if image_data is None:
            image_data = getattr(output, 'images', [None])[0]
        
        # Create a completion output with both text and image
        completion_output = CompletionOutput(
            index=0,
            text=text_data,
            token_ids=getattr(output, 'token_ids', []),
            cumulative_logprob=getattr(output, 'cumulative_logprob', 0.0),
            logprobs=getattr(output, 'logprobs', None),
            finish_reason="stop"
        )
        
        # Add image data to the completion output
        completion_output.image = image_data
        
        request_output = RequestOutput(
            request_id=getattr(output, 'request_id', 'unknown'),
            prompt=getattr(output, 'prompt', ''),
            prompt_token_ids=getattr(output, 'prompt_token_ids', []),
            outputs=[completion_output],
            finished=getattr(output, 'finished', True)
        )
        
        return [request_output]
    
    def _process_latents_output(self, output: Any) -> List[RequestOutput]:
        """Process latent representation output."""
        # Extract latent data
        latent_data = getattr(output, 'latents', None)
        if latent_data is None:
            latent_data = getattr(output, 'latent_representation', None)
        
        # Create a completion output with latent data
        completion_output = CompletionOutput(
            index=0,
            text="",  # No text for latent output
            token_ids=[],
            cumulative_logprob=0.0,
            logprobs=None,
            finish_reason="stop"
        )
        
        # Add latent data to the completion output
        completion_output.latents = latent_data
        
        request_output = RequestOutput(
            request_id=getattr(output, 'request_id', 'unknown'),
            prompt=getattr(output, 'prompt', ''),
            prompt_token_ids=getattr(output, 'prompt_token_ids', []),
            outputs=[completion_output],
            finished=getattr(output, 'finished', True)
        )
        
        return [request_output]
    
    def _process_pooling_output(self, output: Any) -> List[RequestOutput]:
        """Process pooling output (hidden states, embeddings, etc.)."""
        # Extract pooling data
        pooling_data = getattr(output, 'pooler_output', None)
        if pooling_data is None:
            pooling_data = getattr(output, 'hidden_states', None)
        
        # Create a completion output with pooling data
        completion_output = CompletionOutput(
            index=0,
            text="",  # No text for pooling output
            token_ids=[],
            cumulative_logprob=0.0,
            logprobs=None,
            finish_reason="stop"
        )
        
        # Add pooling data to the completion output
        completion_output.pooler_output = pooling_data
        
        request_output = RequestOutput(
            request_id=getattr(output, 'request_id', 'unknown'),
            prompt=getattr(output, 'prompt', ''),
            prompt_token_ids=getattr(output, 'prompt_token_ids', []),
            outputs=[completion_output],
            finished=getattr(output, 'finished', True)
        )
        
        return [request_output]
    
    def process_outputs(self, engine_core_outputs: List[EngineCoreOutput], **kwargs) -> List[RequestOutput]:
        """Process multiple engine core outputs."""
        all_outputs = []
        
        for engine_core_output in engine_core_outputs:
            outputs = self.process_output(engine_core_output)
            all_outputs.extend(outputs)
        
        return all_outputs
    
    def add_output_handler(self, output_type: str, handler: Callable) -> None:
        """Add a custom output handler for a specific output type."""
        self.output_handlers[output_type] = handler
    
    def remove_output_handler(self, output_type: str) -> None:
        """Remove an output handler for a specific output type."""
        if output_type in self.output_handlers:
            del self.output_handlers[output_type]
