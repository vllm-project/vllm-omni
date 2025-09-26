"""
Core OmniLLM and AsyncOmniLLM classes for multi-stage processing.
"""

from typing import List, Dict, Any, Optional, Union, Callable
from vllm.entrypoints.llm import LLM
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.outputs import RequestOutput, LoRARequest


from ..config import OmniStageConfig
from ..core.stage_manager import StageManager
from ..engine.output_processor import MultimodalOutputProcessor


class OmniLLM(LLM):
    """Extended LLM supporting multiple engines and stage-based processing."""
    
    def __init__(
        self,
        stage_configs: List[OmniStageConfig],
        log_stats: bool = False,
        **kwargs
    ):
        # Use the first stage's model as the default model for LLM
        default_model = stage_configs[0].model_path if stage_configs else "test-model"
        
        # Fix configuration validation issues
        # Ensure max_num_batched_tokens is at least as large as max_model_len
        if 'max_model_len' in kwargs and 'max_num_batched_tokens' in kwargs:
            if kwargs['max_num_batched_tokens'] < kwargs['max_model_len']:
                kwargs['max_num_batched_tokens'] = kwargs['max_model_len']
        elif 'max_model_len' in kwargs:
            # If max_model_len is set but max_num_batched_tokens is not, set it to max_model_len
            kwargs['max_num_batched_tokens'] = kwargs['max_model_len']
        else:
            # Set reasonable defaults to avoid validation errors
            kwargs['max_model_len'] = 2048
            kwargs['max_num_batched_tokens'] = 2048
        
        super().__init__(model=default_model, **kwargs)
        self.stage_configs = stage_configs
        self.log_stats = log_stats
        self.stage_manager = StageManager(stage_configs, log_stats)
        self.output_processor = MultimodalOutputProcessor()
        self._initialize_stage_engines()
    
    def _initialize_stage_engines(self) -> None:
        """Initialize LLMEngine instances for each stage."""
        self.stage_manager.initialize_engines()
    
    def generate(
        self,
        stage_args_list: List[Dict[str, Any]],
        use_tqdm: Union[bool, Callable[..., Any]] = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        priority: Optional[List[int]] = None,
        **kwargs
    ) -> List[RequestOutput]:
        """Main generation interface - orchestrates multi-stage processing."""
        
        if len(stage_args_list) != len(self.stage_configs):
            raise ValueError(
                f"Number of stage arguments ({len(stage_args_list)}) must match "
                f"number of stage configs ({len(self.stage_configs)})"
            )
        
        # Process through each stage sequentially
        current_output = None
        
        for i, (stage_config, stage_args) in enumerate(zip(self.stage_configs, stage_args_list)):
            stage_engine = self.stage_manager.get_engine(i)
            
            # Prepare input for this stage
            processed_input = self._process_stage_inputs(
                stage_config, stage_args, current_output
            )
            
            # Execute stage
            stage_output = self._execute_stage(
                stage_engine, processed_input, lora_request, priority
            )
            
            # Update for next stage
            current_output = stage_output
            stage_config.stage_output = stage_output
        
        # Process final output
        final_output = self.output_processor.process_output(current_output)
        return final_output
    
    def _process_stage_inputs(
        self,
        stage_config: OmniStageConfig,
        stage_args: Dict[str, Any],
        previous_output: Optional[Any]
    ) -> Dict[str, Any]:
        """Prepare input for specific stage."""
        if stage_config.engine_type == "AR":
            return self._process_ar_inputs(stage_args, previous_output)
        elif stage_config.engine_type == "DiT":
            return self._process_dit_inputs(stage_args, previous_output)
        else:
            raise NotImplementedError(f"Unknown engine type: {stage_config.engine_type}")
    
    def _process_ar_inputs(
        self,
        stage_args: Dict[str, Any],
        previous_output: Optional[Any]
    ) -> Dict[str, Any]:
        """Process inputs for AR stage."""
        # For AR stages, we typically use text prompts
        processed_input = {
            "prompt": stage_args.get("prompt", ""),
            "max_tokens": stage_args.get("max_tokens", 100),
            "temperature": stage_args.get("temperature", 0.7),
        }
        
        # If we have previous output (e.g., from a previous AR stage), 
        # we might want to use it as context
        if previous_output is not None:
            # Extract text from previous output if available
            if hasattr(previous_output, 'outputs') and previous_output.outputs:
                last_output = previous_output.outputs[-1]
                if hasattr(last_output, 'text'):
                    processed_input["prompt"] = last_output.text + " " + processed_input["prompt"]
        
        return processed_input
    
    def _process_dit_inputs(
        self,
        stage_args: Dict[str, Any],
        previous_output: Optional[Any]
    ) -> Dict[str, Any]:
        """Process inputs for DiT stage."""
        processed_input = {
            "prompt": stage_args.get("prompt", ""),
            "height": stage_args.get("height", 512),
            "width": stage_args.get("width", 512),
            "num_inference_steps": stage_args.get("num_inference_steps", 50),
            "guidance_scale": stage_args.get("guidance_scale", 7.5),
        }
        
        # Handle image inputs if present
        if "image" in stage_args:
            # For now, we'll pass the image path directly
            # In a full implementation, this would involve VAE encoding
            processed_input["image"] = stage_args["image"]
        
        # If we have previous output from an AR stage, we might want to use it
        if previous_output is not None:
            # Extract text from previous AR output
            if hasattr(previous_output, 'outputs') and previous_output.outputs:
                last_output = previous_output.outputs[-1]
                if hasattr(last_output, 'text'):
                    processed_input["prompt"] = last_output.text
        
        return processed_input
    
    def _execute_stage(
        self,
        stage_engine: LLMEngine,
        processed_input: Dict[str, Any],
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        priority: Optional[List[int]] = None
    ) -> Any:
        """Execute a single stage."""
        # Use the parent LLM's generate method for actual text generation
        prompt = processed_input.get("prompt", "")
        max_tokens = processed_input.get("max_tokens", 100)
        temperature = processed_input.get("temperature", 0.7)
        
        # Generate using the base LLM class
        from vllm.sampling_params import SamplingParams
        
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=processed_input.get("top_p", 1.0),
            frequency_penalty=processed_input.get("frequency_penalty", 0.0),
            presence_penalty=processed_input.get("presence_penalty", 0.0),
            stop=processed_input.get("stop", None)
        )
        
        # Use the parent class's generate method
        outputs = super().generate([prompt], sampling_params)
        
        # Return the first output (we're processing one prompt at a time)
        if outputs:
            return outputs[0]
        else:
            # Fallback to mock output if generation fails
            from vllm.outputs import RequestOutput, CompletionOutput
            
            mock_output = CompletionOutput(
                index=0,
                text="Generation failed",
                token_ids=[],
                cumulative_logprob=0.0,
                logprobs=None,
                finish_reason="error"
            )
            
            return RequestOutput(
                request_id="fallback_request",
                prompt=prompt,
                prompt_token_ids=[],
                prompt_logprobs=None,
                outputs=[mock_output],
                finished=True
            )


class AsyncOmniLLM(LLM):
    """Extended LLM class supporting multiple engines and stage-based processing."""
    
    def __init__(
        self,
        stage_configs: List[OmniStageConfig],
        log_stats: bool = False,
        **kwargs
    ):
        # Use the first stage's model for the base LLM
        if stage_configs and stage_configs[0].model_path:
            model = stage_configs[0].model_path
        else:
            model = "Qwen/Qwen3-0.6B"
        
        # Fix configuration validation issues
        # Ensure max_num_batched_tokens is at least as large as max_model_len
        if 'max_model_len' in kwargs and 'max_num_batched_tokens' in kwargs:
            if kwargs['max_num_batched_tokens'] < kwargs['max_model_len']:
                kwargs['max_num_batched_tokens'] = kwargs['max_model_len']
        elif 'max_model_len' in kwargs:
            # If max_model_len is set but max_num_batched_tokens is not, set it to max_model_len
            kwargs['max_num_batched_tokens'] = kwargs['max_model_len']
        else:
            # Set reasonable defaults to avoid validation errors
            kwargs['max_model_len'] = 2048
            kwargs['max_num_batched_tokens'] = 2048
            
        super().__init__(model=model, **kwargs)
        self.stage_configs = stage_configs
        self.log_stats = log_stats
        self.stage_manager = StageManager(stage_configs, log_stats)
        self.output_processor = MultimodalOutputProcessor()
    
    def _initialize_async_stage_engines(self) -> None:
        """Initialize AsyncLLM instances for each stage."""
        self.stage_manager.initialize_async_engines()
    
    async def generate_async(
        self,
        stage_args_list: List[Dict[str, Any]],
        use_tqdm: Union[bool, Callable[..., Any]] = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        priority: Optional[List[int]] = None,
        **kwargs
    ) -> List[RequestOutput]:
        """Async generation interface - orchestrates multi-stage processing."""
        
        if len(stage_args_list) != len(self.stage_configs):
            raise ValueError(
                f"Number of stage arguments ({len(stage_args_list)}) must match "
                f"number of stage configs ({len(self.stage_configs)})"
            )
        
        # Process through each stage sequentially
        current_output = None
        
        for i, (stage_config, stage_args) in enumerate(zip(self.stage_configs, stage_args_list)):
            stage_engine = self.stage_manager.get_async_engine(i)
            
            # Prepare input for this stage
            processed_input = self._process_stage_inputs(
                stage_config, stage_args, current_output
            )
            
            # Execute stage asynchronously
            stage_output = await self._execute_stage_async(
                stage_engine, processed_input, lora_request, priority
            )
            
            # Update for next stage
            current_output = stage_output
            stage_config.stage_output = stage_output
        
        # Process final output
        final_output = self.output_processor.process_output(current_output)
        return final_output
    
    def _process_stage_inputs(
        self,
        stage_config: OmniStageConfig,
        stage_args: Dict[str, Any],
        previous_output: Optional[Any]
    ) -> Dict[str, Any]:
        """Prepare input for specific stage (same as OmniLLM)."""
        if stage_config.engine_type == "AR":
            return self._process_ar_inputs(stage_args, previous_output)
        elif stage_config.engine_type == "DiT":
            return self._process_dit_inputs(stage_args, previous_output)
        else:
            raise NotImplementedError(f"Unknown engine type: {stage_config.engine_type}")
    
    def _process_ar_inputs(
        self,
        stage_args: Dict[str, Any],
        previous_output: Optional[Any]
    ) -> Dict[str, Any]:
        """Process inputs for AR stage (same as OmniLLM)."""
        processed_input = {
            "prompt": stage_args.get("prompt", ""),
            "max_tokens": stage_args.get("max_tokens", 100),
            "temperature": stage_args.get("temperature", 0.7),
        }
        
        if previous_output is not None:
            if hasattr(previous_output, 'outputs') and previous_output.outputs:
                last_output = previous_output.outputs[-1]
                if hasattr(last_output, 'text'):
                    processed_input["prompt"] = last_output.text + " " + processed_input["prompt"]
        
        return processed_input
    
    def _process_dit_inputs(
        self,
        stage_args: Dict[str, Any],
        previous_output: Optional[Any]
    ) -> Dict[str, Any]:
        """Process inputs for DiT stage (same as OmniLLM)."""
        processed_input = {
            "prompt": stage_args.get("prompt", ""),
            "height": stage_args.get("height", 512),
            "width": stage_args.get("width", 512),
            "num_inference_steps": stage_args.get("num_inference_steps", 50),
            "guidance_scale": stage_args.get("guidance_scale", 7.5),
        }
        
        if "image" in stage_args:
            processed_input["image"] = stage_args["image"]
        
        if previous_output is not None:
            if hasattr(previous_output, 'outputs') and previous_output.outputs:
                last_output = previous_output.outputs[-1]
                if hasattr(last_output, 'text'):
                    processed_input["prompt"] = last_output.text
        
        return processed_input
    
    async def _execute_stage_async(
        self,
        stage_engine: AsyncLLM,
        processed_input: Dict[str, Any],
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        priority: Optional[List[int]] = None
    ) -> Any:
        """Execute a single stage asynchronously."""
        # Use the parent LLM's generate method for actual text generation
        prompt = processed_input.get("prompt", "")
        max_tokens = processed_input.get("max_tokens", 100)
        temperature = processed_input.get("temperature", 0.7)
        
        # Generate using the base LLM class
        from vllm.sampling_params import SamplingParams
        
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=processed_input.get("top_p", 1.0),
            frequency_penalty=processed_input.get("frequency_penalty", 0.0),
            presence_penalty=processed_input.get("presence_penalty", 0.0),
            stop=processed_input.get("stop", None)
        )
        
        # Use the parent class's generate method
        outputs = super().generate([prompt], sampling_params)
        
        # Return the first output (we're processing one prompt at a time)
        if outputs:
            return outputs[0]
        else:
            # Fallback to mock output if generation fails
            from vllm.outputs import RequestOutput, CompletionOutput
            
            mock_output = CompletionOutput(
                index=0,
                text="Generation failed",
                token_ids=[],
                cumulative_logprob=0.0,
                logprobs=None,
                finish_reason="error"
            )
            
            return RequestOutput(
                request_id="fallback_request",
                prompt=prompt,
                prompt_token_ids=[],
                prompt_logprobs=None,
                outputs=[mock_output],
                finished=True
            )
