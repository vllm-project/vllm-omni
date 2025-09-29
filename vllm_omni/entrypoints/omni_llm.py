"""
Core OmniLLM and AsyncOmniLLM classes for multi-stage processing.
"""

from typing import List, Dict, Any, Optional, Union, Callable
from vllm.entrypoints.llm import LLM
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.outputs import RequestOutput, LoRARequest

from ..config import OmniStageConfig
from .stage_manager import StageManager
from ..engine.output_processor import MultimodalOutputProcessor
from ..engine.diffusion_engine import DiffusersPipelineEngine



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

        # Track whether we should launch engines in multiprocess mode.
        self.multiprocess_mode = kwargs.pop("multiprocess_mode", False)

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
        stage_args_list: Optional[List[Dict[str, Any]]] = None,
        use_tqdm: Union[bool, Callable[..., Any]] = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        priority: Optional[List[int]] = None,
        *,
        prompt: Optional[str] = None,
        stage_overrides: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> List[RequestOutput]:
        """Main generation interface - orchestrates multi-stage processing."""
        if stage_args_list is None:
            if prompt is None:
                raise ValueError(
                    "prompt must be provided when stage_args_list is not supplied"
                )
            stage_args_list = self._build_stage_args_from_config(
                prompt, stage_overrides or {}
            )

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
                stage_config, stage_args or {}, current_output
            )
            
            # Execute stage
            stage_output = self._execute_stage(
                stage_engine, processed_input, lora_request, priority, stage_config
            )
            
            # Update for next stage
            current_output = stage_output
            stage_config.stage_output = stage_output
        
        # Process final output
        final_output = self.output_processor.process_output(current_output)
        return final_output
    
    def _build_stage_args_from_config(
        self,
        prompt: str,
        stage_overrides: Dict[int, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Derive per-stage argument dictionaries from configuration defaults."""
        stage_args: List[Dict[str, Any]] = []
        for stage_config in self.stage_configs:
            combined: Dict[str, Any] = dict(stage_config.default_stage_args or {})
            override = stage_overrides.get(stage_config.stage_id)
            if override:
                combined.update(override)
            if stage_config.engine_type == "AR":
                combined["prompt"] = prompt
            stage_args.append(combined)
        return stage_args

    def _process_stage_inputs(
        self,
        stage_config: OmniStageConfig,
        stage_args: Dict[str, Any],
        previous_output: Optional[Any]
    ) -> Dict[str, Any]:
        """Prepare input for specific stage."""
        if stage_config.engine_type == "AR":
            return self._process_ar_inputs(stage_config, stage_args, previous_output)
        elif stage_config.engine_type == "DiT":
            return self._process_dit_inputs(stage_config, stage_args, previous_output)
        else:
            raise NotImplementedError(f"Unknown engine type: {stage_config.engine_type}")

    def _process_ar_inputs(
        self,
        stage_config: OmniStageConfig,
        stage_args: Dict[str, Any],
        previous_output: Optional[Any]
    ) -> Dict[str, Any]:
        """Process inputs for AR stage."""
        combined = dict(stage_config.default_stage_args or {})
        combined.update(stage_args)
        combined.setdefault("prompt", "")
        combined.setdefault("max_tokens", 100)
        combined.setdefault("temperature", 0.7)
        
        # If we have previous output (e.g., from a previous AR stage), 
        # we might want to use it as context
        if previous_output is not None:
            # Extract text from previous output if available
            if hasattr(previous_output, 'outputs') and previous_output.outputs:
                last_output = previous_output.outputs[-1]
                if hasattr(last_output, 'text'):
                    combined["prompt"] = last_output.text + " " + combined["prompt"]

        return combined

    def _process_dit_inputs(
        self,
        stage_config: OmniStageConfig,
        stage_args: Dict[str, Any],
        previous_output: Optional[Any]
    ) -> Dict[str, Any]:
        """Process inputs for DiT stage."""
        combined = dict(stage_config.default_stage_args or {})
        combined.update(stage_args)

        dit = stage_config.dit_config
        if dit is not None:
            combined.setdefault("height", getattr(dit, "height", 512))
            combined.setdefault("width", getattr(dit, "width", 512))
            combined.setdefault(
                "num_inference_steps", getattr(dit, "num_inference_steps", 50)
            )
            combined.setdefault(
                "guidance_scale", getattr(dit, "guidance_scale", 7.5)
            )
        else:
            combined.setdefault("height", 512)
            combined.setdefault("width", 512)
            combined.setdefault("num_inference_steps", 50)
            combined.setdefault("guidance_scale", 7.5)

        # Handle image inputs if present
        if "image" in stage_args:
            # For now, we'll pass the image path directly
            # In a full implementation, this would involve VAE encoding
            combined["image"] = stage_args["image"]

        # If we have previous output from an AR stage, we might want to use it
        if previous_output is not None:
            # Extract text from previous AR output
            if hasattr(previous_output, 'outputs') and previous_output.outputs:
                last_output = previous_output.outputs[-1]
                if hasattr(last_output, 'text'):
                    combined["prompt"] = last_output.text

        combined.setdefault("prompt", stage_args.get("prompt", ""))

        return combined
    
    def _execute_stage(
        self,
        stage_engine: Optional[Union[LLMEngine, DiffusersPipelineEngine]],
        processed_input: Dict[str, Any],
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        priority: Optional[List[int]] = None,
        stage_config: Optional[OmniStageConfig] = None,
    ) -> Any:
        """Execute a single stage."""
        # DiT via diffusers backend
        if stage_config and stage_config.engine_type == "DiT":
            dit = stage_config.dit_config
            if dit and getattr(dit, "use_diffusers", False):
                # Lazy-init executor per stage
                if not hasattr(self, "_dit_engines"):
                    self._dit_engines = {}
                exec_inst = self._dit_engines.get(stage_config.stage_id)
                if exec_inst is None:
                    exec_inst = DiffusersPipelineEngine(
                        dit_config=dit,
                        model_path=stage_config.model_path,
                        log_stats=self.log_stats,
                        multiprocess_mode=self.multiprocess_mode,
                    )
                    
                    self._dit_engines[stage_config.stage_id] = exec_inst

                return exec_inst.generate(
                    prompt=processed_input.get("prompt", ""),
                    height=processed_input.get("height", getattr(dit, "height", 512)),
                    width=processed_input.get("width", getattr(dit, "width", 512)),
                    num_inference_steps=processed_input.get(
                        "num_inference_steps", getattr(dit, "num_inference_steps", 30)
                    ),
                    guidance_scale=processed_input.get(
                        "guidance_scale", getattr(dit, "guidance_scale", 5.0)
                    ),
                    negative_prompt=processed_input.get("negative_prompt"),
                    seed=processed_input.get("seed"),
                    image=processed_input.get("image"),
                )

        # Use the parent LLM's generate method for AR text generation
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
        stage_args_list: Optional[List[Dict[str, Any]]] = None,
        use_tqdm: Union[bool, Callable[..., Any]] = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        priority: Optional[List[int]] = None,
        *,
        prompt: Optional[str] = None,
        stage_overrides: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> List[RequestOutput]:
        """Async generation interface - orchestrates multi-stage processing."""

        if stage_args_list is None:
            if prompt is None:
                raise ValueError(
                    "prompt must be provided when stage_args_list is not supplied"
                )
            stage_args_list = self._build_stage_args_from_config(
                prompt, stage_overrides or {}
            )

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
                stage_config, stage_args or {}, current_output
            )
            
            # Execute stage asynchronously
            stage_output = await self._execute_stage_async(
                stage_engine, processed_input, lora_request, priority, stage_config
            )
            
            # Update for next stage
            current_output = stage_output
            stage_config.stage_output = stage_output
        
        # Process final output
        final_output = self.output_processor.process_output(current_output)
        return final_output
    
    def _build_stage_args_from_config(
        self,
        prompt: str,
        stage_overrides: Dict[int, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        stage_args: List[Dict[str, Any]] = []
        for stage_config in self.stage_configs:
            combined: Dict[str, Any] = dict(stage_config.default_stage_args or {})
            override = stage_overrides.get(stage_config.stage_id)
            if override:
                combined.update(override)
            if stage_config.engine_type == "AR":
                combined["prompt"] = prompt
            stage_args.append(combined)
        return stage_args

    def _process_stage_inputs(
        self,
        stage_config: OmniStageConfig,
        stage_args: Dict[str, Any],
        previous_output: Optional[Any]
    ) -> Dict[str, Any]:
        """Prepare input for specific stage (same as OmniLLM)."""
        if stage_config.engine_type == "AR":
            return self._process_ar_inputs(stage_config, stage_args, previous_output)
        elif stage_config.engine_type == "DiT":
            return self._process_dit_inputs(stage_config, stage_args, previous_output)
        else:
            raise NotImplementedError(f"Unknown engine type: {stage_config.engine_type}")

    def _process_ar_inputs(
        self,
        stage_config: OmniStageConfig,
        stage_args: Dict[str, Any],
        previous_output: Optional[Any]
    ) -> Dict[str, Any]:
        """Process inputs for AR stage (same as OmniLLM)."""
        combined = dict(stage_config.default_stage_args or {})
        combined.update(stage_args)
        combined.setdefault("prompt", "")
        combined.setdefault("max_tokens", 100)
        combined.setdefault("temperature", 0.7)

        if previous_output is not None:
            if hasattr(previous_output, 'outputs') and previous_output.outputs:
                last_output = previous_output.outputs[-1]
                if hasattr(last_output, 'text'):
                    combined["prompt"] = last_output.text + " " + combined["prompt"]

        return combined

    def _process_dit_inputs(
        self,
        stage_config: OmniStageConfig,
        stage_args: Dict[str, Any],
        previous_output: Optional[Any]
    ) -> Dict[str, Any]:
        """Process inputs for DiT stage (same as OmniLLM)."""
        combined = dict(stage_config.default_stage_args or {})
        combined.update(stage_args)

        dit = stage_config.dit_config
        if dit is not None:
            combined.setdefault("height", getattr(dit, "height", 512))
            combined.setdefault("width", getattr(dit, "width", 512))
            combined.setdefault(
                "num_inference_steps", getattr(dit, "num_inference_steps", 50)
            )
            combined.setdefault(
                "guidance_scale", getattr(dit, "guidance_scale", 7.5)
            )
        else:
            combined.setdefault("height", 512)
            combined.setdefault("width", 512)
            combined.setdefault("num_inference_steps", 50)
            combined.setdefault("guidance_scale", 7.5)

        if "image" in stage_args:
            combined["image"] = stage_args["image"]

        if previous_output is not None:
            if hasattr(previous_output, 'outputs') and previous_output.outputs:
                last_output = previous_output.outputs[-1]
                if hasattr(last_output, 'text'):
                    combined["prompt"] = last_output.text

        combined.setdefault("prompt", stage_args.get("prompt", ""))

        return combined
    
    async def _execute_stage_async(
        self,
        stage_engine: AsyncLLM,
        processed_input: Dict[str, Any],
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        priority: Optional[List[int]] = None,
        stage_config: Optional[OmniStageConfig] = None,
    ) -> Any:
        """Execute a single stage asynchronously."""
        # DiT via diffusers backend (sync call inside async for MVP)
        if stage_config and stage_config.engine_type == "DiT":
            dit = stage_config.dit_config
            if dit and getattr(dit, "use_diffusers", False):
                if not hasattr(self, "_dit_engines"):
                    self._dit_engines = {}
                exec_inst = self._dit_engines.get(stage_config.stage_id)
                if exec_inst is None:
                    from vllm_omni.engine.diffusion_engine import (
                        DiffusersPipelineEngine,
                    )

                    pipeline_name = getattr(dit, "diffusers_pipeline", None)
                    device_cfg = getattr(dit, "device_config", None)
                    model_cfg = getattr(dit, "model_config", None)
                    if isinstance(device_cfg, dict):
                        device = device_cfg.get("device")
                        dtype = device_cfg.get("dtype")
                    else:
                        device = getattr(device_cfg, "device", None)
                        dtype = getattr(device_cfg, "dtype", None)

                    if dtype is None:
                        if isinstance(model_cfg, dict):
                            dtype = model_cfg.get("dtype")
                        else:
                            dtype = getattr(model_cfg, "dtype", None)

                    exec_inst = DiffusersPipelineEngine(
                        model_path=stage_config.model_path,
                        pipeline_name=pipeline_name,
                        device=device,
                        dtype=dtype,
                    )
                    self._dit_engines[stage_config.stage_id] = exec_inst

                return exec_inst.generate(
                    prompt=processed_input.get("prompt", ""),
                    height=processed_input.get("height", getattr(dit, "height", 512)),
                    width=processed_input.get("width", getattr(dit, "width", 512)),
                    num_inference_steps=processed_input.get(
                        "num_inference_steps", getattr(dit, "num_inference_steps", 30)
                    ),
                    guidance_scale=processed_input.get(
                        "guidance_scale", getattr(dit, "guidance_scale", 5.0)
                    ),
                    negative_prompt=processed_input.get("negative_prompt"),
                    seed=processed_input.get("seed"),
                    image=processed_input.get("image"),
                )

        # Use the parent LLM's generate method for AR text generation
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
