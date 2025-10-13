"""
Core OmniLLM and AsyncOmniLLM classes for multi-stage processing.
"""

from typing import List, Dict, Any, Optional, Union, Callable
import cloudpickle
from pydantic import ValidationError

from vllm.entrypoints.llm import LLM
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.outputs import RequestOutput, LoRARequest
from vllm.usage.usage_lib import UsageContext
from vllm.config import CompilationConfig, is_init_field
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.utils import log_non_default_args
from vllm.plugins.io_processors import get_io_processor
from vllm.utils import Counter
from vllm.logger import init_logger
import vllm.envs as envs

from vllm_omni.engine.arg_utils import OmniEngineArgs
# from .stage_manager import StageManager
from ..engine.output_processor import MultimodalOutputProcessor
from ..engine.diffusion_engine import DiffusersPipelineEngine
from .utils import load_stage_configs

logger = init_logger(__name__)



class OmniLLM(LLM):
    """Extended LLM supporting multiple engines and stage-based processing."""
    
    def __init__(
        self,
        model: str,
        omni_args: List[OmniEngineArgs] = None,
        log_stats: bool = False,
        **kwargs
    ):
        # Replicate parent LLM.__init__ logic WITHOUT creating self.llm_engine
        # This is necessary because we manage multiple engines through engine_list
        
        # 1. Handle kwargs preprocessing (from parent LLM.__init__)
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True

        if "worker_cls" in kwargs:
            worker_cls = kwargs["worker_cls"]
            if isinstance(worker_cls, type):
                kwargs["worker_cls"] = cloudpickle.dumps(worker_cls)

        if "kv_transfer_config" in kwargs and isinstance(
                kwargs["kv_transfer_config"], dict):
            from vllm.config.kv_transfer import KVTransferConfig
            raw_config_dict = kwargs["kv_transfer_config"]
            try:
                kwargs["kv_transfer_config"] = KVTransferConfig(
                    **raw_config_dict)
            except ValidationError as e:
                logger.error(
                    "Failed to convert 'kv_transfer_config' dict to "
                    "KVTransferConfig object. Dict: %s. Error: %s",
                    raw_config_dict, e)
                raise ValueError(
                    f"Invalid 'kv_transfer_config' provided: {e}") from e

        # 2. Handle hf_overrides
        hf_overrides = kwargs.get('hf_overrides')
        if hf_overrides is None:
            kwargs['hf_overrides'] = {}

        # 3. Handle compilation_config
        compilation_config = kwargs.get('compilation_config')
        if compilation_config is not None:
            if isinstance(compilation_config, int):
                compilation_config_instance = CompilationConfig(
                    level=compilation_config)
            elif isinstance(compilation_config, dict):
                predicate = lambda x: is_init_field(CompilationConfig, x[0])
                compilation_config_instance = CompilationConfig(
                    **dict(filter(predicate, compilation_config.items())))
            else:
                compilation_config_instance = compilation_config
        else:
            compilation_config_instance = CompilationConfig()
        kwargs['compilation_config'] = compilation_config_instance

        # 4. Create EngineArgs for validation/logging purposes
        # (but we won't use it to create self.llm_engine)
        engine_args = EngineArgs(model=model, **kwargs)
        log_non_default_args(engine_args)

        # 5. Initialize attributes that parent would have set
        # NOTE: We skip creating self.llm_engine here!
        self.engine_class = LLMEngine  # Set default engine class
        self.request_counter = Counter()
        self.default_sampling_params: Union[dict[str, Any], None] = None
        
        # We'll set these after initializing our stage engines
        self.supported_tasks = None
        self.io_processor = None

        # 6. Initialize OmniLLM-specific attributes
        omni_args = omni_args or OmniEngineArgs(model=model, **kwargs)
        self.initalize_stage_configs(omni_args)
        self.log_stats = log_stats
        self.engine_list = []  # List of LLMEngine instances for each stage
        self.initialize_stage_engines(model=model)
        self.output_processor = MultimodalOutputProcessor()
        
        # 7. Set supported_tasks and io_processor from first engine
        if self.engine_list:
            first_engine = self.engine_list[0]
            if envs.VLLM_USE_V1:
                self.supported_tasks = first_engine.get_supported_tasks()
            else:
                self.supported_tasks = first_engine.model_config.supported_tasks
            
            logger.info("Supported_tasks: %s", self.supported_tasks)
            
            # Load the Input/Output processor plugin if any
            io_processor_plugin = first_engine.model_config.io_processor_plugin
            self.io_processor = get_io_processor(first_engine.vllm_config,
                                                 io_processor_plugin)
    
    def initalize_stage_configs(self, omni_args: OmniEngineArgs) -> None:
        """Initialize stage configs from model."""
        stage_configs = load_stage_configs(omni_args=omni_args)
        self.stage_configs = stage_configs
    
    def initialize_stage_engines(self, model: str) -> None:
        """Initialize LLMEngine instances for each stage."""
        for stage_config in self.stage_configs:
            engine_args = OmniEngineArgs(model=model, **stage_config)
            engine = LLMEngine.from_engine_args(
                    engine_args=engine_args,
                    usage_context=UsageContext.LLM_CLASS
                )
            self.engine_list.append(engine)
    
    # Override parent methods that reference self.llm_engine
    def get_tokenizer(self, lora_request=None):
        """Get tokenizer from the first engine."""
        if not self.engine_list:
            raise RuntimeError("No engines initialized")
        return self.engine_list[0].get_tokenizer_group().get_lora_tokenizer(
            lora_request)
    
    def set_tokenizer(self, tokenizer):
        """Set tokenizer for all engines."""
        from vllm.transformers_utils.tokenizer import get_cached_tokenizer
        
        for engine in self.engine_list:
            tokenizer_group = engine.get_tokenizer_group()
            if tokenizer.__class__.__name__.startswith("Cached"):
                tokenizer_group.tokenizer = tokenizer
            else:
                tokenizer_group.tokenizer = get_cached_tokenizer(tokenizer)
    
    def get_default_sampling_params(self):
        """Get default sampling params from the first engine."""
        from vllm.sampling_params import SamplingParams
        
        if self.default_sampling_params is None and self.engine_list:
            self.default_sampling_params = (
                self.engine_list[0].model_config.get_diff_sampling_param())
        if self.default_sampling_params:
            return SamplingParams.from_optional(**self.default_sampling_params)
        return SamplingParams()
    
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
        stage_config,
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
        stage_config,
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
        stage_config,
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
        stage_config = None,
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
        
        # Note: We can't use super().generate() because we don't have self.llm_engine
        # Instead, use the stage_engine or first engine from engine_list
        engine_to_use = stage_engine if stage_engine else (
            self.engine_list[0] if self.engine_list else None)
        
        if engine_to_use is None:
            raise RuntimeError("No engine available for generation")
        
        # Manually implement generation using the engine
        # This replicates what LLM.generate does internally
        from vllm.inputs import TextPrompt
        
        inputs = TextPrompt(prompt=prompt)
        
        # Process through the engine
        request_id = f"omni-{self.request_counter()}"
        engine_to_use.add_request(request_id, inputs, sampling_params)
        
        outputs = []
        while engine_to_use.has_unfinished_requests():
            step_outputs = engine_to_use.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
        
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
        stage_configs,
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
        # self.stage_manager = StageManager(stage_configs, log_stats)
        self.output_processor = MultimodalOutputProcessor()
    
    def _initialize_async_stage_engines(self) -> None:
        """Initialize AsyncLLM instances for each stage."""
        # self.stage_manager.initialize_async_engines()
    
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
        stage_config,
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
        stage_config,
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
        stage_config,
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
        stage_config = None,
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
