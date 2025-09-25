# vLLM-omni Implementation Architecture

## 1. Package Structure

```
vllm_omni/
├── __init__.py                 # Main package exports
├── config/                     # Configuration management
│   ├── __init__.py
│   ├── stage_config.py        # OmniStageConfig implementation
│   ├── dit_config.py          # DiT-specific configurations
│   └── cache_config.py        # Caching configurations
├── core/                       # Core processing components
│   ├── __init__.py
│   ├── omni_llm.py            # OmniLLM and AsyncOmniLLM
│   ├── stage_manager.py       # Multi-stage orchestration
│   ├── dit_cache_manager.py   # DiT caching system
│   └── sched/                 # Schedulers
│       ├── __init__.py
│       ├── scheduler.py       # Base scheduler interface
│       └── diffusion_scheduler.py  # DiT scheduler
├── engine/                     # Engine components
│   ├── __init__.py
│   ├── processor.py           # Output processing
│   └── output_processor.py    # Multimodal output handling
├── executor/                   # Executor implementations
│   ├── __init__.py
│   ├── base_executor.py       # Base executor interface
│   └── diffusers_executor.py  # Diffusers pipeline executor
├── model_executor/            # Model execution components
│   ├── __init__.py
│   ├── ar_model_runner.py     # OmniARModelRunner
│   └── dit_model_runner.py    # OmniDiffusionModelRunner
├── worker/                     # Worker implementations
│   ├── __init__.py
│   └── omni_worker.py         # Extended worker for DiT
├── entrypoints/               # Entry points and CLI
│   ├── __init__.py
│   ├── omni.py               # OmniServeCommand
│   └── cli/                  # CLI integration
│       ├── __init__.py
│       └── main.py           # CLI main entry point
├── request.py                 # Request handling
├── dit_cache_interface.py     # DiT cache interface
└── utils/                     # Utility functions
    ├── __init__.py
    ├── multimodal.py         # Multimodal utilities
    └── vae.py                # VAE utilities for image processing
```

## 2. Core Module Dependencies

### 2.1 vLLM Integration Points

```python
# Key vLLM imports and extensions
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.core.sched.scheduler import SchedulerInterface
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.worker.gpu_worker import Worker as GPUWorker
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.outputs import ModelRunnerOutput
```

### 2.2 Internal Dependencies

```python
# Internal module dependencies
vllm_omni.config → vllm_omni.core
vllm_omni.core → vllm_omni.engine, vllm_omni.executor, vllm_omni.worker
vllm_omni.engine → vllm_omni.model_executor
vllm_omni.entrypoints → vllm_omni.core
```

## 3. Configuration System

### 3.1 Stage Configuration

```python
# vllm_omni/config/stage_config.py
@dataclass
class OmniStageConfig:
    stage_id: int
    engine_type: Literal["AR", "DiT"]
    model_path: str
    input_modalities: List[str]
    output_modalities: List[str]
    vllm_config: Optional[VllmConfig] = None
    executor_class: type[Executor] = MultiprocExecutor
    dit_config: Optional[DiTConfig] = None
    cache_config: Optional[DiTCacheConfig] = None
    stage_output: Optional[Any] = None
```

### 3.2 DiT Configuration

```python
# vllm_omni/config/dit_config.py
@dataclass
class DiTConfig:
    model_type: str
    scheduler_type: str
    num_inference_steps: int
    guidance_scale: float
    use_diffusers: bool = False
    diffusers_pipeline: Optional[str] = None
```

### 3.3 Cache Configuration

```python
# vllm_omni/config/cache_config.py
@dataclass
class DiTCacheConfig:
    cache_tensors: List[DiTCacheTensor]
    max_cache_size: int
    cache_strategy: str = "fifo"
    enable_optimization: bool = True
```

## 4. Core Implementation Details

### 4.1 OmniLLM Implementation

```python
# vllm_omni/core/omni_llm.py
class OmniLLM(LLM):
    def __init__(self, stage_configs: List[OmniStageConfig]):
        super().__init__()
        self.stage_configs = stage_configs
        self.engine_list: List[LLMEngine] = []
        self.output_processor = MultimodalOutputProcessor()
        self._initialize_stage_engines()
    
    def _initialize_stage_engines(self) -> None:
        """Initialize LLMEngine instances for each stage"""
        for stage_config in self.stage_configs:
            if stage_config.engine_type == "AR":
                engine = self._create_ar_engine(stage_config)
            elif stage_config.engine_type == "DiT":
                engine = self._create_dit_engine(stage_config)
            self.engine_list.append(engine)
    
    def generate(self, stage_args_list: List[Dict], **kwargs) -> List[RequestOutput]:
        """Main generation interface - orchestrates multi-stage processing"""
        current_output = None
        
        for i, (stage_config, stage_args) in enumerate(zip(self.stage_configs, stage_args_list)):
            stage_engine = self.engine_list[i]
            
            # Prepare input for this stage
            processed_input = self._process_stage_inputs(stage_config, stage_args, current_output)
            
            # Execute stage
            stage_output = self._execute_stage(stage_engine, processed_input)
            
            # Update for next stage
            current_output = stage_output
            stage_config.stage_output = stage_output
        
        # Process final output
        return self.output_processor.process_output(current_output)
```

### 4.2 AsyncOmniLLM Implementation

```python
# vllm_omni/core/omni_llm.py
class AsyncOmniLLM(AsyncLLM):
    def __init__(self, stage_configs: List[OmniStageConfig]):
        super().__init__()
        self.stage_configs = stage_configs
        self.async_engine_list: List[AsyncLLM] = []
        self.output_processor = MultimodalOutputProcessor()
        self._initialize_async_stage_engines()
    
    async def generate_async(self, stage_args_list: List[Dict], **kwargs) -> List[RequestOutput]:
        """Async generation interface"""
        # Similar to OmniLLM but with async/await patterns
        pass
```

### 4.3 DiT Scheduler Implementation

```python
# vllm_omni/core/sched/diffusion_scheduler.py
class OmniDiffusionScheduler(SchedulerInterface):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        dit_cache_config: DiTCacheConfig,
        structured_output_manager: StructuredOutputManager,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        super().__init__(vllm_config, kv_cache_config, structured_output_manager, 
                        mm_registry, include_finished_set, log_stats)
        self.dit_cache_manager = DiTCacheManager(dit_cache_config)
    
    def schedule(self) -> SchedulerOutput:
        """Schedule DiT requests with caching optimization"""
        # Implement DiT-specific scheduling logic
        pass
```

### 4.4 Model Runners

```python
# vllm_omni/model_executor/dit_model_runner.py
class OmniDiffusionModelRunner(GPUModelRunner):
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:
        # DiT model execution logic
        return ModelRunnerOutput(
            req_ids=[...],
            req_id_to_index={...},
            sampled_token_ids=[],
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[tensor, ...],  # DiT output tensors
            kv_connector_output=None,
            num_nans_in_logits=None,
        )

# vllm_omni/model_executor/ar_model_runner.py
class OmniARModelRunner(GPUModelRunner):
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:
        # AR model execution with hidden state output
        return ModelRunnerOutput(
            req_ids=[...],
            req_id_to_index={...},
            sampled_token_ids=[],
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[hidden_states, ...],  # Hidden states
            kv_connector_output=None,
            num_nans_in_logits=None,
        )
```

### 4.5 Output Processing

```python
# vllm_omni/engine/output_processor.py
class MultimodalOutputProcessor(OutputProcessor):
    def __init__(self):
        self.output_handlers: Dict[str, Callable] = {
            "image": self._process_image_output,
            "text+image": self._process_text_image_output,
            "latents": self._process_latents_output,
            "text": self._process_text_output,
        }
    
    def process_outputs(self, engine_core_outputs: List[EngineCoreOutput], ...):
        """Process multimodal outputs based on type"""
        for engine_core_output in engine_core_outputs:
            output_type = self._detect_output_type(engine_core_output)
            handler = self.output_handlers.get(output_type, self._process_pooling_output)
            handler(engine_core_output)
```

## 5. CLI Integration

### 5.1 Entry Point Override

```python
# vllm_omni/entrypoints/cli/main.py
def main():
    """Main CLI entry point that intercepts vLLM commands"""
    if "--omni" in sys.argv:
        omni_args = [arg for arg in sys.argv[1:] if arg != "--omni"]
        omni_serve = OmniServeCommand()
        omni_serve.run(omni_args)
    else:
        from vllm.entrypoints.cli.main import main as vllm_main
        vllm_main()
```

### 5.2 Package Configuration

```toml
# pyproject.toml updates
[project.scripts]
vllm = "vllm_omni.entrypoints.cli.main:main"
vllm-omni = "vllm_omni.entrypoints.cli.main:main"

[project.entry-points."vllm.plugins"]
omni = "vllm_omni.plugin:OmniPlugin"
```

## 6. Testing Strategy

### 6.1 Unit Tests
- Individual component testing
- Mock vLLM dependencies
- Configuration validation

### 6.2 Integration Tests
- End-to-end pipeline testing
- vLLM compatibility testing
- Multi-stage processing validation

### 6.3 Performance Tests
- Benchmarking against native vLLM
- Memory usage profiling
- Latency measurements

## 7. Installation and Setup

### 7.1 Package Installation
```bash
pip install vllm>=0.2.0
pip install vllm-omni
```

### 7.2 Development Setup
```bash
git clone https://github.com/hsliuustc0106/vllm-omni
cd vllm-omni
pip install -e ".[dev]"
```

### 7.3 Usage
```bash
vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8000
```

## 8. Memory Management

### 8.1 DiT Cache Management
- Tensor caching for intermediate results
- Memory pooling for efficient allocation
- Cache eviction strategies

### 8.2 Multi-Stage Memory
- Inter-stage data passing optimization
- Memory sharing between stages
- Garbage collection optimization

## 9. Error Handling

### 9.1 Stage Failure Handling
- Graceful degradation on stage failures
- Error propagation and reporting
- Recovery mechanisms

### 9.2 vLLM Compatibility
- Version compatibility checks
- API change detection
- Fallback mechanisms

## 10. Monitoring and Logging

### 10.1 Performance Metrics
- Stage execution times
- Memory usage per stage
- Throughput measurements

### 10.2 Debug Information
- Request tracing across stages
- Cache hit/miss ratios
- Error logging and reporting
