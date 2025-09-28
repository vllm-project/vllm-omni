# vLLM-omni Software Design Document

## Overview

vLLM-omni extends vLLM with multi-stage, multimodal capabilities. It supports both autoregressive (AR) stages and diffusion (DiT) stages, enabling pipelines such as text→image generation.

## Architecture Principles

1. **vLLM V1 Compatibility** – Reuse LLMEngine/AsyncLLM and executor abstractions.
2. **Stage-based Processing** – Each stage encapsulates its own engine and configuration.
3. **Executor/Worker Separation** – Executors remain thin; model-specific logic lives in workers and runners.
4. **Extensibility** – New modalities can be added by implementing dedicated stage configs, workers, and output processors.

## Key Data Flow

```
Request → OmniLLM/AsyncOmniLLM → StageManager → LLMEngine/Executor → Worker → ModelRunner → OutputProcessor → Response
```

For diffusion stages, the executor hands off to `DiffusionGPUWorker`, which wraps `DiffusionModelRunner` (responsible for diffusers pipeline loading and inference).

## Core Components (Ordered by Data Flow)

### 1. Installation
```bash
pip install vllm
pip install vllm-omni
```

### 2. Online inference launch (Entry Point)
```bash
vllm serve Qwen/Qwen3-0.6B --omni --port 8000
```

### 3. Omni Serve Command (CLI)
```python
# vllm_omni/entrypoints/cli/main.py
if "--omni" in sys.argv:
    omni_args = [arg for arg in sys.argv[1:] if arg not in {"--omni", "serve"}]
    OmniServeCommand().run(omni_args)
else:
    from vllm.entrypoints.cli.main import main as vllm_main
    vllm_main()
```

### 4. Offline Inference (OmniLLM)
```python
# vllm_omni/entrypoints/omni_llm.py
class OmniLLM(LLM):
    def __init__(..., stage_configs):
        super().__init__(model=stage_configs[0].model_path if stage_configs else "test-model")
        self.stage_manager = StageManager(stage_configs)
        self.output_processor = MultimodalOutputProcessor()
        self.stage_manager.initialize_engines()

    def generate(...):
        current_output = None
        for idx, (config, args) in enumerate(zip(self.stage_configs, stage_args_list)):
            engine = self.stage_manager.get_engine(idx)
            processed = self._process_stage_inputs(config, args, current_output)
            stage_output = self._execute_stage(engine, processed, stage_config=config)
            current_output = stage_output
            config.stage_output = stage_output
        return self.output_processor.process_output(current_output)
```

### 5. Stage Manager
```python
# vllm_omni/core/stage_manager.py
class StageManager:
    def initialize_engines(self):
        if self._initialized:
            return
        for config in self.stage_configs:
            if config.engine_type == "AR":
                engine = self._create_ar_engine(config)
            elif config.engine_type == "DiT":
                engine = self._create_dit_engine(config)
            self.engine_list.append(engine)
        self._initialized = True
```

### 6. Diffusion Execution (DiT Stage)
```python
# vllm_omni/worker/diffusion_model_runner.py
class DiffusionModelRunner:
    def __init__(self, model_path, pipeline_name=None, device=None, dtype=None):
        self.device, self.torch_dtype = self._resolve_device_and_dtype(device, dtype)
        self._pipeline = self._load_pipeline(model_path, pipeline_name)

    def generate(...):
        output = self._pipeline(...)
        return DiffusionRunnerOutput(prompt=prompt, images=getattr(output, "images", []) or [])
```

```python
# vllm_omni/worker/gpu_diffusion_worker.py
class DiffusionGPUWorker:
    def __init__(self, model_path, ...):
        self.model_runner = DiffusionModelRunner(model_path, ...)

    def generate(...):
        return self.model_runner.generate(...)
```

```python
# vllm_omni/executor/diffusers_executor.py
class DiffusersPipelineExecutor:
    def __init__(self, model_path, ...):
        self.worker = DiffusionGPUWorker(model_path, ...)

    def generate(...):
        return self.worker.generate(...)
```

This mirrors the standard executor→worker→runner layering in vLLM.

### 7. Stage Configuration
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
    executor_class: Optional[type[Executor]] = None
    dit_config: Optional[DiTConfig] = None
```

- AR stages default to `UniProcExecutor`.
- DiT stages set `executor_class` to `DiffusersPipelineExecutor` when `use_diffusers=True`.

### 8. Output Processing
```python
# vllm_omni/engine/output_processor.py
class MultimodalOutputProcessor(OutputProcessor):
    def process_output(self, engine_core_output):
        output_type = self._detect_output_type(engine_core_output)
        handler = self.output_handlers.get(output_type, self._process_pooling_output)
        return handler(engine_core_output)
```

### 9. CLI Package Configuration
```toml
# pyproject.toml
[project.scripts]
vllm = "vllm_omni.entrypoints.cli.main:main"
vllm-omni = "vllm_omni.entrypoints.cli.main:main"
```

## Future Work
- Scheduler adaptations for diffusion batching.
- Advanced output processors for audio/video.
- Multi-worker execution (ray / multiprocess) for heavy DiT stages.
