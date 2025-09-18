# v1 Diffusion Components: Executor, Worker, Model Runner (Interfaces & Flow)

## Introduction

Background
- vLLM v1 was originally optimized for autoregressive text generation. Diffusion models (e.g., UNet/DiT + VAE + scheduler) are non‑autoregressive and run a fixed number of denoising steps. They do not use the KV cache or token sampling/logprobs pipeline.

Purpose
- Add first‑class diffusion support while strictly reusing v1 interfaces and the canonical execution flow: EngineCore → Executor → Worker → GPUModelRunner. External entry points and the main run loop remain unchanged.

Scope
- Define a DiffusionModelRunner and a Worker configuration that minimally adapts the GPU worker path for diffusion.
- Provide a second, no‑worker DiffusersPipelineExecutor for direct Diffusers pipeline execution (single process), alongside the standard MultiprocExecutor path.
- Keep EngineCore and Scheduler interfaces stable; diffusion behavior is expressed through the same methods and data containers.

Core design features
- Inheritance strategy: DiffusionModelRunner extends GPUModelRunner; Worker remains GPUWorker (with optional minimal overrides only to initialize the diffusion runner); standard MultiprocExecutor is reused as‑is. A no‑worker DiffusersPipelineExecutor extends Executor for single‑process use.
- Data flow parity: schedule() → executor.execute_model() → scheduler.update_from_output(); Worker delegates to ModelRunner; ModelRunner returns ModelRunnerOutput.
- Outputs: Reuse ModelRunnerOutput; diffusion results are carried as tensors (e.g., via pooler_output), leaving text‑specific fields as orginal vllm.
- KV cache: Treated as not required; when not registered in vllm_config, KV‑related initialization becomes a no‑op automatically.
- Distributed: Existing TP/PP/DP initialization, process orchestration, profiling, and sleep/wake behaviors remain intact for the worker‑based path.
- Acceleration: Torch compile/CUDA Graph warmup follows existing compile_or_warm_up_model hooks or the pipeline executor’s warmup helper.

Assumptions and non‑goals
- No prompt logprobs, grammar bitmask, or token sampler in diffusion.
- No new public RPCs are added; we rely on existing EngineCore/Executor/Worker/Runner calls.
- Scheduler retains the same interface (see the separate diffusion scheduler design doc for bucketing key definition and step/shape grouping).

Deliverables
- This design and the interface skeletons for: DiffusionModelRunner, (config‑selected) Worker with minimal overrides, MultiprocExecutor reuse, and DiffusersPipelineExecutor (no worker).
- Clear data flow and compatibility notes to ensure coherence across EngineCore, executors, workers, and runners.

Reading guide
- Canonical v1 Call Path gives the end‑to‑end flow we preserve.
- Executor covers both the reused MultiprocExecutor and the no‑worker DiffusersPipelineExecutor.
- Worker explains the minimal inheritance strategy from GPUWorker.
- Model Runner enumerates overridden vs. kept methods for diffusion.

## Canonical v1 Call Path (for context)
```python
# class: EngineCore
# EngineCore.step (simplified)
class EngineCore:
    def step(self):
        scheduler_output = scheduler.schedule()
        model_output = executor.execute_model(scheduler_output)
        engine_outputs = scheduler.update_from_output(
            scheduler_output, model_output
        )
        return engine_outputs
```

```python
# class: MultiprocExecutor (v1)
from concurrent.futures import Future
from typing import Union
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.executor.abstract import Executor

class MultiprocExecutor(Executor):
    # Single RPC hop
    def execute_model(
        self, scheduler_output
    ) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:
        output = collective_rpc("execute_model", args=(scheduler_output, ))
        return output
```

```python
# class: GPUWorker (v1)
# Required v1 method
from vllm.worker.worker_base import WorkerBase
from vllm.v1.outputs import ModelRunnerOutput

class GPUWorker(WorkerBase):
    def execute_model(self, scheduler_output) -> ModelRunnerOutput:
        # (driver/TP metadata broadcast handled by LocalOrDistributedWorkerBase subclasses)
        return self.model_runner.execute_model(
            scheduler_output=scheduler_output,
            intermediate_tensors=None,
        )
```


## Executor

### Function map (Executor)

#### 1) Inherited and overridden
```python
# None required for diffusion. Reuse MultiprocExecutor as-is.
# Worker class selection is driven by vllm_config; no _init_executor override.
```
<!-- 

#### Multi-worker (TP/PP/DP) behavior and aggregation (Executor)
- No KVConnector in diffusion by default → `kv_output_aggregator` is not used; the path short-circuits as in v1 when `kv_transfer_config` is None.
- Executor collects output only from the output rank, consistent with v1:
  - `output_rank = world_size - tensor_parallel_size` (i.e., TP0 of the last PP stage).
  - Other ranks participate in compute but do not emit final `ModelRunnerOutput`.
- Pipeline Parallel (PP):
  - Intermediate PP stages return `IntermediateTensors` only; last PP stage returns `ModelRunnerOutput`.
  - Executor receives only from `output_rank`.
- Tensor Parallel (TP):
  - TP ranks collaborate; TP0 produces/holds the final tensors for return.
  - Executor still receives only from `output_rank`.
- Data Parallel (DP):
  - Each DP group executes independently; each group’s executor returns its own `output_rank` result.
- Async batches (batch queue / `max_concurrent_batches>1`):
  - Results are returned via `Future`, but still only from `output_rank`; no cross-worker aggregation is needed.
 - Worker/Runner type is resolved from vllm_config (e.g., `worker_cls`, `model_cls`), not hardcoded in the executor. -->

### Diffusers Pipeline Executor (no worker)

A single-process executor that directly runs the Diffusers pipeline without spawning workers or using RPC. Interfaces remain identical to `Executor` so the EngineCore loop is unchanged.

#### Function map（Pipeline Executor）

##### 1) Inherited and overridden
```python
from concurrent.futures import Future
from typing import Optional, Union, Callable, Any
import torch
import torch.nn as nn
from vllm.v1.executor.abstract import Executor
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.tasks import SupportedTask
from diffusers import DiffusionPipeline

class DiffusersPipelineExecutor(Executor):
    supports_pp: bool = False  # Single-process, no TP/PP/DP

    def _init_executor(self) -> None:
        # Called by ExecutorBase.__init__
        self._failure_callback: Optional[Callable[[], None]] = None
        self._device = self._resolve_device()
        self._dtype = self._resolve_dtype()
        self._pipeline = self._build_pipeline(device=self._device, dtype=self._dtype)
        self._profiler = None
        self._is_failed = False
        self.is_sleeping = False
        self.sleeping_tags: set[str] = set()
    
    # major functions to build/run diffusers pipeline
    def _build_pipeline(self, device:torch.device, dtype:torch.dtype)->DiffusionPipeline:
        model_name = "Qwen/Qwen-Image"

        self.pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=dtype)
        self.pipe = pipe.to(device)

    def _run_pipeline(self, scheduler_output) -> ModelRunnerOutput: ...
        positive_magic = {
            "en": ", Ultra HD, 4K, cinematic composition.", # for english prompt
            "zh": ", 超清，4K，电影级构图." # for chinese prompt
        }

        # Generate image
        prompt_embeds = self._get_and_process_prompt_embeds(scheduler_output, positive_magic)
        negtive_prompt_embeds = self.pipe.embed_prompt(" ")


        # Generate with different aspect ratios
        aspect_ratios = {
            "1:1": (1328, 1328),
            "16:9": (1664, 928),
            "9:16": (928, 1664),
            "4:3": (1472, 1140),
            "3:4": (1140, 1472),
            "3:2": (1584, 1056),
            "2:3": (1056, 1584),
        }

        width, height = aspect_ratios["16:9"]

        image = pipe(
            prompt_embeds=prompt_embeds,
            negtive_prompt_embeds=negtive_prompt_embeds,
            width=width,
            height=height,
            num_inference_steps=50,
            true_cfg_scale=4.0,
            generator=torch.Generator(device="cuda").manual_seed(42)
        ).images[0]

        output = self.wrap_image_as_ModelRunnerOutput(image)
        return output

    # ---- Internal helpers (implementation-specific, not public API) ----
    def _resolve_device(self): ...
    def _resolve_dtype(self): ...
    def _get_model(self) -> nn.Module: ...
    def _get_and_process_prompt_embeds(self, scheduler_output, positive_magic):
        ...
        #append the positive_magic to prompt and embed them to prompt embed tensors

    # Functions related to workers should either raise NotImplementedError
    # or return empty defaults in this no-worker executor.

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None) -> list[Any]:
        # No workers in pipeline executor
        raise NotImplementedError("No workers in DiffusersPipelineExecutor")

    def initialize_from_config(self, kv_cache_configs: list[KVCacheConfig]) -> None:
        return  # no-op (pipeline already built in _init_executor)

    def register_failure_callback(self, callback):
        self._failure_callback = callback

    def determine_available_memory(self) -> list[int]:  # bytes
        # Single device; return [available_bytes]. If CPU-only, return [0].
        return [self._determine_available_bytes(self._device)]

```



## Worker

### Inheritance strategy
Prefer reusing the mature GPU Worker end-to-end. Worker class is selected by configuration (vllm_config). Do not add a new executor-specific worker binding. If customization is needed, override only `init_device` to construct the `DiffusionModelRunner`; all other behaviors (device init details, profiling, sleep/wake, PP/TP comms, execute path) remain from `vllm/v1/worker/gpu_worker.py::Worker`.



### Function map (Worker)
#### 1) Inherited and overridden
```python
# Optional: only if you need to plug a custom DiffusionModelRunner.
from vllm.v1.worker.gpu_worker import Worker as GPUWorker
from vllm.v1.worker.diffusion_model_runner import DiffusionModelRunner

class Worker(GPUWorker):
    def init_device(self) -> None:
        #those related to device check and init
        ...
        
        self.model_runner = DiffusionModelRunner(self.vllm_config, ...)
```

## Model Runner

### Function map (Model Runner)
#### 1) Inherited and overridden
Those parts relied to the KV Cache will be omitted if we do not register the model to the vllm config. The engine core will view it as do not require KV Cache, and handle it properly

Reuse `vllm/v1/outputs.py::ModelRunnerOutput`：
- DiffusionModelRunner: Use the `pooler_output=[Tensor,...]` to return multi modal tensors
- ARModelRunner: Use the `pooler_output=[Tensor,...]` to return hidden states.
```python
from typing import Optional, Union
import torch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.outputs import ModelRunnerOutput


class DiffusionModelRunner(GPUModelRunner):
    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:
        ...
        return ModelRunnerOutput(
            req_ids=[...],
            req_id_to_index={...},
            sampled_token_ids=[],
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[Tensor,...],         # Return Hidden states
            kv_connector_output=None,
            num_nans_in_logits=None,
        )# return multi modal tensors via pooler_output=[Tensor,...]


class ARModelRunner(GPUModelRunner):
    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:
        ...
        return ModelRunnerOutput(
            req_ids=[...],
            req_id_to_index={...},
            sampled_token_ids=[],
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[Tensor,...],         # Return Hidden states
            kv_connector_output=None,
            num_nans_in_logits=None,
        )# return hidden states via pooler_output=[Tensor,...]
```


## Minimal data flow (end-to-end)
1) EngineCore (DiffusionEngineCore): `_initialize_kv_caches` produces empty KV Cache; calls executor to warm up.
2) EngineCore.step(): `schedule()` (shape/step bucketing) → `executor.execute_model(scheduler_output)` → `scheduler.update_from_output(...)`.
3) Executor: single RPC to Workers `execute_model`.
4) Worker: prepares/broadcasts metadata if needed → `runner.execute_model(...)`.
5) Runner: runs diffusion (fixed T steps or pipeline)/ runs AR Model with hidden state output → returns `ModelRunnerOutput(pooler_output=...)`.
6) Scheduler: marks those requests finished and yields `EngineCoreOutputs` back to EngineCore.


