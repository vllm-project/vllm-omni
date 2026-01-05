---
toc_depth: 4
---

# Diffusion Module Architecture Design

The vLLM-Omni diffusion module (`vllm_omni/diffusion`) is a high-performance inference engine for diffusion models, designed with a modular architecture that separates concerns across multiple components. It provides efficient execution for non-autoregressive generation tasks such as image and video generation.

This document describes the architecture design of the diffusion module, including the diffusion engine, scheduler, worker, diffusion pipeline, and acceleration components.

<p align="center">
   <img src="https://github.com/user-attachments/assets/cde1ebea-4006-4d30-8283-15a5721a09d2" alt="vLLM-Omni Diffusion Module Components" width="80%">
</p>
<p align="center">
  <em> Main Components of the Diffusion Module </em>
</p>


**Table of Content:**

- [Architecture Overview](#architecture-overview)
- [Diffusion Engine](#1-diffusion-engine)
- [Scheduler](#2-scheduler)
- [Worker](#3-worker)
- [Diffusion Pipeline](#4-diffusion-pipeline)
- [Acceleration Components](#5-acceleration-components)
    - [Attention Backends](#51-attention-backends)
    - [Parallel Attention](#52-parallel-attention)
    - [Cache Backends](#53-cache-backends)
    - [Parallel Strategies](#54-parallel-strategies)
- [Data Flow](#6-data-flow)

---

## Architecture Overview

The diffusion module follows a **multi-process, distributed architecture** with clear separation of concerns:

<p align="center">
  <img src="https://github.com/user-attachments/assets/78c4d446-d238-4406-a057-eb17d2ccc8e0" alt="vLLM-Omni Diffusion Module Architecture" width="100%">
</p>
<p align="center">
  <em> Diffusion Architecture Overview </em>
</p>


---

## 1. Diffusion Engine

**Location**: `vllm_omni/diffusion/diffusion_engine.py`

### Responsibilities

The `DiffusionEngine` is the **orchestrator** of the diffusion inference system. It manages the lifecycle of worker processes and coordinates the execution flow.

### Key Components

#### 1.1 Initialization

```python
class DiffusionEngine:
    def __init__(self, od_config: OmniDiffusionConfig):
        self.od_config = od_config
        self.post_process_func = get_diffusion_post_process_func(od_config)
        self.pre_process_func = get_diffusion_pre_process_func(od_config)
        self._processes: list[mp.Process] = []
        self._make_client()
```

**Key Features**:

- **Pre/Post Processing**: Registers model-specific pre-processing and post-processing functions via registry pattern

- **Worker Management**: Launches and manages multiple worker processes (one per GPU)

- **Process Isolation**: Uses multiprocessing for true parallelism

#### 1.2 Worker Launch Process

The engine launches workers using a **spawn** method:

```python
def _launch_workers(self, broadcast_handle):
    # Creates one process per GPU
    for i in range(num_gpus):
        process = mp.Process(
            target=worker_proc.worker_main,
            args=(i, od_config, writer, broadcast_handle),
            name=f"DiffusionWorker-{i}",
        )
        process.start()
```

**Design Decisions**:

- **Spawn Method**: Ensures clean state for each worker (no shared memory issues)

- **Pipe Communication**: Uses `mp.Pipe` for initialization handshake

- **Device Selection**: Each worker is assigned a specific GPU (`cuda:{rank}`)

#### 1.3 Request Processing Flow

```python
def step(self, requests: list[OmniDiffusionRequest]):
    # 1. Pre-process requests
    requests = self.pre_process_func(requests)

    # 2. Send to scheduler and wait for response
    output = self.add_req_and_wait_for_response(requests)

    # 3. Post-process results
    result = self.post_process_func(output.output)
    return result
```

**Flow**:

1. **Pre-processing**: Applies model-specific transformations

2. **Scheduling**: Delegates to scheduler for distribution

3. **Post-processing**: Converts raw outputs to final format (e.g., PIL images)

---

## 2. Scheduler

**Location**: `vllm_omni/diffusion/scheduler.py`

### Architecture

The `Scheduler` is implemented as a **Singleton** pattern to ensure a single coordination point across the system, i.e., only one scheduler instance exists for coordination.

### Key Components

#### 2.1 Message Queue System

```python
class Scheduler:
    def initialize(self, od_config: OmniDiffusionConfig):
        # Broadcast queue: scheduler -> all workers
        self.mq = MessageQueue(
            n_reader=od_config.num_gpus,
            n_local_reader=od_config.num_gpus,
            local_reader_ranks=list(range(od_config.num_gpus)),
        )

        # Result queue: rank 0 worker -> scheduler
        self.result_mq = None  # Initialized later
```

**Communication Pattern**:

- **Broadcast Queue**: One-to-many communication (scheduler → all workers)

- **Result Queue**: One-to-one communication (rank 0 → scheduler)

- **Shared Memory**: Uses `MessageQueue` (ZMQ-based) for efficient IPC

#### 2.2 Request Distribution

```python
def add_req(self, requests: list[OmniDiffusionRequest]) -> DiffusionOutput:
    # Broadcast request to all workers
    self.mq.enqueue(requests)

    # Wait for result from Rank 0
    output = self.result_mq.dequeue()
    return output
```

**Design Features**:

- **Broadcast Model**: All workers receive the same request (for tensor parallelism)

- **Single Response**: Only rank 0 sends results back (avoids duplicate outputs)

- **Synchronous**: Blocks until result is received (can be made async)

#### 2.3 Singleton Pattern

```python
class Scheduler:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

# Global singleton instance
scheduler = Scheduler()
```

**Benefits**:

- **Single Point of Control**: Ensures consistent state

- **Easy Access**: Global `scheduler` instance accessible everywhere

- **Resource Management**: Centralized queue management

---

## 3. Worker

**Location**: `vllm_omni/diffusion/worker/gpu_worker.py`

### Architecture

Workers are **independent processes** that execute the actual model inference. Each worker runs on a dedicated GPU and participates in distributed inference.

### Key Components

#### 3.1 Worker Process Structure

```python
class WorkerProc:
    def __init__(self, od_config, gpu_id, broadcast_handle):
        # Initialize ZMQ context for IPC
        self.context = zmq.Context(io_threads=2)

        # Connect to broadcast queue (receive requests)
        self.mq = MessageQueue.create_from_handle(broadcast_handle, gpu_id)

        # Create result queue (only rank 0)
        if gpu_id == 0:
            self.result_mq = MessageQueue(n_reader=1, ...)

        # Initialize GPU worker
        self.worker = GPUWorker(local_rank=gpu_id, rank=gpu_id, od_config=od_config)
```

**Initialization Steps**:

1. **IPC Setup**: Creates ZMQ context and message queues

2. **Distributed Environment Setup**: Initializes PyTorch distributed communication

    - For CUDA GPUs: Uses NCCL (fast GPU communication)

    - For NPU: Uses HCCL (Huawei Collective Communications Library)

    - For other devices: Uses appropriate backend (GLOO, MCCL, etc.)

3. **Model Loading**: Loads diffusion pipeline on assigned GPU

4. **Cache Setup**: Enables cache backend if configured.

#### 3.2 GPU Worker

```python
class GPUWorker:
    def init_device_and_model(self):
        # Set distributed environment variables
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        # Initialize PyTorch distributed
        init_distributed_environment(world_size, rank)
        parallel_config = self.od_config.parallel_config
        initialize_model_parallel(
            data_parallel_size=parallel_config.data_parallel_size,
            cfg_parallel_size=parallel_config.cfg_parallel_size,
            sequence_parallel_size=parallel_config.sequence_parallel_size,
            tensor_parallel_size=parallel_config.tensor_parallel_size,
            pipeline_parallel_size=parallel_config.pipeline_parallel_size,
        )

        # Load model
        model_loader = DiffusersPipelineLoader(load_config)
        self.pipeline = model_loader.load_model(od_config, load_device=f"cuda:{rank}")

        # Setup cache backend
        from vllm_omni.diffusion.cache.selector import get_cache_backend
        self.cache_backend = get_cache_backend(od_config.cache_backend, od_config.cache_config)

        if self.cache_backend is not None:
            self.cache_backend.enable(self.pipeline)
```

**Key Features**:

- **Tensor Parallelism**: Supports multi-GPU tensor parallelism via PyTorch distributed

- **Model Loading**: Uses `DiffusersPipelineLoader` for efficient weight loading

- **Cache Integration**: Enables cache backends (TeaCache, cache-dit, etc.) transparently

#### 3.3 Worker Busy Loop

```python
def worker_busy_loop(self):
    while self._running:
        # 1. Receive unified message (generation request, RPC request, or shutdown)
        msg = self.recv_message()

        # 2. Route message based on type
        if isinstance(msg, dict) and msg.get("type") == "rpc":
            # Handle RPC request
            result, should_reply = self.execute_rpc(msg)
            if should_reply:
                self.return_result(result)

        elif isinstance(msg, dict) and msg.get("type") == "shutdown":
            # Handle shutdown message
            self._running = False

        else:
            # Handle generation request (OmniDiffusionRequest list)
            output = self.worker.execute_model(msg, self.od_config)
            self.return_result(output)
```

**Execution Flow**:

1. **Receive**: Dequeues unified messages from shared memory queue

2. **Route**: Handles different message types (generation, RPC, shutdown)

3. **Execute**: Runs forward pass through pipeline for generation requests

4. **Respond**: Sends results back (rank 0 for generation, specified rank for RPC)

#### 3.4 Model Execution

```python
@torch.inference_mode()
def execute_model(self, reqs: list[OmniDiffusionRequest], od_config):
    req = reqs[0]  # TODO: support batching

    # Refresh cache backend if enabled
    if self.cache_backend is not None and self.cache_backend.is_enabled():
        self.cache_backend.refresh(self.pipeline, req.num_inference_steps)

    # Set forward context for parallelism
    with set_forward_context(
        vllm_config=self.vllm_config,
        omni_diffusion_config=self.od_config
    ):
        output = self.pipeline.forward(req)
    return output
```

The model execution leverages multiple parallelism strategies that are transparently applied during the forward pass. The `set_forward_context()` context manager makes parallel group information available throughout the forward pass:

```python
# Inside transformer layers, parallel groups are accessed via:
from vllm_omni.diffusion.distributed.parallel_state import (
    get_sp_group, get_dp_group, get_cfg_group, get_pp_group
)
```

**Optimizations**:

- **Cache Refresh**: Clears cache state before each generation for clean state

- **Context Management**: Forward context ensures parallel groups are available during execution

- **Single Request**: Currently processes one request at a time (batching TODO)

---

## 4. Diffusion Pipeline

**Location**: `vllm_omni/diffusion/models/*/pipeline_*.py`

The pipeline is the **model-specific implementation** that orchestrates the diffusion process. Different models (QwenImage, Wan2.2, Z-Image) have their own pipeline implementations.

Most pipeline implementation are referred from `diffusers`. The multi-step diffusion loop is usually the most time-consuming part during the overall inference process, which is defined by the `diffuse` function in the pipeline class. An example is as follows:

```python
def diffuse(self, ...):
    for i, t in enumerate(timesteps):
        # Forward pass for positive prompt
        transformer_kwargs = {
            "hidden_states": latents,
            "timestep": timestep / 1000,
            "encoder_hidden_states": prompt_embeds,
        }
        noise_pred = self.transformer(**transformer_kwargs)[0]

        # Forward pass for negative prompt (CFG)
        if do_true_cfg:
            neg_transformer_kwargs = {...}
            neg_transformer_kwargs["cache_branch"] = "negative"
            neg_noise_pred = self.transformer(**neg_transformer_kwargs)[0]

            # Combine predictions
            comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
            noise_pred = comb_pred * (cond_norm / noise_norm)

        # Scheduler step
        latents = self.scheduler.step(noise_pred, t, latents)[0]

    return latents
```

**Key Features**:

- **CFG Support**: Handles classifier-free guidance with separate forward passes

- **Cache Branching**: Uses `cache_branch` parameter for cache-aware execution

- **True CFG**: Implements advanced CFG with norm preservation

To learn more about the diffusion pipeline and how to add a new diffusion pipeline, please view [Adding Diffusion Model](https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/model/adding_diffusion_model)

---

## 5. Acceleration Components

### 5.1 Attention Backends

**Location**: `vllm_omni/diffusion/attention/`

#### Architecture

The attention system uses a **backend selector pattern** that automatically chooses the optimal attention implementation based on hardware and model configuration.

#### Backend Selection

**Location**: `vllm_omni/diffusion/attention/selector.py`

```python
class Attention(nn.Module):
    def __init__(self, num_heads, head_size, causal, softmax_scale, ...):
        # Auto-select backend
        self.attn_backend = get_attn_backend(-1)
        self.attn_impl_cls = self.attn_backend.get_impl_cls()
        self.attention = self.attn_impl_cls(...)
```

**Available Backends**:

- **FlashAttention**: Optimized CUDA kernel (FA2/FA3) - memory efficient via tiling

- **SDPA**: PyTorch's scaled dot-product attention - default, cross-platform

- **SageAttention**: Sparse attention implementation from SageAttention library

- **AscendAttention**: NPU-optimized attention for Ascend hardware

These backends provide the **kernel implementations** for attention computation. For attention-level sequence parallelism strategies (Ring Attention, Ulysses), see [Parallel Attention](#52-parallel-attention).

#### Backend Selection Mechanism

```python
def get_attn_backend(head_size: int) -> type[AttentionBackend]:
    # Check environment variable
    backend_name = os.environ.get("DIFFUSION_ATTENTION_BACKEND")

    if backend_name:
        return load_backend(backend_name.upper())

    # Default to SDPA
    return SDPABackend
```

**Selection Priority**:

1. **Environment Variable**: `DIFFUSION_ATTENTION_BACKEND` for manual override

    - Valid values: `FLASH_ATTN`, `TORCH_SDPA`, `SAGE_ATTN`, `ASCEND`

    - Example: `export DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN`

2. **Automatic Fallback**: Falls back to SDPA if selected backend unavailable

3. **Hardware Detection**: Can select based on device type (NPU, CUDA, etc.)

**Backend Availability**:

- **SDPA**: Always available (PyTorch built-in)

- **FlashAttention**: Requires `flash-attn` package installed

- **SageAttention**: Requires `sage-attention` package (from THU-ML GitHub)

- **AscendAttention**: Only available on Ascend NPU hardware

#### Attention Backend Registry

**Location**: `vllm_omni/diffusion/attention/selector.py`

The attention system uses a **registry pattern** to manage and dynamically load attention backends. This allows for easy extension and runtime selection of backends.


**Registry Structure**:

```python
# Registry mapping backend names to their module paths and class names
_BACKEND_CONFIG = {
    "FLASH_ATTN": {
        "module": "vllm_omni.diffusion.attention.backends.flash_attn",
        "class": "FlashAttentionBackend",
    },
    "TORCH_SDPA": {
        "module": "vllm_omni.diffusion.attention.backends.sdpa",
        "class": "SDPABackend",
    },
    "SAGE_ATTN": {
        "module": "vllm_omni.diffusion.attention.backends.sage_attn",
        "class": "SageAttentionBackend",
    },
    "ASCEND": {
        "module": "vllm_omni.diffusion.attention.backends.ascend_attn",
        "class": "AscendAttentionBackend",
    },
}
```

#### Attention Backend Integration

The `Attention` layer integrates backends through a unified interface. Here's how **FlashAttentionBackend** is integrated as an example:

```python
# attention/backends/flash_attn.py

class FlashAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        return FlashAttentionImpl

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 96, 128, 192, 256]  # FlashAttention supports these head sizes


class FlashAttentionImpl(AttentionImpl):
    def __init__(self, num_heads, head_size, softmax_scale, causal, ...):
        self.num_heads = num_heads
        self.causal = causal
        self.softmax_scale = softmax_scale

    def forward(self, query, key, value, attn_metadata=None):
        # Call FlashAttention kernel
        out = flash_attn_func(
            query, key, value,
            causal=self.causal,
            softmax_scale=self.softmax_scale,
        )
        return out
```

---

### 5.2 Parallel Attention

**Location**: `vllm_omni/diffusion/attention/parallel/`

#### Architecture

Parallel attention strategies implement **Sequence Parallelism (SP) at the attention layer level**. These strategies distribute attention computation across multiple GPUs by splitting the sequence dimension, using different communication patterns. They work **on top of** AttentionBackend implementations (FlashAttention, SDPA, etc.), handling the parallelization/communication while the backends handle the actual attention computation.

**Key Distinction**: Unlike AttentionBackend (which provides kernel implementations), ParallelAttentionStrategy provides communication patterns for multi-GPU attention parallelism. These strategies implement the `ParallelAttentionStrategy` interface and use AttentionBackend implementations internally.

Both Ring Attention and Ulysses are forms of Sequence Parallelism (SP) that:

- Split the sequence dimension across GPUs

- Contribute to `sequence_parallel_size` (via `ring_degree` and `ulysses_degree`)

- Work at the attention layer level (not model/pipeline level)

#### Ulysses Sequence Parallelism (USP)

**Location**: `vllm_omni/diffusion/attention/parallel/ulysses.py`

USP is a sequence-parallel attention strategy that splits attention computation across multiple GPUs by distributing both the sequence dimension and attention heads. It uses **all-to-all communication** to efficiently parallelize attention for very long sequences. Specifically, it uses **all-to-all** collective operations to redistribute Q/K/V tensors before attention computation and gather results afterward.

Ulysses splits attention computation in two dimensions:

1. **Sequence Dimension**: Splits the sequence length across GPUs

2. **Head Dimension**: Splits attention heads across GPUs

**Configuration**: `ulysses_degree` contributes to `sequence_parallel_size`

#### Ring Sequence Parallelism

**Location**: `vllm_omni/diffusion/attention/parallel/ring.py`

Ring Attention is a **parallel attention strategy** that implements sequence parallelism using ring-based point-to-point (P2P) communication. Unlike attention backends that provide the attention kernel implementation, Ring Attention is a **communication pattern** that works on top of attention backends (FlashAttention or SDPA).

Ring Attention splits sequence dimension across GPUs in a ring topology, implemented via the `ParallelAttentionStrategy` interface, instead of `AttentionBackend`. P2P ring communication is applied to circulate Key/Value blocks across GPUs. Internally, `ring_flash_attn_func` or `ring_pytorch_attn_func` is used depending on available backends.

**Architecture**:
```python
class RingParallelAttention:
    """Ring sequence-parallel strategy."""

    def run_attention(self, query, key, value, attn_metadata, ...):
        # Selects underlying attention kernel (FlashAttention or SDPA)
        if backend_pref == "sdpa":
            return ring_pytorch_attn_func(...)  # Uses SDPA kernel
        else:
            return ring_flash_attn_func(...)    # Uses FlashAttention kernel
```

**Integration**:

- Ring Attention is activated when `ring_degree > 1` in parallel config

- It's selected by `build_parallel_attention_strategy()` in the attention layer

- The `Attention` layer routes to `_run_ring_attention()` when Ring is enabled

- Works alongside attention backends: Ring handles communication, backends handle computation

**Configuration**: `ring_degree` contributes to `sequence_parallel_size`

#### Relationship with AttentionBackend

Parallel attention strategies (Ring, Ulysses) work **on top of** AttentionBackend implementations:

- They use AttentionBackend for the actual attention computation (FlashAttention, SDPA, etc.)

- They handle the multi-GPU communication/parallelization layer

- They implement `ParallelAttentionStrategy` interface (not `AttentionBackend`)

For general parallelism strategies (Data Parallelism, Tensor Parallelism, Pipeline Parallelism), see [Parallel Strategies](#54-parallel-strategies).

---

### 5.3 Cache Backends

**Location**: `vllm_omni/diffusion/cache/`

#### Architecture

Cache backends provide a **unified interface** for applying different caching strategies to accelerate diffusion inference. The system supports multiple backends (TeaCache, cache-dit) with a consistent API for enabling and refreshing cache state.

#### Cache Backend Interface

```python
class CacheBackend(ABC):
    def __init__(self, config: DiffusionCacheConfig):
        self.config = config
        self.enabled = False

    @abstractmethod
    def enable(self, pipeline: Any) -> None:
        """Enable cache on the pipeline."""
        raise NotImplementedError

    @abstractmethod
    def refresh(self, pipeline: Any, num_inference_steps: int, verbose: bool = True) -> None:
        """Refresh cache state for new generation."""
        raise NotImplementedError

    def is_enabled(self) -> bool:
        """Check if cache is enabled."""
        return self.enabled
```

**Design Pattern**:

- **Abstract Base Class**: Defines contract for all cache backends

- **Pipeline-based**: Works with pipeline instances (not just transformers)

- **State Management**: Provides refresh mechanism for clean state between generations

#### Available Backends

**1. TeaCache Backend**

**Location**: `vllm_omni/diffusion/cache/teacache/backend.py`

```python
class TeaCacheBackend(CacheBackend):
    def enable(self, pipeline: Any):
        # Extract transformer from pipeline
        transformer = pipeline.transformer
        transformer_type = transformer.__class__.__name__

        # Create TeaCacheConfig from DiffusionCacheConfig
        teacache_config = TeaCacheConfig(
            transformer_type=transformer_type,
            rel_l1_thresh=self.config.rel_l1_thresh,
            coefficients=self.config.coefficients,
        )

        # Apply hooks to transformer
        apply_teacache_hook(transformer, teacache_config)
        self.enabled = True

    def refresh(self, pipeline: Any, num_inference_steps: int, verbose: bool = True):
        transformer = pipeline.transformer
        if hasattr(transformer, "_hook_registry"):
            transformer._hook_registry.reset_hook(TeaCacheHook._HOOK_NAME)
```

**TeaCache Features**:

- **Timestep-aware**: Caches based on timestep embedding similarity

- **Adaptive**: Dynamically decides when to reuse cached computations

- **CFG-aware**: Handles positive/negative branches separately

- **Custom Hook System**: Uses a custom forward interception mechanism (via `HookRegistry`) that wraps the module's `forward` method, allowing transparent integration without modifying model code

**2. Cache-DiT Backend**

**Location**: `vllm_omni/diffusion/cache/cache_dit_backend.py`

```python
class CacheDiTBackend(CacheBackend):
    def enable(self, pipeline: Any):
        # Uses cache-dit library for acceleration
        # Supports DBCache, SCM (Step Computation Masking), TaylorSeer
        # Works with single and dual-transformer architectures
        ...
        self.enabled = True

    def refresh(self, pipeline: Any, num_inference_steps: int, verbose: bool = True):
        # Updates cache context with new num_inference_steps
        ...
```

**Cache-DiT Features**:

- **DBCache**: Dynamic block caching with configurable compute blocks

- **SCM**: Step Computation Masking for additional speedup

- **TaylorSeer**: Advanced calibration for cache accuracy

- **Dual-transformer Support**: Handles models like Wan2.2 with two transformers

#### Cache Backend Selector

**Location**: `vllm_omni/diffusion/cache/selector.py`

```python
def get_cache_backend(
    cache_backend: str | None,
    cache_config: dict | DiffusionCacheConfig
) -> CacheBackend | None:
    """Get cache backend instance based on cache_backend string.

    Args:
        cache_backend: Cache backend name ("cache_dit", "tea_cache", or None)
        cache_config: Cache configuration (dict or DiffusionCacheConfig)

    Returns:
        Cache backend instance or None if cache_backend is "none"
    """
    if cache_backend is None or cache_backend == "none":
        return None

    if isinstance(cache_config, dict):
        cache_config = DiffusionCacheConfig.from_dict(cache_config)

    if cache_backend == "cache_dit":
        return CacheDiTBackend(cache_config)
    elif cache_backend == "tea_cache":
        return TeaCacheBackend(cache_config)
    else:
        raise ValueError(f"Unsupported cache backend: {cache_backend}")
```

**Usage Flow**:

1. **Selection**: `get_cache_backend()` returns appropriate backend instance

2. **Enable**: `backend.enable(pipeline)` called during worker initialization

3. **Refresh**: `backend.refresh(pipeline, num_inference_steps)` called before each generation

4. **Check**: `backend.is_enabled()` verifies cache is active

### 5.4 Parallel Strategies

**Location**: `vllm_omni/diffusion/distributed/parallel_state.py`

#### Parallelism Types

The system supports multiple orthogonal parallelism strategies:

**Sequence Parallelism (SP)**

- **Purpose**: Split sequence dimension across GPUs

- **Attention-level SP**: Ring Attention and Ulysses (USP) implement SP at the attention layer level

    - See [Parallel Attention](#52-parallel-attention) for details

    - Configuration: `ulysses_degree` × `ring_degree` = `sequence_parallel_size`

- **Use Case**: Very long sequences (e.g., high-resolution images)

**Data Parallelism (DP)**

- **Purpose**: Replicate model across GPUs, split batch

- **Use Case**: Batch processing, throughput optimization

**Tensor Parallelism (TP)** (Experimental)

- **Purpose**: Split model weights across GPUs

- **Implementation**: Uses vLLM's tensor parallel groups

- **Use Case**: Large models that don't fit on single GPU

**CFG Parallelism**  (under development)

- **Purpose**: Parallelize Classifier-Free Guidance (positive/negative prompts)

- **Infrastructure**: CFG parallel groups are initialized and available via `get_cfg_group()`

#### Parallel Group Management

```python
def initialize_model_parallel(
    data_parallel_size: int = 1,
    cfg_parallel_size: int = 1,
    sequence_parallel_size: int | None = None,
    ulysses_degree: int = 1,
    ring_degree: int = 1,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    vae_parallel_size: int = 0,
):
    # Generate orthogonal parallel groups
    rank_generator = RankGenerator(
        tensor_parallel_size,
        sequence_parallel_size,
        pipeline_parallel_size,
        cfg_parallel_size,
        data_parallel_size,
        "tp-sp-pp-cfg-dp",
    )

    # Initialize each parallel group
    _DP = init_model_parallel_group(rank_generator.get_ranks("dp"), ...)
    _CFG = init_model_parallel_group(rank_generator.get_ranks("cfg"), ...)
    _SP = init_model_parallel_group(rank_generator.get_ranks("sp"), ...)
    _PP = init_model_parallel_group(rank_generator.get_ranks("pp"), ...)
    _TP = init_model_parallel_group(rank_generator.get_ranks("tp"), ...)
```

**Rank Order**: `tp-sp-pp-cfg-dp` (tensor → sequence → pipeline → cfg → data)

**Note**: For attention-level Sequence Parallelism implementations (Ring Attention and Ulysses), see [Parallel Attention](#52-parallel-attention). This section covers higher-level parallelism strategies.


---

## 6. Data Flow

### Complete Request Flow

<p align="center">
   <img src="https://github.com/user-attachments/assets/6e093a0d-29c0-4efd-85d5-747002bd2fed" alt="vLLM-Omni Diffusion Module Components" width="100%">
</p>
<p align="center">
  <em> End-to-end Data Flow in the vLLM-Omni Diffusion Module </em>
</p>


```
1. User Request
   └─> OmniDiffusion.generate(prompt)
       └─> Prepare OmniDiffusionRequest
           └─> DiffusionEngine.step(requests)

2. Pre-processing
   └─> pre_process_func(requests)
       └─> Model-specific transformations

3. Scheduling
   └─> scheduler.add_req(requests)
       └─> Broadcast via MessageQueue to all workers

4. Worker Execution
   └─> WorkerProc.worker_busy_loop()
       └─> GPUWorker.execute_model(reqs)
           └─> Pipeline.forward(req)
               ├─> encode_prompt()
               ├─> prepare_latents()
               ├─> diffuse() [loop]
               │   ├─> transformer.forward() [with cache backend hooks]
               │   └─> scheduler.step()
               └─> vae.decode()

5. Result Collection
   └─> Rank 0 sends DiffusionOutput via result queue
       └─> Scheduler receives and returns

6. Post-processing
   └─> post_process_func(output)
       └─> Convert to PIL images / final format
```

---
