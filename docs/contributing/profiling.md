# Profiling vLLM-Omni

Performance Profiling Guidelines Profiling capabilities in vLLM-Omni are reserved for development and maintenance tasks aimed at temporal analysis of the codebase. Production use is **strongly discouraged**; enabling the profiler incurs a substantial overhead that negatively impacts inference latency.

**Mechanism**: vLLM-Omni implements cross-stage profiling via the PyTorch Profiler. To accommodate the architecture—where stages operate as distinct engine instances in separate processes—the profiling interface supports both holistic capturing (all stages) and targeted capturing (specific stages).

**1. Enabling the Profiler**

Before running your script, you must set the ```VLLM_TORCH_PROFILER_DIR``` environment
variable.

```Bash
export VLLM_TORCH_PROFILER_DIR=/path/to/save/traces
```

**Highly Recommended: Limit Profiling to a Single Iteration**  
For most use cases (especially when profiling audio stages), you should limit the profiler to just **one iteration** to keep trace files small and readable.


```bash
export VLLM_PROFILER_MAX_ITERS=1
```

**2. Offline Inference**

For offlinie processing using ```OmniLLM```, you can wrap your ```generate``` calls with ```start_profile``` and ```stop_profile()```.

Basic Usage(All Stages)
```Python
from vllm_omni import OmniLLM

omni_llm = OmniLLM.from_engine_args(engine_args)

# Start profiling all active stages
omni_llm.start_profile()

outputs = omni_llm.generate(prompts, sampling_params)

# Stop profiling and save traces
omni_llm.stop_profile()
```

**Selective Stage Profiling**
The profiler is default to function across all stages. But It is highly recommended to profile specific stages by passing the stages list, preventing from producing too large trace files:
```python
# Only profile Stage 1
omni_llm.start_profile(stages=[1])
```

```python
# Stage 0 (Thinker) and Stage 2 (Audio Decoder) for qwen omni
omni_llm.start_profile(stages=[0, 2])
```

**Examples**:

1. **Qwen-omni 2.5**:  [https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/qwen2_5_omni/end2end.py](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/qwen2_5_omni/end2end.py)

2. **Qwen-omni 3.0**:   [https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/qwen3_omni/end2end.py](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/qwen3_omni/end2end.py)

**3. Online Inference(Async)**

For online serving using AsyncOmni, the methods are asynchronous. This allows you to toggle profiling dynamically without restarting the server.

```python
from vllm_omni import AsyncOmni

# Inside an async function:
async_omni = AsyncOmni.from_engine_args(engine_args)

await async_omni.start_profile()

async for output in async_omni.generate(prompt, sampling_params, request_id):
    # Process outputs...
    pass

await async_omni.stop_profile()
```

**4. Analyzing Omni Traces**

After ``stop_profile()`` completes (and the file write wait time has passed), the directory specified in ```VLLM_TORCH_PROFILER_DIR``` will contain the trace files.

```
Output/
│── ...rank-0.pt.trace.json.gz   # GPU 0 trace (TP=2 Example)
│── ...rank-1.pt.trace.json.gz   # GPU 1 trace (TP=2 Example)
│       # Load these into Perfetto to visualize synchronization
│
│── profiler_out_.txt            # Summary tables (CPU/CUDA time %)
```

**Viewing Tools:**
     - [Perfetto](https://ui.perfetto.dev/): (Recommended): Best for handling large audio trace files.
     - ```chrome://tracing```: Good for smaller text-only traces.


**Note**: vLLM-Omni reuses the PyTorch Profiler infrastructure from vLLM.  
For more advanced configuration options (memory profiling, custom activities, etc.), see the official vLLM profiler documentation:  [vLLM Profiling Guide](https://docs.vllm.ai/en/latest/dev/profiling.html)
