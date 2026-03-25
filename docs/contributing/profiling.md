# Profiling vLLM-Omni

> **Warning:** Profiling incurs significant overhead. Use only for development and debugging, never in production.

vLLM-Omni uses the PyTorch Profiler to analyze performance across both **multi-stage omni-modality models** and **diffusion models**.

### 1. Configure Profiling in the Stage YAML

Enable profiling by adding `profiler_config` under `engine_args` for the stage(s) you want to profile in your stage config YAML:

```yaml
stage_args:
  - stage_id: 0
    stage_type: llm
    engine_args:
      # ... other engine args ...
      profiler_config:
        profiler: torch
        torch_profiler_dir: ./perf
```

| Field | Description |
|---|---|
| `profiler` | Profiler backend to use. Currently supports `torch`. |
| `torch_profiler_dir` | Directory where trace files are saved. Created automatically if it doesn't exist. |

> **Tip:** Only enable `profiler_config` on stages you actually need to profile. Stages without it will not start a profiler, keeping overhead minimal.

### 2. Profiling Omni-Modality Models

**Selective Stage Profiling**

It is highly recommended to profile specific stages to prevent producing overly large trace files:

```python
# Profile all stages
omni_llm.start_profile()

# Only profile Stage 1
omni_llm.start_profile(stages=[1])

# Stage 0 (Thinker) and Stage 2 (Audio Decoder) for qwen omni
omni_llm.start_profile(stages=[0, 2])
```

> **Important:** Always pass the same `stages` list to both `start_profile()` and `stop_profile()`. If you omit `stages` from `stop_profile()`, it defaults to stopping all stages — including ones that were never started — which will produce errors.

**Python Usage**: Wrap your generation logic with `start_profile()` and `stop_profile()`.

```python
profiler_stages = [0]  # Only profile the stages you need

# 1. Start profiling
omni.start_profile(stages=profiler_stages)

# Initialize generator
omni_generator = omni.generate(prompts, sampling_params_list, py_generator=args.py_generator)

total_requests = len(prompts)
processed_count = 0

# Main Processing Loop
for stage_outputs in omni_generator:

    # ... [Output processing logic for text/audio would go here] ...

    # Update count to track when to stop profiling
    processed_count += len(stage_outputs.request_output)

    # 2. Check if all requests are done to stop the profiler safely
    if profiler_enabled and processed_count >= total_requests:
        print(f"[Info] Processed {processed_count}/{total_requests}. Stopping profiler inside active loop...")

        # Stop the profiler while workers are still active
        # Pass the same stages list used in start_profile()
        omni_llm.stop_profile(stages=profiler_stages)

        # Wait for traces to flush to disk
        print("[Info] Waiting 30s for workers to write trace files to disk...")
        time.sleep(30)
        print("[Info] Trace export wait time finished.")

omni_llm.close()
```


**CLI Usage** (using `end2end.py`):
```bash
# Profile only Stage 0 (Thinker)
python end2end.py --output-wav output_audio \
    --query-type text --enable-profiler --profiler-stages 0

# Profile Stage 0 and Stage 2
python end2end.py --output-wav output_audio \
    --query-type text --enable-profiler --profiler-stages 0 2

# Profile all stages (omit --profiler-stages)
python end2end.py --output-wav output_audio \
    --query-type text --enable-profiler
```

**Examples**:

1. **Qwen2.5-Omni**:  [https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/qwen2_5_omni/end2end.py](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/qwen2_5_omni/end2end.py)

2. **Qwen3-Omni**:   [https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/qwen3_omni/end2end.py](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/qwen3_omni/end2end.py)

### 3. Profiling diffusion models

Diffusion profiling is End-to-End, capturing encoding, denoising loops, and decoding. Standalone diffusion scripts use `--profiler-dir` to enable profiling.

**CLI Usage:**
```bash
python image_to_video.py \
    --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
    --image qwen-bear.png \
    --prompt "A cat playing with yarn, smooth motion" \
    --profiler-dir \
    \
    # Minimize Spatial Dimensions (Optional but helpful):
    #    Drastically reduces memory usage so the profiler doesn't
    #    crash due to overhead, though for accurate performance
    #    tuning you often want target resolutions.
    --height 48 \
    --width 64 \
    \
    # Minimize Temporal Dimension (Frames):
    #    Video models process 3D tensors (Time, Height, Width).
    #    Reducing frames to the absolute minimum (2) keeps the
    #    tensor size small, ensuring the trace file doesn't become
    #    multi-gigabytes in size.
    --num-frames 2 \
    \
    # Minimize Iteration Loop (Steps):
    #    This is the most critical setting for profiling.
    #    Diffusion models run the same loop X times.
    #    Profiling 2 steps gives you the exact same performance
    #    data as 50 steps, but saves minutes of runtime and
    #    prevents the trace viewer from freezing.
    --num-inference-steps 2 \
    \
    --guidance-scale 5.0 \
    --guidance-scale-high 6.0 \
    --boundary-ratio 0.875 \
    --flow-shift 12.0 \
    --fps 16 \
    --output i2v_output.mp4
```

> **Note:** For diffusion stages within a multi-stage omni pipeline, use `profiler_config` in the stage YAML instead (see Section 1).

**Examples**:

1. **Qwen image edit**:  [https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/image_to_image/image_edit.py](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/image_to_image/image_edit.py)

2. **Wan-AI/Wan2.2-I2V-A14B-Diffusers**:   [https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/image_to_video](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/image_to_video)

### 4. Profiling Online Serving

When `profiler_config` is set in the stage YAML, the server automatically exposes `/start_profile` and `/stop_profile` HTTP endpoints.

**1. Start the server** with a stage YAML that has `profiler_config` enabled:
```bash
vllm serve Qwen/Qwen2.5-Omni-7B \
    --omni \
    --stage-configs-path qwen2_5_omni.yaml \
    --port 8091
```

Or for one stage diffusion models:

```bash
vllm serve Wan-AI/Wan2.2-I2V-A14B-Diffusers --omni --port 8091 --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile"}'
```

**2. Start profiling** by sending a POST request:
```bash
# Profile all stages that have profiler_config set
curl -X POST http://localhost:8091/start_profile

# Profile specific stages only
curl -X POST http://localhost:8091/start_profile \
    -H "Content-Type: application/json" \
    -d '{"stages": [0]}'
```

**3. Send your inference requests** as normal while the profiler is running.

**4. Stop profiling** and collect traces:
```bash
# Stop all stages
curl -X POST http://localhost:8091/stop_profile

# Stop specific stages (must match the stages you started)
curl -X POST http://localhost:8091/stop_profile \
    -H "Content-Type: application/json" \
    -d '{"stages": [0]}'
```

Trace files are written to the `torch_profiler_dir` specified in your stage YAML.

> **Important:** Always stop the same stages you started. Stopping a stage that was never started will produce errors.

### 5. Analyzing Traces

Output files are saved to the `torch_profiler_dir` specified in your stage YAML config.

**Output**
**Chrome Trace** (`.json.gz`): Visual timeline of kernels and stages. Open in Perfetto UI.

**Viewing Tools:**

- [Perfetto](https://ui.perfetto.dev/) (recommended)
- `chrome://tracing` (Chrome only)

**Note**: vLLM-Omni reuses the PyTorch Profiler infrastructure from vLLM. See the official vLLM profiler documentation:  [vLLM Profiling Guide](https://docs.vllm.ai/en/stable/contributing/profiling/)
