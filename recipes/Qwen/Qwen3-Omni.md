# Qwen3-Omni

> Online serving for multimodal chat
>
> Ray batch inference

## Summary

- Vendor: Qwen
- Models:
  - **Instruct:** `Qwen/Qwen3-Omni-30B-A3B-Instruct` — multimodal chat with text
    and audio output (3-stage pipeline)
  - **Thinking:** `Qwen/Qwen3-Omni-30B-A3B-Thinking` — multimodal understanding
    with text-only output (thinker-only, single stage)
- Task: Multimodal chat with text/image/audio/video input
- Mode: Online serving with the OpenAI-compatible API; Ray offline batch inference
- Maintainer: Community

## When to use this recipe

Use this recipe as a practical baseline for running Qwen3-Omni with the same
serving paths already exercised by repository examples and tests.

- Use **Instruct** for multimodal chat with optional speech synthesis.
- Use **Thinking** for reasoning-focused workloads that only need text output.

## References

- User guide:
  [`docs/user_guide/examples/online_serving/qwen3_omni.md`](../../docs/user_guide/examples/online_serving/qwen3_omni.md)
- Example guide:
  [`examples/online_serving/qwen3_omni/README.md`](../../examples/online_serving/qwen3_omni/README.md)
- Thinking stage config (verified on 2× H100-80G, TP=2):
  [`vllm_omni/deploy/qwen3_omni_moe_thinking_stage_config.yaml`](../../vllm_omni/deploy/qwen3_omni_moe_thinking_stage_config.yaml)

## Environment

- OS: Linux
- Python: 3.10+
- vLLM / vLLM-Omni: use versions from your current checkout, >=0.18.0

> Online serving for multimodal chat

## 1. Instruct (`Qwen/Qwen3-Omni-30B-A3B-Instruct`)

### Start server (single command)

From repository root:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091
```

Notes:

- `--omni` is required.
- The default deploy config `vllm_omni/deploy/qwen3_omni_moe.yaml` is loaded
  automatically by model registry.
- `async_chunk` is enabled by default in this deploy config.
- Platform deltas under `platforms:` (NPU/ROCm/XPU) are merged automatically on
  matching runtimes.

For advanced customization, pass an overlay YAML:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
  --deploy-config /path/to/your_qwen3_omni_overrides.yaml
```

### Runtime tuning

Prefer CLI overrides for day-to-day tuning:

```bash
# Disable async chunking when using /v1/realtime
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
  --no-async-chunk

# Example per-stage tuning in unified launch
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
  --stage-overrides '{"1": {"gpu_memory_utilization": 0.5}}'

# Tune max_num_seqs per stage (single process launch)
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
  --stage-overrides '{
    "0": {"max_num_seqs": 8},
    "1": {"max_num_seqs": 4},
    "2": {"max_num_seqs": 4}
  }'
```

### Stage-based launch (one stage per process)

Use three terminals (one per stage). Start with the default commands below, then
add `--max-num-seqs` only if you need explicit per-stage concurrency control.

Default stage-based commands:

```bash
# Stage 0: Thinker + API server
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
  --port 8091 \
  --stage-id 0 \
  --omni-master-address 127.0.0.1 \
  --omni-master-port 26000 &

# Stage 1: Talker
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
  --stage-id 1 \
  --headless \
  --omni-master-address 127.0.0.1 \
  --omni-master-port 26000 &

# Stage 2: Code2Wav
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
  --stage-id 2 \
  --headless \
  --omni-master-address 127.0.0.1 \
  --omni-master-port 26000 &
```

Optional: explicit per-stage `max_num_seqs` tuning:

```bash
# Stage 0
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
  --port 8091 \
  --stage-id 0 \
  --max-num-seqs 8 \
  --omni-master-address 127.0.0.1 \
  --omni-master-port 26000 &

# Stage 1
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
  --stage-id 1 \
  --headless \
  --max-num-seqs 4 \
  --omni-master-address 127.0.0.1 \
  --omni-master-port 26000 &

# Stage 2
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
  --stage-id 2 \
  --headless \
  --max-num-seqs 4 \
  --omni-master-address 127.0.0.1 \
  --omni-master-port 26000 &
```

If you use custom deploy YAML, add `--deploy-config` to each stage command.

### Verification

After server startup, run a multimodal example client:

```bash
python examples/online_serving/openai_chat_completion_client_for_multimodal_generation.py \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --query-type use_image \
  --port 8091 \
  --host localhost
```

Quick API smoke test (text-only output):

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "messages": [{"role": "user", "content": "Describe vLLM in brief."}],
    "modalities": ["text"]
  }'
```

`{"modalities":["text","audio"]}` means the model returns both text and audio in
the same response. Use it when you want transcription/content text and TTS audio
together.

Quick API smoke test (text + audio output):

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "messages": [{"role": "user", "content": "Describe vLLM in brief."}],
    "modalities": ["text", "audio"]
  }'
```

Realtime WebSocket check (`/v1/realtime`) requires async chunk disabled:

```bash
python examples/online_serving/qwen3_omni/openai_realtime_client.py \
  --url ws://localhost:8091/v1/realtime \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --input-wav /path/to/input_16k_mono.wav \
  --output-wav realtime_output.wav
```

### Benchmark with `vllm bench`

After the server is up, you can run online serving benchmarks with
`vllm bench serve --omni`.

Text-focused random workload:

```bash
vllm bench serve \
  --omni \
  --host localhost \
  --port 8091 \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --endpoint /v1/chat/completions \
  --backend openai-chat-omni \
  --dataset-name random \
  --num-prompts 40 \
  --max-concurrency 4 \
  --random-input-len 2500 \
  --random-output-len 900 \
  --ignore-eos \
  --extra-body '{"modalities":["text"]}' \
  --percentile-metrics ttft,tpot,itl,e2el
```

If you want benchmark requests to return both text and audio, switch
`--extra-body` to:

```bash
--extra-body '{"modalities":["text","audio"]}'
```

Synthetic multimodal workload (`random-mm`):

```bash
vllm bench serve \
  --omni \
  --host localhost \
  --port 8091 \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --endpoint /v1/chat/completions \
  --backend openai-chat-omni \
  --dataset-name random-mm \
  --num-prompts 40 \
  --max-concurrency 4 \
  --random-input-len 2500 \
  --random-output-len 900 \
  --random-mm-base-items-per-request 2 \
  --random-mm-limit-mm-per-prompt '{"image":1,"video":1,"audio":1}' \
  --random-mm-bucket-config '{"(32, 32, 1)": 0.5, "(0, 1, 1)": 0.5}' \
  --ignore-eos \
  --extra-body '{"modalities":["text"]}' \
  --percentile-metrics ttft,tpot,itl,e2el
```

### Notes

- `/v1/realtime` is unsupported while `async_chunk` is enabled.
- The default deploy uses `SharedMemoryConnector`; this is for single-host
  stage wiring.

## 2. Thinking (`Qwen/Qwen3-Omni-30B-A3B-Thinking`)

Thinker-only (single stage). Text-only output — no talker or code2wav stages.

### Start server (single command)

From repository root:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Thinking --omni --port 8091 \
  --stage-configs-path vllm_omni/deploy/qwen3_omni_moe_thinking_stage_config.yaml \
  --stage-init-timeout 1200 --init-timeout 1800 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --max-num-seqs 1024 \
  --max-num-batched-tokens 32768
```

### Verification

After server startup, run a multimodal example client. Request text-only output:

```bash
python examples/online_serving/openai_chat_completion_client_for_multimodal_generation.py \
  --model Qwen/Qwen3-Omni-30B-A3B-Thinking \
  --query-type use_image \
  --port 8091 \
  --host localhost \
  --modalities text
```

Quick API smoke test (text-only output):

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Omni-30B-A3B-Thinking",
    "messages": [{"role": "user", "content": "Describe vLLM in brief."}],
    "modalities": ["text"]
  }'
```

Quick API smoke test (image + text prompt):

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Omni-30B-A3B-Thinking",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}},
        {"type": "text", "text": "What is in this image?"}
      ]
    }],
    "modalities": ["text"]
  }'
```

### Benchmark with `vllm bench`


Synthetic multimodal workload (`random-mm`):

```bash
vllm bench serve \
  --omni \
  --host localhost \
  --port 8091 \
  --model Qwen/Qwen3-Omni-30B-A3B-Thinking \
  --endpoint /v1/chat/completions \
  --backend openai-chat-omni \
  --dataset-name random-mm \
  --seed 5678 \
  --num-warmups 16 \
  --num-prompts 1024 \
  --max-concurrency 2048 \
  --random-input-len 1024 \
  --random-output-len 4096 \
  --random-range-ratio 0.0 \
  --random-mm-num-mm-items-range-ratio 0.0 \
  --random-mm-limit-mm-per-prompt '{"image":1,"video":1,"audio":1}' \
  --random-mm-bucket-config '{"(256, 256, 1)":0.34,"(256, 256, 2)":0.33,"(0, 5, 3)":0.33}' \
  --ignore-eos \
  --extra-body '{"modalities":["text"]}' \
  --percentile-metrics ttft,tpot,itl,e2el
```

> Ray batch inference

## 3. Thinking — single Ray batch run
```bash
python examples/batch_inference_ray.py \
  model=qwen3_omni_30b_a3b_thinking \
  dataset=random_mm \
  query_type=mixed \
  ignore_eos=true \
  num_prompts=1024 \
  input_len=4096 \
  output_len=1024 \
  batch_size=256 \
  n_repeats=1 \
  vllm.tensor_parallel_size=8 \
  vllm.gpu_memory_utilization=0.94 \
  vllm.max_model_len=32768 \
  vllm.max_num_seqs=1024 \
  vllm.max_num_batched_tokens=32768 \
  vllm.max_concurrent_batches=8 \
  vllm.kv_cache_dtype=fp8 \
  vllm.enable_expert_parallel=true \
  vllm.enable_async_scheduling=true
```
