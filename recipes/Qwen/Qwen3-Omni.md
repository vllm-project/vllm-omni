# Qwen3-Omni

## Summary

- Vendor: Qwen
- Model: `Qwen/Qwen3-Omni-30B-A3B-Instruct`
- Task: Multimodal chat with text/image/audio/video input
- Mode: Online serving with the OpenAI-compatible API
- Maintainer: Community

## When to use this recipe

Use this recipe as a practical baseline for running
`Qwen/Qwen3-Omni-30B-A3B-Instruct` with the same serving paths already
exercised by repository examples and tests.

## References

- User guide:
  [`docs/user_guide/examples/online_serving/qwen3_omni.md`](../../docs/user_guide/examples/online_serving/qwen3_omni.md)
- Example guide:
  [`examples/online_serving/qwen3_omni/README.md`](../../examples/online_serving/qwen3_omni/README.md)

## Environment

- OS: Linux
- Python: 3.10+
- vLLM / vLLM-Omni: use versions from your current checkout, >=0.18.0

## Start server (single command)

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

#### Runtime tuning

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

#### Stage-based launch (one stage per process)

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

#### Verification

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

#### Benchmark with `vllm bench`

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

#### Notes

- `/v1/realtime` is unsupported while `async_chunk` is enabled.
- The default deploy uses `SharedMemoryConnector`; this is for single-host
  stage wiring.
