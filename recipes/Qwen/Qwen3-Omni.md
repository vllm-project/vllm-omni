# Qwen3-Omni for multimodal chat on 1x A100 80GB

## Summary

- Vendor: Qwen
- Model: `Qwen/Qwen3-Omni-30B-A3B-Instruct`
- Task: Multimodal chat with text, image, audio, or video input
- Mode: Online serving with the OpenAI-compatible API
- Maintainer: Community

## When to use this recipe

Use this recipe when you want a known-good starting point for serving
`Qwen/Qwen3-Omni-30B-A3B-Instruct` with vLLM-Omni on a single 80 GB A100 and
validate the deployment with the existing multimodal client examples in this
repository.

## References

- Upstream or canonical docs:
  [`docs/user_guide/examples/online_serving/qwen3_omni.md`](../../docs/user_guide/examples/online_serving/qwen3_omni.md)
- Related example under `examples/`:
  [`examples/online_serving/qwen3_omni/README.md`](../../examples/online_serving/qwen3_omni/README.md)
- Related issue or discussion:
  [RFC: add recipes folder](https://github.com/vllm-project/vllm-omni/issues/2645)

## Hardware Support

This recipe currently documents one tested-style reference configuration for
CUDA GPU serving. Add more sections for other hardware as community validation
lands.

## GPU

### 1x A100 80GB

#### Environment

- OS: Linux
- Python: 3.10+
- Driver / runtime: NVIDIA CUDA environment with an A100 80 GB GPU
- vLLM version: Match the repository requirements for your checkout
- vLLM-Omni version or commit: Use the commit you are deploying from

#### Command

Start the server from the repository root:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091
```

To enable async chunking, use the bundled stage config:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --omni \
  --port 8091 \
  --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_omni_moe_async_chunk.yaml
```

#### Verification

Run one of the existing example clients after the server is ready:

```bash
python examples/online_serving/openai_chat_completion_client_for_multimodal_generation.py \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --query-type use_image \
  --port 8091 \
  --host localhost
```

For a quick API smoke test, request text-only output:

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "messages": [{"role": "user", "content": "Describe vLLM in brief."}],
    "modalities": ["text"]
  }'
```

#### Notes

- Memory usage: Size depends on runtime options and output modalities; leave headroom for multimodal workloads.
- Key flags: `--omni` is required; `--stage-configs-path` is optional for custom or async-chunk stage configs.
- Known limitations: This starter recipe is intentionally narrow and focuses on the single-GPU online-serving path already documented in the repo examples.
