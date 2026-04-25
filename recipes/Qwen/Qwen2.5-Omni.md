# Qwen2.5-Omni for text and speech chat on 1x A100 80GB

## Summary

- Vendor: Qwen
- Model: `Qwen/Qwen2.5-Omni-7B`
- Task: Multimodal chat with text and audio output
- Mode: Online serving with the OpenAI-compatible API
- Maintainer: Community

## When to use this recipe

Use this recipe to serve `Qwen/Qwen2.5-Omni-7B` on one 80 GB A100 by placing
the thinker, talker, and code2wav stages on the same GPU.

## References

- Upstream or canonical docs:
  [`docs/user_guide/examples/online_serving/qwen2_5_omni.md`](../../docs/user_guide/examples/online_serving/qwen2_5_omni.md)
- Related example under `examples/`:
  [`examples/online_serving/qwen2_5_omni/`](../../examples/online_serving/qwen2_5_omni/)
- Related issue or discussion:
  [RFC: add recipes folder](https://github.com/vllm-project/vllm-omni/issues/2645)

## Hardware Support

This recipe documents one validated CUDA GPU configuration.

## GPU

### 1x A100 80GB

#### Environment

- Python: 3.10.12
- GPU: `NVIDIA A100-SXM4-80GB`, 81920 MiB reported in `nvidia-smi` samples
- Runtime: CUDA `cu130` wheel stack
- vLLM version: `0.19.0+cu130`
- vLLM-Omni version or commit: `0.1.dev1415+g6b52db9e2`
- Related package versions: `torch==2.10.0+cu130`,
  `transformers==4.57.6`, `accelerate==1.12.0`, `soundfile==0.13.1`

#### Command

Start the server from the repository root. The bundled deploy config places
stage 1 on device `1`, so this recipe explicitly places all three stages on
device `0` and assigns per-stage memory budgets.

```bash
vllm-omni serve Qwen/Qwen2.5-Omni-7B \
  --omni \
  --host 127.0.0.1 \
  --port 8091 \
  --stage-overrides '{"0":{"devices":"0","gpu_memory_utilization":0.45},"1":{"devices":"0","gpu_memory_utilization":0.30},"2":{"devices":"0","gpu_memory_utilization":0.15}}'
```

If your installation keeps the vLLM-Omni entrypoint installed as `vllm`, the
same arguments can be used with `vllm serve`.

#### Verification

Check server health after startup:

```bash
curl -fsS http://localhost:8091/health
```

Validate a text-only response:

```bash
curl -sS http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Omni-7B",
    "messages": [{"role": "user", "content": "Describe vLLM in brief."}],
    "modalities": ["text"],
    "max_tokens": 64
  }' > text_response.json

python - <<'PY'
import json

with open("text_response.json", encoding="utf-8") as f:
    data = json.load(f)
content = data["choices"][0]["message"]["content"]
assert content
print(content)
PY
```

Validate an audio-output response:

```bash
curl -sS http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Omni-7B",
    "messages": [{"role": "user", "content": "Say one short sentence about vLLM."}],
    "modalities": ["audio"],
    "max_tokens": 32
  }' > audio_response.json

python - <<'PY'
import json

with open("audio_response.json", encoding="utf-8") as f:
    data = json.load(f)
audio = data["choices"][0]["message"]["audio"]
assert audio
print("audio response present")
PY
```

#### Notes

- Memory usage: In the validated run, `nvidia-smi` samples peaked at
  `63236 MiB` while serving all three stages on one A100.
- Validation result: The text and audio validation requests both completed
  successfully and parsed as expected.
- Key flags: `--omni` enables vLLM-Omni serving; `--stage-overrides` is
  required here to place stages 0, 1, and 2 on the single visible GPU.
- Known limitations: This validation covered text input with text output and
  text input with audio output. It did not validate image or video input on
  this 1x A100 configuration.
