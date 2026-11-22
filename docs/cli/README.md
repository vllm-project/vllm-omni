# vLLM-Omni CLI Guide

The CLI for vLLM-Omni inherits from vllm with some additional arguments.

Environment variables are process configuration rather than CLI-only options.
See the [Environment Variables](../configuration/environment_variables.md)
reference for ownership, precedence, evaluation time, and per-stage `env`
scoping.

## serve

Starts the vLLM-Omni OpenAI Compatible API server.

Start with a model:

```bash
vllm serve Qwen/Qwen2.5-Omni-7B --omni
```

Specify the port:

```bash
vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8091
```

Load a custom deploy configuration with `--deploy-config`:

```bash
vllm serve Qwen/Qwen2.5-Omni-7B --omni --deploy-config /path/to/deploy_config.yaml
```

## bench

Run benchmark tests for online serving throughput.
Available Commands:

```bash
vllm bench serve --omni \
    --model Qwen/Qwen2.5-Omni-7B \
    --host server-host \
    --port server-port \
    --random-input-len 32 \
    --random-output-len 4  \
    --num-prompts  5
```

See [vllm bench serve](./bench/serve.md) for serving benchmark arguments and
dataset-specific examples, including OmniInteract native-duplex sessions.

## doctor

Reports the devices on this machine and the deployment variants whose recipe
documents a minimum device memory at or below what the device exposes. It reads
no weights and starts no server, so it is safe to run before downloading
anything:

```bash
vllm-omni doctor
```

```
vLLM-Omni doctor

Devices:
- NVIDIA GeForce RTX 5080 Laptop GPU (cuda, 15.9 GiB dedicated)
  11 documented variants across 11 models fit: `MiniMaxAI/MiniMax-H3-Image`,
  `OpenMOSS-Team/MOSS-TTS-Nano`, `OpenMOSS-Team/MOSS-TTS-Realtime`,
  `OpenMOSS-Team/MOSS-VoiceGenerator`, `Qwen/Qwen3-TTS-12Hz-0.6B-Base`,
  `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, `Qwen/Qwen3-TTS-12Hz-1.7B-Base`,
  `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`, and 3 more

A documented minimum is what a recipe measured or bounded, not a promise for
your workload. Anything below it is not yet measured, not unsupported.
Full index: docs/models/memory_tiers.md
```

A device whose memory is below every documented minimum is reported as
*not yet measured*, not unsupported; see
[Minimum memory tiers](../models/memory_tiers.md) for the index and
[tools/gen_memory_tiers.py](https://github.com/vllm-project/vllm-omni/blob/main/tools/gen_memory_tiers.py)
for how it is generated.

Pass `--json` for the detected devices and their fitting variants in a
machine-readable form.

On Apple Silicon this reports unified memory rather than a dedicated GPU pool,
so treat the tier it prints as an upper bound: the CPU, the OS, and the
tokenizer all draw from the same pool.
