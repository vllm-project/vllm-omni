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

Reports the devices on this machine and how each documented deployment variant
relates to them. It reads no weights and starts no server, so it is safe to run
before downloading anything:

```bash
vllm-omni doctor
```

```text
vLLM-Omni doctor

Devices:
- NVIDIA GeForce RTX 4090 (cuda, 24.0 GiB dedicated)
  no variant is documented for this device at or below 24 GiB
  19 more variants fit on capacity alone, compatibility unknown (recipe
  declares no backend): `MiniMaxAI/MiniMax-H3-Image`,
  `OpenMOSS-Team/MOSS-TTS-Nano`, and 17 more
  3 variants fit on capacity but need specific hardware:
  `black-forest-labs/flux-3-image-klein` (nvfp4: b200, gb200, b300, gb300),
  `black-forest-labs/flux-3-image-klein` (fp8r: h100, h200, b200, gb200,
  b300, gb300), `black-forest-labs/flux-3-image-klein` (default: h100, h200,
  b200, gb200, b300, gb300, rtx_pro_6000)

A documented minimum is what a recipe measured or bounded, not a promise for
your workload. Anything below it is not yet measured, not unsupported.
A fit also requires the recipe to declare your backend on dedicated memory and,
where it lists hardware, to include your card; capacity alone is reported apart.
Full index: docs/models/memory_tiers.md
```

Memory is only the first gate. A variant is reported as a fit only when all of
the following hold, and anything that fails a gate is reported with its reason
instead of being counted:

- The device has a dedicated memory pool. System memory (CPU) and unified
  memory (Apple Silicon) are not device memory, so a `vram_minimum_gb` bound
  does not describe them.
- The recipe declares the device's backend (`cuda`, `rocm`, ...) through its
  `supported_hardware` entries. Recipes that declare no platform stay
  *capacity only*; the index never assumes CUDA.
- Where a recipe lists specific cards, the device must be one of them. A card
  that cannot be resolved to a documented slug is not assumed to match.

A device whose memory is below every documented minimum is reported as
*not yet measured*, not unsupported; see
[Minimum memory tiers](../models/memory_tiers.md) for the index and
[tools/gen_memory_tiers.py](https://github.com/vllm-project/vllm-omni/blob/main/tools/gen_memory_tiers.py)
for how it is generated.

Pass `--json` for the same report in a machine-readable form. Every variant
that fits on capacity appears with a `status` of `compatible`,
`capacity_only`, or `needs_hardware`, plus a `reason` code and the declared
backends and hardware slugs, so a caller never has to re-derive the gates.

On Apple Silicon this reports unified memory rather than a dedicated GPU pool,
so treat the tier it prints as an upper bound: the CPU, the OS, and the
tokenizer all draw from the same pool.
