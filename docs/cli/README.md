# vLLM-Omni CLI Guide

The CLI for vLLM-Omni inherits from vllm with some additional arguments.

Environment variables are process configuration rather than CLI-only options.
See the [Environment Variables](../configuration/environment_variables.md)
reference for ownership, precedence, evaluation time, and per-stage `env`
scoping.

## doctor

Reports the current machine's acceleration tier and the documented recipe
minimums. It is read-only: it does not load a model, initialize a connector or
allocate GPU memory.

```bash
vllm-omni doctor
```

The report covers the OS, backend (CUDA / MPS / CPU), device and memory. It also
lists the documented recipe minimums and marks a tier below every documented
minimum as `not yet measured`, not `unsupported`.

Use `--json` for a machine-readable report:

```bash
vllm-omni doctor --json
```

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
