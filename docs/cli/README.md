# vLLM-Omni CLI Guide

The CLI for vLLM-Omni inherits from vllm with some additional arguments.

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

For a migrated model, load a custom deploy configuration with `--deploy-config`:

```bash
vllm serve Qwen/Qwen2.5-Omni-7B --omni --deploy-config /path/to/deploy_config.yaml
```

The deprecated `--stage-configs-path` flag is retained for models that still use the legacy `stage_args` schema:

```bash
vllm serve ByteDance-Seed/BAGEL-7B-MoT --omni --stage-configs-path /path/to/legacy_stage_config.yaml
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

See [vllm bench serve](./bench/serve.md) for the full reference of all available arguments.
