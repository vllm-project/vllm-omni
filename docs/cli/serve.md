# vllm-omni serve

## Multiple API frontends

For local multi-stage EngineCore deployments, `--api-server-count` starts
multiple API frontend processes that share one parent-owned set of stage
engines:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
    --api-server-count 2 \
    --port 8091
```

This mode is intended for workloads whose request parsing, multimodal input
processing, or response encoding can bottleneck a single frontend. It
currently requires local multi-process EngineCore stages with one local
process group per replica. A stage may still use multiple `num_replicas`.
It cannot be combined with diffusion, headless or
remote stages, intra-stage data parallelism, Ray, fault tolerance, elastic
expert parallelism, sleep mode, or runtime LoRA updating. Runtime voice upload
and deletion are also disabled because those mutations are process-local;
built-in voices, inline reference audio, and voices restored at startup remain
available. The `/v1/omni/sleep` and `/v1/omni/wakeup` control routes return HTTP
409 for the same reason: their bookkeeping is process-local while stage engines
are shared.

## Stage-based CLI quickstart

The stage-based CLI is designed for deployments that require launching each pipeline stage in an isolated process
(e.g., across separate operating system processes, distinct GPUs, or distributed hosts).

- For **migrated models** that utilize the bundled deployment YAML configurations located in
  `vllm_omni/deploy/`, the `--deploy-config` flag is only required to override the default configuration. By default, executing `vllm serve MODEL --omni ...`
  automatically loads the bundled deployment configuration.

Example: Initializing Stage 0 (Orchestrator and API Server):
The commands below show a common device mapping where Stage 0 uses GPU 0 and
worker stages use GPU 1 via `CUDA_VISIBLE_DEVICES`.

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
    --port 8091 \
    --stage-id 0 \
    --omni-master-address 127.0.0.1 \
    --omni-master-port 26000
```

Example: Initializing a Headless Worker Stage (Stage 1):

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
    --stage-id 1 \
    --headless \
    --omni-master-address 127.0.0.1 \
    --omni-master-port 26000
```

When utilizing a custom deployment YAML, append `--deploy-config /path/to/override.yaml` to each command execution.

In the standard execution paradigm, the `--stage-overrides` argument is utilized to apply stage-specific configurations from a single CLI command.
However, under the **stage-based CLI** paradigm, where each process strictly encapsulates a single stage, it is recommended to specify tuning parameters directly via discrete command-line flags for the respective stage, rather than constructing a composite `--stage-overrides` JSON string.

For example, as an alternative to the following composite configuration:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
    --stage-overrides '{"1": {"gpu_memory_utilization": 0.5}}'
```

the stage-based CLI permits the direct initialization of Stage 1 with explicit parameters:

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
    --stage-id 1 \
    --headless \
    --gpu-memory-utilization 0.5 \
    --omni-master-address 127.0.0.1 \
    --omni-master-port 26000
```

## JSON CLI Arguments

--8<-- "docs/cli/json_tip.inc.md"

## Arguments

--8<-- "docs/generated/argparse_omni/omni_serve.inc.md"
