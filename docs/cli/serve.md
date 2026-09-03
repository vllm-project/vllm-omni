# vllm-omni serve

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

## Standalone stage mode

`--standalone` boots a single stage as an independent HTTP server with no orchestrator
dependency. Unlike `--headless` (which registers with a ZMQ master and receives work
from the orchestrator), standalone stages expose their own HTTP endpoint and can be
deployed, scaled, and managed independently by external infrastructure.

```bash
# Stage 0 (talker) on GPU 0
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --omni \
    --standalone --stage-id 0 --port 8000 --trust-remote-code

# Stage 1 (code2wav) on GPU 1
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --omni \
    --standalone --stage-id 1 --port 8001 --trust-remote-code
```

Standalone stages communicate via `/v1/stage/run`. Entry mode (no `stage_output` in
body) runs the model and returns raw multimodal output. Downstream mode (has
`stage_output` in body) accepts upstream output and returns the final result. An
external system chains stages by calling entry mode on the first stage and forwarding
the response to downstream mode on the next. See
`examples/online_serving/text_to_speech/qwen3_tts/standalone_disagg_client.py` for a
reference coordinator.

Async-chunk streaming is not supported in standalone mode. `--standalone` is mutually
exclusive with `--headless`.

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
