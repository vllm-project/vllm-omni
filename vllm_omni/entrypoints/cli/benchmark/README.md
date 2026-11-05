# Benchmark CLI

This package owns Omni benchmark subcommands exposed through the CLI.

## What Belongs Here

- CLI parser extensions for benchmark commands.
- Benchmark command dispatch.
- Benchmark-only argument groups.

## What Does Not Belong Here

- Runtime server route helpers.
- OpenAI endpoint implementation details.
- Benchmark execution internals that belong under `vllm_omni.benchmarks`.

This package is not part of the OpenAI endpoint helper migration. It may need
import updates only if CLI/bootstrap ownership changes during #5227.

## Realtime video

`vllm bench realtime-video --omni` measures the `/v1/realtime/video` WebSocket
protocol using the packaged LingBot-World workload and metrics. For example:

```bash
vllm bench realtime-video --omni --host 127.0.0.1 --port 8000 \
  --num-chunks 40 --warmup-sessions 1 --sessions 3 --target-fps 12 \
  --output-json /tmp/lingbot-world.json
```

Start the server separately. The command dispatches to
`vllm_omni.benchmarks.lingbot_world.realtime`; argument validation, warmup,
measurement, failed-session evidence and result output share one implementation
with the source-checkout compatibility script. See the
[LingBot benchmark guide](../../../../benchmarks/lingbot_world/README.md) for
server configurations, workload files and the meaning of each metric.
