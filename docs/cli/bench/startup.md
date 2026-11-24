# Startup Benchmark

`vllm bench startup --omni` times Omni pipeline construction for cold and warm starts, which helps benchmark startup optimizations. Usage is the same as `vllm bench startup`; the only Omni-specific flag is an optional `--deploy-config` override. An example is shown below.

```bash
vllm bench startup --omni --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --num-iters-cold 1 --num-iters-warm 3 \
  --output-json startup.json
```

Each iteration constructs the pipeline in a new process and times initialization. Cold iterations reuse vLLM's helpers to run against empty temporary cache directories; your existing caches are left untouched.

Note that Omni currently reports only total startup time, not the compilation metrics that vLLM reports. Contributions for more granular metrics are welcome.
