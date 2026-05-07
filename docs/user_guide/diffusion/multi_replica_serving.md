# Diffusion Multi-Replica Serving

Diffusion stages can be replicated in online serving by setting `runtime.num_replicas` on a `stage_type: diffusion` stage. Each replica owns an independent diffusion engine process and receives requests through round-robin routing with request affinity.

## When to Use

Use multi-replica serving when one diffusion process does not saturate the available GPUs, or when you want higher request throughput for many independent image/video generation requests. It improves concurrency throughput; it does not make a single request faster.

## Configuration

A replica consumes the devices required by that diffusion stage. For single-device diffusion pipelines, set `runtime.devices` to one device per replica:

```yaml
stages:
  - stage_id: 0
    stage_type: diffusion
    runtime:
      devices: "0,1,2,3"
      num_replicas: 4
    engine_args:
      model: Qwen/Qwen-Image
      parallel_config:
        world_size: 1
```

For multi-device diffusion pipelines, `parallel_config.world_size` is treated as the number of devices per replica. For example, four 2-GPU replicas need eight devices:

```yaml
stages:
  - stage_id: 0
    stage_type: diffusion
    runtime:
      devices: "0,1,2,3,4,5,6,7"
      num_replicas: 4
    engine_args:
      model: Wan-AI/Wan2.2-T2V-A14B-Diffusers
      parallel_config:
        world_size: 2
```

The runtime splits the device list into contiguous chunks per replica. With the second example, replica 0 uses `0,1`, replica 1 uses `2,3`, replica 2 uses `4,5`, and replica 3 uses `6,7`.

## Serving

Start the server with the stage config:

```bash
vllm serve Qwen/Qwen-Image \
  --omni \
  --port 8099 \
  --stage-configs-path /path/to/diffusion_multi_replica.yaml
```

Then send normal OpenAI-compatible diffusion chat requests. The orchestrator routes new requests across replicas and keeps updates/abort calls on the same replica for each request.

## Benchmarking

Benchmark each replica count by changing `runtime.num_replicas` and `runtime.devices`, restarting the server, then running:

```bash
python benchmarks/diffusion/diffusion_benchmark_serving.py \
  --base-url http://localhost:8099 \
  --model Qwen/Qwen-Image \
  --task t2i \
  --dataset random \
  --num-prompts 8 \
  --request-rate inf \
  --num-inference-steps 2 \
  --width 256 --height 256 \
  --output-file diffusion_replica_4.json
```

For a quick scaling sweep, test `num_replicas` values `1`, `2`, `3`, and `4` with the same prompt count and generation parameters. Compare throughput and end-to-end latency from the JSON outputs.

## Notes and Limitations

- Multi-replica serving is currently supported for diffusion stages. Single-stage mode still rejects non-diffusion replicas.
- `runtime.devices` must provide exactly `num_replicas * devices_per_replica` devices.
- Request-level latency may stay similar or slightly increase because each request still runs on one replica.
- Throughput improves only when enough concurrent requests are issued and each replica has enough GPU memory.
