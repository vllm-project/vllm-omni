# LingBot realtime Replica-DP validation

This recipe validates session-affine routing with two independent LingBot
AR-Diffusion replicas. Each replica uses TP=2, for four GPUs total:

```text
GPU 0,1 -> replica 0, TP=2
GPU 2,3 -> replica 1, TP=2
```

It is a hardware-validation recipe for PR #6233. The checked-in CPU tests
validate the routing state machine; this run validates real worker-local AR
state, concurrent execution, and the four-GPU process layout.

## Run

Use a single node with four mutually visible GPUs. Record the code revision,
topology, driver, and neighboring processes before loading the checkpoint:

```bash
git rev-parse HEAD
git status --short
nvidia-smi
nvidia-smi topo -m
```

Then run two sessions concurrently:

```bash
python examples/offline_inference/diffusion/lingbot_world_v2_realtime.py \
  --model robbyant/lingbot-world-v2-14b-causal-fast-diffusers \
  --deploy-config examples/offline_inference/diffusion/lingbot_world_v2_replica_dp_tp2.yaml \
  --image /absolute/path/to/input.png \
  --prompt "A vehicle travels through an outdoor scene" \
  --events examples/offline_inference/diffusion/lingbot_world_v2_realtime_events.jsonl \
  --output-dir runs/lingbot_replica_dp_tp2 \
  --session-id world \
  --num-sessions 2 \
  --require-distinct-replicas
```

The driver submits both first ticks with one `asyncio.gather()`, then advances
both sessions concurrently for every later chunk. It fails if:

- a completed tick has no session route;
- a session moves to another replica;
- `--require-distinct-replicas` is set and the two sessions share an owner;
- metadata validation, model execution, or finite-output checks fail.

Output is grouped by session:

```text
runs/lingbot_replica_dp_tp2/
  summary.json
  world-0/chunk_000.{pt,json}
  world-0/chunk_001.{pt,json}
  world-1/chunk_000.{pt,json}
  world-1/chunk_001.{pt,json}
```

`summary.json` records the replica owner of every session, per-round wall time,
aggregate chunks/s, aggregate latent-frames/s, and every chunk's latency,
latent shape, finite flag, and correlated AR metadata. Preserve the raw
stdout/stderr and `nvidia-smi dmon` output with this directory.

## Pass criteria

1. `world-0` and `world-1` have distinct `route_owners`.
2. Each session reports the same `replica_id` for all chunks.
3. Every latent is finite and every metadata session/chunk id matches.
4. Both replica GPU groups are active during the same wall-clock interval.
5. Closing the sessions completes without a leaked route warning.

This run proves routing and concurrent execution. It does not by itself prove
video quality or realtime RGB delivery: the driver writes latent chunks, and
the current AR path still needs the separate streaming-VAE validation for
end-to-end frame delivery.

## Recommended comparison

Run these with the same checkpoint, image, events, seed, and compile mode:

```text
1 replica x TP=2, one session       single-session latency baseline
1 replica x TP=2, two sessions      serialized-session throughput baseline
2 replicas x TP=2, two sessions     session-affine Replica-DP result
```

Report `tick_wall_seconds`, per-session chunk latency,
`aggregate_latent_frames_per_second`, per-GPU utilization, and peak HBM. The
reported latent-frame rate is a generation throughput metric, not delivered RGB
FPS or delivery RTF. Do not compare the two-replica result only against TP=4:
TP changes single-session model parallelism, while replicas change cross-session
throughput.
