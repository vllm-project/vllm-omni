# Replica Data Parallelism for Video Generation

A runnable recipe + benchmark for **replica data parallelism** (replica-DP) on a
video DiT. Replica-DP runs `N` independent diffusion engine replicas, one per GPU,
and routes each request to a single replica. It scales **request throughput**
near-linearly; it does **not** change single-request latency (for that, use an
intra-request axis such as tensor/sequence/CFG parallelism).

This example uses `Wan-AI/Wan2.2-TI2V-5B-Diffusers` with one GPU per replica.

## Files

| File | Purpose |
|------|---------|
| `wan2_2_ti2v_dp.yaml` | Deploy config; `num_replicas` + `devices` drive the fan-out. |
| `run_server.sh` | Substitutes `NUM_REPLICAS` / `DEVICES` into the config and serves. |
| `bench_replica_dp.py` | Replica-agnostic load driver; reports throughput, latency, and a per-input isolation check. |

## Run

Start the server with the replica count you want (one GPU per replica):

```bash
# baseline
NUM_REPLICAS=1 DEVICES=0        ./run_server.sh
# 2 replicas
NUM_REPLICAS=2 DEVICES=0,1      ./run_server.sh
# 4 replicas
NUM_REPLICAS=4 DEVICES=0,1,2,3  ./run_server.sh
```

In another shell, drive a fixed workload and read the throughput:

```bash
python bench_replica_dp.py --url http://127.0.0.1:8098 \
    --num-requests 28 --concurrency 8 --label "replicas=4"
```

Re-run the server at `NUM_REPLICAS=1,2,4` (same client workload each time) and
compare `videos/min`. Pass `--baseline-tpm <N=1 value>` to print a scaling factor.

### Isolation check

Identical outputs under a *shared* prompt+seed prove nothing -- every request
would be identical regardless of isolation. Instead each request gets a distinct
input (a prompt from the pool + a unique seed `base_seed+i`), and we check that a
request's output is unchanged whether it runs alone or concurrently amid other
distinct requests. Capture a per-input baseline on `N=1`, then verify it under
`N>=2` concurrency:

```bash
# 1) against NUM_REPLICAS=1 (serial), record each input's output hash
python bench_replica_dp.py --num-requests 8 --concurrency 1 \
    --write-baseline baseline.json

# 2) against NUM_REPLICAS>=2, replay the SAME inputs concurrently and verify
python bench_replica_dp.py --num-requests 8 --concurrency 8 \
    --check-baseline baseline.json
```

Step 2 exits non-zero if any request's output differs from its own baseline (a
mismatch means cross-request state bleed or misrouting). Keep `--num-requests`,
`--base-seed`, and the prompt source identical between the two runs so the input
sets match.

## Measured scaling

Wan2.2-TI2V-5B (832×480, 33 frames, 30 steps), 4× A800-80GB (NVLink), one GPU per replica:

| Replicas | Throughput (videos/min) | Scaling | Efficiency |
|----------|-------------------------|---------|------------|
| 1 | 4.71 | 1.00× | — |
| 2 | 9.20 | 1.95× | 98% |
| 4 | 18.02 | 3.83× | 96% |

Per-request latency stayed flat (~13–14 s) across all replica counts. Isolation
is checked per-input via the baseline/verify protocol above: each request's
output under `N>=2` concurrency must match its own single-replica baseline.

## Notes on the config surface

- `runtime.num_replicas: N` fans out `N` replicas; `runtime.devices` assigns their
  GPUs. With `tensor_parallel_size = 1`, list one GPU per replica
  (`num_replicas * tensor_parallel_size` entries, pool mode).
- Replica fan-out here comes from the deploy config, not a CLI flag. (The headless /
  multi-runtime launch path uses the process-local `--omni-dp-size-local`, which
  requires `--stage-id`.)
- This recipe serves via `--deploy-config`. Replica fan-out comes from the deploy
  config's `num_replicas` / `devices`; stage topology comes from the registered
  `WanPipeline`.
