# Replica Data Parallelism for Image and Video Generation

A runnable recipe + benchmark for **replica data parallelism** (replica-DP) on a
video DiT. Replica-DP runs `N` independent diffusion engine replicas, one per GPU,
and routes each request to a single replica. It scales **request throughput**
near-linearly; it does **not** change single-request latency (for that, use an
intra-request axis such as tensor/sequence/CFG parallelism).

This example uses `Wan-AI/Wan2.2-TI2V-5B-Diffusers` with one GPU per replica.
The MiniMax-H3-Image configuration below uses the same replica runtime for
text-to-image serving.

## Files

| File | Purpose |
| ------ | --------- |
| `wan2_2_ti2v_dp.yaml` | Deploy config; `num_replicas` + `devices` drive the fan-out. |
| `minimax_h3_image_dp.yaml` | MiniMax-H3-Image profile: SP1, encoder TP1 and one full pipeline per GPU. |
| `run_server.sh` | Substitutes `NUM_REPLICAS` / `DEVICES` into the config and serves. |
| `bench_replica_dp.py` | Replica-agnostic load driver; reports throughput, latency, and a per-input isolation check. |

The benchmark driver in this directory sends video requests. For images, use
the image benchmark command in the [AMD recipe](../../../recipes/MiniMaxAI/MiniMax-H3-Image-AMD.md).

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

### MiniMax-H3-Image: four SP1 replicas

Prepare the image checkpoint and its external text encoder using the
[AMD recipe](../../../recipes/MiniMaxAI/MiniMax-H3-Image-AMD.md).
Each selected GPU must have enough memory for the complete model, text encoder
and VAE. From this directory, in the prepared runtime:

```bash
MODEL=/data/hf_hub_cache/MiniMax-H3-Image \
NUM_REPLICAS=4 DEVICES=0,1,2,3 PORT=8001 \
DEPLOY_TEMPLATE="$PWD/minimax_h3_image_dp.yaml" \
    ./run_server.sh
```

`DEVICES` uses the launcher's visible device numbering. For example, with both
`HIP_VISIBLE_DEVICES=4,5,6,7` and `CUDA_VISIBLE_DEVICES=4,5,6,7`, use
`DEVICES=0,1,2,3`. Do not combine conflicting visibility masks.
Stop any existing server on the selected GPUs and port before launching.

The template keeps every intra-request parallel degree at one, including
text-encoder TP, and `max_num_seqs=1` per replica. `num_replicas` distributes
requests across four engines; it does not shard one image across four GPUs.
Do not pass `--num-gpus 4`: that specifies GPUs per engine rather than the
number of independent replicas. Additional CLI arguments are forwarded
unchanged by `run_server.sh`.

This profile disables VAE graph decode and encoder cuDNN SDP to retain the
historical experiment's settings. It uses the existing runtime's replica
routing and lifecycle; it does not install another HTTP proxy or a tuned GEMM
table. Historical SP4-versus-four-SP1 measurements and their single-request
latency tradeoff are recorded in the AMD recipe. The new native-runtime
configuration has CPU configuration coverage; those historical numbers are
not a benchmark of this configuration at the current revision.

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
| ---------- | ------------------------- | --------- | ------------ |
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
