# π0.5 Benchmarks

Latency benchmarks for π0.5 (`lerobot/pi05_base`) serving, used for the measurements in [`recipes/lerobot/Pi05.md`](../../recipes/lerobot/Pi05.md).

| Script | Measures | Needs |
| --- | --- | --- |
| `e2e_latency_pi05.py` | Client E2E round trip over the OpenPI websocket: msgpack pack, send, engine, denoise, unpack. Reports p50/p95/min/max and the device memory peak sampled from `nvidia-smi`. | A running server |
| `bench_pi05.py` | Model-level `sample_actions` latency and output shape over a batch size × camera-view sweep, with no serving stack. Writes `pi05_sweep.csv` and `pi05_sweep.md`. | A GPU and the checkpoint |

## Client E2E latency

Start a server with the recipe's command, then run the client per camera count.
It serves the optimized path in the deploy config's `float32`; pass `--enforce-eager` to the server for the eager baseline and `--dtype bfloat16` for bfloat16.

```bash
# optimized path (CUDA graphs + fused kernels)
vllm serve lerobot/pi05_base --omni --port 8000 --deploy-config vllm_omni/deploy/pi05.yaml --disable-log-stats
# eager baseline
vllm serve lerobot/pi05_base --omni --port 8000 --deploy-config vllm_omni/deploy/pi05.yaml --enforce-eager --disable-log-stats

python benchmarks/pi05/e2e_latency_pi05.py --host 127.0.0.1 --port 8000 --views 3 --warmup 3 --iters 30 --out e2e.json
```

## Model-level latency

The script builds the model through `Pi05Pipeline`, as the server does, from the deploy config's `model_config`.
It runs the optimized path by default; pass `--enforce-eager` for the eager baseline and `--dtype bfloat16` for bfloat16.

```bash
# optimized path (CUDA graphs + fused kernels)
python benchmarks/pi05/bench_pi05.py --bs 1 --views 3 --warmup 5 --iters 30 --out pi05_bench/
# eager baseline
python benchmarks/pi05/bench_pi05.py --enforce-eager --bs 1 --views 3 --warmup 5 --iters 30 --out pi05_bench_eager/
```

## Experimental P0 options

The model benchmark supports independent `--vision-precision` (`fp32`,
`bf16_autocast`, `selective_bf16`) and `--batch-cameras` / `--no-batch-cameras`
options. The corresponding stage `custom_pipeline_args` keys are
`pi05_vision_precision` and `pi05_batch_cameras`; CLI values override them.
Defaults preserve R0's FP32 vision/projector and sequential cameras. Both BF16
experiments restore FP32 image features before the prefix LM, preserving the
default legacy residual policy. Selective BF16 retains FP32 patch/position embedding
weights and casts their combined output to BF16 before the vision encoder; the
remaining vision tower, including its encoder and final LayerNorm parameters,
and the projector use BF16 weights. Autocast instead retains all FP32
vision/projector weights and uses BF16 autocast for their forward pass. Neither
option changes the LM/expert norm weights or their residual precision policy.
Changing either precision or camera batching can change rounding; action parity
and task quality need to be evaluated separately from latency.
The 1/2-camera cases retain three image slots with false masks for missing views.

`--lm-precision legacy|lerobot_bf16` maps to `custom_pipeline_args.pi05_lm_precision`
and defaults to `legacy`.
`lerobot_bf16` applies LeRobot's BF16 decoder-entry and residual policy to both
prefix and action expert when their projection weights are BF16, keeping vision
and the Euler schedule unchanged.
This option covers LM activation precision; full pipeline parity requires separate validation.
Norm and softmax fusion are audited for Torch 2.13.0; softmax requires BF16
`[1,8,50,1018]` expert scores or `[1,8,968,968]` prefix scores (CUDA 13.2). Unsupported cases use PyTorch.
The validated 50-token expert also uses fused QKV GEMMs on CUDA 13.2 / SM90.
Its exact BF16 QK path supports CUDA 13.2 / SM90 and SM120 (the latter requires Triton 3.7.1 Gluon); other cases retain cuBLAS.

Three independent runtime flags are also available, all disabled by default:
`--precompute-adarms` prepares timestep embeddings, the time MLP, and every
AdaRMS modulation projection for batch size 1 and the configured default
denoising schedule (10 steps in the supplied deploy config), after checkpoint
load and before capture. Other step counts or batch sizes use live projections;
normal writes to tracked conditioning weights invalidate the table. Such
requests bypass the captured denoise step; compatible prefix graphs can still
replay. `--hoist-denoise-invariants` computes denoise masks, position IDs, and
RoPE once per request and shares one prefix RoPE table across its layers;
`--prefix-kv-only` omits the final prefix layer's Q/attention/MLP and final norm
when action generation only consumes KV. Their stage
`custom_pipeline_args` keys are `pi05_precompute_adarms`,
`pi05_hoist_denoise_invariants`, and `pi05_prefix_kv_only`. Each flag also has a
`--no-...` form for ablation. General prefix forward keeps its full hidden output.

These options are fixed during pipeline initialization, before graph capture;
use a new pipeline to change them. Request masks, positions, and prefix KV are
rebuilt for each observation, not retained as history across requests.

`--execution-mode eager|fused|graph` selects the implementation independently from
these numerical/algorithm settings. `--enforce-eager` remains an alias for eager;
combining it with fused/graph is an error. Default execution remains graph.

```bash
python benchmarks/pi05/bench_pi05.py --dtype bfloat16 --bs 1 --views 1 2 3 \
  --vision-precision selective_bf16 --batch-cameras --warmup 20 --iters 30 \
  --out pi05_p0_selective_batched/

# Full P0 candidate; use --no-<flag> for individual ablations, and compare the
# same options under --execution-mode eager / fused / graph.
python benchmarks/pi05/bench_pi05.py --dtype bfloat16 --bs 1 --views 1 2 3 \
  --vision-precision selective_bf16 --lm-precision lerobot_bf16 \
  --batch-cameras --precompute-adarms \
  --hoist-denoise-invariants --prefix-kv-only --execution-mode graph \
  --warmup 20 --iters 30 --out pi05_p0_all_on/
```

These are latency/finite-output checks, not a fixed-noise numerical parity gate.
Run each policy in a fresh process; correctness, capture/replay,
stage profiling, and task-level quality require separate validation. Compare
latency only at the same model/serving boundary, with matching inputs, denoising
steps, warmup, GPU, and precision policy.

## Reference results

These historical RTX 5080 measurements use R0's default vision/runtime settings;
they do not measure the experimental P0 options above.

### Client E2E latency

RTX 5080 16 GB, batch size 1, 10 denoising steps, p50/p95 of 30 round trips after 3 warmup, untimed server:

| dtype | path | p50 ms, 1 / 2 / 3 cameras | p95 ms, 1 / 2 / 3 cameras | device peak (GiB) |
| --- | --- | ---: | ---: | ---: |
| float32 | eager | 263.3 / 264.2 / 264.8 | 272.5 / 272.2 / 267.3 | 14.67 |
| float32 | **CUDA graphs + fused kernels** | **240.5 / 241.9 / 242.4** | 243.1 / 245.9 / 243.8 | 14.95 |
| bfloat16 | eager | 166.4 / 166.7 / 167.7 | 167.4 / 168.2 / 190.6 | 8.71 |
| bfloat16 | **CUDA graphs + fused kernels** | **118.0 / 118.1 / 118.2** | 118.9 / 120.1 / 119.6 | 9.07 |

### Model-level latency

RTX 5080 16 GB, batch size 1, 3 cameras, 10 steps, 5 warmup:

| dtype | PR 6950 (eager) | eager path | optimized path (CUDA graph + fused Triton kernels) |
| --- | ---: | ---: | ---: |
| float32 | 264.0 ms | 260.4 ms | 237.7 ms |
| bfloat16 | 173.0 ms | 168.3 ms | 114.0 ms |
