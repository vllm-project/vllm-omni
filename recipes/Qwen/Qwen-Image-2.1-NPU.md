# Qwen-Image-2.1 on Ascend NPU

> Low-latency text-to-image and image-conditioned generation on a single-machine
> 8-card Ascend 910B1 (Atlas 800I A2) — a serving reference measured for
> single-image latency.

## Summary

- Vendor: Qwen
- Model: [`Qwen/Qwen-Image-2.1`](https://huggingface.co/Qwen/Qwen-Image-2.1),
  [`Qwen/Qwen-Image-2.1-Turbo`](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo)
  (8-step distilled)
- Task: Text-to-image and image-conditioned generation (editing)
- Mode: OpenAI-compatible `/v1/images/generations` and `/v1/images/edits` serving
- Hardware: Ascend 910B1 (Atlas 800I A2), 8x NPU, 64 GB HBM per device
- Maintainer: Community

This recipe adapts [Qwen-Image-2.1.md](Qwen-Image-2.1.md) for Ascend NPU and
adds a measured low-latency serving reference. Differences from the GPU path:

- **No FP8.** Ascend 910B1 has no FP8 datatype, so the base recipe's FP8
  quantization and FP8 prefix-KV sections do not apply. Keep everything BF16.
- **No diffusion graph on NPU.** Ascend has its own NPU graph
  (`torch.npu.NPUGraph`, the ACL graph) that other vLLM-Omni models use, but
  this DiT does not: its decode-graph manager is CUDA-only and logs "not on
  CUDA ... falling back to eager decode" on `npu:0`, and Ascend does not
  `torch.compile` this transformer. The denoising loop runs eager either way.
- **`--enforce-eager` is not a latency lever here.** It flips this DiT's NPU
  attention-fusion switch, but measured steady-state latency at 1024×1024 is
  identical with or without it (2.67 s either way). It only skips a doomed
  first-request CUDA-graph-capture attempt, trimming the warmup request
  (~6.7 s vs ~8.4 s). The NPU optimizations that do run here — NZ weight
  storage, fused SwiGLU, broadcast modulation, residual reuse — are selected
  automatically by device and shape (see
  [Qwen-Image-2.1.md § NPU performance defaults](Qwen-Image-2.1.md#npu-performance-defaults)),
  not by this flag.

## Low-latency at a glance

Three facts drive the low-latency config on an 8-card 910B1.

- **The Turbo checkpoint is the fastest way to an image.** Its fixed 8-step
  distilled schedule beats the base model's 20- or 50-step run outright.
- **Ulysses sequence parallelism (`--ulysses-degree`) lowers single-image
  latency.** It shards the 4096-token DiT sequence across cards, so each rank
  denoises 1/N of the tokens. At 1024×1024 it measured **1.82x** on the
  distilled Turbo (2.67 s → 1.47 s at degree 4) and **2.18x** on the base model
  at 20 steps (6.25 s → 2.87 s).
- **Tensor parallelism does not.** `--tensor-parallel-size 2` measured
  **6.20 s** on the base model at 20 steps — statistically equal to one card's
  6.25 s — because sharding the 7.1B DiT weights buys little at batch 1 while
  it disables the TP1-only NPU fusions and adds per-rank all-reduce.

The whole model (text encoder 17.5 GB + 7.1B DiT 14.2 GB + VAE 1.4 GB) fits in
one device's 64 GB HBM, so single-card TP1 is the simplest configuration, and a
4-card Ulysses server is the lowest-latency one.

| Goal | Config | Checkpoint | Steps | Cards | Time/image |
| --- | --- | --- | ---: | ---: | ---: |
| Fastest, square | SP4 | Turbo | 8 (fixed) | 4 | **1.47 s** (1024×1024) |
| Fastest, banner, one card | TP1 | Turbo | 8 (fixed) | 1 | **2.27 s** (1280×704) |
| Fastest, square, one card | TP1 | Turbo | 8 (fixed) | 1 | 2.67 s (1024×1024) |
| Base, low latency | SP4 | base | 20 | 4 | 2.87 s (1024×1024) |
| Base, one card | TP1 | base | 20 | 1 | 6.25 s (1024×1024) |
| Base, default steps | TP1 | base | 50 | 1 | 15.26 s (1024×1024) |

## Prerequisites

### Checkpoint

```bash
export TURBO=/path/to/Qwen-Image-2.1-Turbo
export BASE=/path/to/Qwen-Image-2.1
# hf download Qwen/Qwen-Image-2.1-Turbo --local-dir "${TURBO}"
# hf download Qwen/Qwen-Image-2.1        --local-dir "${BASE}"
```

`vllm serve` accepts the repository id or a local path and resolves the nested
text-encoder / DiT / VAE components automatically.

### Environment

Use the official vLLM-Omni NPU image for Atlas 800I A2 (see the
[NPU installation guide](../../docs/getting_started/installation/npu.md)) and a
CPU-only (aarch64) PyTorch build with `torch_npu`. No CUDA runtime is present.
Within the image:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
```

## Start a server (single card)

This is the simplest configuration and the recommended starting point.
`--enforce-eager` makes the eager intent explicit (there is no diffusion graph
on NPU; see above) but does not change steady-state latency. `--num-gpus 1`
keeps tensor and sequence parallelism off. For the lowest single-image latency,
use the 4-card Ulysses server in
[Scale out](#scale-out-on-one-8-card-machine) instead (~1.8x faster at
1024×1024).

```bash
export PORT=8091

vllm serve /path/to/Qwen-Image-2.1-Turbo \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --served-model-name Qwen/Qwen-Image-2.1-Turbo \
  --num-gpus 1 \
  --enforce-eager \
  --trust-remote-code
```

To serve the base checkpoint, replace the model path **and** the
`--served-model-name` (use `Qwen/Qwen-Image-2.1`), then send that same name as
the request `model`; leaving the Turbo alias in place would reject base-model
requests with HTTP 400. Set `num_inference_steps` per request (`20` for lower
latency, `40` for the reference default, `50` is the server default). To change
the shape, use `size` in the request.

## Scale out on one 8-card machine

The 8 cards do two different jobs, and they do not stack for a single image:

- **Lower single-image latency** — add Ulysses sequence parallelism. Replace
  `--num-gpus 1` with `--num-gpus 4 --ulysses-degree 4` (or 2). This is the
  latency lever; it trades cards for wall-clock on one image.
- **Higher throughput** — run independent single-card servers (data parallelism)
  and balance requests across them. Each request still takes the single-card
  latency, but N servers serve N images at once.

```bash
# Lower latency for one image: one 4-card sequence-parallel server
vllm serve /path/to/Qwen-Image-2.1-Turbo \
  --omni --num-gpus 4 --ulysses-degree 4 --ulysses-mode strict \
  --served-model-name Qwen/Qwen-Image-2.1-Turbo \
  --enforce-eager --trust-remote-code --port 8091

# Higher throughput: run several single-card servers and balance requests across
# them (each request keeps single-card latency; N servers serve N images at once)
vllm serve /path/to/Qwen-Image-2.1-Turbo \
  --omni --num-gpus 1 --served-model-name Qwen/Qwen-Image-2.1-Turbo \
  --enforce-eager --trust-remote-code --port 8092
```

## HTTP API

The request contract is identical to the GPU recipe; see
[Qwen-Image-2.1.md § Online Serving](Qwen-Image-2.1.md#online-serving). The
Turbo checkpoint runs a fixed 8-step schedule, so omit `num_inference_steps` and
keep `true_cfg_scale` at `1.0`:

```bash
curl -sS -X POST "http://127.0.0.1:${PORT}/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen-Image-2.1-Turbo",
       "prompt":"A ceramic teapot on a wooden table",
       "size":"1280x704",
       "true_cfg_scale":1.0,
       "seed":42}' \
  | jq -r '.data[0].b64_json' | base64 -d > out.png
```

## Benchmarks (measured)

Measured on one Ascend 910B1 (64 GB) with CANN 9.1.0, `torch_npu` 2.10.0,
`triton_ascend` 3.2.2, vLLM 0.32.0, vLLM-Omni commit `5917730a4`, BF16, CFG 1,
`--enforce-eager`, fixed prompt `"A ceramic teapot on a wooden table"` and
seed 42, batch 1. Each row is the engine-side request latency from `vllm serve`
(the `OmniTiming ... engine=` field): start a server, send one warmup request
(dropped — it includes the one-time regional kernel compile), then take the
median of 5 steady-state requests. End-to-end HTTP latency runs ~0.1–0.6 s
higher than the engine numbers.

| Checkpoint | Resolution | Steps | Parallelism | Time per image (s) |
| --- | --- | ---: | --- | ---: |
| Turbo | 1280×704 | 8 (fixed) | TP1, 1 card | 2.27 |
| Turbo | 1024×1024 | 8 (fixed) | TP1, 1 card | 2.67 |
| Turbo | 1024×1024 | 8 (fixed) | SP4 (Ulysses 4), 4 cards | 1.47 |
| base | 1024×1024 | 20 | TP1, 1 card | 6.25 |
| base | 1024×1024 | 20 | SP4 (Ulysses 4), 4 cards | 2.87 |
| base | 1024×1024 | 20 | SP2 (Ulysses 2), 2 cards | 5.37 |
| base | 1024×1024 | 20 | TP2, 2 cards | 6.20 |
| base | 1024×1024 | 50 | TP1, 1 card | 15.26 |

These were measured on a shared multi-tenant cluster; read the relative ordering
as the robust result and leave a few percent of headroom on the absolutes. The
ordering held across re-measurements at different load levels: Turbo < base, and
for the base model SP4 < SP2 < TP1 ≈ TP2.

## Multi-card parallelism

At 1024×1024 / 20 steps on the base model, measured against **6.25 s** on one
card:

| Parallelism | Cards | Time/image | vs one card |
| --- | ---: | ---: | --- |
| TP1 (none) | 1 | 6.25 s | — |
| TP2 | 2 | 6.20 s | ≈ equal (0.99x) |
| SP2 (Ulysses) | 2 | 5.37 s | 1.16x faster |
| SP4 (Ulysses) | 4 | 2.87 s | 2.18x faster |

Tensor parallelism shards the 7.1B DiT weights but replicates the 17.5 GB text
encoder per rank and disables the TP1-only NPU fusions, so at batch 1 it nets
out to nothing — TP2 lands at the same latency as one card, not faster. Ulysses
sequence parallelism shards the 4096-token denoising sequence, which directly
reduces each rank's attention and MLP work; at this resolution the all-to-all
traffic is small enough that SP4 wins by ~2.2x. SP2 gives only 1.16x, so 4 cards
is the measured sweet spot for 1024×1024.

The distilled Turbo gets the same lever: at 1024×1024, SP4 measured 1.47 s
against 2.67 s on one card (1.82x). Its fixed 8-step loop is short, so the
speedup is a little less than the 20-step base schedule's 2.18x, but SP4 is still
the fastest way to an image here — faster than single-card Turbo even at the
smaller 1280×704 banner shape (2.27 s).

Multi-card parallelism is therefore a latency lever for every schedule, and a
throughput lever too. Do not use `--tensor-parallel-size` expecting a latency
win; prefer `--ulysses-degree` for latency or independent single-card servers
for throughput.

## Known limitations

- Ascend 910B1 has no FP8, so FP8 quantization and FP8 prefix-KV are unavailable
  here.
- The DiT runs eager on NPU — no diffusion graph is captured (see above). The
  NPU kernel optimizations that apply here (NZ weight storage, fused SwiGLU) are
  selected automatically by the 910B1 device and input shape, not by
  `--enforce-eager`.
- `--text-encoder-tp-size` is not supported for this model on NPU; the text
  encoder runs replicated on every rank.
- The first `vllm serve` request after startup is a warmup (regional kernel
  compile) and runs ~3x steady state, so drop it from any latency measurement.
  The numbers above are steady-state engine timings, confirmed by real serving
  runs: single-card Turbo 1280×704 serves real images in ~2.27 s (engine), and
  the 4-card Ulysses `--ulysses-degree 4` server in ~1.47 s at 1024×1024
  (engine).

## Additional resources

- [Qwen-Image-2.1.md](Qwen-Image-2.1.md) — full GPU guide (CUDA graphs, FP8,
  prefix KV cache, quality comparison)
- [Qwen-Image-2.1.md § NPU performance defaults](Qwen-Image-2.1.md#npu-performance-defaults)
  — the NPU auto-fusion eligibility and validation notes
- [MiniMax-H3-NPU.md](../MiniMaxAI/MiniMax-H3-NPU.md) — larger Ascend NPU
  deployment reference (multi-card USP/text-encoder-TP/VAE-tile patterns)
- [NPU installation](../../docs/getting_started/installation/npu.md)
