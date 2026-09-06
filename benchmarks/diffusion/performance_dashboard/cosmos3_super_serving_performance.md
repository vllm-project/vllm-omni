# Cosmos3-Super serving performance dashboard

This document describes how to deploy and benchmark **nvidia/Cosmos3-Super** using vLLM-Omni. It covers service startup configuration, parallelism options, benchmark methodology, dataset settings, and performance results on 8x H200.

---

# 1. Overview

Cosmos3-Super is a 64B Mixture-of-Transformers world foundation model. It carries an autoregressive reasoner tower and a diffusion generator tower in a single checkpoint. This document covers the **generator** path only, served through vLLM-Omni's `/v1/videos` endpoint.

Contents:

* Service launch configuration across five parallelism strategies
* Benchmark entry and arguments
* Dataset and workload settings
* Single request latency, a latency decomposition model, and a concurrency envelope
* Reproducibility guidelines, including a determinism caveat that affects output comparison
* B200 addendum: parallel-strategy pair, scaling ladder, concurrency spot-check, and batch-path results on 8x B200

---

# 2. Test environment

| Component | Specification |
|---|---|
| GPU | 8x NVIDIA H200 141GB SXM, NV18 full mesh at 26.562 GB/s per link |
| Driver | 580.126.09 |
| Host | 128 vCPU, 1,574 GB RAM |
| Container | `vllm/vllm-omni:cosmos3`, digest `sha256:6d2630c7d637b699557573f2c3fee8df5d4d0cd718977aa22549ed6a6ef30587` |
| vllm / vllm-omni | 0.25.0 / 0.25.0rc2.dev62+g9c1b7504b |
| torch / transformers / diffusers | 2.11.0+cu130 / 5.13.0 / 0.38.0 |
| Model snapshot | `e0262be9d8f7586bc24c069a2aed2b665bdff266`, 124 GB on disk |
| Diffusion attention backend | FlashAttention |

The image ships a pre-release dev build of vllm-omni rather than a tagged release. A cross-image control on `v0.26.0` at the anchor cell (guardrails off, n=1 per image) accepted the same flags and matched: 124 s wall against 123 s on this image, with output parity inside the same configuration restart band described in 8.1. Guardrails-on behavior on `v0.26.0` was not tested.

---

# 3. Service launch configuration

## 3.1 Basic serving command

```bash
vllm serve nvidia/Cosmos3-Super --omni \
    --host 0.0.0.0 --port 8000 \
    --cfg-parallel-size 2 --ulysses-degree 4 \
    --use-hsdp --hsdp-shard-size 8 \
    --init-timeout 1800
```

## 3.2 Two deployment prerequisites

Both of these cause startup failure on a clean host and are worth stating explicitly.

**The guardrail repository is gated.** `nvidia/Cosmos-1.0-Guardrail` is pulled at server startup and requires license acceptance, even though the main model is ungated and downloads without authentication. Without it the server exits 1 with `GatedRepoError: 401`. Accept the license and supply `HF_TOKEN`, or pass `--no-guardrails`. A useful diagnostic: an authenticated fetch returns 403 while an anonymous fetch returns 401, which separates a missing license acceptance from a bad token.

**The Xet download client fails on this image's pinned versions.** With guardrail access granted, startup still fails inside `huggingface_hub` with `RuntimeError: Task error: Unable to parse string as hex hash value`. This is huggingface/xet-core#895, which reproduces on `hf-xet 1.5.1` plus `huggingface_hub 1.23.0`. This image pins exactly that pair. Set `HF_HUB_DISABLE_XET=1`.

## 3.3 Key parameters

| Parameter | Description |
|---|---|
| `--cfg-parallel-size` | CFG parallelism degree (capped at 2) |
| `--ulysses-degree` | Ulysses sequence parallel degree |
| `--ring-degree` | Ring attention degree |
| `--use-hsdp`, `--hsdp-shard-size` | Hybrid sharded data parallelism |
| `--vae-patch-parallel-size` | VAE patch parallelism degree |
| `--tensor-parallel-size` | Tensor parallelism degree |

Record these when reporting results. Configuration was verified as applied rather than assumed, by reading `Building SP subgroups from explicit sp_group_ranks` and the resulting `sp_group` membership out of the server log.

---

# 4. Benchmark script

## 4.1 Benchmark entry

```bash
python benchmarks/diffusion/diffusion_benchmark_serving.py \
    --base-url http://localhost:8000 \
    --model nvidia/Cosmos3-Super \
    --endpoint /v1/videos \
    --task t2v \
    --dataset random \
    --num-prompts <N> \
    --max-concurrency <C> \
    --width 1280 --height 720 --num-frames 189 --fps 24 \
    --num-inference-steps 35 \
    --warmup-requests <C> --warmup-concurrency <C> \
    --warmup-num-inference-steps 4 \
    --output-file results.json
```

## 4.2 Key benchmark arguments

| Parameter | Description |
|---|---|
| `--endpoint` | Use `/v1/videos` for the generator |
| `--dataset` | `random`, `vbench`, or `trace` |
| `--task` | `t2v` for text to video |
| `--num-frames` | Frame count. 189 is the model card default |
| `--max-concurrency` | Client side concurrency |
| `--warmup-concurrency` | Match to `--max-concurrency` so the first measured batch excludes compile overhead |
| `--slo` | Enables SLO attainment reporting |

Note that CLI `--width` and `--height` override per row values when `--dataset trace` is used, which flattens a mixed shape trace into a single shape. Omit them for trace replay.

---

# 5. Dataset and workload settings

The anchor cell used throughout is the model card's own text to video profile.

* Resolution 1280x720, 189 frames, 24 fps
* `num_inference_steps` 35 and 50
* `guidance_scale` 6.0, `flow_shift` 10.0, `max_sequence_length` 4096
* Fixed `seed` 17
* Prompts: the repository's pre-upsampled `assets/example_t2v_prompt.json` with `assets/negative_prompt.json` for single request measurement, and `--dataset random` for concurrency cells

---

# 6. Performance metrics

| Metric | Description | Unit |
|---|---|---|
| `stage_gen_time_ms` | Server side generation stage | milliseconds |
| Denoise loop | Sampler loop elapsed, read from the progress bar | seconds |
| Per step | Measured sampler step rate | s/it |
| Mean / P99 latency | Client side request latency | seconds |
| Throughput | Completed clips per hour | clips/hr |
| Peak memory | Per GPU peak | MB |

Read the per step rate from the sampler progress bar. Dividing `stage_gen_time_ms` by step count returns a derived figure that reproduces the total by construction and hides the fixed cost described in section 7.2.

---

# 7. Performance results

## 7.1 Parallel strategy comparison

Anchor cell, 35 steps, guardrails off, n=3 per configuration.

| Configuration | CFG | Ulysses | Ring | TP | VAE PP | `stage_gen_time_ms` (s) | Denoise loop (s) | Per step (s/it) | Spread |
|---|---|---|---|---|---|---|---|---|---|
| Recommended | 2 | 4 | 1 | Off | Off | **120.57** | 106 | 3.03 to 3.06 | 0.18% |
| Recommended + VAE PP | 2 | 4 | 1 | Off | 2 | 121.13 | 105.5 | 3.00 to 3.03 | 0.17% |
| Ulysses + Ring hybrid | 2 | 2 | 2 | Off | Off | 123.90 | 110 | 3.11 to 3.15 | 0.16% |
| Sequence parallel only | Off | 8 | 1 | Off | Off | 125.7 | not recorded | 3.18 (derived) | 0.0% |
| Tensor parallel | Off | Off | Off | 8 | Off | 131.26 | 117 | 3.33 to 3.36 | 0.13% |

At 50 steps the ordering holds: the recommended configuration measures 166.26 s against tensor parallel at 181.31 s. The recommended 50 step figure is server side n=1, with wall times of 168, 169, and 168 s at n=3; the tensor parallel figure is n=3.

The model card's recommended configuration is the fastest of the five. Tensor parallelism, which is the methodology behind NVIDIA's published benchmark grid for 4 and 8 GPU configurations, measures about 9% slower end to end (8.9% at 35 steps, 9.1% at 50; its per step rate is 9.9% slower). The Ulysses plus Ring hybrid serves correctly and deterministically, which is worth noting given the known ring attention issue on B200 (#5611).

The sequence parallel only row explains why the recommendation splits rather than widens. CFG parallel has a hard ceiling of 2, so the alternative use of those GPUs is a wider Ulysses degree. Doubling Ulysses from 4 to 8 and running the two classifier free guidance passes sequentially costs 4.3% against the recommendation, so the 2 way CFG split earns more than those four GPUs would earn on the sequence axis. Its stage timing was captured at whole second resolution, so it carries roughly plus or minus 0.5 s where the other rows carry plus or minus 0.2 s, its per step figure is derived from the fixed cost in 7.2 rather than read from its own progress bar, and its 0.0% spread is wall clock based (130 s on all three repetitions) where the other rows report stage based spread.

## 7.2 Latency decomposes into a fixed cost and a per step cost

Every cell above fits `latency = 14.4 s + N x per_step` to within 0.3%:

| Cell | Predicted | Measured |
|---|---|---|
| Recommended, 35 steps | 14.4 + 35 x 3.04 = 120.8 s | 120.57 s |
| Recommended, 50 steps | 14.4 + 50 x 3.04 = 166.4 s | 166.26 s |
| Tensor parallel, 35 steps | 14.3 + 35 x 3.34 = 131.2 s | 131.26 s |
| Tensor parallel, 50 steps | 14.3 + 50 x 3.34 = 181.3 s | 181.31 s |

Two results follow.

The fixed term sits at 14.3 plus or minus 0.9 s across every parallelism strategy and both step counts. Work outside the denoise loop, which includes text encode, VAE decode, and muxing, does not benefit from any of the configurations tested. At 35 steps that is 12% of total latency.

The entire difference between strategies is denoise step cost. None of it is startup, scheduling, or teardown.

Patch parallel VAE decode was tested directly against the fixed term and does not reduce it. The denoise loop is unchanged and the fixed term rises by about a second.

What composes the 14.4 s remains open: the server emits no per stage VAE or text encode timing, and the profiler endpoints do not register for this model through the documented serve path, so the fixed cost could not be decomposed here. That limitation is reported separately as an issue (#5925).

## 7.3 Concurrency envelope

Production shape: 1280x720, 189 frames, 35 steps, guardrails on, `--dataset random`, `--request-rate inf`, n equal to twice the concurrency per cell.

| Concurrency | Throughput | Mean latency | p50 | p99 | Video seconds per wall second | Peak memory |
|---|---|---|---|---|---|---|
| 1 | 28.21 clips/hr | 127.6 s | 127.6 s | 127.8 s | 0.0617 | 38,562 MB |
| 2 | 30.43 | 208.2 s | 230.1 s | 243.9 s | 0.0666 | 38,562 MB |
| 4 | 30.78 | 382.0 s | 461.4 s | 474.6 s | 0.0673 | 38,564 MB |
| 8 | **30.96** | 729.0 s | 923.3 s | 936.2 s | **0.0677** | 38,564 MB |

Throughput saturates near 31 clips/hr, a gain of 9.8% from concurrency 1 to 8, with most of it collected by concurrency 2. Median latency is almost exactly linear in concurrency, doubling at each step, which is the signature of serialization rather than batching. Stated as a trade, concurrency buys 9.8% of throughput and costs 7.2x of median latency on this shape.

A clip is 189 frames at 24 fps, so 7.875 s of video. At saturation the node sustains about one second of generated 720p video for every 15 seconds of wall clock, and concurrency moves that figure by less than 10%. That rate, rather than clips per hour, is the one to carry into a synthetic data volume estimate.

Three controls each remove an objection to the table. VBench prompts match `--dataset random` to within 0.1 s, so prompt source does not move latency. Poisson arrivals match all at once arrivals, so arrival pattern does not matter under serialization. Guardrails on versus off costs 1.3 to 1.8% under load, and the gap narrows from concurrency 2 to 4, so the guardrail stage does not serialize against generation.

**Peak memory moves by 2 MB across an eightfold increase in in flight work.** A batching implementation could not hold allocation flat while in flight work grows eightfold. This confirms by measurement what RFC #4340's feature matrix records for Cosmos3-Super: step level batching (Step Execution) is unsupported, so concurrent requests queue and serialize. The reduced shape series in 7.3.2 shows the same flat allocation signature.

## 7.3.1 The stock benchmark harness reports a throughput collapse that does not exist

`benchmarks/diffusion/backends.py` hardcodes `timeout_seconds = 600.0` with no command line override. Mean latency on this shape reaches 729 s at concurrency 8, so the client kills its own requests, counts them as failures, and then divides successes by the full benchmark duration.

| Cell | Client timeout | Completed | Reported throughput |
|---|---|---|---|
| Concurrency 8 | 600 s, stock | 5 of 16 | 14.96 clips/hr |
| Concurrency 8 | 5400 s, patched | 16 of 16 | 30.96 clips/hr |

The stock harness understates throughput by 52% on this shape and presents it as a collapse under load rather than as a client timeout. Any concurrency figure taken with the unmodified harness, for a shape whose latency under load approaches 600 s, should be re-checked. Raising the timeout, or exposing it as an argument, is a one line change.

## 7.3.2 Reduced shape envelope and the queue wait metric

An earlier sweep at a reduced shape (720p, 81 frames, 20 steps, guardrails on, one pass per cell): throughput saturates at 119.72, 131.17, 134.59, 136.49 clips/hr for concurrency 1, 2, 4, 8, a gain of 14.0% collected almost entirely by concurrency 2. Mean latency scales like serialization at 1.60x, 1.81x, and 1.90x per doubling. **Peak memory is byte identical at 39,804 MB across concurrency 1 through 8 in this series**, the same flat allocation signature as the production shape in 7.3.

The harness reports `queue_wait_ms` at 0.21 to 0.53 ms through concurrency 4 and 131 ms at concurrency 8 in this series, while end to end latency grows 5.5x from concurrency 1 to 8 (30.07 s to 165.47 s). The waiting happens inside `stage_0_gen_ms`, not in the reported queue. Diagnosing saturation on this stack from `queue_wait_ms` alone will show an idle server that is in fact fully serialized.

## 7.4 Attention backend: leave the platform default alone

The server resolves `FLASH_ATTN` via platform default on H200 (sm_90). The in-repo diagnostic `benchmarks/diffusion/bench_attention_backends.py` suggests that is a poor choice, ranking FlashInfer at 1.13x and cuDNN at 1.00x against torch SDPA's flash kernel at 0.50x on the no mask path. Measured end to end at the anchor cell, n=3 per arm, with the applied backend verified from the selector's own resolution line rather than from the flag:

| Backend | `stage_gen_time_ms` (s) | Wall (s) | vs default | Diagnostic predicted |
|---|---|---|---|---|
| Platform default, resolves `FLASH_ATTN` | **119** | 123, 123, 123 | baseline | slowest, 0.50x |
| `FLASHINFER_ATTN` | 120 | 124, 125, 125 | +0.8% | fastest, 1.13x |
| `CUDNN_ATTN` | 131 | 135, 135, 135 | +10.1% | 1.00x |

The ranking nearly inverts. The backend the diagnostic rates slowest by 2x is the fastest in the real model, and the one it rates fastest ties. **Do not select a diffusion attention backend from the kernel diagnostic.** It exercises a synthetic shape through torch SDPA dispatch, while the server reaches a different implementation at Cosmos3's own dimensions, which are 64 heads at head dim 128 with 8 key value heads and a much longer sequence than any preset. The diagnostic remains useful for its stated purpose, which is finding silent cuDNN fallbacks, and its ranking does not hold across the mask boundary either, where FLASH_ATTENTION fails outright.

Output differs bytewise between backends but not measurably. Against the platform default, cuDNN measures 28.81 dB, inside the same configuration band described in 8.1, and FlashInfer 29.85 dB, at or above that band's top, which is closer to its control than same configuration clips are to each other.

## 7.5 Guardrail cost is content dependent

| Prompt | Guardrails | Generation time | n |
|---|---|---|---|
| Repository example prompt | Off | 120.57 s | 3 |
| Repository example prompt | On | 139.01 s | 2 |
| `--dataset random` prompts | On | 122.57 s | one harness series |

The same enabled guardrail stage costs 18.4 s on one prompt and about 2 s on another at identical resolution, frame count, and step count. Per frame face detection and blurring is a plausible driver, since that work runs only when generated content triggers it. Report guardrail overhead as a range tied to content rather than as a constant. Guardrails also add about 20 s of one time initialization and 17 GB of weights on disk.

## 7.6 Mixed workload trace

Every other cell in this dashboard holds shape constant. This one does not. `--dataset trace`, 16 requests at max concurrency 4, guardrails on, 12 s nominal arrival spacing: twelve requests at 832x480, 61 frames, 20 steps, interleaved with four at 1280x720, 189 frames, 35 steps, one long request in every fourth slot.

Solo baselines measured under the same conditions, so the comparison is like for like:

| Shape | Solo latency | n |
|---|---|---|
| 832x480, 61 frames, 20 steps | 25 s | 3 |
| 1280x720, 189 frames, 35 steps | 139.01 s | 1 |

| Metric | Value |
|---|---|
| Completed and failed | 16 and 0 |
| Wall duration | 580.06 s |
| Serialized prediction from solo latencies | 856 s |
| Mean, p50, p95, p99 latency | 120.28, 141.41, 146.25, 154.43 s |
| Peak memory max and mean | 39,612 and 37,257 MB |

**About a third of the work overlaps.** Fully serialized, these requests would take 856 s. The arrival spread is only 180 s, so a serialized server could not have finished before roughly that figure, and the measured 580 s is 32% below it. Some per request stage runs off the critical GPU path.

This also explains why 7.3 shows so little concurrency gain. There, every request is an expensive 720p clip whose non generation fraction is small. Here three quarters of requests are cheap, that fraction dominates, and concurrency finally pays. **The near flat scaling in 7.3 is a property of the homogeneous 720p workload, not of the server.** A mixed or cheaper workload gets materially more out of concurrency.

**Peak memory is not invariant once shapes are mixed.** It holds within 2 MB across the entire homogeneous sweep, then rises about 1 GB here, and the mean falls below the max where the two previously coincided. Serving two geometries appears to need resident buffers for both. The effect is small against 141 GB cards, but a homogeneous benchmark reports this quantity as constant when it is not.

Two limits are worth stating. The harness reports only aggregate latencies, with no per request breakdown, so this cell cannot answer what a short request pays when it queues behind a long one. That needs a client side per request timer. And SLO attainment reads 0.00 here at p99 154 s while a Poisson cell in the same series passes at p99 462 s, so the same `slo_scale` implies very different thresholds between datasets. The harness normalizes against a per dataset reference it does not surface, which makes the reported SLO figure uninterpretable as published.

---

# 8. Reproducibility checklist

* Record GPU type, driver, and NVLink topology
* Record the container digest, not just the tag
* Record the parallel configuration, and verify it applied by reading `sp_group` membership from the server log rather than trusting the flags
* Record resolution, frame count, step count, seed, `guidance_scale`, and `flow_shift`
* Match `--warmup-concurrency` to `--max-concurrency`
* Ensure no background workload on the GPUs
* Report the per step rate from the sampler progress bar, not from `stage_gen_time_ms` divided by step count
* State whether guardrails were enabled, and state which prompt set was used, because guardrail cost varies with content

## 8.1 Determinism caveat, important for output comparison

Output is bitwise reproducible **within a single server instance**. Repeated requests at a fixed seed return byte identical files in every configuration tested.

Determinism holds across frame counts, resolutions, parallel configurations, and guardrail state. Eighteen clips spanning six shapes were byte identical within their shape, and the guardrails on cells are byte identical too.

Output is **not** reproducible across a server restart, and container recreation alone is enough. No host reboot is required. Three same configuration clips at the same seed produced three distinct checksums and three distinct file sizes, with no bitwise identical frames in any pairing. The determinism boundary is exactly the server process.

The size of that effect is a band, not a number. Four same configuration crossings measure medians of 26.81, 27.56, 28.12, and 28.99 dB PSNR, a spread of 2.18 dB. For scale, adjacent frames within one clip measure 27.91 dB, and numerical noise alone would sit far higher.

This matters for anyone comparing configurations by output, and it is easy to get wrong. Each configuration runs on its own server instance, so every cross configuration comparison carries a restart effect. In this study, tensor parallel against the recommended configuration measured 26.93, 28.75, and 27.49 dB across three crossings, entirely inside the same configuration band and not even its lowest member. There is no evidence that tensor parallelism alters output.

Measured against a single control, the first of those figures had looked like a real 1.19 dB shortfall. **One control is worse than none, because it supplies a precision the measurement does not have.** Measure several same configuration crossings on your own hardware, report the band rather than a floor, and treat any cross configuration result inside that band as unresolved rather than as agreement or as difference.

---

# 9. B200 addendum

### Environment (B200 lane)

| Item | Value |
| --- | --- |
| Hardware | 8x NVIDIA B200 SXM (192 GB HBM3e per GPU), 160 vCPU, 1792 GB RAM |
| Driver / host image | 580.126.09 / ubuntu24.04-cuda13.0.0.2.711 |
| Cloud | Nebius us-central1, gpu-b200-sxm, preset 8gpu-160vcpu-1792gb, preemptible |
| Image | `vllm/vllm-omni:cosmos3` @ sha256:6d2630c7d637b699557573f2c3fee8df5d4d0cd718977aa22549ed6a6ef30587 |
| Model revision | `nvidia/Cosmos3-Super` @ e0262be9d8f7586bc24c069a2aed2b665bdff266 |
| Harness | in-image `benchmarks/diffusion/diffusion_benchmark_serving.py`, client timeout patched 600 s -> 5400 s (same patch as the H200 lane) |

### 1. Parallel-strategy pair, anchor shape (720p / 189 frames / 35 steps, seed 17, guardrails off, n=3; plus one 50-step cell per arm)

| Arm | Wall (s), 35 steps | Wall (s), 50 steps |
| --- | --- | --- |
| Recommended: CFG2 x Ulysses4 x HSDP8 | 70 / 68 / 69 | 91 |
| Tensor parallelism 8 | 78 / 77 / 77 | 104 |

The recommended config is 10.7% faster at 35 steps and 12.5% faster at 50 steps. The ~9% recommended-over-TP advantage measured on 8x H200 (PR #5926 section 7.1) generalizes across GPU generations and slightly widens on B200.

Steps-axis decomposition on the recommended arm, using the same derivation as section 7.2 of the H200 entry (step delta, guardrails off, example prompt):

- Per-step rate: from the pair walls 69 s (35 steps, mean of 70/68/69) and 91 s (50 steps), (91 - 69)/15 = **1.467 s/step**. MP4 encode cancels inside a step delta.
- Cross-generation scaling: the H200 entry's matched figures give (166.26 - 120.57)/15 = **3.046 s/step**, so the per-step rate scales **2.08x** from H200 to B200.
- Fixed cost, with the basis difference stated: H200's stage-based fixed cost is 14.4 s. B200's wall-derived fixed cost is 17.7 s and includes MP4 encode and request overhead, which were not recorded separately on this host. Net of a few-second encode, the B200 fixed cost lands in the same ~14 s class, so the fixed cost reads as approximately GPU-independent.
- Derivation warning: neither figure above uses the startup-stats field `denoise_step_latency_ms`. That field is stage_gen/steps by construction (3,448 ms on H200 guardrails off; 2,388.4 ms on B200 guardrails on at this prompt, matching 35 x 2,388.4 ≈ 83,595 ms stage-gen), so it amortizes the fixed cost, and on B200 also the guardrail cost, into every step. Section 7.6's guidance already bars that derivation.

### 2. Scaling ladder spot-cells (720p / 189 frames / 35 steps, seed 17, guardrails off, wall, n=1)

| GPUs | Measured (s) | NVIDIA published (s; tensor parallelism per their methodology) |
| --- | --- | --- |
| 1 | 377 | 390.28 |
| 4 | 122 | 113.31 |
| 8 | 77.3 (mean of 3, TP-8) | 62.11 |

Within about 3-8% at 1 and 4 GPUs; 24% above the published 8-GPU number. Direction matches the H200 lane's finding that the card's published grid is not exactly reproducible on third-party hosts even with the vendor image; the gap is wider here than on H200 (8%). Comparing against the server-side stage-generation time (which, like NVIDIA's grid, excludes MP4 encode and request overhead): even subtracting ~3 s of encode, the 8-GPU delta stands at ~19%.

### 3. Concurrency spot-check, production shape (720p / 189 frames / 35 steps, recommended config, guardrails on, `--dataset random`, patched client)

| Concurrency | Clips/hr | stage-gen mean (s) | Peak memory max (MB) |
| --- | --- | --- | --- |
| 1 | 49.06 | 68.5 | 38,562 |
| 2 | 54.99 | 110.7 | 38,562 |

+12.1% throughput at concurrency 2 with byte-identical peak memory (38,562 MB at both). Throughput saturates by concurrency 2 on B200 as well, consistent with the H200 entry's finding that added concurrency queues rather than batches. The B200's larger per-GPU memory changes nothing about the envelope at this shape.

### 4. Blackwell-specific notes for the entry

- The pinned `cosmos3` image serves the recommended config on B200 with no #5611-class failure; server logs resolve the layout to ulysses=4, ring=1, so ring degree > 1 remains unverified on B200 by this lane (see the datapoint comment on RFC #5631).
- `vllm/vllm-omni:v0.26.0` fails to start with guardrails at the default ON (missing `cosmos-guardrail` package; the pinned image ships 0.3.1). With `--no-guardrails` it serves at 72 s (vs 69.0 s) and its output differs materially from the pinned image at the same seed (median PSNR 30.54 dB, below even the B200 restart comparisons that follow).
- B200 same-config restart drift, measured from two restart comparisons (guardrails-on example prompt): median PSNR 32.17 and 32.13 dB. Two comparisons are an early estimate, but they already sit outside the H200 range (26.8-29.0 dB over four comparisons); use B200-measured drift for B200 parity claims.

### 5. Batch-path addendum: `npa-cosmos3` on B200

- `npa-cosmos3:1.2.2-cu130` (nebius-physical-ai @ 3fe8584, cosmos-framework 5e67049) on B200: Super t2v anchor shape, guardrails on, seed 17. Warm-cache invocations complete in 473 and 472 s wall (n=2); a first invocation including one-time guardrail-asset downloads took 740 s. Sampling 394 s (35 steps, ~11.25 s/step) on a single B200 card; the full 124 GB checkpoint fits one 192 GB card. Load-to-first-step ~80 s warm. All four runs byte-identical (sha256 3c96f88d9710d58b82399a2c8fe0df7964a07b179b902d179838f2578df5af77): the batch path is deterministic across process restarts on B200, in contrast to the serving path's per-server-instance determinism.
- The serving container built from the pinned image (PR #268 branch @ dbc59ca) was also validated on B200: build gate pass, all three startup preflights refuse correctly (no token; unwritable cache; 4-of-8 GPUs), readiness 261 s, anchor clip content-verified (wall 87 s, stage-gen 83.7 s at 35 steps), per-rank memory 43.3-43.4 GiB during generation (49.7 GiB on rank 0, which also hosts the API server and guardrails; nvidia-smi readings 44,320-44,472 MiB and 50,880 MiB).

### Artifacts

Measurement logs, sweep JSONs, parity logs, sha256 manifests, and clips are retained and available on request; all numbers above are reproducible from them.
