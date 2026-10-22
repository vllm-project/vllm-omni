# MammothModa2

> MammothModa2-Preview and MammothModa2-Dev unified understanding and generation

## Summary

- Vendor: ByteDance Research
- Models: `bytedance-research/MammothModa2-Preview`, `bytedance-research/MammothModa2-Dev`
- Tasks: Preview and Dev text-to-image (AR → DiT); Dev text/image understanding
- Mode: Offline inference
- Maintainer: Community

## When to use this recipe

Use this recipe to run MammothModa2-Preview through shared task-oriented
offline examples. Text-to-image uses the shared image example (`text_to_image.py`) instead of a model-specific script.
The generic example formats the AR prompt, drives the AR → DiT stage pipeline,
and forwards MammothModa2-specific generation parameters through the
pipeline-declared `extra_body` contract.

MammothModa2's DiT stage runs in the shared diffusion runtime. The default
`mammoth_moda2.yaml` uses request-level batching with up to eight compatible
requests (`max_num_seqs: 8`) and one image per request
(`num_outputs_per_prompt: 1`). TeaCache and Cache-DiT are optional alternative
cache backends in request mode. Enable TeaCache with `--cache-backend tea_cache`
as shown below, or enable Cache-DiT using the commented settings in that YAML.
Compilation, DiT quantization, parallelism, and offload are not enabled by
these presets.

To use step execution and continuous batching, select
[`mammoth_moda2_step.yaml`](../../vllm_omni/deploy/mammoth_moda2_step.yaml)
with `--deploy-config vllm_omni/deploy/mammoth_moda2_step.yaml` in the
text-to-image commands below. This preset inherits the same stage placement
and capacity, enables step execution for Stage 1, and disables request-batch
admission waiting and diffusion cache acceleration. Compatible requests can
join between denoising steps and finish independently, including requests
with different inference-step counts. Step mode cannot be combined with
TeaCache, Cache-DiT, or other diffusion cache backends.

Image size, seed, guidance, and denoising steps use the standard diffusion
request fields. `cfg_range` remains a MammothModa2-specific `extra_body`
parameter. For compatibility, the runtime also accepts the former
`text_guidance_scale` and `num_inference_steps` keys in `extra_body`; when
present and non-null, those keys take precedence over the standard fields.

## References

- Upstream model:
  [`bytedance-research/MammothModa2-Preview`](https://huggingface.co/bytedance-research/MammothModa2-Preview)
- Dev model:
  [`bytedance-research/MammothModa2-Dev`](https://huggingface.co/bytedance-research/MammothModa2-Dev)
- Related offline example:
  [`examples/offline_inference/text_to_image/text_to_image.py`](../../examples/offline_inference/text_to_image/text_to_image.py)
- Related T2T/I2T example:
  [`examples/offline_inference/x_to_text/x_to_text.py`](../../examples/offline_inference/x_to_text/x_to_text.py)
- Declared parameters:
  [`vllm_omni/model_extras/mammothmodal2_preview.py`](../../vllm_omni/model_extras/mammothmodal2_preview.py)
- Deploy config:
  [`vllm_omni/deploy/mammoth_moda2.yaml`](../../vllm_omni/deploy/mammoth_moda2.yaml)

## Hardware Support

The default deploy config places both the AR and DiT stages on one GPU
(`devices: "0"`). Its committed `gpu_memory_utilization` split is 0.5 for
stage 0 and 0.3 for stage 1. The A800 validation section below also shows a
two-GPU placement with one stage per GPU for attributable timing and memory;
the measured results are summarized below.

## GPU

### Optional FP8 AR KV cache

For CUDA deployments, `mammoth_moda2_fp8_kv.yaml` is an opt-in preset that
keeps the Stage 0 AR KV cache of decoder layer 0 in BF16 and stores the other
27 layers as FP8 E4M3 (`kv_cache_dtype_skip_layers: ["0"]`; write the layer
indices as quoted strings). Stage 1 remains on
`kv_cache_dtype=auto`; its DiT execution is unaffected. This setting quantizes
only the autoregressive KV cache. It is neither FP8 weight/activation
quantization nor vLLM-Omni diffusion KV-cache quantization.

Use the preset in place of the default deploy config:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model ./MammothModa2-Preview \
  --deploy-config vllm_omni/deploy/mammoth_moda2_fp8_kv.yaml \
  --prompt "A stylish woman riding a motorcycle in NYC, movie poster style" \
  --height 1024 \
  --width 1024 \
  --seed 42 \
  --extra-body '{"text_guidance_scale": 4.0, "cfg_range": [0.0, 1.0], "num_inference_steps": 50}' \
  --output mammoth_t2i.png
```

The preset was first validated on one NVIDIA H800 80GB with CUDA and
FlashAttention 3, with every layer in FP8. The native and FP8 runs used the
same model, code revision, and downstream configuration.

| Metric | Native BF16 (`kv_cache_dtype=auto`) | FP8 E4M3 (`fp8_e4m3`) | Change |
| --- | ---: | ---: | ---: |
| KV cache memory | 15.59 GiB | 15.55 GiB | -0.04 GiB |
| GPU KV cache size | 145,904 tokens | 291,232 tokens | +99.6% |
| Maximum concurrency at 8,192 tokens | 17.81x | 35.55x | +99.6% |
| Steady-state AR median | 71.191 s | 79.816 s | +12.1% |
| Steady-state end-to-end median | 83.840 s | 92.473 s | +10.3% |
| Steady-state DiT median | 12.574 s | 12.561 s | effectively unchanged |

The FP8 run used `kv_cache_dtype=fp8_e4m3` only for Stage 0; Stage 1 used
`kv_cache_dtype=auto`. The nearly unchanged reserved cache memory holds almost
twice as many tokens because FP8 reduces the bytes per cached token. This is a
capacity/concurrency tradeoff: the measured AR and end-to-end latencies were
higher than the native-BF16 baseline.

On H800, a 1024x1024 fixed-seed smoke test with every layer in FP8 completed
successfully with no obvious visual failure. FP8 is lossy, so numerical or
image-quality equivalence with BF16 is not implied.

On one A800 80GB (vLLM 0.30.0), FP8 KV layers run on FlashInfer and BF16 layers
on FlashAttention 2. Text-to-image at 1024x1024, 50 steps, guidance 4.0, seeds 42
and 1-5, with a studio tabby cat prompt and a peephole-view Samoyed prompt;
an image counts when it shows the prompted subject and scene.

| Stage 0 KV cache | GPU KV cache size | Cat images that follow the prompt | Samoyed images that follow the prompt |
| --- | ---: | ---: | ---: |
| BF16 on all 28 layers (`auto`) | 147,408 tokens | 6/6 | 6/6 |
| FP8 E4M3 on all 28 layers | 294,816 tokens | 1/6 | 0/6 |
| Layer 0 BF16, other 27 layers FP8 E4M3 (this preset) | 284,640 tokens | 6/6 | 6/6 |

The KV cache sizes above still count the 28 attention layers of the replaced
Qwen-VL language model, which #8095 removes: with it, BF16 goes from 147,408 to
294,816 tokens and all-FP8 from 294,816 to 589,632. The prompt-following
columns do not depend on it.

With all 28 layers in FP8, most cat images become a framed print on a wall.
Keeping layer 0 in BF16 restores them; keeping only layer 27, which has the
largest key magnitude, does not. The H800 numbers above were measured with all 28
layers in FP8; this preset has been run on A800 only.

### 1x L40S 48GB

> **48 GB config adjustment:** the committed
> `vllm_omni/deploy/mammoth_moda2.yaml` uses
> `gpu_memory_utilization` 0.5 / 0.3 (sized for ~80 GB). To fit on a 48 GB L40S,
> set the stage-0 (AR) value to `0.8` and the stage-1 (DiT) value to `0.16`
> before running. (On an ~80 GB GPU, leave the defaults unchanged.)

### 1x NVIDIA A800 80GB

#### Environment

- OS: Linux
- Python: Match the repository requirements for your checkout
- Driver / runtime: NVIDIA CUDA environment with one A800 80 GB
- vLLM version: Match the repository requirements for your checkout
- vLLM-Omni version or commit: Use the commit you are deploying from

#### Offline Commands

Download the model:

```bash
hf download bytedance-research/MammothModa2-Preview --local-dir ./MammothModa2-Preview
```

Run text-to-image with the shared offline example from the repository root. The
deploy config sets `trust_remote_code`, so no extra flag is needed:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model ./MammothModa2-Preview \
  --deploy-config vllm_omni/deploy/mammoth_moda2.yaml \
  --prompt "A stylish woman riding a motorcycle in NYC, movie poster style" \
  --height 1024 \
  --width 1024 \
  --seed 42 \
  --guidance-scale 4.0 \
  --num-inference-steps 50 \
  --extra-body '{"cfg_range": [0.0, 1.0]}' \
  --output mammoth_t2i.png
```

The standard diffusion request fields are `height`, `width`, `seed`,
`guidance_scale`, and `num_inference_steps`; use their corresponding CLI flags
shown above. `--height` and `--width` must be multiples of 16.

`cfg_range` is the only recommended MammothModa2 field in `--extra-body`; it
sets the relative step range `[start, end]` over which CFG is applied (default
`[0.0, 1.0]`). For compatibility, `text_guidance_scale` and
`num_inference_steps` remain accepted `extra_body` aliases and, when non-null,
take precedence over the standard request fields. Model extras are filtered
against the declared `extra_body_params`.

TeaCache can be enabled for the DiT stage with the same user-facing sampling
parameters:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model ./MammothModa2-Preview \
  --deploy-config vllm_omni/deploy/mammoth_moda2.yaml \
  --prompt "A stylish woman riding a motorcycle in NYC, movie poster style" \
  --height 1024 \
  --width 1024 \
  --guidance-scale 4.0 \
  --num-inference-steps 50 \
  --cache-backend tea_cache \
  --extra-body '{"cfg_range": [0.0, 1.0]}' \
  --output mammoth_t2i_teacache.png
```

The bundled TeaCache coefficients were fitted from MammothModa2 full-compute
traces. MammothModa2 uses the model-specific default `rel_l1_thresh=0.075`,
selected for the evaluated 1024x1024, 50-step configuration.

The model-specific keys are declared in
[`vllm_omni/model_extras/mammothmodal2_preview.py`](../../vllm_omni/model_extras/mammothmodal2_preview.py)),
so unknown MammothModa2 extras may be dropped.

Run text-to-text through the shared understanding example. It recognizes the
MammothModa2 checkpoint and automatically selects `mammoth_moda2_ar.yaml`:

```bash
python examples/offline_inference/x_to_text/x_to_text.py \
  --model ./MammothModa2-Preview \
  --prompt "Explain multimodal generation in three sentences."
```

Add an image for image-to-text or image summarization. The shared example
uses MammothModa2's chat and vision-token template:

```bash
python examples/offline_inference/x_to_text/x_to_text.py \
  --model ./MammothModa2-Preview \
  --image /path/to/input.jpg \
  --prompt "Please summarize the content of this image."
```

#### Verification

The example writes the generated image to the `--output` path. Confirm the file
exists and is a valid image:

```bash
ls -lh mammoth_t2i.png
python -c "from PIL import Image; print(Image.open('mammoth_t2i.png').size)"
```

### 2x NVIDIA A800 80GB validation

Use one A800 per stage so AR and DiT memory and timing are attributable. The
per-stage override changes placement only; both stages remain single-rank.

```bash
VLLM_LOGGING_LEVEL=DEBUG vllm serve ./MammothModa2-Preview --omni \
  --deploy-config vllm_omni/deploy/mammoth_moda2.yaml \
  --stage-overrides '{"0":{"devices":"0"},"1":{"devices":"1"}}' \
  --port 8099 \
  --log-stats
```

Startup logs should identify stage 1 as `StageDiffusionClient` and resolve it
to `MammothModa2DiTPipeline`. `DiffusionEngine` step timing is a DEBUG-level,
per-request message, so it appears only after sending a text-to-image request
with `VLLM_LOGGING_LEVEL=DEBUG`; it is not a startup marker. Seeing the legacy
generation model runner for stage 1 is a failed migration.

#### Experimental two-rank VAE patch decode

To split the DiT stage's tiled VAE decode across two GPUs, keep AR on a
separate GPU and configure stage 1 as a two-rank group (at least three GPUs
total):

```bash
vllm serve ./MammothModa2-Preview --omni \
  --deploy-config vllm_omni/deploy/mammoth_moda2.yaml \
  --stage-overrides '{"0":{"devices":"0"},"1":{"devices":"1,2","ulysses_degree":2,"vae_patch_parallel_size":2}}' \
  --port 8099
```

Mammoth's DiT attention is replicated, not sequence-sharded: the two-rank
group is used to coordinate VAE tiles. The registry enables VAE tiling
automatically when `vae_patch_parallel_size=2`. The spatial VAE path uses
`vae_parallel_mode: tile`; `spatial_shard_height` and `spatial_shard_width`
are unsupported for this VAE and raise an error. `vae_use_slicing: true`
also applies to `gen_vae`, including single-rank slicing-only configurations.
With distributed spatial decode, slicing adds a batch coordinate to each
tile/patch task. Each decoder input has a single batch row, and merge
restores the original image order.

The VAE-only path was measured
on two RTX 3090s; this three-GPU AR→DiT deployment has not been validated
end-to-end and needs sufficient memory for a full DiT copy on each stage-1
GPU. Compare it against a one-rank stage-1 run on the same checkpoint, prompt,
seed, and image size before drawing request-level performance conclusions.

#### Migration benchmark

The request-mode migration was checked on 2x NVIDIA A800 80GB PCIe with AR on
GPU 0 and DiT on GPU 1. Each revision ran one warmup followed by 10 serial
measured requests in the same initialized process. Both used BF16 eager mode,
1024x1024 output, 50 denoising steps, guidance scale 4.0, seed 42, and no
diffusion cache. The baseline was the pre-migration revision `caed3061`; the
candidate was `19de562a`. Lower latency is better.

| Metric | Baseline p50 | Baseline p95 | Candidate p50 | Candidate p95 |
| --- | ---: | ---: | ---: | ---: |
| End-to-end latency | 105.36 s | 106.01 s | 104.94 s | 105.79 s |
| AR stage latency | 86.97 s | 87.61 s | 86.26 s | 87.11 s |
| DiT stage latency | 18.27 s | 18.41 s | 18.60 s | 18.62 s |

Peak sampled device memory was 39,209 MiB on the AR GPU for both revisions.
The DiT GPU used 11,089 MiB for the baseline and 10,967 MiB for the candidate.
The candidate's shared runtime reported 372.02 ms p50 per denoising step and a
5.89 ms p50 AR-to-diffusion adapter time. All measured requests completed and
both revisions produced valid, prompt-aligned 1024x1024 RGB images. The small
latency differences are regression evidence, not a statistically significant
speedup claim.

### VAE decode memory options (slicing / tiling)

The DiT stage decodes latents with its own `gen_vae` (`AutoencoderKL`), which supports the diffusers slicing and tiling memory modes. Both are off by default and are enabled per deployment through the DiT stage's standard `vae_use_slicing` / `vae_use_tiling` fields:

```yaml
stages:
  - stage_id: 1
    vae_use_slicing: true   # decode the latent in slices instead of at once
    vae_use_tiling: true    # decode the latent tile by tile
```

Notes:

- These are capacity options: they bound VAE-decode peak memory for memory-constrained or high-resolution workloads and may increase decode latency. Measure both before enabling them in production.
- Tiling geometry comes from the checkpoint's VAE config (`sample_size`, `tile_sample_min_size`). A resolution below the tiling threshold decodes in a single tile: the mode is enabled but not exercised.
- VAE slicing splits the decode along the batch dimension. A request always decodes one image — the pipeline rejects `num_outputs_per_prompt != 1` — so the batch slicing bounds is the one that forms when several single-image requests decode together (`max_num_seqs` above). Under step execution (`mammoth_moda2_step.yaml`) the DiT decodes each request's latents on its own in `post_decode`, so slicing has no effect there. The decode sweep below measures that batched axis, batch 1-4.

To measure the modes on a target card without the AR stage, use `benchmarks/diffusion/bench_mammoth_moda2_vae_decode.py`: it decodes with the checkpoint's real `gen_vae` weights and reports decode latency (mean with the min-max of the measured decodes), peak memory and output deviation per resolution, batch size and mode.

#### Measured VAE decode, batch 1-4 (RTX PRO 6000 Blackwell 96 GB)

The current-source measurement of the modes, on the axis slicing acts on: a
serving batch of single-image requests decodes together (the DiT stage's
`max_num_seqs` is 8).

```bash
python benchmarks/diffusion/bench_mammoth_moda2_vae_decode.py \
  --model ./MammothModa2-Preview --sizes 1024,1536,2048,3072 --batches 1,2,4 \
  --modes baseline,slicing,tiling,slicing+tiling
```

##### Environment

- Source: branch head `dc4f490e6`, extracted from git and loaded with
  `PYTHONPATH` (the container's installed `vllm_omni` predates this change)
- Container image `vllm/vllm-omni:nightly`; Python 3.12; PyTorch 2.13.0+cu130;
  vLLM 0.30.0; transformers 5.14.1; diffusers 0.40.0
- GPU: one NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,251 MiB
  (96,329 MiB free before the sweep)
- VAE weights read from the checkpoint shards (`gen_vae.*`), bf16
- 2 warmup + 5 measured decodes per row, `seed=42`; the latency figures are the
  mean with the min-max of the five decodes
- Tiling geometry: `tile_sample_min_size=1024`, `tile_latent_min_size=128`,
  `tile_overlap_factor=0.25`, scale 8 — 1024x1024 is below the threshold, so
  tiling only engages at 1536 and above

Peak decode memory (MiB), the reason to enable the modes:

| Size | Batch | baseline | slicing | tiling | slicing + tiling |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1024x1024 | 1 | 2,638 | 2,638 | 2,638 | 2,638 |
| 1024x1024 | 2 | 4,307 | 2,650 | 4,307 | 2,650 |
| 1024x1024 | 4 | 8,416 | 2,675 | 8,416 | 2,675 |
| 1536x1536 | 1 | 5,687 | 5,687 | 2,647 | 2,647 |
| 1536x1536 | 2 | 9,443 | 5,714 | 4,323 | 2,674 |
| 1536x1536 | 4 | 16,960 | 5,770 | 8,448 | 2,730 |
| 2048x2048 | 1 | 9,953 | 9,953 | 2,678 | 2,678 |
| 2048x2048 | 2 | 16,634 | 10,003 | 4,388 | 2,729 |
| 2048x2048 | 4 | 28,973 | 10,103 | 8,578 | 2,828 |
| 3072x3072 | 1 | 22,146 | 22,146 | 2,748 | 2,748 |
| 3072x3072 | 2 | 44,092 | 22,258 | 4,525 | 2,860 |
| 3072x3072 | 4 | 73,944 | 22,483 | 8,852 | 3,085 |

Decode latency (ms), mean with the min-max of the five decodes:

| Size | Batch | baseline | slicing | tiling | slicing + tiling |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1024x1024 | 1 | 101.6 (101.6-101.6) | 101.7 (101.6-101.9) | 101.7 (101.6-101.7) | 101.6 (101.6-101.7) |
| 1024x1024 | 2 | 200.5 (200.4-200.6) | 204.1 (204.0-204.4) | 200.5 (200.4-200.5) | 203.8 (203.8-203.9) |
| 1024x1024 | 4 | 359.3 (359.1-359.4) | 407.3 (407.2-407.4) | 359.4 (359.3-359.5) | 407.5 (407.4-407.9) |
| 1536x1536 | 1 | 256.9 (256.8-257.2) | 256.8 (256.8-256.9) | 345.1 (343.9-346.0) | 345.3 (344.7-345.6) |
| 1536x1536 | 2 | 570.4 (569.7-571.0) | 515.3 (515.2-515.4) | 631.3 (630.6-631.7) | 689.9 (689.6-690.5) |
| 1536x1536 | 4 | 1,049.0 (1,048.2-1,049.5) | 1,029.9 (1,029.8-1,030.2) | 1,099.8 (1,099.4-1,100.3) | 1,379.6 (1,378.6-1,380.3) |
| 2048x2048 | 1 | 488.1 (487.9-488.3) | 488.3 (488.2-488.5) | 745.7 (744.9-747.2) | 745.2 (744.7-746.0) |
| 2048x2048 | 2 | 1,134.7 (1,134.1-1,134.9) | 978.1 (977.9-978.2) | 1,346.6 (1,346.0-1,347.1) | 1,488.8 (1,486.3-1,490.8) |
| 2048x2048 | 4 | 2,107.6 (2,107.3-2,108.1) | 1,955.9 (1,955.8-1,956.1) | 2,317.9 (2,316.4-2,319.6) | 2,988.5 (2,981.2-2,991.7) |
| 3072x3072 | 1 | 1,366.0 (1,365.8-1,366.2) | 1,366.6 (1,366.3-1,367.2) | 1,713.4 (1,711.1-1,716.5) | 1,714.3 (1,710.3-1,719.6) |
| 3072x3072 | 2 | 2,697.1 (2,696.6-2,697.8) | 2,731.4 (2,731.1-2,731.7) | 3,054.0 (3,051.8-3,055.8) | 3,420.7 (3,417.5-3,424.4) |
| 3072x3072 | 4 | 40,652.0 (30,302.4-43,240.7) | 5,467.1 (5,466.4-5,468.2) | 5,256.3 (5,253.5-5,260.9) | 6,837.0 (6,828.9-6,848.3) |

Deviation from the untiled baseline at the same size and batch: slicing is
byte-identical in every row except 3072x3072 batch 4 (68.29 dB PSNR, max abs
diff 0.0156); tiling is byte-identical below the threshold and 53.5 - 55.8 dB
above it (0.074 - 0.112).

What the sweep shows:

- **Slicing is flat in batch, at a small latency cost.** The untiled peak grows
  2,638 -> 8,416 MiB from batch 1 to 4 at 1024x1024 while sliced decode stays at
  2,650 - 2,675 MiB; from batch 2 up it is also the faster of the two at some
  sizes (1,029.9 against 1,049.0 ms at 1536 batch 4, 1,955.9 against 2,107.6 at
  2048 batch 4). At batch 1 it changes nothing, which is why the end-to-end
  table below cannot see it.
- **Tiling is flat in resolution, and it is a memory-for-time trade.** It caps
  the spatial extent at one tile, so peak stays at 2,647 - 8,852 MiB across the
  sweep where untiled reaches 73,944 MiB; where untiled still fits comfortably
  it costs latency (745.7 against 488.1 ms at 2048 batch 1, 345.1 against 256.9
  at 1536 batch 1) and on a card with headroom it is not a speed-up.
- **The two compose.** Slicing + tiling holds 2,638 - 3,085 MiB in all 12 rows
  (four sizes × three batch sizes), including 3072x3072 batch 4, where untiled
  decode needs 73,944 MiB.
- **The one row with a wide spread is the one under memory pressure.**
  3072x3072 batch 4 untiled reads 40,652.0 ms with a 30,302.4 - 43,240.7 ms
  range — its ~38.6 GB allocations are retried until they fit — while tiling the
  same row is 5,256.3 (5,253.5 - 5,260.9) ms. Every other row is tight.

#### Measured end-to-end (RTX PRO 6000 Blackwell 96 GB)

*Historical: the environment, protocol and tables directly below were measured
at `624ebea` with vLLM 0.29.0 and batch size 1. The re-measured current-source
data, including the batch > 1 rows, is at the end of this subsection.*

##### Environment

- OS: Ubuntu 24.04.3 LTS, Linux 7.0.0-30-generic, x86_64
- Container: `docker.m.daocloud.io/vllm/vllm-omni:nightly`
- Python: 3.12.3
- PyTorch: 2.13.0+cu130
- Driver / runtime: NVIDIA 595.84 / CUDA 13.2
- GPU: one NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB
- vLLM version: 0.29.0
- transformers: 5.14.1; diffusers: 0.40.0
- vLLM Omni version or commit: `624ebea19ec298d4f5332d9fb15cc7c5095df610`
- The measured code was loaded with `PYTHONPATH=/app/vllm_omni`. The container's
  installed `vllm_omni` package metadata is `0.29.0rc2.dev104+g21d86ec92`, which
  does not contain this change — without the override the run exercises the
  released pipeline and the flags never reach `gen_vae`.

##### Measurement protocol

Fixed prompt, `seed=42`, `text_guidance_scale=9.0`, `num_inference_steps=50`,
batch size 1, both stages on one device with the deploy config's
`gpu_memory_utilization` (0.5 / 0.3). End-to-end latency is the mean of five
requests after four warmups; the stage split is the per-stage wall time reported
under `--log-stats`; device peak is whole-device memory sampled every 0.5 s.

At 1536x1536 — above the tiling threshold, so tiling is actually exercised:

| Config | Stage 0 (AR) ms | Stage 1 (DiT + VAE) ms | End-to-end s | Device peak MiB | PSNR vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 202,346 | 24,423 | 225.9 | 67,354 | — |
| slicing | 196,867 | 24,379 | 221.3 | 67,354 | identical |
| tiling | 197,883 | 24,471 | 222.9 | **61,814** | 46.75 dB |
| slicing + tiling | 196,289 | 24,464 | 220.8 | **61,814** | 46.75 dB |

At 1024x1024 the latent (128) does not exceed `tile_latent_min_size`, so tiling
is enabled but decodes in a single tile: all four configs produce byte-identical
images and peak at 61,004 MiB.

What these measurements show:

- **Tiling bounds the device peak by 5.4 GiB at 1536x1536** (67,354 -> 61,814 MiB)
  once it engages, and is a no-op below the threshold. Tiled decode holds roughly
  one tile at a time while untiled decode holds the whole latent, so the saving
  grows with resolution.
- **Slicing alone changes nothing at batch size 1** — same peak, byte-identical
  output. It splits the decode along the batch dimension, so it is a batched-
  serving option; this table measures single requests.
- **End-to-end latency is a poor instrument for this option.** The AR stage is
  ~89% of the wall time and never touches the VAE; it drifts by more between runs
  (baseline 202.3 s vs slicing 196.9 s) than stage 1 varies across all four
  configs (24,379 - 24,471 ms, a 0.4% spread). Judge this option on device peak
  or a VAE-level decode benchmark, not on end-to-end timing.
- **Tiling introduces no visible seams.** Tiled output differs from baseline at
  46.75 dB PSNR (max 61/255, mean 0.8/255 over 85% of pixels). An 8x-amplified
  difference image traces image content — edges and contours — rather than the
  tile grid, and an autocorrelation test on the detrended row/column difference
  profiles finds no consistent periodic peak on both axes. Treat the difference
  as tiled-decode numerical noise.

##### Re-measured on the current source (with batch > 1)

Re-measurement on the same box for the merge head, with the same fixed prompt,
`seed=42`, 50 steps, `text_guidance_scale=9.0` and `cfg_range=[0, 1]`, loaded
with `PYTHONPATH`; vLLM 0.30.0 (required by the merge head; upgraded from
0.29.0), PyTorch 2.13.0+cu130. The batch-1 rows and every device peak come from
the 2026-10-05 sweeps, `6873f69b8` (batch 1) and `1c87b798d` (batch 4; only
benchmark files differ between the two), the batch-4 latency and stage rows are
the 2026-10-07 re-run at the scratch branch's `86a9d74dc`, whose model code is
identical to those revisions'.

Protocol: batch-1 cells are 4 warmups + 5 measured single requests, so n = 5 and
the tables carry the min-max of the five. Batch-4 cells measure whole waves of
four concurrent requests, wave wall time divided by four: the 2026-10-05 sweep
gave each cell a single measured wave (n = 1, so no spread), and the re-run
gives each cell one warmup wave plus four measured waves (`--num-prompts 16`),
n = 4, reproducing the 2026-10-05 peaks in seven of eight cells (the exception,
1024 batch-4 baseline, read 570 MiB higher in one 0.5 s sampling window). The
re-run ran on a shared node and its AR stage read 1-3% slower than in the
October sweep (stage 1 flat within ±0.8%), so compare batch-1 and batch-4
latencies within their own run.

At 1536x1536, batch 1:

| Config | Stage 0 (AR) ms | Stage 1 (DiT + VAE) ms | End-to-end s (min-max) | Device peak MiB |
| --- | ---: | ---: | ---: | ---: |
| baseline | 193,030 | 24,643 | 217.8 (216.8-218.7) | 67,766 |
| slicing | 189,610 | 24,664 | 214.4 (212.9-215.3) | 67,766 |
| tiling | 193,797 | 24,781 | 218.7 (217.8-219.8) | **61,656** |
| slicing + tiling | 190,430 | 24,783 | 215.3 (214.0-216.7) | **61,654** |

At 1536x1536, batch 4:

| Config | Slicing | Tiling | Stage 0 (AR) ms | Stage 1 (DiT + VAE) ms | End-to-end s / image (min-max) | Device peak MiB |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| baseline | off | off | 209,407 | 87,188 | 79.4 (79.3-79.5) | 87,078 |
| slicing | on | off | 207,806 | 87,032 | 79.0 (78.8-79.1) | 67,200 |
| tiling | off | on | 213,104 | 87,200 | 80.3 (79.9-81.0) | 74,090 |
| slicing + tiling | on | on | 209,291 | 87,372 | 79.5 (78.9-80.2) | **66,020** |

Under a wave the two stage figures are per-request durations that overlap across
the four concurrent requests (in the batch-4 table they are the mean over the
sixteen measured requests of the four waves), so they do not add up to the
per-image end-to-end time (at batch 1 each row is one request and they do).

At 1024x1024 (below the tiling threshold, so tiling is enabled but decodes in
a single tile):

| Batch | Config | End-to-end s/image (min-max) | Device peak MiB |
| ---: | --- | ---: | ---: |
| 1 | baseline | 105.4 (104.8-105.9) | 60,844 |
| 1 | slicing | 105.7 (104.5-107.9) | 60,844 |
| 1 | tiling | 106.0 (104.3-107.7) | 60,844 |
| 1 | slicing + tiling | 106.5 (106.1-107.2) | 60,844 |
| 4 | baseline | 37.0 (36.8-37.1) | 69,688 |
| 4 | slicing | 36.8 (36.8-36.9) | 60,846 |
| 4 | tiling | 36.8 (36.6-37.1) | 69,688 |
| 4 | slicing + tiling | 36.6 (36.3-36.8) | 60,846 |

All four batch-1 configs produce byte-identical output; slicing bounds the
decode peak flat at batch > 1, tiling bounds it above the threshold only (at
1024x1024 it sits below the threshold and is a no-op), and the batch-4
latencies, 36.6 - 37.0 s per image, all overlap within their min-max ranges.

Isolated at equal concurrency and size (1536x1536 batch 4, against `baseline`),
tiling saves 12,988 MiB, slicing 19,878 MiB and the two together 21,058 MiB; the
untiled batch-4 peak grows by 19,312 MiB over the single-image baseline, which
is the growth slicing removes. Latency barely discriminates between the four
configs (79.0 - 80.3 s per image at 1536, 36.6 - 37.0 s at 1024 — a couple of
percent, tiling at the slow end at 1536) because the AR stage dominates and
never touches the VAE.

### 1x AMD MI300X, MammothModa2 Preview (pre-migration baseline)

#### Environment

- OS: Linux 6.8.0-134-generic, x86_64
- Container: official ROCm image built from `docker/Dockerfile.rocm`
- Python: 3.12.13
- PyTorch: 2.11.0+gitd0c8b1f
- Driver / runtime: AMD 6.19.14.31400000 / ROCm 7.2.53211
- GPU: one AMD Instinct MI300X, `gfx942:sramecc+:xnack-`, 191.69 GiB visible HBM
- vLLM version: 0.27.0+rocm723
- vLLM Omni version or commit: `73e1368c7bb940efe1a025859c9d6c8eeeb2e3f0`
- Installed vLLM Omni package metadata: `0.27.0rc2.dev44+g55abdade9.rocm`

#### Offline Commands

The checked run used the committed stage split, with `gpu_memory_utilization` set to 0.5 for AR and 0.3 for DiT:

```bash
python3 examples/offline_inference/text_to_image/text_to_image.py \
    --model bytedance-research/MammothModa2-Preview \
    --deploy-config vllm_omni/deploy/mammoth_moda2.yaml \
    --prompt "A stylish woman riding a motorcycle in NYC, movie poster style" \
    --height 1024 \
    --width 1024 \
    --seed 42 \
    --extra-body '{"text_guidance_scale": 4.0, "cfg_range": [0.0, 1.0], "num_inference_steps": 50}' \
    --enable-diffusion-pipeline-profiler \
    --log-stats \
    --output mammoth_t2i.png
```

#### Verification

The first request took 85.224 seconds. The AR stage generated 4,161 visual tokens in 72.996 seconds, and the DiT stage took 12.163 seconds. AR weight loading used 21.4 GiB and took 8.250 seconds. DiT weight loading used 5.49 GiB and took 1.824 seconds. The largest one second whole device memory sample was 106.57 GiB, including the AR KV cache reserved by the 0.5 memory setting.

### 1x A800 80GB, MammothModa2 Preview (serving)

#### Environment

- OS: Linux, x86_64
- CPU: 18 cores
- GPU: one NVIDIA A800-SXM4-80GB
- Deploy config: `vllm_omni/deploy/mammoth_moda2.yaml` (default split: AR `gpu_memory_utilization` 0.5, DiT 0.3)
- vLLM-Omni version or commit: use the commit you are deploying from

#### Serving Commands

```bash
vllm-omni serve bytedance-research/MammothModa2-Preview --omni \
    --port 8091 \
    --trust-remote-code

python benchmarks/diffusion/diffusion_benchmark_serving.py \
    --model bytedance-research/MammothModa2-Preview \
    --endpoint /v1/images/generations \
    --host 127.0.0.1 --port 8091 \
    --height 1024 --width 1024 --num-inference-steps 50 \
    --extra-body '{"text_guidance_scale": 9.0, "cfg_range": [0.0, 1.0]}' \
    --num-prompts 8 --seed 142 --warmup-requests 0 \
    --output-file mm2_1024_s50_c1.json
```

#### Verification

On 1024x1024 / 50 steps, single-concurrency end-to-end mean latency is ~96 s
(P99 ~101 s) per image and concurrency-4 mean is ~151 s (~0.024 img/s);
steady-state combined GPU memory is ~50.4 GiB (AR ~39.2 GiB, DiT ~11.1 GiB).
The AR stage dominates (~77 s of a ~96 s request) because it decodes a fixed
4,161-token visual grid per image, so latency is insensitive to prompt length
and scales only partially with DiT step count. See the
[serving performance dashboard](../../benchmarks/diffusion/performance_dashboard/mammoth_moda2_serving_performance.md)
for the full sweep, peak-memory, and component-attribution data.

The output was a valid 1024 by 1024 RGB PNG.

### Startup benchmark

The [startup benchmark](../../benchmarks/mammoth_moda2/README.md) provides
commands, measurement boundaries and a single-A800 eager baseline. On that
16-CPU-quota machine, parallel stage initialization and 16 CPU threads reduced
`Omni()` initialization from about 60 s to 37 s. See the benchmark for the
configuration and limitations; CUDA-graph mode needs separate memory and
quality validation.

## MammothModa2-Dev unified inference

MammothModa2-Dev uses a Qwen3-VL AR backbone, while MammothModa2-Preview uses
Qwen2.5-VL. vLLM-Omni selects the matching implementation from the nested
`llm_config.model_type`; no checkpoint edits or `trust_remote_code` flag are
required.

Text-to-text and image-to-text use the AR-only deploy. Text-to-image loads the
Qwen3 generation experts (`gen_mlp`), extra visual vocabulary and image head,
then sends the generated visual tokens and hidden states to the DiT stage.

Download the checkpoint:

```bash
hf download bytedance-research/MammothModa2-Dev --local-dir ./MammothModa2-Dev
```

Run text-to-text through the shared understanding example. It recognizes the
Dev checkpoint as MammothModa2 and automatically selects
`mammoth_moda2_ar.yaml`:

```bash
python examples/offline_inference/x_to_text/x_to_text.py \
  --model ./MammothModa2-Dev \
  --prompt "Explain multimodal generation in three sentences."
```

Add an image for image-to-text or image summarization:

```bash
python examples/offline_inference/x_to_text/x_to_text.py \
  --model ./MammothModa2-Dev \
  --image ./image.png \
  --prompt "Please summarize the content of this image."
```

The Dev checkpoint is approximately 47.55 GiB on disk. In the verified AR-only
run, loaded model weights used approximately 16.97 GiB of GPU memory before KV
and encoder caches. Allow additional GPU memory for those caches and the input
image.
