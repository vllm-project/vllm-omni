# Mage-Flow

> Text-to-image and multi-reference editing with tensor, CFG, and sequence parallelism

## Summary

- Vendor: Microsoft
- Model: `microsoft/Mage-Flow`, `microsoft/Mage-Flow-Turbo`, `microsoft/Mage-Flow-Edit` (Base / RL / Turbo variants)
- Task: Text-to-image generation and multi-reference image editing
- Mode: Online serving and offline inference, single or multi-GPU
- Maintainer: Community

## When to use this recipe

Use this recipe to pick a starting configuration for serving Mage-Flow, and to
choose between tensor, CFG, and sequence parallelism when more than one GPU is
available. The measured numbers below are the basis for those recommendations.

The upstream checkpoints are released for research purposes and are not
intended for product or service deployment.

## References

- Online serving guide:
  [`docs/user_guide/examples/online_serving/text_to_image.md`](../../docs/user_guide/examples/online_serving/text_to_image.md)
- Editing example:
  [`examples/offline_inference/image_to_image/image_edit.py`](../../examples/offline_inference/image_to_image/image_edit.py)
- Parallelism overview:
  [`docs/user_guide/diffusion/parallelism/overview.md`](../../docs/user_guide/diffusion/parallelism/overview.md)

## Hardware Support

Mage-Flow is a 4B NR-MMDiT. In BF16 the pipeline (transformer + Qwen3-VL text
encoder + VAE) needs roughly **19 GB** of device memory at 1024x1024, so a
single 24 GB card is the practical floor and anything at 40 GB or above is
comfortable.

This recipe documents CUDA configurations validated on 4x H20 (NVLink) and
2x RTX 5090 (PCIe, no NVLink).

## GPU

### 1x GPU (>= 24 GB)

#### Environment

- OS: Linux
- Python: 3.12
- Driver / runtime: NVIDIA CUDA 13.0, driver 580+
- vLLM version: 0.24.0
- vLLM-Omni version or commit: Use the commit you are deploying from

#### Command

```bash
vllm serve microsoft/Mage-Flow --omni --port 8091 --dtype bfloat16 \
  --max-num-seqs 2 --request-batch-max-wait-ms 20
```

Offline:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model microsoft/Mage-Flow-Turbo \
  --prompt "A serene mountain lake at sunrise" \
  --height 1024 --width 1024 --output out.png
```

#### Verification

```bash
curl -X POST http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model": "microsoft/Mage-Flow", "prompt": "a cup of coffee on the table"}'
```

#### Notes

- BF16 only. Quantization and LoRA are not supported.
- `Mage-Flow-Base` / `Mage-Flow` / `Mage-Flow-Turbo` default to 30 / 20 / 4
  denoising steps and CFG 5.0 / 5.0 / 1.0 when those values are omitted.
- Raise `max_num_seqs` only after measuring activation-memory headroom.

### 2x GPU

#### Command

CFG parallelism is the recommended two-GPU configuration — it splits the
positive and negative guidance branches across ranks and exchanges only one
prediction per denoising step:

```bash
vllm serve microsoft/Mage-Flow --omni --port 8091 --dtype bfloat16 \
  --cfg-parallel-size 2
```

Use tensor parallelism instead when the constraint is memory rather than
latency:

```bash
vllm serve microsoft/Mage-Flow --omni --port 8091 --dtype bfloat16 \
  --tensor-parallel-size 2
```

#### Benchmark

4x H20 (NVLink NV18) · `Mage-Flow-Turbo` · 1024x1024 · 4 steps · seed 42 ·
single request. SSIM/PSNR are measured against the single-GPU output at the
same guidance scale.

| Config | Latency | Speedup | Peak VRAM | SSIM | PSNR |
|---|---|---|---|---|---|
| 1 GPU (guidance 4.0) | 11.96 s | — | 18288 MB | — | — |
| TP=2 | 11.59 s | 1.03x | 15662 MB | 0.9921 | 37.89 dB |
| CFG=2 | 6.26 s | **1.91x** | 18014 MB | 0.9908 | 37.24 dB |

#### Notes

- **CFG parallelism requires guidance > 1.0.** With `Mage-Flow-Turbo` at its
  native guidance 1.0 there is no negative branch to split, and the flag has no
  effect. Use the `Mage-Flow` or `Mage-Flow-Base` checkpoints, or pass an
  explicit guidance scale.
- `cfg_parallel_size` must be 1 or 2 — there are exactly two branches to split.
- TP mainly buys memory (18288 -> 15662 MB) rather than latency. On a PCIe host
  without NVLink, TP=2 measured 0.94x, i.e. a regression: the per-layer
  all-reduce costs more than the sharded compute saves at this model size.

### 4x GPU

#### Command

Stack sequence parallelism on top of the guidance split:

```bash
vllm serve microsoft/Mage-Flow --omni --port 8091 --dtype bfloat16 \
  --cfg-parallel-size 2 --usp 2
```

Offline equivalent:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model microsoft/Mage-Flow --cfg-parallel-size 2 --ulysses-degree 2 \
  --prompt "A serene mountain lake at sunrise" --output out.png
```

#### Benchmark

Same setup as above. The sequence-parallel rows use guidance 1.0, so they are
compared against a guidance-1.0 single-GPU reference — mixing the two guidance
settings would conflate the guidance change with the parallelism effect.

| Config | Latency | Speedup | Peak VRAM | SSIM | PSNR |
|---|---|---|---|---|---|
| 1 GPU (guidance 4.0) | 11.96 s | — | 18288 MB | — | — |
| TP=4 | 11.87 s | 1.01x | **14348 MB** | 0.9911 | 37.68 dB |
| TP=2 x CFG=2 | 3.99 s | 3.00x | 15644 MB | 0.9893 | 36.22 dB |
| SP=2 x CFG=2 | 1.68 s | **7.12x** | 18136 MB | 0.9889 | 35.51 dB |
| 1 GPU (guidance 1.0) | 1.50 s | — | 18140 MB | — | — |
| SP=2 | 1.30 s | 1.15x | 18136 MB | **0.9969** | **46.71 dB** |
| SP=4 | 1.23 s | 1.22x | 18136 MB | 0.9968 | 46.49 dB |

#### Notes

- **Sequence parallelism cannot shard a padded batch.** Sharding splits the
  padded sequence blindly, leaving real tokens and filler on different ranks
  with no mask to separate them. Serving therefore requires `--max-num-seqs 1`
  under `--usp`, which is the default; raising it is refused at startup,
  because prompt length is not part of the request-batch key, so any two
  concurrent requests could land in one batch with differing token counts.
- **`--usp` alone is correct, but pairing it with `--cfg-parallel-size 2` is
  faster.** Packed CFG pads the shorter of the two guidance branches, which SP
  cannot shard, so under `--usp` a guided request runs its positive and
  negative branches as two sequential forwards per step instead. That is
  correct but does no guidance work in parallel. `--cfg-parallel-size 2` puts
  one branch on each rank and is the configuration measured above. The
  distinction only matters at guidance > 1: `Mage-Flow-Turbo` runs at guidance
  1.0 and has no negative branch either way.
- Combining dimensions beats the product of the parts (SP2 x CFG2 reaches
  7.12x). Attention is quadratic in sequence length, so sharding the sequence
  pays off super-linearly.
- SP tracks single-GPU output most closely (SSIM 0.997). Ulysses all-to-all
  only redistributes tokens and heads; unlike row-parallel splitting it leaves
  every GEMM's reduction order intact.
- SP combined with TP is untested. Ring attention (`--ring-degree`) is
  untested; only Ulysses was exercised.

## Choosing a configuration

| Goal | Use |
|---|---|
| Lowest latency, 2 GPUs | `--cfg-parallel-size 2` |
| Lowest latency, 4 GPUs | `--cfg-parallel-size 2 --usp 2` |
| Fit a memory-constrained GPU | `--tensor-parallel-size 2` or `4` |
| Highest fidelity to single-GPU output | `--usp N` |

Unsupported multi-GPU modes (pipeline parallelism, VAE patch parallelism,
HSDP) fail at startup with an explicit error rather than degrading silently, as
does sequence parallelism combined with `--max-num-seqs > 1`.

## Known limitations

### Content safety and provenance

The upstream reference implementation ships two governance mechanisms that this
integration does **not** provide:

- **Prompt and reference-image screening.** Upstream runs a fail-closed
  content-policy classifier on the same Qwen3-VL weights that produce the
  diffusion conditioning, covering sexual, hate, self-harm, violence,
  copyright, and public-figure categories; the editing path also inspects the
  reference images, where recognizing a real person or a copyrighted character
  is itself a violation. Upstream marks that gate mandatory with no opt-out and
  returns a blank refusal image when it fires. vLLM-Omni does not implement it:
  every prompt and reference image reaches the model.
- **Gaussian-Shading provenance watermark.** Upstream unconditionally replaces
  the initial noise with a distribution-preserving watermarked sample and ships
  a flow-ODE inversion detector for it. vLLM-Omni draws plain Gaussian noise,
  so outputs carry no provenance mark and upstream's detector will not identify
  them as Mage-Flow output.

Deployments are responsible for their own content moderation, and for their own
provenance or synthetic-content marking where that is required. This follows
upstream's own guidance that "downstream users are responsible for applying
additional safeguards" before broader use, alongside its statement that the
checkpoints are released for research rather than for product or service
deployment.

The weight repositories are gated. Accept the access terms on the model page
and set `HF_TOKEN` before running, and confirm there which license governs the
weights: the MIT license in the upstream GitHub repository covers the source
code, and the gate may present separate terms.

One practical consequence: because the watermark replaces the initial noise
rather than being applied to the finished image, a seeded vLLM-Omni run does
not reproduce the upstream image for the same seed. Compare the two
implementations by injecting an identical initial latent through `latents`
rather than by matching seeds.

### Other constraints

See the per-configuration **Notes** above for BF16-only operation, unsupported
quantization and LoRA, the `--max-num-seqs 1` requirement under sequence
parallelism, and the untested SP+TP and ring-attention combinations.

## A note on output differences

Mage-Flow is unusually sensitive to numerical perturbation: a 9.5e-07 float32
difference in the RoPE frequency table amplifies, through 12 double-stream
blocks x 4 turbo steps, into SSIM 0.986 in the final image. Bit-identical
output across parallel configurations is therefore not achievable, and the
SSIM values above (all >= 0.9889) reflect BF16 reduction-order effects rather
than logic errors. Runs are deterministic within a fixed configuration: the
same seed and flags reproduce byte-identical images.
