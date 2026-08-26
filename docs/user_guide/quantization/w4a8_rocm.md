# W4A8 Quantization on AMD MI355X

## Overview

W4A8 quantizes diffusion transformer weights to MXFP4 (`float4_e2m1fn_x2`,
groups of 32 K-dimension elements sharing one `float8_e8m0fnu` exponent) and
quantizes activations to MXFP8 **dynamically inside the kernel**. Nothing about
the activation side is stored in the checkpoint.

The GEMM comes from AMD Quark's FlyDSL kernels and needs CDNA4 scaled MFMA, so
it runs on **gfx950 (MI355X) only**. CDNA3 (gfx942 / MI325X) has neither scaled
MFMA nor native FP4.

Three accuracy tiers across two `--quantization` methods (all one kernel launch):

| Tier | `--quantization` | Correction branch | Offline setup |
| --- | --- | --- | --- |
| **plain W4A8 (RTN)** | `quark_w4a8` | none | none |
| **online W4A8 SVD** | `quark_svdquant` | low-rank, derived at load (uncalibrated) | none |
| **calibrated W4A8 SVD** | `quark_svdquant` | low-rank, activation-calibrated | one export |

### How each tier works

All three share one runtime: at inference the BF16 activation is quantized to
**MXFP8 inside the kernel**, then multiplied against the **MXFP4 weight** on the
CDNA4 scaled-MFMA GEMM — one kernel launch, BF16 out plus bias. They differ only
in how the weight (and, for SVD, the correction) is prepared:

- **plain W4A8 (RTN)** — each BF16 weight is round-to-nearest quantized to MXFP4
  and packed at load; the BF16 copy is freed. No correction branch:
  `y = Q(x) @ Q(W).T + bias`.
- **online W4A8 SVD** — same, plus a rank-R branch derived from the weight *at
  load* with `torch.svd_lowrank`. Only the residual `Wr = W - L2·L1` is 4-bit;
  `L1`/`L2` stay BF16 and `d @ L2.T` is fused into the GEMM epilogue. Zero setup,
  but uncalibrated (sees the weight only).
- **calibrated W4A8 SVD** — the residual and `L1`/`L2` come from an offline Quark
  calibration (SmoothQuant + exact SVD on the smoothed weight) and are read from
  the checkpoint. Same runtime as online SVD, higher accuracy.

The SVD tiers are the **same method** (`quark_svdquant`); the checkpoint
(`is_checkpoint_w4a8_serialized`) picks online vs calibrated. Their low-rank branch
absorbs the weight's dominant singular directions, leaving a residual with a
narrower dynamic range that quantizes more accurately.

**plain W4A8 (RTN)** and **online W4A8 SVD** read a stock BF16 checkpoint and do all
their work at load — no offline step. **calibrated W4A8 SVD** reads a serialized
checkpoint produced offline (see [Offline calibrated
export](#offline-calibrated-export)).

!!! warning "online W4A8 SVD is an uncalibrated weight SVD"
    The published SVDQuant method derives `L1` / `L2` from calibration
    activations, which also migrates activation outliers into the low-rank
    branch. **online W4A8 SVD** instead takes a randomized truncated SVD of the
    weight alone (`torch.svd_lowrank`), so you get the low-rank correction and
    none of the outlier migration. Treat its accuracy as a floor.

    **calibrated W4A8 SVD** (`is_checkpoint_w4a8_serialized`) closes that gap: it
    runs Quark's `SVDQuantProcessor` (SmoothQuant smoothing + exact SVD on the
    smoothed weight) and ships the calibrated factors in the checkpoint.

## Hardware Support

| Device | Support |
| --- | --- |
| AMD ROCm (gfx950 / MI355X) | ✅ |
| AMD ROCm (gfx942 / MI325X) | ❌ no CDNA4 scaled MFMA |
| NVIDIA / Intel XPU / Ascend NPU | ❌ |

An unsupported device raises `NotImplementedError` at model build. It does not
fall back to BF16 — a whole-model silent fallback looks exactly like a
successful quantized run.

## Requirements

- ROCm with a gfx950 device
- AMD Quark from PR
  [#6079](https://gitenterprise.xilinx.com/AMDNeuralOpt/Quark/pull/6079)
  (branch `xiaoyu/svd-quant-flydsl`), installed from source. It vendors the
  FlyDSL A8W4 kernels
  (`quark.torch.quantization.nn.modules.flydsl_a8w4_inference_linear`) and the
  SVDQuant algorithm; the released `amd-quark` wheel ships **neither**.
- `aiter` (used for the MXFP4 pack and the `(16, 16)` weight shuffle)

Quark is imported lazily: `import vllm_omni.quantization` does not pull it in.
It loads the first time a W4A8 linear layer is constructed.

## Configuration

No preprocessing step. BF16 weights are read from the stock checkpoint and
packed to MXFP4 one layer at a time as they load; the BF16 copy is freed
immediately, so peak load memory stays close to the quantized footprint.

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(model="Wan-AI/Wan2.2-TI2V-5B-Diffusers", quantization="quark_w4a8")
outputs = omni.generate(
    "A cat sitting on a windowsill",
    OmniDiffusionSamplingParams(num_inference_steps=50),
)
```

```bash
python text_to_video.py --model <your-model> --quantization quark_w4a8

vllm serve <your-model> --omni --quantization quark_w4a8
```

Ready-made deploy configs:

- `vllm_omni/deploy/wan2_2_ti2v_w4a8.yaml`
- `vllm_omni/deploy/wan2_2_t2v_a14b_w4a8.yaml`

`quark-w4a8` is accepted as an alias. The bare name `svdquant` is deliberately
not claimed; it belongs to Nunchaku upstream.

### Options

| Key | Default | Meaning |
| --- | --- | --- |
| `svd_rank` (alias `rank`) | `None`, or `32` under `quark_svdquant` | Rank of the low-rank correction branch. Absent selects the plain variant. |
| `ignored_layers` (alias `modules_to_not_convert`) | `[]` | Layer prefixes to keep in BF16. |

### Model support

| Model | `quark_w4a8` | Notes |
| --- | --- | --- |
| Wan2.2-TI2V-5B | ✅ | Single transformer |
| Wan2.2-T2V-A14B | ✅ | Dual-expert cascade; `transformer/` and `transformer_2/` both quantized |
| Wan2.2-I2V-A14B | ⭕ | Not validated; the image-conditioning linears have their own shapes |

## Offline calibrated export

**online W4A8 SVD** is convenient but sees only the raw weight. **calibrated W4A8
SVD** gets the published SVDQuant accuracy — activation-aware smoothing plus an
exact SVD — by exporting a checkpoint once with
`examples/quantization/export_quark_svdquant_w4a8.py` and serving it with
`is_checkpoint_w4a8_serialized`.

```bash
# One-time offline export (needs Quark + a gfx950 GPU; ~tens of minutes on A14B).
python examples/quantization/export_quark_svdquant_w4a8.py \
    --model /path/to/Wan2.2-TI2V-5B-Diffusers --variant svdquant \
    --svd_rank 32 --n_calib_prompts 2 --n_calib_steps 20 \
    --output_dir /path/to/wan5b-w4a8-svd
```

This runs Quark's `SVDQuantProcessor` (SmoothQuant smoothing folded into the
weights, then exact SVD on the smoothed weight) and writes, per transformer:

```text
<output_dir>/<comp>/diffusion_pytorch_model.safetensors
<output_dir>/<comp>/quant_config.json      # {"quantization_config": {...}}
```

where `<comp>` is `transformer` (plus `transformer_2` for the A14B cascade).
Weights are stored **unpacked BF16** (residual under `weight`, factors under
`proj_down`/`proj_up`) and packed to the MXFP4 kernel layout at load. Copy or
symlink the non-transformer pipeline components (`vae`, `text_encoder`,
`scheduler`, `model_index.json`) into `<output_dir>` to make it directly
loadable, then serve it exactly like a stock model:

```bash
vllm serve /path/to/wan5b-w4a8-svd --omni --quantization quark_w4a8
```

The `quantization_config` stanza carries `is_checkpoint_w4a8_serialized: true` and
`svd_rank`, so the loader selects the serialized SVD path automatically.

Notes:

- **Self-attention QKV is pre-fused in the exporter.** Wan fuses `to_q/k/v` into
  `to_qkv`, so the factors are emitted fused: `proj_down` stacked to rank
  `3 x svd_rank`, `proj_up` block-diagonal. The runtime needs no special handling.
- **`--variant plain`** writes a portable pre-quantized artifact, but it is the
  RTN tier — always uncalibrated, RTN-equivalent to the online plain path
  (smoothing folds back to identity without a low-rank branch). `--gptq` (which
  GPTQ-quantizes the SVD residual) applies only to `--variant svdquant`.
- **SVD convention.** Quark places all singular values in `proj_down` (`L1`) and
  keeps `proj_up` (`L2`) orthonormal — the opposite of the online path's
  `sqrt(S)` split. The loader consumes them verbatim; do not rebalance.
- **`--pack-format` (compact 4-bit formats).** By default (`bf16`) the residual is
  stored unpacked and packed at load. Two opt-in 4-bit formats are **~4× smaller**
  on disk (low-rank factors stay BF16); both couple the checkpoint to the kernel's
  pack version, so keep a BF16 copy as the archival format.
    - `--pack-format packed`: preshuffled into the kernel layout
      (`weight_shuffle`/`weight_scale`) — fastest load, but **TP=1 only** (the
      shuffle bakes in K/N).
    - `--pack-format unshuffled`: natural-order MXFP4 (`weight_packed`/
      `weight_scale`) that vLLM can shard for **TP>1**; each rank shuffles its
      shard at load. TP=1 output is bit-identical to `packed`.

## Layers that stay in BF16

Routing falls back to `UnquantizedLinearMethod` and logs a warning when a layer
cannot be tiled. The two variants have **different** limits:

| Variant | Requirement on `in_features` / `out_features` |
| --- | --- |
| `quark_w4a8` | `in_features` `>= 256` and a multiple of 256; `out_features` a multiple of 32 |
| `quark_svdquant` | both `>= 256` **and** a multiple of 256 |

`in_features` is the strict one even for the plain variant: it is the GEMM's K,
and Quark validates `K >= 256 and K % 256 == 0` outright, because below 256 the
MXFP4 packer emits an *unshuffled* layout that the preshuffle kernel cannot
read. `out_features` only has to divide the smallest `tile_n`, which is 32.

The SVD epilogue's 256 floor is stricter because of the preshuffled B layout.
Wan's `proj_out` (`out_features=192`) is the motivating case: Quark refuses it
rather than emitting garbage, so under `quark_svdquant` that layer runs in BF16
while under `quark_w4a8` it is quantized normally.

## Selecting the kernel provider

`VLLM_OMNI_SVDQUANT_PROVIDER` picks the backend:

| Value | Meaning |
| --- | --- |
| `auto` (default) | Prefer upstream FlyDSL, fall back to Quark's vendored kernels |
| `quark` | Force Quark's vendored kernels |
| `flydsl` | Force upstream FlyDSL — currently always fails; the released wheel is the compiler only, with no kernels |

An unrecognised value raises `ValueError`. A missing provider is reported as a
capability answer, not an error.

## Limitations

- **Tensor parallelism** requires an `--pack-format unshuffled` checkpoint (the
  preshuffled/BF16 paths raise on `tp_size > 1`). Both plain W4A8 and
  `quark_svdquant` support **column- and row-parallel**; the SVD low-rank term
  needs no extra collective — by linearity its per-rank partial `d_p @ proj_up.T`
  rides the layer's existing output all-reduce (`proj_up` shares the N axis and is
  sharded/replicated with it; `proj_down` shares K). The sharding math is verified
  on gfx950 (`test_flydsl_w4a8_tp_rocm.py` decomposition tests), but the end-to-end
  multi-GPU run is **unvalidated** (the dev box has one GPU) — run
  `test_svd_row_parallel_end_to_end_multigpu` on a ≥ 2-GPU server. TP=1 is
  bit-identical to the single-GPU path.
- **Load time.** `quark_svdquant` runs a randomized truncated SVD per layer at
  load. It is O(rank) work, not a full decomposition, but it is not free on a
  28B dual-expert model.
- **No MXFP4-packed checkpoints.** Serialized checkpoints store **unpacked BF16**
  weights (packed to the kernel layout at load), because the shuffled MXFP4
  layout is kernel-specific rather than a stable on-disk format. See
  [Offline calibrated export](#offline-calibrated-export).
- **BF16 activations only.** The kernel emits BF16 and quantizes its own inputs;
  accepting FP16 would silently round-trip through BF16.

## Accuracy

Measured on gfx950 at the Wan prefill token count (M = 4680, deliberately not
tile-aligned so the ragged-M padding path is exercised), cosine similarity of the
layer output against an unquantized BF16 reference:

| Shape (N x K) | `quark_w4a8` | `quark_svdquant`, rank 32 |
| --- | --- | --- |
| 3072 x 3072 | 0.9897 | 0.9962 |
| 5120 x 5120 | 0.9897 | 0.9962 |

The SVDQuant figures use a weight with a decaying singular spectrum, which is
what the low-rank branch is designed for. On i.i.d. Gaussian weights, where
there is no dominant subspace to extract, it offers no benefit.

Reproduce with:

```bash
pytest tests/diffusion/quantization/test_flydsl_w4a8_rocm.py
```

The test file gates on a runtime `gfx950` check as well as the `MI355` marker, so
it skips cleanly on other hardware.

### End to end

Wan2.2, t2v, 832x480, 49 frames, seed 42, prompt "a cat walks on the grass,
realistic style", against the BF16 pipeline:

| Model | Variant | Cosine | PSNR (dB) | Peak memory |
| --- | --- | --- | --- | --- |
| TI2V-5B, 30 steps | `quark_w4a8` | 0.932 | 14.4 | 55.1 GB vs 62.4 GB BF16 |
| TI2V-5B, 30 steps | `quark_svdquant` r32 | 0.945 | 15.5 | 55.5 GB |
| T2V-A14B, 20 steps | `quark_w4a8` | 0.885 | 14.5 | 37.3 GB vs 76.6 GB BF16 |
| T2V-A14B, 20 steps | `quark_svdquant` r32 | 0.879 | 13.5 | 38.1 GB |

**Read those numbers as drift, not as quality.** Every one of these outputs is a
sharp, prompt-faithful video; 4-bit weights perturb the sampler trajectory into a
*different* valid sample rather than degrading the current one, and per-pixel
metrics score that as catastrophic. Use them as a regression tripwire — a sudden
drop means something broke — and judge quality with a distributional benchmark
such as VBench.
