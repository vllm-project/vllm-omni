# Wan2.2 Text-to-Video

> Text-to-video serving (Wan2.2 14B), with optional Skip-Softmax sparse attention on Blackwell

## Summary

- Vendor: Wan-AI
- Model: `Wan-AI/Wan2.2-T2V-A14B-Diffusers`
- Task: Text-to-video generation
- Mode: Online serving with the OpenAI-compatible API (offline `Omni` also supported)
- Maintainer: Community

## When to use this recipe

Use this recipe to deploy the Wan2.2 14B text-to-video model with vLLM-Omni. The
standard plan is multi-card serving with sequence/CFG parallelism (below). On
**datacenter Blackwell** (sm_100 / sm_103) you can additionally enable
**Skip-Softmax** through the `TRTLLM_ATTN` backend for extra speed — see the
Blackwell section.

## References

- Upstream model card: <https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers>
- Attention backends guide: [`docs/user_guide/diffusion/attention_backends.md`](../../docs/user_guide/diffusion/attention_backends.md)
- Skip-Softmax design: [`docs/design/feature/skip_softmax.md`](../../docs/design/feature/skip_softmax.md)

## Hardware Support

## GPU

### 8x NVIDIA H20 / H100 / A100 (standard multi-card serving)

The recommended parallel strategy depends on whether the checkpoint needs
classifier-free guidance:

1. **Distilled model (no CFG)** — higher throughput; use when the checkpoint does
   not require negative-prompt computation.
2. **Official open-source model (with CFG)** — uses CFG parallelism to run the
   negative and positive samples for the original released weights.

#### Environment

- OS: Linux
- Python: 3.10+
- Driver / runtime: NVIDIA driver with a CUDA runtime supported by your PyTorch build
- vLLM version: Match the repository requirements for your checkout
- vLLM-Omni version or commit: Use the commit you are deploying from

#### Command

**Distilled model (no CFG, recommended for distilled checkpoints):**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --omni \
  --use-hsdp \
  --usp 8 \
  --vae-patch-parallel-size 8 \
  --vae-use-tiling
```

**Official open-source model (with CFG):**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --omni \
  --use-hsdp \
  --cfg-parallel-size 2 \
  --usp 4 \
  --vae-patch-parallel-size 8 \
  --vae-use-tiling
```

The official model splits the positive and negative CFG branches with
`--cfg-parallel-size 2`, keeping Ulysses sequence parallelism at `--usp 4` so the
two branches cover all 8 GPUs (`usp * cfg = 8`).

#### Verification

For online serving client examples and request formats, see
[`examples/online_serving/text_to_video`](../../examples/online_serving/text_to_video).

#### Notes

- **Key flags:**
    - `--omni` — enables vLLM-Omni diffusion serving.
    - `--use-hsdp` — Hybrid Sharded Data Parallelism for the 14B DiT weights.
    - `--usp <N>` — Unified (Ulysses) Sequence Parallelism degree.
    - `--cfg-parallel-size <N>` — CFG parallelism; set to 2 for the official model,
    omit for distilled checkpoints.
    - `--vae-patch-parallel-size 8` / `--vae-use-tiling` — parallel + tiled VAE
    decoding; disabling patch parallelism can significantly increase VAE latency.

## XPU

### 1x Intel Arc Pro B70 (32 GB)

Offline text-to-video at a reduced 576x320, 17-frame shape. Layerwise offload
streams the two 14B expert transformers; VAE slicing bounds the decode
activations.

#### Environment

- OS: Linux
- Python: 3.10+
- torch: 2.13.0+xpu
- vLLM: 0.29.0 (`98dff2a8`)
- vLLM-Omni: `main` at `4c7a98c2`

#### Command

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --prompt "A serene lakeside sunrise with mist over the water." \
  --negative-prompt "blurry, noisy, artifacts, distorted, low quality, streaks, sparkles, blocky, pixelated" \
  --height 320 --width 576 --num-frames 17 \
  --num-inference-steps 60 \
  --guidance-scale 4.0 \
  --flow-shift 5.0 \
  --boundary-ratio 0.875 \
  --enable-layerwise-offload \
  --vae-use-slicing \
  --enforce-eager \
  --seed 12345 \
  --output wan22_t2v_output.mp4
```

Given spare cards, `--enable-cpu-offload` shards the two expert transformers
instead of streaming them, which nearly halves the denoise loop. Four is the
fastest count; eight is slower.

```bash
  --tensor-parallel-size 4 \
  --enable-cpu-offload
```

| Offload | Cards | Peak VRAM | Denoise |
| --- | ---: | ---: | ---: |
| `--enable-layerwise-offload` | 1 | 9.9 GiB | 160 s |
| `--enable-cpu-offload` | 2 | 18.9 GiB | 113 s |
| `--enable-cpu-offload` | 4 | 12.4 GiB | 86 s |

#### Verification

Confirm `wan22_t2v_output.mp4` decodes and that frame 0 is clean.

#### Notes

- Key flags: `--boundary-ratio 0.875` sets the high-noise to low-noise expert
  switch. `--flow-shift 5.0` with `--guidance-scale 4.0` over 60 steps clears
  the frame-0 VAE corruption seen at the stock 50-step, `flow_shift=12.0`
  settings.
- Known limitations: only offline T2V at 576x320 was qualified. Larger shapes,
  TI2V, HSDP, CFG parallelism, and online serving are out of scope for this
  profile.

## Accelerating with TRTLLM_ATTN + Skip-Softmax (datacenter Blackwell)

On **datacenter Blackwell** (sm_100 / sm_103), the mask-free Wan pipeline
auto-routes to the `TRTLLM_ATTN` backend, which can enable **Skip-Softmax** — a
sparse-attention mode that trades a little fidelity for speed. This composes with
the multi-card plan above (Ulysses SP is compatible; ring SP is not).

### Environment

- Same as above, plus: datacenter Blackwell (sm_100 / sm_103), `head_dim == 128`,
  and `flashinfer` (already a hard dependency of vLLM — no extra install).

### Command

Dense BF16 (Skip-Softmax off — `TRTLLM_ATTN` is already the Blackwell auto-route
for mask-free Wan, so no flags are needed):

```bash
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --omni
```

With Skip-Softmax (calibrated curve — the `(a, b)` curve is read from the
checkpoint's `sparse_attention_config`; you only pick the operating point):

```bash
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --omni \
  --diffusion-attention-config '{"default": {"backend": "TRTLLM_ATTN",
      "skip_softmax": {"target_sparsity": 0.65, "disabled_until_timestep": 0.86}}}'
```

Offline, the same config is a typed object (or the equivalent dict) — values are
validated at construction:

```python
from vllm_omni import Omni
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, SkipSoftmaxSpec

llm = Omni(
    model="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    diffusion_attention_config=AttentionConfig(
        default=AttentionSpec(
            backend="TRTLLM_ATTN",
            skip_softmax=SkipSoftmaxSpec(target_sparsity=0.65, disabled_until_timestep=0.86),
        ),
    ),
)
```

Calibration-free path (no checkpoint curve needed — set an absolute threshold
instead of `target_sparsity`):

```bash
--diffusion-attention-config '{"default": {"backend": "TRTLLM_ATTN",
    "skip_softmax": {"threshold": 0.02, "disabled_until_timestep": 0.86}}}'
```

### Skip-Softmax controls

| Key | Valid values | Meaning |
| --- | --- | --- |
| `target_sparsity` | finite, `[0, 1]` | Operating point on the checkpoint's calibrated curve (`a·exp(b·s)`). |
| `threshold` | finite, `≥ 0` | Direct skip threshold, no calibration needed. Mutually exclusive with `target_sparsity`. |
| `disabled_until_timestep` | finite, `[0, 1]` | Keeps early, high-noise steps dense; skip turns on once the normalized timestep `t ≤ D`. |

**Start at `target_sparsity=0.65, disabled_until_timestep=0.86`** (near-lossless)
and raise `target_sparsity` only as far as your quality bar allows — the output
diverges from dense as it climbs. Gate the noisy early steps first (`D`), then
raise sparsity; the timestep gate buys more speed per unit of divergence than
sparsity does. The gain grows with sequence length, since Skip-Softmax only
accelerates attention.

Measured on B300 / SM103, 1280×720 / 81f / 50 steps, torch.compile, official
ModelOpt calibration (speedup and divergence relative to dense):

| config | speedup | LPIPS (divergence vs dense) |
| --- | --- | --- |
| dense | 1.000x | - |
| `s=0.65, D=0.86` | 1.062x | 0.041 |
| `s=0.75, D=1.00` | 1.181x | 0.377 |
| `s=1.00` | 1.202x | 0.597 |

### Notes

- **Requirements:** datacenter Blackwell (sm_100 / sm_103), `head_dim == 128`, and
  `flashinfer`. Elsewhere, or without FlashInfer, selecting `TRTLLM_ATTN` raises
  rather than silently degrading.
- **Known limitations:**
    - Skip-Softmax is not supported under ring sequence parallelism; use Ulysses SP
    (`--usp`), or run Skip-Softmax without ring.
    - `TRTLLM_ATTN` is BF16-only (no attention-level quantization).
