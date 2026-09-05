# Wan VAE Decoder Fast Path

Video pipelines built on the Wan 2.1/2.2 causal VAE (Cosmos3, Wan2.2, LingBot,
Helios, SANA-Video, LongCat-Video, DreamZero) spend a large share of their
end-to-end time decoding latents to pixels. The decoder is a chunked 3D
convolutional network whose reference implementation in diffusers moves every
activation through memory several times per layer (normalization, activation,
causal padding, feature-cache bookkeeping, shortcut upsampling). vLLM-Omni
installs a fast path on every loaded Wan VAE that fuses this data movement into
a handful of Triton kernels while leaving the convolutions themselves untouched.

## Levels

The fast path is controlled by `--vae-fast-path` (engine argument
`vae_fast_path`, deploy-config key `vae_fast_path`):

| Level | Default | Output vs. diffusers | What it does |
|---|---|---|---|
| `lossless` | yes | bit-identical | Fused RMSNorm epilogue, fused causal-conv input assembly and cache refresh, fused shortcut upsampling and residual adds (with the neighbouring convolution biases folded in), single-pass nearest 2x upsampling, preallocated output assembly. |
| `channels_last` | no | within tolerance (PSNR typically > 60 dB) | Everything in `lossless`, plus decoder convolution weights converted to channels-last memory format so cuDNN picks its channels-last kernels, and a single-pass channels-last RMSNorm+SiLU kernel that also absorbs the bias of the preceding `conv1`. |
| `off` | no | bit-identical | Reference diffusers decoder. |

```bash
# Default: bit-exact fast path
vllm serve nvidia/Cosmos3-Nano --omni

# Fastest: channels-last decoder (not bit-exact)
vllm serve nvidia/Cosmos3-Nano --omni --vae-fast-path channels_last

# Reference implementation, e.g. to bisect a quality issue
vllm serve nvidia/Cosmos3-Nano --omni --vae-fast-path off
```

The equivalent per-stage deploy configuration is:

```yaml
stages:
  - stage_id: 0
    vae_fast_path: channels_last
```

## Behavior and limitations

- Only CUDA is supported. On other platforms the reference decoder runs.
- The fast path is installed once per VAE instance when the pipeline is
  initialized. It rebinds the forwards of the loaded decoder modules; parameter
  names, `state_dict` keys and weight loading are unchanged.
- Every fused kernel validates its inputs and falls back to the exact PyTorch
  expression for anything it does not support (unusual dtypes or layouts, CPU
  tensors, autograd enabled, `torch.compile` tracing).
- The SiLU activation is folded into the normalization kernel only after an
  exhaustive self-test over all bf16/fp16 values proves the fused epilogue
  bit-identical to `F.silu` on the running toolkit; otherwise SiLU stays a
  separate operation. The startup log reports `fused_silu=...`.
- Spatially sharded VAE decode (`--vae-parallel-mode spatial_shard_height` or
  `spatial_shard_width`) is not combined with the fast path; the installer skips
  the VAE and logs why. Tiled and tile-parallel decode work unchanged.
- The `channels_last` level changes the order in which cuDNN accumulates
  convolutions, so outputs differ from the reference in the last bits. Use
  `lossless` when bitwise reproducibility against diffusers matters.
- For VAEs kept in fp32, cuDNN's channels-last convolution algorithms run in
  TF32 under PyTorch's default `torch.backends.cudnn.allow_tf32 = True`, which
  dominates the difference to the reference (about 1e-3). Set
  `torch.backends.cudnn.allow_tf32 = False` if full fp32 convolutions are
  required.

## Measured

Cosmos3-Nano VAE, 1280x720 x 189 frames, bf16, one GB200 GPU, `bench_wan_vae_decode.py`
(best of 2 runs after warmup):

| `--vae-fast-path` | Decode time | Speedup | Output vs. `off` |
|---|-------------|---------|---|
| `off` | 6.00 s      | 1.00x   | reference |
| `lossless` | 3.26 s      | 1.84x   | bit-identical |
| `channels_last` | 2.51 s      | 2.39x   | PSNR 62.6 dB, max abs diff 4.6e-2 |

Peak decode memory was unchanged (about 13 GiB): it is set by the largest
activations and the cuDNN workspace, not by the output assembly.

## Benchmarking

`benchmarks/diffusion/bench_wan_vae_decode.py` decodes seeded latents with the
real VAE at each level and reports decode time, peak memory, bitwise equality
and PSNR against the `off` level. `--profile` prints a per-kernel table with the
convolution share and any layout-transpose kernels:

```bash
python benchmarks/diffusion/bench_wan_vae_decode.py --model nvidia/Cosmos3-Nano \
    --size 1280x720 --frames 189 --fast-path off,lossless,channels_last --profile
```

The same script benchmarks multi-GPU VAE decode when launched with `torchrun`;
the decode is timed across all ranks and rank 0 reports:

```bash
torchrun --nproc-per-node 2 benchmarks/diffusion/bench_wan_vae_decode.py --model nvidia/Cosmos3-Nano \
    --vae-patch-parallel-size 2 --vae-parallel-mode tile
```
