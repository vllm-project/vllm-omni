# Kernel Benchmarks

Kernel-level micro-benchmarks and auto-tuners for custom operators in vLLM-Omni.

## mot_linear_benchmarks.py

Benchmark and auto-tune MoT (Mixture-of-Tokens) GEMM Triton kernels. MoT layers route text tokens and VAE (image) tokens to different weight matrices within the same linear layer. This script finds optimal Triton tile configurations for each (M, K, N) shape and saves them as JSON configs consumed at runtime by `vllm_omni.diffusion.layers.mot.ops.mot_gemm`.

### Prerequisites

- One or more NVIDIA GPUs
- Ray (for multi-GPU parallel tuning)
- A HuggingFace model config or local checkpoint (e.g., `ByteDance-Seed/BAGEL-7B-MoT`)

### Usage

```bash
# Auto-tune and save configs
python benchmarks/kernels/mot_linear_benchmarks.py \
    --model ByteDance-Seed/BAGEL-7B-MoT \
    --tp-size 1 --dtype w16a16 --tune \
    --save-dir vllm_omni/diffusion/layers/mot/configs/

# Benchmark only (measure with existing configs, no search)
python benchmarks/kernels/mot_linear_benchmarks.py \
    --model ByteDance-Seed/BAGEL-7B-MoT \
    --tp-size 1 --dtype w16a16
```

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | (required) | HuggingFace model name or local checkpoint path |
| `--tp-size` | `1` | Tensor parallel size |
| `--dtype` | `w16a16` | Weight/activation dtype (`w16a16`, `fp8_w8a8`, `int8_w8a16`) |
| `--batch-size` | `1 2 4 8 16` | Image counts to tune/benchmark |
| `--tune` | off | Enable auto-tuning mode |
| `--save-dir` | `./` | Directory to save tuned config JSON |
| `--seed` | `0` | Random seed |
| `--trust-remote-code` | off | Trust remote code when loading HF config |

### How it works

1. Extracts GEMM shapes (QKV, O, FFN gate+up, FFN down) from the model config, applying TP divisors
2. Computes the M dimension from batch size: `M = image_count * (VAE_CHUNK_SIZE + 2)`
3. Generates a pruned Triton tile search space (BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps, num_stages), filtering by SRAM capacity, register pressure, and SM occupancy
4. Distributes tuning across Ray GPU workers, each benchmarking configs via CUDA graphs
5. Saves per-(M, K, N) optimal configs as JSON with streaming checkpoints

### Output

JSON config file at `device_name=<gpu>,dtype=<dtype>.json`, consumed by `invoke_mot_gemm` at runtime.

## bench_attention_backends.py

Diagnostic benchmark for diffusion attention kernel backends. Exercises the same synthetic attention shape through each backend (SDPA, cuDNN, Flash Attention, FA4, FlashInfer, SageAttn3) with and without `attn_mask`, to identify backends that underperform on specific GPU SKUs.

### Usage

```bash
# Preset shapes matching real DiT models
python benchmarks/kernels/bench_attention_backends.py --preset hv15
python benchmarks/kernels/bench_attention_backends.py --preset wan22

# Custom shape
python benchmarks/kernels/bench_attention_backends.py \
    --batch 1 --heads 24 --seq 14336 --head-dim 128
```

### Optional dependencies

```bash
pip install --pre flash-attn-4    # FA4 (pre-release)
pip install -U flashinfer          # FlashInfer
```
