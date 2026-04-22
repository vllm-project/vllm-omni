# Wan2.2 Image To Video

## Summary

- Vendor: Wan-AI
- Model: `Wan-AI/Wan2.2-I2V-A14B-Diffusers`
- Task: Image-to-video generation
- Mode: Online serving with the OpenAI-compatible API
- Maintainer: Community

## When to use this recipe

Use this recipe when you want to deploy the Wan2.2 14B image-to-video model
with vLLM-Omni using multi-card parallelism. Two configurations are provided:

1. **Distilled model (no negative-prompt / CFG computation)** — higher
   throughput, recommended when using a distilled checkpoint that does not
   require classifier-free guidance.
2. **Official open-source model (with CFG)** — uses `--cfg 2` to run negative
   and positive samples in parallel for the original released weights.

## References

- Upstream model card: <https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers>

## Hardware Support

## NPU

### 8x Ascend A2 / A3

#### Environment

- OS: Linux
- Python: 3.10+
- Driver / runtime: Ascend NPU driver with CANN toolkit
- Recommended operator library: **mindie-sd** (Ascend high-performance fused
  operators — enables `adalayernorm` and other fused kernels automatically upon
  installation)
- vLLM version: Match the repository requirements for your checkout
- vLLM-Omni version or commit: Use the commit you are deploying from

#### Prerequisites

Install the **mindie-sd** operator library to enable Ascend-optimized fused
operators (`adalayernorm`, etc.):

```bash
git clone https://gitcode.com/Ascend/MindIE-SD.git && cd MindIE-SD

# Comment out the tik_ops build step (not needed for this use case)
sed -i 's|^\(\s*\)source ${current_script_dir}/build_tik_ops.sh|\1# source ${current_script_dir}/build_tik_ops.sh|' build/build_ops.sh

python setup.py bdist_wheel
cd dist
pip install mindiesd-*.whl
```

After installation, enable the Laser Attention kernel for significant
long-sequence speedups (up to ~40% at 720p in tested workloads):

```bash
export MINDIE_SD_FA_TYPE=ascend_laser_attention
```

When using HSDP with FSDP2, set the following environment variable to work
around a PyTorch NPU multi-stream memory reuse issue
([pytorch/pytorch#147168](https://github.com/pytorch/pytorch/issues/147168)).
This issue has been fixed on CUDA but still applies to NPU:

```bash
export MULTI_STREAM_MEMORY_REUSE=2
```

#### Command

**Distilled model (no CFG, recommended for distilled checkpoints):**

```bash
export MINDIE_SD_FA_TYPE=ascend_laser_attention
export MULTI_STREAM_MEMORY_REUSE=2

vllm serve \
  --omni Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --use-hsdp \
  --usp 8 \
  --vae-patch-parallel-size 8 \
  --vae-use-tiling
```

**Official open-source model (with CFG):**

```bash
export MINDIE_SD_FA_TYPE=ascend_laser_attention
export MULTI_STREAM_MEMORY_REUSE=2

vllm serve \
  --omni Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --use-hsdp \
  --usp 4 \
  --cfg 2 \
  --vae-patch-parallel-size 8 \
  --vae-use-tiling
```

> **Why the difference?** With `--cfg 2`, two copies of the input (positive and
> negative prompts) are processed in parallel, effectively doubling the compute
> for the DiT backbone. USP is therefore halved from 8 to 4 so that the total
> parallelism across the 8 cards remains balanced (`usp * cfg = 8`).

#### Verification

After the server is ready, see
[`examples/online_serving/image_to_video/README.md`](../../examples/online_serving/image_to_video/README.md)
for complete client examples and request formats.

#### Notes

- **Key flags:**
  - `--omni` — enables vLLM-Omni diffusion serving.
  - `--use-hsdp` — enables Hybrid Sharded Data Parallelism for the DiT model
    weights.
  - `--usp <N>` — Unified Sequence Parallelism degree.
  - `--cfg <N>` — Classifier-Free Guidance parallelism; set to 2 for models
    that require negative-prompt computation, omit for distilled models.
  - `--vae-patch-parallel-size 8` — parallelizes VAE decoding across all 8
    cards.
  - `--vae-use-tiling` — enables tiled VAE decoding to reduce peak memory.
- **Performance tips:**
  - Installing mindie-sd and enabling Laser Attention
    (`MINDIE_SD_FA_TYPE=ascend_laser_attention`) provides up to ~40%
    performance improvement at 720p resolution due to long-sequence attention
    optimization.
- **Known limitations:**
  - `MULTI_STREAM_MEMORY_REUSE=2` is required on NPU when using HSDP/FSDP2
    due to a multi-stream memory reuse bug. This is not needed on CUDA.
