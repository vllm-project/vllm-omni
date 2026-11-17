# Online FP8 on ROCm

## Online block scaling

ROCm can opt into finer online FP8 groups when per-tensor or fixed 128-block
scaling loses too much diffusion quality. Set `online_block_size` to `32`,
`64`, or `128` while loading an ordinary BF16/FP16 checkpoint:

```python
from vllm_omni import Omni

omni = Omni(
    model="<your-model>",
    quantization_config={
        "method": "fp8",
        "online_block_size": 64,
        "ignored_layers": ["<model-specific-sensitive-layer>"],
    },
)
```

Serving accepts the same configuration as JSON:

```bash
vllm serve <your-model> --omni \
  --diffusion-quantization-config \
  '{"method":"fp8","online_block_size":64,"ignored_layers":["<model-specific-sensitive-layer>"]}'
```

The option uses square `online_block_size × online_block_size` weight groups
and dynamic `1 × online_block_size` activation groups. Existing component
routing and `ignored_layers` still apply. The option is ROCm-only, requires
dynamic activation scaling, and cannot be combined with serialized FP8
checkpoint metadata or `weight_block_size`.

Z-Image's ROCm quality configuration uses `64` together with its complete
checked-in sensitive-layer exclusions. Do not copy only part of that layer
policy into production. Validate every model and block size against a BF16
baseline with the same prompt, seed, resolution, and sampling settings.

The AITER block kernel currently accepts `1 × 128` activation groups. Finer
groups can select vLLM's ROCm Triton block kernel instead, so measure latency
and throughput as well as quality. Configurable online blocks are not yet
supported with the Diffusers backend, DLO AllGather, or HSDP.

See [FP8 quantization](fp8.md) for component scope, ignored-layer guidance,
and serialized-checkpoint configuration.

## Experimental MiniMax-H3-Image MLP FP8 with Hadamard64

MiniMax-H3-Image can opt into online FP8 for its 100 MLP linears while keeping
attention projections, text encoding, the image VAE and protected FP32 heads
in their existing precision. The optional
`minimax_h3_mlp_fp8_hadamard` boolean in `additional_config` adds a signed,
normalized 64-point Hadamard rotation to the 50 MLP `fc2` input/weight pairs.
Its default is `false`; enabling it also requires the explicit MLP-only
quantization configuration below.

Add these options to a working single-GPU MiniMax-H3-Image launch:

```bash
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_SKINNY_GEMM=0 \
VLLM_OMNI_ROCM_NATIVE_DIFFUSION_OPS=1 \
vllm serve <model-path> --omni \
  --num-gpus 1 --usp 1 --ring 1 --text-encoder-tp-size 1 \
  --additional-config '{"minimax_h3_mlp_fp8_hadamard":true}' \
  --diffusion-quantization-config \
  '{"default":null,"text_encoder":null,"vae":null,"transformer":{"method":"online","targets":{"re:blocks\\.[0-9]+\\.mlp\\.(fc1|fc2)":"fp8_per_channel"}}}'
```

Enable AITER for the supported per-token FP8 path. The native-ops and
skinny-GEMM settings above reproduce the tested CI runtime, whose precompiled
vLLM kernels lack `gfx950` support. A multi-architecture runtime with compatible
kernels may not require those two overrides; see the
[AMD H3 image recipe](../../../recipes/MiniMaxAI/MiniMax-H3-Image-AMD.md).
Keep the same runtime settings for BF16 and FP8 comparisons.

The supported path uses a BF16 checkpoint, ROCm `gfx950`, TP1/SP1 and vLLM's
online per-channel-weight/per-token-activation FP8 method with the row-wise
Torch FP8 scaled-matrix-multiplication kernel. Keep weights resident; CPU
offload, sleep mode, LoRA adapters and runtime weight updates are unsupported.
Other quantization recipes,
backends or feature combinations are rejected when the rotation is enabled.
Keep `online_block_size` unset for this recipe. The value 64 specifies the
rotation's channel-group size; the FP8 scaling remains per channel for weights
and per token for activations.

Weights rotate in FP32 during loading before FP8 conversion. At runtime one
Triton kernel fuses activation rotation and FP8 quantization, retaining the
native main FP8 GEMM without an extra GEMM or a materialized rotated activation.
The matching input/weight rotations preserve the ideal matrix product; FP8
rounding still changes model outputs.

This is an experimental performance option, not a claim of BF16-equivalent
accuracy. Controlled tests showed better mean SSIM/PSNR than ordinary MLP FP8,
but some cases regressed and substantial BF16 differences remained. Validate
the prompts, geometry and sampling settings required by your application.
For a BF16 baseline, omit both the quantization configuration and the Hadamard
option. Independent single-GPU replicas can each use this same configuration.
