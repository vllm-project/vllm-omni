# CPU Offloading for Diffusion Models

vLLM-Omni provides three CPU-offload strategies for diffusion models. Choose
the coarsest strategy that meets your memory target: finer-grained streaming
saves more device memory but adds more transfer and setup overhead.

For the shared factory, discovery, and lifecycle contract, see the
[CPU Offloading design](../../design/feature/offloader/README.md).

## Choose a strategy

| Strategy | Device residency | Parallel scope | Guide |
| --- | --- | --- | --- |
| Model-level (sequential) | One pipeline component group at a time | Single device | [Model-Level Offloading](offloader/cpu_offload.md) |
| Layerwise (blockwise) | One transformer block, with next-block prefetch | Single device | [Layerwise Offloading](offloader/layerwise_offload.md) |
| Distributed layerwise | Fixed two-block device buffer; optional host-weight sharding and AllGather | Multiple GPU/NPU ranks | [Distributed Layerwise Offloading](offloader/distributed_layerwise_offload.md) |

All strategies use pinned host memory for faster transfers where applicable.
Configuration priority is:

1. Distributed layerwise offloading.
2. Layerwise offloading.
3. Model-level offloading.

Treat the flags as mutually exclusive. If more than one is enabled, the
higher-priority strategy is selected.

## Quick selection

- Use [model-level offloading](offloader/cpu_offload.md) when swapping whole
  encoders and DiTs is enough to fit and phase-boundary transfers are
  acceptable.
- Use [layerwise offloading](offloader/layerwise_offload.md) for compute-heavy video
  DiTs where block transfers can overlap computation.
- Use [distributed layerwise offloading](offloader/distributed_layerwise_offload.md)
  when a multi-rank deployment also needs bounded device residency and,
  optionally, sharded host weights.

## Supported models

| Architecture | Example models | DiT class | Model-level | Layerwise | Distributed layerwise | Layerwise block attributes |
| --- | --- | --- | :---: | :---: | :---: | --- |
| Flux2Pipeline | `black-forest-labs/FLUX.2-dev` | `Flux2Transformer2DModel` | yes | yes | — | `transformer_blocks`, `single_transformer_blocks` |
| LongCatImagePipeline | `meituan-longcat/LongCat-Image` | `LongCatImageTransformer2DModel` | — | yes | — | `transformer_blocks`, `single_transformer_blocks` |
| NextStep11Pipeline | `stepfun-ai/NextStep-1.1` | `NextStepModel` | — | yes | — | `layers` |
| OvisImagePipeline | `AIDC-AI/Ovis-Image-7B` | `OvisImageTransformer2DModel` | — | yes | — | `transformer` |
| QwenImagePipeline | `Qwen/Qwen-Image` | `QwenImageTransformer2DModel` | yes | yes | — | `transformer_blocks` |
| StableDiffusionXLPipeline | `stabilityai/stable-diffusion-xl-base-1.0` | `SDXLUNet2DConditionModel` | yes | yes | — | `down_blocks`, `up_blocks` |
| StableDiffusion3Pipeline | `stabilityai/stable-diffusion-3.5-medium` | `SD3Transformer2DModel` | — | yes | — | `transformer_blocks` |
| Wan22I2VPipeline | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | `WanTransformer3DModel` | yes | yes | — | `blocks` |
| Wan22Pipeline | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | `WanTransformer3DModel` | yes | yes | — | `blocks` |
| SoulXSingerPipeline / SoulXSingerSVCPipeline | `Soul-AILab/SoulX-Singer` | `DiffLlama` | yes | yes | — | `layers` |
| BagelPipeline | `ByteDance-Seed/BAGEL-7B-MoT` | `Qwen2MoTModel` | — | yes | — | `layers`, customized modules |
| Cosmos3OmniDiffusersPipeline | `nvidia/Cosmos3-Nano`, `nvidia/Cosmos3-Super` | `Cosmos3VFMTransformer`, `Cosmos3LanguageModel` | yes | yes | yes | `layers`, `gen_layers` |

Model-level support requires discoverable DiT and encoder components.
Layerwise support requires transformer block topology. Distributed support
reuses that topology but still requires validation for the model, checkpoint,
and parallel configuration.

## Compatibility anchors

These headings preserve links to sections that moved into dedicated guides.

## Model-level (Sequential) Offloading

See [Model-Level Offloading](offloader/cpu_offload.md).

## Layerwise (Blockwise) Offloading

See [Layerwise Offloading](offloader/layerwise_offload.md).

## Distributed Layerwise Offloading

See [Distributed Layerwise Offloading](offloader/distributed_layerwise_offload.md).
