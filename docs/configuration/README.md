# Configuration Options

This section lists the most common options for running vLLM-Omni.

For options within a vLLM Engine. Please refer to [vLLM Configuration](https://docs.vllm.ai/en/v0.16.0/configuration/index.html)

Currently, the main options are maintained by stage configs for each model.

For specific example, please refer to [Qwen2.5-omni stage config](stage_configs/qwen2_5_omni.yaml)

For introduction, please check [Introduction for stage config](./stage_configs.md)

## Memory Configuration

- **[GPU Memory Calculation and Configuration](./gpu_memory_utilization.md)** - Guide on how to calculate memory requirements and set up `gpu_memory_utilization` for optimal performance

## Multi-Stage Recipes

- **[Prefill-Decode Disaggregation](./pd_disaggregation.md)** - How to derive a PD-aware Qwen3-Omni stage config from the default config without introducing another bundled YAML

## Optimization Features

- **[Diffusion Features Overview](../user_guide/diffusion_features.md)** - Complete overview of all diffusion model features and supported models
