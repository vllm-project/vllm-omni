# Configuration Options

This section lists the most common options for running vLLM-Omni.

For options within a vLLM Engine. Please refer to [vLLM Configuration](https://docs.vllm.ai/en/v0.14.0/configuration/index.html)

Currently, the main options are maintained by stage configs for each model.

For specific example, please refer to [Qwen2.5-omni stage config](stage_configs/qwen2_5_omni.yaml)

For introduction, please check [Introduction for stage config](./stage_configs.md)

## Memory Configuration

- **[GPU Memory Calculation and Configuration](./gpu_memory_utilization.md)** - Guide on how to calculate memory requirements and set up `gpu_memory_utilization` for optimal performance

## Optimization Features

- **[TeaCache Configuration](../user_guide/diffusion/teacache.md)** - Enable TeaCache adaptive caching for DiT models to achieve 1.5x-2.0x speedup with minimal quality loss
- **[Cache-DiT Configuration](../user_guide/diffusion/cache_dit_acceleration.md)** - Enable Cache-DiT as cache acceleration backends for DiT models
- **[Parallelism Configuration](../user_guide/diffusion/parallelism_acceleration.md)** - Enable parallelism (e.g., sequence parallelism) for for DiT models
