# Configuration Options

This section lists the most common options for running vLLM-Omni.

For options within a vLLM Engine. Please refer to [vLLM Configuration](https://docs.vllm.ai/en/v0.11.0/configuration/index.html)

Currently, the main options are maintained by stage configs for each model.

For specific example, please refer to [Qwen2.5-omni stage config](stage_configs/qwen2_5_omni.yaml)

For introduction, please check [Introduction for stage config](./stage_configs.md)

## Optimization Features

- **[TeaCache Configuration](../user_guide/teacache.md)** - Enable TeaCache adaptive caching for DiT models to achieve 1.5x-2.0x speedup with minimal quality loss
