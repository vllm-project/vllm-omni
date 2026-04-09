# Frequently Asked Questions

> Q: How many chips do I need to infer a model in vLLM-Omni?

A: Now, we support natively disaggregated deployment for different model stages within a model. There is a restriction that one chip can only have one AutoRegressive model stage. This is because the unified KV cache management of vLLM. Stages of other types can coexist within a chip. The restriction will be resolved in later version.

> Q: When trying to run examples, I encounter error about backend of librosa or soundfile. How to solve it?

A: If you encounter error about backend of librosa, try to install ffmpeg with command below.
```
sudo apt update
sudo apt install ffmpeg
```

> Q: I see GPU OOM or "free memory is less than desired GPU memory utilization" errors. How can I fix it?

A: Refer to [GPU memory calculation and configuration](../configuration/gpu_memory_utilization.md) for guidance on tuning `gpu_memory_utilization` and related settings.

> Q: I encounter some bugs or CI problems, which is urgent. How can I solve it?

A: At first, you can check current [issues](https://github.com/vllm-project/vllm-omni/issues) to find possible solutions. If none of these satisfy your demand and it is urgent, please find these [volunteers](https://docs.vllm.ai/projects/vllm-omni/en/latest/community/volunteers/) for help.

> Q: Does vLLM-Omni support AWQ or any other quantization?

A: We plan to introduce GGUF FP8 prequantized models and online FP8 quantization in version 0.16.0. Support for other quantization types will follow in future releases. For details, please see our [Q1 quantization roadmap](https://github.com/vllm-project/vllm-omni/issues/1057).

> Q: Does vLLM-Omni support multimodal streaming input and output?

A: Not yet. We already put it on the [Roadmap](https://github.com/vllm-project/vllm-omni/issues/165). Please stay tuned!
