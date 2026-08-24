# Frequently Asked Questions

> Q: How many chips do I need to infer a model in vLLM-Omni?

A: Now, we support natively disaggregated deployment for different model stages within a model. There is a restriction that one chip can only have one AutoRegressive model stage. This is because the unified KV cache management of vLLM. Stages of other types can coexist within a chip. The restriction will be resolved in later version.

> Q: I see GPU OOM or "free memory is less than desired GPU memory utilization" errors. How can I fix it?

A: Refer to [GPU memory calculation and configuration](../configuration/gpu_memory_utilization.md) for guidance on tuning `gpu_memory_utilization` and related settings.

> Q: I encounter some bugs or CI problems, which is urgent. How can I solve it?

A: Search the current [issues](https://github.com/vllm-project/vllm-omni/issues) first. If no existing issue matches a bug or CI failure, open a [bug report](https://github.com/vllm-project/vllm-omni/issues/new?template=400-bug-report.yml) with a minimal reproducer, environment details, and relevant logs. For technical or usage questions, use the `sig-omni` channel in [vLLM Slack](https://slack.vllm.ai/) or the [vLLM Forum](https://discuss.vllm.ai/).

> Q: Does vLLM-Omni support AWQ or any other quantization?

A: AWQ is available through the inherited vLLM quantization registry, and vLLM-Omni supports several additional methods. Actual support and validation vary by model, component, hardware, and checkpoint. See the [quantization overview](../user_guide/quantization/overview.md) for validated combinations and method-specific guides.

> Q: Does vLLM-Omni support multimodal streaming input and output?

A: Yes, for specific models and serving paths. Current capabilities include [streaming video input](../serving/video_stream_api.md), [streaming diffusion output](../user_guide/diffusion/execution_modes.md#streaming-output), and experimental full-duplex real-time audio described in the [project overview](../README.md). Check the model and API documentation because input and output support is model-specific.
