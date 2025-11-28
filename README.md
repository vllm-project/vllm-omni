<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="./docs/source/logos/vllm-omni-logo-text-dark.png">
    <img alt="vllm-omni" src="./docs/source/logos/vllm-omni-logo-text-dark.png" width=55%>
  </picture>
</p>
<h3 align="center">
Easy, fast, and cheap omni-modality model serving for everyone
</h3>

<p align="center">
| <a href="https://vllm-omni.readthedocs.io/en/latest/"><b>Documentation</b></a> | <a href="https://tinyurl.com/vllm-omni-meeting"><b>Weekly Meeting</b></a> |
</p>

---

*Latest News* 🔥

- [2025/11] vLLM community officially released [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni) in order to support omni-modality models serving.

---

## About

[vLLM](https://github.com/vllm-project/vllm) was originally designed to support large language models for text-based autoregressive generation tasks. vLLM-Omni extends its support for omni-modality model inference and serving:

- **Omni-modality**: Text, image, video, and audio data processing
- **Non-autoregressive Architectures**: extend the AR support of vLLM to Diffusion Transformers (DiT) and other parallel generation models
- **Heterogeneous outputs**: from traditional text generation to multimodal outputs

<p align="center">
  <picture>
    <img alt="vllm-omni" src="./docs/source/architecture/omni-modality model architecture.png" width=55%>
  </picture>
</p>

vLLM-omni is fast with:

- Seamless AR support for efficient KV Cache management with vLLM
- Pipelined stage execution overlapping
- Fully disaggregation based on omniConnector and dynamic resource allocation across stages

vLLM-omni is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data and expert parallelism support for distributed inference
- Streaming outputs
- OpenAI-compatible API server

vLLM seamlessly supports most popular open-source models on HuggingFace, including:

- Multi-modality generation models (e.g. Qwen-image)
- Omni-modality models (e.g. Qwen-omni)

## Getting Started

Visit our [documentation](https://vllm-omni.readthedocs.io/en/latest/) to learn more.

Visit our documentation to learn more.

- [Installation](https://vllm-omni.readthedocs.io/en/latest/getting_started/installation/)
- [Quickstart](https://vllm-omni.readthedocs.io/en/latest/getting_started/quickstart/)
- [List of Supported Models](https://vllm-omni.readthedocs.io/en/latest/models/supported_models/)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM-omni](https://vllm-omni.readthedocs.io/en/latest/contributing/) for how to get involved.

## Weekly Meeting

- vLLM Omni Weekly Meeting: https://tinyurl.com/vllm-omni-meeting
- Wednesday, 11:30 - 12:30 (UTC+8, [Convert to your timezone](https://dateful.com/convert/gmt8?t=15))

## License

Apache License 2.0, as found in the [LICENSE](./LICENSE) file.
