---
hide:
  - navigation
  - toc
---

# Welcome to vLLM-Omni

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" src="./source/logos/vllm-omni-logo-text-dark.png">
    <img alt="vllm-omni" src="./source/logos/vllm-omni-logo-text-dark.png" width=55%>
  </picture>
</p>
<h3 align="center">
Easy, fast, and cheap omni-modality model serving for everyone
</h3>

<p style="text-align:center">
<script async defer src="https://buttons.github.io/buttons.js"></script>
<a class="github-button" href="https://github.com/vllm-project/vllm-omni" data-show-count="true" data-size="large" aria-label="Star">Star</a>
<a class="github-button" href="https://github.com/vllm-project/vllm-omni/subscription" data-show-count="true" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
<a class="github-button" href="https://github.com/vllm-project/vllm-omni/fork" data-show-count="true" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
</p>


## About

[vLLM](https://github.com/vllm-project/vllm) was originally designed to support large language models for text-based autoregressive generation tasks. vLLM-Omni extends its support for omni-modality model inference and serving:

- **Omni-modality**: Text, image, video, and audio data processing
- **Non-autoregressive Architectures**: extend the AR support of vLLM to Diffusion Transformers (DiT) and other parallel generation models
- **Heterogeneous outputs**: from traditional text generation to multimodal outputs

<p align="center">
  <picture>
    <img alt="vllm-omni" src="./source/architecture/omni-modality model architecture.png" width=55%>
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

vLLM-Omni seamlessly supports most popular open-source models on HuggingFace, including:

- Multi-modality generation models (e.g. Qwen-image)
- Omni-modality models (e.g. Qwen-omni)

For more information, checkout the following:
