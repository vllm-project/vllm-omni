# Installation

!!! important
    vLLM-Omni is released against the matching upstream vLLM major/minor version. This source checkout targets the vLLM 0.32.0 wheel at `5df63a7203cdb9328d824a27b682e77c7b9928be`, tested with CUDA 13.0 and PyTorch 2.13.0. The [vLLM 0.32.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.32.0), PyPI wheels and upstream CUDA/ROCm/XPU images are published. The Dockerfiles now select those matching release images; CI pins the CUDA image by digest. Full CI validation of the final CUDA base and ROCm/XPU runtime validation remain pending. The published vLLM-Omni 0.32.0 wheel instructions use vLLM 0.32.0; choose the installation path that matches your Omni revision.

!!! note "Documentation version"
    These instructions target vLLM-Omni 0.32.0. The [`latest` documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/) follows development on `main`, not this release. Select documentation matching the version you install; do not combine release dependencies with an unpinned `main` checkout.

vLLM-Omni supports the following hardware platforms:

- [GPU](gpu.md)
    - [NVIDIA CUDA](gpu.md)
    - [AMD ROCm](gpu.md)
    - [Intel XPU](gpu.md)
    - [MThreads MUSA](gpu.md)
- [NPU](npu.md)
