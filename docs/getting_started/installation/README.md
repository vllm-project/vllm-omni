# Installation

!!! important
    [vLLM-Omni 0.32.0](https://github.com/vllm-project/vllm-omni/releases/tag/v0.32.0) is published with a Python wheel and CUDA Docker images. It requires the matching vLLM 0.32.0 release and Python `>=3.11,<3.14`; the CUDA installation uses CUDA 13.0 and PyTorch 2.13.0. This source checkout also targets vLLM 0.32.0 at `5df63a7203cdb9328d824a27b682e77c7b9928be`, but `main` can contain changes beyond the tagged Omni release. Choose the installation path for your platform: CUDA release artifacts do not establish ROCm, XPU, MUSA, or NPU runtime compatibility. The platform guides retain their separate version and validation requirements.

vLLM-Omni supports the following hardware platforms:

- [GPU](gpu.md)
    - [NVIDIA CUDA](gpu.md)
    - [AMD ROCm](gpu.md)
    - [Intel XPU](gpu.md)
    - [MThreads MUSA](gpu.md)
- [NPU](npu.md)
