# Installation

!!! important
    This source checkout targets vLLM 0.33.0rc3.dev3+g4ba787ec9 at `4ba787ec9959e566d3b587a796dff07673324da5`, with CUDA 13.0 and PyTorch 2.14.0. Full CI and ROCm/XPU runtime validation remain pending. CUDA Dockerfiles upgrade the published 0.32.0 bootstrap image using the frozen candidate wheel; ROCm builds the candidate from source. The published Omni 0.32.0 wheel instructions below retain their matching vLLM 0.32.0 dependency.

vLLM-Omni supports the following hardware platforms:

- [GPU](gpu.md)
    - [NVIDIA CUDA](gpu.md)
    - [AMD ROCm](gpu.md)
    - [Intel XPU](gpu.md)
    - [MThreads MUSA](gpu.md)
- [NPU](npu.md)
