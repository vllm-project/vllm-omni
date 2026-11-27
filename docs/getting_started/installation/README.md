# Installation

!!! important
    This source checkout targets the official [vLLM 0.34.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.34.0), at commit `62d4bc3c7e33b1b6c906e233e6af2d3506c482ca`. CUDA uses PyTorch 2.14.0 and CUDA 13.0; ROCm and XPU retain their upstream platform runtimes. The CUDA, ROCm and XPU Dockerfiles use published upstream 0.34.0 images. Published Omni wheel instructions below retain the vLLM dependency matching that Omni release.

    CI results obtained with vLLM 0.34.0rc3 do not qualify the final 0.34.0 artifacts. Final-release model validation and ROCm/XPU runtime qualification remain pending.

The CUDA Dockerfiles rebuild LMCache 0.5.6 against the selected Torch 2.14
runtime to preserve the extension's native ABI. They retain its CUDA architecture
coverage and check dependency consistency. Release-container validation remains pending.

vLLM-Omni supports the following hardware platforms:

- [GPU](gpu.md)
    - [NVIDIA CUDA](gpu.md)
    - [AMD ROCm](gpu.md)
    - [Intel XPU](gpu.md)
    - [MThreads MUSA](gpu.md)
- [NPU](npu.md)

## Source development and demo dependencies

For this 0.34 source checkout, the `[dev]` and `[demo]` extras need a temporary
Gradio source backport to satisfy vLLM's Pillow 13 requirement. From the repository
root, after installing the matching vLLM runtime for your platform. The builder
requires Git and uv 0.11.8 or newer:

```bash
python tools/build_gradio_pillow13.py --output-dir /tmp/gradio-pillow13
uv pip install -e '.[dev,demo]' \
  /tmp/gradio-pillow13/gradio-6.31.0+omni.pillow13-py3-none-any.whl
uv pip check
```

Select just `.[dev]` or `.[demo]` when appropriate, keeping the explicit wheel
in the same install command. Preserve the platform's target-device and build
options shown in the GPU installation instructions. The isolated build verifies
pinned source and wheel hashes and retains the original licenses.

The local distribution `gradio==6.31.0+omni.pillow13` derives from the exact
official Gradio 6.31.0 source archive. It backports the Pillow `<14` bound and
I-mode PNG error guidance from upstream [Gradio PR #14005](https://github.com/gradio-app/gradio/pull/14005),
merged to `7.0-dev`; it is not an official Gradio 6 release. Hash-pinned source,
patch and build-tool versions are recorded under `tools/gradio_pillow13/`;
the builder preserves the original licenses. I-mode PNG output requires an explicit `I;16`
conversion for in-range values or a format such as TIFF to preserve 32-bit values.
