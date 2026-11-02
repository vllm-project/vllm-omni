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

## Source development and demo dependencies

For this 0.33 source checkout, the `[dev]` and `[demo]` extras need a temporary
Gradio source backport to satisfy vLLM's Pillow 13 requirement. From the repository
root, after installing the matching vLLM runtime for your platform:

```bash
python tools/build_gradio_pillow13.py --output-dir /tmp/gradio-pillow13
uv pip install -e '.[dev,demo]' \
  /tmp/gradio-pillow13/gradio-6.31.0+omni.pillow13-py3-none-any.whl
uv pip check
```

Select just `.[dev]` or `.[demo]` when appropriate, keeping the explicit wheel
in the same install command. Preserve the platform's target-device and build
options shown in the GPU installation instructions. The builder requires uv
0.11.8 or newer, uses an isolated build, and retains its source hashes, license
verification and build receipt beside the wheel.

The local distribution `gradio==6.31.0+omni.pillow13` derives from the exact
official Gradio 6.31.0 source archive. It backports the Pillow `<14` bound and
I-mode PNG error guidance from upstream [Gradio PR #14005](https://github.com/gradio-app/gradio/pull/14005),
merged to `7.0-dev`; it is not an official Gradio 6 release. Hash-pinned source,
patch and build-tool versions are recorded under `tools/gradio_pillow13/`;
the builder preserves the original licenses. I-mode PNG output requires an explicit `I;16`
conversion for in-range values or a format such as TIFF to preserve 32-bit values.

A clean Python 3.12.13 environment passed normal dependency installation and
checks with the targeted vLLM revision, Pillow 13 and VoxCPM 2.0.3, plus selected
CPU Image, ImageEditor, Gallery and helper serialization tests. This does not
establish browser, native model, full CI or ROCm/XPU/MUSA runtime acceptance.
The builder reproduced the same wheel on Python 3.11.15, 3.12.13 and 3.13.13;
full environments on Python 3.11 and 3.13 remain unverified. Released 0.32
installation examples retain their published dependency sets.
