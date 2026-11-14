# Installation

!!! important
    This source checkout targets vLLM 0.34.0rc3 at `9950ce36a2e1f4307e99bb4972202c1618b4787e`, with CUDA 13.0 and PyTorch 2.14.0. Full CI and ROCm/XPU runtime validation remain pending. CUDA Dockerfiles retain the published 0.33.0 bootstrap image and verify the frozen candidate wheel; ROCm builds the same candidate commit from source on its 0.33.0 bootstrap base. The published Omni 0.32.0 wheel instructions below retain their matching vLLM 0.32.0 dependency.

The CUDA Dockerfiles rebuild LMCache 0.5.6 against Torch 2.14 because the base's
Torch 2.13 extension fails to import. They retain its CUDA architecture coverage
and check dependency consistency. Release-container validation remains pending.

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
