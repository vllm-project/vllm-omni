# NPU

vLLM-Omni supports NPU through the vLLM Ascend Plugin (vllm-ascend). This is a community maintained hardware plugin for running vLLM on NPU.

## Requirements

- OS: Linux
- Python: 3.12

!!! note
    vLLM-Omni is currently not natively supported on Windows.

=== "NPU"

    --8<-- "docs/getting_started/installation/npu/npu.inc.md:requirements"

## Installation

### Set up using Docker

=== "NPU"

    --8<-- "docs/getting_started/installation/npu/npu.inc.md:pre-built-images"

### Build wheel from source

=== "NPU release"

    --8<-- "docs/getting_started/installation/npu/npu.inc.md:installation-release"

=== "NPU from main"

    --8<-- "docs/getting_started/installation/npu/npu.inc.md:installation-main"

## CosyVoice3 on Ascend 950 (A5)

The A5 platform setup enables CPU STFT and ISTFT for the CosyVoice3 HiFiGAN
vocoder because native STFT is unavailable on this platform. Both transforms,
including complex-spectrum construction, run in float32 on CPU; their real
outputs return to the caller's device and dtype. Other platforms keep the
native transform path.

This fallback introduces CPU/device transfers and is a correctness workaround,
not a throughput optimization. End-to-end latency and CER must be measured on
the exact hardware and software stack used for deployment. Use a matching
vLLM/vLLM-Ascend/vLLM-Omni installation; this change does not make older runner
interfaces compatible with current main.
