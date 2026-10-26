# MiniMax H3 Video VAE Omni operators (experimental)

Set `MINIMAX_H3_VAE_USE_OMNI_OPS=1` before starting the process to enable
the native H3 decoder's Omni RMSNorm and RoPE adapters:

```bash
MINIMAX_H3_VAE_USE_OMNI_OPS=1 \
vllm serve /path/to/MiniMax-H3/FL2VA --omni --trust-remote-code
```

The switch defaults to off. Values `1`, `true`, `yes`, and `on` enable it;
matching ignores case and surrounding whitespace. Recreate the VAE or restart
the serving process after changing it. It can also be enabled without INT8
for separate operator validation and performance comparisons.

Installation requires a native decoder, an NPU target, and the Ascend NPU
platform. CPU-stored weights during initialization or offload do not prevent
installation. Encode-only VAEs, missing decoders, replacement NVFP4 decoders,
and non-NPU targets/platforms skip this adapter with a reason in startup logs.
Online VAE INT8 is a separate feature. When that feature is available, its
configuration restrictions still apply when combining both optimizations.

Affine RMSNorm uses the Omni NPU implementation backed by
`torch_npu.npu_rms_norm`. Non-affine Q/K RMSNorm with an explicit epsilon
uses the same NPU primitive with a fixed all-ones gamma buffer, without adding
checkpoint keys. Q/K normalization retains the remote code's FP32 input and
casts the result back to the original Q/K dtype.
Norms with a dynamic epsilon (`eps=None`) keep their original implementation.
RoPE uses `torch_npu.npu_rotary_mul` directly for nonempty FP16, BF16, or FP32
inputs on NPU, without requiring MindIE or restricting batch size to one.
Shared and per-batch coefficients are supported, and dimensions outside the
rotary prefix remain unchanged. Other devices, empty inputs, and unsupported
dtypes use the H3 native calculation. RMSNorm and RoPE remain separate kernels.
A selected kernel's runtime failure is
reported rather than silently switching implementations.

Norm and Attention replacements occur after FP32 weight loading and before
online INT8 conversion, device-only placement, temporal patching, and residency
snapshots. Startup logs report adapter installation counts; RoPE dispatch logs
identify its backend and selection reason. CPU tests cover the switch, initialization order,
Attention/RoPE contracts, and mixed parameter residency. Combined NPU operator
accuracy, video quality, memory, and latency gains still require A2/A3
validation; CPU test results do not establish those gains.

When the separate online VAE INT8 feature is also installed, both can be enabled:

```bash
MINIMAX_H3_VAE_USE_OMNI_OPS=1 \
vllm serve /path/to/MiniMax-H3/FL2VA --omni --trust-remote-code \
  --diffusion-quantization-config '{"transformer":null,"video_vae":{"method":"int8","activation_scheme":"dynamic","is_checkpoint_int8_serialized":false},"text_encoder":null,"audio_vae":null,"default":null}'
```
