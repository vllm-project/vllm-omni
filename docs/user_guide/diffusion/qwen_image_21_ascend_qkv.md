# Qwen-Image-2.1 Ascend QKV preprocessing

The optional MindIE-SD adapter combines Q/K affine RMS normalization and
interleaved RoPE before the existing Attention layer. Prefix-cache ownership,
CFG branches, and attention masking stay in the model. The second option keeps
head-major storage while concatenating cached and current K/V during decode.

Both options default to `false`. Enable them through `Omni`:

```python
from vllm_omni.entrypoints.omni import Omni

omni = Omni(
    model="Qwen/Qwen-Image-2.1",
    enforce_eager=True,
    additional_config={
        "qwen21_mindiesd_qkv_fusion": True,
        "qwen21_head_major_kv_concat": True,
    },
)
```

The initial supported slice is A5, BF16, batch 1, head dimension 128,
TP1/SP1, eager, unquantized weights, and native prefix/attention KV.
The installed MindIE-SD package must expose the callable Python API
`mindiesd.norm_rope_concat`. Unsupported configurations retain the original
path and log the reason. Failures after selecting the operator propagate.
Head-major concatenation applies only when the fused path is active.

The fused BF16 normalization/RoPE rounding differs from the original model
path. Do not claim bitwise equivalence to the unfused model or transfer
single-input quality screens to an entire quality suite. Turbo shares these
modules, but its distilled checkpoint has not been validated with this option.
Keep the options disabled for unvalidated configurations.

## Validation

CPU dispatch, configuration, prefix ownership, and concatenation tests:

```bash
python -m pytest -q \
  tests/diffusion/models/qwen_image_21/test_mindiesd_qkv.py \
  tests/diffusion/models/qwen_image_21/test_prefix_concat.py \
  tests/diffusion/models/qwen_image_21/test_route_observation.py
```

A5 operator correctness:

```bash
python -m pytest -q -s \
  tests/diffusion/models/qwen_image_21/test_mindiesd_qkv_npu.py
```

Full-request A/B, with two warmups and three interleaved measured requests per
arm and scenario, BF16/1024x1024/50 steps/seed 42/CFG1:

```bash
QWEN_IMAGE_21_TEST_MODEL=/path/to/Qwen-Image-2.1 \
QWEN_IMAGE_21_TEST_IMAGE=/path/to/input.png \
python -m pytest -q -s \
  tests/e2e/offline_inference/test_qwen_image_21_mindiesd.py
```

Warmup requests assert zero optimized calls in the off arm and actual
`mindiesd.norm_rope_concat` prefill/decode plus head-major prefix concatenation
in the on arm. Measured requests run after all observation wrappers are removed.

The image variable is optional; omit it to run T2I only. The test reports
synchronized `Omni.generate` wall time and allocated/reserved memory peaks;
initialization, warmup, PNG saving, and quality analysis are outside measured
time. Output images are generated artifacts. RGB PSNR >= 30 dB is a bounded
single-seed screen, not formal quality acceptance. The test does not enforce a
latency threshold on shared hardware.

Disable both options and create a fresh engine to restore the original path.
