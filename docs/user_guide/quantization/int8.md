# Int8 Quantization

## Overview

Int8 quantization supports W8A8 diffusion transformer inference on CUDA and
Ascend NPU. It can quantize BF16/FP16 weights at load time, or load serialized
Int8 checkpoints that already contain quantized weights and scales.

Only online activation scaling is currently supported.

## Hardware Support

| Device | Support |
|--------|---------|
| NVIDIA Blackwell GPU (SM 100+) | ✅ |
| NVIDIA Ada/Hopper GPU (SM 89+) | ✅ |
| NVIDIA Ampere GPU (SM 80+) | ✅ |
| AMD ROCm | ⭕ |
| Intel XPU | ⭕ |
| Ascend NPU | ✅ |

Legend: `✅` supported, `❌` unsupported, `⭕` not verified in this
guide.

## Model Type Support

### Diffusion Model (Qwen-Image, Wan2.2)

| Model | HF models | CUDA | Ascend NPU | Mode | Recommendation |
|-------|-----------|:----:|:----------:|------|----------------|
| Qwen-Image | `Qwen/Qwen-Image`, `Qwen/Qwen-Image-2512` | Yes | Yes | Online W8A8 | All layers |
| Wan2.2 | Wan2.2 diffusion pipelines | Not validated | Not validated | Online W8A8 | Validate before enabling in docs |
| Z-Image | `Tongyi-MAI/Z-Image-Turbo` | Yes | Yes | Online W8A8 | All layers |

Other diffusion models may work if their transformer uses supported linear
layers, but they are not validated in this guide.

### Multi-Stage Omni/TTS Model (Qwen3-Omni, Qwen3-TTS)

| Model | Scope | Status | Notes |
|-------|-------|--------|-------|
| Qwen3-Omni | Thinker language-model stage | Not validated | Prefer checkpoint-supported ModelOpt FP8 or AutoRound paths |
| Qwen3-TTS | TTS language-model stage | Not validated | No Int8 TTS stage support is documented |

### Multi-Stage Diffusion Model (BAGEL, GLM-Image)

| Model | Scope | Status | Notes |
|-------|-------|--------|-------|
| BAGEL | Stage-specific transformer or DiT module | Not validated | Requires explicit stage routing |
| GLM-Image | Stage-specific transformer or DiT module | Not validated | Requires quality comparison with BF16 |

## Configuration

Python API:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(model="<your-model>", quantization="int8")

omni_with_skips = Omni(
    model="<your-model>",
    quantization_config={
        "method": "int8",
        "ignored_layers": ["<layer-name>"],
    },
)

outputs = omni.generate(
    "A cat sitting on a windowsill",
    OmniDiffusionSamplingParams(num_inference_steps=50),
)
```

CLI:

```bash
python text_to_image.py --model <your-model> --quantization int8
python text_to_image.py --model <your-model> --quantization int8 --ignored-layers "img_mlp"
vllm serve <your-model> --omni --quantization int8
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | - | Quantization method (`"int8"`) |
| `activation_scheme` | str | `"dynamic"` | `"dynamic"` selects online activation scaling; static is not supported |
| `ignored_layers` | list[str] | `[]` | Layer name patterns to keep in BF16/FP16 |
| `is_checkpoint_int8_serialized` | bool | `False` | Set by checkpoint config when loading serialized Int8 weights |

## Validation and Notes

Int8 quantization can be combined with cache acceleration:

```python
omni = Omni(
    model="<your-model>",
    quantization="int8",
    cache_backend="tea_cache",
    cache_config={"rel_l1_thresh": 0.2},
)
```

Only add a new model to the supported table after comparing the Int8 output
against a BF16 baseline and documenting any required `ignored_layers`.


## MiniMax H3 Video VAE on Ascend NPU (experimental)

The native H3 video decoder has an explicit online W8A8 path that reuses
`NPUInt8OnlineLinearMethod`. This path has CPU unit coverage; A2/A3 operator,
video-quality, and performance validation are still required before deployment.
It requires the normal floating-point VAE checkpoint and `--trust-remote-code`.

Enable it with the exact `video_vae` component key. A global `int8` config,
a `default` config, or a transformer-only config does not enable VAE quantization.
For example, the following starts with VAE-only quantization; keep the other
serving and offload options appropriate for your deployment:

```bash
vllm serve /path/to/MiniMax-H3/FL2VA --omni --trust-remote-code \
  --diffusion-quantization-config '{"transformer":null,"video_vae":{"method":"int8","activation_scheme":"dynamic","is_checkpoint_int8_serialized":false},"text_encoder":null,"audio_vae":null,"default":null}'
```

To quantize both DiT and Video VAE, replace `"transformer":null` with
`"transformer":{"method":"int8"}`. The same component config is accepted
through the Python `quantization_config` argument and by the H3 decoder-only
pipeline's diffusion-stage configuration.

Only these native decoder Transformer-block projections are replaced:

- `video_vae.decoder.transformer_blocks.<i>.attn.to_qkv`
- `video_vae.decoder.transformer_blocks.<i>.attn.to_out`
- `video_vae.decoder.transformer_blocks.<i>.ff.w1`
- `video_vae.decoder.transformer_blocks.<i>.ff.w2`

The reference 36-block decoder has 144 candidate linears. Convolutions,
`decoder.x_embedder`, `decoder.proj_out`, the video encoder, Audio VAE, and
Attention core remain unchanged. The two boundary projections retain their
floating weight dtype because the native helper reads that dtype to cast inputs.
Use `ignored_layers` inside the `video_vae` config to keep selected candidates
floating, using the existing INT8 matching rules and the canonical names above:

```json
{
  "video_vae": {
    "method": "int8",
    "ignored_layers": ["video_vae.decoder.transformer_blocks.0.attn.to_out"]
  },
  "default": null
}
```

Weights are converted once at startup, one Linear at a time on NPU. The adapter
stores transposed INT8 weights and FP32 per-output-channel scales/biases; it
uses the existing per-token activation quantization and NPU quantized matmul
at forward time. Internal compute defaults to FP16, follows active NPU
FP16/BF16 autocast, and restores the input dtype when autocast is disabled.
Restoring FP32 output does not imply FP32 matmul arithmetic. BF16 requires
validation on the target software stack.

With legacy CPU/layerwise/DLO offload, conversion finishes before residency
snapshots, which preserve the mixed parameter dtypes. Move the converted VAE
by device only; do not call whole-module `.half()` or `.bfloat16()`, which also
cast its scales and biases. Startup logs report the number of quantized and
ignored linears. No floating-weight fallback is retained after installation.

Serialized INT8 VAE checkpoints, nested `video_vae.*` component routes,
encode-only quantization, and replacement NVFP4 decoders are rejected. This
includes automatically detected `vae_decoder_nvfp4` assets; set
`additional_config.video_vae_decoder` to `null` to use the native decoder
instead. This minimal path does not add HWR-required or HSDP support.

### Combine native VAE INT8 with Omni operators (experimental)

Set `MINIMAX_H3_VAE_USE_OMNI_OPS=1` before starting the process to enable
the native H3 decoder's Omni RMSNorm and RoPE adapters alongside INT8:

```bash
MINIMAX_H3_VAE_USE_OMNI_OPS=1 \
vllm serve /path/to/MiniMax-H3/FL2VA --omni --trust-remote-code \
  --diffusion-quantization-config '{"transformer":null,"video_vae":{"method":"int8","activation_scheme":"dynamic","is_checkpoint_int8_serialized":false},"text_encoder":null,"audio_vae":null,"default":null}'
```

The switch defaults to off. Values `1`, `true`, `yes`, and `on` enable it;
matching ignores case and surrounding whitespace. Recreate the VAE or restart
the serving process after changing it. It can also be enabled without INT8
for separate operator validation and performance comparisons.

Installation requires a native decoder, an NPU target, and the Ascend NPU
platform. CPU-stored weights during initialization or offload do not prevent
installation. Encode-only VAEs, missing decoders, replacement NVFP4 decoders,
and non-NPU targets/platforms skip this adapter with a reason in startup logs.
The existing INT8 configuration restrictions still apply.

Affine RMSNorm uses the Omni NPU implementation backed by
`torch_npu.npu_rms_norm`. RoPE uses the MindIE kernel only when it is available
and the actual input is on NPU within the initial batch-size-one scope;
otherwise it uses the H3 native calculation while preserving the batch
dimension. A missing MindIE RoPE kernel does not
disable INT8. This path does not fuse Q/K normalization with RoPE; the native
non-affine Q/K norms remain unchanged. A selected kernel's runtime failure is
reported rather than silently switching implementations.

Norm and Attention replacements occur after FP32 weight loading and before
online INT8 conversion, device-only placement, temporal patching, and residency
snapshots. Startup logs report adapter installation counts; RoPE dispatch logs
identify its backend and selection reason. CPU tests cover the switch, initialization order,
Attention/RoPE contracts, and mixed parameter residency. Combined NPU operator
accuracy, video quality, memory, and latency gains still require A2/A3
validation; CPU test results do not establish those gains.
