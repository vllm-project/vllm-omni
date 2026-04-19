# Qwen-Image-Layered — 2x RTX 5880 48GB

> Offline layered image decomposition on two 48 GB GPUs

## Summary

- Vendor: Qwen
- Model: `Qwen/Qwen-Image-Layered`
- Task: Layered image decomposition (one input image → multiple RGBA layers)
- Mode: Offline inference with the shared image-edit runner
- Hardware: 2x NVIDIA RTX 5880 48GB
- Maintainer: Community

## When to use this recipe

Use this recipe when you do **not** have an 80 GB-class GPU but have two 48 GB
GPUs. The checkpoint is about 45 GB, which exceeds a single 48 GB card once
runtime overhead is included. Tensor parallelism shards the DiT across two
GPUs. Model-level CPU offload is required as well: tensor parallelism does
**not** shard the Qwen2.5-VL text encoder, so each rank would otherwise keep a
full ~30 GB encoder copy on device.

This recipe qualifies one offline profile. Online serving of the same model is
documented separately and is not re-measured here.

## Supported model contract

| Task | Required input | Qualified output | Entrypoint |
| --- | --- | --- | --- |
| Layer decomposition | Exactly one image; prompt may be empty | Exactly `layers` RGBA PNGs | [`image_edit.py`](../../examples/offline_inference/image_to_image/image_edit.py) |

| Parameter | Supported values | Notes |
| --- | --- | --- |
| `layers` | 3–10 inclusive; default 4 | The OpenAI image/chat APIs reject values outside this range |
| `resolution` | `640` or `1024`; default `640` | Bucket used to compute output `H×W` from the input aspect ratio |
| `color-format` | `RGBA` | Required for this model |
| Input images | Exactly one | The pipeline rejects multiple images |

Output geometry is not the raw input size. With `resolution=640` and a
720×1280 input, the pipeline produces 480×864 layers.

| Profile | Devices | Purpose | Qualification |
| --- | ---: | --- | --- |
| TP=2 + model-level CPU offload | 2 | Fit the model on 48 GB cards | Command below; performance numbers pending re-measurement on current main |

## References

- Model card: <https://huggingface.co/Qwen/Qwen-Image-Layered>
- Offline image-to-image guide:
  [`docs/user_guide/examples/offline_inference/image_to_image.md`](../../docs/user_guide/examples/offline_inference/image_to_image.md)
- Online layered serving example:
  [`docs/user_guide/examples/online_serving/image_to_image.md`](../../docs/user_guide/examples/online_serving/image_to_image.md)
- Shared runner:
  [`examples/offline_inference/image_to_image/image_edit.py`](../../examples/offline_inference/image_to_image/image_edit.py)
- CPU offload:
  [`docs/user_guide/diffusion/cpu_offload.md`](../../docs/user_guide/diffusion/cpu_offload.md)
- Tensor parallelism:
  [`docs/user_guide/diffusion/parallelism/tensor_parallel.md`](../../docs/user_guide/diffusion/parallelism/tensor_parallel.md)
- Related discussion:
  [#2905](https://github.com/vllm-project/vllm-omni/pull/2905)

## Hardware

- Accelerator model and per-device memory: NVIDIA RTX 5880-Ada-48Q, 49,152 MiB
- Number of devices: 2
- Device interconnect: PCIe (no NVLink on the qualification host)
- Host memory: 377 GiB on the qualification host. CPU offload stages the
  ~30 GB text encoder in host RAM; hosts with only ~64 GB class memory are
  not qualified here.
- Qualification scope: BF16, batch size 1, 50 denoising steps,
  `resolution=640`, `layers=4`, empty prompt, TP=2, model-level CPU offload.
  Similar 48 GB cards (for example RTX A6000 or L40S) are not separately
  qualified.

## Software environment

- OS: Ubuntu 22.04
- Python: 3.12 (repository current requirement)
- Driver / runtime: NVIDIA Driver 570.172.18, CUDA 12.8
- vLLM: 0.29.0, matching the current vLLM-Omni development line
- vLLM-Omni version or commit: current `main` checkout after rebase

Install from source on the 0.29 line, as in
[`docs/getting_started/installation/gpu.md`](../../docs/getting_started/installation/gpu.md).
The default vLLM 0.29.0 wheel targets CUDA 13.0; if the host driver cannot
run that variant, install a CUDA 12.x-compatible vLLM wheel or build vLLM
from source before `uv pip install -e .`.

## Command

Run from the repository root. Replace `input.jpg` with a local RGBA-capable
image (JPEG is converted to RGBA by the runner):

```bash
python examples/offline_inference/image_to_image/image_edit.py \
    --model Qwen/Qwen-Image-Layered \
    --image input.jpg \
    --prompt "" \
    --output layered \
    --num-inference-steps 50 \
    --layers 4 \
    --resolution 640 \
    --color-format RGBA \
    --tensor-parallel-size 2 \
    --enable-cpu-offload \
    --enable-diffusion-pipeline-profiler
```

Key parameters:

| Parameter | Value | Purpose |
| --- | --- | --- |
| `--tensor-parallel-size 2` | 2 | Shard the DiT transformer across two GPUs |
| `--enable-cpu-offload` | flag | Model-level sequential offload: swap the Qwen2.5-VL text encoder and DiT at phase boundaries |
| `--layers 4` | 4 | Number of RGBA layers to emit (supported range 3–10; default 4) |
| `--resolution 640` | 640 | Output-size bucket (`640` or `1024`) |
| `--enable-diffusion-pipeline-profiler` | flag | Print stage timings (`text_encoder.forward`, `vae.encode`, `diffuse`, `vae.decode`) |

`--cfg-scale` only takes effect together with `--negative-prompt`. This
profile uses an empty prompt and does not pass either flag.

## Verification

After the command finishes:

```bash
ls -lh layered_0.png layered_1.png layered_2.png layered_3.png
```

Expected: four RGBA PNG files. For a 720×1280 input at `resolution=640`, each
layer is 480×864. Inspect mode and size with:

```bash
python - <<'PY'
from pathlib import Path
from PIL import Image

for path in sorted(Path(".").glob("layered_*.png")):
    with Image.open(path) as image:
        image.load()
        print(f"{path.name}: format={image.format} mode={image.mode} size={image.size}")
        assert image.format == "PNG"
        assert image.mode == "RGBA"
PY
```

## Notes

- Memory: with TP=2 and CPU offload, weights previously loaded at ~15.7 GiB
  per GPU and peak reserved memory was ~34.2 GB per GPU. Re-measure these
  numbers on the current checkout before treating them as current.
- `--tensor-parallel-size 2` is required on 48 GB cards. The full model does
  not fit on one 48 GB device.
- `--enable-cpu-offload` is strongly recommended. Without it, each rank keeps
  a full text-encoder copy and per-GPU usage previously rose to ~41.8 GiB,
  leaving little headroom for denoising and VAE decode. See the
  [tensor-parallel limitation](../../docs/user_guide/diffusion/parallelism/tensor_parallel.md)
  that the text encoder is not sharded.
- `--enable-cpu-offload` remains the compatibility flag for model-level
  (`mode="module"`) offload. Do not combine it with `--enable-layerwise-offload`
  on this profile; the offload strategies are mutually exclusive.
- If VAE decode is tight on memory, add `--vae-use-tiling`.
- Increasing `--layers` (within 3–10) mainly grows the denoising latent; text
  encoding stays essentially unchanged because it is dominated by CPU↔GPU
  transfer of the encoder.

## Performance

Numbers below are from the original April 2026 qualification on this hardware
(`vLLM-Omni` `a683b1dd`, `--layers 4`, 50 steps, 720×1280 input). They are
**not** yet re-measured on current `main` and should be replaced after the
command above is re-run with `--enable-diffusion-pipeline-profiler`.

| Phase | Time | Notes |
| --- | --- | --- |
| Pre-processing (VAE encode) | ~38 ms | Image → latent |
| Text encoding (incl. CPU↔GPU offload) | ~24 s | Qwen2.5-VL encoder moved from CPU to GPU, executed, then moved back |
| Denoising (50 steps) | ~61 s | ~1.23 s/step |
| VAE decode + post-processing | < 1 s | Latent → output images |
| **End-to-end total** | **~85.9 s** | |
| Peak GPU memory | 34.23 GB | Reserved per GPU; dominated by model weights |

## Supported features

| Feature | Status for this profile | Shared guide |
| --- | --- | --- |
| Tensor parallel | Qualified at TP=2 | [Tensor parallel](../../docs/user_guide/diffusion/parallelism/tensor_parallel.md) |
| Model-level CPU offload | Required; `--enable-cpu-offload` | [CPU offload](../../docs/user_guide/diffusion/cpu_offload.md) |
| Pipeline profiler | Enabled in the command above | [Diffusion pipeline profiler](../../docs/user_guide/diffusion_features.md) |
| VAE tiling | Available; not required on this 48 GB pair | [VAE parallelism](../../docs/user_guide/diffusion/parallelism/vae_parallelism.md) |
| Cache-DiT / TeaCache | Supported by the model, not qualified here | [Feature matrix](../../docs/user_guide/diffusion_features.md) |
| CFG-Parallel | Supported when `cfg_scale > 1` and a negative prompt is set; not used here | [CFG parallel](../../docs/user_guide/diffusion/parallelism/cfg_parallel.md) |
| Layerwise / distributed layerwise offload | Not qualified on this profile | [CPU offload](../../docs/user_guide/diffusion/cpu_offload.md) |
| Quantization / step execution | Not supported for this model | [Feature matrix](../../docs/user_guide/diffusion_features.md) |
