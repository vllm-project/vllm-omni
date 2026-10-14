# SenseNova-U1

> Unified image generation and understanding

## Summary

- Vendor: SenseNova
- Model: `SenseNova/SenseNova-U1-8B-MoT`
- Task: text2img, img2img, img2text (visual understanding), text2text (chat)
- Mode: Offline inference, Online serving (OpenAI-compatible API)
- Maintainer: Community

## When to use this recipe

Use this recipe to run SenseNova-U1-8B-MoT via vLLM-Omni. SenseNova-U1 is a
unified Qwen3-based LLM with Mixture-of-Tokenizers (MoT) attention that handles
text encoding, optional chain-of-thought reasoning, flow-matching image
denoising, and visual understanding in a single pipeline — no separate text
encoder or VAE needed. It supports four task modalities: text-to-image,
image-to-image editing (with dual CFG), image-to-text understanding, and
text-to-text chat.

## References

- Offline text-to-image:
  [`examples/offline_inference/text_to_image/text_to_image.py`](../../examples/offline_inference/text_to_image/text_to_image.py)
- Offline image-to-image:
  [`examples/offline_inference/image_to_image/image_edit.py`](../../examples/offline_inference/image_to_image/image_edit.py)
- Online serving:
  [`examples/online_serving/sensenova_u1/`](../../examples/online_serving/sensenova_u1/)
- E2E tests:
  [`tests/e2e/offline_inference/test_sensenova_u1_text2img.py`](../../tests/e2e/offline_inference/test_sensenova_u1_text2img.py),
  [`tests/e2e/offline_inference/test_sensenova_u1_img2img.py`](../../tests/e2e/offline_inference/test_sensenova_u1_img2img.py)
- HuggingFace model page:
  [SenseNova/SenseNova-U1-8B-MoT](https://huggingface.co/SenseNova/SenseNova-U1-8B-MoT)

## A3B (MoE) variant

[sensenova/SenseNova-U1-A3B-MoT](https://huggingface.co/sensenova/SenseNova-U1-A3B-MoT)
runs on this same recipe and the same `SenseNovaU1Pipeline` — the checkpoint keeps
`model_type: neo_chat`, so it resolves without `--model-class-name`. Only the `--model`
id changes.

The backbone is Qwen3-MoE instead of dense Qwen3, and both MoT branches are sparse:
`mlp` routes over 128 experts and `mlp_mot_gen` over 32, top-8 each, with per-expert
width `moe_intermediate_size=768`. `SenseNovaU1Config` picks `SenseNovaU1MoELLMConfig`
whenever `llm_config` carries `num_experts`, so the dense 8B path is untouched. The
gen-path knobs (`gen_num_experts`, `gen_num_experts_per_tok`, `gen_moe_intermediate_size`)
fall back to their understanding-path counterparts when absent.

Experts run through vLLM's `FusedMoE`, whose `moe_forward` op resolves its layer through
`vllm.forward_context` rather than the diffusion runner's own context, so
`SenseNovaU1ForCausalLM.forward` enters that context when the model has MoE layers.

### Memory

The weights dominate, and every expert stays resident even though only the top-8 of each
tower run per token -- the "A3B" in the name is the ~4B activated, not the 38.7B stored.
Counted from the checkpoint's safetensors headers:

| Component | Params | BF16 |
| --- | --- | --- |
| Understanding-path experts (128/layer) | 28.99 B | 54.0 GiB |
| Generation-path experts (32/layer) | 7.25 B | 13.5 GiB |
| Attention (dual tower) | 1.81 B | 3.4 GiB |
| Embedding + LM head | 0.62 B | 1.2 GiB |
| Routers, norms, vision and flow-matching heads | 0.06 B | 0.1 GiB |
| Total | 38.74 B | 72.2 GiB |

At 1024x1024 the 72.9 GiB peak is those 72.2 GiB of weights plus about 0.7 GiB of
activations and workspace. The 2048x2048 edit, the largest activation case below, peaks at
74.6 GiB, 2.4 GiB above the weights. Text-to-image only routes through the generation tower,
leaving the 54 GiB of understanding-path experts idle; that tower is reached by the text
prefix, think mode, `img2text` and `text2text`. One 96 GB card fits TP=1, and TP=2 halves
the per-GPU footprint.

### Measured (1x H20 96GB, BF16)

Fixed case used for the stage split and the torch.profiler kernel table:

- Prompt: `Close portrait of an elderly woman by a farmhouse window, warm natural light.`
- 1024x1024, 50 Euler steps, think off, seed 42, `cfg_scale` 4.0, `cfg_norm` none,
  `timestep_shift` 3.0, `t_eps` 0.02
- Python 3.12, vLLM 0.30.0, torch 2.13.0+cu130
- Peak GPU memory is the reserved high-water mark the runner records (`peak_memory_mb`)

| | Time | Memory |
| --- | --- | --- |
| Weights after load | ~13 s | 72.2 GiB |
| Pipeline `forward` | 7.68 s | 72.9 GiB peak |

`--enable-diffusion-pipeline-profiler` on that request:

| Stage | Time | Share |
| --- | --- | --- |
| Text prefix (`_t2i_prefix_forward`, understanding tower) | 0.06 s | 0.7% |
| Denoising loop (`_run_denoising_loop`, generation tower, 50 steps) | 7.56 s | 98.5% |
| Pipeline `forward` | 7.68 s | 100% |

There is no VAE. The loop ends in unpatchify / denorm / PIL, which is inside the denoise
row and is a few tens of milliseconds at 1024x1024.

A warm 8-step generate at the same resolution, wrapped in `torch.profiler` after one
unprofiled warmup request, spent 1.19 s of self-CUDA time as:

| Kernel family | Self CUDA | Share | What it is |
| --- | --- | --- | --- |
| `fused_moe_kernel` | 576 ms | 48% | top-8 expert GEMMs on the 32-expert generation tower |
| FlashAttention `flash_fwd_kernel` | 222 ms | 19% | dual-tower attention |
| `nvjet` GEMMs | ~233 ms | 20% | `qkv_proj` / `o_proj` (and the small dense leftovers) |
| `qk_norm_rope_kernel` | 58 ms | 5% | fused RMSNorm + RoPE |
| MoE routing / `act_and_mul` / elementwise | ~40 ms | 3% | top-k, SwiGLU, residual |

So a denoise step is about half FusedMoE and about two-fifths attention + dense GEMM.
`--cfg-parallel-size 2` on the same 8-step t2i request (2-branch `cond`+`uncond`)
is 0.95 s vs 1.48 s on one GPU (~1.56x). Img2img CFG-P=2 also completes for both
branch counts at 8-step 1024x1024: 2-way (`img_cfg_scale=1`, `cond`+`img_cond`)
1.09 s; 3-way (`cfg_scale=4`, `img_cfg_scale=2`, `cond`+`img_cond`+`uncond`)
1.74 s (3 branches on 2 ranks dispatch as `[[0, 2], [1]]`). Cache-DiT and
layerwise CPU offload do not take effect on A3B; see
[diffusion features](../../docs/user_guide/diffusion_features.md) note 8.

Other configurations on the same machine (second run, warm kernels):

| Case | Total | Peak GPU memory |
| --- | --- | --- |
| text2img 1024x1024, 50 steps, think on | 9.16 s | 73.0 GiB |
| text2img 1024x1024, 50 steps, think off, TP=2 | 5.64 s | 36.7 GiB per GPU |
| img2img 2048x2048 output, 25 steps, think off | 20.93 s | 74.6 GiB |

Think-on spends 1.37 s in `_generate_think` (215 tokens). At the same seed TP=2 differs
from TP=1 only numerically (MAE 2.97/255, identical composition) from the changed
reduction order.

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
    --model sensenova/SenseNova-U1-A3B-MoT \
    --prompt "Close portrait of an elderly woman by a farmhouse window, warm natural light." \
    --width 1024 --height 1024 \
    --seed 42 --num-inference-steps 50 --cfg-scale 4.0 \
    --extra-body '{"think": false, "cfg_norm": "none", "timestep_shift": 3.0, "t_eps": 0.02}' \
    --output sensenova_a3b_t2i.png
```

`img2text` and `text2text` go through the server, as for the dense checkpoint:

```bash
vllm serve sensenova/SenseNova-U1-A3B-MoT --omni --port 8091

python examples/online_serving/sensenova_u1/openai_chat_client.py \
    -s http://127.0.0.1:8091 -m img2text -i input.png -p "Describe this image."
```

All four modalities were verified on A3B: text2img (with and without think), img2img
editing, img2text and text2text.

## Hardware Support

## GPU

### 1x H200 (144GB)

#### Environment

- OS: Linux
- Python: 3.12
- Driver / runtime: NVIDIA 590.48.01, CUDA 13.1
- vLLM-Omni version: 0.18.1.dev

#### Command

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
    --model SenseNova/SenseNova-U1-8B-MoT \
    --prompt "Close portrait of an elderly woman by a farmhouse window, textured skin, gentle smile, warm natural light, emotional documentary look. The portrait should feel polished and natural, with sharp eyes, realistic skin texture, accurate facial anatomy, and premium lighting that keeps the face as the main focus." \
    --width 1536 --height 2720 \
    --seed 42 --num-inference-steps 50 \
    --cfg-scale 4.0 \
    --extra-body '{"think": true, "cfg_norm": "none", "timestep_shift": 3.0, "t_eps": 0.02}' \
    --output outputs/sensenova_u1_output.png
```

#### Verification

```bash
pytest -s -v tests/e2e/offline_inference/test_sensenova_u1_text2img.py \
    -m "advanced_model" --run-level "advanced_model"
```

#### Notes

- E2E latency: **32.1s** (1536×2720, 50 steps, think mode, CFG scale 4.0)
- Peak VRAM: **35.9 GB** reserved, 35.1 GB allocated
- Model loading: 32.8 GiB, 8.7s
- No deploy YAML needed — the engine auto-generates a single-stage diffusion config.
- Think mode (`--think`) is recommended for higher image quality.

#### Text-to-Image via the shared example

SenseNova-U1 is registered with the shared text-to-image example. Forward
SenseNova-specific generation parameters as a JSON object through `--extra-body`:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
    --model SenseNova/SenseNova-U1-8B-MoT \
    --prompt "A beautiful sunset over mountains" \
    --width 2048 --height 2048 \
    --num-inference-steps 50 \
    --seed 42 \
    --extra-body '{"think": true, "cfg_scale": 4.0, "cfg_norm": "none", "timestep_shift": 3.0, "t_eps": 0.02}' \
    --output sensenova_text2img.png
```

The `--extra-body` keys are filtered against the model's declared
`extra_body_params` (see
[`vllm_omni/model_extras/sensenova_u1.py`](../../vllm_omni/model_extras/sensenova_u1.py)):
`think`, `cfg_scale`, `cfg_norm`, `timestep_shift`, `t_eps`, `img_cfg_scale`,
and `max_tokens`. No deploy YAML is needed — the engine auto-generates a
single-stage diffusion config.

#### Image-to-Image Editing (img2img)

```bash
python examples/offline_inference/image_to_image/image_edit.py \
    --model SenseNova/SenseNova-U1-8B-MoT \
    --prompt "Turn this into an oil painting" \
    --image input.png \
    --resolution 2048 \
    --seed 42 --num-inference-steps 50 \
    --cfg-scale 4.0 \
    --extra-args '{"think": true, "img_cfg_scale": 1.0, "cfg_norm": "none", "timestep_shift": 3.0}' \
    --output outputs/sensenova_u1_edit.png
```

- img2img uses dual CFG: `--cfg-scale` controls text guidance, while
  `img_cfg_scale` in `--extra-args` controls image guidance (1.0 = image CFG
  disabled).
- Pass multiple `--image` paths for multi-reference editing when the underlying
  pipeline supports them.

#### Image Understanding (img2text)

```bash
vllm serve SenseNova/SenseNova-U1-8B-MoT --omni --port 8091
python examples/online_serving/sensenova_u1/openai_chat_client.py \
    --prompt "Describe this image in detail" \
    --modality img2text \
    --image-url photo.jpg
```

#### Text-to-Text Chat (text2text)

```bash
vllm serve SenseNova/SenseNova-U1-8B-MoT --omni --port 8091
python examples/online_serving/sensenova_u1/openai_chat_client.py \
    --prompt "Explain the theory of relativity in simple terms" \
    --modality text2text
```

- For img2text and text2text, use the online chat-compatible example. The
  offline generic image examples intentionally focus on image-producing tasks.

### 2x H200 (144GB) — TP=2

#### Environment

- OS: Linux
- Python: 3.12
- Driver / runtime: NVIDIA 590.48.01, CUDA 13.1
- vLLM-Omni version: 0.18.1.dev

#### Command

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
    --model SenseNova/SenseNova-U1-8B-MoT \
    --prompt "Close portrait of an elderly woman by a farmhouse window, textured skin, gentle smile, warm natural light, emotional documentary look. The portrait should feel polished and natural, with sharp eyes, realistic skin texture, accurate facial anatomy, and premium lighting that keeps the face as the main focus." \
    --width 1536 --height 2720 \
    --seed 42 --num-inference-steps 50 \
    --cfg-scale 4.0 \
    --extra-body '{"think": true, "cfg_norm": "none", "timestep_shift": 3.0, "t_eps": 0.02}' \
    --tensor-parallel-size 2 \
    --output outputs/sensenova_u1_output.png
```

#### Verification

Verify the output image is generated at `outputs/sensenova_u1_output.png`
with the expected 1536×2720 resolution.

#### Notes

- E2E latency: **28.3s** (1536×2720, 50 steps, think mode, CFG scale 4.0)
- Peak VRAM (per GPU): **18.2 GB** reserved, 17.9 GB allocated
- Model loading: 16.5 GiB per GPU, 7.0s
- TP=2 provides ~12% speedup over TP=1; limited by serial CFG dual-forward
  and communication overhead.
- The LLM transformer uses `QKVParallelLinear` and `MergedColumnParallelLinear`
  for fused QKV and gate/up projections with TP support.

### 1x AMD MI300X 192GB

#### Environment

- OS: Linux 6.8.0-134-generic, x86_64
- Container: official ROCm image built from `docker/Dockerfile.rocm`
- Python: 3.12.13
- PyTorch: 2.11.0+gitd0c8b1f
- Driver / runtime: AMD 6.19.14.31400000 / ROCm 7.2.53211
- GPU: one AMD Instinct MI300X, `gfx942:sramecc+:xnack-`, 191.69 GiB visible HBM
- vLLM version: 0.27.0+rocm723
- vLLM Omni version or commit: `a704c8759c96e123c0d7c89b11f120b1c0f120cf`
- Installed vLLM Omni package metadata: `0.27.0rc2.dev44+g55abdade9.rocm`

#### Command

```bash
python3 examples/offline_inference/text_to_image/text_to_image.py \
    --model SenseNova/SenseNova-U1-8B-MoT \
    --prompt "Close portrait of an elderly woman by a farmhouse window, textured skin, gentle smile, warm natural light, emotional documentary look." \
    --width 1536 \
    --height 2720 \
    --seed 42 \
    --num-inference-steps 50 \
    --cfg-scale 4.0 \
    --extra-body '{"think": true, "cfg_norm": "none", "timestep_shift": 3.0, "t_eps": 0.02}' \
    --enable-diffusion-pipeline-profiler \
    --log-stats \
    --output sensenova_u1_mi300x.png
```

#### Verification

The command completed and wrote a valid 1536 by 2720 RGB PNG.

#### Notes

- Generation with 50 inference steps took 34.011 seconds.
- Model loading used 32.774 GiB and took 13.987 seconds.
- The internal profiler recorded 35.67 GB reserved and 35.10 GB allocated for the request.
- The highest one second whole device memory sample was 37.93 GiB.

## Online Serving

SenseNova-U1 supports all four modalities via the OpenAI-compatible
`/v1/chat/completions` API.

### Launch

```bash
vllm serve SenseNova/SenseNova-U1-8B-MoT --omni --port 8091
```

### Send Requests

```bash
cd examples/online_serving/sensenova_u1

# Text-to-image
python openai_chat_client.py \
    --prompt "A beautiful sunset" --modality text2img

# Image-to-image editing
python openai_chat_client.py \
    --prompt "Turn this into an oil painting" \
    --modality img2img --image-url input.jpg

# Image understanding
python openai_chat_client.py \
    --prompt "Describe this image" \
    --modality img2text --image-url photo.jpg

# Text chat
python openai_chat_client.py \
    --prompt "What is the capital of France?" \
    --modality text2text
```

For full API documentation and curl examples, see
[`examples/online_serving/sensenova_u1/README.md`](../../examples/online_serving/sensenova_u1/README.md).
