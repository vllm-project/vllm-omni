# SenseNova-U1.5

> Unified image generation and understanding, with an official 8-step distilled LoRA

## Summary

- Vendor: SenseNova
- Model: `sensenova/SenseNova-U1.5-8B-MoT`
- LoRA: `sensenova/SenseNova-U1.5-8B-MoT-LoRAs` (`SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors`)
- Task: text2img, img2img, img2text (visual understanding), text2text (chat)
- Mode: Offline inference, Online serving (OpenAI-compatible API)
- Maintainer: Community

## When to use this recipe

U1.5 runs on the same `SenseNovaU1Pipeline` as U1 — the checkpoint keeps
`model_type: neo_chat`, so it resolves without `--model-class-name`. Relative to U1 the
config flips two fields, `use_pixel_head` to `true` (the flow-matching head becomes a
`ConvDecoder`) and `noise_scale_max_value` from 8.0 to 16.0; both are already read by
`SenseNovaU1Config`. The checkpoint is 13 shards / 50.2 GB on disk, 30.3 GB of that fp32, and
loads to roughly 34 GB in bf16.

Use this recipe for U1.5 specifically, including its distilled few-step LoRA. For U1 see
[SenseNova-U1](SenseNova-U1.md).

## Hardware Support

### GPU

#### 1x A800 80GB

- 1x NVIDIA A800 80GB PCIe with an Intel Xeon Gold 6336Y, driver 595.84, Linux
- Python 3.12.13, vLLM 0.28.0, torch 2.13.0+cu130, CUDA 13.0, diffusers 0.40.0,
  transformers 5.14.1, flashinfer 0.6.16.post3, BF16, TP=1
- Peak GPU memory: 34.3 GB at 1024x1024, 36.4 GB at 1536x2720

##### Text-to-Image

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
    --model sensenova/SenseNova-U1.5-8B-MoT \
    --prompt "Close portrait of an elderly woman by a farmhouse window, warm natural light." \
    --width 1024 --height 1024 \
    --seed 42 --num-inference-steps 50 --cfg-scale 4.0 \
    --extra-body '{"think": false, "cfg_norm": "none", "timestep_shift": 3.0, "t_eps": 0.02}' \
    --output sensenova_u15_t2i.png
```

Think mode (`"think": true`) is recommended for higher image quality.

##### Image-to-Image Editing

```bash
python examples/offline_inference/image_to_image/image_edit.py \
    --model sensenova/SenseNova-U1.5-8B-MoT \
    --prompt "Turn this into an oil painting" \
    --image input.png --resolution 1024 \
    --seed 42 --num-inference-steps 25 --cfg-scale 4.0 \
    --extra-args '{"think": true, "img_cfg_scale": 1.0, "cfg_norm": "none", "timestep_shift": 3.0}' \
    --output sensenova_u15_edit.png
```

##### 8-step distilled LoRA

The distilled LoRA is fused into the generation tower at load time:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
    --model sensenova/SenseNova-U1.5-8B-MoT \
    --lora-path SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors --lora-backend distill \
    --prompt "Close portrait of an elderly woman by a farmhouse window, warm natural light." \
    --width 1024 --height 1024 \
    --seed 42 --num-inference-steps 8 --cfg-scale 1.0 \
    --extra-body '{"think": false, "cfg_norm": "none", "timestep_shift": 3.0, "t_eps": 0.02}' \
    --output sensenova_u15_lora8.png
```

**Use `--cfg-scale 1.0` with this LoRA.** It is distilled with DMD and runs without
classifier-free guidance; the default `4.0` applies guidance twice and produces a
blown-out, posterised image.

Online FP8 does not support the 8-step distilled LoRA.
Use BF16 without `--quantization fp8` for this LoRA.

##### Online serving

```bash
vllm serve sensenova/SenseNova-U1.5-8B-MoT --omni --port 8091

python examples/online_serving/sensenova_u1/openai_chat_client.py \
    -s http://127.0.0.1:8091 -m img2text -i input.png -p "Describe this image."
```

`-s` takes the base URL; the client appends `/v1` itself.

##### Mixed-traffic readiness warmup

For a deployment alternating image generation and text/vision chat, opt in to a
bounded warmup profile for the shapes it serves most often. This profile and the
measurements below were validated only on SenseNova-U1.5-8B-MoT. SenseNova-U1
and U1-A3B share the pipeline and inherit the key, but have no benchmark
evidence for this profile:

```bash
TORCH_LOGS=recompiles VLLM_LOGGING_LEVEL=DEBUG \
vllm serve sensenova/SenseNova-U1.5-8B-MoT --omni --port 8091 \
  --additional-config \
  '{"sensenova_mixed_warmup":{"resolutions":[[1024,1024],[1536,1536]],"text_to_text":true,"image_to_text":true}}'
```

The existing generic 512x512 image-conditioned dummy runs first. The extra
profile then runs one-token text-to-text and image-to-text requests and one-step,
think-off text-to-image requests at the listed resolutions. The text warmup
also exercises the existing paged AR decode runner, so its first graph capture
happens before readiness. This profile does not implement or change the decode
graph capture machinery in #7666. Think-on image generation can reuse that
decode graph; it is not separately prewarmed by this profile. Shapes not listed
continue through the normal request path.

The profile is off by default because each additional shape increases startup
time and may increase peak or reserved GPU memory. `resolutions` accepts at most
three distinct `[width,height]` pairs, each divisible by the model's patch/merge
factor and no larger than 4096² pixels. `text_to_text` and `image_to_text` are
optional booleans. For image shapes, `cfg_scale` defaults to 4.0 on the
base model and 1.0 when the distilled LoRA is fused at load time; it can also
be set explicitly in the warmup object. Set only the paths and resolutions
your workload actually uses. A failed explicitly requested warmup fails startup
so the server does not
silently claim to be ready without the selected shapes.

To measure an alternating sequence, start the server with its output redirected
to a log and run:

```bash
python benchmarks/diffusion/sensenova_mixed.py \
  --base-url http://127.0.0.1:8091 --rounds 2 --steps 2 \
  --server-log /path/to/server.log --output /path/to/mixed-result.json
```

Use `--case` repeatedly to select the exact sequence; supported forms are
`t2i:WxH`, `t2i-think:WxH`, `t2t`, and `i2t`. The report includes per-request
latency, P50/P100, and serving-time recompile/graph-capture log counts. Keep
server revision, GPU, compile-cache state, and sequence identical in the
baseline and warmup runs; record startup time and GPU memory separately. For
the distilled LoRA profile, pass `--steps 8 --cfg-scale 1.0` to the benchmark.

In one cold-cache H20-3e BF16 comparison (vLLM 0.30.0, torch 2.13.0+cu132,
model revision `9feeeab8`, TP=1), the default server ran vLLM-Omni commit
`a038b3817` and the mixed-warmup server ran committed code `ed5c7a3c8`.
Each server used its own CUDA, Triton, Inductor, and vLLM cache directory.
The [JSON reports, startup timestamps, GPU process memory snapshots, and
server-log excerpts](../../benchmarks/diffusion/evidence/sensenova_u15_mixed_h20/README.md)
are available for review. The alternating sequence was `t2i:1024x1024`,
`t2t`, `i2t`, `t2i:1536x1536`, repeated twice; image requests used seed 42,
two denoising steps, and CFG 4.0, while text requests used `max_tokens=2`.
These short requests isolate first-hit overhead rather than represent
production image quality or throughput.

| Metric | Default warmup | Mixed warmup |
| --- | ---: | ---: |
| Startup to `/health` | 26.5 s | 88.6 s |
| GPU process memory at readiness | 34,432 MiB | 35,600 MiB |
| Mixed-request P50 | 1.3735 s | 0.6585 s |
| Mixed-request P100 | 58.571 s | 2.226 s |
| First text-to-text request | 58.571 s | 0.065 s |
| Decode graphs captured after readiness | 1 | 0 |
| `torch.compile` recompilations after readiness | 0 | 0 |

The first text request dominated the baseline tail and captured a paged decode
graph. With mixed warmup, that capture occurred during readiness; the text
warmup itself took 58.65 s in the isolated-cache run. Regional compilation
was skipped for this model, so these measurements do not establish a
`torch.compile` speedup. Memory values are `nvidia-smi` process allocations
at readiness, not peak-memory measurements. In an earlier development-run
smoke test, a think-on 1024x1024 request completed in 2.709 s without a
serving-time capture or recompile, and an uncovered 1280x1280 request
completed in 1.765 s; those two requests were not rerun at `ed5c7a3c8`.

With the 8-step distilled LoRA revision `f33b8fe` fused, the same sequence
used eight denoising steps and CFG 1.0. The warmup profile automatically chose
CFG 1.0 for its image requests. Separate cold-cache runs on the same H20-3e
gave:

| Metric | Default warmup | Mixed warmup |
| --- | ---: | ---: |
| Startup to `/health` | 35.5 s | 92.6 s |
| GPU process memory at readiness | 34,432 MiB | 35,600 MiB |
| Mixed-request P50 | 2.0915 s | 1.0275 s |
| Mixed-request P100 | 57.790 s | 4.037 s |
| Decode graphs captured after readiness | 1 | 0 |
| `torch.compile` recompilations after readiness | 0 | 0 |

##### Online FP8 quantization

`--quantization fp8` quantizes only eligible attention and MLP linears in the
`language_model` understanding and generation branches at load time. Other layers
retain their existing precision.

On the tested A800 (SM80) setup with vLLM 0.29.0,
`VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel` is **required** for online
FP8 to start successfully. Without it, the selected CUTLASS kernel fails during
the startup dummy run with `RuntimeError: cutlass_scaled_mm_sm80_epilogue`.
Disabling this kernel selects Marlin W8A16 (FP8 weights with BF16 activations).

```bash
VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel \
vllm serve sensenova/SenseNova-U1.5-8B-MoT --omni \
    --quantization fp8 --port 8091
```

#### Measured BF16 vs FP8 performance (1x A800 80GB, 28 steps, P95 over 10 prompts)

vLLM 0.29.0, BF16 base dtype, seed 42, TORCH_SDPA, eager execution, TP=1, batch size 1,
paged decode off, no LoRA. Each resolution uses the same 10 prompts for BF16 and FP8.
Both runs set `VLLM_DISABLED_KERNELS=CutlassFP8ScaledMMLinearKernel`.

| Resolution | Peak reserved memory (BF16 → FP8) | Step latency (BF16 → FP8) |
| --- | --- | --- |
| 512x512 | 33.166 → 18.697 GiB | 63.674 → 81.136 ms |
| 1024x1024 | 33.555 → 18.561 GiB | 187.867 → 229.115 ms |

Step latency is the P95 of per-request mean step times, excluding warmup requests; FP8 reduced
memory usage but increased step latency.

#### Measured BF16 latency (1x A800 80GB, 25 steps, median of 3 after a warmup)

| Resolution | Step latency | Total |
| --- | --- | --- |
| 1024x1024 | 209.5 ms | 5.24 s |
| 1536x1536 | 458.5 ms | 11.46 s |

At 1536x2720 with 50 steps, end-to-end is 43.0 s.

#### Measured latency (1x NVIDIA H200 139GB, single run, no warmup)

Reported by @hsliuustc0106 against PR head `29c090d`, on one reserved CUDA device.

- Python 3.12.13, vLLM 0.28.0, torch 2.13.0+cu130, CUDA 13.0, diffusers 0.40.0,
  transformers 5.14.1, flashinfer 0.6.16.post3, BF16, TP=1
- `sensenova/SenseNova-U1.5-8B-MoT` with `SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors`,
  seed 42, 1024x1024

| Case | Stage latency | Peak GPU memory |
| --- | --- | --- |
| 50 steps, cfg 4.0, think off | 9017 ms | 34384 MiB |
| 8 steps, cfg 1.0, no LoRA control | 624.00 ms, 77.85 ms/step | 34376 MiB |
| 8 steps, cfg 1.0, distilled LoRA | 623.85 ms, 77.98 ms/step | 34364 MiB |
| 50 steps, cfg 4.0, think on, `VLLM_OMNI_SENSENOVA_PAGED_DECODE=1` | 81622 ms | 34594 MiB |

Each row runs the Text-to-Image command above with `--num-inference-steps` and
`--cfg-scale` as listed and `think` set through `--extra-body`; the LoRA row adds
`--lora-path` and `--lora-backend distill`.

Single runs without a preceding warmup, so the think-on row carries the first-request
compile rather than a steady state. The distilled LoRA fused into 168 parameters, and its
image differs from the same-seed no-LoRA control in 1,048,574 of 1,048,576 pixels, MAE 58.62.

```bash
pytest -m "cpu and not cuda" tests/diffusion/models/sensenova_u1/ tests/diffusion/lora/test_loader.py -q
pytest -m "cpu and not cuda" tests/config/test_environment_variables.py tests/diffusion/test_diffusion_worker.py -q
pytest -m cuda tests/diffusion/models/sensenova_u1/test_sensenova_u1_attention_gqa.py \
    tests/diffusion/models/sensenova_u1/test_sensenova_u1_paged_decode.py -q --tb=short
```

71, 13 and 9 passed. Serving reached `/health`; `img2text` returned a 1921-character
description and `text2text` the expected answer.

#### Verification

```bash
pytest -q tests/diffusion/models/sensenova_u1/
```

## Notes

- Both `use_pixel_head` and `noise_scale_max_value` come from `config.json`; no flag is needed.
- The checkpoint carries no `configuration_neo_chat.py`, so the loader logs a few
  "does not appear to have a file named configuration_neo_chat.py" errors during startup and
  then proceeds on the in-tree config. Generation is unaffected.
- The LoRA targets the generation tower only (`*_mot_gen`); understanding-tower weights are
  untouched.
- Autoregressive decode (think, text-to-text and image-to-text) runs on a paged K/V cache under
  a captured CUDA graph. It falls back to the ordinary cache when the device or the bundled
  `flash_attn_varlen_func` cannot support it; set `VLLM_OMNI_SENSENOVA_PAGED_DECODE=0` to force
  that fallback.
- The first request after startup costs about 0.7 s more than the steady state whether the paged
  path is on or off. Measured on one A800 with the inductor, triton and vLLM compile caches all
  cleared, median of three runs: 718 ms above steady with the path on, 679 ms with it off.
- With paged decode enabled, the pipeline owns the decode cache and captured graphs. The first
  decode in a bucket captures its graph; later requests reuse it while the cache remains
  compatible. A think request may enter a larger bucket and capture an additional graph there.
  Dynamically served LoRA adapters disable this cross-request reuse, so their decode cache and
  graphs are built per request. The distilled LoRA fused at load time uses the ordinary reused
  path. If paged decode is unavailable or disabled, decoding falls back to the ordinary cache.
