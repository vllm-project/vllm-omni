# Qwen-Image-2.1

> Text-to-image and image-conditioned generation with Qwen-Image 2.1

## Summary

- Vendor: Qwen
- Model: `Qwen/Qwen-Image-2.1`
- Task: Text-to-image generation and image-conditioned generation (editing)
- Mode: Offline inference and online serving with optional step-wise execution
- Maintainer: Community

> **Model id:** `Qwen/Qwen-Image-2.1` follows the reference diffusers
> documentation and may change before the official release — the final
> checkpoint name is subject to the official release.

## When to use this recipe

Use this recipe as a starting point for running `Qwen/Qwen-Image-2.1`. A single
pipeline class, `QwenImage21Pipeline`, handles both pure text-to-image requests
(no condition image) and image-conditioned requests with up to 4 condition
images: prompt and condition images are encoded together by a Qwen3-VL text
encoder, so editing-style requests go through the same serving path as
text-to-image.

## References

- Related offline examples:
  [`examples/offline_inference/text_to_image/text_to_image.py`](../../examples/offline_inference/text_to_image/text_to_image.py),
  [`examples/offline_inference/image_to_image/image_edit.py`](../../examples/offline_inference/image_to_image/image_edit.py)
- Supported models table:
  [`docs/models/supported_models.md`](../../docs/models/supported_models.md)
- Feature compatibility matrix:
  [`docs/user_guide/diffusion_features.md`](../../docs/user_guide/diffusion_features.md)
- Sibling recipes: [`Qwen-Image.md`](./Qwen-Image.md),
  [`Qwen-Image-Edit.md`](./Qwen-Image-Edit.md)

## Offline Inference

### Text-to-image

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Qwen/Qwen-Image-2.1 \
  --prompt "A ceramic teapot on a wooden table" \
  --output qwen_image_21_t2i.png \
  --num-inference-steps 50 \
  --cfg-scale 1.0
```

### Image-conditioned generation (editing)

```bash
python examples/offline_inference/image_to_image/image_edit.py \
  --model Qwen/Qwen-Image-2.1 \
  --color-format RGBA \
  --seed 42 \
  --image qwen_bear.png \
  --prompt "Let this mascot dance under the moon, surrounded by floating stars" \
  --output qwen_image_21_edit.png \
  --num-inference-steps 50 \
  --cfg-scale 1.0
```

or via the bundled launcher:

```bash
bash examples/offline_inference/image_to_image/run_qwen_image_21.sh
```

Up to 4 condition images can be passed via `--image`:

```bash
python examples/offline_inference/image_to_image/image_edit.py \
  --model Qwen/Qwen-Image-2.1 \
  --color-format RGBA \
  --seed 42 \
  --image input1.png input2.png \
  --prompt "Combine these images into a single scene" \
  --output qwen_image_21_multi.png \
  --num-inference-steps 50 \
  --cfg-scale 1.0
```

Use `--color-format RGBA` to preserve transparency in condition images. The
shared example otherwise loads inputs as RGB. Add `--width` and `--height`
(multiples of 32) to set the output size; when omitted, Qwen-Image 2.1 derives
it from the last condition image's aspect ratio at approximately 1024×1024.
Use `--num-outputs-per-prompt` to generate multiple images and `--vae-use-tiling`
to reduce VAE memory usage. The launcher accepts these options as well.

## Online Serving

Start the server:

```bash
vllm serve Qwen/Qwen-Image-2.1 --omni --port 8091
```

To enable the step-wise runtime:

```bash
vllm serve Qwen/Qwen-Image-2.1 --omni \
  --port 8091 \
  --step-execution \
  --max-num-seqs 8
```

Step execution batches compatible requests at the same KV-cache phase. The
pipeline tags every request with a `batch_compatibility_key` (conditioning
mode plus condition-image layout), so a newly arriving request that would
otherwise join a batch already past its first denoising step is isolated by
the scheduler and deferred until the in-flight batch drains — it no longer
fails the whole batch. `--max-num-seqs` controls batch capacity.

### CUDA Graph decode

Qwen-Image-2.1 automatically uses CUDA Graph for supported fixed-shape denoising
decode steps when `enable_cuda_graph_decode=True` (the default) and
`enforce_eager=False`. Prefill remains eager. This works with both request
execution and `--step-execution`.

To disable graph capture while keeping the rest of the compilation setup:

```bash
vllm serve Qwen/Qwen-Image-2.1 --omni --no-enable-cuda-graph-decode
```

The same control is available as `Omni(..., enable_cuda_graph_decode=False)`
and as `enable_cuda_graph_decode: false` on a stage in the deployment YAML.
`--enforce-eager` / `enforce_eager: true` remains the broader switch: it
disables both graph capture and automatic `torch.compile`. The two
optimizations stack: the DiT blocks are regionally `torch.compile`'d and
graph capture records the compiled (fused) kernels, while inductor's own
cudagraphs stay disabled so the two graph layers never nest. Measured on a
single GB200 (1024×1024, 50 steps, seed 42, BF16, `true_cfg_scale=1.0`,
warmup 1 + median of 3): compile+graph 3.04 s, compile-only 3.2 s,
graph-only 4.0 s, eager 4.3 s end-to-end per image; combo output vs eager is
44.5–45.6 dB PSNR (the same magnitude as pure `torch.compile` fusion
divergence), and graph-only vs eager is bit-identical.

The autoregressive engine's `compilation_config.cudagraph_mode` does not
control this diffusion path; use `enable_cuda_graph_decode` (or
`enforce_eager` for everything) to force eager decode execution.

TP/SP/ring parallelism, HSDP, offload/cache hooks, quantized KV caches,
dynamic LoRA, padded text masks, and a second in-flight request whose cache
aliases an already-owned graph key fall back to eager decode.
Graphs keep separate entries for different image layouts and copy
request-owned prefix K/V into static buffers before replay; each graph key
has a single live owner at a time, so concurrent same-shape requests never
overwrite each other's prefix. Those buffers require additional memory; graph
entries are bounded by the transformer's cache limit.

### Verification

For a direct API smoke test:

```bash
curl http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen-Image-2.1",
    "prompt": "A ceramic teapot on a wooden table",
    "size": "1024x1024",
    "num_inference_steps": 50,
    "true_cfg_scale": 1.0,
    "seed": 42
  }'
```

For image-conditioned generation, pass condition images through the
OpenAI-compatible multimodal endpoint (`/v1/chat/completions` with
`image_url` content parts, or `/v1/images/edits`), same as
[`Qwen-Image-Edit.md`](./Qwen-Image-Edit.md).

## Parameters

| Parameter | Default | Notes |
| --- | --- | --- |
| `true_cfg_scale` | 1.0 | Disables the negative-prompt branch by default. Set above 1 with a `negative_prompt` to enable true CFG. Qwen-Image 2.1 supports true CFG only — there is no `guidance_scale` parameter. |
| `num_inference_steps` | 50 | Number of denoising steps. |
| `height` / `width` | 1024×1024 | Must be multiples of 32. For image-conditioned requests, leaving them unset derives the output size from the last condition image's aspect ratio at ~1024×1024. |
| `negative_prompt` | None | Ignored at the default CFG scale of 1.0; used only when `true_cfg_scale > 1`. |
| condition images | — | Up to 4 input images per request; more than 4 raises an error. |
| `seed` | — | Fix for reproducible outputs. |

The default CFG scale of 1.0 runs only the positive-prompt branch at each
denoising step. To enable negative-prompt guidance, explicitly set
`--cfg-scale` (offline) or `true_cfg_scale` (API) above 1 and provide a
negative prompt. This changes the generated output as well as compute cost.

### Prefix KV cache

The transformer caches the text/condition-image prefix across denoising steps
(a prefill/decode split keyed by CFG branch). The cache is enabled
automatically when the checkpoint declares `causal_condition: true` and is
re-created per generation, so no state leaks between requests. It changes the
compute path — cached prefix vs. full recompute at every step — so exact
numerical reproduction against a no-cache reference (e.g. the diffusers
pipeline) requires matching the cache setting on both sides. The prefix KV
cache is orthogonal to weight quantization and works unchanged under FP8.

### FP8 quantization

Online FP8 quantization of the transformer is supported; the fused
`to_qkv` projections load correctly, and the prefix KV cache path is
unaffected. Only block-internal linears are eligible — the boundary
projections (`img_in`/`txt_in`/`modulation`/`proj_out`/`time_text_embed`) stay
in BF16 by construction:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Qwen/Qwen-Image-2.1 \
  --prompt "A ceramic teapot on a wooden table" \
  --output qwen_image_21_fp8.png \
  --num-inference-steps 50 --cfg-scale 1.0 \
  --quantization fp8 --ignored-layers "img_mlp"
```

`ignored_layers` entries are matched as name patterns for this model (e.g.
`img_mlp` matches every block's `img_mlp.proj`/`img_mlp.gate_layer`/
`img_mlp.out`); exact full prefixes like
`transformer.transformer_blocks.3.attn.to_qkv` also work.

A per-module sensitivity sweep (fixed prompt embeds / latent / timestep, 5
timesteps across the trajectory, relative L2 of the noise prediction vs BF16)
ranks the groups: `img_mlp` is the most sensitive (3.8% max error on its own,
driven by `img_mlp.proj` 2.8% and `img_mlp.out` 2.5%), then `attn.to_out`
(2.7%), then `attn.to_qkv` (1.6%); errors are largest at the late (low-sigma)
steps. End-to-end on 4 fixed prompts at 1024x1024, 50 steps, seed 42
(GB200, single GPU):

| Config | Quantized block linears | Avg PSNR vs BF16 | Min PSNR | Steady-state time/image | Peak memory |
| --- | --- | --- | --- | --- | --- |
| BF16 | 0 / 160 | — | — | 7.2 s | 40.0 GB |
| FP8 all layers | 160 / 160 | 26.1 dB | 19.1 dB | 8.4 s | 33.4 GB |
| FP8, `ignored_layers=["img_mlp"]` | 64 / 160 | 29.7 dB | 21.6 dB | 8.8 s | 38.3 GB |

All-layer FP8 is usable but its composition can drift on some prompts
(fine detail and layout shift, worst case ~19 dB); keeping `img_mlp` in BF16
preserves composition noticeably better and is the recommended setting. On
Blackwell (GB200) FP8 here is a memory optimization, not a speedup: dynamic
per-token activation quantization costs more than the FP8 GEMM saves at these
shapes, so BF16 remains the fastest option.

## Quantization

Online FP8 is supported per component via `quantization_config`:

```python
from vllm_omni import Omni

omni = Omni(
    model="Qwen/Qwen-Image-2.1",
    quantization_config={
        # DiT: skip the sensitive image-stream MLPs if quality regresses.
        "transformer": {"method": "fp8", "ignored_layers": ["img_mlp"]},
        # Qwen3-VL text encoder: only the language-model linear layers are
        # quantized; the vision tower and the (unused) LM head stay BF16.
        "text_encoder": {"method": "fp8"},
    },
)
```

Either component can be quantized on its own. Measured on GB200 at 1024×1024
(seed 42, 50 steps): text-encoder FP8 lowers peak GPU memory from ~41.0 GiB to
~34.4 GiB (−6.6 GiB); adding DiT FP8 reaches ~28.1 GiB. Output quality stays
close to BF16 (T2I PSNR vs BF16 ≈ 28–32 dB with text-encoder FP8; the
image-conditioned edit path ≈ 36 dB). FP8 saves memory but does not speed up
generation on this hardware. The edit path routes condition images through the
BF16 vision tower, so it is unaffected by text-encoder FP8. See
[`docs/user_guide/quantization/fp8.md`](../../docs/user_guide/quantization/fp8.md)
for the scope rules.

### FP8 prefix KV storage

The cached prefix K/V can be stored in FP8 E4M3 (with per-token-per-head fp32
scales), halving the prefix-cache memory — relevant for long prompts (up to
8192 text tokens) and up to 4 condition images (~4096 latent tokens each).
Prefill quantizes once; decode dequantizes per step, so attention compute is
unchanged. Enable it through the model-specific `extras` config (e.g. in the
deploy YAML's stage `extras` or as an `Omni(...)` engine kwarg):

```yaml
extras:
  prefix_kv_cache_dtype: "fp8"
```

Accepted values: `"fp8"` / `"fp8_e4m3"` (K and V quantized), `"fp8_v"` (V
only — K stays in the native dtype), and `None` / `"auto"` (default, native
dtype — behavior unchanged). This is independent of
`diffusion_kv_cache_dtype`, which quantizes attention Q/K/V *compute* per
forward pass on supported backends.

Measured on T2I 1024x1024 (seed 42, 50 steps, true CFG 4.0, PSNR vs. the
bf16 baseline): **`"fp8"` 34.9 dB**, **`"fp8_v"` 40.9 dB** — same composition
and semantics, with texture-level drift in fine detail for `"fp8"`, and
visually indistinguishable output for `"fp8_v"`. The error is dominated by
K quantization: post-RoPE keys are the precision-sensitive half of the
cache, so `"fp8_v"` buys back ~6 dB at 75% (instead of 50%) of the original
cache size. The residual error is inherent e4m3 precision accumulated
coherently over the denoising trajectory; finer scale granularity
(per-tensor/per-head/per-token were compared) or Hadamard-rotated V did not
improve it. The saving scales with prefix length — at the limit
(8192 text tokens + 4 condition images, ~24.6k prefix tokens) the prefix
cache is ~12.9 GB per CFG branch in bf16, ~6.6 GB in `"fp8"` and ~9.7 GB in
`"fp8_v"`, which is where this option matters. Treat it as an opt-in for
memory-bound long-prompt / multi-image workloads, not a free lunch;
`"fp8_v"` is the better default trade-off when quality matters.

Note: a quantized prefix cache is not CUDA-graph capturable — requests using
this option stay on the eager decode path (a warning is logged once).

### Why not the scheduler-managed paged KV (`DiffusionKVCacheMode.PAGED_SCHEDULER`)?

Evaluated and deliberately not adopted (see the `qwen21-p0-kv` commit message
for the full analysis): the paged scheduler targets *cross-request* KV
management — a statically sized page pool, block sharing/dedup, and eviction —
whereas this prefix cache lives exactly as long as one generation and is
sized once per request. Migrating would require routing decode attention
through the worker paged adapter (writing fresh target K/V into pages every
step), per-request page tables, and re-validating the block-causal
piecewise-span masking and Ulysses/CFG constraints, for no memory benefit over
the per-request dense tensors. The FP8 storage above captures the actual
memory win at ~1% of the integration cost.

## Quality Comparison (head vs diffusers)

The reference is the diffusers Qwen-Image 2.1 pipeline
([huggingface/diffusers#14804](https://github.com/huggingface/diffusers/pull/14804)).
Both sides must pin the same checkpoint, inputs, seed, precision (BF16), and
hardware.

vLLM-Omni (this recipe):

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Qwen/Qwen-Image-2.1 \
  --prompt "A ceramic teapot on a wooden table" \
  --seed 42 \
  --output qwen_image_21_vllm_omni.png \
  --num-inference-steps 50 \
  --cfg-scale 1.0
```

diffusers reference:

```python
import torch
from diffusers import QwenImage21Pipeline  # diffusers PR #14804

pipe = QwenImage21Pipeline.from_pretrained(
    "Qwen/Qwen-Image-2.1", torch_dtype=torch.bfloat16
).to("cuda")
image = pipe(
    prompt="A ceramic teapot on a wooden table",
    num_inference_steps=50,
    true_cfg_scale=1.0,
    generator=torch.Generator(device="cuda").manual_seed(42),
    height=1024,
    width=1024,
).images[0]
image.save("qwen_image_21_diffusers.png")
```

| Metric | vLLM-Omni | diffusers (#14804) | Status |
| --- | --- | --- | --- |
| Output image (1024×1024, 50 steps, seed 42, BF16) | pending measurement | pending measurement | not yet run |
| PSNR / LPIPS vs reference | pending measurement | — | not yet run |
| Latency (time/image, same GPU) | pending measurement | pending measurement | not yet run |

The prefix KV cache changes the compute path (cached prefix vs. full recompute
at every step), so exact numerical parity is not expected; match the cache
setting on both sides or compare with PSNR/LPIPS rather than exact pixels.

The following capabilities are implemented but the PR-body checkboxes predate
any attached measurements; treat them as unverified until the A/B above is
filled in:

- [ ] TP=2/4 output parity vs single-card — pending measurement.
- [ ] VAE tiling vs non-tiled decode — pending measurement.
- [x] CUDA graph decode vs eager — measured on GB200 (1024×1024, 50 steps,
  seed 42, BF16): compile+graph 3.04 s/image vs eager 4.3 s/image (~1.4×),
  44.5–45.6 dB PSNR vs eager; graph-only is bit-identical to eager. See
  "CUDA Graph decode" above.

## Known Limitations

- Cache acceleration backends (`cache_dit`, `tea_cache`) are not supported;
  `QwenImage21Pipeline` is registered in `_NO_CACHE_ACCELERATION`.
- Sequence parallelism supports Ulysses only; ring attention is not supported.
- Only true CFG is exposed; there is no `guidance_scale` knob.
- VAE tiling (`--vae-use-tiling`, also implied by `--vae-patch-parallel-size > 1`)
  decodes in 512px tiles with 384px stride (32 latent pixels per tile at 16x
  spatial compression). On CUDA OOM the tile size is automatically halved
  (keeping 25% overlap) and decoding retried, down to 128px tiles. Tile sizes
  are not yet exposed as engine arguments.
