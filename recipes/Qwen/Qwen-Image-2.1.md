# Qwen-Image-2.1

> Text-to-image and image-conditioned generation with Qwen-Image 2.1

## Summary

- Vendor: Qwen
- Model: `Qwen/Qwen-Image-2.1`, `Qwen/Qwen-Image-2.1-Turbo` (8-step distilled)
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

## Qwen-Image-2.1-Turbo (8-step distilled)

`Qwen/Qwen-Image-2.1-Turbo` is the 8-step distilled variant of
Qwen-Image-2.1 and runs through the same `QwenImage21Pipeline`. Keep these
differences from the base checkpoint in mind:

- **Do not pass `num_inference_steps`.** The Turbo checkpoint's
  `model_index.json` ships a fixed `sample_sigmas` grid, which vLLM-Omni
  loads automatically. An explicit `num_inference_steps` is overridden by
  `sample_sigmas`, so omit it.
- **Do not enable CFG.** As a distilled model it runs at CFG=1: leave
  `true_cfg_scale` at its default (1.0); setting `true_cfg_scale > 1` (and a
  `negative_prompt`) is not meaningful for the distilled weights.
- **License:** Qwen Research License (not Apache-2.0) — check the checkpoint
  card before commercial use.

Offline:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Qwen/Qwen-Image-2.1-Turbo \
  --prompt "A ceramic teapot on a wooden table" \
  --output qwen_image_21_turbo_t2i.png \
  --cfg-scale 1.0
```

Online serving works the same as the base model
(`vllm serve Qwen/Qwen-Image-2.1-Turbo --omni --port 8091`). The sampling
grid can be overridden per request through the whitelisted `sigmas`
extra-body parameter (both `/v1/chat/completions` and the multi-stage
`/v1/images/generations` path). The default 8-step grid is
`[1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]`:

```bash
curl http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen-Image-2.1-Turbo",
    "prompt": "A ceramic teapot on a wooden table",
    "size": "1024x1024",
    "seed": 42,
    "sigmas": [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]
  }'
```

### Four-GPU BF16 eager inference

Start a four-GPU server with the checkpoint's native eight-step schedule:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen-Image-2.1-Turbo \
  --omni --enforce-eager --dtype bfloat16 --num-gpus 4 \
  --tensor-parallel-size 1 --usp 4 --ring 1 --cfg-parallel-size 1 \
  --vae-patch-parallel-size 1 --max-num-seqs 1 \
  --diffusion-attention-backend FLASH_ATTN --port 8098
```

The default communication backend remains the reference all-to-all. On a
qualified CUDA installation, opt into symmetric-memory exchange with
`--ulysses-a2a-permute`. This requires a working CUDA compiler/toolkit and
NCCL 2.28 or later with symmetric-memory device support, including the wheel
library/header layout required by the current extension builder. Missing
requirements fail explicitly at initialization when opted in. No rank-local
fallback is attempted after collectives have started.

Eligible equal BF16 Q/K/V tensors then share one exchange; other layouts
retain the separate exchanges. Backend selection uses the common parallel
configuration; it does not depend on a model name or a hardware-name heuristic.

Shared kernels live under `vllm_omni/diffusion/layers/`: pointwise fusion
preserves BF16 intermediate roundings. Both standalone RoPE and full head128
Q/K norm/RoPE require SM103 with CUDA/PyTorch 2.13, where the reduction and
complex FMA order were validated. Other architectures, including SM90, retain
the reference normalization and complex RoPE arithmetic. The channel-first VAE norm
helper and epilogue are shared with Wan; Wan's channels-last policy stays in
its own adapter.

Exact pure-T2I text conditioning is bounded by 16 entries and 16 MiB.
Image conditioning, batched prompts, training/gradients, LoRA and CPU offload
bypass and invalidate this cache. Geometry follows the request's KV lifetime,
with weak ownership and denoise-finally cleanup that preserves decode graphs.
Generated images and sampled latents are never reused by these optimizations.

After the server reports ready, send a full eight-step image request:

```bash
curl --fail --silent --show-error http://127.0.0.1:8098/health
curl --fail --silent --show-error \
  http://127.0.0.1:8098/v1/images/generations \
  -H 'Content-Type: application/json' \
  --data-binary @- > qwen-image-response.json <<'JSON'
{"model":"Qwen/Qwen-Image-2.1-Turbo","prompt":"A ceramic teapot on a wooden table","seed":42,"n":1,"size":"1280x704","response_format":"b64_json","output_format":"png","output_compression":100,"true_cfg_scale":1.0,"sigmas":[1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568]}
JSON
```

The response contains a base64-encoded PNG in `data[0].b64_json`. This request
checks generation; it does not measure latency. For performance comparisons,
freeze the benchmark and accuracy checks independently of the optimization,
use identical inputs and declared configurations, and compare native RGB and
alpha outputs. Report cold/unseen prompts separately from warmed repeated
prompts, which can reuse exact text conditioning.

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
single H200 (1024×1024, 50 steps, seed 42, BF16, `true_cfg_scale=1.0`,
warmup 1 + median of 3): compile+graph **5.93 s**, compile-only **6.88 s**,
graph-only **9.52 s**, eager **8.28 s** end-to-end per image; combo and
compile-only vs eager are **38.10 dB** PSNR (compile fusion drift), and
graph-only vs eager is bit-identical.

The autoregressive engine's `compilation_config.cudagraph_mode` does not
control this diffusion path; use `enable_cuda_graph_decode` (or
`enforce_eager` for everything) to force eager decode execution.

TP/SP/ring parallelism, HSDP, cache hooks, quantized KV caches,
dynamic LoRA, padded text masks, and a second in-flight request whose cache
aliases an already-owned graph key fall back to eager decode.
Model-level CPU offload does not disable decode graphs while
`enable_cuda_graph_decode` stays on (the default). DiT weights stay on fixed
device storage so captured pointers remain valid, and that storage is not
released while the text encoder runs. Pass `--no-enable-cuda-graph-decode`
(or `enable_cuda_graph_decode: false`) for the eager offload path, which
moves DiT weights off the device and frees that VRAM. Peak memory for the
default graph-plus-offload combination was not re-measured in this PR.
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
steps. End-to-end on H200, single GPU, same prompt as Quality Comparison
(`"A ceramic teapot on a wooden table"`, 1024×1024, 50 steps, seed 42,
eager):

| Config | Quantized block linears | PSNR vs BF16 | Steady-state time/image | Peak memory |
| --- | --- | --- | --- | --- |
| BF16 | 0 / 160 | — | **7.34 s** | **36.9 GB** |
| FP8 all layers | 160 / 160 | **33.0 dB** | **6.06 s** | **30.4 GB** |
| FP8, `ignored_layers=["img_mlp"]` | 64 / 160 | **34.3 dB** | **6.98 s** | **34.9 GB** |

Keeping `img_mlp` in BF16 is still the higher-fidelity DiT setting vs
all-layer FP8. On H200, all-layer FP8 is both leaner and faster than BF16
at this shape.

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

Either component can be quantized on its own. Measured on H200 at 1024×1024
(seed 42, 50 steps, same teapot prompt, eager): text-encoder FP8 lowers peak
GPU memory from **36.9 GB** (BF16) to **30.4 GB** (−6.5 GB); T2I PSNR vs BF16
**32.1 dB**. DiT FP8-all is **30.4 GB** / **6.06 s**. Combined DiT+text-encoder
FP8 was not re-measured on H200. The edit path routes condition images through
the BF16 vision tower, so it is unaffected by text-encoder FP8. See
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

Measured on one NVIDIA H200, T2I 1024x1024 (seed 42, 50 steps, true CFG 4.0,
eager, PSNR vs. the bf16 baseline on the same GPU): **`"fp8"` 34.2 dB**,
**`"fp8_v"` 44.0 dB**. The error is dominated by
K quantization: post-RoPE keys are the precision-sensitive half of the
cache, so `"fp8_v"` buys back about 9.8 dB at 75% (instead of 50%) of the original
cache size. The residual error is inherent e4m3 precision accumulated
coherently over the denoising trajectory; finer scale granularity
(per-tensor/per-head/per-token were compared) or Hadamard-rotated V did not
improve it. The saving scales with prefix length. These sizes are calculated at the
limit (8192 text tokens + 4 condition images, ~24.6k prefix tokens), not
measured on the 1024 T2I run above: ~12.9 GB per CFG branch in bf16, ~6.6 GB
in `"fp8"` and ~9.7 GB in `"fp8_v"`. That is where this option matters. Treat it as an opt-in for
memory-bound long-prompt / multi-image workloads, not a free lunch;
`"fp8_v"` is the better default trade-off when quality matters.

Note: a quantized prefix cache is not CUDA-graph capturable — requests using
this option stay on the eager decode path (a warning is logged once).

### Scheduler-managed paged KV and cross-request prefix caching (`DiffusionKVCacheMode.PAGED_SCHEDULER`)

In addition to the default per-request dense prefix cache (`dense_legacy`),
`QwenImage21Pipeline` supports Scheduler-managed paged KV (`diffusion_kv_mode="paged_scheduler"`)
with cross-request prefix caching (`enable_prefix_caching=True`). Because reference
images precede the edit instruction in the block-causal sequence and modulate from
`t = 0`, repeated requests that share leading reference images reuse their cached
KV pages across requests, slice the Step-0 prefill query to the uncached suffix,
and skip redundant condition-image `vae.encode` (and `encode_prompt` on a full-prefix hit):

```python
from vllm_omni import Omni

omni = Omni(
    model="Qwen/Qwen-Image-2.1",
    diffusion_kv_mode="paged_scheduler",
    diffusion_kv_max_rows_per_request=2,  # 1 when true_cfg_scale <= 1.0, 2 for standard CFG
    enable_prefix_caching=True,
    max_num_batched_tokens=32768,
)
```

With CFG, both rows currently use the minimum safe hit length across branches.
A full common-prefix hit can therefore still encode the longer branch's text;
prompt encoding is skipped separately only when that branch is fully covered.

The startup memory profile defines the safe paged-KV admission envelope. By
default it profiles 4 reference images, both CFG branches, an 8192-token VLM
prompt budget, and a 1024x1024 target (4096 DiT target tokens). Requests may
use other aspect ratios when their aligned target-token count does not exceed
4096. To serve a larger target envelope, set
`additional_config["diffusion_kv_profile_height"]`,
`additional_config["diffusion_kv_profile_width"]`, and, when needed,
`additional_config["diffusion_kv_profile_max_sequence_length"]`; increase
`max_model_len` and the KV memory budget consistently. Requests outside the
profiled sequence or target-token envelope are rejected before execution
rather than risking an activation-memory OOM.

For 4-reference-image requests with `true_cfg_scale > 1.0` (~20.5k tokens per CFG branch at 1024×1024), single-rank CFG executes the positive and negative branches sequentially so `max_num_batched_tokens=32768` covers each branch's prefill pass; ensure the KV cache pool (`kv_cache_memory_bytes` or `gpu_memory_utilization`) can hold both branches (~21 GiB in BF16 on 1 GPU, or halved per rank under `cfg_parallel_size=2` / `tensor_parallel_size=2` / `ulysses_degree=2`).

#### Choosing dense or paged execution

`dense_legacy` remains the default. Its DiT KV belongs to one generation,
and eligible single-GPU configurations retain CUDA-graph decode. Choose
`paged_scheduler` when repeated condition prefixes justify a retained GPU
pool and cross-request lookup. A hit saves conditioning and Step-0 work once
per request; it does not eliminate the remaining denoise steps.

Paged execution disables decode CUDA graphs and rejects FP8 prefix KV
(`extras.prefix_kv_cache_dtype`). It also requests FP32 GEMM reductions throughout
the DiT forward; HSDP with Ulysses additionally requires native CUDA
FlashAttention batch-invariant attention. These per-step costs can outweigh
the one-time savings, especially for long denoise schedules or cold requests.
Compare with the default dense path at the intended step count and memory
budget before selecting paged mode.

Bitwise parity is an observation for a particular GPU, workload, and backend,
not a guarantee for every request. The VLM prompt encoder runs outside the
DiT precision scopes, and changing text or suffix lengths can select different
floating-point kernels. Report bitwise equality where observed and PSNR
otherwise, including a separate paged-versus-dense comparison. The GPU
regression compares cache-off with changed-text and changed-later-image hits,
checks actual Worker tail-page copies, and independently requires RGB and
alpha (when present) PSNR of at least 40 dB when outputs are not bitwise equal:

```bash
QWEN_IMAGE21_TEST_MODEL=Qwen/Qwen-Image-2.1 python -m pytest \
  tests/diffusion/models/qwen_image_21/test_prefix_cache_gpu.py -q -s
```

The test uses eight CUDA GPUs, HSDP8/Ulysses8, BF16 eager execution, and
two denoise steps; it is a correctness test, not a serving benchmark.

The 40 dB threshold applies to the selected in-tree regression inputs.
Numerical agreement is configuration-specific and is reported separately
for each tested workload.

The cache deliberately isolates image-count/size layouts and CFG branches.
Replacing a later reference image reuses earlier images only when the layout
is unchanged. Increasing the reference count or changing a later image's
aspect ratio currently starts a different namespace. These are conservative
numerical guards, not a claim that the block-causal graph makes an earlier
image depend on a later one. CFG rows also retain separate image KV, which
must be included in capacity estimates.

For request isolation, supply a non-empty string `cache_salt` in the offline prompt,
the images generation JSON body, the images edit multipart form, or the
diffusion chat request. Different salts cannot share Qwen prefix hits.
Applications should assign salts from their own authenticated isolation
policy; a client-supplied string is not authentication. Omitting the salt
preserves shared reuse within that engine. Pools are local to one loaded
model instance; cross-process/model-revision KV sharing is outside this path.

Exact image boundaries use token-granular hashes. Hash construction,
Scheduler lookup, and request-side hash metadata grow with prefix length;
include that work in end-to-end latency rather than counting only GPU time.

#### Reproducing the dense/paged comparison

Run each mode in a fresh process on the same GPU allocation. This example
uses two references, a fixed 14 GiB paged pool, two warmup requests, and the
median of five measured requests per cold/partial/full-hit case:

```bash
for mode in dense paged_off paged_on; do
  CUDA_VISIBLE_DEVICES=0 python benchmarks/diffusion/benchmark_qwen_image_21_prefix_cache.py \
    --model Qwen/Qwen-Image-2.1 --layout single --mode "$mode" \
    --steps 50 --cfg 4 --size 1024 --references 2 --kv-cache-gib 14 \
    --warmup 2 --repeats 5 --capacity-probes 32 --output "prefix-cache-results/$mode"
done
```

Use a fresh output directory for each run. Repeat with
`Qwen/Qwen-Image-2.1-Turbo --steps 8 --cfg 1`, and with eight visible GPUs
and `--layout hsdp8`. The script limits paged admission to 24,576 tokens so
the declared maximum sequence fits the single-GPU pool. Untimed preparation
and capacity requests ask for two steps to materialize the same static prefix;
Turbo's fixed sigma grid overrides that request and still runs eight steps.
Each output directory records sample latencies,
actual cache hits/COW copies, decode-graph replay observations, peak GPU
memory, output arrays, and retained-prefix capacity after lookup-only probes.
Cold trials use distinct salts so the shared text preamble cannot hit;
partial/full trials share the unsalted namespace.
These are sequential offline request latencies, not concurrent serving
throughput. Keep each speedup comparison on one host and GPU configuration.

#### Validation on vLLM 0.33

After merging upstream `a2d3b4b2`, the prefix-cache path was revalidated at
`8c463de7` with vLLM 0.33.0 (frozen wheel commit `4ba787ec9`),
PyTorch 2.14.0+cu130, CUDA 13.0, cuDNN 9.25.1, Transformers 5.19.0 and
diffusers 0.40.0. All comparisons below use the same host and allocation.

Affected diffusion/KV/scheduler/runner CPU suites passed **910 tests**;
image/chat API and profiler endpoints passed **122 tests**, with one existing
expected failure. The eight-GPU partial-prefix/COW regression passed in
317.39 seconds on eight RTX PRO 5000 72GB Blackwell cards,
BF16, HSDP8/Ulysses8, 256² and two steps. All four cases matched cache-off
bitwise in RGB and alpha, including the changed-text and changed-later-image
hits with two tail-page copies each. A separate 480×608 control also matched
bitwise for `mask_sp_padding=False` and `True`, cold and warm: 1,140 logical
target tokens and four synthetic SP padding tokens.

The representative performance follow-up uses one RTX PRO 5000 72GB card,
Base at 1024², 50 steps, CFG scale 4, two references and seed 42.
It uses the benchmark above: fresh processes per mode, two warmups and
five measured trials per cold/partial/full case, with 14 GiB KV in paged modes.
References are deterministic synthetic 128×128 RGB inputs processed normally.

| Case | Dense s | Paged off s | Paged on s | Dense / on | Off / on |
| --- | ---: | ---: | ---: | ---: | ---: |
| cold | 48.983 | 50.248 | 50.488 | 0.970× | 0.995× |
| partial | 49.006 | 50.313 | 49.554 | 0.989× | 1.015× |
| full | 49.048 | 50.328 | 48.157 | 1.019× | 1.045× |

All **15/15 measured cache-on/off full-image SHA256 pairs match**, including
alpha. Each case's first saved on/off output array also matches bitwise.
The following on/dense comparisons use those first saved matching inputs:

| Case | On/off RGBA | On/dense RGB dB | On/dense RGB max | On/dense alpha max |
| --- | --- | ---: | ---: | ---: |
| cold | bitwise | 34.37 | 131 | 8 |
| partial | bitwise | 36.71 | 159 | 5 |
| full | bitwise | 34.91 | 161 | 4 |

Actual decode-graph replay was observed in dense mode; paged modes did not
replay decode graphs. Whole-run NVML sampled peaks were
**57.76/52.16/52.15 GiB** for dense/off/on.
These samples include startup, warmup, untimed preparation and retention
probes, rather than per-request incremental memory. The paged-on pool has
**1,792 physical blocks / 14 GiB**, retaining **1/32 full common CFG prefixes**
after lookup-only probes. Capacity depends on image and text lengths.

Full common-prefix hits give **1.045× versus paged-off** and **1.019× versus
dense** in this Base single-GPU workload. Cold and changed-last-image partial
requests remain slower than dense. This follow-up does not relabel the earlier
Base/Turbo single-/multi-GPU matrix as vLLM 0.33 results.

#### Earlier vLLM 0.32 measurements

NVIDIA RTX PRO 5000 Blackwell, BF16, PyTorch 2.13.0+cu130, vLLM 0.32.0.
Single-GPU cohorts use one 72GB card; HSDP8/Ulysses8 cohorts use eight
48GB cards on another host. Each dense/off/on comparison stays on one
host and allocation. These numbers do not measure cross-host or parallel
scaling. All requests use 1024×1024, two references, seed 42, two warmups
and the median of five trials per case. Base uses CFG 4; Turbo uses CFG 1.
Paged modes reserve 14 GiB per rank; all modes use memory utilization 0.9.

References are deterministic synthetic 128×128 RGB images processed by
the model's normal image preprocessing. Each trial changes reference seeds;
the corresponding trials match across all three modes.

| Model (steps) | Allocation | Case | Dense s | Paged off s | Paged on s | Dense / on | Off / on |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Base (50) | 1×72GB | cold | 49.069 | 50.321 | 50.367 | 0.974× | 0.999× |
| Base (50) | 1×72GB | partial | 49.165 | 50.306 | 49.512 | 0.993× | 1.016× |
| Base (50) | 1×72GB | full | 49.225 | 50.320 | 48.199 | 1.021× | 1.044× |
| Base (50) | HSDP8/UP8, 8×48GB | cold | 75.332 | 49.977 | 50.139 | 1.502× | 0.997× |
| Base (50) | HSDP8/UP8, 8×48GB | partial | 75.270 | 50.057 | 49.367 | 1.525× | 1.014× |
| Base (50) | HSDP8/UP8, 8×48GB | full | 75.189 | 50.066 | 48.202 | 1.560× | 1.039× |
| Turbo (8) | 1×72GB | cold | 5.991 | 5.309 | 5.374 | 1.115× | 0.988× |
| Turbo (8) | 1×72GB | partial | 6.172 | 5.391 | 5.020 | 1.229× | 1.074× |
| Turbo (8) | 1×72GB | full | 6.139 | 5.413 | 4.118 | 1.491× | 1.315× |
| Turbo (8) | HSDP8/UP8, 8×48GB | cold | 5.344 | 5.468 | 5.493 | 0.973× | 0.995× |
| Turbo (8) | HSDP8/UP8, 8×48GB | partial | 5.332 | 5.487 | 5.110 | 1.043× | 1.074× |
| Turbo (8) | HSDP8/UP8, 8×48GB | full | 5.314 | 5.508 | 4.093 | 1.298× | 1.346× |

Ratios above one favor paged cache-on. Partial hits replace the last
reference with a new image of the same shape. Full means a full common
CFG prefix; the longer branch may still encode its remaining text.
Dense/on includes execution-backend and compilation differences;
off/on measures enabling cross-request reuse within paged execution.

The following comparisons use the first measured matching request in
each case. PSNR uses native RGB; alpha reports its maximum absolute
8-bit difference. ∞ means identical RGB values.

| Model | Allocation | Case | On/off RGB dB | On/dense RGB dB | On/off RGB max | On/dense RGB max | Alpha max on/off; on/dense |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Base (50) | 1×72GB | cold | ∞ | 33.59 | 0 | 150 | 0; 6 |
| Base (50) | 1×72GB | partial | ∞ | 36.73 | 0 | 151 | 0; 4 |
| Base (50) | 1×72GB | full | ∞ | 34.58 | 0 | 128 | 0; 10 |
| Base (50) | HSDP8/UP8, 8×48GB | cold | ∞ | 39.39 | 0 | 110 | 0; 2 |
| Base (50) | HSDP8/UP8, 8×48GB | partial | ∞ | 32.08 | 0 | 180 | 0; 5 |
| Base (50) | HSDP8/UP8, 8×48GB | full | ∞ | 30.80 | 0 | 198 | 0; 8 |
| Turbo (8) | 1×72GB | cold | ∞ | 41.30 | 0 | 43 | 0; 4 |
| Turbo (8) | 1×72GB | partial | ∞ | 37.80 | 0 | 53 | 0; 87 |
| Turbo (8) | 1×72GB | full | ∞ | 39.43 | 0 | 60 | 0; 28 |
| Turbo (8) | HSDP8/UP8, 8×48GB | cold | ∞ | 43.80 | 0 | 40 | 0; 4 |
| Turbo (8) | HSDP8/UP8, 8×48GB | partial | ∞ | 38.87 | 0 | 54 | 0; 26 |
| Turbo (8) | HSDP8/UP8, 8×48GB | full | ∞ | 39.04 | 0 | 61 | 0; 31 |

All 60/60 measured cache-on/off output SHA256 pairs also match, including alpha.

| Model | Allocation | Pool blocks | Allocated GiB/rank | Retained common prefixes / 32 | Dense graph replay | Sampled peak GiB (dense/off/on), max rank |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Base (50) | 1×72GB | 1792 | 14.000 | 1/32 | yes | 57.59/51.98/52.14 |
| Base (50) | HSDP8/UP8, 8×48GB | 14336 | 14.000 | 13/32 | no | 35.95/41.51/41.41 |
| Turbo (8) | 1×72GB | 1792 | 14.000 | 2/32 | yes | 47.83/52.10/52.10 |
| Turbo (8) | HSDP8/UP8, 8×48GB | 14336 | 14.000 | 27/32 | no | 31.05/41.41/41.41 |

Graph replay is observed in the actual decode path, not inferred from a
configuration flag. Paged decode graphs are disabled. Memory is sampled
through NVML every 0.2 seconds, including engine startup, warmup and
capacity probes; it is whole-run device usage, not per-request incremental
memory or an exact PyTorch allocation peak. Retention uses lookup-only probes after 32 distinct
prefixes; it does not reinsert evicted entries. These capacity values
apply to the tested two-reference workloads, not every image/text length.

Raw sample commit identities are preserved. Earlier baselines remain
valid: subsequent changes lower paged admission to fit the fixed pool,
move native paged metadata construction outside Dynamo, and isolate cold
trials from the shared text header with unique salts. These changes do
not alter dense execution. Sample commits: `5ff0cc9e`, `96045624`, `9c9a59f3`.

Later review hardening rejects invalid salts and retries shorter prefix
hits when COW would exceed the physical pool. The latter retry is not
reached by this matrix: even two 12,318-token rows plus two COW sources
need at most 1,542 physical 16-token blocks, below the smallest pool's 1,791 usable
blocks. The recorded salt values are valid. These measurements retain
their original source labels; the added edge cases have separate regression
coverage.

For the 50-step base model, warm reuse alone yields 1.014–1.044× versus
paged cache-off. The 1.560× full-hit HSDP8 comparison against dense includes
execution-path differences already visible with cache-off (1.507× on cold
requests). For Turbo, full hits yield 1.315× and 1.346× versus paged cache-off,
and 1.491× and 1.298× versus dense, on the single- and eight-GPU allocations
respectively. Cold or partial requests can still lose to dense. Memory use
also varies: a retained 14 GiB pool is not always smaller than dense storage.
Choose from the measured workload and memory trade-off, rather than treating
cross-request caching as an unconditional speedup.

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
| Output image (1024×1024, 50 steps, seed 42, BF16, H200) | generated (eager + default CUDA-graph) | generated | measured |
| PSNR / LPIPS vs reference | eager vs Diffusers **47.17 dB**; **default compile+graph** vs Diffusers **38.02 dB**; **default compile+graph** vs eager **38.10 dB** | — | measured (H200) |
| Latency (time/image, same GPU, H200) | **default compile+graph** **6.04 s** (120.8 ms/step); eager **7.31 s** (146.2 ms/step) | **7.74 s** (154.9 ms/step) | measured |

The prefix KV cache changes the compute path (cached prefix vs. full recompute
at every step), so exact numerical parity is not expected; match the cache
setting on both sides or compare with PSNR/LPIPS rather than exact pixels.
Eager stays numerically close to Diffusers (~47 dB). The **default** serving
path stacks regional `torch.compile` with CUDA-graph decode; on H200 that
combo lands at **~38.1 dB** vs eager (~1.21× on the Quality Comparison timed
run). Graph-only vs eager is bit-identical; the ~38 dB drift is the compile
fusion, not CUDA-graph replay. See "CUDA Graph decode" for the four-mode
H200 breakdown (warmup 1 + median of 3).

Parity / feature measurements (same prompt/seed/steps/CFG, BF16, eager decode
unless noted; 4× NVIDIA H200):

- [x] TP=2/4 output parity vs single-card — measured (1024×1024): TP1 vs TP2
  **39.32 dB**, TP1 vs TP4 **38.37 dB**, TP2 vs TP4 **45.77 dB**. Latency:
  TP1 **7.30 s**, TP2 **4.89 s**, TP4 **5.12 s** per image.
- [x] VAE tiling vs non-tiled decode — measured: 1024×1024 nontile vs
  `--vae-use-tiling` **43.55 dB** (peak mem ~37.7 GB → ~32.7 GB); 2048×2048
  nontile vs tiled **44.14 dB** (peak mem ~57.9 GB → ~33.3 GB; gen ~36.8 s vs
  ~37.1 s).
- [x] CUDA-graph four-mode vs eager — measured on H200 (1024×1024, 50 steps,
  seed 42, BF16, warmup 1 + median of 3): compile+graph **5.93 s**,
  compile-only **6.88 s**, graph-only **9.52 s**, eager **8.28 s**. Combo /
  compile-only vs eager **38.10 dB**; graph-only vs eager bit-identical.
  Quality Comparison table uses a separate single-run A/B vs Diffusers
  (compile+graph **6.04 s**, eager **7.31 s**).

## Known Limitations

- Cache acceleration backends (`cache_dit`, `tea_cache`) are not supported;
  `QwenImage21Pipeline` is registered in `_NO_CACHE_ACCELERATION`.
- Ulysses sequence parallelism is wired; ring attention is not a supported path.
  A short H200 smoke (1024×1024, 8 steps, seed 42, true CFG 1.0, eager) finished:
  Ulysses SP=2 vs SP=1 is **46.5 dB**. That size's target length is divisible by 4,
  so SP padding was not exercised, and this is not the 50-step parity next to the
  TP tables. The features matrix therefore stays ❓. Ring SP=2 also ran, at
  **29.9 dB** vs SP=1, which is why it stays unsupported.
- Only true CFG is exposed; there is no `guidance_scale` knob.
- VAE tiling (`--vae-use-tiling`, also implied by `--vae-patch-parallel-size > 1`)
  decodes in 512px tiles with 384px stride (32 latent pixels per tile at 16x
  spatial compression). On CUDA OOM the tile size is automatically halved
  (keeping 25% overlap) and decoding retried, down to 128px tiles. Tile sizes
  are not yet exposed as engine arguments.

## NPU performance defaults

Qwen-Image-2.1 enables the following optimizations automatically on eligible
NPU paths. The measured workload is Qwen-Image-2.1-Turbo, BF16, TP1, eager
execution, batch 1, CFG1 and 1280×704 on Ascend 910B1. These defaults do not
change the sampling schedule or select a different attention backend.

These optimizations are selected automatically, without environment switches:

- Interleaved NPU RoPE.
- Broadcast modulation during BF16 batch-1 eager decode without SP.
- Residual buffer reuse during eligible BF16 eager inference.
- NZ storage for eligible BF16 NPU linear weights.
- Model-local fusion attention for eligible calls with SDPA selected.
- Triton Ascend SwiGLU fusion on eligible Ascend910B1 BF16 inputs.

`QWEN21_FUSION_PRECISE` still controls fusion attention precision (default `0`).

The automatic attention path requires TP1, no SP, no CPU offload, no
quantization, batch 1, equal Q/K/V head counts, no metadata, no gradients and
no compilation. It executes inside shared Attention dispatch. Other models,
other selected backends, masked calls and graph execution keep their existing
paths. Causal attention and unequal Q/K/V head counts also retain ordinary SDPA.

Automatic NZ conversion requires unquantized TP1 execution without CPU offload,
sequence parallelism, HSDP or a configured `lora_path`, BF16 NPU weights, and
`allow_internal_format=True`
already configured in the runtime. Otherwise it is skipped. Eligibility follows
eager execution, not `enable_cuda_graph_decode`. On NPU this transformer stays
eager either way: Ascend does not torch.compile it, and CUDA graph capture does
not run, so the default graph manager falls back to eager and still converts.
The model never changes the global internal-format setting. Conversion is
skipped when its prerequisites are absent.
Conversion runs while the transformer loads its weights. Distilled LoRA fusion
adds ND deltas into those weights afterwards, and HSDP shards or broadcasts
them, so both configurations keep ND storage.
Residual reuse requires the eager prepared-modulation path and does not mutate
its input or combine multiplication/addition rounding into one fused operation.

VAE spatial-padding folding, prompt caching, request geometry reuse and shared
modulation preparation apply under their own guards.

VAE normalization also reuses its fresh FP32 conversion buffer on NPU for
FP16/BF16 inference without autograd or compilation. It preserves the L2 norm,
epsilon clamp and sequential scaling operations, and never overwrites the
input. A separate 1024×1024,
8-step Ascend 910B1 comparison validated pixel-identical outputs; its results
should not be combined with the 1280×704 measurements above.

Unquantized NPU BF16 MLP inference also reuses fresh gate-projection storage
for separate SiLU and multiplication operations. Autograd, compilation and
other dtypes retain the original path. VAE normalization skips its fixed Python
zero bias, while trainable tensor biases are still applied. These changes need
no additional environment flags and were validated with the 1024×1024 workload.

### Triton Ascend SwiGLU fusion

This path is enabled automatically when its eligibility checks pass.
Earlier default-configuration measurements excluded this fusion and must not
be treated as measurements of the new default. With Triton Ascend
installed, unquantized eager BF16 inference on Ascend910B1 fuses the
two-input SiLU/multiply path for batch 1, hidden width
4096 and sequence length at least 4096. Both projected tensors must have width
12288 and contiguous ND storage. Other cases use the existing implementation.
The kernel reuses the fresh gate projection, preserves BF16 rounding after
SiLU, and does not concatenate gate/up tensors. Sequence lengths share a single
compiled kernel; initial compilation is excluded from warmed latency results.

Focused validation (requires the project's NPU test environment):

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python -m pytest \
  tests/diffusion/models/qwen_image_21/test_npu_fused_ops.py \
  --run-level core_model -q
```

### Conditioning cache and graph fallback scope

The model-local text-only prompt cache matches upstream main. A single string
prompt is cached directly and does not consult `enable_prompt_embed_cache`.
Editing, multi-prompt calls, training, autograd, LoRA, and offload bypass the
cache and drop stored text entries. The shared runner cache keeps its own
default and collective-safety checks.

Single-request step decode retains its request KV owner. Geometry is stored
per CFG branch on that owner; rebuilding an equivalent mask no longer invalidates
it. Prefix layout must remain fixed for the lifetime of that branch's KV cache.

Shared modulation applies to eager prefill and eager decode fallback even when
a decode graph manager exists. Captured graph execution is unchanged; no new
graph-mode performance improvement is claimed.
