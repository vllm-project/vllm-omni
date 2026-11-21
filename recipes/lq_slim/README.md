# LQ-Slim

The recommended LQ-Slim configuration combines eight-step MiniMax-H3 Ref2VA
LoRA, NPU-resident reference KV, mixed-precision attention and independent
reference-image sparsity.
The target sparsity is 0.8 for steps 0–1 and 0.85 for steps 2–7; reference-image
candidate blocks use an image deletion ratio of 0.9. These are approximate optimizations;
output quality must be evaluated for the intended inputs.

The complete eight-step recipe pairs the Larry v4-step600 EMA adapter with
fixed sampling settings so deployments can reproduce the same configuration
when evaluating reference fidelity and audio/video quality. Adapter integration
and the Ref extensions are included together, but each Ref option remains
independent of the adapter. Selecting this LoRA alone does not establish
output quality for a particular sparsity or precision setting.

For deployments that only need the two Ref extensions, see the independent
[reference optimization guide](../h3_ref_optimizations/README.md). They do not
require the fixed adapter or the eight-step preset below.

## Install

Use the project's normal [Ascend installation](../../docs/getting_started/installation/npu.md)
and a MindIE-SD build exposing the public `image_spans/image_sparsity` API.
The dependency source checked for this port is
[MindIE-SD `6965ff2`](https://gitcode.com/Ascend/MindIE-SD/tree/6965ff2f397c3370ef50d06b6564067d3205fccb).
Install its matching Python and native components using MindIE-SD's normal
installation instructions; the old RBS helper patch is insufficient.

```bash
git clone --branch feat/lq-slim-upstream --single-branch https://github.com/march7th-26/vllm-omni.git
cd vllm-omni
VLLM_OMNI_TARGET_DEVICE=npu pip install -e . --no-build-isolation
```

Prepare [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) and the
[Larry v4-step600 EMA adapter](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora).
The supported file is `minimax_h3_turbo_v4_step600_ema.safetensors`, SHA256
`5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3`.
It has 259 BF16 A/B pairs, including rank-16 AdaLN and rank-64 attention/MLP
deltas. It uses contiguous Q/K/V layout and is a different format from the
FlashGen and LightX2V Turbo adapters.

## Serve

Select four Ascend A5 devices using the standard device environment settings,
then replace the model and adapter paths:

```bash
export MODEL=/path/to/MiniMax-H3
export LORA=/path/to/minimax_h3_turbo_v4_step600_ema.safetensors
export VLLM_WORKER_MULTIPROC_METHOD=spawn

vllm serve "$MODEL" --omni --task-type ref2va \
    --host 127.0.0.1 --port 9098 --trust-remote-code \
    --num-gpus 4 --usp 4 --ring 1 --text-encoder-tp-size 4 \
    --vae-parallel-mode tile --vae-use-tiling --vae-patch-parallel-size 4 \
    --lora-path "$LORA" --lora-backend distill --lora-scale 1 \
    --diffusion-quantization-config '{"transformer":{"method":"mxfp8"}}' \
    --diffusion-attention-config '{"default":{"backend":"RAINFUSION_ATTN","block_sparse":{"sparsity":0.8,"precision":"mix"},"rainfusion":{"image_sparsity":0.9}}}' \
    --diffusion-attention-schedule '{"profiles":{"late":{"default":{"backend":"RAINFUSION_ATTN","block_sparse":{"sparsity":0.85,"precision":"mix"},"rainfusion":{"image_sparsity":0.9}}}},"default":[{"start":2,"end":null,"profile":"late"}]}' \
    --kv-offload-config '{"mode":"on","kv_refresh_interval":8,"kv_host_quantization":"none","kv_storage":"npu"}'
```

This uses the existing [attention schedule](../../docs/user_guide/diffusion/attention_backends/schedules.md)
interface. Profile options are complete configurations; include `rainfusion.image_sparsity`
and `precision` in each profile. The immutable adapter is applied as a BF16
forward delta after each base projection, including quantized projections. It
requires DiT TP=1, resident DiT weights, eight denoising steps and scale 1;
Ulysses SP is supported. Per-request LoRA switching is disabled for this adapter.

For workloads that need more NPU memory, append the normal text-encoder offload
option to the command above:

```bash
--diffusion-offload-config '{"mode":"module","components":["text_encoder"],"pin_memory":true}'
```

## Request

Use the standard synchronous video endpoint:

```bash
curl --fail-with-body http://127.0.0.1:9098/v1/videos/sync \
    --form-string 'prompt=Use the object in <Picture 1> as the subject. A slow camera push-in in soft studio lighting, with quiet room ambience.' \
    --form-string 'short_edge=768' \
    --form-string 'aspect_ratio=9:16' \
    --form-string 'fps=24' \
    --form-string 'num_inference_steps=8' \
    --form-string 'flow_shift=12.0' \
    --form-string 'audio_flow_shift=3.0' \
    --form-string 'seed=42' \
    --form-string 'extra_params={"task":"ref2va","duration":4}' \
    --form 'input_references=@/path/to/reference.jpg;type=image/jpeg' \
    --output output.mp4
```

Repeat `input_references` in prompt order to supply several references.

## Controls and limits

| Control | Meaning |
| --- | --- |
| `rainfusion.image_sparsity` | Fraction of candidate image blocks dropped per image, head and eligible video query. Range `[0, 1]`; default 0 disables image selection. At least one candidate is kept. |
| `sparsity` | Nominal fraction of pooled key blocks dropped. Protected blocks and reference selection change the realized mask. |
| `kv_storage` | `host` retains the existing layerwise host cache; `npu` keeps unquantized reference KV on the device and avoids host copies. Default `host`. |
| `kv_refresh_interval` | With eight-step inference, 8 refreshes at the first step and reuses references for seven steps. It is unrelated to INT8 cache storage. |
| `precision=mix` | Q/K are quantized to INT8 and V to FP8 inside attention. Inputs, cached reference KV and outputs remain BF16. |

Image selection applies independently to eligible reference-video and target-video
query blocks; protected queries and image boundary blocks remain protected.
It changes attention connections, without shrinking the reference KV pool.
When image selection is active, compact reuse pads dummy reference queries after
Ulysses to satisfy MindIE-SD's equal-length Q/K/V interface, then discards their
outputs. Reference projections and KV are still reused. Without image selection,
A5 resident KV uses rectangular BF16/mix attention. Compact SP transport padding
is excluded from attention and restored for reverse exchange. Ring is unsupported.

## Validation

This branch ports the complete scheme onto official main
`62d0e880a2c7c42903cc384d3b805af4c21a3326`. Earlier results from the separate
`lq-slim` branch do not validate this revision. NPU end-to-end execution,
performance and output quality for this port remain unverified.

Focused CPU tests cover configuration, image ownership, step-profile dispatch,
resident/host KV values and cleanup, transport padding and LoRA
arithmetic/binding. Native attention calls in routing tests
are spies; no device operator or distributed execution is simulated.

For the optional dependency-source tests, set `MINDIE_SD_SOURCE` to the checked
MindIE-SD checkout; otherwise those tests skip.

For a minimal CPU environment, install PyTorch, safetensors, pytest,
pytest-mock, pytest-asyncio and einops, then run:

```bash
python -m pytest --noconftest -o addopts='' tests/diffusion/models/minimax_h3/test_reference_block_selection.py tests/diffusion/models/minimax_h3/test_reference_kv_resident.py tests/diffusion/models/minimax_h3/test_ref2va_lora.py tests/diffusion/models/minimax_h3/test_image_sparse_mask.py -q
```

With the full repository test dependencies, the CI-aligned command is:

```bash
python -m pytest tests/diffusion/models/minimax_h3/test_reference_block_selection.py tests/diffusion/models/minimax_h3/test_reference_kv_resident.py tests/diffusion/models/minimax_h3/test_ref2va_lora.py tests/diffusion/models/minimax_h3/test_image_sparse_mask.py -m 'core_model and cpu' --run-level core_model -q
```
