# MiniMax-H3 reference optimizations

These opt-in extensions add NPU-resident reference KV and per-image reference
block selection to MiniMax-H3 Ref2VA. Both features can be composed with your
existing model, LoRA, weight quantization, sampling and serving configuration.
An optional Larry adapter integration is included with the recommended
eight-step setup below. The Ref extensions do not require that adapter or a
fixed step count.

## Install

Use the normal [Ascend installation](../../docs/getting_started/installation/npu.md)
with this source branch:

```bash
git clone --branch h3-ref-opt-main --single-branch https://github.com/march7th-26/vllm-omni.git
cd vllm-omni
VLLM_OMNI_TARGET_DEVICE=npu pip install -e . --no-build-isolation
```

Reference-image sparsity uses the official MindIE-SD implementation from
[PR #859](https://gitcode.com/Ascend/MindIE-SD/pull/859). Build MindIE-SD from
official `dev` at `b7386be8b8397581a63dcc78d73feb64eabe92a0` and install it
separately from vLLM-Omni. The rectangular reuse path uses the image-aware RF
permutation helper from that revision.

## Add reference KV reuse

On Ascend, append the existing KV configuration flag to your normal
`vllm serve ... --omni --task-type ref2va` command:

```bash
--kv-offload-config '{"mode":"on","kv_storage":"npu","kv_refresh_interval":8,"kv_host_quantization":"none"}'
```

The first step captures reference KV on the device. Subsequent steps reuse it
until the refresh interval is reached. For example, interval 8 refreshes at
steps 0, 8, 16 and so on; a different denoise step count does not need a code
change. Choose the interval for your workload and evaluate output quality.

`kv_storage` defaults to `host`, which preserves the existing layerwise host
prefetch. `npu` avoids host copies and staging-ring transfers but increases
device memory use. It requires unquantized cache storage. This restriction is
about stored reference KV; existing DiT weight quantization stays independent.

The cache is local to one denoise request and is released when it closes.
Model weights and the active adapter must remain fixed during that request.
No SP and pure Ulysses SP are supported; Ring and paged KV are unsupported.
Compact SP transport padding is excluded from attention and cache capture.

## Add reference-image sparsity

For an existing RainFusion deployment, retain its `block_sparse` options and add
`rainfusion.image_sparsity` in the same attention spec. For example:

```bash
--diffusion-attention-config '{"default":{"backend":"RAINFUSION_ATTN","block_sparse":{"sparsity":0.8,"precision":"mix"},"rainfusion":{"image_sparsity":0.9}}}'
```

MiniMax-H3 supplies each reference image's start and two-dimensional token grid.
`image_sparsity` is the fraction of candidate blocks dropped, in `[0, 1]`, with
default `0` disabling image sparsity. The option can be used with or without
reference KV reuse and independently of video sparsity.

RainFusion packs each image on a block boundary. For each eligible video query
block, head and image, it keeps `max(1, ceil(N * (1 - image_sparsity)))` of the
image's `N` complete candidate blocks. First-frame query blocks and mixed or
partial image blocks retain their protection. This changes attention
connections while retaining the reference KV allocation.

Full-length steps use the MindIE-SD API. On compact reuse steps, Omni maps
the same block layout to the cached KV and surviving Q positions, then selects
blocks from the rectangular score matrix.

When combined with compact reference KV on A5, the rectangular attention path
supports BF16 and mix. Mix uses the existing MindIE-SD Q/K INT8 and V FP8
quantization functions and operator; cached KV and outputs remain BF16. Other precision and backend
combinations require separate compatibility validation.

## Recommended eight-step LoRA configuration

For eight-step Ref2VA, prepare the
[Larry v4-step600 EMA adapter](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)
separately from the source installation. The supported file is
`minimax_h3_turbo_v4_step600_ema.safetensors`, SHA256
`5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3`.
It contains 259 BF16 A/B pairs with rank-64 attention/MLP and rank-16 AdaLN
deltas, using contiguous Q/K/V. The loader validates this artifact before
binding; FlashGen and LightX2V formats retain their existing loading paths.

Select four Ascend A5 devices through the standard device settings. This
example uses existing MXFP8 weight quantization and mix attention, with the
two Ref options enabled:

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
    --kv-offload-config '{"mode":"on","kv_storage":"npu","kv_refresh_interval":8,"kv_host_quantization":"none"}'
```

For workloads that need more NPU memory, append the normal text-encoder offload
option; DiT weights must remain resident for this adapter:

```bash
--diffusion-offload-config '{"mode":"module","components":["text_encoder"],"pin_memory":true}'
```

Use the normal synchronous endpoint and an eight-step request:

```bash
curl --fail-with-body http://127.0.0.1:9098/v1/videos/sync \
    --form-string 'prompt=Use the object in <Picture 1> as the subject. A slow camera push-in with quiet room ambience.' \
    --form-string 'short_edge=768' --form-string 'aspect_ratio=9:16' \
    --form-string 'fps=24' --form-string 'num_inference_steps=8' \
    --form-string 'flow_shift=12.0' --form-string 'audio_flow_shift=3.0' \
    --form-string 'seed=42' \
    --form-string 'extra_params={"task":"ref2va","duration":4}' \
    --form 'input_references=@/path/to/reference.jpg;type=image/jpeg' \
    --output output.mp4
```

The adapter is installed once at startup and adds BF16 `B(A(x))` deltas after
base projections, including quantized projections. It requires scale 1,
eight steps, DiT TP=1 and resident DiT weights; Ulysses SP is supported.
Per-request adapter switching is disabled for this artifact. These restrictions
apply when this adapter is selected, independently of the Ref options.

Use a fixed adapter and sampler when evaluating reference fidelity and
audio/video quality. Selecting this LoRA alone does not establish quality for
a particular sparsity or precision setting. The Ref features remain usable
with other compatible adapters and sampling configurations.

## Optional per-step profiles

Use the existing [attention schedule](../../docs/user_guide/diffusion/attention_backends/schedules.md)
if your workload benefits from different sparsity at different steps. The
following is an optional working point; neither extension requires it:

```bash
--diffusion-attention-schedule '{"profiles":{"late":{"default":{"backend":"RAINFUSION_ATTN","block_sparse":{"sparsity":0.85,"precision":"mix"},"rainfusion":{"image_sparsity":0.9}}}},"default":[{"start":2,"end":null,"profile":"late"}]}'
```

Each profile is a complete configuration. Keep the base attention configuration
from the preceding example for steps outside the schedule. Continue using the
normal request endpoint and your existing sampling and LoRA flags.

## Validation

The contribution is based on official main
`4bc340899368f817c1c481b9c5a060c705270304`, which already includes host reference
KV reuse. This branch adds device storage, compact SP alignment and image
selection integration plus the optional adapter hookup. Earlier experiments
of a complete LoRA/quantization/reference configuration do not validate this
revision or attribute performance to the two extensions individually.

CPU tests cover typed configuration, ownership, default-path preservation,
step-profile dispatch, host/resident KV values, refresh intervals and cleanup,
SP padding/metadata, and LoRA arithmetic/binding. Native calls in routing tests
are recording doubles;
they do not execute or simulate device operators or distributed communication.
NPU end-to-end execution, performance and output quality remain unverified.

For a minimal CPU environment, install PyTorch, safetensors, pytest, pytest-mock
and pytest-asyncio. Run:

```bash
python -m pytest --noconftest -o addopts='' tests/diffusion/models/minimax_h3/test_reference_block_selection.py tests/diffusion/models/minimax_h3/test_reference_kv_resident.py tests/diffusion/models/minimax_h3/test_ref2va_lora.py -q
```

With the full repository test dependencies, the CI-aligned command is:

```bash
python -m pytest tests/diffusion/models/minimax_h3/test_reference_block_selection.py tests/diffusion/models/minimax_h3/test_reference_kv_resident.py tests/diffusion/models/minimax_h3/test_ref2va_lora.py -m 'core_model and cpu' --run-level core_model -q
```
