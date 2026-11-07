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

### Ascend A3: BF16 text-to-image

The A3 path targets BF16, TP=1 and TP=2, with either the base checkpoint or
startup fusion of the **original** `SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors`.
The V2 adapter is a different checkpoint and is not used by these commands.

The 2026-10-26 A3 experiment used PR #9197 commit
`7d0eb181ceaa6dd311e91c8547faa705f6c37e9b`. Its base-model TP1 smoke and original
LoRA TP1/TP2 generation checks completed. The repeated performance run covered
TP1/TP2 with profiling on/off, all using the original LoRA; it did not cover
four different base/LoRA configurations. These are generation and performance
observations, not a full regression suite or dataset-quality evaluation.

| Configuration | Generation settings | A3 validation |
| --- | --- | --- |
| BF16, TP=1, no LoRA | 50 steps, CFG 4.0 | 512×512 smoke passed |
| BF16, TP=2, no LoRA | 50 steps, CFG 4.0 | Not run |
| BF16, TP=1, original LoRA | 8 steps, CFG 1.0 | 2720×1536, 20 measured requests per profiling setting |
| BF16, TP=2, original LoRA | 8 steps, CFG 1.0 | 1024×1024 and 2560×1760 smoke; 2720×1536 repeated run |

#### Recorded image and runtime

The Pod used the private container tag
`vllm:nightly-a3-20261018170400_aarch64`, with 16 CPU cores, 200 GiB host memory,
200 GiB persistent storage and two allocated A3 compute devices. Each device
reported 64 GiB HBM. TP1 used one logical device; TP2 used both. This device
count is not a statement about the number of physical boards.

The container additionally used an existing isolated `runtime032` Python
environment copied from the reference installation. **The container tag alone
does not recreate the measured environment.** No immutable container digest
was recorded. The actual service environment was:

| Component | Recorded version |
| --- | --- |
| OS / architecture | Linux, aarch64 |
| Python | 3.12.13 |
| CANN / driver | 9.1.0 / 25.5.1.1 |
| PyTorch / torch-npu | 2.10.0+cpu / 2.10.0.post4 |
| vLLM | 0.32.0+empty |
| vLLM-Ascend | 0.19.1rc2.dev2959+g450e55698 |
| vLLM-Omni installation label | 0.32.0.dev0+g7d0eb181.npu |
| Transformers / Diffusers | 5.19.0 / 0.40.0 |
| MindIE-SD | 3.1.0 |

These are the recorded package labels, including `+cpu` and `+empty`; the
service executed on NPU through `torch_npu`. The Omni installation label was
set with `VLLM_OMNI_VERSION_OVERRIDE` for the pinned archive source, not an
official release. The deployment audit compared 5,320 archive files and
reported zero source changes; installation metadata was tracked separately.

The backend was `FLASH_ATTN`, with BF16, eager execution and CFG parallel size 1.
Runtime settings included:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN
export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=1024
export VLLM_OMNI_SENSENOVA_PAGED_DECODE=0
export LD_PRELOAD=/lib/aarch64-linux-gnu/libjemalloc.so.2
```

The model was `sensenova/SenseNova-U1.5-8B-MoT`, with the original adapter
SHA-256 `3ef32180cdf1e30a870a83f4f136e897ea50b7ee467f863d75633464ebb25708`.
Resolved public checkpoint revisions were not recorded; the evidence records
the mounted model config and adapter hashes.

After provisioning that runtime, the unprofiled TP2 service used the following
launch configuration. Set `MODEL` and `LORA_PATH` to the local checkpoint and
original adapter; use one logical device and TP1 for the single-device arm.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
vllm serve "$MODEL" --omni --served-model-name SenseNova-U1.5-8B-MoT \
    --host 0.0.0.0 --port 8091 --trust-remote-code --dtype bfloat16 \
    --tensor-parallel-size 2 --cfg-parallel-size 1 --enforce-eager \
    --lora-path "$LORA_PATH" --lora-backend distill
```

#### Measured generation and performance

Initial smoke requests used a mountain-lake prompt, seed 42 and `think=false`:

| Configuration | Width×height | Steps / CFG | HTTP | Single-request client time |
| --- | --- | --- | --- | --- |
| TP1, base | 512×512 | 50 / 4.0 | 200 | 6.975 s |
| TP2, original LoRA | 1024×1024 | 8 / 1.0 | 200 | 2.179 s |
| TP2, original LoRA | 2560×1760 | 8 / 1.0 | 200 | 8.256 s |

These are single-request smoke observations, not steady-state medians.

The subsequent fixed-prompt performance run used **width 2720, height 1536**,
the original 8-step LoRA, CFG 1.0, `think=false`, `cfg_norm=none`,
`timestep_shift=3.0`, `t_eps=0.02`, seed 42 and batch/concurrency 1. Each of
TP1/TP2 with profiling on/off ran three warmups followed by 20 measurements:
92 requests total, 80 measured, all returning images.

| TP | Profiled text mean | Profiled 8-step denoise mean | Profiled generation mean | Plain E2E mean | Plain E2E median | Plain E2E P95 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.2726 s | 3.6148 s | 3.6979 s | 6.2191 s | 6.2193 s | 6.2251 s |
| 2 | 0.1962 s | 2.3250 s | 2.4113 s | 4.8812 s | 4.8795 s | 4.8892 s |

Stage columns come from a separate profiling copy with a
[timing-only patch](validation/u15-a3-20261026-timing.patch); plain E2E comes
from unchanged PR production code with profiling disabled. Pure denoising is
included in generation, so the columns must not be added. Stage boundaries
synchronize NPU execution and reduce complete intervals by MAX over TP ranks.
E2E includes the complete HTTP body and JSON parsing, but excludes service
startup, loading, image decoding and saving. P95 uses linear interpolation
over 20 measurements. The same-TP first measured PNG images with profiling on/off
were byte-identical; this does not establish TP1-versus-TP2 equivalence.

The [recorded workload and request timings](validation/u15-a3-20261026.json)
contain the exact prompt, settings, all 92 request records, package versions,
source checks and saved image hashes. The final health/model endpoints
returned 200 at experiment completion; this is not a current service-status
claim. Native NPU/CUDA regression completion and base-only TP2 remain pending.

#### Dependencies and checkpoints

Follow the [NPU installation guide](../../docs/getting_started/installation/npu.md).
Use matching Ascend driver/firmware, CANN, `torch`, `torch_npu`, `vllm`, and
`vllm-ascend` versions for the checked-out Omni revision. In particular, the
older release container in that guide is not proof of compatibility with
current main. Install this checkout in the validated environment; do not
replace its dependency pins with the GPU versions listed above. The serving
benchmark additionally uses the existing `aiohttp` and `tqdm` client dependencies.
The existing NPU FlashAttention backend requires a compatible MindIE-SD
installation; without it, use the existing SDPA backend. Additive prefix masks
and unqualified dense kernel shapes use SDPA even when FlashAttention is selected.

Obtain the base checkpoint and exactly the original adapter from their public
repositories (these commands download weights only when you run them):

```bash
hf download sensenova/SenseNova-U1.5-8B-MoT --local-dir models/SenseNova-U1.5-8B-MoT
hf download sensenova/SenseNova-U1.5-8B-MoT-LoRAs \
    SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors --local-dir models/SenseNova-U1.5-LoRAs
```

Record the resolved Hugging Face revisions when downloading; pass `--revision`
to repeat a run with the same weights. No GitCode patch or offline merged
checkpoint is needed. The existing `distill` loader fuses the adapter once
while loading each worker's base weights; requests do not attach a PEFT adapter.
Omitting `LORA_PATH` preserves the no-LoRA path.

#### Launch and generate: all four configurations

Run from the repository root. Choose one row below, then start the server.
Stop the preceding server before selecting another row; each configuration
requires fresh base weights so that a previous fusion is not reused.

```bash
export MODEL=models/SenseNova-U1.5-8B-MoT
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Choose one configuration:
export ASCEND_RT_VISIBLE_DEVICES=0 TP=1 LORA_PATH=""       # TP1, base
# export ASCEND_RT_VISIBLE_DEVICES=0,1 TP=2 LORA_PATH=""   # TP2, base
# export ASCEND_RT_VISIBLE_DEVICES=0 TP=1 LORA_PATH=models/SenseNova-U1.5-LoRAs/SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors
# export ASCEND_RT_VISIBLE_DEVICES=0,1 TP=2 LORA_PATH=models/SenseNova-U1.5-LoRAs/SenseNova-U1.5-8B-MoT-LoRA-8step.safetensors

bash recipes/SenseNova/run_u15_a3.sh serve
```

The script uses the formal `vllm serve --omni --dtype bfloat16
--tensor-parallel-size 1|2 --enforce-eager` interface. With `LORA_PATH` it adds
`--lora-path "$LORA_PATH" --lora-backend distill`. There is no `torchrun` wrapper:
the engine owns worker creation. Choose visible device IDs appropriate to the
host and check available device memory before a high-resolution run.

In another terminal, set the **same** `MODEL`, `TP`, `LORA_PATH` and `PORT`
(default `8091`) as the server, then run:

```bash
OUTPUT=sensenova-u15-test bash recipes/SenseNova/run_u15_a3.sh generate
```

This reuses the existing chat client and saves a 2560×1760 PNG. It sends one
prompt/image, seed 42, think off, `cfg_norm=none`, `timestep_shift=3.0` and the
pipeline's default `t_eps=0.02`. Base weights use 50 steps/CFG 4.0; the original
LoRA uses 8 steps/CFG 1.0. Do not leave `LORA_PATH` set on the client while using
a base-only server: the client cannot inspect the adapter loaded by a server.

The existing offline runner is also usable for both TP sizes and optional
startup LoRA. Its unquantized diffusion default is BF16:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
    --model "$MODEL" --tensor-parallel-size "$TP" --enforce-eager \
    --prompt "A mountain lake at sunrise, detailed reflections, natural colors." \
    --width 2560 --height 1760 --seed 42 --num-inference-steps 50 --cfg-scale 4.0 \
    --extra-body '{"think": false, "cfg_norm": "none", "timestep_shift": 3.0, "t_eps": 0.02}' \
    --output sensenova-u15-base.png
```

For the original LoRA, add `--lora-path "$LORA_PATH" --lora-backend distill`,
change to `--num-inference-steps 8 --cfg-scale 1.0`, and use a different output
filename. Run it once with `TP=1`, then with `TP=2` and two visible devices.

#### Steady-state benchmark

Restart the LoRA server with profiling enabled, using the same configuration
variables in both terminals:

```bash
PROFILE=1 bash recipes/SenseNova/run_u15_a3.sh serve
```

Then, in the **same host/software environment** as the server:

```bash
export SERVER_METADATA='{"hardware":"Ascend A3","base_model_revision":"REPLACE_WITH_RESOLVED_REVISION","adapter_revision":"REPLACE_WITH_RESOLVED_REVISION","cann":"REPLACE_WITH_INSTALLED_VERSION","driver":"REPLACE_WITH_INSTALLED_VERSION"}'
WARMUP=2 REPEATS=5 OUTPUT=results/sensenova-u15-a3-tp1 \
    bash recipes/SenseNova/run_u15_a3.sh benchmark
```

Use actual metadata values before publishing results. Repeat after restarting
with `TP=2`, two visible devices and `OUTPUT=results/sensenova-u15-a3-tp2`.
The benchmark requires the original adapter filename, records its SHA-256,
Omni commit/dirty state and installed software versions, and writes `.json`
and `.csv`. Preserve the source diff separately when `omni_dirty` is true.
Server metadata is declared by the operator, not discovered remotely; confirm
that the running server has the same TP, dtype and adapter settings.

The generic benchmark reuses the existing chat request backend. It runs
**2560×1760, original LoRA, 8 steps, batch=1, concurrency=1**, discards two full
warmup requests by default, then records every measured request separately:

- `text_seconds`: tokenization, conditional/unconditional prefix prefill,
  optional think generation and KV cache preparation. The supplied workload
  disables think. Initial image noise and scheduler setup are excluded.
- `generation_seconds`: the complete denoising loop, image feature extraction,
  scheduler updates, cache cleanup and conversion to a CPU PIL image.
- `end_to_end_seconds`: client wall time from request dispatch through receiving
  and parsing the complete response, including queuing, image encoding and
  transport; excludes server startup, weight loading and saving results.

The existing pipeline profiler synchronizes the active device before and after
stages, including NPU execution. After each worker completes a request, stage
metrics use MAX across its TP group. Each phase is timed as a whole before
reduction, rather than summing the maximum of individual rank timings. Text
and generation maxima may come from different ranks, so their sum is not an
end-to-end measurement. Profiling adds synchronization overhead; use the same
profiling setting for comparisons. A client-side NPU synchronization cannot
synchronize the remote workers and is deliberately unnecessary here.

Missing stage metrics, failed warmups and failed requests abort the benchmark
instead of producing zero timing values. Completed measurements are preserved
if a later request fails. JSON contains raw stage durations and run settings;
CSV contains one row per measured request. The results above came from the
recorded workload, which used 2720×1536 and a separate benchmark-only timing
copy; this helper defaults to 2560×1760 and does not export pure-loop timing.

#### Current scope and limitations

This adaptation covers eager BF16 text-to-image and TP=1/2 startup distilled
LoRA fusion. FP8, dynamic PEFT switching, image editing/understanding, CFG
parallelism, CPU offload, graph capture and other TP degrees have not been
validated on A3 by this change. CUDA paged decode remains a CUDA-specific
optimization; NPU uses the ordinary cache path. Unsupported optimized-kernel
conditions use the existing correct fallback; runtime kernel failures are
surfaced rather than retried silently. CPU mathematical checks do not replace
the remaining A3 configuration checks or numerical parity checks on real NPU
kernels. The recorded generation/performance results do not establish
dataset-quality equivalence or validate untested feature combinations.

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
- Each request captures its own graphs, and a think request captures twice because the sequence
  grows past the 512 bucket, so the capture cost is paid per request rather than once at
  startup.
