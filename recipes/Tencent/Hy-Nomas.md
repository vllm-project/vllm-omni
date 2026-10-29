# Hy-Nomas video generation — H200

## Summary

- Vendor: Tencent Hunyuan
- Model: [`tencent/Hy-Nomas-Diffusers`](https://huggingface.co/tencent/Hy-Nomas-Diffusers), BF16 distilled checkpoint
- Task: text-to-video with segment prompts and optional camera control
- Modes: offline generation, `/v1/videos`, realtime streaming, and multi-turn continuation
- Hardware: one or two NVIDIA H200 GPUs, 141 GB per device
- Runtime: `HyNomasPipeline`, BF16, with the pinned Hy-Nomas reference package
- Maintainer: vLLM-Omni community

Use this recipe for text-controlled video generation and interactive continuation.
The pipeline manages generation and loads the DiT and PixelUMM shards through
vLLM-Omni's component weight loader. It uses the checkpoint's bundled text encoder
and tokenizer. Model computation, text preprocessing, and the Euler scheduler
retain the pinned Hy-Nomas reference implementation.
`HyNomasModularPipeline` in the checkpoint index selects this vLLM-Omni pipeline.
The author's Diffusers fork is not needed for this integration.

## Model architecture

Qwen text conditioning and optional camera geometry guide a segment-autoregressive
dual-stream DiT; retained KV connects segments, and PixelUMM decodes DCAE-V latents.

| Component | Structure | Parameters | BF16 weights (GiB) |
| --- | --- | --- | --- |
| DiT | 54 blocks, hidden 2048, 16×128 heads, FFN 8192; token refiner included | 8.55B | 15.93 |
| Qwen2.5-VL language component | 28 layers, hidden 3584 | 7.07B | 13.17 |
| PixelUMM decoder | 36 layers, hidden 4096, 32 query / 8 KV heads, one decode step | 6.97B | 12.99 |

The loaded components total 22.60B parameters / 42.09 GiB BF16 weights, excluding
KV, staging, activations and workspaces. DCAE-V uses 32 channels, 32× spatial and
4× temporal compression; PixelUMM uses 10-latent windows with overlap 5.
Parameter counts use checkpoint-index bytes divided by two and agree with
actual loaded component bytes; 8.55B describes the DiT alone.

## Supported model contract

| Task | Entrypoint | Input | Output |
| --- | --- | --- | --- |
| Video generation | Shared `text_to_video.py` or `/v1/videos` | One prompt or multiple text segments; optional camera trajectory | One silent video |
| Live interaction | `/v1/realtime/video` and the shared streaming client | Initial prompt, then prompt/camera updates at chunk boundaries | Stable video chunks followed by the final decoder tail |
| Multi-turn continuation | `hy_nomas_session.py` using AR-Diffusion ticks | Successive prompt/control snapshots with one session ID | Stable video chunks sharing the session history |

| Parameter | Contract |
| --- | --- |
| Geometry | Height and width divisible by 32; default 704×1280 |
| Frame count | Current API validation accepts `num_frames=4*n+1`, with `n >= 1`; default 121 |
| Frame rate | 24 fps |
| Sampling | 4, 6, or 8 steps; default 8; guidance scale 1 and flow shift 5 |
| Negative prompt | Omit or leave empty; nonempty values are rejected because the distilled checkpoint uses CFG=1 |
| Segment boundaries | Pixel-frame indices starting at 0 and ending at `num_frames`; internal boundaries divisible by 4 |
| Camera | Reference generator name, such as `pan_left`, or a worker-accessible JSON/NPZ trajectory file |
| Persistent turn | At most 89 frames with the default nominal 16-latent chunk and short-tail merging; height, width, and seed remain fixed within a session |
| Default AR capacity | 120 latent frames, 16 prompt segments, one active session, 704×1280 spatial limits |
| Explicit recent mode | Shared video/text history budget; independent total frame/segment limits; incremental-v1 sampling |

Recent history applies to AR sessions. Use the session example or the recent
deployment profile to enable it; the ordinary offline pipeline uses full history.

Each `|||`-separated text describes one segment. Supply prepared reference segment
JSON as its text when using structured prompts. Prompt enhancement with Nomas-PE
and the reference dataset prompt-file loader are separate upstream tools.
Input images, audio generation, and latent output are unsupported.
PixelUMM decodes four video frames per latent frame and crops to the requested
length. The `4*n+1` restriction is enforced by this API's current validator.

The reference's 720-pixel height is reduced to 704 by its latent grid; this API
requires the actual output height. To specify duration, omit `num_frames` and pass
`duration` in model extras: 4 seconds gives 97 frames, including the initial frame.
An explicit frame count must agree with duration; fractional durations must span
a positive multiple of four frame intervals. Use these parameters for clip length
instead of the video endpoint's top-level `seconds` field.

## Deployment profiles

| Profile | Devices | Configuration | Behavior |
| --- | --- | --- | --- |
| Ordinary generation | 1 H200 | `hy_nomas.yaml`, BF16, `mp`, eager | Offline and video API, resident and CPU offload |
| Sequence parallel generation | 2 H200 | `hy_nomas.yaml` plus `--ulysses-degree 2` | Offline and video API, resident and CPU offload |
| Realtime streaming | 2 H200 | `hy_nomas_ar.yaml`, Ulysses degree 2 | Prompt/camera updates, cancellation, and a subsequent request |
| Multi-turn continuation | 2 H200 | Session example, Ulysses degree 2 | Successive turns with a final decoder flush |
| Bounded recent history | 2 H200 | `hy_nomas_ar_recent.yaml`, Ulysses degree 2 | Explicit recent window and long-session limits; qualification described below |

The converted checkpoint was qualified on one and two H200s at vLLM-Omni commit
`81620d2d69dc661d694fdc66ae737be0c7d8993d`. See [Verification](#verification)
for the tested workloads.

Weights, text encoding, and PixelUMM decoding remain replicated with sequence
parallelism. The degree must divide the DiT's 16 attention heads and the spatial
token count per latent frame. Real-model validation covers degrees one and two.
Memory and latency depend on the workload; see [Verification](#verification)
for the measured configurations and revisions.
CPU offload requires sufficient GPU memory for initial component loading and host
memory for the staged weights.
The legacy `--enable-cpu-offload` flag swaps the DiT, text encoder, and PixelUMM
decoder. Explicit `diffusion_offload_config.components` accepts `dit` and
`text_encoder`; unselected components, including the decoder, remain resident.
Selecting `vae` is unsupported and raises an error.

## Software and checkpoints

Validation used Linux, Python 3.12.13, PyTorch 2.13.0+cu130, CUDA 13.0,
vLLM 0.32.0, Transformers 5.16.1, Diffusers 0.40.0, and FlashAttention 2.8.3.
The NVIDIA driver was 570.133.20.

Install vLLM-Omni first, then install the pinned reference checkout without
replacing the environment's existing dependency versions:

```bash
git clone https://github.com/wtz2333/Hy-Nomas.git
git -C Hy-Nomas checkout --detach 16a4c228fd6f4770ed4a111b360031f7686c4a19
python -m pip install --no-deps -e ./Hy-Nomas
python -m pip install hydra-core==1.3.2 omegaconf==2.3.0 loguru einops
```

Keep the editable checkout: the integration reads its configuration files.
This fork adds opt-in rolling PixelUMM decoding to reference revision
`911cea6458a89c2663513ff7099561ec190dd491`. The default reference decoder is
unchanged. AR sessions use stable-prefix rolling decoding without retained
output history, replace-tail preview, or rollback.
Install FlashAttention 2 for the environment's PyTorch/CUDA versions using the
[reference installation instructions](https://github.com/Tencent-Hunyuan/Hy-Nomas/tree/911cea6458a89c2663513ff7099561ec190dd491).
Verify imports before loading weights:

```bash
python -c 'from hymm.pipeline.causal_ar import CPTCausalInferencePipeline'
```

Download the component checkpoint:

```bash
hf download tencent/Hy-Nomas-Diffusers \
  --revision a1ae04d11ae4ea176d1f76de634c115944f9882d \
  --local-dir ./Hy-Nomas-Diffusers
```

Keep `modular_model_index.json` and the `transformer/`, `decoder/`, `text_encoder/`,
`tokenizer/`, and `scheduler/` directories. The pipeline uses each weight index to
load its shards and checks required parameters, duplicate names, and tensor shapes.
The published DiT attention names are mapped to the reference component's names.
The Hub ID also works directly; standard modular-checkpoint discovery selects the
pipeline without a generated index or model-name special case.

The bundled text encoder is `Qwen2_5_VLTextModel`. To override it with an existing
Qwen text encoder and its tokenizer,
pass `--custom-pipeline-args '{"text_encoder":"/path/to/Qwen2.5-VL-7B-Instruct"}'`
to the server, or `custom_pipeline_args={"text_encoder": "/path/to/Qwen2.5-VL-7B-Instruct"}`
to the Python API.

## Offline generation

Run commands from the vLLM-Omni repository:

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model ./Hy-Nomas-Diffusers \
  --deploy-config vllm_omni/deploy/hy_nomas.yaml \
  --prompt 'A forest beside a river. ||| The camera moves along the river.' \
  --height 704 --width 1280 --num-frames 121 --num-inference-steps 8 \
  --guidance-scale 1 --fps 24 --seed 42 \
  --extra-body '{"segment_frame_boundaries":[0,60,121]}' \
  --output /path/to/results/hy_nomas.mp4
```

Add `--ulysses-degree 2` for two GPUs. The default single-segment boundary is
`[0, num_frames]`. Optional `camera_trajectory` and `segment_has_camera` fields
go in `--extra-body`; the latter contains one boolean per segment.

The deployment selects the `mp` executor and eager execution. Python callers
must set `distributed_executor_backend="mp"` and `enforce_eager=True`.
The reference package needs process isolation. The first real request includes
reference-kernel compilation; generic startup warmup is disabled because its
sampling parameters are outside this checkpoint's contract.

## Online generation

```bash
vllm serve ./Hy-Nomas-Diffusers --omni --port 8098 \
  --deploy-config vllm_omni/deploy/hy_nomas.yaml
```

Submit a video job from another terminal:

```bash
curl -sS http://localhost:8098/v1/videos \
  -F 'prompt=A forest beside a river.' -F 'size=1280x704' \
  -F 'num_frames=121' -F 'fps=24' -F 'num_inference_steps=8' \
  -F 'guidance_scale=1' -F 'seed=42'
```

Poll `GET /v1/videos/{id}` until the returned job is `completed` or `failed`.
Download completed output with `GET /v1/videos/{id}/content`.
Model extras use the JSON `extra_params` form field, for example
`-F 'extra_params={"duration":4}'` when `num_frames` is omitted, or
`-F 'extra_params={"segment_frame_boundaries":[0,60,121],"camera_trajectory":"pan_left","segment_has_camera":[true,true]}'`
with a two-segment prompt and 121 frames. `generate_sound=false` is accepted.

## CPU offload

Select the components to offload with the shared configuration:

```bash
vllm serve ./Hy-Nomas-Diffusers --omni \
  --deploy-config vllm_omni/deploy/hy_nomas.yaml \
  --diffusion-offload-config '{"mode":"module","components":["dit","text_encoder"]}'
```

This configuration leaves PixelUMM resident. The compatibility flag
`--enable-cpu-offload` stages all three components. Module offload also works
with Ulysses SP; each rank stages its own weights. AR step execution transfers
DiT weights for each forward, increasing transfer overhead.

## Streaming and live interaction

Start the two-GPU AR-Diffusion deployment:

```bash
vllm serve ./Hy-Nomas-Diffusers --omni --port 8098 \
  --deploy-config vllm_omni/deploy/hy_nomas_ar.yaml
```

Connect the shared streaming client:

```bash
python examples/online_serving/streaming_video_generation/streaming_video_client.py \
  --model ./Hy-Nomas-Diffusers --port 8098 --no-helios-distilled-preset \
  --prompt 'A forest beside a river.' --height 704 --width 1280 \
  --num-frames 241 --num-inference-steps 8 --guidance-scale 1 --fps 24 --seed 42 \
  --prompt-updates '[{"at":2,"prompt":"The camera approaches a wooden bridge.","transition_chunks":0}]' \
  --camera-updates '[{"at":3,"camera":{"camera_trajectory":"pan_left"},"transition_chunks":0}]' \
  --output /path/to/results/hy_nomas_stream.mp4
```

Updates take effect at chunk boundaries. Set `transition_chunks=0`, or omit it
when sending protocol messages directly. A camera event uses
`event.multi_modal_data.camera={"camera_trajectory":"pan_left"}`; a null
`camera_trajectory` disables camera control. Composite prompt/camera updates
are validated together. Chunk metadata acknowledges applied and coalesced event IDs.

PixelUMM emits stable frames and retains an overlapping tail between chunks.
Short chunks may emit no frames. Completion flushes the tail; cancellation or
disconnect releases the session and discards the withheld tail.

Change `ar_height`, `ar_width`, `ar_chunk_latent_frames`, `ar_max_latent_frames`,
`ar_max_segments`, and `ar_session_capacity` through `custom_pipeline_args`.
Requests must fit the configured limits; increasing capacity requires more memory.
Keep `ar_max_latent_frames` fixed when comparing seeds because it determines the
noise timeline.

### Bounded recent history

Use `vllm_omni/deploy/hy_nomas_ar_recent.yaml` for explicit `recent` history.
The default `full` deployment retains its fixed noise timeline and limits.

| Custom pipeline argument | Recent-mode meaning |
| --- | --- |
| `ar_history_mode` | `full` by default; set `recent` explicitly |
| `ar_history_latent_frames` | Shared video/text token budget in latent-frame equivalents; the recent profile uses 36 |
| `ar_max_total_latent_frames` | Absolute generation limit, default 4096, independent of resident history |
| `ar_max_total_segments` | Cumulative prompt-segment limit, default 1024 |
| `ar_chunk_latent_frames` | Nominal chunk size, default 16; it must fit the recent budget |

The retained budget includes text. Eviction removes the smallest complete-frame
prefix that satisfies it, dropping a frame's trailing text together with that
frame. It preserves absolute RoPE/frame progress. Publication waits for every
layer and SP rank before releasing old KV. Pending pages, per-segment page
rounding, and contiguous attention staging are reserved separately.
The reference merges a tail shorter than half a nominal chunk into its
preceding chunk, within the same prompt segment. With a nominal size of 16,
pending allocation therefore covers up to 23 latents. A 33-latent request uses
16+17; it does not generate a separate one-latent tail.

`recent` uses sampling version `incremental_v1`: a CPU generator draws one
latent frame at a time, and a separately seeded CPU generator supplies KV-commit
perturbations. Decoder noise uses the reference's seed plus absolute frame
index. Changing only the history budget does not change the generation-noise
sequence. This explicit mode has a different noise sequence from legacy
`full`; windowing also changes conditioning and requires its own quality check.

PixelUMM keeps only latents and active mappings needed for future windows.
Normal completion marks the final segment and flushes the tail. Cancellation
discards withheld pixels. Every public chunk, including empty stable output,
can carry `frame_start`, `generated_raw_frames`, `generated_frames`, and
`withheld_frames`; these distinguish raw decoder positions from visible output.
Reaching a total limit fails the session explicitly and releases its state.

At 1280×704, set `streaming_pacing.max_output_bytes` explicitly when enabling
playback pacing. For the default 16-latent chunks, 512 MiB
(`536870912` bytes) covers decoded RGB chunks including the final held tail.
The shared 64 MiB default can reject a decoded chunk before a playback idle
timeout is reached. The limit applies to decoded payloads and encoded bytes.
The playback window must also admit a complete chunk; the tested request uses
`max_lead_ms=10000`. Send `session.playback` progress while playing.

For a 120-second request, start the recent deployment and use the shared client
above with `--num-frames 2881`. The default remains eight steps; four steps are
an explicit option. Recent history is functional qualification work, not a
real-time performance guarantee. GPU qualification status must be read with
the revision and workloads in [Verification](#verification).

## Multi-turn continuation

```bash
python examples/offline_inference/text_to_video/hy_nomas_session.py \
  --model ./Hy-Nomas-Diffusers --ulysses-degree 2 \
  --prompts 'A forest beside a river.' 'The camera approaches a wooden bridge.' \
  --height 704 --width 1280 --num-frames 49 --num-inference-steps 8 \
  --output-dir /path/to/results/hy_nomas_session
```

The example uses `ARDiffusionOmniTickConsumer` with `step_execution=false` and
writes stable chunks, `session.mp4`, and `chunks.json`. Ticks share a `session_id`
and use consecutive `chunk_index` values starting at zero. The last tick sets
`close_session=true` to flush the decoder; `reset=true` starts a fresh timeline.
Close, cancellation, failure, and eviction release the session's KV and decoder state.

Typed camera controls use `track="camera"`, `schema="hy_nomas.camera.v1"`, and
`data={"camera_trajectory":"pan_left"}`. A turn can include frames withheld from
the previous turn; concatenate returned chunks in `metadata.hy_nomas.frame_start`
order. `metadata.ar_diffusion` preserves tick identity and applied event IDs.
Pass `--history-mode recent --history-latent-frames 36` to the session example
for cumulative continuation beyond the legacy limits. It writes the combined
MP4 incrementally instead of retaining all decoded frames in host memory.

## Supported features

| Feature | Hy-Nomas support |
| --- | --- |
| [Sequence parallelism](../../docs/user_guide/diffusion/parallelism/sequence_parallel.md) | Strict Ulysses with the existing worker group. Ring, AllGather-KV, and UAA are unsupported. |
| [CPU offload](../../docs/user_guide/diffusion/cpu_offload.md) | Module offload, including with SP; layerwise and distributed layerwise offload are unsupported. |
| [Other parallelism](../../docs/user_guide/diffusion/parallelism/overview.md) | TP, PP, DP, CFG, HSDP, text encoder sharding, and VAE parallelism are unsupported. |
| [Other acceleration](../../docs/user_guide/diffusion_features.md) | Quantization, LoRA, and Cache-DiT are unsupported. Use eager execution; full-transformer compilation is rejected and generic regional compilation leaves the cached reference forward unchanged. FlexAttention retains its own kernel compilation. |

## Verification

### Scope relative to the RFC

This first PR implements model integration and qualifies offline/full/recent
accuracy plus normal realtime output. Relative to
[RFC #14, §5](https://github.com/wtz2333/vllm-omni/issues/14):

| RFC item | Included here | Deferred / not claimed here |
| --- | --- | --- |
| A1 | Absolute progress, resident budgets, incremental-v1 noise, legacy full behavior and model-state guards | Complete production resource/lifecycle qualification |
| A2 | Pinned companion reference rolling decoder at `16a4c228` | Reference-side source is not duplicated in this Omni diff |
| A3 | Session-bound rolling decode, positions, empty stable metadata, normal final flush | Full cancellation/transport qualification |
| A4 | Full/recent transactional retention; offline AR and normal realtime/typed output; 18/36 precision and 120+ second delivery | Remaining A4 service matrix, 72 precision, complete high-water accounting and broad W6/full-vs-recent quality |
| B1–B9 | Basic SP/module offload remains supported; no new B-series optimization claimed | Media/kernel/attention/staging optimization, chunk residency/prefetch, decode parallelism, encoder sharding, regional compile/graphs and sparsity |
| C1–C8 | Full/recent through AR state interfaces | CPU archival, other history sampling, KV/weight quantization, undo and pause/resume |
| D1–D3 | Camera trajectory conditioning | Nomas-PE, official LoRA and browser camera UI |

The shared runtime retains its existing cancellation, timeout, quota and
backpressure mechanisms. Their complete production qualification, including
mid-compute cancellation and real blocked TCP sending, is outside this PR's
acceptance scope; this is not a declaration that the entire A4 milestone passed.

### Scope and reference accuracy

This integration covers offline generation, full/recent AR sessions, and normal
`/v1/realtime/video` generation with text/camera updates and final flush.
Implementation `209771901` passed 234 related CPU checks and four isolated
resident/offload continuation/tail-reference cases; the pinned PixelUMM revision
passed 34 CPU checks. All applicable local hooks passed.

The latest precision campaign used one H200, SP1, resident BF16, FlexAttention,
704×1280, 24 fps, seed 42, CFG=1, and flow shift=5:

| Comparison | Workload | Result |
| --- | --- | --- |
| Full vs upstream | Three 49-frame turns, separately at 4/8 steps | Exact complete latents and pre-codec uint8 video |
| Recent vs upstream | Same turns/steps, budget 36, equal actual noise | Exact at every turn |
| Offline AR vs realtime | Full/recent, 129 frames each; recent includes text/camera segments | Exact latents/pixels; encoded frames and PTS correct |
| Long recent vs upstream | Budgets 18/36, 48×61=2928 frames, 4 steps | All 48 turns exact; 122 seconds each |

Full uses the configured upstream noise timeline. Recent precision comparisons
supply upstream with the same actual `incremental_v1` generation and KV noise;
they do not establish same-seed agreement with the unchanged upstream RNG.
The reference uses its supported `seg_ar` route. Its additional 193-frame
`recent + chunk_ar` probe fails at an upstream cache write and is not counted
as a passed comparison. Reported recent precision covers budgets 18/36;
the interrupted 72-budget campaign is unqualified and was not retried.

Upstream native-RNG controls also show late overexposure, yellowing and distortion
in this forest/river scene. At equal noise, Omni adds no observed numerical
difference. The mechanism remains unresolved; one scene/seed does not establish
broad long-horizon quality. The 48-turn comparison is distinct from the
single-request WebSocket diagnostic below.

[Public data, calculations, complete viewing videos, images and reproduction commands](https://github.com/wtz2333/vllm-omni/blob/16278847a58dbde41b8d5e15017c7b092585340e/test-evidence/PR9338/section-a-20261029/README.md).

### Current long-stream diagnostic

Two H200s, SP2, 2881 frames, four steps, seed 42, resident BF16, eager/mp,
20 prompt/camera updates, nominal 16-latent blocks and a merged 17-latent tail:

| History budget | First playable frame (ms) | Client E2E (ms) | RTF | Peak allocated / reserved per GPU (GiB) |
| --- | --- | --- | --- | --- |
| 18 | 17175.3 | 538357.1 | 4.485 | 64.80 / 67.40 |
| 36 | 19042.5 | 554444.1 | 4.619 | 72.30 / 74.79 |

Each row is one diagnostic session without excluded matching warmup. Timing
excludes model startup and includes controls, encoding, writes, client decoding,
and worker observations; compilation effects are not excluded. RTF is E2E divided
by `2881/24=120.0417` seconds. These are not steady-state benchmark results.
The decoder retained at most 19 latents and one active mapping record.
The local timing/control harness is not included; the public scripts reproduce
the precision and entrypoint checks, rather than this diagnostic workload.

### Historical repeated performance

Earlier same-H200 offline 4/8-step data at `257856d3` is also retained: 704×1280,
121 frames, resident BF16, SP1, ten unprofiled measured requests per step count
(five before/five after diagnostics; two initial and one recovery warmup each).

| Steps | Samples | Median SDK E2E (ms) | Mean ± SD (ms) | Median drift |
| --- | --- | --- | --- | --- |
| 8 | 10 | 25653.8 | 25787.4 ± 555.7 | +0.055% |
| 4 | 10 | 19199.9 | 19202.1 ± 64.1 | -0.396% |

Both have peak allocated/reserved 59.09/63.37 GiB. The observed 1.336× step-count
ratio is not a quality-equivalence claim; cross-step outputs differ.

Two separate event diagnostics per step count give this decomposition:

| Stage | 8 steps (ms) | 4 steps (ms) |
| --- | --- | --- |
| Text encoding | 41.4 | 41.3 |
| DiT denoising | 12682.1 | 6356.5 |
| KV commit | 1583.5 | 1583.8 |
| VAE / PixelUMM decode, six windows | 8758.8 | 8750.2 |
| Other pipeline residual | 8.6 | 8.6 |
| Post-pipeline return residual | 2383.1 | 2461.7 |

Diagnostic E2E means are 25457.4/19202.2 ms, rather than the formal medians above.
Event stream elapsed can include host gaps; residuals are calculated differences,
not isolated IPC/PCIe measurements. Do not add nested stage wrappers twice.

The separate SP/module-offload SDK matrix belongs to the same earlier implementation:
704×1280, 121 frames, eight steps, two warmups and five measured requests per row.
It has not been repeated on the current bounded-history implementation.

| H200s / SP | Module CPU offload | Median SDK E2E (ms) | Peak allocated / reserved per GPU (GiB) |
| --- | --- | --- | --- |
| 1 / 1 | no | 30894.7 | 59.09 / 63.37 |
| 2 / 2 | no | 21123.7 | 53.05 / 57.06 |
| 1 / 1 | yes | 41248.6 | 30.56 / 32.07 |
| 2 / 2 | yes | 38466.6 | 23.80 / 25.99 |

SDK E2E excludes loading, encoding and writes. The SP1 recheck median was
25669.2 ms, a -16.91% change on the shared host. Raw samples, mean/SD/p95,
earlier ordinary/offload accuracy and benchmark commands remain in the
[historical evidence](https://github.com/wtz2333/vllm-omni/blob/a518affb28c71bf072bae2e4397dd5f227822e75/test-evidence/PR9338/hy-nomas-diffusers/README.md).

### Historical per-chunk stages and current chunk records

The earlier `b5c9b68f5` full-history stream used one H200/SP1, 253 frames, four
steps, 16-latent chunks and explicit capacity 64 latents / 1 prompt. Unprofiled
median first-playable frame was 7595.3 ms, E2E 58517.9 ms, RTF 5.551 (n=5).
The following stage values are means of two separate event diagnostics:

| Previous latents | Stable frames | DiT denoise (ms) | KV commit (ms) | PixelUMM prediction (ms) | Decode/convert inclusive (ms) |
| --- | --- | --- | --- | --- | --- |
| 0 | 24 | 2581.5 | 655.0 | 2919.8 | 3970.9 |
| 16 | 64 | 3860.9 | 979.8 | 4376.2 | 7148.2 |
| 32 | 64 | 4878.6 | 1229.1 | 4376.7 | 7425.4 |
| 48, final flush | 101 | 5916.8 | 1493.3 | 8749.3 | 13762.5 |

PixelUMM prediction is inside decode/convert; staging is inside DiT/commit.
Event elapsed is not pure GPU kernel time. The final flush changes decode work.
The explicit single-H200 capacity is not evidence that the default AR capacity fits.

Current 18/36 SP2 diagnostics publish all 90 chunk records (45 per session),
including positions, metadata intervals, decoded-frame timestamps and hashes.
No per-stage event timers were collected for those recent rollouts; historical
full-mode stage values do not replace that missing measurement.
[Raw stage samples, all chunk records, architecture calculations and RFC mapping](https://github.com/wtz2333/vllm-omni/blob/16278847a58dbde41b8d5e15017c7b092585340e/test-evidence/PR9338/section-a-20261029/architecture-stages-scope.md).

Production service hardening, compute-mid-generation cancellation and real TCP
blocked sending are outside this integration's acceptance scope. GPU CI and a
matched 240-frame SGLang benchmark have not run.

## References

- [Canonical reference implementation](https://github.com/Tencent-Hunyuan/Hy-Nomas/tree/911cea6458a89c2663513ff7099561ec190dd491)
- [Shared text-to-video example](../../examples/offline_inference/text_to_video/text_to_video.md)
- [Multi-turn session example](../../examples/offline_inference/text_to_video/hy_nomas_session.py)
- [Streaming client](../../examples/online_serving/streaming_video_generation/streaming_video_client.py)
