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
Device interconnect, peak memory, and host memory were not measured in these checks.
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

### Section A implementation

Implementation `209771901` passed 234 related CPU checks, all applicable local
hooks, and four isolated 704×1280 reference cases. Four/eight steps cover three
49-frame turns and a 129-frame request; resident/offload outputs preserve exact
complete latents and uint8 pixels under the reference's actual chunk sequence.
The pinned PixelUMM revision passed 34 CPU checks against the legacy decoder.

Post-correction SP2 runs for history budgets 18 and 36 delivered 2,881 frames
with 20 prompt/camera controls. The remaining sustained matrix and measurement
summaries are still being collected. Long-session perceptual quality requires
separate qualification; duration and resource checks do not establish it.

### Baseline integration evidence

For a quick check, change the offline command to one prompt, 256×256, nine frames,
and four steps. Verify that the output has nine frames at 24 fps. This checks
execution; it does not establish video quality or performance.

CPU coverage checks modular discovery, sharded component loading, parameter-name
mapping, component initialization, and AR state handling. All 18 GPU cases passed
with the converted checkpoint:

- Six reference comparisons cover text, multiple prompt segments, and camera
  control in resident and CPU-offload modes. Complete decoded videos match the
  independent original checkpoint exactly at 384×640, 49 frames, eight steps,
  BF16, and seed 42 (`rtol=0`, `atol=0`).
- Two continuation comparisons cover resident and CPU-offload modes. Per-turn
  latents and the complete 98-frame output match the reference exactly across
  two 49-frame turns with a prompt change and camera control.
- Ten offline and video API cases cover single-device and Ulysses degree-two
  generation, including CPU offload, segment/camera inputs, and duration requests.

On two H200s at 1280×704 and 24 fps, the session example produced 147 frames
across three turns, emitted as stable chunks of 12, 49, and 86 frames. Closing
the session flushed all withheld frames. Realtime acceptance completed a
477-frame request with prompt/camera updates, stopped a separate request after
23 decoded frames, and completed another 477-frame request on the same service.
Update acknowledgements, stop status, transport order, and decoded frame counts
were checked. These checks do not establish latency or throughput performance.

## References

- [Canonical reference implementation](https://github.com/Tencent-Hunyuan/Hy-Nomas/tree/911cea6458a89c2663513ff7099561ec190dd491)
- [Shared text-to-video example](../../examples/offline_inference/text_to_video/text_to_video.md)
- [Multi-turn session example](../../examples/offline_inference/text_to_video/hy_nomas_session.py)
- [Streaming client](../../examples/online_serving/streaming_video_generation/streaming_video_client.py)
