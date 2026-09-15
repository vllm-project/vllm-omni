# MiniMax-H3 on RTX 4090

[Model guide](MiniMax-H3.md) · [Deployment choices](MiniMax-H3.md#choose-a-deployment) · [HTTP API](MiniMax-H3.md#http-api-examples)

The single-GPU configuration streams BF16 DiT and text-encoder blocks from
host memory and loads the VAEs only for their encode/decode phases. VAE tiling
reduces decode activation memory. A separately validated four-GPU deployment
is provided below.

## Capacity requirements

| Resource | One RTX 4090 | Four RTX 4090s |
| --- | ---: | ---: |
| GPU memory | 24 GiB | 24 GiB per GPU |
| Checkpoint storage | 135 GiB per partition | 135 GiB per partition |
| Available system RAM | Minimum not established | 200 GiB minimum |
| Recommended system RAM | See the host-memory requirements below | 384 GiB |

`FL2VA` and `Ref2VA` are separate 135 GiB checkpoint partitions. Start one
server at a time. See the shared
[host-memory requirements](MiniMax-H3-CUDA.md#single-gpu-low-memory-serving):
GPU offload does not establish a 32 GiB system-RAM configuration.

## Single GPU

Complete the [CUDA installation](MiniMax-H3-CUDA.md#installation), then start
one worker with no retained DiT layers:

```bash
vllm serve MiniMaxAI/MiniMax-H3 --omni --trust-remote-code \
  --task-type fl2va \
  --diffusion-offload-config \
  '{"mode":"layer","components":["dit","text_encoder","vae"],"layer_options":{"dit":{"weight_transfer":"rank-local"}}}' \
  --vae-use-tiling --enforce-eager \
  --diffusion-attention-backend CUDNN_ATTN
```

Start with the shared [480P T2VA request](MiniMax-H3-CUDA.md#single-gpu-low-memory-serving).
This single-GPU configuration has not completed target-hardware end-to-end
validation; neither a 768P capacity guarantee nor a latency result is available.

## Four RTX 4090s: 1024x576, 5 seconds

Keep TP2 and raise Ulysses sequence parallel to 2 so the world size is
`TP × USP = 4`. Text-encoder TP and VAE patch parallel follow the GPU count.
Keep 12 resident DiT layers. TP shards the DiT weights; Ulysses shards the
sequence without further reducing resident weight memory.

```bash
vllm serve /path/to/MiniMax-H3/FL2VA \
  --omni --trust-remote-code \
  --num-gpus 4 --tensor-parallel-size 2 --text-encoder-tp-size 4 \
  --usp 2 --ring 1 --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile --vae-use-tiling \
  --diffusion-offload-config \
  '{"mode":"layer","components":["dit","text_encoder","vae"],"layer_options":{"dit":{"weight_transfer":"rank-local","resident_layers":12}}}' \
  --enforce-eager \
  --diffusion-attention-backend CUDNN_ATTN
```

For Ref2VA, stop the FL2VA server and restart the same command with
`/path/to/MiniMax-H3/Ref2VA`. Pin four dedicated GPUs; if the launcher exposes
fewer than four devices, startup fails with
`Stage 0 requires 4 device(s) based on parallel_config`.

## Four-GPU request examples

`t2va` requires an explicit `aspect_ratio`. Without it the request fails with
`t2va requires an explicit aspect_ratio`, even when `width` and `height` are set.

Text to video and audio, against the `FL2VA` server:

```bash
curl -sS -D headers_t2va.txt -o out_t2va.mp4 --max-time 14400 \
  -X POST "http://127.0.0.1:8000/v1/videos/sync" \
  -F 'prompt=傍晚的小厨房，夕阳余晖从窗边洒进来，旧木桌上放着洗到一半的马克杯和起雾的玻璃瓶，悬挂的抹布轻轻晃动。画面带有手持拍摄的轻微晃动和逆光曝光波动，生活感十足。环境音是安静的厨房底噪。' \
  -F 'aspect_ratio=16:9' \
  -F 'width=1024' -F 'height=576' -F 'fps=24' \
  -F 'num_inference_steps=60' -F 'flow_shift=12' -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":5,"audio_flow_shift":3.0}'
```

Reference to video and audio, against the `Ref2VA` server:

```bash
curl -sS -D headers_ref2va.txt -o out_ref2va.mp4 --max-time 14400 \
  -X POST "http://127.0.0.1:8000/v1/videos/sync" \
  -F 'prompt=傍晚的小厨房，夕阳余晖从窗边洒进来，旧木桌上放着洗到一半的马克杯和起雾的玻璃瓶，悬挂的抹布轻轻晃动。画面带有手持拍摄的轻微晃动和逆光曝光波动，生活感十足。环境音是安静的厨房底噪。' \
  -F 'input_reference=@/path/to/reference.jpg;type=image/jpeg' \
  -F 'audio_reference=</path/to/audio_reference.json' \
  -F 'width=1024' -F 'height=576' -F 'fps=24' \
  -F 'num_inference_steps=60' -F 'flow_shift=12' -F 'seed=1101' \
  -F 'extra_params={"task":"ref2va","duration":5,"audio_flow_shift":3.0}'
```

`audio_reference` is a JSON object, which is why the example loads it from a file
with curl's `<` syntax. Its `audio_url` must be an `http(s)` URL or a data URL;
a bare filesystem path is rejected with `Invalid audio_reference.audio_url`.

```json
{"audio_url": "https://example.com/reference.wav"}
```

Reference images are validated before inference: the short edge must be at least
256 pixels, the long edge at most 5760, and the aspect ratio must fall between
0.4 and 2.5.

The response headers carry the measurements used in the validation table below:

```bash
cat headers_t2va.txt
ffmpeg -v error -i out_t2va.mp4 -f null - && echo DECODE_OK
```

## Target-hardware validation

Measured on four RTX 4090s (24,564 MiB each, driver 580.126.09) with the
topology above and the legacy rank-local DLO flags retaining 12 DiT layers,
at vLLM-Omni `0.26.1.dev55+g81b48e83e`, vLLM `0.26.0`, and PyTorch
`2.11.0+cu130`. The structured offload configuration above has not been rerun
on this hardware. GPUs used by each server were dedicated.
`Server elapsed` is the `x-inference-time-s` response header, measured within
the synchronous server handler; it excludes client upload and response download.
`Rank-0 peak` is the reported CUDA reserved high-water mark from
`x-peak-memory-mb`, not an externally sampled maximum across all devices.
See the [response-header definitions](../../docs/serving/videos_api.md#synchronous-response).

| GPUs | Topology | Task | Shape | Frames | Steps | Server elapsed | Rank-0 peak | Output validation |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 4 | TP2 × USP2 | T2VA (`FL2VA`) | 1024x576 | 124 at 24 FPS | 60 | 4 min 29 s | 15.2 GiB | H.264 video + 32 kHz stereo AAC; full `ffmpeg` decode passed |
| 4 | TP2 × USP2 | Ref2VA | 1024x576 | 124 at 24 FPS | 60 | 9 min 5 s | 16.1 GiB | H.264 video + 32 kHz stereo AAC; successful `/v1/videos/sync` 200 |

All rows use `seed=1101`, `flow_shift=12`, and `audio_flow_shift=3.0`.

Four-GPU T2VA repeats: 273.8 s and 269.5 s, peaking at 15,560 MiB and
15,612 MiB. Four-GPU Ref2VA is a single successful end-to-end run (545.2 s,
16,448 MiB). Follow-up Ref2VA requests on this build completed diffusion but
then hit the engine's async output wait during post-compute D2H, so they are
not reported as repeats. That wait was a hardcoded 30 s when these numbers were
taken; it is now `VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT` (default 600 s), so a rerun on
current main should not lose these repeats to the bound.

These are single-request validation runs, not concurrent throughput benchmarks.
