# Video output transport

This feature implements the API-output portion of the roadmap in
[RFC #6212](https://github.com/vllm-project/vllm-omni/issues/6212). Video output
policy is configured independently from the typed pre-D2H media contract in
[RFC #6541](https://github.com/vllm-project/vllm-omni/issues/6541). The model
runner owns device-side preparation; this page covers the API-side container,
encoder, and delivery sink selected after the engine finalizes the video.

The defaults preserve the existing synchronous API contract: MP4 bytes are
returned directly and device-side postprocessing remains disabled.

## Configuration

Pass `video_output_transport` as a Python mapping:

```python
from vllm_omni import Omni

engine = Omni(
    model="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    video_output_transport={
        "enable_device_postprocess": True,
        "transport_mode": "bytes",
        "output_format": "mp4",
    },
)
```

The server CLI accepts the same object as JSON:

```bash
vllm serve Wan-AI/Wan2.2-TI2V-5B-Diffusers --omni \
  --video-output-transport \
  '{"enable_device_postprocess":true,"transport_mode":"bytes","output_format":"mp4"}'
```

| Field | Default | Effect |
| --- | --- | --- |
| `enable_device_postprocess` | `false` | Enables the runner-owned pre-D2H preparation described in the [device-side postprocessing guide](device_side_video_postprocess.md). |
| `transport_mode` | `"bytes"` | Selects `bytes`, `base64`, `url`, or `shared_memory`. |
| `shared_memory_ttl_seconds` | `300` | Bounds an unclaimed shared-memory response. |
| `output_format` | `"mp4"` | Selects the MP4 or WebM container. |
| `video_codec` | `null` | Uses the container default when unset: H.264 for MP4 and VP9 for WebM. |
| `video_codec_options` | `{}` | Uses codec-specific fast defaults when empty. |

`video_codec`, `video_codec_options`, and `output_format` may also be overridden
for one request through `extra_params`. These transport-only keys are removed
before model sampling parameters are built. `transport_mode` and
`shared_memory_ttl_seconds` are deployment-only; request-level values are
rejected before generation.

## HTTP response modes

`POST /v1/videos/sync` supports all four modes:

| Mode | Response | Encoding |
| --- | --- | --- |
| `bytes` | `video/mp4` or `video/webm` body | Encoded once. |
| `base64` | `VideoGenerationResponse` JSON with `b64_json` | Encoded once, then base64 encoded. |
| `url` | `VideoGenerationResponse` JSON with `url` | Encoded once and stored under an expiring key. |
| `shared_memory` | `VideoGenerationResponse` JSON with `shm_handle` | No MP4/WebM encoding; raw uint8 RGB frames. |

The asynchronous `POST /v1/videos` job API accepts only `bytes`. Immediate
response modes are rejected before generation because that endpoint initially
returns job metadata rather than the generated artifact.

## Containers and encoders

Container defaults are resolved as one policy so video, audio, and MIME types
cannot drift:

| Container | Video | Audio | MIME |
| --- | --- | --- | --- |
| MP4 | H.264 | AAC | `video/mp4` |
| WebM | VP9 | Opus | `video/webm` |

An explicit codec must be compatible with the selected container. A compatible
encoder that cannot be opened on the current host falls back to the container's
software default. Options from the unavailable encoder are discarded because
FFmpeg does not accept options from another encoder family.

Hardware encoding is optional. In particular, Hopper data-center GPUs do not
provide an NVENC block; requesting `h264_nvenc` there exercises the verified
software fallback rather than hardware acceleration.

The fragmented WebSocket stream remains MP4-only and resolves low-latency codec
options independently of the artifact container.

## URL artifacts

URL mode requires an expiration policy before generation. Enable the existing
file TTL manager with:

```bash
export VLLM_OMNI_SERVER_STORAGE__FILE_TTL=3600
```

Without a static server, responses point to the built-in route:

```text
/v1/videos/artifacts/{storage_key}
```

The route streams local files and derives MP4/WebM MIME from the extension. To
publish the same storage directory through a static server or CDN, configure:

```bash
export VLLM_OMNI_SERVER_STORAGE__PUBLIC_BASE_URL="https://cdn.example.com/videos"
```

Only the local file backend is supported in-tree; no S3 or OSS client is
included.

## Same-host shared memory

Shared-memory mode is accepted only when the API server binds to `127.0.0.1`,
`::1`, or `localhost`:

```bash
vllm serve <model> --omni --host 127.0.0.1 \
  --video-output-transport '{"transport_mode":"shared_memory"}'
```

The response contains a versioned `VideoSharedMemoryHandle` for one contiguous
`uint8` array shaped `(frames, height, width, 3)`. Consume it with the context
manager:

```python
from vllm_omni.entrypoints.openai.video_output_shm import borrowed_video_frames

handle = response.json()["data"][0]["shm_handle"]
with borrowed_video_frames(handle) as frames:
    consume(frames)
```

Leaving the block unlinks the segment. If no consumer claims it, the server's
lease sweeper unlinks it after `shared_memory_ttl_seconds`; process shutdown also
cleans outstanding leases. Audio-bearing video output is rejected in this mode
rather than silently dropping audio.

!!! warning

    The borrowed array aliases shared memory and is invalid after the context
    exits. Copy inside the block if data must outlive it. The view is writable,
    so mutations are visible to every holder.

This sink is consumer-side zero-copy: publishing the frames still copies them
once into shared memory, while the consumer avoids MP4 decoding and a private
payload copy. It is separate from the worker-to-engine IPC managed by the typed
media runtime.

Run the checked-in cross-process benchmark with:

```bash
python benchmarks/diffusion/bench_video_output_sinks.py \
  --frames 48 --height 512 --width 768 --rounds 3
```

For a 54.0 MiB uint8 video on the development host, three fresh consumer
processes per sink measured:

| Sink | JSON boundary | Consumer RSS mean [range] | Frame hash |
| --- | ---: | ---: | --- |
| `base64` | 14.70 MiB | 101.3 [101.3, 101.4] MiB | Lossy after MP4 decode |
| `shared_memory` | Less than 1 KiB | 52.3 [51.8, 53.4] MiB | Exact |

The benchmark starts a fresh consumer for each measurement, takes its RSS
baseline after imports, reads every frame byte, and verifies that shared-memory
segments are removed. These are per-payload memory measurements, not throughput
results.
