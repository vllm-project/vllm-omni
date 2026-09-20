# Videos API

vLLM-Omni provides an OpenAI-compatible video generation API for diffusion
video models. The API supports asynchronous video jobs through `/v1/videos` and
a synchronous benchmark-oriented endpoint through `/v1/videos/sync`.

Each server instance runs a single model specified at startup with
`vllm serve <model> --omni`.

## Quick Start

### Start the Server

```bash
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --omni --port 8091
```

### Create a Video Job

```bash
create_response=$(curl -s http://localhost:8091/v1/videos \
  -F "prompt=A cinematic tracking shot of a mountain lake at sunrise" \
  -F "width=1280" \
  -F "height=720" \
  -F "num_frames=80" \
  -F "fps=16" \
  -F "num_inference_steps=40")

video_id=$(echo "${create_response}" | jq -r '.id')
```

### Poll and Download

```bash
curl -s "http://localhost:8091/v1/videos/${video_id}" | jq .
curl -L "http://localhost:8091/v1/videos/${video_id}/content" -o output.mp4
```

## API Reference

### Endpoints

| Endpoint | Method | Description |
| ---------- | -------- | ------------- |
| `/v1/videos` | `POST` | Create an asynchronous video generation job |
| `/v1/videos/sync` | `POST` | Generate a video synchronously and return raw video bytes |
| `/v1/videos/{video_id}` | `GET` | Retrieve job status and metadata |
| `/v1/videos` | `GET` | List stored video jobs |
| `/v1/videos/{video_id}/content` | `GET` | Download generated video content |
| `/v1/videos/{video_id}` | `DELETE` | Delete a video job and stored output |

### Request Parameters

`POST /v1/videos` and `POST /v1/videos/sync` accept `multipart/form-data`.

#### OpenAI-style fields

| Parameter | Type | Default | Description |
| ----------- | ------ | --------- | ------------- |
| `prompt` | string | **required** | Text prompt for video generation |
| `model` | string | server's model | Optional model name |
| `seconds` | string | null | Requested clip duration in seconds |
| `size` | string | null | Requested output size in `WIDTHxHEIGHT` format |
| `user` | string | null | Optional user identifier |

#### vLLM-Omni extension fields

| Parameter | Type | Default | Description |
| ----------- | ------ | --------- | ------------- |
| `input_reference` | file | null | Uploaded reference image or video for image-to-video/video-to-video requests |
| `timeline_guides` | string | null | H3-only ordered JSON manifest of pixel-frame guide placements; see [H3 Timeline Guides](#h3-timeline-guides) |
| `guide_files` | repeated file | null | Separate uploads addressed by zero-based `upload_index` in `timeline_guides` |
| `control_reference` | file | null | Optional uploaded image/video control, up to 512 MiB, for models that declare control-upload support |
| `control_type` | string | null | Model control name associated with `control_reference`; currently Cosmos3 supports `edge`, `blur`, `depth`, `seg`, and `wsm` |
| `image_reference` | string | null | JSON-encoded reference image payload; do not combine with `input_reference` or `video_reference` |
| `video_reference` | string | null | JSON-encoded reference video payload; do not combine with `input_reference` or `image_reference` |
| `audio_reference` | string | null | JSON-encoded audio reference for speech-to-video: `{"audio_url": "..."}` — supports HTTP(s) URLs or base64 data URLs |
| `source_video` | file | null | MiniMax H3 latent-edit source video (`.mp4` or `.mov`, up to 512 MiB) |
| `source_audio` | file | null | Optional MiniMax H3 latent-edit source audio (`.wav` or `.mp3`, up to 512 MiB) |
| `video_noise_mask` | file | null | UTF-8 JSON MiniMax H3 video mask; `0` preserves and `1` regenerates a token |
| `audio_noise_mask` | file | null | UTF-8 JSON MiniMax H3 audio mask; `0` preserves and `1` regenerates a token |
| `width` | integer | model default | Output video width |
| `height` | integer | model default | Output video height |
| `num_frames` | integer | 1 | Number of generated frames |
| `fps` | integer | model default | Output frames per second |
| `num_inference_steps` | integer | model default | Number of diffusion steps |
| `guidance_scale` | number | null | CFG guidance scale for the low-noise stage |
| `guidance_scale_2` | number | null | CFG guidance scale for the high-noise stage |
| `boundary_ratio` | number | null | Boundary split ratio for multi-stage denoising |
| `flow_shift` | number | null | Scheduler flow-shift value |
| `true_cfg_scale` | number | null | True CFG scale when supported by the model |
| `seed` | integer | null | Random seed for reproducibility |
| `generate_sound` | boolean | false | Request model-generated audio for video models that support sound generation |
| `sound_duration` | number | null | Duration in seconds for generated audio; defaults to generated video duration |
| `negative_prompt` | string | null | Text describing what to avoid in the generated video |
| `enable_frame_interpolation` | boolean | null | Enable post-generation frame interpolation |
| `frame_interpolation_exp` | integer | null | Interpolation exponent; `1=2x`, `2=4x`, and so on |
| `frame_interpolation_scale` | number | null | RIFE inference scale |
| `frame_interpolation_model_path` | string | null | Local path or Hugging Face repo for the interpolation model |
| `lora` | string | null | JSON-encoded LoRA configuration object |
| `extra_params` | string | null | JSON-encoded object for additional model-specific parameters |

### Create Response

`POST /v1/videos` returns a job record:

```json
{
  "id": "video-123",
  "status": "queued",
  "created_at": 1701234567
}
```

The final content is available from `/v1/videos/{video_id}/content` after the
job status becomes `completed`.

`queued` means the request is still waiting for diffusion scheduler admission.
The status changes to `in_progress` when the scheduler first selects the
request for execution.

`DELETE /v1/videos/{video_id}` issues a bounded engine abort
(`VLLM_OMNI_ABORT_TIMEOUT`, default 2s), then cancels the frontend
task. Cancellation cleanup is also bounded and best-effort: it confirms
the abort was queued, and the current request batch may still drain.
The job is then re-read so a completed save is not orphaned.
Guided requests are the exception; see
[Cancellation and Cleanup](#cancellation-and-cleanup).

### Synchronous Response

`POST /v1/videos/sync` blocks until generation finishes and returns raw video
bytes. It is useful for benchmarks and simple scripts that do not need job
storage or polling.

## Examples

### Image-to-Video

```bash
curl -s http://localhost:8091/v1/videos \
  -F "prompt=animate this image with subtle camera movement" \
  -F "input_reference=@input.png" \
  -F "width=1280" \
  -F "height=720" \
  -F "num_frames=80" \
  -F "fps=16"
```

### Video-to-Video

For models that support video conditioning, upload the reference video with
`input_reference`:

```bash
curl -s http://localhost:8091/v1/videos \
  -F "prompt=continue this motion with consistent subjects and lighting" \
  -F "input_reference=@input.mp4;type=video/mp4" \
  -F "width=1280" \
  -F "height=720" \
  -F "num_frames=80" \
  -F "fps=16"
```

You can also pass a JSON-safe video URL or `data:video/...;base64,...` payload
through `video_reference`. Do not send `video_reference` together with
`input_reference` or `image_reference`.

```bash
curl -s http://localhost:8091/v1/videos \
  -F "prompt=continue this motion with consistent subjects and lighting" \
  -F 'video_reference={"video_url":"https://example.com/input.mp4"}' \
  -F "width=1280" \
  -F "height=720" \
  -F "num_frames=80" \
  -F "fps=16"
```

JSON references currently support `image_url`/`video_url`; `file_id` references
are not implemented yet. Models may expose additional V2V controls through
`extra_params`. For example, Cosmos3 supports
`condition_frame_indexes_vision` and `condition_video_keep` to select which
decoded reference frames are used as clean conditioning. Cosmos3 transfer mode
also accepts `edge`, `blur`, `depth`, `seg`, or `wsm` control hints. Each hint
may specify its own `control_path` and `control_weight`. Request-level transfer
options include
`control_guidance`, `control_guidance_interval`,
`emphasize_control_in_prompt`, `num_video_frames_per_chunk`,
`num_conditional_frames`, `show_control_condition`, and `show_input`. Transfer
uses its transfer-specific system prompt and, by default, appends a
control-adherence directive to the positive prompt. It adds duration/FPS and
resolution metadata to both CFG branches but does not add a negative prompt
automatically. The Cosmos3 recipe includes an optional reference negative
prompt and shows how to pass it. Set `emphasize_control_in_prompt`,
`use_duration_template`, or `use_resolution_template` to `false` to disable the
corresponding addition. `negative_metadata_mode` accepts `same`, `inverse`, or
`none` and defaults to `same` for transfer. See the Cosmos3 recipe for complete
examples.

A client that cannot place the control on the server filesystem can upload one
control with `control_reference` and identify it with `control_type`. The API
streams the upload to request-scoped storage, supplies its path to the model,
and removes it after synchronous or asynchronous generation completes. Other
options for the selected control can remain in `extra_params`; do not also set
`control` or `control_path` there. Uploads larger than 512 MiB are rejected.

```bash
curl -s http://localhost:8091/v1/videos/sync \
  -F "prompt=Preserve the scene while following the world-state control" \
  -F "input_reference=@input.mp4;type=video/mp4" \
  -F "control_reference=@wsm.mp4;type=video/mp4" \
  -F "control_type=wsm" \
  -F 'extra_params={"wsm":{"control_weight":1.0}}' \
  -o output.mp4
```

HTTP redirects for `image_reference.image_url` follow vLLM's
`VLLM_MEDIA_URL_ALLOW_REDIRECTS` setting. Before starting the server, set it to
`1` (the default) to allow redirects or `0` to reject them. A redirect target
can differ from the original host, and vLLM-Omni does not yet fully implement
upstream vLLM's media URL allowlist protection. Deployments should therefore
treat remote media URLs as untrusted and choose this setting as part of their
URL access policy.

### MiniMax H3 Latent-Mask Editing

MiniMax H3 accepts request-scoped source media and video/audio noise masks.
At least one mask is required. A nontrivial `video_noise_mask` requires
`source_video`, while a nontrivial `audio_noise_mask` requires either
`source_audio` or a `source_video` with an audio stream. Mask values are in
`[0, 1]`: `0` preserves the source, `1` regenerates it, and fractional values
blend the two behaviors. Exact all-one masks are no-ops and do not require a
source. Source uploads without a mask are rejected. Masks may be a JSON scalar
or arrays matching the H3 latent/token grid; pixel-resolution masks must be
resized or pooled by the client before upload.

For an aligned output of `F` frames at `W x H`, the video latent grid is
`[Tv, H/16, W/16]`, where `Tv = 2 + 5 * ((F - 5) / 17)`. The video mask may be
a scalar, a flat token vector, `[Tv, H/32, W/32]`, or the full latent grid. For
the model input, timestep, and velocity, a full-grid mask is max-pooled over
each 2x2 spatial token and fractional values are rounded upward to 1/256
levels. The final x0 restore uses the original, unquantized mask, so full-grid
masks retain cell-level preservation inside a model token. The audio length is
`Ta = round(F * 40 / 24)`, and its mask may be a scalar, `[Ta]`, `[2, Ta]`, or
a flat `2 * Ta` vector. If source audio is short, its missing latent tail is
forced to mask value `1` so H3 generates that portion. Each mask must be sent
as a UTF-8 JSON file part and is limited to 8 MiB. A file may contain either a
JSON scalar or an array.

```bash
curl -s http://localhost:8091/v1/videos/sync \
  -F "prompt=Partially restyle the complete clip and soundtrack" \
  -F 'extra_params={"task":"t2va","duration":4.0,"aspect_ratio":"16:9"}' \
  -F "source_video=@source.mp4;type=video/mp4" \
  -F "source_audio=@source.wav;type=audio/wav" \
  -F "video_noise_mask=@video-mask.json;type=application/json" \
  -F "audio_noise_mask=@audio-mask.json;type=application/json" \
  -o edited.mp4
```

For example, each mask file in the request above may contain the JSON scalar
`0.5`.

The server streams source files to temporary request-scoped storage and removes
them after synchronous or asynchronous generation finishes. Models that do not
declare latent-mask editing support reject these fields.

### Speech-to-Video

For models that support audio-driven generation (e.g., Wan2.2-S2V), pass both
an image reference and an audio reference. The `audio_reference` field accepts a
JSON string with `audio_url` pointing to an HTTP(s) URL or base64 data URL.

```bash
curl -s http://localhost:8091/v1/videos \
  -F "prompt=A person singing" \
  -F 'image_reference={"image_url": "https://example.com/face.png"}' \
  -F 'audio_reference={"audio_url": "https://example.com/speech.mp3"}' \
  -F "width=832" \
  -F "height=480" \
  -F "num_inference_steps=40" \
  -F "guidance_scale=4.5" \
  -F "fps=16"
```

### H3 Timeline Guides

For implementation boundaries, packing invariants, and request ownership, see
the [H3 timeline guide implementation](../design/feature/minimax_h3_timeline_guides.md).

MiniMax H3 accepts ordered image, clip, audio-only, and visual-plus-audio guides
on both video endpoints. Guided HTTP requests require Python 3.11 or newer for
reliable cancellation-state tracking; older runtimes reject them before admission.
No-guide requests retain their existing runtime behavior.
These are timeline conditions, not ordinary references:
they never enter Qwen reference presentation or change `<Picture N>` labels.
Use base H3 weights, dense attention, and cache-free execution. Send
`quality=lossless`; `quality=high` enables Cache-DiT and is rejected with guides,
even when caching was disabled at startup. Active/fused LoRA (including Turbo),
FastH3, few-step distilled schedules, sparse attention, and cache acceleration are not
supported for guided requests. No-guide requests retain their existing behavior.

```bash
curl --fail-with-body http://localhost:8091/v1/videos/sync \
  -F 'prompt=A bird flies across the lake and lands beside the reeds.' \
  -F 'width=1344' -F 'height=768' -F 'num_frames=124' -F 'fps=24' \
  -F 'num_inference_steps=50' -F 'seed=1101' -F 'quality=lossless' \
  -F 'extra_params={"task":"t2va","aspect_ratio":"16:9","audio_flow_shift":3.0}' \
  -F 'timeline_guides=[{"frame_index":36,"image":{"upload_index":0}},{"frame_index":-22,"video":{"upload_index":1},"audio":{"upload_index":2}}]' \
  -F 'guide_files=@middle.png;type=image/png' \
  -F 'guide_files=@tail.mp4;type=video/mp4' \
  -F 'guide_files=@tail.flac;type=audio/flac' \
  -o guided.mp4
```

Use `/v1/videos` instead to create a job, then poll and download normally.
`tail.mp4` in this example must normalize to no more than 22 frames. For an
audio-only guide, use a manifest such as
`[{"frame_index":0,"audio":{"upload_index":0}}]` and upload only its audio file.
Guide audio supports WAV, MP3, and FLAC, including clips shorter than two seconds.
A guide video's soundtrack is **not** extracted automatically: upload explicit
audio when it should condition the output.

Manifest indices must be JSON integers, not booleans, floats, or strings. Each
entry requires `frame_index` and at least one of `image`, `video`, or `audio`;
`image` and `video` cannot coexist in one entry. Source objects contain only
`upload_index`. Unknown fields, missing/out-of-range indices, unreferenced
uploads, and mismatched media are rejected. Reusing an upload is allowed;
each occurrence is conditioned in insertion order, including overlaps and
nonchronological entries. Empty or omitted guides with no files use the legacy
path. Do not put the manifest in `extra_params`, send server paths or URLs as
guides, or supply the reserved `_minimax_h3_timeline_guides` field yourself.

#### Placement and Normalization

- H3 outputs 24 FPS. Let `N` be the actual output frame count after upward
  alignment to `17k+5`; negative index `i` resolves to `N+i`.
- Starts are exact pixel-frame indices, not rounded to VAE latent boundaries.
- A source with 1-4 visual frames becomes its first frame. Otherwise it becomes
  `G=5+17*floor((M-5)/17)` frames. Empty visuals are rejected.
- Clips are decoded using elapsed timestamps at 24 FPS before normalization.
  The normalized clip must satisfy `0 <= start` and `start+G <= N`; it is not
  trimmed again to fit. For a 22-frame clip, `-22` fits at the end but `-1` does not.
- Guide visuals are center-cropped to the output canvas. Ordinary FL2VA
  first/last images retain their existing stretching behavior.
- Explicit audio starts at the same frame and may outlast the visual guide.
  After encoding, each channel is cropped to at most
  `floor(round(N*40/24)-(5/3)*start)` latent positions. Audio-only guides use a
  one-frame start check and must leave at least one audio latent position.
- Guides condition generation; pixel-identical or sample-identical reconstruction
  is not promised.

Ordinary inputs still determine routing: text plus guides uses `t2va`/FL2VA
weights, first/last inputs use `fl2va`, and ordinary visual references use
`ref2va`. Guides coexist with `input_references` and typed ordinary references.
A `t2va` request still needs an explicit ratio in the multipart `aspect_ratio`
field or `extra_params.aspect_ratio`, even when `width` and `height` are supplied.
H3's existing 4-15 second output-duration
restriction remains in force; short guide inputs do not permit shorter outputs.
A Ref2VA-only server still requires an ordinary visual reference; guides are
not implicitly converted into references. Existing step-mode, fanout, batching,
and offload restrictions continue to apply.

#### Admission Limits

The server owns the `od_config.model_config["minimax_h3_timeline_guides"]`
configuration block, configurable through the existing deploy YAML/stage
overrides. Requests cannot override it. Every value must be finite and positive.

For example, merge this block into the H3 diffusion stage of a deploy YAML
(not the text-encoder stage):

```yaml
model_config:
  minimax_h3_timeline_guides:
    max_entries: 4
    max_outstanding_requests: 2
```

| Configuration key | Default |
| --- | ---: |
| `max_entries` / `max_unique_files` | 8 / 16 |
| `max_image_bytes` | 31,457,280 (30 MiB) |
| `max_video_bytes` | 52,428,800 (50 MiB) |
| `max_audio_bytes` | 15,728,640 (15 MiB) |
| `max_total_upload_bytes` | 134,217,728 (128 MiB) |
| `max_guide_rows` / `max_packed_rows` | 65,536 / 262,144 |
| `max_source_pixels` | 16,777,216 |
| `max_decoded_visual_pixels` | 268,435,456 |
| `max_decoded_audio_samples` | 8,388,608 stereo scalar samples at 32 kHz |
| `subprocess_timeout_seconds` | 60 per probe/decode operation |
| `max_outstanding_requests` | 4 per handler, including abandoned running work |

Reused files count once for upload bytes, but every occurrence counts toward
decode and conditioning work. Visual work counts source/canvas pixel-frames,
including frames discarded during normalization. For clips, work is
`max(source_frames, resampled_24_fps_frames) * max(source_pixels, canvas_pixels)`;
high-FPS source frames count even when resampling discards them. Packed rows include text,
ordinary references, first/last anchors, guides, targets, and padding. Oversized
sources must be trimmed or resized by the client. These are admission-policy
defaults, **not measured safe GPU capacity**. Multipart parsing may spool data
before admission; configure reverse-proxy ingress limits separately.

#### Cancellation and Cleanup

For guided requests, a sync timeout, client cancellation, or job DELETE abandons
the response but does **not** interrupt already-submitted generation. GPU work
may continue. The server retains all file-backed inputs, including ordinary
references, and the outstanding-job reservation until the inner task completes
safely, then discards the result and removes temporary files. Unsubmitted work
can be cleaned up immediately. DELETE cannot be undone by a late completion or
output-store write. Guided DELETE deliberately skips the bounded engine abort and
frontend task cancellation used for no-guide jobs: an abort acknowledgment does
not establish that a worker stopped reading the file-backed guide inputs.
Graceful shutdown drains submitted guided work before engine
teardown; this can take as long as generation. No immediate GPU abort or
cross-host file transfer is provided. No-guide cancellation is unchanged.
Guided jobs report `queued` until the engine reports inference start, matching
no-guide jobs.

If an engine failure or unexpected inner-task cancellation prevents confirmation
that workers finished reading, cleanup is deliberately conservative: inputs and
admission slots remain retained, including after the graceful drain. The current
engine's shutdown return is not a worker-termination acknowledgment. Such abnormal
retention has no automatic reclamation; operators must independently confirm
reader termination before removing retained files or restarting admission.

Schema/association errors are rejected before queueing where possible.
Output-dependent validation failures use normal synchronous client errors or
structured asynchronous failed-job errors.

### Synchronous Generation

```bash
curl -X POST http://localhost:8091/v1/videos/sync \
  -F "prompt=A small robot walking through a neon city" \
  -F "width=854" \
  -F "height=480" \
  -F "num_frames=80" \
  -F "fps=16" \
  -o output.mp4
```

## Output Encoding

These `extra_params` control how the server turns decoded frames into MP4 bytes.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `preencode_mp4` | boolean | false | Encode the MP4 on the worker while the VAE is still decoding, instead of after the full video is materialized |
| `preencode_batch_frames` | positive integer | 17 (H3, Wan T2V/I2V); 1 (Wan S2V) | Minimum accumulated frames per worker transfer/encoding batch; used only with `preencode_mp4=true` |
| `video_codec_options` | object | null | Encoder options passed through to the H.264 encoder, such as `{"preset": "ultrafast", "threads": "0"}` |

With `preencode_mp4` enabled, each committed VAE chunk leaves the accelerator and
is encoded while later chunks are still decoding, so host transfer and CPU
encoding overlap the remaining decode instead of following it. The response is
unchanged: the same complete MP4, byte-for-byte equivalent frames.

```bash
curl -X POST http://localhost:8091/v1/videos/sync \
  -F "prompt=A small robot walking through a neon city" \
  -F 'extra_params={"preencode_mp4": true, "preencode_batch_frames": 33, "video_codec_options": {"preset": "ultrafast"}}' \
  -o output.mp4
```

Set `preencode_batch_frames` in `extra_params` (or `extra_args` for offline
sampling) to tune batching. The worker accumulates complete VAE chunks until
it has at least this many frames, then transfers and encodes them together.
It always flushes the final partial batch. This is a threshold, not an exact
chunk length: a value of 1 submits every native chunk immediately, and a value
smaller than a native chunk does not split it. Larger values reduce transfers
but retain more frames on the accelerator and delay encoding. The VAE decode
window and output frame count stay unchanged. Wan S2V keeps its existing
per-clip behavior by default. Zero, negative, fractional, boolean, string, and
null values are rejected when pre-encoding is enabled.

`preencode_mp4` applies to the complete-MP4 response paths only. The
`/v1/realtime/video` WebSocket endpoint rejects it, because that path already
overlaps encoding through its own incremental fragmented-MP4 encoder. Wan also
rejects it together with `enable_frame_interpolation`, which needs the decoded
frames the pre-encoded path no longer materializes.

Support is per model: MiniMax-H3 and Wan 2.2 (T2V, I2V, and S2V) implement it,
and other models ignore the flag and take the full-decode path.

## Storage

Set `VLLM_OMNI_SERVER_STORAGE__PATH` to control where asynchronous video outputs are
stored:

```bash
export VLLM_OMNI_SERVER_STORAGE__PATH=/var/tmp/vllm-omni-videos
```

> `VLLM_OMNI_STORAGE_PATH` is deprecated and will be removed in a future release;
> use `VLLM_OMNI_SERVER_STORAGE__PATH` instead.

## Model-Specific Examples

For complete text-to-video, image-to-video, and model-specific video-to-video
walkthroughs, see:

- [Text-to-Video](../user_guide/examples/online_serving/text_to_video.md)
- [Image-to-Video](../user_guide/examples/online_serving/image_to_video.md)
- [Speech-to-Video](../user_guide/examples/online_serving/speech_to_video.md)
  for Wan2.2-S2V audio-driven lip-sync generation
- [Cosmos3 recipes](https://github.com/vllm-project/vllm-omni/blob/main/recipes/cosmos3/Cosmos3-Nano.md)
  for model-specific video-to-video examples and conditioning controls
