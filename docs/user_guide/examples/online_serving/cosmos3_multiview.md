# Cosmos3 multiview and numeric LiDAR

## Multi-GPU serving

Use the existing engine parallelism flags; the video API payload is unchanged:

```bash
vllm serve /models/cosmos3-multiview --omni \
  --model-class-name Cosmos3MultiviewPipeline --num-gpus 4 \
  --cfg-parallel-size 2 --ulysses-degree 2 --port 8091
```

For HSDP on those same four GPUs, add `--use-hsdp --hsdp-shard-size 4`.
For TP2 x CP2, replace `--cfg-parallel-size 2` with `--tensor-parallel-size 2`.
HSDP and TP are mutually exclusive. Single-GPU execution remains the default.
Triton and FA4 use the same sparse visibility rules; FA4 requires datacenter
Blackwell and the optional `fa4` extra.

See the
[offline script](https://github.com/vllm-project/vllm-omni/blob/main/examples/offline_inference/multiview_video/cosmos3_multiview.py)
for usage examples.

## Upload contract

`POST /v1/videos` and `POST /v1/videos/sync` accept all camera inputs in the HTTP request.
See the [video API reference](../../../serving/videos_api.md) for endpoint and response details.
Send the files as repeated `input_references` parts and the camera manifest as
a JSON-encoded `extra_params` form field. Each view's
`control_reference_index` or `vision_reference_index` is a zero-based index
into the uploaded file list. Filenames do not identify cameras; duplicate
filenames are allowed. The server substitutes temporary paths before inference.

For example, a view can reference control upload 0 and vision upload 11:

```json
{
  "camera_key": "camera_front_wide_120fov",
  "control_reference_index": 0,
  "vision_reference_index": 11
}
```

For schema-version-2 checkpoints with `variable_view_count=true`, select any
nonempty subset of the checkpoint cameras in `extra_params.multiview.views`.
Request order determines caption association and RGB output order. Unversioned
WSM artifacts require all eleven cameras in canonical order; versioned artifacts
with `variable_view_count=false` require their full exported camera order.
The production rig supports:

```text
camera_front_wide_120fov
camera_cross_right_120fov
camera_rear_right_70fov
camera_rear_tele_30fov
camera_rear_left_70fov
camera_cross_left_120fov
camera_front_tele_30fov
camera_front_fisheye_200fov
camera_left_fisheye_200fov
camera_right_fisheye_200fov
camera_rear_fisheye_200fov
```

Select conditioning by request mode, including camera-only requests on joint checkpoints:

| Mode | Controls | RGB conditions |
|---|---|---|
| Joint camera + LiDAR | WSM for every camera plus numeric LiDAR; exactly `wsm` | Every selected camera or none |
| Ordinary camera generation | No controls or hints | T2V, I2V, or video prefix |
| Camera transfer | Every camera; exactly one of `wsm`, `edge`, `depth`, `seg`, `blur` | Every selected camera or none |
| View completion | Every camera; exactly one hint | Complete videos for known cameras; omitted unknown views |

Set the selected hint to `true`, `{}`, or `{"weight": 1.0}`. A single hint's
weight is relative and normalizes to one. Completion fixes all latent frames
of known views and rejects short videos, partial images, or explicit condition
indexes. Joint requests reject partial-view conditioning and non-WSM hints.
Within each role, use all videos or all images. Uploads support MP4, MOV, MKV, WebM and
the existing control image formats (BMP, GIF, JPEG, PNG, TIFF, WebP).

Every upload must be referenced exactly once, and indexes must be integers.
You may retain existing server-side `control_path`/`vision_path` entries for
other inputs, but cannot provide a path and an upload index for the same
camera and role. Do not combine this upload mode with `input_reference`,
`image_reference`, `video_reference`, `audio_reference`, `control_reference`,
or `control_type`.

The online client accepts an existing offline JSON manifest containing
`prompt`, `wsm`, and `multiview`, or a video API manifest containing
`extra_params`. It opens local `control_path`/`vision_path` files (also accepting
the `control`/`vision` path aliases) and replaces them with upload indexes:

```bash
python examples/online_serving/multiview_video/cosmos3_multiview_client.py \
  request.json --server http://localhost:8091 --output multiview.mp4

# Use the synchronous endpoint:
python examples/online_serving/multiview_video/cosmos3_multiview_client.py \
  request.json --server http://localhost:8091 --sync --output multiview.mp4

# Override the manifest's frame count, resolution, and aspect ratio:
python examples/online_serving/multiview_video/cosmos3_multiview_client.py \
  request.json --num-frames 29 --resolution 720 --aspect-ratio 9:16 --output multiview.mp4
```

`--num-frames` must be positive. `--resolution` selects `480` or `720`; when the
CLI and manifest omit it, the checkpoint supplies the default (480 for legacy WSM).
`--aspect-ratio` accepts `auto` (default), `1:1`, `4:3`, `3:4`, `16:9`, or `9:16`;
comma spellings are also accepted. All selected cameras share the selected size:

| Aspect ratio | 480p (width × height) | 720p (width × height) |
|---|---|---|
| 1:1 | 640 × 640 | 960 × 960 |
| 4:3 | 736 × 544 | 1104 × 832 |
| 3:4 | 544 × 736 | 832 × 1104 |
| 16:9 | 832 × 480 | 1280 × 720 |
| 9:16 | 480 × 832 | 720 × 1280 |

Automatic mode selects the nearest Cosmos3 bucket from the original dimensions
of the first selected camera's control input, using the first frame for video.
Without controls it uses that camera's vision input, or 16:9 for ordinary T2V. Detection happens in the pipeline after uploaded references
have been resolved. Other cameras and optional vision inputs do not determine
the ratio; all are resized and center-cropped to the same output size. Input
pixel count does not select the resolution tier. An unreadable first WSM input
fails generation. Specify `--aspect-ratio 16:9` to retain the previous fixed
landscape behavior.

Offline manifests accept top-level `resolution`/`aspect_ratio` or their fields
inside `multiview`; API manifests also accept them inside `extra_params`.
Duplicate declarations must agree after normalization. Each CLI override
replaces declarations for its own setting and clears stale width/height
constraints. Automatic mode or an omitted resolution sends no computed width/height; user-supplied
constraints are retained when there is no geometry override and must match the
pipeline's resolved size. The client does not decode media or edit the manifest.

For direct HTTP requests, use the existing `aspect_ratio` form field or
`extra_params.aspect_ratio` / `extra_params.multiview.aspect_ratio`. Nested
multiview fields take precedence over `extra_params` fields, which take
precedence over the top-level aspect-ratio field. Resolution defaults to the
checkpoint's `inference_defaults.resolution` (480 for legacy WSM); aspect ratio
defaults to `"auto"`. For portrait output at 720p, set
`extra_params.multiview.resolution` to `"720"` and
`extra_params.multiview.aspect_ratio` to `"9:16"`, including the selected camera
entries as described above. Omit width/height or explicitly supply width `720`
and height `1280`. Only canonical 480p and 720p buckets are supported.
Single-dimension constraints are also checked by the pipeline.

720p uses approximately 2.3 times as many spatial tokens as 480p. Measure
latency and peak memory for the desired clip length and execution topology.

Relative input paths resolve against the manifest's directory. The default
client submits a background job, polls it, and downloads the existing video
output. Set `VLLM_API_KEY` or `OPENAI_API_KEY` when the server requires a bearer
token. Output packaging and per-camera conditioning settings are unchanged.

Each uploaded file must be nonempty and no larger than 512 MiB; at most 23
files are accepted. Invalid manifests return HTTP 400 before inference.
Media decoding happens during generation, so a corrupt clip can produce a
failed asynchronous job even after the upload was accepted.

Uploaded files are retained until the job finishes or is cancelled, and are
removed on errors and synchronous timeouts. Existing caller-owned paths are
never included in upload cleanup. API and inference workers must share access
to the temporary filesystem. Configure proxy body-size limits and temporary
disk capacity for concurrent requests: the maximum payload is 11.5 GiB, and
multipart spooling plus persisted inputs can temporarily require extra disk
space. Application file limits are checked after multipart parsing; they do
not replace ingress limits. This API does not provide resumable uploads or
cross-host file transport.


## Joint input and caption contract

New exports have `transformer/config.json` → `multiview.schema_version=2`.
The joint artifact includes `lidar_vae/config.json` and
`lidar_vae/diffusion_pytorch_model.safetensors`, containing the complete V1.2
encoder, decoder, coordinates, and latent statistics. Joint deployments load
independent FP32 encoder and decoder components at startup. By default, inference
returns RGB; numeric LiDAR decoding is enabled per request.

The VAE config stores resolved network constructor defaults in addition to the
explicit settings in `transformer/config.json` → `multiview.lidar`. The runtime
constructs both components from the VAE config and requires every explicit
transformer setting and shared metadata field to agree. The component specifies
`dtype="float32"` and `sample_posterior=false`. `apply_validity_mask` describes
decoder behavior and does not change encoder input masking.

Install the optional runtime dependencies in the supported CUDA environment:

```bash
uv pip install -e '.[cosmos3-lidar]'
```

The encoder and decoder use PyTorch FlexAttention for spatial neighborhood
attention; Cosmos3 does not require NATTEN or a separate CUDA extension build.
Use a supported PyTorch/CUDA runtime with FlexAttention. Each local attention
shape compiles on first use, including partial streaming chunks; the rest of
the VAE executes eagerly. Even with `--enforce-eager`, local LiDAR attention
compiles to keep memory bounded. Compilation/kernel failures raise an error.

LiDAR attention stays in FP32 with IEEE multiplication independently of the
camera model's precision settings. Its spatial masks and compiled operators
use bounded caches; temporal state remains local to each request. Mask geometry
comes from each encoder/decoder level, including asymmetric decoder artifacts.
CPU execution is available only for small correctness fixtures (at most 4,096
spatial tokens); production LiDAR grids require CUDA.

Missing VAE files, incompatible metadata, missing projection weights, malformed
encoder/decoder tensors, unsupported streaming architectures, and invalid latent
statistics fail loading. The component is loaded directly and does not require
a LiDAR entry in the Diffusers pipeline indexes.

Each production camera entry must include a nonempty plain-text `prompt`.
JSON-object per-camera captions are rejected, including objects without metadata
fields; this path does not support the reference's dictionary-caption format. Do not
include camera labels, duration/FPS/resolution templates, rig descriptions,
or control-emphasis sentences: the pipeline applies reference formatting once
per camera. Captions are tokenized independently with separate causal boundaries.
Camera tokens read their own caption; LiDAR tokens read all camera captions.
Positive emphasis follows `inference_defaults.emphasize_control_in_prompt`
(true in production exports); a request can explicitly enable or disable it.
Joint requests use the reference joint WSM/LiDAR sentence; camera-only
transfer uses the corresponding single-hint sentence. Production unconditional
captions are empty, and supplied negative prompts are ignored. Unversioned WSM
artifacts retain their top-level caption and negative-prompt behavior.

The online client also accepts older manifests whose top-level `prompt` is a
JSON-encoded object with `num_views` and a `views` list containing `view_index`
and `caption`. It copies each plain-text `caption` into the corresponding
`multiview.views[view_index].prompt` when that field is absent. Counts must match
the selected camera list, and indexes must cover `0` through `num_views - 1`
exactly once. Indexes refer to request camera order; keep that order aligned
with the aggregate captions. Explicit per-camera prompts take precedence.
The original top-level prompt is retained for older checkpoints, while camera
role/type metadata stays out of the extracted captions. Plain-text top-level
prompts are not automatically copied to every camera. Direct HTTP and offline
requests should supply `multiview.views[].prompt` explicitly for new checkpoints.

HTTP admission validates supplied per-camera captions but has no checkpoint
metadata to determine whether they are required. The worker enforces
`separate_view_text_tokenization`; a request missing a required camera prompt
can receive HTTP 200 at asynchronous admission and then fail as a job instead
of returning HTTP 400 at admission.

Offline input example (one selected camera):

```json
{
  "prompt": "",
  "num_frames": 201,
  "fps": 30,
  "resolution": "480",
  "seed": 42,
  "wsm": {},
  "lidar": {"control_path": "lidar_control.safetensors"},
  "multiview": {
    "condition_video_as_image": true,
    "views": [{
      "camera_key": "camera_front_wide_120fov",
      "prompt": "A car travels along a tree-lined road.",
      "control_path": "front_wsm.mp4",
      "vision_path": "front_rgb.mp4"
    }]
  }
}
```

The online client also uploads `lidar.control_path`, replacing it with
`extra_params.lidar.control_reference_index`. Numeric files are accepted only
in this role. The safetensors file must contain exactly one contiguous float32
tensor `frames` of shape `[3,T,128,1800]`: range in metres, unit intensity,
and validity in `[0,1]`. All values must be finite and ranges nonnegative.
Do not upload `.pt`, tar archives, sparse NPZ files, normalized tensors, or RGB
previews as numeric inputs.

HTTP admission checks the safetensors header, tensor name, shape, and dtype
without scanning tensor values. The worker checks finite values and physical
ranges after selecting the required sweeps. Invalid values in selected sweeps
therefore fail during generation; unused trailing sweeps are not scanned.

Start sweeps at the same instant as the RGB clip. Required sweeps are
`round(resolved_num_frames * lidar_fps / camera_fps)`; extra sweeps are truncated,
short clips rejected. At 201 frames, 30 camera FPS, and 10 LiDAR FPS, provide
67 sweeps. Camera frame counts round up to the VAE's `4k+1` grid. The encoder
adds four circular columns on each side, normalizes with checkpoint physical
limits, and runs FP32 posterior-mean streaming inference. Camera and LiDAR
control/target geometries stay independent. Camera targets are always decoded;
LiDAR targets are decoded when requested as described below.

New artifact defaults are 480p, 30 FPS, 35 steps, guidance 6, shift 10, and
control guidance 1. Resolution, FPS, and control emphasis come from artifact
metadata when absent from the request; explicit request values take precedence.
Export selects the largest active training rig and requires every other active
rig to be its ordered subset. Incompatible rigs fail export instead of creating
an untrained camera union or increasing `max_views` beyond the trained rig.
`guidance_interval` and `control_guidance_interval` accept
strict `[lo,hi]` bounds in scheduler timestep units (0–1000); CFG is enabled
only inside the interval. `normalize_cfg` preserves reference sample-wide
normalization. `sigma_max` is retained for request compatibility; the UniPC
schedule is determined by steps and shift, as in reference rectified-flow
inference. The preparation utility supplies caption-derived duration.

See [preparation and GPU comparison commands](cosmos3_multiview_lidar_validation.md).

## Generated numeric LiDAR

Set `lidar.return_output` to `true` in an offline manifest or in the asynchronous
HTTP request's `extra_params`. Keep the existing LiDAR control and camera WSM
inputs:

```json
"lidar": {"control_path": "lidar_control.safetensors", "return_output": true}
```

For direct multipart HTTP requests use `control_reference_index` instead of
`control_path`. The example client maps local paths to upload indexes and
preserves `return_output`. It must be a JSON boolean and defaults to `false`.

The offline engine returns `multimodal_output["lidar"]` as a contiguous CPU FP32
tensor `[1,3,T,128,1800]`, alongside video and `metadata.lidar`. The offline
example saves `lidar.safetensors` with a tensor named `frames`, shaped
`[3,T,128,1800]`, and records its metadata in `sample_outputs.json`.

Channels are range in metres, intensity in `[0,1]`, and validity. With the
checkpoint's `apply_validity_mask=true`, validity is binary and invalid ranges
and intensities are zero. Otherwise validity contains probabilities. The
threshold comes from `range_projection.validity_threshold` (default `0.5`).
Outputs use the physical 1800-column grid: four circular padding columns are
removed from each side. No smoothing, point-cloud conversion, or RGB processing
is applied to these tensors.

LiDAR uses its own checkpoint FPS and sweep count, starting at the camera clip's
time origin. Metadata records these values, channel units, validity settings,
and the projection configuration; the same metadata is embedded in the
safetensors header under `lidar` as JSON.

For `POST /v1/videos`, a completed job includes a `lidar` descriptor with `url`,
`file_name`, `format`, `shape`, `dtype`, `fps`, and sensor metadata. Download the
numeric file with `GET /v1/videos/{video_id}/lidar`; `/content` continues to return
the MP4. The example HTTP client downloads `<output-stem>.lidar.safetensors`
and `<output-stem>.lidar.json` alongside the MP4.

Both files follow the job's retention and deletion lifecycle. A job is marked
complete only after both requested artifacts are persisted. Failed or cancelled
jobs clean up partial outputs. Synchronous requests (`/v1/videos/sync`, or the
client's `--sync`) reject `return_output=true`; use an asynchronous job for LiDAR.

Cancelling a joint output job allows an active write up to one second to finish.
Repeated cancellation does not extend this period; RGB-only jobs have no save
grace period. Partial-file deletions run concurrently with a one-second wait
limit. DELETE waits up to two seconds for an in-progress job to stop, then returns
HTTP 409 if cancellation is still pending. Writes or deletions that outlive these
limits retain background ownership, and late writes trigger another cleanup.
These limits bound request waits; they cannot interrupt blocked filesystem
threads or guarantee cleanup after the server process exits.

Decoder execution is eager FP32 with checkpoint-owned chunk/context settings.
It runs independently on each pipeline rank and follows existing VAE device
placement; no decoder tensor or sequence parallelism is introduced. Joint
deployments therefore load decoder weights even when an individual request
does not ask for LiDAR output, but those requests skip decoder execution.
