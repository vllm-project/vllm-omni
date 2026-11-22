# Cosmos3-Nano-Transfer-Auto — GB200

## Summary

- Vendor: NVIDIA
- Model: Cosmos3-Nano-Transfer-Auto, a local Diffusers checkpoint loaded with
  `--model-class-name Cosmos3MultiviewPipeline`
- Task: multiview driving video generation (T2V, I2V, video prefix, WSM
  transfer, view completion), optionally joint with numeric LiDAR
- Mode: online serving (`vllm serve --omni`, `/v1/videos`)
- Hardware: NVIDIA GB200
- Maintainer: Maciej Bala

## When to use this recipe

Use it to generate up to eleven synchronized MADS camera views, and optionally a
LiDAR range sequence, in one bidirectional denoising pass. The model reuses the
Cosmos3 Nano architecture and weights. It adds camera-major VAE processing, a
sparse attention mask, required rig embeddings and optional LiDAR projection
tables (see [Checkpoint](#checkpoint)).

## Supported model contract

### Tasks

| Mode | Controls | RGB conditions |
| --- | --- | --- |
| Ordinary camera generation | None, and no hint | None (T2V), images (I2V), or a video prefix |
| Camera transfer | One control per selected camera, plus exactly one hint | Every selected camera, or none |
| View completion | One control per selected camera, plus exactly one hint | Complete videos for the known cameras; unknown cameras omit `vision_path` |
| Joint camera + LiDAR | WSM for every camera, exactly the `wsm` hint, and a numeric LiDAR control | Every selected camera, or none |

WSM is the only control hint any released checkpoint was trained on. Within a
role (control or vision), use all images or all videos.

### Cameras

Requests may select any non-empty subset of the exported `cameras`, in any order.
Request order sets caption association and output order. Every camera uses its
physical ID from the required `rig_view_embedding` table, so reordered subsets
retain each camera's trained identity.

### Inputs

| Input | Format and limits |
| --- | --- |
| Prompt | A plain-text `prompt` is required per view; runtime camera labels and metadata sentences are rejected. |
| Camera control / vision | Server-local paths or multipart uploads. MP4, MOV, MKV, WebM, or BMP, GIF, JPEG, PNG, TIFF, WebP. |
| LiDAR control | `lidar.control_path`: a `.safetensors` file holding one float32 tensor `frames` of shape `[3, T, 128, 1800]` (range in metres, intensity and validity in `[0, 1]`) |
| LiDAR condition (optional) | `lidar.condition_path`, same format; `lidar.num_conditional_sweeps` (default 1) measured sweeps condition the start of the generated LiDAR |
| Prompt length | `max_sequence_length` may be lowered but not raised above 4096 (see [Prompt length](#prompt-length)) |
| Negative captions | Optional `per_view_negative_prompt` (see [Negative captions](#negative-captions)); shared negative prompts are rejected |

### Outputs

| Output | Contract |
| --- | --- |
| Video | One camera-major MP4 per request: all frames of the first requested camera, then the next. All cameras share one size. |
| Frames and rate | `num_frames` per camera (default 201) is rounded up to the VAE's `4k+1` grid. fps defaults to 30, the training rate. Other rates are accepted, with a warning outside [10, 30]. |
| Geometry | `resolution` `"480"` or `"720"` and `aspect_ratio` (see [Resolution and aspect ratio](#resolution-and-aspect-ratio)) |
| LiDAR (joint checkpoints, opt-in) | `lidar.return_output: true` returns float32 `[3, T, 128, 1800]` sweeps at the checkpoint's LiDAR rate, starting at the camera clip's time origin |

Sampling defaults come from the checkpoint's required `inference_defaults`.
The current export uses 35 steps, guidance 6.0, flow shift 10 and 480p.
Guidance intervals use open timestep bounds, as in reference inference.

### Resolution and aspect ratio

`resolution` selects the `"480"` or `"720"` bucket (default: the checkpoint's
`inference_defaults.resolution`, else `"480"`), independent of the input's pixel
count. `aspect_ratio` defaults to `"auto"`, which picks the nearest bucket from
the first view's control input, or its vision input when it has no control
(first frame for videos). An unreadable input fails generation; requests with
no camera media (T2V) use `16:9`. All other inputs are resized and
center-cropped to the selected size.

| `aspect_ratio` | 480p (width × height) | 720p (width × height) |
| --- | --- | --- |
| `1:1` | 640 × 640 | 960 × 960 |
| `4:3` | 736 × 544 | 1104 × 832 |
| `3:4` | 544 × 736 | 832 × 1104 |
| `16:9` | 832 × 480 | 1280 × 720 |
| `9:16` | 480 × 832 | 720 × 1280 |

These are canonical buckets, so their dimensions need not form the exact ratio.
Other resolutions and arbitrary sizes are unsupported. Comma spellings such as
`"9,16"` are accepted. Requests may set both fields at the top level of
`extra_params` or inside `multiview`; the `multiview` value wins. Explicit
width/height must match the resolved bucket.

### Prompt length

The sparse attention pads the text (UND) stream to one fixed capacity, 4096
prompt tokens plus the two framing tokens. The pad is numerically free: pad
keys are excluded from every real query by the visibility predicate. Requests
may lower `max_sequence_length` but not raise it above 4096; a larger value is
rejected at admission, and longer prompts are truncated.

### Negative captions

Every camera has its own caption, and so does its unconditional (CFG) branch.
By default the unconditional caption is empty, as with training's caption
dropout. Set `per_view_negative_prompt` at the top level of `extra_params` to
apply one negative caption to every camera. It carries the same duration/FPS
and resolution sentences as the positive caption.

There is no shared negative prompt. Requests that set `negative_prompt` (form
field or `extra_params`) or `negative_metadata_mode` are rejected rather than
ignored. Requests with uploaded references fail with HTTP 400 before a job is
created; other requests fail when generation starts.

### Deployment profiles on GB200

| Profile | GPUs | Flags | Status |
| --- | --- | --- | --- |
| Single GPU | 1 | none | Runtime-qualified |
| CFG parallel | 2 | `--cfg-parallel-size 2` | Runtime-qualified |
| Strict Ulysses CP | N | `--ulysses-degree N` | Runtime-qualified |
| Tensor parallel | N | `--tensor-parallel-size N` | Configuration-only |
| HSDP | N | `--use-hsdp --hsdp-shard-size N` | Configuration-only |

The checkpoint's query and KV head counts must be divisible by TP × CP, which
is checked at load time. TP and HSDP cannot be combined.

## References

- [Video API](../../docs/serving/videos_api.md)
- [Supported models](../../docs/models/supported_models.md) and the
  [diffusion feature matrix](../../docs/user_guide/diffusion_features.md)
- Base model recipe: [Cosmos3-Nano](Cosmos3-Nano.md)

## Checkpoint

The checkpoint is a Diffusers directory whose `model_index.json` names
`Cosmos3MultiviewPipeline` and whose `transformer/config.json` sets
`backbone_type="cosmos3_multiview"`. Tensor names and weights follow Cosmos3
Nano, plus the extra weights listed below.

There is one strict `multiview` object. Missing required fields
and unknown fields are rejected, including old version markers, backend
settings, attention switches and caption aliases. A camera-only deployment
example with the full 11-camera rig is:

```json
{
  "backbone_type": "cosmos3_multiview",
  "multiview": {
    "cameras": [
      "camera_front_wide_120fov",
      "camera_cross_right_120fov",
      "camera_rear_right_70fov",
      "camera_rear_tele_30fov",
      "camera_rear_left_70fov",
      "camera_cross_left_120fov",
      "camera_front_tele_30fov",
      "camera_front_fisheye_200fov",
      "camera_left_fisheye_200fov",
      "camera_right_fisheye_200fov",
      "camera_rear_fisheye_200fov"
    ],
    "cross_view_past_window_seconds": 0.4,
    "rig_view_embedding": {
      "num_embeddings": 12,
      "camera_ids": {
        "camera_front_wide_120fov": 0,
        "camera_cross_right_120fov": 1,
        "camera_rear_right_70fov": 2,
        "camera_rear_tele_30fov": 3,
        "camera_rear_left_70fov": 4,
        "camera_cross_left_120fov": 5,
        "camera_front_tele_30fov": 6,
        "camera_front_fisheye_200fov": 7,
        "camera_left_fisheye_200fov": 8,
        "camera_right_fisheye_200fov": 9,
        "camera_rear_fisheye_200fov": 10
      },
      "lidar_id": 11
    },
    "inference_defaults": {
      "resolution": "480",
      "fps": 30.0,
      "num_steps": 35,
      "guidance": 6.0,
      "shift": 10.0,
      "control_guidance": 1.0,
      "emphasize_control_in_prompt": true,
      "guidance_interval": null,
      "control_guidance_interval": null,
      "normalize_cfg": false
    }
  }
}
```

The exported `cameras` list describes the checkpoint's supported rig, and its
length is the maximum camera count. Requests may select and reorder subsets;
`camera_ids` retains each camera's physical ID.

Joint exports additionally require complete `lidar` tokenizer metadata and
`lidar_latent_patch_size_hw`, the transformer's `[height, width]` patches over
LiDAR latents. The current checkpoint uses `[1, 1]`, independently of the camera
patch.

Attention is intrinsic to Cosmos3-Nano-Transfer-Auto: same-view target and control
attention spans the full clip, in both temporal directions. Cross-view target
keys are visible in `[query_time - cross_view_past_window_seconds, query_time]`,
including both boundaries with `1e-4` tolerance. Targets and controls read their
own camera caption; LiDAR reads all camera captions. Cross-view controls remain
isolated. The architecture uses aligned temporal positions and counts each key
exactly once.

Weights beyond Cosmos3 Nano, checked at load time:

| Contract | Extra transformer weights | Extra directory |
| --- | --- | --- |
| Every Cosmos3-Nano-Transfer-Auto checkpoint | `rig_view_embed.weight` | None |
| Joint (`multiview.lidar` present) | `lidar_proj_in.{weight,bias}`, `lidar_proj_out.{weight,bias}` | `lidar_vae/` (`config.json`, `diffusion_pytorch_model.safetensors`) |

The LiDAR component includes the full VAE, decoder and latent statistics; its
architecture is validated through the tokenizer fields. The scheduler directory
must describe the regular FlowUniPC scheduler. Deployment defaults omit the
ignored `sigma_max` and inapplicable `negative_metadata_mode` settings.

## Hardware

- Accelerator model and per-device memory: NVIDIA GB200 (Blackwell, SM100),
  186 GB HBM3e per GPU
- Number of devices: 1, or 2 and more for the parallel profiles in
  [Deployment profiles on GB200](#deployment-profiles-on-gb200)
- Device interconnect: NVLink
- Host memory: not recorded
- Qualification scope: the single-GPU, CFG-parallel and strict Ulysses CP
  profiles have been run on GB200. TP and HSDP are configuration-only. No
  memory or latency profile is recorded yet; see [Verification](#verification).

## Software environment

- OS: Linux (distribution not recorded)
- Python: not recorded (vLLM-Omni supports 3.10–3.13)
- Driver / runtime: NVIDIA driver and CUDA version not recorded; PyTorch 2.13.0
- vLLM version: a CUDA build of vLLM, whose bundled FlashAttention-4 provides
  the default sparse kernel on GB200; exact version not recorded
- vLLM-Omni version or commit: the commit that adds this recipe
- Guardrails: `cosmos-guardrail` and access to the gated
  `nvidia/Cosmos-1.0-Guardrail` model (see [Safety guardrails](#safety-guardrails))

## Command

### Safety guardrails

As for the other Cosmos3 models, safety guardrails are **on by default**
(NVIDIA Open Model License). Before generation, the shared prompt and every
per-camera caption pass the text guardrail; after decoding, each camera clip
passes the video guardrail (face blur) separately. The guardrails load the
**gated** `nvidia/Cosmos-1.0-Guardrail` model, so to keep them on you must:

1. `pip install cosmos-guardrail`
2. Accept the license at <https://huggingface.co/nvidia/Cosmos-1.0-Guardrail>
3. Export a token with access: `export HF_TOKEN=hf_...`

To run **without** guardrails (you are responsible for license compliance), add
`--no-guardrails` to `vllm serve`; this needs neither the token nor
`cosmos-guardrail`. When the server loads guardrails, a request can skip them
with `"guardrails": false` in its `extra_params`; a request cannot turn them on
for a server started with `--no-guardrails`.

### Server

Single GPU:

```bash
vllm serve /models/Cosmos3-Nano-Transfer-Auto --omni \
  --model-class-name Cosmos3MultiviewPipeline --port 8091
```

Two GPUs with CFG parallelism:

```bash
vllm serve /models/Cosmos3-Nano-Transfer-Auto --omni \
  --model-class-name Cosmos3MultiviewPipeline --num-gpus 2 \
  --cfg-parallel-size 2 --port 8091
```

For strict Ulysses CP, set `--num-gpus` to the total GPU count and add
`--ulysses-degree N`.

### Request

`POST /v1/videos` and `POST /v1/videos/sync` take the request manifest as the
JSON-encoded `extra_params` form field. The manifest carries `multiview.views`,
one entry per camera with `camera_key`, `prompt` and, for transfer or view
completion, `control_path`; I2V or prefix conditioning adds `vision_path`.
Transfer requests also set exactly one hint (`"wsm": {}`). Set
`multiview.condition_video_as_image: true` to condition on the first frame only.
Add a top-level `lidar` object for joint requests:

```json
"lidar": {"control_path": "lidar_control.safetensors", "return_output": true}
```

Media may be server-local `control_path`/`vision_path` entries or multipart
`input_references` parts referenced by zero-based
`control_reference_index`/`vision_reference_index` (LiDAR:
`control_reference_index`/`condition_reference_index`). Every upload must be
referenced exactly once, and one camera role cannot have both a path and an
index.

`/content` returns one camera-major MP4. For joint jobs with
`lidar.return_output: true`, the completed job carries a `lidar` descriptor and
`GET /v1/videos/{video_id}/lidar` returns the numeric file. LiDAR output requires the
asynchronous endpoint; `/v1/videos/sync` rejects it.

## Verification

With a server from [Server](#server) running, generate a two-camera T2V clip and
check its geometry:

```bash
curl -sf http://localhost:8091/v1/videos/sync \
  -F "prompt=Driving through a suburban street." \
  -F "num_frames=29" \
  -F 'extra_params={"multiview": {"views": [
        {"camera_key": "camera_front_wide_120fov", "prompt": "A car drives along a tree-lined suburban street."},
        {"camera_key": "camera_cross_left_120fov", "prompt": "Parked cars and houses pass on the left."}]}}' \
  -o multiview.mp4
ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=width,height,nb_read_frames -of csv=p=0 multiview.mp4
```

Expected output: `832,480,58`. The default 480p bucket for T2V is 16:9, and the
MP4 holds 29 frames for each of the two cameras, front-wide first.

Run the contract tests:

```bash
pytest -q \
  tests/diffusion/models/cosmos3/test_cosmos3_pipeline.py \
  tests/diffusion/models/cosmos3/test_cosmos3_transformer.py \
  tests/diffusion/models/cosmos3/test_multiview_config.py \
  tests/diffusion/models/cosmos3/test_multiview_attention.py \
  -k "multiview or lidar or rig"
```

These cover strict checkpoint metadata, single-camera and reordered subsets,
per-camera captions and negatives, LiDAR conditioning, required rig weights,
and kernel selection. The attention suite compares both runtime implementations
to an independent dense reference with mixed sensor clocks, inclusive window
bounds, unrestricted same-view attention, controls, captions and padding.
On GB200, the CUDA tests also run Triton's compiled Flex kernel and FA4
full-graph execution; CPU-only runs skip them.

To qualify a checkpoint, generate 29-frame clips in WSM-only and
vision-conditioned modes for all five ratios at both resolutions with the same
seed and settings, using explicit overrides as well as automatic detection.
Check every video for camera order, frame count and exact dimensions, then run
one 201-frame 720p portrait generation. Record the checkpoint, backend, GPU
count, parallel topology, steps, cold/warm latency and peak memory.

## Notes

- Memory usage: no memory profile is recorded yet. Memory and latency depend on
  clip length, camera count, resolution, backend and topology. The transformer
  pads spatial axes and crops back before VAE decode; the buckets use 390–400
  spatial tokens per latent frame and camera at 480p and 900–920 at 720p.
- Attention backend: the pipeline picks a kernel strategy from the shared vLLM
  FlashAttention version resolver. On GB200 (and other SM100-family GPUs) it
  selects `fa4`, vLLM's bundled FlashAttention-4 with a 256 × 128 block-sparse
  mask. Where the resolver reports FlashAttention 3 or 2 (Hopper, Ampere, Ada,
  consumer Blackwell) it selects `maskless`: the training implementation's
  strategy of a few unmasked variable-length FlashAttention passes (same view,
  deduplicated capture-time cross-view rectangles, captions) merged in FP32 by
  log-sum-exp, which replaces the much slower Triton FlexAttention kernel.
  `triton` (64 × 64 FlexAttention) remains the fallback when no bundled
  FlashAttention is importable. All three implement the same visibility rules;
  the maskless planner verifies at build time that its passes cover the sparse
  predicate exactly once. Set `VLLM_OMNI_COSMOS3_MULTIVIEW_BACKEND=fa4|maskless|triton`
  to pin one; unknown values fail at load time. Goldens taken on one backend must
  be re-calibrated before they gate another, or pinned to that backend.
- Compilation: the pipeline's `setup_compile()` hook compiles the GEN layers
  statically and regionally, so they specialize per output geometry.
  `--diffusion-compile-dynamic` and its negative form do not change this;
  `--diffusion-compile-granularity full` also uses regional compilation, with a
  warning; `--enforce-eager` skips it. FA4 and maskless attention run inside
  the compiled layers as opaque custom ops (the maskless plan's only
  prompt-dependent tensor is marked dynamic); the Triton attention call runs
  outside the compiled layers with its own dynamic-shape compile. The fixed
  text capacity keeps new prompts and both CFG branches from recompiling the
  GEN layers.
- Known limitations:
    - Other accelerators, including Hopper, are not qualified by this recipe.
      Hopper resolves to the `maskless` backend; its parity and latency against
      `fa4`/`triton` are pending a GPU benchmark.
    - LiDAR CUDA parity, memory and latency have not been qualified.
    - Joint checkpoints load the LiDAR decoder even when a request does not ask
    for LiDAR output; such requests skip decoder execution.

## Supported features

| Feature | Status on GB200 | Guide |
| --- | --- | --- |
| CFG parallelism | ✅ 2-way (`--cfg-parallel-size 2`) | [CFG parallel](../../docs/user_guide/diffusion/parallelism/cfg_parallel.md) |
| Sequence parallelism | ✅ strict Ulysses only (`--ulysses-degree`) | [Sequence parallel](../../docs/user_guide/diffusion/parallelism/sequence_parallel.md) |
| Tensor parallelism | Configuration-only (`--tensor-parallel-size`); not with HSDP | [Tensor parallel](../../docs/user_guide/diffusion/parallelism/tensor_parallel.md) |
| HSDP | Configuration-only (`--use-hsdp --hsdp-shard-size N`); not with TP | [HSDP](../../docs/user_guide/diffusion/parallelism/hsdp.md) |
| Regional compilation | ✅ static GEN layers; `--enforce-eager` disables it | [Regional compilation](../../docs/user_guide/diffusion/regional_compilation.md) |
| Cache-DiT / TeaCache | ❌ disabled at startup with a warning | [Cache-DiT](../../docs/user_guide/diffusion/cache_acceleration/cache_dit.md) |
| Session state | ❌ rejected at load time | — |
| LiDAR decoder parallelism | ❌ the decoder runs eager FP32 on every rank | — |
