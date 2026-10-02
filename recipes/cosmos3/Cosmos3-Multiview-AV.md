# Cosmos3 Multiview-AV

## Summary

- Vendor: NVIDIA
- Model: Cosmos3 Nano multiview AV exports, e.g.
  `wsm_transfer_nano_480p_11view_decomposed_attn_16n` (camera-only) and the
  versioned joint camera+LiDAR exports
- Task: multiview driving video generation (T2V, I2V, video prefix, WSM
  transfer, view completion), optionally joint with numeric LiDAR
- Mode: offline (`Omni`) and online (`vllm serve --omni`, `/v1/videos`)
- Hardware: NVIDIA CUDA GPUs; sparse FA4 defaults on supported SM100/SM110
  (Blackwell), with an explicit, unverified SM90 (Hopper) option
- Maintainer: Maciej Bala

## When to use this recipe

Use it to generate up to eleven synchronized MADS camera views, and optionally a
LiDAR range sequence, in one bidirectional denoising pass. The model reuses the
Cosmos3 Nano architecture and weights, adds camera-major VAE processing and a
weight-free sparse attention mask, and, for versioned exports, a few small
projection and embedding tables (see [Checkpoint](#checkpoint)).

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

| Checkpoint contract | Camera selection |
| --- | --- |
| Unversioned, or versioned with `variable_view_count: false` | All exported cameras, in exported order |
| Versioned (`schema_version` 2 or 3) with `variable_view_count: true` | Any non-empty subset of the exported cameras, in any order |

Request order sets caption association and output order. Version-3
checkpoints with `rig_view_embedding` embed each camera by its physical rig ID,
so subsets and reordered views keep the trained per-camera identity.

### Inputs

| Input | Format and limits |
| --- | --- |
| Prompt | One shared prompt. Checkpoints with `per_view_captions: true` also require a plain-text `prompt` per view; runtime camera labels and metadata sentences are rejected. |
| Camera control / vision | Local paths (offline, or server-local over HTTP) or multipart uploads. MP4, MOV, MKV, WebM, or BMP, GIF, JPEG, PNG, TIFF, WebP. |
| LiDAR control | `lidar.control_path`: a `.safetensors` file holding one float32 tensor `frames` of shape `[3, T, 128, 1800]` (range in metres, intensity and validity in `[0, 1]`) |
| LiDAR condition (optional) | `lidar.condition_path`, same format; `lidar.num_conditional_sweeps` (default 1) measured sweeps condition the start of the generated LiDAR |
| Prompt length | `max_sequence_length` may be lowered but not raised above 4096 (see [Prompt length](#prompt-length)) |

### Outputs

| Output | Contract |
| --- | --- |
| Video | One camera-major clip per request; all cameras share one size. The offline example writes one MP4 per camera. |
| Frames and rate | `num_frames` per camera (default 201) is rounded up to the VAE's `4k+1` grid. fps defaults to 30, the training rate. Other rates are accepted, with a warning outside [10, 30]. |
| Geometry | `resolution` `"480"` or `"720"` and `aspect_ratio` (see [Resolution and aspect ratio](#resolution-and-aspect-ratio)) |
| LiDAR (joint checkpoints, opt-in) | `lidar.return_output: true` returns float32 `[3, T, 128, 1800]` sweeps at the checkpoint's LiDAR rate, starting at the camera clip's time origin |

Sampling defaults come from the checkpoint's `inference_defaults` for versioned
exports. Unversioned exports use the Cosmos3 video defaults (35 steps,
guidance 6.0, flow shift 10, 480p) and cap guidance at 7.0.

## References

- Offline example: [`examples/offline_inference/multiview_video/cosmos3_multiview.py`](../../examples/offline_inference/multiview_video/cosmos3_multiview.py)
- Online client: [`examples/online_serving/multiview_video/cosmos3_multiview_client.py`](../../examples/online_serving/multiview_video/cosmos3_multiview_client.py)
- [Video API](../../docs/serving/videos_api.md)
- [Supported models](../../docs/models/supported_models.md) and the
  [diffusion feature matrix](../../docs/user_guide/diffusion_features.md)
- Base model recipe: [Cosmos3-Nano](Cosmos3-Nano.md)

## Checkpoint

Export with the normal Cosmos3 EMA-to-Diffusers conversion, then set
`model_index.json` to:

```json
{"_class_name": "Cosmos3MultiviewPipeline"}
```

`transformer/config.json` keeps all Cosmos3 Nano fields and adds
`"backbone_type": "cosmos3_multiview"` plus a `multiview` object. The
imaginaire4 exporter writes versioned objects (`schema_version` 2 or 3) that
also carry `per_view_captions`, `variable_view_count`, `inference_defaults`
and, for joint checkpoints, `lidar`. A versioned object with an unknown field is
rejected at load time. The minimal unversioned form for the camera-only WSM
checkpoint is:

```json
{
  "backbone_type": "cosmos3_multiview",
  "multiview": {
    "causal_training_strategy": "none",
    "attention_scope": "decomposed",
    "decomposed_temporal_window_seconds": null,
    "control_attends_sensor": false,
    "align_temporal_positions_across_views": false,
    "backend": "triton",
    "max_views": 11,
    "share_vision_temporal_positions": true,
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
    ]
  }
}
```

Unversioned artifacts require the canonical camera order above and cannot be
joint or maskless.

Weights beyond Cosmos3 Nano, checked at load time:

| Contract | Extra transformer weights | Extra directory |
| --- | --- | --- |
| Unversioned / version 2, camera-only | None | None |
| `rig_view_embedding` declared (version 3) | `rig_view_embed.weight` | None |
| Joint (`multiview.lidar` present) | `lidar_proj_in.{weight,bias}`, `lidar_proj_out.{weight,bias}` | `lidar_vae/` (`config.json`, `diffusion_pytorch_model.safetensors`) |

The scheduler directory must describe the regular FlowUniPC scheduler.

### Sparse attention backend

`multiview.backend` selects how the visibility mask is executed:

| `backend` | Kernel | Sparse block `(q, kv)` | Requirements |
| --- | --- | --- | --- |
| `"triton"` | PyTorch FlexAttention, Triton template | 64 × 64 | Any CUDA GPU |
| `"fa4"` | vLLM's bundled FlashAttention-4 CuTe (`vllm.vllm_flash_attn.cute`) | 256 × 128 | CUDA build of vLLM; SM100/SM110 (Blackwell), or explicit SM90 (Hopper) pending GPU verification |
| `"maskless"` | Dense FlashAttention over per-branch key folds | — | Versioned checkpoint trained with maskless semantics (the v2 AV model) |

Triton and FA4 implement the same visibility predicate and differ in block
geometry and rounding. For a checkpoint declaring `"triton"`, Cosmos3 selects
sparse FA4 automatically when its shared version resolver returns FA4. This
happens on SM100/SM110 when vLLM reports FA4 support. On Hopper (SM90), the
resolver prefers FA3, or FA2 when FA3 is unavailable, so the sparse checkpoint
stays on Triton. If the version resolver raises an import or availability error,
automatic selection also keeps Triton. Maskless uses the resolved FlashAttention
version for its dense kernels; overlapping branch keys count twice, so it
requires a matching checkpoint.

Set `VLLM_OMNI_COSMOS3_MULTIVIEW_BACKEND=triton|fa4` to pin a sparse backend
without editing the checkpoint. An unknown name, or a switch to or from
`maskless`, fails at load time. To request FA4 explicitly on Hopper, set this
before starting the offline process or server:

```bash
export VLLM_OMNI_COSMOS3_MULTIVIEW_BACKEND=fa4
```

vLLM supports FA4 on Hopper, and the Cosmos3 adapter uses a scalar mask callback
there. Its vector mask callback is specific to SM100/SM110. For Cosmos3's usual
`head_dim=128`, source inspection indicates that the 256 × 128 sparse block map
is compatible with Hopper's 128 × 128 compute tile. Actual Hopper FA4
correctness and performance have not been verified; run the CUDA checks in
[Verification](#verification) before qualifying an H100/H200 deployment.

Goldens taken on Triton must be re-calibrated before they gate FA4, or pinned
with `VLLM_OMNI_COSMOS3_MULTIVIEW_BACKEND=triton`.

## Hardware

- Accelerator: NVIDIA CUDA GPU. vLLM's bundled FA4 supports SM90 (Hopper) and
  SM100/SM110 (Blackwell); see [Sparse attention backend](#sparse-attention-backend)
  for Cosmos3's selection policy and qualification status.
- Devices: 1 by default. CFG parallelism (2-way), strict Ulysses CP, TP and HSDP
  are supported through the engine flags (see [Supported features](#supported-features)).
- Qualification scope: no memory or latency profile is recorded in this
  repository yet. Record one per the [Verification](#verification) section
  before claiming a hardware profile.

## Software environment

- vLLM-Omni: this branch; FA4 uses the copy bundled with vLLM's CUDA build, so no extra is needed.
- Guardrails: `cosmos-guardrail` and access to the gated
  `nvidia/Cosmos-1.0-Guardrail` model (see [Safety guardrails](#safety-guardrails)).

## Command

### Offline

The input JSON carries the prompt, one hint (`"wsm": {}`), and
`multiview.views`, one entry per camera with `camera_key`, `control_path` and,
for I2V or prefix conditioning, `vision_path`. Set
`multiview.condition_video_as_image: true` to condition on the first frame only.
Add a top-level `lidar` object for joint requests:

```json
"lidar": {"control_path": "lidar_control.safetensors", "return_output": true}
```

```bash
python examples/offline_inference/multiview_video/cosmos3_multiview.py \
  --model /models/cosmos3-multiview-av \
  --input /data/mv_i2v_wsm.json \
  --negative-prompt-json recipes/cosmos3/negative_prompt.json \
  --output-dir outputs/mv_i2v_wsm \
  --seed 42 --fps 30 --num-frames 200
```

The script writes `vision_viewNN_<camera>.mp4` per camera (plus
`combined_views.mp4` with `--combine-views`), `lidar.safetensors` when LiDAR
output was requested, and `sample_outputs.json` with the resolved geometry and
metadata. `--negative-prompt-json` applies the required serialization; a
`negative_prompt` string in the input wins over it. `--fps`, `--num-frames`,
`--resolution` and `--aspect-ratio` override every record. Records may use
`guidance`, `num_steps` and `shift` as aliases for `guidance_scale`,
`num_inference_steps` and `flow_shift`; the vLLM-Omni names win when both are
present. The negative prompt carries the same duration/FPS and resolution
sentences as the positive prompt; set `negative_metadata_mode` to change that.

Camera files are encoded concurrently by default (at least two, at most four
FFmpeg threads per camera, bounded by the CPU affinity mask);
`--video-encoding-mode serial` is a diagnostic fallback.

### Online

```bash
vllm serve /models/cosmos3-multiview-av --omni \
  --model-class-name Cosmos3MultiviewPipeline --port 8091
```

`POST /v1/videos` and `POST /v1/videos/sync` take the request manifest as the
JSON-encoded `extra_params` form field. Media may be server-local
`control_path`/`vision_path` entries or multipart `input_references` parts
referenced by zero-based `control_reference_index`/`vision_reference_index`
(LiDAR: `control_reference_index`/`condition_reference_index`). Every upload
must be referenced exactly once, and one camera role cannot have both a path and
an index. The client uploads the local paths of an offline manifest:

```bash
python examples/online_serving/multiview_video/cosmos3_multiview_client.py \
  request.json --server http://localhost:8091 --output multiview.mp4
```

`/content` returns one camera-major MP4. For joint jobs with
`lidar.return_output: true`, the completed job carries a `lidar` descriptor and
`GET /v1/videos/{video_id}/lidar` returns the numeric file; the client saves
`<output-stem>.lidar.safetensors` and `.lidar.json`. LiDAR output requires the
asynchronous endpoint; `/v1/videos/sync` rejects it.

Setting `extra_params.parallel_multiview_encoding: true` opts in to
camera-parallel MP4 encoding on the server. It needs at least two cameras, no
audio, and enough affinity CPUs; otherwise the server logs why and uses the
regular encoder.

## Resolution and aspect ratio

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
`extra_args`/`extra_params` or inside `multiview`; the `multiview` value wins.
Explicit sampling width/height must match the resolved bucket.

## Prompt length

The sparse attention pads the text (UND) stream to one fixed capacity, 4096
prompt tokens plus the two framing tokens. The pad is numerically free: pad
keys are excluded from every real query by the visibility predicate. Requests
may lower `max_sequence_length` but not raise it above 4096; a larger value is
rejected at admission, and longer prompts are truncated.

The fixed capacity keeps the mask plan and packing buffers identical across
prompts. The Triton attention call runs outside the regionally compiled GEN
layers, through FlexAttention's own dynamic-shape compile, and the cached UND
keys/values are marked dynamic in their sequence dimension. New prompts and the
two CFG branches therefore do not recompile the GEN layers. The GEN layers
themselves are compiled statically and specialize per output geometry.

## Safety guardrails

As for the other Cosmos3 models, safety guardrails are **on by default**
(NVIDIA Open Model License). Before generation, the shared prompt and every
per-camera caption pass the text guardrail; after decoding, each camera clip
passes the video guardrail (face blur) separately. The guardrails load the
**gated** `nvidia/Cosmos-1.0-Guardrail` model, so to keep them on you must:

1. `pip install cosmos-guardrail`
2. Accept the license at <https://huggingface.co/nvidia/Cosmos-1.0-Guardrail>
3. Export a token with access: `export HF_TOKEN=hf_...`

To run **without** guardrails (you are responsible for license compliance), add
`--no-guardrails` to the offline script or to `vllm serve`; neither needs
the token nor `cosmos-guardrail`. When the server loads guardrails, a request
can skip them with `"guardrails": false` in its `extra_params`; a request cannot
turn them on for a server started with `--no-guardrails`.

## Verification

Run the CPU contract tests:

```bash
pytest -q \
  tests/diffusion/models/cosmos3/test_cosmos3_pipeline.py \
  tests/diffusion/models/cosmos3/test_cosmos3_transformer.py \
  -k "multiview or lidar or rig"
```

These cover checkpoint contract validation, camera selection, per-camera
captions, LiDAR admission and conditioning, rig-view embedding, guardrail hooks,
and FA4 backend selection, mask metadata validation and full-graph custom-op
capture with mocked kernels. The FA4 tests do not verify CUDA kernel
correctness or performance. The dedicated GPU attention, LiDAR decoder,
parallelism, recompilation and HTTP upload suites are not in the tree; run a
CUDA generation to cover those paths.

For checkpoint validation, generate 29-frame clips in WSM-only and
vision-conditioned modes for all five ratios at both resolutions with the same
seed and settings, including explicit overrides as well as automatic detection.
Check every exported video for camera order, frame count and exact dimensions,
then run one 201-frame 720p portrait generation. Record the checkpoint, backend,
GPU model and count, parallel topology, steps, cold/warm latency and peak
memory. To qualify Hopper FA4, repeat these checks with
`VLLM_OMNI_COSMOS3_MULTIVIEW_BACKEND=fa4` and
`VLLM_OMNI_COSMOS3_MULTIVIEW_BACKEND=triton`, using identical checkpoints,
seeds and sampling settings, and compare outputs, latency and peak memory.

## Supported features

| Feature | Status | Guide |
| --- | --- | --- |
| CFG parallelism | ✅ 2-way (`--cfg-parallel-size 2`) | [CFG parallel](../../docs/user_guide/diffusion/parallelism/cfg_parallel.md) |
| Sequence parallelism | ✅ strict Ulysses only (`--ulysses-degree`) | [Sequence parallel](../../docs/user_guide/diffusion/parallelism/sequence_parallel.md) |
| Tensor parallelism | ✅ (`--tensor-parallel-size`); not with HSDP | [Tensor parallel](../../docs/user_guide/diffusion/parallelism/tensor_parallel.md) |
| HSDP | ✅ (`--use-hsdp --hsdp-shard-size N`); not with TP | [HSDP](../../docs/user_guide/diffusion/parallelism/hsdp.md) |
| Regional compilation | ✅ static GEN layers; `--enforce-eager` disables it | [Regional compilation](../../docs/user_guide/diffusion/regional_compilation.md) |
| Cache-DiT / TeaCache | ❌ disabled at startup with a warning | [Cache-DiT](../../docs/user_guide/diffusion/cache_acceleration/cache_dit.md) |
| Session state | ❌ rejected at load time | — |
| LiDAR decoder parallelism | ❌ the decoder runs eager FP32 on every rank | — |

Example: four GPUs with CFG parallelism and Ulysses CP:

```bash
vllm serve /models/cosmos3-multiview-av --omni \
  --model-class-name Cosmos3MultiviewPipeline --num-gpus 4 \
  --cfg-parallel-size 2 --ulysses-degree 2 --port 8091
```

For HSDP on the same four GPUs, add `--use-hsdp --hsdp-shard-size 4`. For
TP2 × CP2, replace `--cfg-parallel-size 2` with `--tensor-parallel-size 2`.

## Notes

- Memory and latency depend on clip length, camera count, resolution, backend
  and topology. The transformer pads spatial axes and crops back before VAE
  decode; the buckets use 390–400 spatial tokens per latent frame and camera at
  480p and 900–920 at 720p.
- Joint checkpoints load the LiDAR decoder even when a request does not ask for
  LiDAR output; such requests skip decoder execution.
- LiDAR CUDA parity, memory and latency have not been qualified in this
  repository.
