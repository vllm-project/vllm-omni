# Frustum KV retrieval

Frustum retrieval keeps the full clean KV archive and attends to an exact subset
of past latent frames: initially 8 sinks, 32 retrieved frames, and 32 recent
frames. Retrieved frames maximize approximate camera-volume overlap with the
current chunk; ties prefer newer frames. The selected frames retain their
original positions and include both action and video tokens.

This bounds attended history, not archive memory. It is opt-in per request;
checkpoint weights, action contracts and sampling schedules are unchanged.

## Input

Add `kv_cache_frustum_path` to an example JSONL record. The path is relative to
the JSONL file. Its JSON object must contain:

- `c2w`: absolute OpenCV camera poses, shape `[RGB_frames, 4, 4]`, or a
  `pose_source` as described below.
- `fps`: the same FPS used for inference; omit for `camera_batch`.
- `image_size`: calibration `[width, height]`, not padded model dimensions.
- Either `intrinsics` (`[3,3]` or `[RGB_frames,3,3]`) or
  `horizontal_fov_degrees` (centered square-pixel estimate).
- `near`, `far`: finite camera-z distances in the pose translation units.
- Optional `sink_size`, `retrieved_size`, `recent_size` (defaults 8, 32, 32).

For example, set `image_size` to `[640,360]`, `horizontal_fov_degrees` to 90,
`near` to 0.1 and `far` to 10, alongside the actual pose array and FPS. Those
values are an estimated calibration, not a resizing instruction.

Instead of storing a pose array, use either pose source:

- `agibot_unified_v1_chunk16`: reconstruct poses from normalized unified actions
  using the checkpoint's bundled global-asinh statistics. Only AgiBot/camera
  domains are supported. Reconstruction continues across domain transitions.
- `camera_batch`: use the camera recipe's resolved trajectory, or RGB-aligned
  `gt_c2w`/`c2w` in a reference record. FPS comes from the request; do not specify
  it in the geometry config. Source-image calibration is not used.

For example, the same small config can be reused across AgiBot inputs:

```json
{
  "pose_source": "agibot_unified_v1_chunk16",
  "image_size": [640, 360], "fps": 30,
  "horizontal_fov_degrees": 90, "near": 0.1, "far": 10,
  "sink_size": 8, "retrieved_size": 32, "recent_size": 32
}
```

For camera inputs, change `pose_source` to `camera_batch` and remove `fps`.
Do not combine `pose_source` with `c2w`. The model action tensor is unchanged;
inverse normalization is used only for retrieval geometry.

API callers can resolve shortcuts with `resolve_frustum_geometry`, then pass
the resulting explicit geometry as `sampling_params.extra_args["frustum_geometry"]`.
The frontend loads local sidecars; it does not download S3 geometry or resolve
remote pose-source files. Pose rows at RGB indices `0,4,8,...` represent latent
endpoints when the exported temporal compression factor is four.

## Run

Use the normal deployment YAML with a full-history checkpoint
(`window_frames: null`, `sink_frames: 0`). Do not use the finite-window
overrides: eviction would destroy the retrieval archive.

```bash
python examples/offline_inference/cosmos3_nano_sim_bimanual/cosmos3_nano_sim_bimanual.py \
  --model /path/to/diffusers_format --jsonl /path/to/mixed.jsonl --sample-index 0 \
  --deploy-config vllm_omni/deploy/cosmos3_nano_sim_bimanual.yaml \
  --num-frames 901 --height 480 --width 832 --fps 30 --seed 42 \
  --output outputs/mixed_frustum.mp4
```

Only fresh, complete image-conditioned rollouts are supported; streaming ticks
and resuming sessions are rejected. This prevents reusing an archive with
changed geometry. Omitting geometry preserves ordinary inference.

## Storage and execution

Dense and paged archives both gather exact selected token positions into a
reusable joint-attention buffer. Paged storage is genuine managed KV storage;
the attention kernel reads the gathered buffer, not approximate boundary pages.
History gathering is reused across the chunk's denoising steps. Batched clean
refresh retains causality: each frame sees selected history and preceding
frames of its current chunk. Framewise clean refresh is also supported.

Compilation/CUDA graphs apply to the existing regional transformer blocks;
CPU selection and gathering remain outside capture. Stable staging addresses
are reused across requests. The standalone example currently forces eager
execution; compiled benchmarking uses Omni with `enforce_eager=False` and
`model_config.use_cuda_graphs=True` in its deployment.

The overlap approximation uses an 8x8x8 volume grid and does not model occlusion,
object motion, or stale scene contents. Geometry parity does not establish
end-to-end image parity with another framework.
