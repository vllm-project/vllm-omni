# Cosmos3 Multiview-AV

Cosmos3 Multiview-AV generates the eleven fixed MADS camera views in one
bidirectional denoising pass. It uses the regular Cosmos3 Nano architecture and
weights plus camera-major VAE processing and a weight-free sparse attention
mask. The runtime defaults to single-GPU and sequential-CFG execution and PyTorch
FlexAttention's Triton backend, with an opt-in FlashAttention-4 backend on
Blackwell (see [Sparse attention backend](#sparse-attention-backend)).

## Export contract

Export the `wsm_transfer_nano_480p_11view_decomposed_attn_16n` checkpoint with
the normal Cosmos3 EMA-to-Diffusers conversion. No multiview-only weight keys
are expected. Update `model_index.json` to use:

```json
{"_class_name": "Cosmos3MultiviewPipeline"}
```

Add the following fields to `transformer/config.json` (preserve all existing
Cosmos3 Nano fields):

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

The scheduler directory must describe the regular FlowUniPC scheduler. The
request defaults to 35 steps, guidance 6.0, flow shift 10, and the 480p resolution bucket.
Resolution, frame rate, and per-camera frame count are request-driven. When
omitted, fps defaults to 30 and num_frames to 201.

30 FPS is the training rate: the MADS WSM transfer recipes read their clips at
native 30 FPS and stamp "30 FPS" into the training captions, so the
fps-modulated temporal mRoPE and the prompt metadata are on-distribution only
there. Other rates are accepted with a warning outside [10, 30], and frame
counts are rounded up to the VAE's `4k+1` grid (200 becomes 201) instead of
being rejected.

## Resolution and aspect ratio

All eleven cameras share one output size. `resolution` selects the `"480"`
(default) or `"720"` bucket; it is independent of the input's pixel count.
`aspect_ratio` defaults to `"auto"`, which uses the original dimensions of the
first camera's WSM input (`camera_front_wide_120fov`). Images use their spatial
size and videos use their first frame. The nearest bucket is selected using
Cosmos3's existing target-size matching. Other cameras and vision inputs do
not affect the selection; all inputs are resized and center-cropped to it.
An unreadable first WSM input fails generation rather than selecting a fallback.

| `aspect_ratio` | 480p output (width × height) | 720p output (width × height) |
|---|---|---|
| `1:1` | 640 × 640 | 960 × 960 |
| `4:3` | 736 × 544 | 1104 × 832 |
| `3:4` | 544 × 736 | 832 × 1104 |
| `16:9` | 832 × 480 | 1280 × 720 |
| `9:16` | 480 × 832 | 720 × 1280 |

These are canonical buckets, so their dimensions need not form the exact
mathematical ratio. 256p, 704p, and arbitrary dimensions are unsupported.
Explicit ratios override detection; comma spellings such as `"9,16"` are also
accepted. To retain the previous fixed landscape behavior, specify `"16:9"`.

Offline JSON/JSONL records accept `resolution` and `aspect_ratio` at the top
level or inside `multiview`. Integer resolutions `480` and `720` are accepted.
Duplicate declarations must agree after normalization. Each CLI flag overrides
its own setting for every record; a geometry override clears stale width/height
constraints without editing the input file:

```bash
# Automatically select an aspect ratio from the first WSM input at 720p.
python examples/offline_inference/multiview_video/cosmos3_multiview.py \
  --model /models/cosmos3-multiview-av --input /data/mv_i2v_wsm.json \
  --resolution 720 --aspect-ratio auto --num-frames 29 --output-dir outputs/mv_auto

# Explicitly generate portrait views.
python examples/offline_inference/multiview_video/cosmos3_multiview.py \
  --model /models/cosmos3-multiview-av --input /data/mv_i2v_wsm.json \
  --resolution 720 --aspect-ratio 9:16 --num-frames 29 --output-dir outputs/mv_portrait
```

Direct pipeline requests prefer `extra_args.multiview.resolution` and
`extra_args.multiview.aspect_ratio` over their top-level `extra_args` equivalents,
then default to `"480"` and `"auto"`. The generic image resolution field is not
used. Explicit sampling width/height must match the resolved bucket, including
in automatic mode. Output dimensions and prompt metadata use that same bucket.

Detection runs in the pipeline for both offline and online requests; clients do
not decode media or inject landscape dimensions in automatic mode. See the
[online client guide](../../docs/user_guide/examples/online_serving/cosmos3_multiview.md).

The existing transformer pads either spatial axis and crops back before VAE
decode, preserving exact output dimensions. With spatial compression 16 and
patch size 2, the buckets use 390–400 spatial tokens at 480p and 900–920 at 720p,
per latent frame and item. Memory and latency also depend on clip length,
attention backend, and execution topology.

## Sparse attention backend

`multiview.backend` selects the kernel that consumes the sparse block map. Both
backends are built from the same run-level projection of the visibility
predicate, so they agree on which token pairs are visible; they differ only in
block geometry and floating-point rounding.

| `backend` | Kernel | Sparse block `(q, kv)` | Requirements |
|---|---|---|---|
| `"triton"` (default) | PyTorch FlexAttention, Triton template | 64 × 64 | Any CUDA GPU |
| `"fa4"` | FlashAttention-4 CuTe | 256 × 128 | SM100 (Blackwell), CUDA 13, `pip install 'vllm-omni[fa4]'` |

Because the backend changes only how the mask is executed, it can be overridden
per run without editing the checkpoint — useful for A/B measurement:

```bash
VLLM_OMNI_COSMOS3_MULTIVIEW_BACKEND=fa4 \
  python examples/offline_inference/multiview_video/cosmos3_multiview.py \
  --model /models/cosmos3-multiview-av --input /data/mv_i2v_wsm.json
```

The environment variable wins over `transformer/config.json`; an unset or empty
value falls back to the checkpoint. An unknown name fails at load time rather
than on the first generated frame. Parity thresholds are backend-specific:
goldens taken on Triton must be re-calibrated before they are used to gate the
FA4 path.

### Prompt length is capped by the variant, not the request

The sparse attention pads its text (UND) stream to a fixed capacity so the
compiled kernel sees one input shape for the life of the process. A pad that
tracked each prompt's length would resize the packed key tensor, and the kernel
is compiled with `dynamic=False`, so every distinct prompt length would cost a
recompile — and past Dynamo's default limit of eight the whole attention falls
back to eager FlexAttention, which cannot fit its score matrix at this
sequence length.

Requests may therefore *lower* `max_sequence_length` but not raise it past
`COSMOS3_MULTIVIEW_MAX_SEQUENCE_LENGTH` (4096); a larger value is rejected at
admission. Raise that constant if a golden fixture ever shows the reference
negative prompt being truncated. The padding itself is numerically free: pad
keys are excluded from every real query by the visibility predicate.

## Input and run

The input JSON needs to have the following structure: Each view must appear in
the exact exported camera order and provide `control_path`. For i2v_wsm, every
view also provides `vision_path`; set `condition_video_as_image: true` to use
only its first frame. A top-level empty `wsm` object selects the only supported
control hint.

```bash
python examples/offline_inference/multiview_video/cosmos3_multiview.py \
  --model /models/cosmos3-multiview-av \
  --input /data/mv_i2v_wsm.json \
  --negative-prompt-json recipes/cosmos3/negative_prompt.json \
  --output-dir outputs/mv_i2v_wsm \
  --seed 42 --fps 30 --num-frames 200
```

`--negative-prompt-json` applies the required serialization for you; a
`negative_prompt` string in the input JSON takes precedence over it. Omit both
only for runs where reference parity does not matter.

`--fps` and `--num-frames` override every record, so one input file can be run
at several rates or lengths without editing it. Records may also use the field
names `guidance`, `num_steps`, and `shift` as aliases for `guidance_scale`,
`num_inference_steps`, and `flow_shift`; the vLLM-Omni names win when both are
present.

By default the negative prompt carries the same duration/FPS and resolution
sentences as the positive prompt; set `negative_metadata_mode` in the request's
extra args to change it.

The example writes `vision_viewNN_<camera>.mp4` for all eleven cameras plus
`sample_outputs.json`, including the resolved resolution, aspect ratio, width, and height. Strict Ulysses CP,
CFG parallelism, TP, and HSDP use the existing engine flags; HSDP and TP cannot
be combined. See the offline script's usage examples. Cache-DiT, session state,
LiDAR, camera subsets, and reordered cameras are rejected in v1.

## Verification

Run the CPU contract suite:

```bash
pytest -q \
  tests/diffusion/models/cosmos3/test_multiview_flex_attention.py \
  tests/diffusion/models/cosmos3/test_cosmos3_multiview_pipeline.py \
  tests/diffusion/models/cosmos3/test_cosmos3_transformer.py \
  tests/examples/offline_inference/test_cosmos3_multiview.py \
  tests/model_extras/test_cosmos3_multiview_uploads.py \
  tests/model_extras/test_model_extras.py \
  tests/model_tests/diffusion/test_alignment.py
```

On CUDA, run `test_multiview_recompile.py` and, on supported Blackwell hardware,
`test_multiview_fa4.py` from the same model test directory. These cover all ten warmed
aspect-ratio/resolution geometries and backend parity. Run the existing distributed
multiview tests for the deployment's parallel configuration.

For checkpoint validation, generate 29-frame clips in WSM-only and
vision-conditioned modes for all five ratios at both resolutions with the same
seed and settings. Include explicit overrides as well as automatic detection.
Check all eleven exported videos for camera order, frame count, and exact
dimensions. Follow with a 201-frame 720p portrait generation on sufficient
hardware. Record the backend, GPU topology, steps, cold/warm latency, and peak
memory; no fixed memory or latency target is implied by resolution support.
