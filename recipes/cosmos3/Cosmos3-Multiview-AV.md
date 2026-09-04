# Cosmos3 Multiview-AV

Cosmos3 Multiview-AV generates the eleven fixed MADS camera views in one
bidirectional denoising pass. It uses the regular Cosmos3 Nano architecture and
weights plus camera-major VAE processing and a weight-free sparse attention
mask. The v1 runtime is single-GPU and sequential-CFG; it defaults to PyTorch
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
    "attention_scope": "same_view_or_frame",
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
request pins 35 steps, guidance 6.0, flow shift 10, 93 frames per camera, 10
FPS, and 480p (832×480).

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
  --seed 42
```

`--negative-prompt-json` applies the required serialization for you; a
`negative_prompt` string in the input JSON takes precedence over it. Omit both
only for runs where reference parity does not matter.

The example writes `vision_viewNN_<camera>.mp4` for all eleven cameras plus
`sample_outputs.json`. Sequence parallelism, CFG parallelism, cache-DiT,
session state, LiDAR, camera subsets, and reordered cameras are rejected in v1.

## Verification

Run the CPU contract suite:

```bash
pytest -q \
  tests/diffusion/models/cosmos3/test_multiview_flex_attention.py \
  tests/diffusion/models/cosmos3/test_cosmos3_multiview_pipeline.py \
  tests/diffusion/models/cosmos3/test_cosmos3_transformer.py \
  tests/model_extras/test_model_extras.py \
  tests/model_tests/diffusion/test_alignment.py
```
