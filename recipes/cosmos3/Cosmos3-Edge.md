# Cosmos3-Edge

> Compact Nemotron-based Cosmos3 checkpoint for T2I / T2V / I2V and action-conditioned (Physical-AI) generation.

## Summary

- Vendor: NVIDIA
- Model: `nvidia/Cosmos3-Edge`
- Task: T2I, T2V, I2V, and action-conditioned generation (Physical-AI: `policy` / `forward_dynamics` / `inverse_dynamics`)
- Mode: Online (OpenAI-compatible image/video APIs) plus offline generation
- Maintainer: Community

## When to use

The small Cosmos3 checkpoint, for constrained deployments. It runs on the shared
`Cosmos3OmniDiffusersPipeline` but differs from Nano/Super in two ways:

- **Transformer**: `Cosmos3EdgeVFMTransformer` (Nemotron-based), auto-selected from
  `backbone_type: cosmos3_edge_nemotron_dense` — nothing to pass.
- **Defaults** differ; omit `--height` / `--width` to get them:

| | Edge | Nano / Super |
|---|---|---|
| T2I | 640×640 | 1024² |
| T2V / I2V | 480×832 | 1280×720 |
| Video `guidance_scale` | 5.0 | 6.0 |
| Video `flow_shift` | 3.0 | 10.0 |

For the larger checkpoints see [Cosmos3-Nano](./Cosmos3-Nano.md) and [Cosmos3-Super](./Cosmos3-Super.md).

## Support notes

- **Rejected** (raise at request time): video-to-video, transfer V2V
  (`edge` / `blur` / `depth` / `seg` / `wsm`), sound (`generate_sound=true`).
- **Acceleration**: `--quantization fp8` and attention-backend selection are supported. Cache-DiT and
  Sequence Parallel are inherited from the base Cosmos3 transformer (Edge overrides neither) and act on
  the shared GEN pathway — not separately benchmarked here. Tensor Parallel is untested on this checkpoint.

## Usage

Guardrails are on by default (gated `nvidia/Cosmos-1.0-Guardrail`; `pip install cosmos-guardrail` +
an `HF_TOKEN`). The examples pass `"guardrails": false` for a quick local run — you own license
compliance.

### Offline

```bash
# Text-to-image -> 640x640
python examples/offline_inference/text_to_image/text_to_image.py \
  --model nvidia/Cosmos3-Edge \
  --prompt "A photorealistic red sports car at golden hour, cinematic lighting." \
  --extra-body '{"guardrails": false}' --output t2i.png

# Text-to-video -> 480x832  (for I2V: use image_to_video.py + --image <ref>)
python examples/offline_inference/text_to_video/text_to_video.py \
  --model nvidia/Cosmos3-Edge \
  --prompt "A robot arm is cleaning a plate in the kitchen." \
  --num-frames 49 \
  --extra-body '{"max_sequence_length": 4096, "guardrails": false}' --output t2v.mp4
```

### Online

```bash
vllm serve nvidia/Cosmos3-Edge --omni --host 0.0.0.0 --port 8000 --init-timeout 1800
```

Add `--no-guardrails` to skip the safety checker. Request formats match the
[Cosmos3-Nano recipe](./Cosmos3-Nano.md#verification), minus the rejected modes above.

### Action (Physical-AI)

Action-conditioned generation uses the same shared pipeline — pass
`extra_params={"action_mode": ...}`:

- `forward_dynamics` — first frame/video **plus** an action trajectory → roll out the video
  (sync `POST /v1/videos/sync`).
- `policy` — first frame/video + a language instruction → **predict** the action trajectory
  (async `POST /v1/videos`; read the top-level `action` field).
- `inverse_dynamics` — a video → **recover** the action trajectory (async `POST /v1/videos`).

See the [Cosmos3-Nano action section](./Cosmos3-Nano.md) for request/response shapes. For a
DROID-specialized policy model, use `nvidia/Cosmos3-Edge-Policy-DROID`.

## Verification

```bash
python -c "from PIL import Image; im=Image.open('t2i.png'); print('image', im.size, im.mode)"
ffprobe -v error -select_streams v:0 -show_entries stream=codec_type,nb_frames,width,height -of csv=p=0 t2v.mp4
```

Expected: `image (640, 640) RGB` and `video,49,832,480`.

## References

- Model card: <https://huggingface.co/nvidia/Cosmos3-Edge>
- Pipeline: [`pipeline_cosmos3.py`](../../vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py)
- Edge transformer: [`transformer_cosmos3_edge.py`](../../vllm_omni/diffusion/models/cosmos3/transformer_cosmos3_edge.py)
