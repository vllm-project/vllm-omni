# SkyReels V2 T2V (vLLM-Omni)

Use the shared text-to-video runner. The `skyreels_v2` preset is selected
automatically from the model ID (`SkyReelsV2Pipeline` / `skyreels` in the name).

## Models

- `Skywork/SkyReels-V2-T2V-14B-540P-Diffusers` (preset: 544x960, 97 frames)
- `Skywork/SkyReels-V2-T2V-14B-720P-Diffusers` (override with `--height 720 --width 1280`)

## Run

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model Skywork/SkyReels-V2-T2V-14B-540P-Diffusers \
  --prompt "A cat and a dog baking a cake together in a kitchen."
```

Preset defaults: `flow_shift=8.0`, `guidance_scale=6.0`, `num_inference_steps=50`, `fps=24`.

Smoke (fewer frames/steps):

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model Skywork/SkyReels-V2-T2V-14B-540P-Diffusers \
  --num-frames 33 --num-inference-steps 20
```

Online serve docs: `examples/online_serving/text_to_video/README.md` (SkyReels V2 section).
