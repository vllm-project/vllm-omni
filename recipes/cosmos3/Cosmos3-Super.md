# Cosmos3-Super

> Frontier 64B world model: text-to-image, text-to-video, image-to-video, video-to-video (+ optional audio)

## Summary

- Vendor: NVIDIA
- Model: `nvidia/Cosmos3-Super` (64B; also `Cosmos3-Super-Text2Image`, `Cosmos3-Super-Image2Video`)
- Task: T2I, T2V, I2V, V2V generation, with optional synchronized audio (video + sound)
- Mode: Online serving with the OpenAI-compatible image/video APIs
- Maintainer: Community

## When to use this recipe

Use this recipe to deploy the 64B `nvidia/Cosmos3-Super` for the highest-quality
Cosmos3 generation. It shares the same `Cosmos3OmniDiffusersPipeline` and request
formats as [Cosmos3-Nano](./Cosmos3-Nano.md) — only the checkpoint size and the
recommended parallelism differ. Mode is selected per request (T2I →
`/v1/images/generations`; T2V/I2V/V2V → `/v1/videos/sync`; add
`generate_sound=true` for audio).

## References

- Model card (authoritative usage + example assets): <https://huggingface.co/nvidia/Cosmos3-Super>
- Nano recipe (same APIs/params): [`Cosmos3-Nano.md`](./Cosmos3-Nano.md)
- Pipeline: [`vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py`](../../vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py)

## Hardware Support

## GPU

Requires the `vllm-omni` package (or the `vllm/vllm-omni:cosmos3` container),
which provides the `vllm serve … --omni` entrypoint used below.

### 8x H200/H100/A100 (recommended, per model card)

```bash
vllm serve nvidia/Cosmos3-Super \
  --omni \
  --host 0.0.0.0 --port 8000 \
  --cfg-parallel-size 2 \
  --ulysses-degree 4 \
  --use-hsdp --hsdp-shard-size 8 \
  --init-timeout 1800
```

### 2x H200 / B300 (minimum)

```bash
vllm serve nvidia/Cosmos3-Super \
  --omni \
  --host 0.0.0.0 --port 8000 \
  --cfg-parallel-size 2 \
  --use-hsdp --hsdp-shard-size 2 \
  --init-timeout 1800
```

Guardrails are on by default (gated `nvidia/Cosmos-1.0-Guardrail` — `pip install
cosmos-guardrail`, accept the license, set `HF_TOKEN`); add `--no-guardrails` to
disable. `--enable-layerwise-offload` reduces VRAM on smaller GPUs.

#### Verification

Requests are identical to Nano (see [`Cosmos3-Nano.md`](./Cosmos3-Nano.md) for full
T2I/T2V/I2V/V2V/T2VS curls); official params: `size=1280x720, num_frames=189,
fps=24, num_inference_steps=35, guidance_scale=6.0, flow_shift=10.0,
max_sequence_length=4096`.

```bash
curl http://localhost:8000/v1/models
# T2V (official prompt assets give best quality)
curl -sS -X POST http://localhost:8000/v1/videos/sync -H "Accept: video/mp4" \
  -F "model=nvidia/Cosmos3-Super" -F "prompt=A robot arm is cleaning a plate in the kitchen" \
  -F "size=1280x720" -F "num_frames=189" -F "fps=24" -F "num_inference_steps=35" \
  -F "guidance_scale=6.0" -F "max_sequence_length=4096" -F "flow_shift=10.0" \
  -F 'extra_params={"use_resolution_template":false,"use_duration_template":false,"guardrails":true}' \
  -F "seed=17" -o cosmos3_super_t2v.mp4

# I2V — add an uploaded reference image
curl -sS -X POST http://localhost:8000/v1/videos/sync -H "Accept: video/mp4" \
  -F "model=nvidia/Cosmos3-Super" -F "prompt=The scene comes to life with smooth, natural motion." \
  -F "size=1280x720" -F "num_frames=189" -F "fps=24" -F "num_inference_steps=35" \
  -F "guidance_scale=6.0" -F "max_sequence_length=4096" -F "flow_shift=10.0" \
  -F 'extra_params={"use_resolution_template":false,"use_duration_template":false,"guardrails":true}' \
  -F "seed=1111" -F "input_reference=@/path/to/reference.jpg;type=image/jpeg" \
  -o cosmos3_super_i2v.mp4

# V2V — add an uploaded reference video. condition_video_keep can be "first" or "last".
curl -sS -X POST http://localhost:8000/v1/videos/sync -H "Accept: video/mp4" \
  -F "model=nvidia/Cosmos3-Super" -F "prompt=Continue the same scene with smooth natural motion." \
  -F "size=1280x720" -F "num_frames=189" -F "fps=24" -F "num_inference_steps=35" \
  -F "guidance_scale=6.0" -F "max_sequence_length=4096" -F "flow_shift=10.0" \
  -F 'extra_params={"condition_frame_indexes_vision":[0,1],"condition_video_keep":"first"}' \
  -F "seed=2222" -F "input_reference=@/path/to/reference.mp4;type=video/mp4" \
  -o cosmos3_super_v2v.mp4

# T2V + sound — add generate_sound/sound_duration (output muxes AAC 48 kHz stereo)
curl -sS -X POST http://localhost:8000/v1/videos/sync -H "Accept: video/mp4" \
  -F "model=nvidia/Cosmos3-Super" -F "prompt=A robot arm is cleaning a plate in the kitchen" \
  -F "size=1280x720" -F "num_frames=189" -F "fps=24" -F "num_inference_steps=35" \
  -F "guidance_scale=6.0" -F "max_sequence_length=4096" -F "flow_shift=10.0" \
  -F "generate_sound=true" -F "sound_duration=7.875" \
  -F 'extra_params={"use_resolution_template":false,"use_duration_template":false,"guardrails":true}' \
  -F "seed=17" -o cosmos3_super_t2vs.mp4
```

#### Notes

- **Measured (2x B300, bf16, guardrails off, official 2-GPU config above):**
  - T2I 1024², 50 steps → **~6 s**
  - T2V 1280×720, 189 frames, 35 steps → **~197 s**
  - I2V 1280×720, 189 frames, 35 steps → **~200 s**
  - T2V + sound (189 frames, 35 steps) → **~198 s**, output muxes **AAC 48 kHz stereo**
  - (NVIDIA's reference: 8×H200 @ 50 steps ≈ 55 s/video; 2×H200 @ 35 steps ≈ 3 min/video.)
- **Memory:** ~61.5 GiB per GPU when sharded across 2 GPUs (HSDP shard 2); repo ~135 GB on disk.
- Same generation defaults, supported sizes, V2V reference-video controls
  (`condition_frame_indexes_vision`, `condition_video_keep`), and
  `generate_sound`/`sound_duration`
  semantics as Nano, including the **action** modality: `forward_dynamics`,
  `policy`, and `inverse_dynamics` — see the Cosmos3-Nano recipe for the request
  shapes. Use async `/v1/videos` when you need predicted/recovered action metadata
  under the top-level `action` field. Verified on the 64B Super under
  `--cfg-parallel-size 2`: async `policy` returns the predicted action (`[16, 10]`)
  and the rollout video reliably.
