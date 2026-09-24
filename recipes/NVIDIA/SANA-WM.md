# SANA-WM

> Camera-controllable first-frame image-to-video world model.

## Summary

- Vendor: Efficient-Large-Model / NVlabs SANA
- Model: `BBBBruce/SANA-WM_bidirectional-stage1-diffusers` (standard diffusers layout, converted offline from the NVlabs release; Stage-1 transformer + VAE only)
- Task: First-frame image-to-video generation with camera control
- Mode: Online serving with the OpenAI-compatible video API
- Model weights: about 13 GB for the Stage-1 transformer (10 GB) and VAE (2.3 GB)
- Local disk: reserve about 40 GB for the Hugging Face cache and runtime artifacts
- Recommended GPU: 24 GB or larger CUDA GPU
- Maintainer: Community

## When to use this recipe

Use this recipe when you want to serve SANA-WM through `/v1/videos` or
`/v1/videos/sync`. The model takes a text prompt, a first-frame image, and
either an action DSL string or explicit camera poses. vLLM-Omni serves the
SANA-WM Stage-1 DiT, decoded through the SANA VAE. The optional LTX-2 refiner
stage is not supported by this integration; it is a planned follow-up.

## References

- Upstream model card: <https://huggingface.co/Efficient-Large-Model/SANA-WM_bidirectional>
- Video API: [`docs/serving/videos_api.md`](../../docs/serving/videos_api.md)

## Hardware Support

## GPU

### 1x NVIDIA RTX PRO 6000 Blackwell 96GB

#### Capacity

- Model storage: the Stage-1 transformer is about 10 GB and the VAE about
  2.3 GB. The Gemma text encoder is a separate 4.9 GB repo. The engine prefetches
  the whole model repo at startup (`allow_patterns=["*"]`), which is why the
  Stage-1 weights live in their own repo — the two-stage one carries an
  additional 84 GB `refiner/` that this path never loads.
- Text encoder: the pipeline tries `google/gemma-2-2b-it` first, then falls back
  to the ungated mirror `Efficient-Large-Model/gemma-2-2b-it`. The first repo is
  gated, so without an accepted licence and `hf auth login` you will see one
  failed load before the fallback succeeds — that is expected, not an error to
  chase. Set `VLLM_OMNI_SANA_WM_STAGE1_TEXT_ENCODER` to pin a specific repo or a
  local path and skip both.
- Disk sizing: provision about 40 GB of local disk or Hugging Face cache volume
  so the model, temporary downloads, and generated artifacts fit without cache
  eviction.
- GPU sizing: the default 1280x704, 161-frame, 60-step serving profile peaks at
  22.6 GB of device memory and takes about 133 s to generate on one RTX PRO
  6000 Blackwell. The peak lands in the VAE decode, which is why the pipeline
  forces VAE tiling on regardless of `vae_use_tiling` — without it the same
  request costs about 9 GB more, and a 321-frame one OOMs outright. On smaller
  GPUs, lower `width`, `height`, or `num_frames` before serving production
  requests.

#### Environment

- OS: Linux
- Python: 3.10+
- Driver / runtime: NVIDIA driver with CUDA runtime supported by your PyTorch
  build
- Recommended operator library: Triton, installed through the vLLM/vLLM-Omni
  Python environment
- vLLM version: Match the repository requirements for your checkout
- vLLM-Omni version or commit: Use the commit you are deploying from

#### Command

The repo ships the standard Diffusers layout (`model_index.json` +
`transformer/`, `vae/`), and its `model_index.json` names `SanaWmPipeline`, so
the pipeline class resolves on its own.

```bash
CUDA_VISIBLE_DEVICES=0 \
vllm serve BBBBruce/SANA-WM_bidirectional-stage1-diffusers \
  --omni \
  --host 0.0.0.0 \
  --port 8091
```

No deploy config: single-stage diffusion models are deliberately absent from
`OMNI_PIPELINES` (`vllm_omni/config/pipeline_registry.py`), so stage resolution
falls back to the default stage config and a YAML's stage settings — including
`default_sampling_params` — would not be applied. The production generation
settings therefore live in the model (`num_inference_steps=60`,
`guidance_scale=5.0`); a request that omits a field gets them. The examples
below still pass every field explicitly so the numbers are visible.

If you point this at the older two-stage repo
(`BBBBruce/SANA-WM_bidirectional-diffusers`), startup fails with `Model class
SanaWmTwoStagesPipeline not found in diffusion model registry`, because that
repo's `model_index.json` names a class this build does not register. Add
`--model-class-name SanaWmPipeline` to override it.

#### Verification

Use a short smoke request first:

```bash
curl -sS -X POST http://localhost:8091/v1/videos/sync \
  -H "Accept: video/mp4" \
  -F "prompt=A slow forward camera move through a quiet city street." \
  -F "negative_prompt=blurry, low quality, distorted, watermark" \
  -F "input_reference=@/path/to/first_frame.png;type=image/png" \
  -F "width=1280" \
  -F "height=704" \
  -F "num_frames=9" \
  -F "fps=16" \
  -F "num_inference_steps=2" \
  -F "guidance_scale=5.0" \
  -F "seed=42" \
  --form-string 'extra_params={"sana_wm":{"action":"w-8","translation_speed":0.055,"rotation_speed_deg":1.2,"intrinsics":{"fx":640,"fy":640,"cx":640,"cy":352}}}' \
  -o sana_wm_smoke.mp4
```

For a production-length request, note that the action durations must sum to
`num_frames - 1` — the rollout includes the identity start pose — and a
mismatch is rejected rather than padded or truncated:

```bash
curl -sS -X POST http://localhost:8091/v1/videos/sync \
  -H "Accept: video/mp4" \
  -F "prompt=A slow forward camera move through a quiet city street." \
  -F "negative_prompt=blurry, low quality, distorted, watermark" \
  -F "input_reference=@/path/to/first_frame.png;type=image/png" \
  -F "width=1280" \
  -F "height=704" \
  -F "num_frames=161" \
  -F "fps=16" \
  -F "num_inference_steps=60" \
  -F "guidance_scale=5.0" \
  -F "seed=42" \
  --form-string 'extra_params={"sana_wm":{"action":"w-160","translation_speed":0.055,"rotation_speed_deg":1.2,"intrinsics":{"fx":640,"fy":640,"cx":640,"cy":352}}}' \
  -o sana_wm_output.mp4
```

Use `POST /v1/videos` instead when you want job storage and polling rather than
inline MP4 bytes. It accepts the same form fields as `/v1/videos/sync`.

```bash
create_response=$(curl -sS -X POST http://localhost:8091/v1/videos \
  -H "Accept: application/json" \
  -F "prompt=A slow forward camera move through a quiet city street." \
  -F "negative_prompt=blurry, low quality, distorted, watermark" \
  -F "input_reference=@/path/to/first_frame.png;type=image/png" \
  -F "width=1280" \
  -F "height=704" \
  -F "num_frames=161" \
  -F "fps=16" \
  -F "num_inference_steps=60" \
  -F "guidance_scale=5.0" \
  -F "seed=42" \
  --form-string 'extra_params={"sana_wm":{"action":"w-160","translation_speed":0.055,"rotation_speed_deg":1.2,"intrinsics":{"fx":640,"fy":640,"cx":640,"cy":352}}}')

video_id=$(echo "$create_response" | jq -r '.id')
curl -sS "http://localhost:8091/v1/videos/${video_id}" | jq .
curl -L "http://localhost:8091/v1/videos/${video_id}/content" -o sana_wm_output.mp4
```

#### Notes

- Sequence parallelism is not supported. The bidirectional gated delta
  recurrence carries state across frames, so a rank cannot denoise a slice
  of the token sequence in isolation; supporting it needs a distributed scan
  or an all-gather before the GDN blocks.

- `input_reference` is required for the first frame. Use `image_reference` only
  when you need a JSON-safe image URL or data URL instead of a multipart upload.
- `sana_wm` must provide exactly one of `action` or `camera`.
- Action strings use comma-separated `<keys>-<duration>` segments. Supported
  keys are `w`, `a`, `s`, `d` for translation and `i`, `j`, `k`, `l` for
  pitch/yaw rotation. The durations must sum to `num_frames - 1`.
- Explicit camera control (alternative to `action`): pass
  `"camera": {"poses": [...]}` where `poses` is a list of `num_frames`
  camera-to-world 4x4 matrices (row-major, OpenCV `+X right, +Y down, +Z forward`
  convention), e.g.
  `extra_params={"sana_wm":{"camera":{"poses":[[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], ...]},"intrinsics":{...}}}`.
  Most callers should prefer `action`; explicit poses exist for callers that
  already have a per-frame trajectory.
- Explicit `intrinsics` are recommended and take the mapping form
  `{"fx":640,"fy":640,"cx":640,"cy":352}` (for 1280x704). This `{fx,fy,cx,cy}`
  mapping is the only accepted intrinsics form; omit `intrinsics` to derive them
  from the output resolution. All four values must be finite, and `fx`/`fy`
  must be positive — the ray map divides by them.
- The video API returns decoded MP4 bytes and has no `output_type` field, so
  raw Stage-1 latents are reachable only from the offline API
  (`OmniDiffusionSamplingParams(output_type="latent")`); see
  [`tests/e2e/offline_inference/test_sana_wm.py`](../../tests/e2e/offline_inference/test_sana_wm.py).
  Putting `output_type` in `extra_params` does not work: it lands in
  `sampling_params.extra_args`, while the pipeline reads the top-level field.
