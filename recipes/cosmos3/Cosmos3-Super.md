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
disable. `--enable-layerwise-offload` reduces VRAM on smaller GPUs;
`--quantization fp8` (online, no calibration) cuts peak VRAM for 720p video
generation from ~83 GB to ~55 GB per GPU (2-GPU) with BF16-level quality (T2V
composition can shift at the same seed).

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
- **Measured (8x H200 SXM 141 GB, bf16, official 8-GPU config above):**
  - T2V 1280×720, 189 frames, 35 steps, **guardrails off** → **~121 s**
    (n=3, server-side `stage_gen_time_ms` 120.68 / 120.47 / 120.55 s, spread 0.2%)
  - Same shape and seed with **`--tensor-parallel-size 8`** instead of the
    config above → **~131 s** (n=3, 131.23 / 131.35 / 131.18 s, spread 0.13%).
    At 50 steps: recommended config ~166 s (n=1 server-side; wall 168 / 169 /
    168 s across n=3), TP-8 **~181 s** (n=3). So TP-8 is about **9% slower**
    than the documented CFG-parallel x Ulysses x HSDP config on identical
    hardware at both step counts.
  - Every measured cell fits **`latency ~= 14.4 s + steps x per-step`** to
    within 0.3%: per-step is 3.03-3.06 s for the recommended config and
    3.33-3.36 s for TP-8, and the ~14.4 s non-denoise remainder is the same
    in both (the server emits no per-stage breakdown, so its composition is
    not measured here). The whole difference between the two strategies is
    denoise-step rate.
  - Three more 8-GPU strategies at the same shape, 35 steps, n=3 each:
    hybrid `--ulysses-degree 2 --ring-degree 2` (with CFG 2 x HSDP 8) ~124 s;
    `--ulysses-degree 8` without CFG parallelism ~126 s (widening sequence
    parallelism does not pay for serializing the two CFG passes); and the
    recommended config plus `--vae-patch-parallel-size 2` ~121 s (no
    end-to-end gain at this shape). The documented config was the fastest of
    the five measured.
  - T2V 1280×720, 189 frames, 35 steps, **guardrails on** → **~139 s**
    (n=2, 139.12 / 138.91 s) with the model repo's upsampled example prompt,
    plus ~20 s one-time init at startup and 17 GB of extra weights on disk.
    **Guardrail cost is content-dependent**: about +18 s (+15%) on that
    example prompt, but only ~+2 s on randomized benchmark prompts at the
    identical shape, so treat any guardrail overhead figure as
    prompt-specific rather than a general tax.
  - **Determinism is per server instance, not per seed.** Within one running
    server, output is bitwise reproducible at a fixed seed in both guardrail
    modes (three guardrails-off clips sha256-identical, likewise the two
    guardrails-on clips; the two modes produce different bytes, so guardrails
    alter the output rather than only gating it). Across a server restart the
    same seed and config do **not** reproduce the clip: all 189 frames
    differ, median PSNR 28.1 dB against the pre-restart clip, which is about
    the same magnitude as one frame of motion (adjacent-frame PSNR 27.9 dB
    within a single clip). Same-seed comparisons are only valid within one
    server instance.
  - **Memory (8-GPU):** 17.3 GiB per GPU at model load with HSDP shard 8;
    ~43 GiB per GPU resident while serving.
  - On the reference figures: the ~55 s claim is quoted at 50 steps without a
    frame count; at 189 frames (the model card's own example default) 50
    steps measured ~166 s here, and ~55 s would imply a per-step ~3.7x faster
    than anything measured, so it most plausibly describes a much shorter
    clip. The 111.94 s that `inference_benchmarks.md` reports for 8x H200
    vLLM-Omni T2V 720p is a tensor-parallel figure per its own methodology
    note and does not state steps; at the TP-8 per-step rate measured here it
    corresponds to roughly 29-30 steps, which would explain the number
    without either measurement being wrong.
- **Enabling guardrails on this image needs `HF_HUB_DISABLE_XET=1`.** With guardrails on
  and a token that has `nvidia/Cosmos-1.0-Guardrail` access, startup still fails with
  `RuntimeError: Task error: Unable to parse string as hex hash value` from
  `huggingface_hub` `xet_get`. That is
  [huggingface/xet-core#895](https://github.com/huggingface/xet-core/issues/895), which
  reproduces on exactly the pair this image pins: `hf-xet 1.5.1` and
  `huggingface_hub 1.23.0`. The issue was closed 2026-07-28, after this image was pushed
  on 2026-07-20. Setting `HF_HUB_DISABLE_XET=1` downloads the 146-file, 17 GB guardrail
  repo in under two minutes and the server starts normally.
- **Environment:** `vllm/vllm-omni:cosmos3` (`sha256:6d2630c7d637…`), vllm-omni
  `0.25.0rc2.dev62+g9c1b7504b`, torch 2.11.0+cu130, driver 580.126.09, 8x H200 SXM
  (143,771 MiB each) on NVLink full mesh. Startup to `Application startup complete`
  depends on page-cache state: with the weights already in cache, TP-8 started in
  ~90 s and the HSDP-sharded configs clustered at ~280-290 s; first start after a
  node boot (cold cache, 124 GB off disk) took ~610 s. `--init-timeout 1800` was
  never approached.
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

## NPU

### 8× Ascend910 (A2, A3)

#### Environment

- OS: Linux
- Python: 3.10+
- Driver / runtime: Ascend NPU driver with CANN toolkit
- Recommended operator library: **mindie-sd** (Ascend high-performance fused
  operators — enables `adalayernorm` and other fused kernels automatically upon
  installation)
- vLLM version: Match the repository requirements for your checkout
- vLLM-Omni version or commit: Use the commit you are deploying from

A pre-built Docker image is available on
[Docker Hub](https://hub.docker.com/r/vllm/vllm-omni) and
[Quay.io](https://quay.io/ascend/vllm-omni). Ensure the image tag matches your
vLLM-Omni checkout so that NPU-specific code is in sync with the container.

#### Prerequisites

Install the **mindie-sd** operator library to enable Ascend-optimized fused
operators (`adalayernorm`, etc.):

```bash
git clone https://gitcode.com/Ascend/MindIE-SD.git && cd MindIE-SD

# Comment out the tik_ops build step (not needed for this use case)
sed -i 's|^\(\s*\)source ${current_script_dir}/build_tik_ops.sh|\1# source ${current_script_dir}/build_tik_ops.sh|' build/build_ops.sh

python setup.py bdist_wheel
cd dist
pip install mindiesd-*.whl
```

After installation, enable the Laser Attention kernel for significant
long-sequence speedups:

```bash
export MINDIE_SD_FA_TYPE=ascend_laser_attention
```

#### Command

```bash
export MINDIE_SD_FA_TYPE=ascend_laser_attention

vllm serve nvidia/Cosmos3-Super \
  --omni \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 8 \
  --model-class-name Cosmos3OmniDiffusersPipeline \
  --no-guardrails \
  --init-timeout 1800
```

#### Verification

Same requests as the GPU section above — all modes (T2I, T2V, I2V, V2V,
T2VS, I2VS) work identically on NPU. Quick reference with
`--no-guardrails`:

```bash
curl http://localhost:8000/v1/models

# T2I (1024x1024, 50 steps)
curl -sS -X POST http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Cosmos3-Super",
    "prompt": "A photorealistic red sports car on a city street at golden hour, cinematic lighting.",
    "size": "1024x1024", "n": 1, "response_format": "b64_json",
    "num_inference_steps": 50, "guidance_scale": 7.0, "seed": 42
  }' | python3 -c "import sys,json,base64; open('cosmos3_super_t2i.png','wb').write(base64.b64decode(json.load(sys.stdin)['data'][0]['b64_json']))"

# T2V (1280×720, 189 frames, 35 steps — official params)
curl -sS -X POST http://localhost:8000/v1/videos/sync -H "Accept: video/mp4" \
  -F "model=nvidia/Cosmos3-Super" -F "prompt=A robot arm is cleaning a plate in the kitchen" \
  -F "size=1280x720" -F "num_frames=189" -F "fps=24" -F "num_inference_steps=35" \
  -F "guidance_scale=6.0" -F "max_sequence_length=4096" -F "flow_shift=10.0" \
  -F 'extra_params={"use_resolution_template":false,"use_duration_template":false,"guardrails":false}' \
  -F "seed=17" -o cosmos3_super_t2v.mp4

# I2V — add an uploaded reference image
curl -sS -X POST http://localhost:8000/v1/videos/sync -H "Accept: video/mp4" \
  -F "model=nvidia/Cosmos3-Super" -F "prompt=The scene comes to life with smooth, natural motion." \
  -F "size=1280x720" -F "num_frames=189" -F "fps=24" -F "num_inference_steps=35" \
  -F "guidance_scale=6.0" -F "max_sequence_length=4096" -F "flow_shift=10.0" \
  -F 'extra_params={"use_resolution_template":false,"use_duration_template":false,"guardrails":false}' \
  -F "seed=1111" -F "input_reference=@/path/to/reference.jpg;type=image/jpeg" \
  -o cosmos3_super_i2v.mp4

# V2V — add an uploaded reference video. condition_video_keep can be "first" or "last".
curl -sS -X POST http://localhost:8000/v1/videos/sync -H "Accept: video/mp4" \
  -F "model=nvidia/Cosmos3-Super" -F "prompt=Continue the same scene with smooth natural motion." \
  -F "size=1280x720" -F "num_frames=189" -F "fps=24" -F "num_inference_steps=35" \
  -F "guidance_scale=6.0" -F "max_sequence_length=4096" -F "flow_shift=10.0" \
  -F 'extra_params={"condition_frame_indexes_vision":[0,1],"condition_video_keep":"first","guardrails":false}' \
  -F "seed=2222" -F "input_reference=@/path/to/reference.mp4;type=video/mp4" \
  -o cosmos3_super_v2v.mp4

# T2V + sound — add generate_sound/sound_duration (output muxes AAC 48 kHz stereo)
curl -sS -X POST http://localhost:8000/v1/videos/sync -H "Accept: video/mp4" \
  -F "model=nvidia/Cosmos3-Super" -F "prompt=A robot arm is cleaning a plate in the kitchen" \
  -F "size=1280x720" -F "num_frames=189" -F "fps=24" -F "num_inference_steps=35" \
  -F "guidance_scale=6.0" -F "max_sequence_length=4096" -F "flow_shift=10.0" \
  -F "generate_sound=true" -F "sound_duration=7.875" \
  -F 'extra_params={"use_resolution_template":false,"use_duration_template":false,"guardrails":false}' \
  -F "seed=17" -o cosmos3_super_t2vs.mp4

# I2V + sound — reference image with synchronized audio
curl -sS -X POST http://localhost:8000/v1/videos/sync -H "Accept: video/mp4" \
  -F "model=nvidia/Cosmos3-Super" -F "prompt=The scene comes to life with smooth, natural motion and ambient sound." \
  -F "size=1280x720" -F "num_frames=189" -F "fps=24" -F "num_inference_steps=35" \
  -F "guidance_scale=6.0" -F "max_sequence_length=4096" -F "flow_shift=10.0" \
  -F "generate_sound=true" -F "sound_duration=7.875" \
  -F 'extra_params={"use_resolution_template":false,"use_duration_template":false,"guardrails":false}' \
  -F "seed=1111" -F "input_reference=@/path/to/reference.jpg;type=image/jpeg" \
  -o cosmos3_super_i2vs.mp4
```

#### Notes

- **Parallelism:** `--tensor-parallel-size 8` matches the model's 8 KV heads (`num_key_value_heads: 8`, `num_attention_heads: 64` → GQA). This uses 8 davinci devices (NPU 0–3, both cores each). The remaining NPU 4–7 stay idle.
- **Model config:** 64 layers × 5120 hidden size × 64 attention heads, ~120 GB transformer on disk (27 shards). Each NPU device loads ~15 GB of model weights.
- **Peak HBM:** ~22.7 GB per device at startup (bf16 weights). Additional memory is allocated per-generation for KV cache and activations — resolution and frame count drive the peak.
- **Performance (verified on 8× Ascend910):** T2I 256² / 2 steps / guidance 1.0 → ~1.5 s.
- **Guardrails** are disabled with `--no-guardrails` (guards are on by default). The gated `nvidia/Cosmos-1.0-Guardrail` model and `cosmos-guardrail` package are not shipped. Add `guardrails: false` in `extra_params` for per-request overrides when the server has guardrails enabled.
- **Known limitations:** FP8 quantization not yet validated on Ascend NPU. `--enable-layerwise-offload` is available but untested on NPU.
