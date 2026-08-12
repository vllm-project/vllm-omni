# Cosmos3 First-Last Boundary to Video (FL2V) — vLLM-Omni

Generate a Cosmos3 clip pinned to a **start** and **end** boundary, with the
middle filled from an event prompt. Each boundary may be a **single image** or a
**short video clip**, auto-detected from the file extension, so one script covers
frame+frame, clip+clip, and either mixed. Served over HTTP via a warm vLLM-Omni
server.

**Prefer `nvidia/Cosmos3-Super` for best video quality.** **`nvidia/Cosmos3-Nano`
also works** on the same client (verified). Serve the model you want, then pass
the same id to `--model`.

Scripts, the required vLLM patch, and demo assets live in
`tools/cosmos3_fl2v/`.
`fl2v_generate_vllm.py` is a plain HTTP client for `POST /v1/videos/sync` — no
vLLM-Omni source changes and no weight changes. FL2V is expressed entirely with
documented request fields plus `extra_params`.

Demo package: `testdata/fl2v_from_cosmos_v2v/` (robot pouring; mid-clip diversion
into the left jar, then back to the right glass), self-contained in the
`tools/cosmos3_fl2v/` directory.

## Summary

- Vendor: NVIDIA
- Model: `nvidia/Cosmos3-Super` (recommended) or `nvidia/Cosmos3-Nano`
- Task: First-last boundary to video (FL2V) over the V2V API
- Mode: Online serving (`POST /v1/videos/sync`) with a warm vLLM-Omni server
- Maintainer: Community

---

## Setup

One GPU node with enough free VRAM:

| Model | GPU | Notes |
|---|---|---|
| **Super** (recommended) | **B200** or **H200** | ~125 GB weights |
| **Nano** (also works) | **H100** | ~46 GB VRAM |

Both models are gated — accept the license on Hugging Face, then set `HF_TOKEN`
to a token that has access.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh      # once
export PATH="$HOME/.local/bin:$PATH"
export HF_TOKEN=hf_...                               # your Hugging Face token

cd tools/cosmos3_fl2v
uv venv --python 3.12 --seed --managed-python .venv-client
source .venv-client/bin/activate
uv pip install vllm==0.26.0 vllm-omni==0.26.0
uv pip install -r requirements.txt
python patch_vllm_shm.py                             # required, see below
```

Optional — keep large downloads off `$HOME` (weights need tens of GB free):

```bash
export HF_HOME=/path/to/scratch/.cache/hf
export UV_CACHE_DIR=/path/to/scratch/.cache/uv
export UV_PYTHON_INSTALL_DIR=/path/to/scratch/.cache/uv-python
```

Check the client without a GPU: `python fl2v_generate_vllm.py --dry-run` should
print `latent_frames=48`, `conditioning latents [0, 47]`, and `reference video
frames=189`.

### Required vLLM patch

`patch_vllm_shm.py` fixes an upstream vLLM deadlock on large request payloads.
FL2V reference videos are big enough to trip it; ordinary Cosmos3 requests are
not, so it is unpatched upstream. **Without it the server hangs silently** — no
error, no timeout, 0% GPU. The script is idempotent, backs up what it edits, and
supports `--check` and `--revert`; its docstring has the full analysis.

It edits site-packages, so run it in the environment that hosts the **server**,
and re-run it after any vLLM reinstall or upgrade. Verified against vLLM 0.26.0.

---

## Serve the model

In a second shell (or `tmux` — this stays in the foreground). Activate the same
venv and re-export `HF_TOKEN` (and `HF_HOME` if you set it), then:

```bash
source .venv-client/bin/activate
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=3600   # default 600 s returns 504 mid-job

# Super (recommended; B200 / H200)
vllm-omni serve nvidia/Cosmos3-Super \
  --omni \
  --port 8000 \
  --init-timeout 1800 \
  --no-guardrails

# Nano on H100 (~46 GB VRAM) — same flags, swap the id:
# vllm-omni serve nvidia/Cosmos3-Nano \
#   --omni --port 8000 --init-timeout 1800 --no-guardrails
```

Wait for `Application startup complete`; a cold start spends most of
`--init-timeout` downloading weights. Confirm with
`curl -s http://127.0.0.1:8000/health` before launching a long job.

**The client `--model` must match the served id.** Defaults assume Super. On H100,
serve Nano and pass `--model nvidia/Cosmos3-Nano`.

Use the `vllm-omni` command (not plain `vllm`). Do **not** pass `--enforce-eager`
(it slows sampling and can push jobs past the sync timeout). Add
`--host 0.0.0.0` only to reach the server from another machine. To keep
guardrails, drop `--no-guardrails`, install `cosmos-guardrail`, and accept the
gated `nvidia/Cosmos-1.0-Guardrail` repo.

---

## Run

Run from `tools/cosmos3_fl2v/`. `--url` defaults to
`http://localhost:8000`, and every input path below is also a default, so a bare
`python fl2v_generate_vllm.py` runs the frame+frame demo (against a **Super**
server). Client defaults: `--steps 16 --guidance 8 --flow-shift 15`. Send **one
request at a time**.

On H100: serve Nano, then add `--model nvidia/Cosmos3-Nano` to every client
command below.

```bash
A=testdata/fl2v_from_cosmos_v2v/assets
O=testdata/fl2v_from_cosmos_v2v/outputs

# frame + frame
python fl2v_generate_vllm.py \
  --start $A/seed_start.png --end $A/seed_end.png \
  --output $O/generated_jar_diversion_framecond_vllm.mp4

# clip + clip
python fl2v_generate_vllm.py \
  --start $A/seed_start_clip.mp4 --end $A/seed_end_clip.mp4 \
  --head-frames 9 --tail-frames 9 \
  --output $O/generated_jar_diversion_clipcond_vllm.mp4

# mixed: frame start, clip end
python fl2v_generate_vllm.py \
  --start $A/seed_start.png --end $A/seed_end_clip.mp4 \
  --tail-frames 9 \
  --output $O/generated_jar_diversion_frame_start_clip_end_vllm.mp4

# mixed: clip start, frame end
python fl2v_generate_vllm.py \
  --start $A/seed_start_clip.mp4 --end $A/seed_end.png \
  --head-frames 9 \
  --output $O/generated_jar_diversion_clip_start_frame_end_vllm.mp4
```

With your own files, add `--prompt event.json --negative negative.json` (plain
`.txt` also works). The script echoes resolved paths, the latent timeline, the
chosen conditioning indexes, and the `extra_params` before doing any work, then
start/end PSNR vs boundary after generating.

The frame-start / clip-end run above is the controlled comparison against
frame+frame: the start is identical (same still, same locked latent `[0]`, same
seed), so only the tail differs. Leave `--head-frames` at its default or that no
longer holds. Clip-start / frame-end is the symmetric case for the head.

`--head-frames` / `--tail-frames` mean still-repeat count in frame mode, or how
many frames to take from the start/end of the video in clip mode. For clips prefer
VAE-aligned lengths `k*4+1` (1, 5, 9, 13, …); other lengths are snapped down with
a log line. `num_frames` must satisfy `(num_frames - 1) % 4 == 0`, e.g. 189, 93, 49.

---

## How FL2V works here

Pixel length `T` maps to latents with temporal factor 4: `T_z = (T-1)/4 + 1`, so
`T=189` gives `T_z=48`. Stock Cosmos3 V2V locks a clean **prefix** of latents
(`[0, 1]`); FL2V locks a contiguous range at **each end**.

- **Frame mode:** one latent per end, e.g. `[0, 47]`.
- **Clip mode:** every latent the boundary clip covers, e.g. `[0, 1, 2]` and
  `[45, 46, 47]` for 9-frame boundaries.

Three tricks make that work; on this runtime only one is ours:

| FL2V trick | vLLM-Omni |
|---|---|
| Lock start+end latent ranges instead of prefix `[0, 1]` | **native** — `extra_params.condition_frame_indexes_vision` accepts any indexes with `max(index) < T_lat` |
| Clean head/tail for the causal VAE | **ours** — we build and upload the reference MP4 |
| Re-inject clean boundary latents every UniPC step | **native** — the V2V denoise loop already runs `velocity_mask * latents + (1 - velocity_mask) * condition_latents` |

Trick 2 is ours because the Wan VAE is temporally causal: each latent summarizes a
*chunk* of pixel frames, so the reference video must not mix unrelated content
into a locked chunk. The script uploads `[start head][filler…][end tail]` —
repeated stills in frame mode, real clip frames in clip mode. Filler is
irrelevant; those latents become noise. The server encodes only the first
`max(indexes) * 4 + 1` frames, slicing contiguously without fps resampling.

Boundary pixels *become* the locked latents, so the reference is written
near-losslessly (`libx264`, `yuv444p`, `-qp 0`). Passing `--pix-fmt yuv420p`
costs a few dB on both boundaries; use it only if your decoder rejects 4:4:4.

To re-verify the two **native** rows against the commit you deploy, look for
`_prepare_latents_v2v` and the `velocity_mask` line in
`vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py`.

---

## Verified run

FL2V is verified with both models on this path:

- **Nano** on 1×H100 80GB (`vllm` + `vllm-omni` 0.26.0 with `patch_vllm_shm.py`,
  `--no-guardrails`, 189 / 720p / seed 1234). Early lock checks used **35 steps**;
  today's client defaults are **16 / 8 / 15**.
- **Super** on B200 / H200 (`vllm` + `vllm-omni` 0.26.0 with `patch_vllm_shm.py`,
  `--no-guardrails`, 189 / 720p / seed 1234, defaults **16 / 8 / 15**). Same
  client — `--model nvidia/Cosmos3-Super` (the default). Prefer Super for best
  quality.

Both boundaries lock, and the mid-clip event is generated rather than
interpolated between the seeds. Startup ~3 min once; the server stays warm and
logs `Total pipeline time` per request.

---

## Troubleshooting

FL2V requests are long and large, so they hit limits ordinary Cosmos3 requests
often never see:

- **`HTTP 400` model mismatch.** Client `--model` does not match the served id.
  Align them (Super with Super, or Nano with `--model nvidia/Cosmos3-Nano`).
- **`504 Gateway Timeout` after the server logs `Total pipeline time`.** The clip
  generated fine and was discarded. Raise `VLLM_OMNI_VIDEO_SYNC_TIMEOUT` (default
  600 s) and drop `--enforce-eager`. The client's `--timeout` applies
  independently, so the shorter limit wins.
- **Hangs forever at 0% GPU with no error.** Apply the
  [required vLLM patch](#required-vllm-patch).
- **Server has no `/v1/videos` endpoints.** You started plain `vllm` instead of
  `vllm-omni`.

---

## CLI reference

| Flag | Default | Meaning |
|---|---|---|
| `--url` | `http://localhost:8000` | vLLM-Omni base URL |
| `--model` | `nvidia/Cosmos3-Super` | Served model (`Super` recommended; `nvidia/Cosmos3-Nano` also works) |
| `--start` / `--end` | package seed PNGs | Boundary image **or** video (auto-detected by extension) |
| `--prompt` / `--negative` | package JSON | Event / negative prompts |
| `--package` | `testdata/fl2v_from_cosmos_v2v` | Demo defaults root |
| `--head-frames` / `--tail-frames` | `5` / `9` | Frame mode: still repeats. Clip mode: frames taken from start/end of each video |
| `--num-frames` / `--fps` | `189` / `24` | Temporal length and export rate |
| `--height` / `--width` | `720` / `1280` | Sent as `size=WxH` |
| `--steps` / `--guidance` | `16` / `8` | Recommended Super recipe |
| `--flow-shift` / `--max-sequence-length` | `15.0` / `4096` | |
| `--seed` | `1234` | |
| `--codec` / `--pix-fmt` / `--qp` | `libx264` / `yuv444p` / `0` | Reference-video encode |
| `--guardrails` | off | Safety checker (server must have launched with it) |
| `--resolution-template` | off | Server-side resolution/duration templates |
| `--keep-reference` | none | Save the conditioning video for debugging |
| `--dry-run` | off | Build + print the request, do not POST |
| `--timeout` | `1800` | HTTP timeout in seconds |
| `--output` | `<package>/outputs/generated_jar_diversion_framecond_vllm.mp4` | Output mp4 |
