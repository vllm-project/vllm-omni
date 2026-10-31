# WF-04: MiniMax H3 Multiframe Reference

The [Multiframe Reference template](../example_workflows/vLLM-Omni%20MiniMax-H3%20Multiframe%20Reference.json)
recreates the [official ComfyUI Multiframe Reference workflow](https://github.com/Comfy-Org/workflow_templates/blob/aaac56dd5cc5497533d92cbe50edc35ea660e587/templates/video_minimax_h3_multiframe_reference.json)
([tutorial](https://docs.comfy.org/tutorials/video/minimax/minimax-h3-multiframe))
for a remote vLLM-Omni server. One image is a semantic reference; three more are
timeline guides pinned to 1.5 s, 3.0 s and 5.0 s of a 124-frame, 24 FPS clip
with generated audio. It tracks [RFC #7380, WF-04](https://github.com/vllm-project/vllm-omni/issues/7380).

## Requirements

- A vLLM-Omni server with `/v1/videos` timeline guides
  ([#8838](https://github.com/vllm-project/vllm-omni/pull/8838)) serving base
  MiniMax-H3 Ref2VA weights. Start it from the [MiniMax-H3 recipe](../../../recipes/MiniMaxAI/MiniMax-H3.md)
  with `--task-type ref2va` (or a combined service) and
  `--served-model-name MiniMaxAI/MiniMax-H3`, which the node needs to recognize H3.
- ComfyUI-vLLM-Omni with the **MiniMax H3 Add Guide** node
  ([#9152](https://github.com/vllm-project/vllm-omni/pull/9152)).
- Four images in ComfyUI's `input/` directory. The template uses the official
  sample filenames; select your own files in the four **Load Image** nodes, or
  download the pinned samples:
  [1](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/aaac56dd5cc5497533d92cbe50edc35ea660e587/input/h3_frame_ref_1.png),
  [2](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/aaac56dd5cc5497533d92cbe50edc35ea660e587/input/h3_frame_ref_2.png),
  [3](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/aaac56dd5cc5497533d92cbe50edc35ea660e587/input/h3_frame_ref_3.png),
  [4](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/aaac56dd5cc5497533d92cbe50edc35ea660e587/input/h3_frame_ref_4.png).

## Graph

```text
Load Image 1 ──image_1──► Video References ───────────references──────┐
Load Image 2 ──image──► Add Guide (36) ─┐                              ▼
Load Image 3 ──image──► Add Guide (72) ◄┘ ─┐                     Generate Video ──► Save Video
Load Image 4 ──image──► Add Guide (120) ◄──┘ ──timeline_guides──►     ▲
Diffusion Sampling Params, MiniMax-H3 Video Params ───────────────────┘
```

| Native template | This template | Time |
| --- | --- | --- |
| Reference image 0 (`<Picture 1>`) | Video References `image_1` | Opening composition |
| Add Guide, frame 36 | Add Guide `frame_index=36` | 1.5 s |
| Add Guide, frame 72 | Add Guide `frame_index=72` | 3.0 s |
| Add Guide, frame 120 | Add Guide `frame_index=120` | 5.0 s |

The guides are still images, so each occupies one frame; the `17k+5` length
rule applies only to clip guides. Guides keep chain order (36 → 72 → 120). The
server center-crops each guide to the canvas aspect and scales it to 1344×768,
as the native node does, so the 1368×768 samples lose 12 columns on each side.

Guides are VAE-encoded timeline conditions. In vLLM-Omni, as in native
ComfyUI, they never enter the Qwen3-VL reference presentation and receive no
`<Picture N>` label. The prompt is the official template's prompt, except that
its `<Picture 2>`–`<Picture 4>` tags, which name the guide images, are written
as "the 1.5 s / 3.0 s / 5.0 s guide". Only `<Picture 1>` refers to an image the
encoder sees.

## Settings

| Setting | Value |
| --- | --- |
| Canvas | 1344×768 (16:9), 24 FPS, `duration=5.167` → 124 frames (`17×7+5`) |
| Sampling | 50 steps, `guidance_scale=1`, `true_cfg_scale=1`, seed 738004 (fixed) |
| H3 params | `flow_shift=12`, `audio_flow_shift=3`, `audio_mode=native` |
| Request | Ref2VA, one `input_references` image, three PNG `guide_files`, `quality=lossless` |

The native template samples 20 steps with `res_multistep`; vLLM-Omni uses the
server's H3 scheduler, so the template keeps the base H3 default of 50 steps.

A guided request needs base H3 weights. Generate Video rejects FastH3
Deployment, a request LoRA with a non-zero scale (including Turbo) and Latent
Mask Editing before uploading anything, and sends `quality=lossless` itself.
The server rejects cache acceleration, distilled schedules, continuation and
`audio_mode=lock_source`. Keep **fps** at 24: other values are refused, not
converted.

## Run

1. Start the server and set Generate Video's `url` and `model` to it.
2. Select the four images, then run the workflow. Save Video writes
   `video/MiniMax-H3-Multiframe-Reference_*.mp4` with the generated audio.
3. Check the output:

   ```bash
   ffprobe -v error -count_frames -show_streams -show_format -of json "$OUTPUT"
   ffmpeg -v error -i "$OUTPUT" -f null -
   ```

   Expect H.264 1344×768 at 24 FPS with 124 frames and a stereo AAC track.

## Validation

Recorded on 2026-10-29 with this template unchanged (50 steps, seed 738004)
and the four official sample images.

| Item | Value |
| --- | --- |
| Hardware | 1 × NVIDIA L40S 48 GB, driver 580.159.03 |
| Software | vLLM 0.32.0, vLLM-Omni at `cafd2846a`, PyTorch 2.13.0+cu130, ComfyUI 0.43.0 |
| Server | Single-GPU capacity profile from the recipe, with `--task-type ref2va` |
| ComfyUI | `execution_success` in 1,888 s for Generate Video → Save Video |

The server command was:

```bash
vllm serve /path/to/MiniMax-H3/Ref2VA --omni --trust-remote-code \
  --served-model-name MiniMaxAI/MiniMax-H3 --task-type ref2va --num-gpus 1 \
  --diffusion-offload-config '{"mode":"layer","components":["dit","text_encoder"]}' \
  --enforce-eager --diffusion-attention-backend TORCH_SDPA
```

The saved MP4 is H.264 1344×768 at 24 FPS with 124 decoded frames (5.167 s)
and a 32 kHz stereo AAC track (5.143 s, about −11 dBFS RMS per channel, not
silent). Every video and audio frame decoded with PyAV 19.0.2 (bundled
FFmpeg) without errors.

Each guide was cropped to 1344×768 as the server does and compared with every
output frame:

| Guide | Best-matching frame | PSNR / SSIM at the guide frame | PSNR one and six frames earlier |
| --- | --- | --- | --- |
| Frame 36 | 36 | 29.0 dB / 0.95 | 8.3 dB / 8.2 dB |
| Frame 72 | 72 | 32.6 dB / 0.95 | 29.5 dB / 14.7 dB |
| Frame 120 | 120 | 33.5 dB / 0.94 | 31.6 dB / 32.9 dB |

Each guided frame lands on its index. The hard cuts are at frames 36, 64 and
109: shots 3 and 4 start before their guide frames, and the near-static last
shot already resembles its guide from frame 109. A guide fixes the frame it is
placed on, not where the cut happens. Frame 0 follows the `<Picture 1>`
composition loosely (21.7 dB). An independent 50-step HTTP run of the same
request gave the same guide placement and cut frames. Its frames match the
ComfyUI-saved output at ≥38 dB PSNR, a difference due to re-encoding.

On this single-GPU offload profile each step takes about 37 s, so the request
needs about 31 minutes. That exceeds the Generate Video client's fixed
30-minute job wait, and the stock node stops waiting before the video is ready.
The run above raised that limit to 60 minutes in the local test environment
only. On slow single-GPU servers, use a faster deployment from the
recipe or fewer steps. For comparison, the recipe's two-GPU RTX PRO 6000
profile measured 285 s for a 50-step T2VA request of the same shape. A 2-step run queued from the ComfyUI UI on the same server
completed in 122 s with the same output shape and audio.

## Tests

```bash
python -m pytest -q tests/e2e/features/comfyui/test_h3_multiframe_reference_workflow.py
```

The test checks the graph against the node schemas, then executes it on CPU
with the real vLLM-Omni nodes and asserts the `/v1/videos` request: Ref2VA with
one image reference, guides at 36/72/120 in chain order, 124 frames at 24 FPS,
50 steps and `quality=lossless`. It does not run H3.
