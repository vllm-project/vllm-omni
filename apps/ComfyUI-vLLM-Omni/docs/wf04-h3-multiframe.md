# WF-04: H3 multiframe reference

Recreate the official ComfyUI MiniMax H3 **Multiframe Reference** layout on a
remote vLLM-Omni server: one identity reference plus three timeline anchors that
pin the composition at 1.5 s, 3.0 s, and 5.0 s of a 24 FPS clip.

The native template loads H3 locally and chains `MiniMaxH3AddGuide` nodes. This
version loads nothing but the four images locally; generation, sampling, and VAE
decoding run on the vLLM-Omni server.

## Requirements

- ComfyUI with ComfyUI-vLLM-Omni installed.
- A vLLM-Omni server that includes timeline guides
  ([#8838](https://github.com/vllm-project/vllm-omni/pull/8838)) and the Ref2VA
  partition of MiniMax-H3. Follow the
  [H3 recipe](../../../recipes/MiniMaxAI/MiniMax-H3.md) for server setup, and
  start it with a served model name that matches the H3 model spec, for example
  `--served-model-name MiniMaxAI/MiniMax-H3`.
- Base H3 weights. Timeline guides reject FastH3, a request-level LoRA
  (including Turbo), latent-mask editing, cache acceleration, and distilled
  schedules.

## Use

Open **vLLM-Omni MiniMax-H3 Multiframe Reference.json** from the example
workflows, then:

1. Set **Generate Video** `url` and `model` to the server.
2. Select your own four images in the **Load Image** nodes. The template's
   filenames are placeholders; no sample assets are bundled.
3. Rewrite the prompt for your assets, keeping the timing words aligned with the
   guide indices (1.5 s / 3.0 s / 5.0 s).
4. Run the workflow. **Save Video** writes an MP4 under
   `output/video/MiniMax-H3-Multiframe-Reference` and keeps the generated audio.

## The four anchors

| Native frame reference | vLLM-Omni connection | Role |
| --- | --- | --- |
| Frame 1 | **Video References** `image_1` | Identity and style reference, visible to the prompt as `<Picture 1>` |
| Frame 2 | **Add Guide** `frame_index=36` | Composition anchored at 1.5 s |
| Frame 3 | **Add Guide** `frame_index=72` | Composition anchored at 3.0 s |
| Frame 4 | **Add Guide** `frame_index=120` | Composition anchored at 5.0 s, held to the last frame |

Guides are clean timeline conditions: they are packed ahead of the references
and do not enter the Qwen reference presentation, so they are not `<Picture N>`
tags. That is why the template prompt tags `<Picture 1>` only and describes the
three later anchors by time. Point the prompt at the reference for identity, and
let the anchors carry composition.

Guide indices live on the 24 FPS timeline. Negative indices count from the end,
and a guide must fit inside the generated clip; the server shortens a clip guide
to `17k+5` frames and rejects an index that does not fit.

## Settings

| Setting | Value |
| --- | --- |
| Canvas | 1344 x 768 |
| FPS | 24 |
| Duration | 5.167 s -> 124 frames (`17k+5`) |
| Sampling | 50 points, guidance 1.0, true CFG 1.0, seed 42 |
| Flow shifts | video 12, audio 3 |
| Audio | `native` |

Other H3 canvas presets are 1024 x 768, 768 x 768, 768 x 1024, and 768 x 1344.
Keep frame counts at `17k+5` within the 4-15 second range (107, 124, 141, 158,
175, 192, 209, 226, 243, 260, 277, 294, 311, 328, 345). References are scaled
to the output area, so a reference image does not have to match the canvas.

Guided requests always send `fps=24` and `quality=lossless`; other FPS values are
rejected rather than changed, and requests without guides are unaffected.

## Validation

Check the graph against the node schemas locally:

```bash
python -m pytest tests/e2e/features/comfyui/test_h3_multiframe_reference_workflow.py -q
```

This verifies link/slot reciprocity, node input and output types, widget counts,
the anchor layout, the route (Ref2VA + guides), the `17k+5` frame count, and
portable asset paths. It does not run H3.

For a real-model check, run the workflow against a guide-capable H3 server and
inspect the saved file:

```bash
ffprobe -v error -count_frames \
  -show_entries stream=codec_type,width,height,r_frame_rate,nb_read_frames,sample_rate,channels \
  -show_entries format=duration -of json "$OUTPUT_MP4"
ffmpeg -v error -i "$OUTPUT_MP4" -f null -
```

Check for 24 FPS video, 124 frames, 1344 x 768, a stereo audio track, and aligned
duration, then watch the clip to confirm the three anchors appear where the
prompt says, and listen for synchronized audio. Record the server commit, model
revision, input images, prompt, settings, and output alongside the result.

No real-model result is recorded for this template yet; the workflow ships with
static validation only.
