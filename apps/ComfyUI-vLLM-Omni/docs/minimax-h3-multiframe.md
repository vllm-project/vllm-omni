# MiniMax H3 Multiframe Reference (WF-04 development draft)

This is development work for [RFC #7380, WF-04](https://github.com/vllm-project/vllm-omni/issues/7380).
It is **not an executable remote multiframe workflow**. GUIDE-01 (model/API),
GUIDE-02 (ComfyUI), and NODE-02 (ordered references) are not integrated in this
branch. The layout targets the current Generate Video schema, with a duration
of 5.167 seconds that converts to 124 frames at 24 FPS. Do not reuse the old
`num_frames=124` widget value as a duration. The FastH3 input remains
disconnected because this workflow requires Ref2VA and timeline guides.
Do not publish this layout in the template directory or enable generation
until the missing dependencies are implemented and validated.

The [draft layout](drafts/MiniMax_H3_Multiframe_Reference.draft.json) uses existing
ComfyUI and vLLM-Omni node types so that the UI arrangement can be reviewed.
Generate Video and Save Video are muted. The guide blocks are explanatory
Markdown Notes, not substitute guide nodes. Unmuting the graph cannot exercise
timeline conditioning. The baseline client also rejects the default image-only
semantic reference: its serializer requires either videos alone or exactly one
image plus one audio input. NODE-02 must remove that restriction before the
default reference request can be submitted.

## Four images, three timeline guides

The source is the [official multiframe template](https://github.com/Comfy-Org/workflow_templates/blob/aaac56dd5cc5497533d92cbe50edc35ea660e587/templates/video_minimax_h3_multiframe_reference.json).
Its default topology does not insert a guide at frame zero:

| Image | Default role | Time | Frame index |
| --- | --- | --- | --- |
| `h3_frame_ref_1.png` | First semantic reference, `<Picture 1>` | Opening composition requested in the prompt | No timeline guide |
| `h3_frame_ref_2.png` | First timeline guide | 1.5 s | 36 |
| `h3_frame_ref_3.png` | Second timeline guide | 3.0 s | 72 |
| `h3_frame_ref_4.png` | Third timeline guide | 5.0 s | 120 |

At 24 FPS, the output has 124 frames (`17 * 7 + 5`), about 5.167 seconds.
Each default guide is a single still image. Its length is 1; the `17k+5` clip
length rule must not be incorrectly applied to still-image guides.

Preserve guide order 36 -> 72 -> 120 when connecting the GUIDE-02 chain. Ordinary
reference ordering does not place images on the output timeline. The first image
is a semantic reference, so the prompt's opening-composition request is not a
guarantee of exact first-frame preservation.

Images 2-4 are not semantic references in the default official topology. If they
are also connected as semantic references after NODE-02, preserve their order
and then use `<Picture 2>` through `<Picture 4>` in the prompt. The draft prompt
uses only `<Picture 1>` to avoid unbound picture tags.

## Inputs and server settings

Download or replace the four images in ComfyUI's `input/` directory. The loader
widgets use portable filenames, not absolute paths. Pinned official inputs:

- [Image 1](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/aaac56dd5cc5497533d92cbe50edc35ea660e587/input/h3_frame_ref_1.png)
- [Image 2](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/aaac56dd5cc5497533d92cbe50edc35ea660e587/input/h3_frame_ref_2.png)
- [Image 3](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/aaac56dd5cc5497533d92cbe50edc35ea660e587/input/h3_frame_ref_3.png)
- [Image 4](https://raw.githubusercontent.com/Comfy-Org/workflow_templates/aaac56dd5cc5497533d92cbe50edc35ea660e587/input/h3_frame_ref_4.png)

Use a server-side H3 Ref2VA checkpoint, not a ComfyUI-repacked checkpoint. The
target canvas is the H3-native 1344x768 preset, with explicit `aspect_ratio=16:9`.
The draft records the intended named ratio in its output contract. The
current H3 Params node exposes only flow shifts; the final Ref2VA request
must be checked for the intended preset when its guide support is integrated.
Do not infer the named preset from the rounded pixel ratio `1344/768`, which is
not exactly `16/9`.

Base target settings follow the Omni H3 recipe: 50 sigma points,
`guidance_scale=1`, `true_cfg_scale=1`, `flow_shift=12`, `audio_flow_shift=3`,
and fixed seed 738004. These are configuration targets, not measured quality or
performance results.

Optional Ref2VA Turbo uses the server-format
`minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors` artifact: 5 sigma points,
4 denoiser forwards, `flow_shift=12`, `audio_flow_shift=3`, LoRA scale 1.
Use the existing Remote LoRA node with the actual server-local file path.
The source template's ComfyUI-repacked LoRA and FastH3 startup-fused T2VA
deployment do not provide this Ref2VA contract. Turbo is only documented in
this draft; its connected execution graph remains to be implemented and tested.

## Integration and validation

The draft's `extra.wf04` metadata records the source revision, image roles,
output contract and unresolved dependencies. It is development metadata, not a
new ComfyUI node type or an HTTP request schema. Guide node names and serialized
fields remain null until the actual GUIDE-02 implementation is available.

1. Pin and integrate GUIDE-01, GUIDE-02, and NODE-02. Verify the named
   aspect-ratio and audio-preservation behavior of the target branch.
2. Replace the three guide notes with the real GUIDE-02 nodes and connect them
   in order. Wire the final guide output to Generate Video using its real input
   schema. Keep the default semantic reference separate.
3. Add actual node-to-request tests for ordered guides, negative indices, still
   images, valid clip lengths, optional audio, and out-of-range rejection. Check
   the request body and server execution path, not just widget values.
4. Add the optional connected Ref2VA Turbo preset and verify its exact artifact
   and schedule. Unmute generation only in an integrated validation checkout.
5. Import the final graph in real ComfyUI, export its API-format prompt, and
   execute that prompt against the pinned remote server. Record source SHAs,
   model revision, media hashes, GPU UUIDs, environment and the exact command.
6. Save through ComfyUI Save Video. Validate 124 frames, 24 FPS, 1344x768,
   nonempty stereo audio, and audio/video duration agreement within one video
   frame. Inspect the actual generated output and anchor behavior. A request
   succeeding or an MP4 containing audio does not by itself prove guide use or
   perceptual audio/video synchronization.
7. Move the completed graph into `example_workflows/`, replace the draft tests
   with executable workflow tests, and include the recorded real-model result
   before claiming WF-04 complete.

Current CPU-only structural checks, from the repository root:

```bash
python -m pytest -o addopts='' --noconftest \
  tests/e2e/features/comfyui/test_minimax_h3_multiframe_draft.py -q
```

`--noconftest` deliberately runs the JSON/AST checks without the existing
ComfyUI test mocks, Torch, or an installed inference engine. These checks
validate the draft's structure and widget compatibility only. They do not
exercise actual ComfyUI import, API serialization, GPU execution or audio.

After real execution, retain both the server MP4 and the ComfyUI-saved MP4:

```bash
ffprobe -v error -count_frames -show_streams -show_format -of json "$OUTPUT"
ffmpeg -v error -i "$OUTPUT" -f null -
```

No real-model result, guide-conditioning result, Turbo result, saved-audio
result, or performance claim is available for this draft.
