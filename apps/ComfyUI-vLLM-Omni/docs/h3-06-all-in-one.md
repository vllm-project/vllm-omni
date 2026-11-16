# H3-06: MiniMax-H3 all-in-one workflow

One entry graph for the three MiniMax-H3 generation modes: text-to-video+audio
(T2VA), first/last-frame (FL2VA), and mixed-reference (Ref2VA). It uses only
the existing **Generate Video**, **Video References**, sampling, and H3 params
nodes plus core ComfyUI nodes; no extra custom nodes are required. ComfyUI does
not load H3 weights. Every mode runs on a remote vLLM-Omni service.

## Requirements

- ComfyUI with ComfyUI-vLLM-Omni installed. The graph uses the core
  `PrimitiveString` and `PrimitiveFloat` nodes for shared settings.
- An H3 service that serves the modes you plan to run. Follow the
  [H3 recipe](../../../recipes/MiniMaxAI/MiniMax-H3.md) for your hardware.
  - A **combined** service (repository ID, no `--task-type`) serves all three
    modes without a restart: `t2va` and `fl2va` use the FL2VA DiT, and
    `ref2va` uses the Ref2VA DiT. See the recipe's
    [four-GPU combined service](../../../recipes/MiniMaxAI/MiniMax-H3.md#four-gpus-throughput-oriented-combined-service)
    for a starting point. It needs both checkpoint partitions.
  - A service started with `--task-type fl2va` serves T2VA and FL2VA only, and
    one started with `--task-type ref2va` serves Ref2VA only.

## Graph layout

| Group | Contents |
| --- | --- |
| Shared settings - all modes | Server URL, served model name, duration, Base sampling (50 steps, seed 42), H3 flow shifts (video 12, audio 3), usage note |
| Mode 1: T2VA | Generate Video with no image/reference input, Save Video |
| Mode 2: FL2VA | First frame and last frame Load Image nodes, Generate Video, Save Video |
| Mode 3: Ref2VA | Load Image, Video References, Generate Video, Save Video |

The shared nodes feed the `url`, `model`, `duration`, `sampling_params`, and
`model_params` inputs of all three Generate Video nodes, so you set them once.
Each mode keeps its own prompt because the modes need different wording:
Ref2VA prompts name their references, and FL2VA prompts describe motion
between keyframes. Generate Video selects the server task from which inputs
are connected:

| Mode | Connected inputs | Server task |
| --- | --- | --- |
| T2VA | none | `t2va` (aspect ratio derived from width/height) |
| FL2VA | `first_frame`, `last_frame`, or both | `fl2va` (`frame_indices` `[0]`, `[-1]`, or `[0, -1]`) |
| Ref2VA | `references` | `ref2va` |

## Use

Open **vLLM-Omni MiniMax-H3 All-in-One.json** from **Templates →
ComfyUI-vLLM-Omni**, or drag the JSON onto the canvas.

1. Set **Server URL** to the service's `/v1` address as seen from ComfyUI and
   **Served model name** to its `--served-model-name` (the repository ID when
   none is given). Defaults: `http://localhost:8000/v1` and
   `MiniMaxAI/MiniMax-H3`.
2. Choose a mode. Only the T2VA group is active when the template opens; the
   FL2VA and Ref2VA groups are muted. Right-click a group title and choose
   **Set Group Nodes to Always** to enable it, and **Set Group Nodes to Never**
   on the groups you are not using. Selecting nodes and pressing **Ctrl+M** also
   toggles muting. Several active groups run together and send one request per
   mode.
3. Fill in the active mode's inputs:
   - **T2VA**: write the scene and its sound in the prompt. Width and height
     select the nearest supported aspect ratio; 1344×768 maps to `16:9`, and
     768×1344 to `9:16`.
   - **FL2VA**: select the first and last frame images. Disconnect one loader
     for first-frame-only or last-frame-only generation. The output ratio
     follows the input image.
   - **Ref2VA**: select a reference image. Add more image, video, or audio
     loaders to **Video References**: up to 9 images, 3 videos, and 3 audio
     clips, 12 inputs in total, with at least one image or video. Name the
     references in the prompt as `<Picture 1>`, `<Video 1>`, and `<Audio 1>`,
     counting each media type in slot order.

   The template's image filenames are placeholders relative to ComfyUI's input
   directory. No sample assets or model weights are bundled.
4. Run the workflow. The active mode's Save Video node writes an MP4 with the
   generated audio under `output/video/MiniMax-H3-AIO-<mode>_*.mp4`.

The default duration is 5.167 seconds, which converts to 124 frames at 24 FPS
and satisfies H3's `17k+5` frame constraint. Keep durations within the 4–15
second output range; 107, 124, 209, and 345 frames are valid examples.

### Turbo

The graph is configured for Base H3. Turbo adapters are task-specific (the
FL2VA and Ref2VA artifacts differ) and must be preloaded by the server, so they
do not fit one shared request setting. To use Turbo for one mode, start a server
with that mode's adapter, add a **Remote LoRA** node to that mode's Generate
Video, and change the shared sampling and flow-shift values as described in the
[Turbo LoRA recipe](../../../recipes/MiniMaxAI/MiniMax-H3.md#turbo-lora). The
[text-to-video](minimax-h3-t2v.md) and
[reference-to-video](../README.md#minimax-h3-reference-to-video) guides give
per-artifact settings.

## Prompt assistance

This graph does not include a prompt-assistance stage. H3 itself does not
supply a prompt LLM, so each mode's prompt is used as written. Prompt
expansion will be added only after the source graph's assistant model and
optional dependencies are identified. It would then be a separate,
optional stage, for example a **Multimodality Understanding** node calling its
own vLLM-Omni service, rather than part of the H3 request.

## Validation

Check the template's wiring and defaults locally:

```bash
python -m pytest tests/e2e/features/comfyui/test_h3_all_in_one_workflow.py -q
```

For a real-model check, run each mode against a served H3 model and inspect
the saved file:

```bash
ffprobe -v error -show_entries stream=codec_type,width,height,r_frame_rate,nb_frames,sample_rate,channels \
  -show_entries format=duration -of json "$OUTPUT_MP4"
```

Expect 24 FPS H.264 video, a stereo AAC audio stream, and aligned video/audio
durations. Watch and listen to each result for keyframe adherence (FL2VA),
reference conditioning (Ref2VA), and audio/video synchronization. Record the
server commit, model checkpoint, ComfyUI version, graph revision, seed, inputs,
runtime, and output. Local schema checks do not establish generation quality.
