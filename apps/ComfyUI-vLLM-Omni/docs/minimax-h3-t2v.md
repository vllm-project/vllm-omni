# MiniMax H3 Text to Video (WF-01)

Import [MiniMax_H3_Text_to_Video.json](../example_workflows/MiniMax_H3_Text_to_Video.json)
by dragging it into ComfyUI, or select it under **Templates → ComfyUI-vLLM-Omni**.
Restart ComfyUI after updating the extension. ComfyUI discovers the plugin's
`example_workflows` directory automatically.

The workflow sends a joint video/audio prompt through Generate Video to a remote
H3 FL2VA service, then passes the returned VIDEO directly to Save Video. No H3
weights are loaded by ComfyUI. The base presets are connected initially; the
optional Turbo presets and Remote LoRA are disconnected.

## Server and client setup

Install vLLM-Omni using the current [installation guide](../../../../docs/getting_started/installation/README.md).
Download the FL2VA partition of `MiniMaxAI/MiniMax-H3`, preserving its directory
layout. Use the [H3 deployment recipe](../../../../recipes/MiniMaxAI/MiniMax-H3.md)
to choose parallelism and offload settings for your hardware.

For a two-GPU server with sufficient memory, this is a starting command. Replace
`MODEL_ROOT` with the server directory containing `FL2VA`, and select free GPUs.
The served name must match the workflow's **model** field.

```bash
export MODEL_ROOT=/path/to/MiniMax-H3
CUDA_VISIBLE_DEVICES=0,1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm serve "${MODEL_ROOT}" --omni --host 127.0.0.1 --port 8000 \
  --served-model-name MiniMaxAI/MiniMax-H3 --trust-remote-code \
  --task-type fl2va --num-gpus 2 --tensor-parallel-size 2 \
  --text-encoder-tp-size 2 --enforce-eager
```

Use a separate environment for ComfyUI and install this extension under
`ComfyUI/custom_nodes/ComfyUI-vLLM-Omni`. For development, a symlink to this
checkout's `apps/ComfyUI-vLLM-Omni` directory keeps edits in one place.

```bash
# Run in the ComfyUI checkout and its environment.
python main.py --cpu --listen 127.0.0.1 --port 8188
```

Set Generate Video's **url** to the endpoint reachable **from ComfyUI**, including
`/v1`; the template defaults to `http://localhost:8000/v1`. If both services run
on a remote host, forward the ComfyUI port to your browser:

```bash
ssh -L 8188:127.0.0.1:8188 YOUR_SERVER
```

## Base and Turbo in one template

Write camera action and accompanying dialogue, sound effects, or music in the
same **prompt**. Leave **frame** and **references** disconnected to select T2VA.
The template uses 1344×768, 24 FPS, 124 frames (17×7+5, about 5.17 seconds), and
seed 1101. H3 Params supplies the explicit `16:9` aspect ratio required by H3;
it is not inferred from rounded pixel dimensions. For portrait output, change
both the dimensions to 768×1344 and the active H3 Params ratio to `9:16`.

| Setting | Base | Turbo v1.0 768p, four forwards |
| --- | --- | --- |
| Sampling node | Base sampling | Turbo sampling |
| Inference steps | 50 | 5 |
| H3 params node | Base H3 params | Turbo H3 params |
| Video flow shift | 12 | 6 |
| Audio flow shift | 3 | 3 |
| Remote LoRA | Disconnected | Connected, scale 1.0 |

For Turbo, download this exact **Diffusers-layout** artifact from
[lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo):

```text
minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors
```

Preload it by adding `--lora-backend peft --lora-path "${TURBO_LORA}"` to the
FL2VA server command, where `TURBO_LORA` is the full server-side artifact path.
Enter that same path in Remote LoRA's **local_path**; this is a server path,
not a browser or ComfyUI path. Keep the filename unchanged. The `_comfyui_`
export is not supported by the remote backend.

Connect **Turbo sampling**, **Turbo H3 params**, and **Remote LoRA** to Generate
Video's `sampling_params`, `model_params`, and `lora` inputs respectively,
replacing the two base links. Four forwards require **5** inference steps in
this backend. Other Turbo filenames can require different settings; consult the
[artifact contract table](../../../../recipes/MiniMaxAI/MiniMax-H3.md#turbo-lora).
To return to base, reconnect both base presets and disconnect LoRA.

## Validation

The focused tests read the shipped JSON, check its links and widget ordering,
and execute its base/Turbo settings through the real nodes and multipart
serializer. They mock HTTP responses, so they do not establish model quality
or audio preservation.

```bash
# Run from the vLLM-Omni checkout with its test dependencies installed.
python -m pytest tests/e2e/features/comfyui/test_minimax_h3_workflow.py -v
```

Queue the imported graph with the configuration being validated (for example,
the named Turbo preset), and state which configuration was actually tested.
Save Video writes to `ComfyUI/output/video/MiniMax_H3_WF01_*.mp4`. Record each exact path,
backend and ComfyUI commits, model/adapter revisions, startup command, parameters,
and peak GPU memory. For each saved file, run:

```bash
export OUTPUT=/path/to/ComfyUI/output/video/MiniMax_H3_WF01_00001_.mp4
ffprobe -v error -show_streams -show_format -of json "${OUTPUT}"
ffmpeg -v error -i "${OUTPUT}" -f null -
```

Require a nonempty video stream and audio stream, 24 FPS, the requested frame
count and dimensions, and audio/video durations agreeing within one video
frame plus audio codec padding. Listen to the saved file to confirm audio is
present and follows the visible action. HTTP success alone is insufficient.

**Shared audio prerequisite:** the existing client can consume the container
while decoding video before reading audio. Follow
[the shared audio work in #7380](https://github.com/vllm-project/vllm-omni/issues/7380)
and [#6782](https://github.com/vllm-project/vllm-omni/pull/6782). This workflow does
not include that fix. Record any prerequisite commit used in real-model
validation; a silent saved file does not pass WF-01 acceptance.

## Recorded validation

Validated on two H20-3e GPUs with vLLM-Omni base commit `bad50980`, vLLM 0.29.0,
PyTorch 2.13.0+cu129, and ComfyUI commit
`1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a` (CPU mode). The audio prerequisite
was PR #6782 at `f1ab717a9692d485ab33e1c0ffa45b87334befb2`.

In addition to the server flags above and the named Turbo LoRA, this run used:

```bash
--usp 1 --ring 1 --vae-patch-parallel-size 2 --vae-parallel-mode tile \
  --vae-use-tiling --diffusion-attention-backend TORCH_SDPA
```

On driver 550.127.08, the available FA2 path failed with an unsupported PTX
error; `TORCH_SDPA` completed without a driver change. This is the measured
configuration, not a Flash Attention performance result.

The complete Turbo run used the template prompt, seed 1101, 5 inference steps,
flow shifts 6/3 and LoRA scale 1. ComfyUI execution took approximately 129 seconds;
GPU memory sampled every 10 seconds peaked at 75,378 MiB per card. The saved MP4
contained 124 H264 frames at 1344×768 and 24 FPS, plus 32 kHz stereo AAC audio.
Video/audio durations were 5.166667/5.167000 seconds. Full ffmpeg decode passed,
and the audio samples were nonzero. Subjective audio alignment is not established
by those checks. Base parameters passed automated tests and a reduced-step smoke;
50-step Base quality was not evaluated.

## Source and scope

The reference is the official
[ComfyUI H3 T2V template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json).
This remote adaptation preserves joint text-to-video/audio generation, native
canvas/frame settings, optional Turbo, and Save Video output. The service owns
text encoding, denoising and both VAEs. Local loaders, ComfyUI-format LoRAs and
optional local prompt-embedding files are not part of this remote workflow.
