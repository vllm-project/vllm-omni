# MiniMax H3 Text to Video (WF-01)

Import [MiniMax_H3_Text_to_Video.json](../example_workflows/MiniMax_H3_Text_to_Video.json)
into ComfyUI, or select it under **Templates → ComfyUI-vLLM-Omni**.
The template connects existing nodes to a remote H3 service and saves the
returned video with audio. ComfyUI does not load H3 weights.

## Setup

Install the [ComfyUI extension](../README.md#installation) and follow the
[H3 deployment recipe](../../../recipes/MiniMaxAI/MiniMax-H3.md) for your hardware.
For a two-GPU FL2VA server with sufficient memory:

```bash
export MODEL=/path/to/MiniMax-H3/FL2VA
CUDA_VISIBLE_DEVICES=0,1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400 \
vllm serve "$MODEL" --omni --host 127.0.0.1 --port 8000 \
  --served-model-name MiniMaxAI/MiniMax-H3 --trust-remote-code \
  --task-type fl2va --num-gpus 2 --tensor-parallel-size 2 \
  --text-encoder-tp-size 2 --enforce-eager
```

Set Generate Video's **url** to the address reachable from ComfyUI, including
`/v1`, and match its **model** to the served name. The defaults are
`http://localhost:8000/v1` and `MiniMaxAI/MiniMax-H3`.

## Base and Turbo

Base presets are connected by default. Write the scene and its sound in the
same **prompt**; leave **frame**, **references**, and **fast_h3** disconnected.
The template uses 1344×768, 24 FPS, seed 1101 and **duration** 5.167 seconds
(124 frames, satisfying H3's `17k+5` constraint). The existing client infers
`16:9` from these dimensions. For portrait output, use 768×1344.

| Setting | Base | Turbo v1.0 768p |
| --- | --- | --- |
| Inference steps | 50 | 5 (4 forwards) |
| Video flow shift | 12 | 6 |
| Audio flow shift | 3 | 3 |
| Remote LoRA | Disconnected | Connected, scale 1 |

For Turbo, use the **Diffusers-layout** artifact from
[lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo):

```text
minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors
```

Add `--lora-backend peft --lora-path "$TURBO_LORA"` to the server command,
where `TURBO_LORA` is the full server-side path to that file. Enter the same
path in Remote LoRA's **local_path**; retain name `h3-turbo-v1.0-768p`, scale
`1.0`, and **int_id** `0` so the existing API derives the ID from the path.
The template leaves the path blank for users to configure. The `_comfyui_`
export is not supported by this backend.

Connect **Turbo sampling**, **Turbo H3 params**, and **Remote LoRA** to Generate
Video's `sampling_params`, `model_params`, and `lora` inputs, replacing the
Base links. To return to Base, restore both Base links and disconnect LoRA.
Other Turbo artifacts may need different settings; see the
[artifact table](../../../recipes/MiniMaxAI/MiniMax-H3.md#turbo-lora).

If importing an older template with a frame-count widget, reimport this JSON
or set **duration** to **5.167**; do not interpret 124 frames as 124 seconds.

## Validation

Queue the configured workflow and inspect the MP4 saved under
`ComfyUI/output/video/MiniMax_H3_WF01_*.mp4`:

```bash
ffprobe -v error -show_streams -show_format -of json "$OUTPUT"
ffmpeg -v error -i "$OUTPUT" -f null -
```

Check the requested dimensions, frame count, 24 FPS and both video/audio
streams, then watch and listen to the saved file. The current upstream
extension includes the shared audio-preserving decoder fix.

On September 16, the narrowed workflow was validated through ComfyUI using
existing upstream Python code at `4a8ac297`, two H20-3e GPUs, vLLM 0.29.0,
PyTorch 2.13.0+cu129 and ComfyUI `1d48d9cf`. The run used the template's
forest-stream prompt, explicit server-side Turbo LoRA path, and settings
above, with `TORCH_SDPA`, `--usp 1 --ring 1`, and tiled VAE parallelism on two
GPUs. ComfyUI execution took 131.55 seconds. SaveVideo produced 124 H.264
frames at 1344×768 and 24 FPS, plus 32 kHz stereo AAC audio. Video/audio
durations were 5.166667/5.167000 seconds; full decoding passed and audio was
nonzero. This is model-generated output, not a ground-truth reference or a
subjective audio-quality assessment. Base graph validation passed with Turbo
nodes disconnected; full 50-step Base quality was not evaluated.

Based on the official
[ComfyUI H3 T2V template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/video_minimax_h3_t2v.json),
using remote inference instead of local model loaders.
