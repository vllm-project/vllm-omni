# WF-07: H3 video upscale with SeedVR2

Generate video and audio through vLLM-Omni H3, upscale the decoded frames with SeedVR2, and save both the original and upscaled videos. The generated audio and FPS connect directly to Create Video.

## Requirements

- ComfyUI with ComfyUI-vLLM-Omni and [ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler#-installation) installed.
- An H3 service using the FL2VA checkpoint partition. Follow the [H3 recipe](../../../recipes/MiniMaxAI/MiniMax-H3.md) for server setup.
- `seedvr2_ema_3b_fp16.safetensors` and `ema_vae_fp16.safetensors` from [SeedVR2's ComfyUI weights](https://huggingface.co/numz/SeedVR2_comfyUI/tree/main), placed in `ComfyUI/models/SEEDVR2`. The SeedVR2 nodes can also download them on first use.

## Use

Open **vLLM-Omni MiniMax H3 Video Upscale.json** from the example workflows. An API-format version is provided alongside it. Set the Generate Video URL/model and the LoRA path as seen by the H3 server.

The preset uses the LightX2V v1.0 768p Turbo adapter in Diffusers format: `minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors`, five sigma points, video/audio flow shifts 6/3, and LoRA scale 1. Preload the same adapter on the server using the [Turbo LoRA recipe](../../../recipes/MiniMaxAI/MiniMax-H3.md#turbo-lora). For base H3, disconnect LoRA, set inference steps to 50, and set video flow shift to 12.

The H3 preset requests 1344 x 768 at 24 FPS for 5.167 seconds, which gives 124 frames. SeedVR2's `resolution=1536` sets the target short edge, producing 2688 x 1536 for this input. Update this value when changing the input size; it is an output resolution, not a fixed 2x multiplier.

The SeedVR2 preset uses 3B FP16, five-frame batches, one overlapping frame, uniform batch padding, and LAB color correction. The DiT loader uses CPU offload with 32 swapped blocks; the VAE uses tiled encoding and decoding. Select the local GPU in both model loaders. See the [SeedVR2 settings](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler#-usage) when adjusting memory use or batch size.

Run the workflow. The two Save Video nodes write MP4s under `WF07/generated` and `WF07/upscaled` in ComfyUI's output directory.
