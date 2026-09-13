# WF-07: synchronized H3 video upscale

This template generates video and audio on a remote vLLM-Omni H3 service, upscales only the decoded video frames in ComfyUI, then recombines them with the generated audio and frame rate. It saves both the generated and upscaled videos.

## Requirements

- A FL2VA-capable H3 service. T2VA uses the FL2VA checkpoint partition.
- The audio-preserving MP4 decoder from [PR #6782](https://github.com/vllm-project/vllm-omni/pull/6782), or an equivalent fix. A workflow cannot recover audio discarded by its input decoder.
- The optional H3 `aspect_ratio` parameter, serialized into `extra_params`. T2VA requires this even when explicit dimensions are supplied.
- [RealESRGAN_x2plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth) in ComfyUI's `models/upscale_models`. SHA256: `49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb`.

The included preset uses LightX2V's Diffusers-layout `minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors`: five sigma points, video flow shift 6, audio flow shift 3, scale 1. The server must preload that adapter and accept its server-side path. Do not substitute its ComfyUI export. These settings come from the [H3 recipe](../../../recipes/MiniMaxAI/MiniMax-H3.md#turbo-lora).

## Use

Open `vLLM-Omni MiniMax H3 Video Upscale.json` from the example workflows. Set the Generate Video URL/model and the LoRA path as seen by the remote server. The provided relative adapter path assumes the server starts from the directory containing `models/minimax-h3-turbo`.

The preset requests 1344×768, 124 frames, 24 FPS. The graph connects the generated audio and FPS directly to Create Video; it does not interpolate frames or replace the soundtrack. The chosen upscale model produces 2688×1536. For base H3, disconnect LoRA, set inference steps to 50, and set video flow shift to 12.

GPU placement and offload belong to the server command, not to local H3 loader nodes. Use the official recipe for a profile suitable for your hardware. The worktree test's exact environment, startup arguments and real-model results are recorded separately with its evidence.

## Exact validation commands

From the vLLM-Omni checkout, with its test dependencies installed:

```bash
python -m pytest tests/e2e/features/comfyui/test_minimax_h3_upscale.py -q
```

With ComfyUI and the real H3 service running, and NumPy, SciPy, ffmpeg and ffprobe available:

```bash
python apps/ComfyUI-vLLM-Omni/scripts/validate_h3_upscale.py \
  --comfy-url http://127.0.0.1:8188 \
  --server-url http://127.0.0.1:8091/v1 \
  --model MiniMaxAI/MiniMax-H3 \
  --lora-path models/minimax-h3-turbo/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors \
  --output-dir wf07-evidence
```

The script records the submitted graph, ComfyUI history, generated and upscaled MP4s, their hashes, and `validation.json`. It checks dimensions, frame count, 24 FPS, duration, audio presence, sample rate/channels, and audio alignment before and after upscale. These checks measure preservation of the generated timing and soundtrack; they do not score the model's visual quality or its semantic audio-to-action synchronization.

## Recorded runtime configuration

The recorded run used vLLM-Omni commit `33fff2155e0bf9bc8688f7ffb92ef508efdf6954` plus the worktree changes, and ComfyUI commit `cbbc9dab1f03d0d9a6caa8a8be7d77a7e37e1e44`. The server environment contained vLLM `0.29.0+cu129`, PyTorch `2.13.0+cu129`, and NVIDIA driver `570.190`. H3 used two RTX 4090 GPUs with TP2, distributed layerwise offload, eight resident layers and cuDNN attention.

The recorded server command points to the `FL2VA` partition. It sets `VLLM_HOST_IP=127.0.0.1` and `GLOO_SOCKET_IFNAME=lo`. The following version uses a placeholder model root:

```bash
# Run from the directory containing the downloaded models/ folder.
export MODEL_ROOT=/path/to/MiniMax-H3
export LORA=models/minimax-h3-turbo/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=1,2 OMP_NUM_THREADS=8 \
VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_HOST_IP=127.0.0.1 \
GLOO_SOCKET_IFNAME=lo VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200 HF_HUB_OFFLINE=1 \
vllm serve "$MODEL_ROOT/FL2VA" --omni \
  --served-model-name MiniMaxAI/MiniMax-H3 --host 127.0.0.1 --port 8091 \
  --trust-remote-code --task-type fl2va --num-gpus 2 \
  --tensor-parallel-size 2 --usp 1 --ring 1 --text-encoder-tp-size 2 \
  --vae-patch-parallel-size 2 --vae-parallel-mode tile --vae-use-tiling \
  --enable-distributed-layerwise-offload --dlo-no-use-allgather --dlo-resident-layers 8 \
  --enforce-eager --diffusion-attention-backend CUDNN_ATTN \
  --lora-backend peft --lora-path "$LORA"
```

Use the same server-side LoRA path string in the preload command and the workflow. The validation script measures elapsed time after submission setup; model download and server startup occur before that measurement.
