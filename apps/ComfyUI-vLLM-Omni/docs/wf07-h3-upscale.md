# WF-07: synchronized H3 video upscale

This template generates video and audio on a remote vLLM-Omni H3 service, upscales only the decoded video frames in ComfyUI, then recombines them with the generated audio and frame rate. It saves both the generated and upscaled videos.

## Requirements

- A FL2VA-capable H3 service. T2VA uses the FL2VA checkpoint partition.
- ComfyUI-vLLM-Omni with [PR #7456](https://github.com/vllm-project/vllm-omni/pull/7456), which provides the duration input, automatic H3 aspect ratio, and audio decoder fix used by this workflow.
- [RealESRGAN_x2plus.pth](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth) in ComfyUI's `models/upscale_models`. SHA256: `49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb`.

The included preset uses LightX2V's Diffusers-layout `minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors`: five sigma points, video flow shift 6, audio flow shift 3, scale 1. The server must preload that adapter and accept its server-side path. Do not substitute its ComfyUI export. These settings come from the [H3 recipe](../../../recipes/MiniMaxAI/MiniMax-H3.md#turbo-lora).

## Use

Open `vLLM-Omni MiniMax H3 Video Upscale.json` from the example workflows. Set the Generate Video URL/model and the LoRA path as seen by the remote server. The provided relative adapter path assumes the server starts from the directory containing `models/minimax-h3-turbo`.

The preset requests 1344×768 at 24 FPS with `duration=5.167` seconds. Generate Video converts this to 124 frames, satisfying H3's `17k+5` frame constraint. The graph connects the generated audio and FPS directly to Create Video; it does not interpolate frames or replace the soundtrack. The chosen upscale model produces 2688×1536. For base H3, disconnect LoRA, set inference steps to 50, and set video flow shift to 12.

GPU placement and offload belong to the server command, not to local H3 loader nodes. Use the official recipe for a profile suitable for your hardware. The worktree test's exact environment, startup arguments and real-model results are recorded separately with its evidence.

## End-to-end validation

The pytest case runs the supplied workflow against live ComfyUI and H3 services, waits for generation and upscale, downloads both MP4s, and checks the saved media. It does not mock generation, ComfyUI nodes, or media conversion. NumPy, SciPy, ffmpeg and ffprobe must be available to the test environment.

From the vLLM-Omni checkout, with both services configured as above:

```bash
COMFYUI_URL=http://127.0.0.1:8188 \
VLLM_OMNI_URL=http://127.0.0.1:8091/v1 \
WF07_LORA_PATH=models/minimax-h3-turbo/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors \
WF07_OUTPUT_DIR=wf07-evidence \
python -m pytest tests/e2e/features/comfyui/test_minimax_h3_upscale.py \
  -m 'core_model and diffusion and gpu' --run-level core_model -v -s
```

The test selects a new seed for each run and records it with the submitted graph. Set `WF07_SEED` to reproduce a particular request. An identical request may hit ComfyUI's cache; the test fails if the remote generation node was cached. If either service URL is unset, pytest skips the case rather than treating it as a completed E2E run. This case requires separately provisioned ComfyUI and H3 services; the default Buildkite jobs do not provision them.

The same runner can be invoked directly:

```bash
python apps/ComfyUI-vLLM-Omni/scripts/validate_h3_upscale.py \
  --comfy-url http://127.0.0.1:8188 \
  --server-url http://127.0.0.1:8091/v1 \
  --model MiniMaxAI/MiniMax-H3 \
  --lora-path models/minimax-h3-turbo/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors \
  --seed 20260914 --output-dir wf07-evidence
```

The runner records the submitted graph, ComfyUI history, generated and upscaled MP4s, their hashes, and `validation.json`. It checks dimensions, frame count, 24 FPS, duration, audio presence, sample rate/channels, and audio alignment before and after upscale. These checks measure preservation of the generated timing and soundtrack; they do not score visual quality or semantic audio-to-action synchronization. The request also stores the UI graph in ComfyUI history so the recorded run can be opened with its settings and outputs.

## Recorded runtime configuration

The H3 service used vLLM-Omni commit `33fff2155e0bf9bc8688f7ffb92ef508efdf6954` plus the original worktree changes. The ComfyUI extension and E2E runner use this PR on base `58adeec05f151e18542323cb4644b009b83cfef3`, including the interface changes from #7456. The ComfyUI host used commit `cbbc9dab1f03d0d9a6caa8a8be7d77a7e37e1e44`. The server environment contained vLLM `0.29.0+cu129`, PyTorch `2.13.0+cu129`, and NVIDIA driver `570.190`. H3 used two RTX 4090 GPUs with TP2, distributed layerwise offload, eight resident layers and cuDNN attention.

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
