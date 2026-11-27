# FastH3 — four-step Preview V1

[FastH3 family](README.md) · [FastH3 V2](FastH3-V2.md) · [FastH3 Omni](FastH3-Omni.md)

## Summary

- Task: T2VA, text to synchronized video and audio.
- Release: a load-time adapter for `MiniMaxAI/MiniMax-H3`.
- Platform: CUDA with a compatible dense or tile64 VSA backend.
- Mode: existing `/v1/videos` API.

## Hardware and backend selection

The adapter is not tied to a GPU architecture. Select a backend for the
adapter variant and the installed hardware:

| Variant | Backend | Hardware requirements |
| --- | --- | --- |
| Dense | `FLASH_ATTN`, or another H3-compatible dense backend | Follow that backend's GPU and build requirements |
| VSA | `FASTVIDEO_VSA`, BF16 | FastVideo tile64 on compatible CUDA GPUs; automatic FlashInfer selection on SM120/SM121 when available |

The command below uses eight GPUs. GPU memory, interconnect, and output size
determine suitable parallel settings.

## Model and installation

[FastH3](https://haoailab.com/blogs/fasth3-preview/) generates video and audio
in four steps using an adapter for MiniMax-H3. Load one adapter variant when
starting the server.

Install the matching vLLM/vLLM-Omni source environment using the
[CUDA installation guide](../../docs/getting_started/installation/gpu.md).
`ffmpeg` and `ffprobe` are required for video output. Authenticate with
`hf auth login` when the base model or adapter requires access.

Download the Dense / Data-Free adapter and pass its file path to `--lora-path`:

```bash
export FASTH3_DIR=/path/to/fasth3
hf download FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA \
  dense-datafree/adapter_model.safetensors --local-dir "${FASTH3_DIR}"
export FASTH3_LORA="${FASTH3_DIR}/dense-datafree/adapter_model.safetensors"
```

## Serve the Dense / Data-Free adapter

```bash
export FASTH3_DENSE_BACKEND=FLASH_ATTN

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm serve MiniMaxAI/MiniMax-H3 --omni \
  --host 127.0.0.1 --port 8094 --trust-remote-code \
  --task-type fl2va --served-model-name FastH3 \
  --lora-path "${FASTH3_LORA}" \
  --num-gpus 8 --tensor-parallel-size 1 --usp 8 --ring 1 \
  --vae-patch-parallel-size 8 --vae-parallel-mode tile --vae-use-tiling \
  --diffusion-attention-config.default.backend "${FASTH3_DENSE_BACKEND}"
```

T2VA uses H3's FL2VA partition, so `--task-type fl2va` is intentional.
The adapter is fused at load time; `--lora-backend` and per-request `lora=`
selection do not apply.

## Generate video and audio

```bash
curl --fail-with-body -sS http://127.0.0.1:8094/v1/videos/sync \
  -F 'model=FastH3' \
  -F 'prompt=A red fox runs through fresh snow at dawn, with pawsteps, winter wind and distant birds.' \
  -F 'width=1344' -F 'height=768' -F 'aspect_ratio=16:9' -F 'fps=24' \
  -F 'num_inference_steps=4' -F 'guidance_scale=1' -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":4.4}' \
  -o fasth3.mp4

ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate,sample_rate,channels \
  -of json fasth3.mp4
```

Use `num_inference_steps=4` and `task=t2va`. Keep the default noise schedule
and flow shifts. Choose the adapter at startup; per-request adapter
switching and CPU/layer offload are unsupported.

## VSA adapter variants

For a VSA adapter variant, download that variant's adapter file, install the
`vllm-omni[vsa]` extra, and replace the dense attention backend in the serve
command with these flags. The provider is selected for the actual GPU;
SM90 and SM100/SM103 use FastVideo's tile64 route, while compatible SM120/SM121
installations use FlashInfer BF16. The video request API is unchanged:

```bash
--diffusion-attention-config.default.backend FASTVIDEO_VSA \
--diffusion-attention-config.default.fastvideo_vsa_topk 64
```

See
[provider selection](../../docs/user_guide/diffusion/attention_backends/fastvideo_vsa.md#automatic-provider-selection)
for hardware requirements and provider overrides.

For multi-GPU VSA, use Ulysses sequence parallelism. Ring and all-gather
sequence parallelism are unsupported. The Dense / Data-Free variant uses
a dense attention backend and does not require the VSA extra.
