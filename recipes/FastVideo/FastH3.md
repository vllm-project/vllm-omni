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
switching and layerwise offload are unsupported. Dense / Data-Free supports
model-level CPU offload as described below; VSA plus model-level CPU offload
remains unsupported.

## Model-level CPU offload

For a single GPU, the Dense / Data-Free variant may be combined with
model-level CPU offload. New configurations should use the component-selective
API. Replace the multi-GPU parallel settings in the serve command above with:

```bash
--num-gpus 1 \
--diffusion-offload-config \
'{"mode":"module","components":["dit","text_encoder"]}'
```

This compact configuration swaps the active DiT and text encoder while keeping
the VAEs resident. The compatibility alias `--enable-cpu-offload` remains
supported, but MiniMax-H3 uses it as a full-topology lifecycle that also stages
the VAEs. The controlled single-A100 measurements below used that compatibility
path, so they do not establish the compact selector's peak-HBM requirement.
Choose the compatibility alias when the additional VAE staging is required for
capacity; the two forms are not residency-equivalent.

Model-level CPU offload uses the ordinary checkpoint loader, completes FastH3
fusion and validation, and only then installs the offloader. Layer-mode
configurations, including the legacy `--enable-layerwise-offload` and
`--enable-distributed-layerwise-offload` aliases, continue to fail fast because
their loading paths can bypass the fusion. VSA plus model-level CPU offload is
also rejected until that combination receives its own device-level validation.

When model-level CPU offload loads the base parameters on CPU, FastH3 computes
each delta on the accelerator and copies the fused result back to the CPU
parameter before serving. This is startup-only traffic; allow extra startup
time and transient device memory compared with loading the already-fused
checkpoint.

AdaLN projection-weight offload is a separate mechanism. Its automatic default
is disabled when DiT-wide offload is selected; explicitly enabling both is
rejected. The model-level recipes here do not require an AdaLN offload flag.

### Single NVIDIA A100 80GB recipe

The following standalone recipe describes the Dense / Data-Free FastH3 path
previously validated on one NVIDIA A100 80GB. It pins the model and adapter
revisions used for the measurement and downloads only the FL2VA partition.
The A100 measurements below predate the current upstream refresh; they have
not been rerun on the refreshed code and do not establish its latency or
peak-HBM requirement. The attention-backend option below uses the current CLI
spelling:

```bash
export MODEL_DIR=/path/to/MiniMax-H3
export FASTH3_DIR=/path/to/FastH3-Dense-DataFree
export PORT=8091

hf download MiniMaxAI/MiniMax-H3 \
  --revision 42ed227ee7df40d41602854ae760620d6eb651fe \
  --include 'FL2VA/**' \
  --local-dir "${MODEL_DIR}"
hf download FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA \
  dense-datafree/adapter_model.safetensors \
  --revision bcf40ca6f457ed66f8badf13514943e390205fca \
  --local-dir "${FASTH3_DIR}"

export FASTH3_LORA="${FASTH3_DIR}/dense-datafree/adapter_model.safetensors"
```

Start the FL2VA server with the full-topology compatibility offload used by the
A100 run:

```bash
CUDA_VISIBLE_DEVICES=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
vllm serve "${MODEL_DIR}/FL2VA" \
  --omni \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --trust-remote-code \
  --task-type fl2va \
  --num-gpus 1 \
  --enable-cpu-offload \
  --enable-diffusion-pipeline-profiler \
  --diffusion-attention-config.default.backend FLASH_ATTN \
  --lora-path "${FASTH3_LORA}"
```

After `/health` returns HTTP 200, submit the four-step T2VA request:

```bash
curl --fail-with-body -sS -X POST "http://127.0.0.1:${PORT}/v1/videos/sync" \
  -F 'prompt=A cinematic live-action scene with a clear subject moving naturally; the atmosphere includes synchronized environmental sound.' \
  -F 'width=672' \
  -F 'height=384' \
  -F 'fps=24' \
  -F 'num_inference_steps=4' \
  -F 'seed=3101' \
  -F 'extra_params={"task":"t2va","duration":4.4,"aspect_ratio":"16:9"}' \
  -o fasth3-a100.mp4
```

The historical request produced 107 video frames at 24 FPS. In that controlled
run, one warmup followed by three measured requests had a 25.362-second median
client latency and a 14.094-second median denoise time, with 65,989 MiB sampled
peak HBM. Those numbers describe one A100 80GB PCIe, concurrency one, the pinned
artifacts above, and the compatibility offload path; they are not throughput or
production-performance guarantees. Allow additional host RAM and startup time
for the offloaded checkpoint and the one-time adapter fusion. FastH3's
four-step schedule provides the denoising acceleration; this compatibility
change enables offload and does not establish an additional speedup or
perceptual-quality parity with base H3.

## VSA adapter variants

Use a server command without model-level or layerwise CPU offload. Do not
retain `--enable-cpu-offload`, either layerwise-offload alias, or
`--diffusion-offload-config` from an offloaded recipe.

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
