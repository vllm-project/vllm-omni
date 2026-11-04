# DMAD MiniMax-H3

Serve DMAD's released four-step and eight-step students through the existing
H3 LoRA manager. All three adapters generate video and stereo audio from text:

| Adapter | Transformer forwards |
| --- | --- |
| `dmad_minimax_h3_4step_lora_critic.safetensors` | 4 |
| `dmad_minimax_h3_4step_full_critic.safetensors` | 4 |
| `dmad_minimax_h3_8step_full_critic.safetensors` | 8 |

The eight-step release is a separately trained student. The loader reads each
artifact's metadata, validates rank/alpha 128 and all 624 factors, and checks
complete binding to the 312 target modules. This recipe uses the official
Diffusers-layout exports; the `_comfyui` exports use a different packed format.

This recipe covers the released DMAD adapters with the original H3 decoder.
For official numerical comparison, use the [reference configuration](Pika-H3-Flex.md#compare-with-official-numerical-results). The [validation section](#tests) includes complex people videos for the released adapters.

## Download

```bash
hf download MiniMaxAI/MiniMax-H3 \
  --revision 42ed227ee7df40d41602854ae760620d6eb651fe \
  --include 'FL2VA/*' --local-dir /models/MiniMax-H3
hf download ZhengmingYu/DMAD \
  --revision b08c2bf3be8e67039c363781e6eaacc9713340cd \
  --include 'minimax_h3/*_critic.safetensors' LICENSE NOTICE \
  --local-dir /models/DMAD
```

Use the original H3 base. A FastH3 or other already distilled base is
incompatible with these adapters. Select an individual adapter file when a
directory contains several variants.

## Serve

This configuration uses four CUDA GPUs, dense BF16 weights, and H3's existing
VAE tiling. GPU count describes the validation configuration, not a universal
minimum-memory requirement. Use the dependencies required by this checkout.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_WORKER_MULTIPROC_METHOD=spawn \
OMP_NUM_THREADS=8 VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400 \
VLLM_OMNI_VAE_LEGACY_TEMPORAL=1 \
vllm serve /models/MiniMax-H3/FL2VA --omni --trust-remote-code \
  --task-type fl2va --num-gpus 4 --tensor-parallel-size 4 \
  --text-encoder-tp-size 4 --usp 1 --ring 1 \
  --vae-patch-parallel-size 4 --vae-parallel-mode tile --vae-use-tiling \
  --enforce-eager --diffusion-attention-backend CUDNN_ATTN \
  --diffusion-quantization-config '{"transformer":null,"text_encoder":null,"video_vae":null,"audio_vae":null}' \
  --lora-path /models/DMAD/minimax_h3/dmad_minimax_h3_4step_full_critic.safetensors \
  --lora-scale 1 --additional-config '{"minimax_h3_vae_audio_overlap":true}' \
  --host 127.0.0.1 --port 8000
```

## Generate and switch variants

For numerical comparison with the official pipeline, use the
[single-GPU reference-numerics configuration](Pika-H3-Flex.md#compare-with-official-numerical-results),
replacing its `--lora-path` with the selected DMAD checkpoint. Keep
`minimax_h3_reference_numerics=true`, `TORCH_SDPA`, eager execution, and
unquantized TP=SP=1. The reference mode preserves BF16 intermediate rounding
and the FP32 video decoder; the optimized four-GPU command above has a
different numerical and performance profile. DMAD latents are rounded to
BF16 before FP32 decoder normalization, matching the released pipeline.

The startup option preloads an adapter; select it explicitly in each request.
Paths refer to files accessible to the server. Four forwards means
`num_inference_steps=4` here; the integration supplies the five sigma points.

```bash
curl --fail-with-body --max-time 1800 http://127.0.0.1:8000/v1/videos/sync \
  --form-string 'prompt=A golden retriever runs along a beach at sunset, waves crashing.' \
  -F seconds=5 -F size=1344x768 -F fps=24 -F seed=42 \
  -F num_inference_steps=4 \
  -F 'lora={"name":"dmad4","path":"/models/DMAD/minimax_h3/dmad_minimax_h3_4step_full_critic.safetensors","scale":1}' \
  -F 'extra_params={"task":"t2va","aspect_ratio":"16:9","flow_shift":12,"audio_flow_shift":2}' \
  -o dmad4.mp4
```

For eight steps, change the adapter path to
`dmad_minimax_h3_8step_full_critic.safetensors`, the name to `dmad8`, and
`num_inference_steps` to `8`. To use the paper's four-step checkpoint, select
`dmad_minimax_h3_4step_lora_critic.safetensors` with four steps. These switches
do not require a server restart. Scale 0 disables the adapter and restores
the base H3 sampling contract.

DMAD uses a shifted uniform grid, video/audio shifts 12/2, and re-noising
after each prediction. One request-owned CPU generator draws the initial
channel-first video latent, initial channel-major audio rows, and subsequent
packed noise. Pika's independent device generators and six-step grid remain
separate. Both full-request and step execution retain their own random state.
Offline step execution must explicitly specify the selected adapter's count.

## Supported boundaries

The released DMAD task is T2VA. Ref2VA, first/last-frame conditioning,
continuation, locked driving audio, latent editing/refinement and ControlNet
are rejected for an active DMAD adapter. Ordinary model-level and layerwise
DiT offload are also rejected by the dynamic LoRA loader. Quantization,
ComfyUI-layout loading, DLO, and the upstream low-VRAM presets require
separate qualification in vLLM-Omni.

## Tests

Local real-weight HTTP validation on four SM120 GPUs covered both four-step
adapters, the eight-step adapter, and switching
back to Pika's six-step FL2VA adapter without restarting. Every MP4 decoded
fully: 124 frames at 24 fps, with 32 kHz stereo audio. Requests with a
mismatched step count, audio shift, or latent refinement returned HTTP 400.

The published DMAD rollout and this sampler also produced identical video
and audio rows for four/eight steps on CPU and CUDA when fed fixed synthetic
velocities. Independent official/native HTTP comparisons additionally cover
all three released adapters in reference mode: the four-step LoRA-critic on
all three complex people scenes, four-step full-critic in the kitchen, and
eight-step full-critic in the dance studio. Text embeddings, every video/audio
denoising step, final latents after the official BF16 cast, and pre-AAC
waveforms match exactly. The [shared numerical report](Pika-H3-Flex.md#complex-people-examples-and-numerical-evidence)
documents RGB tolerances, hardware and scope.

| Adapter and scene | Official / native comparison |
| --- | --- |
| 4-step LoRA-critic: watch repair | [Video](https://github.com/lishunyang12/vllm-omni/releases/download/h3-pika-dmad-validation-20261103/dmad-people-workshop-comparison.mp4) |
| 4-step LoRA-critic: three chefs | [Video](https://github.com/lishunyang12/vllm-omni/releases/download/h3-pika-dmad-validation-20261103/dmad-people-kitchen-comparison.mp4) |
| 4-step LoRA-critic: three dancers | [Video](https://github.com/lishunyang12/vllm-omni/releases/download/h3-pika-dmad-validation-20261103/dmad-people-dance-comparison.mp4) |
| 4-step full-critic: three chefs | [Video](https://github.com/lishunyang12/vllm-omni/releases/download/h3-pika-dmad-validation-20261103/dmad4full-people-kitchen-comparison.mp4) |
| 8-step full-critic: three dancers | [Video](https://github.com/lishunyang12/vllm-omni/releases/download/h3-pika-dmad-validation-20261103/dmad8full-people-dance-comparison.mp4) |

These checks establish the stated numerical bounds for the recorded inputs
and configuration; they do not establish a universal latency, memory or
perceptual-quality claim. Full prompts and requests are retained in the
validation bundle.

The contract tests need the checkout's development dependencies and no model
weights. They cover artifact validation, noise-stream order, request isolation,
four/eight-step execution, and disabling the adapter:

```bash
pytest -q tests/diffusion/models/minimax_h3/test_dmad_h3.py
pytest -q tests/diffusion/models/minimax_h3/test_dmad_h3.py \
  -m 'core_model and cpu' --run-level core_model
```

## References

- [Official weights and release notes](https://huggingface.co/ZhengmingYu/DMAD)
- [Official sampler](https://github.com/Yzmblog/DMAD/blob/e8d7a5dcf2637b1b6b0102085fc0f89e41181e18/dmad_h3/sampling.py)
