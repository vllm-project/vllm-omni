# PAN 2 on RTX PRO 6000 Blackwell

This recipe runs [PAN2](PAN2.md) text-to-video and image-to-video on 96 GB GPUs
(RTX PRO 6000 Blackwell (96 GB)). One GPU uses distributed layerwise offload;
two GPUs use tensor parallelism.

## Capacity

The profiles are sized for the longest supported output, 704x1248 and 481
frames. Peak memory is per GPU (first GPU / others).

| GPUs | Peak memory (481 frames) | System RAM |
| --- | ---: | ---: |
| 1 GPU | 84.4 GiB | 156 GiB |
| 2 GPUs | 84.5 / 72.7 GiB | 28 GiB |

One 50-step request takes longer than the server's default 600 s timeouts, so
both commands raise `VLLM_OMNI_VIDEO_SYNC_TIMEOUT` and
`VLLM_OMNI_DLO_DP_WAVE_TIMEOUT`:

| Output | 1 GPU | 2 GPUs |
| --- | ---: | ---: |
| 10 s (241 frames) | 18 min | 10 min |
| 13 s (313 frames) | 27 min | 15 min |
| 20 s (481 frames) | 53 min | 29 min |

## 1 GPU

All 48 DiT blocks stay on the GPU; the text encoder streams from pinned host
memory, so the server needs a large amount of system RAM. Without offload the
peak was 97.9 GiB, which does not fit. The guardrail models move to the GPU
only for their checks.

```bash
pip install git+https://github.com/ifm-ai/pan2-guardrail  # PAN 2 guardrails
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400 VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=14400 \
vllm serve IFM/PAN2 --omni --port 8091 \
  --vae-use-tiling \
  --enable-distributed-layerwise-offload --dlo-no-use-allgather --dlo-resident-layers 48 \
  --stage-overrides '{"0": {"model_config": {"offload_guardrail_models": true}}}'
```

## 2 GPUs

Tensor parallelism halves the DiT on each GPU; the guardrails stay on the first
GPU. CFG parallelism (`--cfg-parallel-size 2`) does not fit: it peaked at
102.0 GiB.

```bash
pip install git+https://github.com/ifm-ai/pan2-guardrail  # PAN 2 guardrails
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400 VLLM_OMNI_DLO_DP_WAVE_TIMEOUT=14400 \
vllm serve IFM/PAN2 --omni --port 8091 \
  --num-gpus 2 \
  --tensor-parallel-size 2 \
  --vae-patch-parallel-size 2 --vae-parallel-mode tile
```

Send requests as shown in [PAN2](PAN2.md#requests).
