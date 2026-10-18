# PAN 2 on RTX PRO 6000 Blackwell

This recipe runs [PAN2](PAN2.md) text-to-video and image-to-video on 96 GB GPUs
(RTX PRO 6000 Blackwell (96 GB)). The DiT fits on one 96 GB GPU once the
guardrail models are offloaded to the CPU between checks.

## Capacity

The profiles are sized for the longest supported output, 704x1248 and 481
frames. Peak memory is per GPU (first GPU / others).

| GPUs | Peak memory (481 frames) |
| --- | ---: |
| 1 GPU | 92.0 GiB |
| 2 GPUs | 79.1 / 65.6 GiB |

## 1 GPU

No offload. The guardrail models move to the GPU only for their checks.

```bash
pip install git+https://github.com/ifm-ai/pan2-guardrail  # PAN 2 guardrails
vllm serve IFM/PAN2 --omni --port 8091 \
  --stage-overrides '{"0": {"model_config": {"offload_guardrail_models": true}}}'
```

## 2 GPUs

Tensor parallelism halves the DiT on each GPU; the guardrails stay on the first
GPU. CFG parallelism (`--cfg-parallel-size 2` with the guardrails offloaded)
also fits, but peaked at 93.4 GiB, leaving about 2 GiB of headroom.

```bash
pip install git+https://github.com/ifm-ai/pan2-guardrail  # PAN 2 guardrails
vllm serve IFM/PAN2 --omni --port 8091 \
  --num-gpus 2 \
  --tensor-parallel-size 2 \
  --vae-patch-parallel-size 2 --vae-parallel-mode tile
```

Send requests as shown in [PAN2](PAN2.md#requests).
