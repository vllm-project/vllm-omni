# PAN 2 on RTX PRO 6000 Blackwell

This recipe runs [PAN2](PAN2.md) text-to-video and image-to-video on 96 GB GPUs
(RTX PRO 6000 Blackwell (96 GB)). One GPU uses distributed layerwise offload;
two GPUs use tensor parallelism.

## Capacity

The profiles are sized for the longest supported output, 704x1248 and 481
frames. Peak memory is per GPU (first GPU / others).

| GPUs | Peak memory (481 frames) |
| --- | ---: |
| 1 GPU | 87.0 GiB |
| 2 GPUs | 87.0 / 73.6 GiB |

## 1 GPU

44 of the 48 DiT blocks stay on the GPU; the rest stream from pinned host
memory, so the server needs a large amount of system RAM. Without offload the
peak was 97.9 GiB, which does not fit. The guardrail models move to the GPU
only for their checks.

```bash
pip install git+https://github.com/ifm-ai/pan2-guardrail  # PAN 2 guardrails
vllm serve IFM/PAN2 --omni --port 8091 \
  --vae-use-tiling \
  --enable-distributed-layerwise-offload --dlo-no-use-allgather --dlo-resident-layers 44 \
  --enforce-eager \
  --stage-overrides '{"0": {"model_config": {"offload_guardrail_models": true}}}'
```

## 2 GPUs

Tensor parallelism halves the DiT on each GPU; the guardrails stay on the first
GPU. CFG parallelism (`--cfg-parallel-size 2`) does not fit: it peaked at
102.0 GiB.

```bash
pip install git+https://github.com/ifm-ai/pan2-guardrail  # PAN 2 guardrails
vllm serve IFM/PAN2 --omni --port 8091 \
  --num-gpus 2 \
  --tensor-parallel-size 2 \
  --vae-patch-parallel-size 2 --vae-parallel-mode tile
```

Send requests as shown in [PAN2](PAN2.md#requests).
