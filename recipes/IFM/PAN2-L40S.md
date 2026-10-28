# PAN 2 on L40S and 48 GB GPUs

This recipe runs [PAN2](PAN2.md) text-to-video and image-to-video on 48 GB GPUs
(L40S, RTX 6000 Ada, RTX A6000). All profiles use distributed layerwise offload;
tensor parallelism halves the streamed blocks per GPU.

## Capacity

The profiles are sized for the longest supported output, 704x1248 and 481
frames. Peak memory is per GPU (first GPU / others).

Distributed layerwise offload keeps the DiT and text encoder in pinned host
memory, so the server needs a large amount of system RAM, and each CFG-parallel
branch adds its own copy. Raising `--dlo-resident-layers` keeps more DiT blocks
on the GPU, which reduces host-to-GPU traffic, but uses more GPU memory;
re-measure before raising it.

| GPUs | Peak memory (481 frames) |
| --- | ---: |
| 1 GPU | 40.1 GiB |
| 4 GPUs | 35.3 / 34.7 GiB |

## 1 GPU

10 of the 48 DiT blocks stay on the GPU.

```bash
pip install git+https://github.com/ifm-ai/pan2-guardrail  # PAN 2 guardrails
vllm serve IFM/PAN2 --omni --port 8091 \
  --vae-use-tiling \
  --enable-distributed-layerwise-offload --dlo-no-use-allgather --dlo-resident-layers 10 \
  --enforce-eager \
  --stage-overrides '{"0": {"model_config": {"offload_guardrail_models": true}}}'
```

## 4 GPUs

12 of the 48 DiT blocks stay on each GPU.

```bash
pip install git+https://github.com/ifm-ai/pan2-guardrail  # PAN 2 guardrails
vllm serve IFM/PAN2 --omni --port 8091 \
  --num-gpus 4 \
  --tensor-parallel-size 2 --cfg-parallel-size 2 \
  --vae-patch-parallel-size 4 --vae-parallel-mode tile --vae-use-tiling \
  --enable-distributed-layerwise-offload --dlo-no-use-allgather --dlo-resident-layers 12 \
  --enforce-eager \
  --stage-overrides '{"0": {"model_config": {"offload_guardrail_models": true}}}'
```

Send requests as shown in [PAN2](PAN2.md#requests).
