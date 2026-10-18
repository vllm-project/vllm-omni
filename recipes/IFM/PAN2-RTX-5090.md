# PAN 2 on RTX 5090

This recipe runs [PAN2](PAN2.md) text-to-video and image-to-video on 32 GB GPUs
(GeForce RTX 5090). All profiles use distributed layerwise offload with the
guardrail models offloaded between checks.

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
| 1 GPU | 28.3 GiB |
| 2 GPUs | 27.7 / 24.3 GiB |

## 1 GPU

No resident blocks. With 2 resident blocks the peak was 29.3 GiB, about 2.5 GiB
below the card's usable memory.

```bash
pip install git+https://github.com/ifm-ai/pan2-guardrail  # PAN 2 guardrails
vllm serve IFM/PAN2 --omni --port 8091 \
  --vae-use-tiling \
  --enable-distributed-layerwise-offload --dlo-no-use-allgather --dlo-resident-layers 0 \
  --enforce-eager \
  --stage-overrides '{"0": {"model_config": {"offload_guardrail_models": true}}}'
```

## 2 GPUs

8 resident blocks; 16 peaked at 30.5 GiB. This profile was also checked with
image-to-video through `vllm serve`.

```bash
pip install git+https://github.com/ifm-ai/pan2-guardrail  # PAN 2 guardrails
vllm serve IFM/PAN2 --omni --port 8091 \
  --num-gpus 2 \
  --tensor-parallel-size 2 \
  --vae-patch-parallel-size 2 --vae-parallel-mode tile --vae-use-tiling \
  --enable-distributed-layerwise-offload --dlo-no-use-allgather --dlo-resident-layers 8 \
  --enforce-eager \
  --stage-overrides '{"0": {"model_config": {"offload_guardrail_models": true}}}'
```

Send requests as shown in [PAN2](PAN2.md#requests).
