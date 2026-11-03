# PAN 2 on RTX 5090

This recipe runs [PAN2](PAN2.md) text-to-video and image-to-video on 32 GB GPUs
(GeForce RTX 5090). Two GPUs use tensor parallelism and distributed layerwise
offload, with the guardrail models offloaded between checks. One GPU does not
fit the 481-frame output.

## Capacity

The profile is sized for the longest supported output, 704x1248 and 481
frames. Peak memory is per GPU (first GPU / others).

Distributed layerwise offload keeps the DiT and text encoder in pinned host
memory, so the server needs a large amount of system RAM, and each CFG-parallel
branch adds its own copy. System RAM is the server's peak host memory.

| GPUs | Peak memory (481 frames) | System RAM |
| --- | ---: | ---: |
| 2 GPUs | 29.7 / 29.2 GiB | 193 GiB |

## 2 GPUs

6 resident blocks; 8 peaked at 31.3 GiB, too close to the card's usable memory.

```bash
pip install git+https://github.com/ifm-ai/pan2-guardrail  # PAN 2 guardrails
vllm serve IFM/PAN2 --omni --port 8091 \
  --num-gpus 2 \
  --tensor-parallel-size 2 \
  --vae-patch-parallel-size 2 --vae-parallel-mode tile --vae-use-tiling \
  --enable-distributed-layerwise-offload --dlo-no-use-allgather --dlo-resident-layers 6 \
  --enforce-eager \
  --stage-overrides '{"0": {"model_config": {"offload_guardrail_models": true}}}'
```

Send requests as shown in [PAN2](PAN2.md#requests).
