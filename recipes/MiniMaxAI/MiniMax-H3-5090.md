# MiniMax-H3 on RTX 5090

[Model guide](MiniMax-H3.md) · [Deployment choices](MiniMax-H3.md#choose-a-deployment) · [HTTP API](MiniMax-H3.md#http-api-examples)

This single-GPU configuration streams BF16 DiT and text-encoder blocks from
host memory and loads the VAEs only for their encode/decode phases. VAE tiling
reduces decode activation memory.

## Capacity requirements

| Resource | Requirement |
| --- | --- |
| GPU memory | One RTX 5090 with 32 GiB |
| Checkpoint storage | 135 GiB per partition |
| Available system RAM | Minimum not established; see the host-memory requirements below |

`FL2VA` and `Ref2VA` are separate checkpoint partitions. Start one server at a
time. See the shared
[host-memory requirements](MiniMax-H3-CUDA.md#single-gpu-low-memory-serving):
GPU offload does not establish a 32 GiB system-RAM configuration.

## Single GPU

Complete the [CUDA installation](MiniMax-H3-CUDA.md#installation), then start
one worker with no retained DiT layers:

```bash
vllm serve MiniMaxAI/MiniMax-H3 --omni --trust-remote-code \
  --task-type fl2va \
  --diffusion-offload-config \
  '{"mode":"layer","components":["dit","text_encoder","vae"],"layer_options":{"dit":{"weight_transfer":"rank-local"}}}' \
  --vae-use-tiling --enforce-eager \
  --diffusion-attention-backend CUDNN_ATTN
```

Start with the shared [480P T2VA request](MiniMax-H3-CUDA.md#single-gpu-low-memory-serving).
This single-GPU configuration has not completed target-hardware end-to-end
validation; neither a 768P capacity guarantee nor a latency result is available.
