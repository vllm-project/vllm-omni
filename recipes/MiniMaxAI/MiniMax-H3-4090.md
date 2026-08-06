# MiniMax-H3 on RTX 4090

This recipe uses BF16 weights, tiled VAE decode, tensor parallelism across two
GPUs, and distributed layerwise offload (DLO). It is the 24 GiB sibling of
[MiniMax-H3-5090.md](MiniMax-H3-5090.md): the topology is identical, but the
resident DiT-block count is lower so the per-rank peak fits in 24 GiB. Lower
resident counts reduce HBM use and increase CPU-to-GPU transfer time.

## Capacity requirements

| Resource | Two RTX 4090s |
| --- | ---: |
| GPU HBM | 24 GiB per GPU |
| Checkpoint storage | 135 GiB per partition |
| Available system RAM | 200 GiB minimum |
| Recommended system RAM | 384 GiB |

`FL2VA` and `Ref2VA` are separate 135 GiB checkpoint partitions. Start one
server at a time. DLO keeps rank-local weights in pinned host memory; increasing
`--dlo-resident-layers` improves latency but does **not** reduce host RAM in the
current implementation because resident layers retain pinned CPU master copies.

A single RTX 4090 is not covered here. The one-GPU DLO profile in the RTX 5090
recipe peaked at 26.50 GiB with 12 resident layers, which exceeds 24 GiB. On one
4090, use the model-level CPU offload command in
[MiniMax-H3.md](MiniMax-H3.md#single-gpu-accuracy-and-memory-first) instead, or
lower the resident count and re-measure before trusting it.

> **Modular H3:** after #5720 lands, preserve this recipe's one-partition
> behavior with `--task-type fl2va` or `--task-type ref2va`.

## Two RTX 4090s: 1024x576, 5 seconds
## Two RTX 4090s: 1024x576, 5 seconds

Use TP2 and 12 resident DiT layers. This is the `rtx4090` profile from the main
recipe. A 1024x576 capacity run of this profile peaked at 18,888 MiB per rank on
two B300 ranks; the B300 numbers are an allocation and correctness proxy, not an
RTX 4090 latency claim.

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400

CUDA_VISIBLE_DEVICES=0,1 vllm serve /path/to/MiniMax-H3/FL2VA \
  --omni --trust-remote-code --host 0.0.0.0 --port 8000 \
  --num-gpus 2 --tensor-parallel-size 2 --text-encoder-tp-size 2 \
  --usp 1 --ring 1 --vae-patch-parallel-size 2 \
  --vae-parallel-mode tile --vae-use-tiling \
  --enable-distributed-layerwise-offload --dlo-no-use-allgather \
  --dlo-resident-layers 12 --enforce-eager \
  --diffusion-attention-backend CUDNN_ATTN
```

## request example

*request curl must add -F 'aspect_ratio=16:9'
```bash
curl -sS --max-time 1800 -X POST "http://yourinferenceurl/v1/videos/sync" \
  -F 'model=minimax-h3-4090' \
  -F 'prompt=傍晚小厨房的真人实拍的手 与手绘发光2d动画 融合在一起的影像。夕阳余晖残留在窗边，生活感十足的小厨房里有旧木桌、洗到一半的马克杯、起雾的玻璃瓶、悬挂的抹布。画面带有智能手机单手拍摄的手抖、近距离对焦的犹豫、逆光曝光波动。要像在家中慌忙拍下某个不可思议事件的自然质感，不要广告影像的精心整理。声音只用厨房环境声与手绘生物柔和的电子音、小小的叫声。' \
  -F 'aspect_ratio=16:9' \
  -F 'width=1280' -F 'height=720' -F 'fps=24' \
  -F 'num_inference_steps=60' -F 'flow_shift=12' -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":8,"audio_flow_shift":3.0}' \
  -o "out_t2va.mp4"
```
its need about 25 mins to create finish.(4090*2)
