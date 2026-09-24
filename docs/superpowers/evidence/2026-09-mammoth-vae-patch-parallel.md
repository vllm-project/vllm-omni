# MammothModa2 VAE patch-parallel validation

This note records the final matched FP16 comparison on 4 × RTX 3090 (24 GiB, driver 570.124.04; PyTorch 2.13.0+cu129, vLLM 0.29.0+cu129). The checkpoint is `bytedance-research/MammothModa2-Preview` at revision `ef5a5e41dbf0de1ef6275586b7580f0d4248b4c6`. The PR is based on vLLM-Omni `4c7a98c26` (including #7134).

## Isolated VAE decode

The benchmark strictly loaded all 244 `gen_vae.*` tensors from `model-00006-of-00008.safetensors`. It decoded the same seeded FP16 Gaussian latent (`[1,16,256,256]`, seed 7, transformed by the checkpoint scaling and shift factors), with one warmup and three timed iterations per mode. Timings include distributed collectives and rank-0 stitching, but not weight loading or warmup. GPU 2↔3 has a `PXB` link.

| 2048² decode | Median | Timed range |
| --- | ---: | ---: |
| PP=1, tiled | 1691.6 ms | 1689.8–1698.6 ms |
| PP=2, tiled | 957.5 ms | 955.7–959.3 ms |

PP=2 was **1.77× faster than the like-for-like tiled PP=1 decode** in this isolated run. Both modes produced bit-identical `[1,3,2048,2048]` FP16 outputs and repeated exactly; the recorded cross-rank tile-boundary error is zero. Peak *allocated* memory was about 2.78 GB on PP=1 rank 0 and 2.77/2.74 GB on PP=2 ranks. Tiling, not PP itself, explains the large memory reduction versus untiled decode. These are three timed samples on one GPU pair, not a general throughput or image-quality claim. The [final raw result](mammoth-vae-2048-4gpu-pxb-cu129-final.json) includes per-iteration times, source hashes, and boundary mapping.

Reproduce from source commit `ed1637810` (or matching source hashes in the JSON), with the pinned config and checkpoint shard present at the shown paths. `VLLM_OMNI_SOURCE_COMMIT` records provenance when the experiment copy has no `.git`; it does not select a revision:

```bash
CUDA_VISIBLE_DEVICES=2,3 VLLM_OMNI_SOURCE_COMMIT=ed1637810f00349fe7fa5c7d20a2372dbad5c0c6 \
  python -m torch.distributed.run --standalone --nproc-per-node=2 \
  --module benchmarks.diffusion.bench_mammoth_vae_patch_parallel \
  --model-config /root/autodl-tmp/mammoth-vae-weights/config.json \
  --weights-shard /root/autodl-tmp/mammoth-vae-weights/model-00006-of-00008.safetensors \
  --latent-size 256 --warmup 1 --iterations 3 \
  --output /root/autodl-tmp/mammoth-vae-weights/real-2048-4gpu-pxb-cu129-final.json
```

## Complete request

AR TP=2 ran on GPUs 0–1. DiT/VAE ran on GPU 2 for tiled PP=1 and GPUs 2–3 for SP=2/VAE PP=2. Both requests used the same pinned Preview weights, FP16, seed 42, prompt `A red fox in snow`, guidance 4, VAE tiling, 2048×2048 output, 20 denoising steps, and a complete 16,513-token AR grid (`max_model_len=18432`). The matched stage configs are [PP=1](mammoth-4gpu-2048-pp1.yaml) and [PP=2](mammoth-4gpu-2048-pp2.yaml).

From a local copy of that checkpoint, run the same offline T2I example with the PP=1 or PP=2 config. The second command adds the two-rank VAE settings:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model /root/autodl-tmp/mammoth-vae-weights \
  --deploy-config docs/superpowers/evidence/mammoth-4gpu-2048-pp1.yaml \
  --prompt "A red fox in snow" --seed 42 --guidance-scale 4 \
  --height 2048 --width 2048 --num-inference-steps 20 --vae-use-tiling \
  --output /root/autodl-tmp/mammoth-vae-weights/pp1.png

python examples/offline_inference/text_to_image/text_to_image.py \
  --model /root/autodl-tmp/mammoth-vae-weights \
  --deploy-config docs/superpowers/evidence/mammoth-4gpu-2048-pp2.yaml \
  --prompt "A red fox in snow" --seed 42 --guidance-scale 4 \
  --height 2048 --width 2048 --num-inference-steps 20 --vae-use-tiling \
  --ulysses-degree 2 --vae-patch-parallel-size 2 \
  --output /root/autodl-tmp/mammoth-vae-weights/pp2.png
```

| Request | PP=1 tiled | PP=2 tiled |
| --- | ---: | ---: |
| Total time | 555.343 s | 543.023 s |
| AR time | 447.834 s | 435.624 s |
| DiT/VAE stage time | 107.200 s | 107.123 s |

The AR token IDs and decoded RGB images were bit-identical in this paired run; the image SHA-256 was `5cdde82c5c04e95de431ffa8c1b9dc772a3be5b2b6610dad4931aaf32f67e9d1`. AR time varied by 12.210 s, while stage time differed by only 0.077 s. **No end-to-end speedup is established.** This is one paired request, not a statistical latency benchmark; its raw logs and images remain on the experiment host. BF16 1024² PP=2 repeatability was not established on this host, so no general BF16 parity claim is made.
