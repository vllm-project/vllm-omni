# MiniMax-H3-Image on Intel Arc Pro B70 (XPU)

## Summary

- Vendor: MiniMax
- Model: [`MiniMaxAI/MiniMax-H3-Image`](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)
- Task: text-to-image (`t2i`) and reference-image editing (`edit`)
- Mode: online serving through the OpenAI-compatible Images API
- Hardware: Intel Arc Pro B70 32 GB (Xe2), one, two or four GPUs
- Recommended deployment: 4 GPU, DiT TP 4 with the text encoder layer-offloaded
- Qualification scope: measured on 4x Arc Pro B70 (one node)

## Choose a profile

The checkpoint has 126 GiB of weights: a 61.7 GiB DiT, a 62.1 GiB Qwen3-VL
text encoder and a 2.1 GiB FP32 VAE. A B70 has about 31.9 GiB, so no fully
resident layout fits: DiT TP 4 alone holds about 15.4 GiB of DiT weights per
card and the text encoder must be offloaded. Ulysses SP keeps a full DiT on
every card and is used only together with DLO.

| GPUs | Profile | Extra flags | Peak / GPU | 2048² t2i | 2048² edit |
| --- | --- | --- | ---: | ---: | ---: |
| 4 | DiT TP 4, text encoder offloaded (recommended) | `--num-gpus 4 --tensor-parallel-size 4 --usp 1 --text-encoder-tp-size 4 --diffusion-offload-config '{"mode":"layer","components":["text_encoder"]}'` | 27.7 GiB | 5.6 s | 9.2 s |
| 2 | Ulysses SP 2 + DLO (DiT AllGather) | `--num-gpus 2 --usp 2 --text-encoder-tp-size 2 --diffusion-offload-config '{"mode":"layer","components":["dit","text_encoder"],"layer_options":{"dit":{"weight_transfer":"allgather"},"text_encoder":{"weight_transfer":"rank-local"}}}'` | 16.1 GiB | 15.5 s | 21.0 s |
| 1 | DLO, DiT and text encoder | `--num-gpus 1 --diffusion-offload-config '{"mode":"layer","components":["dit","text_encoder"]}'` | 16.3 GiB | 17.2 s | 25.4 s |

DiT TP 4 with a resident TP 4 text encoder (about 31 GiB of weights per card)
and DiT TP 2 (about 31 GiB of DiT weights per card) do not leave room for
activations and were not run. With the text encoder offloaded,
`--text-encoder-tp-size` must equal `--num-gpus`: ranks without a text-encoder
shard reject the offload config at startup.

## Software environment

| Item | Version |
| --- | --- |
| Base image | `vllm/vllm-openai-xpu` 0.33.0, built into the vLLM-Omni image with `docker/Dockerfile.xpu` |
| PyTorch | 2.14.0+xpu |
| vLLM | 0.33.0 |
| Transformers | 5.19.0 |
| Diffusers | 0.40.0 |
| Host | Xeon Platinum 8468V, Linux 6.17, 1 TiB RAM |

## Deployment

Build the image from the repository root and start a container with the GPUs
and the Hugging Face cache mounted:

```bash
docker build -f docker/Dockerfile.xpu --target vllm-omni -t vllm-omni-xpu .

docker run -d --name minimax-h3-image-xpu \
  --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  --ipc=host --net=host --shm-size 64g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm-omni-xpu sleep infinity
```

Download the release once (126 GiB). The `*_diffusers` folders are a second
copy of the weights in Diffusers format and are not read by vLLM-Omni:

```bash
docker exec minimax-h3-image-xpu hf download MiniMaxAI/MiniMax-H3-Image \
  --exclude "transformer_diffusers/*" "vae_diffusers/*" "assets/*"
```

The release bundles `text_encoder/` (the same shards as
`MiniMaxAI/MiniMax-H3/text_encoder`), so the base repository is not needed.

**4 GPU, DiT TP 4, text encoder offloaded** — recommended:

```bash
docker exec -it minimax-h3-image-xpu vllm serve MiniMaxAI/MiniMax-H3-Image --omni \
  --num-gpus 4 --tensor-parallel-size 4 --usp 1 --text-encoder-tp-size 4 \
  --diffusion-offload-config '{"mode":"layer","components":["text_encoder"]}' \
  --host 0.0.0.0 --port 8000
```

**2 GPU, Ulysses SP 2 + DLO**:

```bash
docker exec -it minimax-h3-image-xpu vllm serve MiniMaxAI/MiniMax-H3-Image --omni \
  --num-gpus 2 --usp 2 --text-encoder-tp-size 2 \
  --diffusion-offload-config '{"mode":"layer","components":["dit","text_encoder"],"layer_options":{"dit":{"weight_transfer":"allgather"},"text_encoder":{"weight_transfer":"rank-local"}}}' \
  --host 0.0.0.0 --port 8000
```

**1 GPU, DLO**:

```bash
docker exec -it minimax-h3-image-xpu vllm serve MiniMaxAI/MiniMax-H3-Image --omni \
  --num-gpus 1 \
  --diffusion-offload-config '{"mode":"layer","components":["dit","text_encoder"]}' \
  --host 0.0.0.0 --port 8000
```

Pass `-e ZE_AFFINITY_MASK=<card ids>` to `docker run` to pick the cards. The
offload profiles keep pinned host copies of the offloaded weights (about
62 GiB for the text encoder and another 62 GiB for the DiT, split across ranks
when sharded). The servers were ready 95 s (4 GPU) and 130 s (1 and 2 GPU)
after launch with a warm page cache.

## Request examples

The request contract is the same on every accelerator; see the
[H200 recipe](./MiniMax-H3-Image-H200.md#supported-model-contract). Text to
image:

```bash
curl -s http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model": "MiniMaxAI/MiniMax-H3-Image",
       "prompt": "Documentary photograph of a lighthouse keeper on a cliff at dawn, Kodak Portra 400",
       "size": "2048x2048", "seed": 42, "response_format": "b64_json"}' \
  | jq -r '.data[0].b64_json' | base64 -d > t2i.png
```

For editing, the release's `examples/edit.py` sends `examples/reference1.png`
with `examples/prompt.edit.txt` (standard library only):

```bash
hf download MiniMaxAI/MiniMax-H3-Image --include "examples/*" --local-dir h3-image
python3 h3-image/examples/edit.py --url http://127.0.0.1:8000 --size 2048x2048 --out edit.png
```

Edit prompts follow the release's structured format (name each reference as
`<Picture N>`, list what is preserved and what changes). A one-line
instruction gives visibly worse edits on every accelerator.

## Measured latency

Client end-to-end time, median of three requests after one warm-up request,
concurrency 1. Text-to-image uses the lighthouse prompt above; edits use the
release example (one 1024² reference). 4096² is one request.

| Profile | t2i 1024² | t2i 2048² | edit 2048² | edit 4096² | Peak / GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 GPU, DiT TP 4, text encoder offloaded | 5.48 s | 5.60 s | 9.22 s | 26.7 s | 27.7 GiB |
| 2 GPU, SP 2 + DLO (AllGather) | 15.41 s | 15.54 s | 20.97 s | 52.0 s | 16.1 GiB |
| 1 GPU, DLO | 16.99 s | 17.19 s | 25.40 s | 83.5 s | 16.3 GiB |

Peak memory is the `xpu-smi` device memory in use, sampled once per second.

## Notes

- The VAE tile decode runs eagerly on XPU; CUDA-graph decode
  (`enable_cuda_graph_decode`) applies to CUDA and ROCm devices only.
- Same-seed requests are not bit-reproducible on XPU. In the tested runtime,
  oneDNN BF16 GEMMs with a narrow output (for example the text encoder's
  per-rank K/V projections, N = 256 at TP 4) are not bitwise deterministic run
  to run (about one BF16 ulp), and the text encoder's large activations carry
  that into different images for the same seed. `torch.use_deterministic_algorithms(True)`
  makes those GEMMs repeatable but is not wired into the server.
- Not qualified on XPU: Ring attention, cache acceleration, quantization,
  `resident_layers` above zero. CFG parallel and HSDP are rejected for this
  model on every accelerator.

## References

- Model card: <https://huggingface.co/MiniMaxAI/MiniMax-H3-Image>
- Other accelerators: [H200](./MiniMax-H3-Image-H200.md), [H100](./MiniMax-H3-Image-H100.md), [RTX PRO 6000 Blackwell](./MiniMax-H3-Image-RTX-PRO-6000.md), [AMD Instinct](./MiniMax-H3-Image-AMD.md)
- Offload guide: [Distributed layerwise offloading](../../docs/user_guide/diffusion/offloader/distributed_layerwise_offload.md)
- XPU installation: [Intel GPU](../../docs/getting_started/installation/gpu/xpu.inc.md)
