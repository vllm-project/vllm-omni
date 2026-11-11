# MiniMax-H3-Image for text-to-image and image editing — B200

> **Configuration-only.** These commands were executed and measured on H200 ([H200 recipe](./MiniMax-H3-Image-H200.md)), not on B200. Memory and latency figures below
> are H200 measurements used for sizing; run the verification step before relying on a profile.

## Summary

- Vendor: MiniMax
- Model: [`MiniMaxAI/MiniMax-H3-Image`](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)
- Task: text-to-image (`t2i`) and reference-image editing (`edit`)
- Mode: online serving through the OpenAI-compatible Images API
- Hardware: NVIDIA B200 180 GB (HGX), one, two or four GPUs
- Recommended deployment: 4 GPU, Ulysses SP 4, resident when four GPUs are free; see the table below for one or two GPUs
- Maintainer: vLLM-Omni MiniMax-H3 maintainers

## Choose a profile

| GPUs you have | Profile | Peak / GPU on H200 |
| --- | --- | ---: |
| 1 | 1 GPU, fully resident | about 135 GiB (estimate) |
| 1, alternative | 1 GPU, DiT resident + text-encoder offload | 73.4 GiB |
| 1, alternative | 1 GPU, DLO (DiT + text encoder) | 14.7 GiB |
| 2 | 2 GPU, Ulysses SP 2, resident | 94.6 GiB |
| 2, alternative | 2 GPU, Ulysses SP 2 + DLO | 12.0 GiB |
| 4 | 4 GPU, Ulysses SP 4, resident | 81.3 GiB |
| 4, alternative | 4 GPU, DiT TP 4, resident | 36.2 GiB |
| 4, alternative | 4 GPU, DiT TP 2 × SP 2, resident | 52.2 GiB |

The checkpoint has 126 GiB of weights: a 61.7 GiB DiT, a 62.1 GiB Qwen3-VL
text encoder and a 2.1 GiB FP32 VAE. Ulysses SP shards the sequence and keeps
a full DiT on every GPU; tensor parallelism shards the DiT itself. When the
weights do not fit, DLO (`--diffusion-offload-config` with `mode: layer`)
keeps them in pinned host memory and streams one block at a time.

## Supported model contract

Same on every accelerator; see the
[H200 recipe](./MiniMax-H3-Image-H200.md#supported-model-contract) for the
parameter table and request examples. In short: `t2i` on
`/v1/images/generations`, `edit` with 1–16 references on `/v1/images/edits`,
six CFG-distilled steps, one image per request, no `negative_prompt`.

## References

- Model card: <https://huggingface.co/MiniMaxAI/MiniMax-H3-Image>
- Other accelerators: [H200](./MiniMax-H3-Image-H200.md), [H100](./MiniMax-H3-Image-H100.md), [B300](./MiniMax-H3-Image-B300.md), [RTX PRO 6000 Blackwell](./MiniMax-H3-Image-RTX-PRO-6000.md)
- Base model recipe: [MiniMax-H3](./MiniMax-H3.md)
- Offload guide: [Distributed layerwise offloading](../../docs/user_guide/diffusion/offloader/distributed_layerwise_offload.md)
- Supported models: [`docs/models/supported_models.md`](../../docs/models/supported_models.md)

## Hardware

- Accelerator: NVIDIA B200 180 GB (HGX), Blackwell (sm100), about 178.0 GiB usable per GPU
- Number of devices: 1, 2 or 4 on one node
- Interconnect: NVLink
- Host memory: offload profiles keep pinned host copies of the offloaded
  weights: about 62 GiB for the text encoder and another 62 GiB for the DiT,
  split across ranks when they are sharded
- Qualification scope: configuration-only

## Software environment

| Item | Version |
| --- | --- |
| OS | Ubuntu 22.04.5 LTS |
| Python | 3.12.15 |
| Driver / CUDA | 590.48.01 / CUDA 13.0 |
| PyTorch | 2.14.0+cu130 |
| vLLM | 0.33.0 |
| Transformers | 5.19.0 |
| vLLM-Omni | source checkout with MiniMax-H3-Image support, or `vllm/vllm-omni:minimax-h3-image` |

These are the versions used for the H200 measurements.

## Deployment

### One GPU

All weights fit on one GPU: full residency needs an estimated 135 GiB (126 GiB of weights plus activations; not measured). Use the offload profiles only when the GPU is shared.

**1 GPU, fully resident** — recommended.

```bash
vllm serve MiniMaxAI/MiniMax-H3-Image --omni \
  --num-gpus 1 \
  --host 0.0.0.0 --port 8000
```

**1 GPU, DiT resident + text-encoder offload**. Peak on H200: 73.4 GiB per GPU. 2048² image in 3.18 s, 2048² edit in 4.6 s on H200.

```bash
vllm serve MiniMaxAI/MiniMax-H3-Image --omni \
  --num-gpus 1 \
  --diffusion-offload-config '{"mode":"layer","components":["text_encoder"]}' \
  --host 0.0.0.0 --port 8000
```

**1 GPU, DLO (DiT + text encoder)**. Peak on H200: 14.7 GiB per GPU. 2048² image in 8.3 s, 2048² edit in 8.75 s on H200.

```bash
vllm serve MiniMaxAI/MiniMax-H3-Image --omni \
  --num-gpus 1 \
  --diffusion-offload-config '{"mode":"layer","components":["dit","text_encoder"]}' \
  --host 0.0.0.0 --port 8000
```

### Two GPUs with Ulysses SP

The resident Ulysses layout peaks at 94.6 GiB per GPU on H200.

**2 GPU, Ulysses SP 2, resident** — recommended. Peak on H200: 94.6 GiB per GPU. 2048² image in 1.48 s, 2048² edit in 2.36 s on H200.

```bash
vllm serve MiniMaxAI/MiniMax-H3-Image --omni \
  --num-gpus 2 --usp 2 --text-encoder-tp-size 2 \
  --host 0.0.0.0 --port 8000
```

**2 GPU, Ulysses SP 2 + DLO**. Peak on H200: 12.0 GiB per GPU. 2048² image in 5.58 s, 2048² edit in 5.84 s on H200.

```bash
vllm serve MiniMaxAI/MiniMax-H3-Image --omni \
  --num-gpus 2 --usp 2 --text-encoder-tp-size 2 \
  --diffusion-offload-config '{"mode":"layer","components":["dit","text_encoder"],"layer_options":{"dit":{"weight_transfer":"allgather"},"text_encoder":{"weight_transfer":"rank-local"}}}' \
  --host 0.0.0.0 --port 8000
```

### Four GPUs

Ulysses SP 4 is the fastest layout measured on H200.

**4 GPU, Ulysses SP 4, resident** — recommended. Peak on H200: 81.3 GiB per GPU. 2048² image in 0.88 s, 2048² edit in 1.35 s on H200.

```bash
vllm serve MiniMaxAI/MiniMax-H3-Image --omni \
  --num-gpus 4 --usp 4 --text-encoder-tp-size 4 \
  --host 0.0.0.0 --port 8000
```

**4 GPU, DiT TP 4, resident**. Peak on H200: 36.2 GiB per GPU. 2048² image in 1.0 s, 2048² edit in 1.51 s on H200.

```bash
vllm serve MiniMaxAI/MiniMax-H3-Image --omni \
  --num-gpus 4 --tensor-parallel-size 4 --usp 1 --text-encoder-tp-size 4 \
  --host 0.0.0.0 --port 8000
```

**4 GPU, DiT TP 2 × SP 2, resident**. Peak on H200: 52.2 GiB per GPU. 2048² image in 1.09 s, 2048² edit in 1.51 s on H200.

```bash
vllm serve MiniMaxAI/MiniMax-H3-Image --omni \
  --num-gpus 4 --tensor-parallel-size 2 --usp 2 --text-encoder-tp-size 4 \
  --host 0.0.0.0 --port 8000
```

`--text-encoder-tp-size` must divide Qwen3-VL's 64 query and 8 KV heads and
must not exceed the number of GPUs. Keep CUDA graphs enabled (do not pass
`--enforce-eager`): the VAE decode tiles are graph-replayed and spread over
all ranks.

## Verification

```bash
curl -s http://localhost:8000/v1/models | jq -r '.data[].id'
# MiniMaxAI/MiniMax-H3-Image

curl -s http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model": "MiniMaxAI/MiniMax-H3-Image", "prompt": "A red bicycle against a white wall",
       "size": "4736x3552", "seed": 42, "response_format": "b64_json"}' \
  | jq -r '.data[0].b64_json' | base64 -d > check.png
```

`check.png` should be a 4736×3552 image of the prompt. This is the largest
request measured, so watch `nvidia-smi` during it to confirm the memory
headroom on your host.

## Notes

- DLO uses the compact offload config. With rank-local transfer and no
  resident layers it runs on the ordinary layerwise backend; DiT AllGather
  selects the distributed backend, which stores half of the DiT on each of
  two ranks and was faster on H200 (5.6 s against 8.6 s for rank-local).
  The text encoder must stay `rank-local`: under text-encoder TP its ranks
  hold different shards.
- `layer_options.dit.resident_layers` above zero runs but is not qualified:
  24 resident blocks took about 15 s per step on H200, slower than streaming
  every block.
- `--additional-config` accepts `min_generation_area` (an integer, or `null`
  to generate at the requested size, which must then be a multiple of 32),
  `vae_tile_parallel` (default `true`) and `text_encoder_cudnn_sdp` (default
  `true`; `false` trades some speed for run-to-run determinism).
- Edits: give references at the canvas size of the requested aspect ratio and
  request that same size; combine several changes into one edit instead of
  chaining edits.

## Supported features

| Feature | Status on this model | Guide |
| --- | --- | --- |
| Ulysses sequence parallel (`--usp`) | Qualified on H200: 2, 4 | [Sequence parallel](../../docs/user_guide/diffusion/parallelism/sequence_parallel.md) |
| Tensor parallel (`--tensor-parallel-size`) | Qualified on H200: 2, 4 | [Tensor parallel](../../docs/user_guide/diffusion/parallelism/tensor_parallel.md) |
| Text-encoder TP (`--text-encoder-tp-size`) | Qualified on H200: 1, 2, 4 | — |
| Layerwise offload / DLO (`mode: layer`) | Qualified on H200: DiT and text encoder, 1 and 2 GPUs, `resident_layers: 0` | [DLO](../../docs/user_guide/diffusion/offloader/distributed_layerwise_offload.md) |
| VAE tile parallel and CUDA-graph decode | On by default | [VAE parallelism](../../docs/user_guide/diffusion/parallelism/vae_parallelism.md) |
| Model CPU offload (`mode: module`) | Not qualified | [CPU offload](../../docs/user_guide/diffusion/cpu_offload.md) |
| CFG parallel, HSDP | Unsupported (CFG-distilled; HSDP rejected) | [CFG parallel](../../docs/user_guide/diffusion/parallelism/cfg_parallel.md) |
| Ring attention, cache acceleration, quantization | Not qualified | — |
