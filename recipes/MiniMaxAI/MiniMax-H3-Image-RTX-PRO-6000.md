# MiniMax-H3-Image for text-to-image and image editing — RTX PRO 6000 Blackwell

## Summary

- Vendor: MiniMax
- Model: [`MiniMaxAI/MiniMax-H3-Image`](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)
- Task: text-to-image (`t2i`) and reference-image editing (`edit`)
- Mode: online serving through the OpenAI-compatible Images API
- Hardware: NVIDIA RTX PRO 6000 Blackwell Server Edition 96 GB, one, two or four GPUs
- Recommended deployment: 4 GPU, Ulysses SP 4, resident; see the table below for one or two GPUs
- Maintainer: vLLM-Omni MiniMax-H3 maintainers

Every profile below was run on 4× RTX PRO 6000 Blackwell (PCIe). The Ulysses
and TP 2 × SP 2 profiles need the VAE tile-graph capture fix
(`[Bugfix] MiniMax-H3-Image: capture each VAE tile-graph replica on its own
stream`). Without it, a later request on these profiles can hang in the VAE
tile all-gather (the first one completes).

## Choose a profile

| GPUs you have | Profile | Peak / GPU | 2048² t2i | 2048² edit |
| --- | --- | ---: | ---: | ---: |
| 1 | 1 GPU, DiT resident + text-encoder offload | 50.8 GiB | 4.31 s | 6.64 s |
| 1, alternative | 1 GPU, DLO (DiT + text encoder) | 14.6 GiB | 7.80 s | 8.17 s |
| 2 | 2 GPU, DiT TP 2, resident | 49.7 GiB | 2.68 s | 4.38 s |
| 2, alternative | 2 GPU, Ulysses SP 2, resident | 68.1 GiB | 2.70 s | 4.26 s |
| 2, alternative | 2 GPU, Ulysses SP 2 + DLO | 10.0 GiB | 13.86 s | 14.53 s |
| 4 | 4 GPU, Ulysses SP 4, resident | 55.2 GiB | 1.60 s | 2.50 s |
| 4, alternative | 4 GPU, DiT TP 2 × SP 2, resident | 37.3 GiB | 1.92 s | 2.97 s |
| 4, alternative | 4 GPU, DiT TP 4, resident | 28.4 GiB | 2.24 s | 3.57 s |

The checkpoint has 126 GiB of weights: a 61.7 GiB DiT, a 62.1 GiB Qwen3-VL
text encoder and a 2.1 GiB FP32 VAE. Ulysses SP shards the sequence and keeps
a full DiT on every GPU; tensor parallelism shards the DiT itself. When the
weights do not fit, DLO (`--diffusion-offload-config` with `mode: layer`)
keeps them in pinned host memory and streams one block at a time.

On PCIe, DLO's DiT AllGather is the slowest part of a request. The two-GPU
SP 2 + DLO profile is slower than one-GPU DLO, so use it only when each GPU
must keep its footprint near 10 GiB.

## Supported model contract

Same on every accelerator; see the
[H200 recipe](./MiniMax-H3-Image-H200.md#supported-model-contract) for the
parameter table and request examples. In short: `t2i` on
`/v1/images/generations`, `edit` with 1–16 references on `/v1/images/edits`,
six CFG-distilled steps, one image per request, no `negative_prompt`.

## References

- Model card: <https://huggingface.co/MiniMaxAI/MiniMax-H3-Image>
- Other accelerators: [H200](./MiniMax-H3-Image-H200.md), [H100](./MiniMax-H3-Image-H100.md), [B200](./MiniMax-H3-Image-B200.md), [B300](./MiniMax-H3-Image-B300.md)
- Base model recipe: [MiniMax-H3](./MiniMax-H3.md)
- Offload guide: [Distributed layerwise offloading](../../docs/user_guide/diffusion/offloader/distributed_layerwise_offload.md)
- Supported models: [`docs/models/supported_models.md`](../../docs/models/supported_models.md)

## Hardware

- Accelerator: NVIDIA RTX PRO 6000 Blackwell Server Edition 96 GB, Blackwell (sm120), about 94.9 GiB usable per GPU
- Number of devices: 1, 2 or 4 on one node
- Interconnect: PCIe, all GPUs behind one host bridge (`PHB`), no NVLink.
  NCCL's default shared-memory transport measured about 32 GB/s per GPU for
  all-to-all on the qualification host; forcing P2P (`NCCL_P2P_LEVEL=SYS`)
  measured 2.4 GB/s there, so keep the NCCL defaults unless your host's P2P
  is faster.
- Host memory: offload profiles keep pinned host copies of the offloaded
  weights: about 62 GiB for the text encoder and another 62 GiB for the DiT,
  split across ranks when they are sharded
- Qualification scope: measured on the host below

## Software environment

| Item | Version |
| --- | --- |
| OS | Ubuntu 24.04.4 LTS |
| Python | 3.12.16 |
| Driver / CUDA | 580.126.09 / CUDA 13.0 |
| PyTorch | 2.14.0+cu130 |
| vLLM | 0.33.0 |
| Transformers | 5.19.0 |
| vLLM-Omni | source checkout with MiniMax-H3-Image support and the VAE tile-graph capture fix |

The attention backend resolves to `CUDNN_ATTN` on sm120 without a flag.

## Deployment

### One GPU

Full residency needs an estimated 135 GiB (126 GiB of weights plus activations) and does not fit. Offloading only the text encoder keeps the DiT resident at a 50.8 GiB peak.

**1 GPU, DiT resident + text-encoder offload** — recommended. Peak 50.8 GiB. 2048² image in 4.31 s, 2048² edit in 6.64 s.

```bash
vllm serve MiniMaxAI/MiniMax-H3-Image --omni \
  --num-gpus 1 \
  --diffusion-offload-config '{"mode":"layer","components":["text_encoder"]}' \
  --host 0.0.0.0 --port 8000
```

**1 GPU, DLO (DiT + text encoder)**. Peak 14.6 GiB. 2048² image in 7.80 s, 2048² edit in 8.17 s.

```bash
vllm serve MiniMaxAI/MiniMax-H3-Image --omni \
  --num-gpus 1 \
  --diffusion-offload-config '{"mode":"layer","components":["dit","text_encoder"]}' \
  --host 0.0.0.0 --port 8000
```

### Two GPUs

Both resident layouts fit. DiT TP 2 matches Ulysses SP 2 in speed (within 3%) at an 18 GiB lower peak, so it is the default. SP 2 + DLO trades a 5× slower 2048² request for a 10 GiB footprint.

**2 GPU, DiT TP 2, resident** — recommended. Peak 49.7 GiB. 2048² image in 2.68 s, 2048² edit in 4.38 s.

```bash
vllm serve MiniMaxAI/MiniMax-H3-Image --omni \
  --num-gpus 2 --tensor-parallel-size 2 --usp 1 --text-encoder-tp-size 2 \
  --host 0.0.0.0 --port 8000
```

**2 GPU, Ulysses SP 2, resident**. Peak 68.1 GiB. 2048² image in 2.70 s, 2048² edit in 4.26 s.

```bash
vllm serve MiniMaxAI/MiniMax-H3-Image --omni \
  --num-gpus 2 --usp 2 --text-encoder-tp-size 2 \
  --host 0.0.0.0 --port 8000
```

**2 GPU, Ulysses SP 2 + DLO**. Peak 10.0 GiB. 2048² image in 13.86 s, 2048² edit in 14.53 s.

```bash
vllm serve MiniMaxAI/MiniMax-H3-Image --omni \
  --num-gpus 2 --usp 2 --text-encoder-tp-size 2 \
  --diffusion-offload-config '{"mode":"layer","components":["dit","text_encoder"],"layer_options":{"dit":{"weight_transfer":"allgather"},"text_encoder":{"weight_transfer":"rank-local"}}}' \
  --host 0.0.0.0 --port 8000
```

### Four GPUs

Ulysses SP 4 is the fastest four-GPU layout on PCIe: tensor parallelism all-reduces every layer, which costs more here than the Ulysses sequence exchange.

**4 GPU, Ulysses SP 4, resident** — recommended. Peak 55.2 GiB. 2048² image in 1.60 s, 2048² edit in 2.50 s.

```bash
vllm serve MiniMaxAI/MiniMax-H3-Image --omni \
  --num-gpus 4 --usp 4 --text-encoder-tp-size 4 \
  --host 0.0.0.0 --port 8000
```

**4 GPU, DiT TP 2 × SP 2, resident**. Peak 37.3 GiB. 2048² image in 1.92 s, 2048² edit in 2.97 s.

```bash
vllm serve MiniMaxAI/MiniMax-H3-Image --omni \
  --num-gpus 4 --tensor-parallel-size 2 --usp 2 --text-encoder-tp-size 4 \
  --host 0.0.0.0 --port 8000
```

**4 GPU, DiT TP 4, resident**. Peak 28.4 GiB. 2048² image in 2.24 s, 2048² edit in 3.57 s.

```bash
vllm serve MiniMaxAI/MiniMax-H3-Image --omni \
  --num-gpus 4 --tensor-parallel-size 4 --usp 1 --text-encoder-tp-size 4 \
  --host 0.0.0.0 --port 8000
```

For throughput rather than latency, run two independent two-GPU servers
(`CUDA_VISIBLE_DEVICES=0,1` and `2,3`, each with SP 2) behind a load
balancer. In a separate run with one request in flight per server, the pair
served 44.8 1024² t2i images and 28.4 1024² edits per minute, against 38.3
and 24.2 for one SP 4 server.

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

`check.png` should be a 4736×3552 image of the prompt. Send it a few times:
without the VAE tile-graph capture fix, a repeated request can hang.

## Qualification evidence

Workload: the release's example t2i prompt and one-reference edit, `seed=42`,
`generator_device=cpu`, six steps. Each value is the median client
end-to-end time of three requests after one warm-up, at concurrency 1. Peak
memory is the per-GPU `nvidia-smi` maximum sampled every 0.5 s, including the
CUDA context. The GPUs ran nothing else.

| Profile | Peak / GPU | t2i 1024² | t2i 2048² | t2i 4736×3552 | edit 2048² | edit vs SP 4 (PSNR) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 GPU, Ulysses SP 4, resident | 55.2 GiB | 1.57 s | 1.60 s | 6.35 s | 2.50 s | reference |
| 4 GPU, DiT TP 2 × SP 2, resident | 37.3 GiB | 1.89 s | 1.92 s | 7.72 s | 2.97 s | 42.5 dB |
| 4 GPU, DiT TP 4, resident | 28.4 GiB | 2.24 s | 2.24 s | 9.19 s | 3.57 s | 39.9 dB |
| 2 GPU, DiT TP 2, resident | 49.7 GiB | 2.65 s | 2.68 s | 11.96 s | 4.38 s | 41.5 dB |
| 2 GPU, Ulysses SP 2, resident | 68.1 GiB | 2.66 s | 2.70 s | 11.86 s | 4.26 s | 36.7 dB |
| 2 GPU, Ulysses SP 2 + DLO | 10.0 GiB | 13.84 s | 13.86 s | 20.52 s | 14.53 s | 38.8 dB |
| 1 GPU, DiT resident + text-encoder offload | 50.8 GiB | 4.27 s | 4.31 s | 18.83 s | 6.64 s | 38.9 dB |
| 1 GPU, DLO (DiT + text encoder) | 14.6 GiB | 7.80 s | 7.80 s | 18.86 s | 8.17 s | 38.9 dB |

Requests up to 2048² share one canvas, so their latency is nearly constant.

Correctness guard:

- Every layout's 2048² edit was compared with the SP 4 output at the same
  seed. The 2048² t2i outputs of all layouts are within PSNR 21.9–23.4 dB of
  SP 4, the same cross-layout spread as on H200.
- DiT offload is exact: the DiT-resident and DiT-offloaded single-GPU runs
  produce byte-identical t2i and edit images.
- With the VAE tile-graph capture fix, repeated requests at one seed return
  byte-identical ONGs, matching the graphs-off (`--enforce-eager`) output.

## Notes

- DLO uses the compact offload config. With rank-local transfer and no
  resident layers it runs on the ordinary layerwise backend; DiT AllGather
  selects the distributed backend, which stores half of the DiT on each of
  two ranks. The text encoder must stay `rank-local`: under text-encoder TP
  its ranks hold different shards.
- `layer_options.dit.resident_layers` above zero is not qualified.
- `--additional-config` accepts `min_generation_area` (an integer, or `null`
  to generate at the requested size, which must then be a multiple of 32),
  `vae_tile_parallel` (default `true`) and `text_encoder_cudnn_sdp` (default
  `true`; `false` trades some speed for run-to-run determinism). Per request,
  pass `min_generation_area` inside `extra_params`; a top-level field is
  ignored.
- Edits: give references at the canvas size of the requested aspect ratio and
  request that same size; combine several changes into one edit instead of
  chaining edits.

## Supported features

| Feature | Status on this model | Guide |
| --- | --- | --- |
| Ulysses sequence parallel (`--usp`) | Qualified on RTX PRO 6000: 2, 4 | [Sequence parallel](../../docs/user_guide/diffusion/parallelism/sequence_parallel.md) |
| Tensor parallel (`--tensor-parallel-size`) | Qualified on RTX PRO 6000: 2, 4 | [Tensor parallel](../../docs/user_guide/diffusion/parallelism/tensor_parallel.md) |
| Text-encoder TP (`--text-encoder-tp-size`) | Qualified on RTX PRO 6000: 1, 2, 4 | — |
| Layerwise offload / DLO (`mode: layer`) | Qualified on RTX PRO 6000: DiT and text encoder, 1 and 2 GPUs, `resident_layers: 0` | [DLO](../../docs/user_guide/diffusion/offloader/distributed_layerwise_offload.md) |
| VAE tile parallel and CUDA-graph decode | On by default; needs the VAE tile-graph capture fix | [VAE parallelism](../../docs/user_guide/diffusion/parallelism/vae_parallelism.md) |
| Model CPU offload (`mode: module`) | Not qualified | [CPU offload](../../docs/user_guide/diffusion/cpu_offload.md) |
| CFG parallel, HSDP | Unsupported (CFG-distilled; HSDP rejected) | [CFG parallel](../../docs/user_guide/diffusion/parallelism/cfg_parallel.md) |
| Ring attention, cache acceleration, quantization | Not qualified | — |
