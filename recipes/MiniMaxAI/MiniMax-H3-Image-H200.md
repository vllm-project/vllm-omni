# MiniMax-H3-Image for text-to-image and image editing — H200

## Summary

- Vendor: MiniMax
- Model: [`MiniMaxAI/MiniMax-H3-Image`](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)
- Task: text-to-image (`t2i`) and reference-image editing (`edit`)
- Mode: online serving through the OpenAI-compatible Images API
- Hardware: NVIDIA H200 141 GB (SXM, HGX), one, two or four GPUs
- Recommended deployment: 4 GPU, Ulysses SP 4, resident when four GPUs are free; see the table below for one or two GPUs
- Maintainer: vLLM-Omni MiniMax-H3 maintainers

## Choose a profile

| GPUs you have | Profile | Peak / GPU on H200 |
| --- | --- | ---: |
| 1 | 1 GPU, DiT resident + text-encoder offload | 73.4 GiB |
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

| Task | Endpoint | Inputs | Output |
| --- | --- | --- | --- |
| `t2i` | `POST /v1/images/generations` (JSON) | prompt | one image |
| `edit` | `POST /v1/images/edits` (multipart) | prompt and 1–16 reference images, one `image` part each | one image |

A request with reference images is an edit; there is no task flag.

| Parameter | Contract |
| --- | --- |
| `size` | Any `WxH`; default `1024x1024`. Smaller requests are generated on a canvas of at least 2048×2048 pixels (same aspect ratio, sides on a 32-pixel grid) and resized back with LANCZOS. |
| `num_inference_steps` | Omit, or `6`. The release ships one fused output head per step. |
| `guidance_scale` | Omit, or `1`. The model is CFG-distilled. |
| `negative_prompt` | Not supported. |
| `n` | `1`. |
| `seed` | Optional. `generator_device=cpu` makes the noise independent of the GPU. |
| `extra_args.min_generation_area` | Optional per-request canvas area, 32² to 4096² pixels. |

References are presented to the text encoder as `Picture 1`, `Picture 2`, …
in upload order, so prompts can name them, for example `<Picture 1> is the
cabin. Only the season changes to winter.` Requesting the canvas size skips
the final resize and keeps edits best aligned with their reference:

| Aspect ratio | 1:1 | 16:9 | 9:16 | 4:3 | 3:2 | 21:9 |
| --- | --- | --- | --- | --- | --- | --- |
| Size | 2048x2048 | 2720x1536 | 1536x2720 | 2368x1760 | 2496x1664 | 3136x1344 |

### Request examples

```bash
curl -s http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model": "MiniMaxAI/MiniMax-H3-Image",
       "prompt": "Documentary photograph of a lighthouse keeper on a cliff at dawn, Kodak Portra 400",
       "size": "2720x1536", "seed": 42, "response_format": "b64_json"}' \
  | jq -r '.data[0].b64_json' | base64 -d > t2i.png

curl -s http://localhost:8000/v1/images/edits \
  -F model=MiniMaxAI/MiniMax-H3-Image \
  -F image=@cabin.png \
  -F 'prompt=<Picture 1> is a cabin by a lake. Everything is preserved exactly; only the season changes to a snowy winter evening.' \
  -F size=2720x1536 -F seed=42 -F response_format=b64_json \
  | jq -r '.data[0].b64_json' | base64 -d > edit.png
```

Repeat `-F image=@...` for up to 16 references, in the order the prompt
names them.

## References

- Model card: <https://huggingface.co/MiniMaxAI/MiniMax-H3-Image>
- Other accelerators: [H100](./MiniMax-H3-Image-H100.md), [B200](./MiniMax-H3-Image-B200.md), [B300](./MiniMax-H3-Image-B300.md), [RTX PRO 6000 Blackwell](./MiniMax-H3-Image-RTX-PRO-6000.md)
- Base model recipe: [MiniMax-H3](./MiniMax-H3.md)
- Offload guide: [Distributed layerwise offloading](../../docs/user_guide/diffusion/offloader/distributed_layerwise_offload.md)
- Supported models: [`docs/models/supported_models.md`](../../docs/models/supported_models.md)

## Hardware

- Accelerator: NVIDIA H200 141 GB (SXM, HGX), Hopper (sm90), about 139.8 GiB usable per GPU
- Number of devices: 1, 2 or 4 on one node
- Interconnect: NVLink
- Host memory: offload profiles keep pinned host copies of the offloaded
  weights: about 62 GiB for the text encoder and another 62 GiB for the DiT,
  split across ranks when they are sharded
- Qualification scope: runtime-qualified

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

## Deployment

### One GPU

Full residency needs an estimated 135 GiB (126 GiB of weights plus activations; not measured), which leaves almost no headroom on a 139.8 GiB H200, so it is not a provided profile. Offloading only the text encoder keeps the DiT resident at 73.4 GiB.

**1 GPU, DiT resident + text-encoder offload** — recommended. Peak on H200: 73.4 GiB per GPU. 2048² image in 3.18 s, 2048² edit in 4.6 s on H200.

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

The resident Ulysses layout peaks at 94.6 GiB per GPU.

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

## Qualification evidence

Workload: the release's example t2i prompt and one-reference edit, `seed=42`,
`generator_device=cpu`, six steps. Each value is the median client
end-to-end time of three requests after one warm-up, at concurrency 1. Peak
memory is the per-GPU `nvidia-smi` maximum sampled every 0.5 s, including the
CUDA context. The GPUs also held an idle MiniMax-H3 server (about 37 GiB each)
that did no work; its memory is subtracted.

| Profile | Peak / GPU | t2i 1024² | t2i 2048² | t2i 4736×3552 | edit 2048² | edit vs SP 4 (PSNR) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 GPU, Ulysses SP 4, resident | 81.3 GiB | 0.77 s | 0.88 s | 3.84 s | 1.35 s | reference |
| 4 GPU, DiT TP 4, resident | 36.2 GiB | 0.91 s | 1.00 s | 4.01 s | 1.51 s | 38.3 dB |
| 4 GPU, DiT TP 2 × SP 2, resident | 52.2 GiB | 0.96 s | 1.09 s | 4.07 s | 1.51 s | 37.1 dB |
| 2 GPU, Ulysses SP 2, resident | 94.6 GiB | 1.36 s | 1.48 s | 6.82 s | 2.36 s | 37.2 dB |
| 2 GPU, DiT TP 2, resident | 63.3 GiB | 1.38 s | 1.49 s | 6.75 s | 2.34 s | 36.8 dB |
| 2 GPU, Ulysses SP 2 + DLO | 12.0 GiB | 5.48 s | 5.58 s | 9.34 s | 5.84 s | 29.8 dB* |
| 1 GPU, DiT resident + text-encoder offload | 73.4 GiB | 3.08 s | 3.18 s | 12.39 s | 4.60 s | 33.3 dB |
| 1 GPU, DLO (DiT + text encoder) | 14.7 GiB | 8.25 s | 8.30 s | 12.85 s | 8.75 s | 35.1 dB |

Requests up to 2048² share one canvas, so their latency is nearly constant.
Offload profiles are bounded by host-to-device copies, so their latency
barely grows with image size.

Correctness guard:

- Every layout's 2048² edit was compared with the SP 4 output at the same
  seed. The t2i outputs of all layouts are within PSNR 21.8–23.9 dB of each
  other; t2i is sensitive to small embedding differences, and the
  bring-up showed the same spread inside the SGLang reference.
- DiT offload is exact: the DiT-resident and DiT-offloaded single-GPU runs
  produce identical t2i images. Rank-local and AllGather DiT transfer
  produce byte-identical edits.
- \* The two-GPU text-encoder offload changes the edit output relative to the
  resident layouts. The difference is confined to the edited region (about
  1% of pixels differ by more than 32/255), and the image is otherwise the
  same.
- During bring-up, the four-GPU SP 4 layout matched the SGLang reference implementation
  within SGLang's own variance between its four- and two-GPU runs.
- Not measured: full single-GPU residency (it needs about 135 GiB, which
  this shared GPU did not have) and `resident_layers` above zero (see Notes).

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
