# MiniMax-H3-Image on Moore Threads MUSA

> Text-to-image and reference-image editing on MTT S5000

## Summary

- Vendor: MiniMaxAI
- Model: [`MiniMaxAI/MiniMax-H3-Image`](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)
- Tasks: text-to-image (`t2i`) and reference-image editing (`edit`)
- Mode: OpenAI-compatible Images API serving (`/v1/images/generations`, `/v1/images/edits`)
- Hardware: Moore Threads MTT S5000, 80 GB per card, two or four GPUs
- Recommended deployment: 4 GPU, DiT TP 4
- Qualification scope: measured on 4x S5000 (one node)
- Maintainer: Community

## Choose a profile

The checkpoint has 126 GiB of weights: a 61.7 GiB DiT, a 62.1 GiB Qwen3-VL
text encoder and a 2.1 GiB FP32 VAE. An S5000 has 79.9 GiB usable per card, so
the DiT has to be sharded rather than replicated.

| GPUs | Profile | Extra flags | Peak / GPU | 2048² t2i | 2048² edit |
| --- | --- | --- | ---: | ---: | ---: |
| 4 | DiT TP 4, resident (recommended) | `--num-gpus 4 --tensor-parallel-size 4 --usp 1 --text-encoder-tp-size 4` | 34.8 GiB | 2.3 s | 3.3 s |
| 2 | DiT TP 2, resident | `--num-gpus 2 --tensor-parallel-size 2 --usp 1 --text-encoder-tp-size 2` | 62.0 GiB | 3.8 s | 5.9 s |

Ulysses sequence parallelism shards the sequence and keeps a full DiT on every
card, which does not fit in 79.9 GiB, so it is not an option here; DiT tensor
parallelism is. Going from two cards to four also halves the per-card footprint.

## Checkpoint

The image release ships the DiT, the VAE, the tokenizer/processor and a
`text_encoder/` copy, while its `model_index.json` names
`MiniMaxAI/MiniMax-H3` as the text-encoder source. The pipeline prefers the
bundled copy and logs which one it used:

```text
MiniMax-H3-Image: using bundled text_encoder/ instead of MiniMaxAI/MiniMax-H3
```

So a full local snapshot works as released, offline, with no metadata edit. If
you would rather not keep the duplicate, download the release without
`text_encoder/` and let the pipeline fetch the encoder from the base repository;
populating the hub cache with it keeps that path offline too:

```bash
export MODEL_ROOT=/path/to/MiniMax-H3-Image

# DiT, VAE, tokenizer, processor, and the fused-head checkpoint
hf download MiniMaxAI/MiniMax-H3-Image \
  --local-dir "${MODEL_ROOT}" \
  --include 'transformer/*' 'vae/*' 'tokenizer/*' 'processor/*' \
            'scheduler/*' 'model_index.json' 'pdd_config.json' \
            'pdd_fused_heads.safetensors'

# optional: text encoder into the hub cache instead (no --local-dir)
hf download MiniMaxAI/MiniMax-H3 --include 'text_encoder/*'
```

Skip `transformer_diffusers/` and `vae_diffusers/`: they duplicate the released
weights byte for byte and cost another 63.9 GiB. What is left is 126.1 GiB.

## Software environment

| Component | Value |
| --- | --- |
| Driver | `5.2.0-server` |
| Image | `sh-harbor.mthreads.com/mcctest/vllm:v0.32.0-ph1-5.2.0-torch2.11.0.post2-20261110` |
| PyTorch / torch-musa | `2.11.0.post2+musa5.2.0` |
| torchada | `0.1.105` |
| MATE / flash_attn_3 | `0.2.8.post1` / `0.2.8.post1+musa` |
| vLLM / vLLM-MUSA | `0.32.0` |
| Transformers | `5.19.0` |
| vLLM-Omni | `0.32.0.dev0+musa` |

Install a compatible PyTorch/torch-musa, torchada, MATE, and vLLM-MUSA stack
first, then vLLM-Omni:

```bash
VLLM_OMNI_TARGET_DEVICE=musa pip install -e . --no-build-isolation
```

`VLLM_OMNI_TARGET_DEVICE=musa` selects `requirements/musa.txt` instead of
letting `setup.py` auto-detect the backend, and `--no-build-isolation` builds
against the preinstalled MUSA wheels. The image above ships Python 3.10 while
`pyproject.toml` requires 3.11 or newer, so add `--ignore-requires-python` when
installing into it.

Do not install the CUDA-only `fa4` optional dependency. On MUSA,
`--diffusion-attention-backend FLASH_ATTN` selects the MATE/FlashAttention-3
implementation.

## Deployment

```bash
export MODEL="${MODEL_ROOT}"
export PORT=8091
```

### Four GPUs

```bash
MUSA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --served-model-name MiniMaxAI/MiniMax-H3-Image \
  --num-gpus 4 \
  --tensor-parallel-size 4 \
  --usp 1 \
  --text-encoder-tp-size 4 \
  --diffusion-attention-backend FLASH_ATTN
```

Each rank loads 30.6 GiB and reports `VAE tile decode uses CUDA graphs: True`.

### Two GPUs

```bash
MUSA_VISIBLE_DEVICES=0,1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --served-model-name MiniMaxAI/MiniMax-H3-Image \
  --num-gpus 2 \
  --tensor-parallel-size 2 \
  --usp 1 \
  --text-encoder-tp-size 2 \
  --diffusion-attention-backend FLASH_ATTN
```

Each rank loads 57.7 GiB; the server answers `/health` roughly 100 s after
launch. `--text-encoder-tp-size` must divide Qwen3-VL's 64 query and 8 KV heads
and must not exceed the number of GPUs.

## Request examples

Text to image:

```bash
curl -s "http://localhost:${PORT}/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{"model": "MiniMaxAI/MiniMax-H3-Image",
       "prompt": "Documentary photograph of a lighthouse keeper on a cliff at dawn, Kodak Portra 400",
       "size": "2720x1536", "seed": 42, "response_format": "b64_json"}' \
  | python3 -c 'import base64,json,sys; sys.stdout.buffer.write(base64.b64decode(json.load(sys.stdin)["data"][0]["b64_json"]))' \
  > t2i.png
```

An edit is the same endpoint with one multipart `image` part per reference
(1-9 of them); there is no task flag. Use `--form-string` for the prompt: in
curl's `-F` syntax a value that begins with `<` means "read this value from a
file", so the model's `<Picture 1>` reference syntax silently fails before the
request is even sent.

```bash
curl -s "http://localhost:${PORT}/v1/images/edits" \
  -F model=MiniMaxAI/MiniMax-H3-Image \
  --form-string 'prompt=<Picture 1> is the cabin. Only the season changes to winter.' \
  -F size=2048x2048 -F seed=7 -F response_format=b64_json \
  -F image=@cabin.png \
  | python3 -c 'import base64,json,sys; sys.stdout.buffer.write(base64.b64decode(json.load(sys.stdin)["data"][0]["b64_json"]))' \
  > edit.png
```

Requests smaller than 2048 pixels are generated on a 2048-class canvas and
resized back with LANCZOS, so latency is nearly flat up to 2048².

## Measured latency

Workload: the release's example t2i prompt and one-reference edit,
`generator_device=cpu`, six steps, concurrency 1, median client end-to-end time
of three requests after one warm-up, including PNG encoding. Peak memory is the
per-card `mthreads-gmi -q -d MEMORY` maximum sampled every 0.5 s.

| Profile | t2i 1024² | t2i 2048² | t2i 4736×3552 | edit 2048² | Peak / GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 GPU, DiT TP 4 | 2.11 s | 2.28 s | 8.96 s | 3.26 s | 34.8 GiB |
| 2 GPU, DiT TP 2 | 3.63 s | 3.81 s | 16.55 s | 5.89 s | 62.0 GiB |

Both layouts log `VAE tile decode uses CUDA graphs: True`, so neither row is an
eager fallback.

The outputs are image content, not blank frames, but this recipe does not claim
prompt adherence, reference-image fidelity, or agreement with a CUDA reference;
none of those were compared.

## Notes

- The VAE tile decode replays captured graphs. `torchada`/`torch.musa` provide
  the graph API and vLLM-Omni enables the path on any cuda-like platform, so
  S5000 replays graphs instead of decoding eagerly. Pass `--enforce-eager`, or
  `--no-enable-cuda-graph-decode`, only to diagnose a capture problem.
- Ulysses sequence parallelism is not validated on S5000: a resident SP 4 run
  loads a full DiT per card and failed to start. Use DiT TP 2 or TP 4.
- H3-Image is CFG-distilled: `num_inference_steps` must be omitted or `6`, and
  `guidance_scale` omitted or `1`. `negative_prompt` is not supported.
- A generation serves one request (`n=1`), and the diffusion batch runs one
  request at a time.
- Keep Ring Attention at degree 1, as in the other H3 recipes.

## References

- [MiniMax-H3-Image-H200.md](MiniMax-H3-Image-H200.md) — full model and API contract, all NVIDIA profiles
- [MiniMax-H3-MUSA.md](MiniMax-H3-MUSA.md) — the video/audio H3 recipe for S5000
- [Supported models](../../docs/models/supported_models.md)
- [Image generation API](../../docs/serving/image_generation_api.md)
- [Image editing API](../../docs/serving/image_edit_api.md)
- [Tensor parallel](../../docs/user_guide/diffusion/parallelism/tensor_parallel.md)
- [VAE parallelism](../../docs/user_guide/diffusion/parallelism/vae_parallelism.md)
