# MiniMax-H3-Image for text-to-image and image editing — Ascend NPU

> Ascend NPU deployment guide covering **Atlas 800I A2**, **Atlas 800T A3**,
> and **Atlas 350 (A5)**. It covers checkpoint preparation, the official
> image and container, deployment commands, request parameters, supported
> image sizes and supported features, plus the measured Atlas 800I A2,
> Atlas 800T A3 and Atlas 350 (A5) performance.

## Summary

- Vendor: MiniMax
- Model: [`MiniMaxAI/MiniMax-H3-Image`](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)
  (ModelScope mirror: [`MiniMax/MiniMax-H3-Image`](https://modelscope.cn/models/MiniMax/MiniMax-H3-Image))
- Task: text-to-image (`t2i`) and reference-image editing (`edit`, 1–9 references)
- Mode: online serving through the OpenAI-compatible Images API
- Hardware:
    - Atlas 800T A3 — 128 GB HBM per card (2 × 64 GB dies), 752 TFLOPS FP16 per card (2 × 376)
    - Atlas 800I A2 — 64 GB HBM per card, 280 TFLOPS FP16
    - Atlas 350 (A5) — 112 GB HBM per card, 425 TFLOPS FP16
- Maintainer: Community

## When to use this recipe

Use this recipe when MiniMax-H3-Image must be served on Ascend NPU.
The provided commands cover one NPU with the DiT offloaded (plus the text
encoder on Atlas 800T A3 / Atlas 800I A2), and eight NPUs with all weights
resident: DiT TP8 on Atlas 800T A3 / Atlas 800I A2, and Ulysses SP8 on
Atlas 350. Atlas 800T A3 and Atlas 800I A2 share one command
set; Atlas 350 has its own. Atlas 800T A3 counts dies: 1, 2, 4 or 8 dies, so
its 8-NPU command is 8 dies (4 cards). Atlas 800I A2 and Atlas 350 provide
1, 2, 4 or 8 cards.

## Supported model contract

| Task | Endpoint | Inputs | Output |
| --- | --- | --- | --- |
| `t2i` | `POST /v1/images/generations` (JSON) | prompt | one image |
| `edit` | `POST /v1/images/edits` (multipart) | prompt and 1–9 reference images, one `image` part each | one image |

A request with reference images is an edit; there is no task flag.

### Request parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `prompt` | string | Yes | Detailed English description of the whole image; truncated to 4096 tokens |
| `size` | string | Yes | `WIDTHxHEIGHT`; both sides must be multiples of 32. The returned image has exactly this size |
| `image` | file | Edit only | Reference image, one `image` part per file, 1–9 files. Upload order defines `Picture 1`, `Picture 2`, ... |
| `seed` | integer | Recommended | Sampling seed |
| `generator_device` | string | Recommended | `cpu` makes the initial noise the same on every machine |
| `response_format` | string | Recommended | `b64_json`; the image is returned in `data[0].b64_json` |
| `output_format` | string | Optional | `png` (default) or `jpeg` |
| `extra_args.min_generation_area` | integer | Optional | Minimum generated canvas area in pixels, 32² to 4096². Default 4194304 (2048×2048). A smaller `size` is generated at the same aspect ratio on a larger canvas and resized back |
| `num_inference_steps` | integer | Optional | Omit or `6`; other values are rejected |
| `guidance_scale` | float | Optional | Omit or `1`; other values are rejected (the model is CFG-distilled) |
| `negative_prompt` | string | Not supported | Rejected |

References are presented to the text encoder as `Picture 1`, `Picture 2`, …
in upload order, so prompts can name them.

### Input and output sizes

The model's effective working range is about 4 MP to 20 MP, with the best
results around 16 MP. The official 16 MP sizes are:

| Aspect ratio | Size (width x height) |
| --- | --- |
| 1:1 | 4096x4096 |
| 4:3 / 3:4 | 4736x3552 / 3552x4736 |
| 3:2 / 2:3 | 5024x3360 / 3360x5024 |
| 16:9 / 9:16 | 5472x3072 / 3072x5472 |
| 21:9 | 6272x2688 |

vLLM-Omni accepts any `WIDTHxHEIGHT` whose sides are multiples of 32 and
returns exactly that size, so every official size above is accepted — each
side is a multiple of 32:

| Aspect ratio | Size | 32-pixel aligned | vLLM-Omni |
| --- | --- | --- | --- |
| 1:1 | 4096x4096 | Yes | Supported |
| 4:3 | 4736x3552 | Yes | Supported |
| 3:4 | 3552x4736 | Yes | Supported |
| 3:2 | 5024x3360 | Yes | Supported |
| 2:3 | 3360x5024 | Yes | Supported |
| 16:9 | 5472x3072 | Yes | Supported |
| 9:16 | 3072x5472 | Yes | Supported |
| 21:9 | 6272x2688 | Yes | Supported |

The same sizes apply to `t2i` and `edit`. Requests smaller than 2048×2048
are generated on a canvas of at least 2048×2048 pixels (same aspect ratio,
sides on a 32-pixel grid) and resized back on device with LANCZOS, so a
`1024x1024` request returns exactly 1024×1024. Set
`extra_args.min_generation_area` to change the canvas floor. For edits,
request the canvas size so the final resize is skipped and the output stays
best aligned with the reference.

## References

- Model card: <https://huggingface.co/MiniMaxAI/MiniMax-H3-Image>
- ModelScope: <https://modelscope.cn/models/MiniMax/MiniMax-H3-Image>
- Model support: <https://github.com/vllm-project/vllm-omni/pull/10120>
- NPU installation: [npu.md](../../docs/getting_started/installation/npu.md)

## Checkpoint

The repository does **not** contain the text encoder. `model_index.json`
names `MiniMaxAI/MiniMax-H3`, subfolder `text_encoder` (about 62 GiB). Fetch
the image release and either let the loader download the encoder, or reuse a
MiniMax-H3 checkpoint that is already on the host:

```bash
export MODEL=/path/to/MiniMax-H3-Image

# ModelScope (CN networks). `--max-workers` uses a hyphen.
modelscope download --model MiniMax/MiniMax-H3-Image \
  --local_dir "${MODEL}" --max-workers 8

# Reuse the H3 video checkpoint's text encoder instead of downloading 62 GiB.
# The pipeline prefers <MODEL>/text_encoder/ when it exists.
ln -sfn /path/to/MiniMax-H3/FL2VA/text_encoder "${MODEL}/text_encoder"
```

Layout:

```text
MiniMax-H3-Image/
├── model_index.json
├── pdd_config.json                 # 6-step distilled sigma schedule
├── pdd_fused_heads.safetensors     # per-step output heads
├── processor/  tokenizer/          # Qwen3-VL tokenizer and processor
├── transformer/                    # DiT, 12 shards, BF16, about 61.7 GiB
├── vae/                            # image VAE, FP32, about 2.2 GiB
└── text_encoder -> ...             # reused Qwen3-VL (optional symlink)
```

## Environment

MiniMax-H3-Image is supported from vLLM-Omni 0.33 and later. Pick the image
matching the platform from the
[vLLM-Omni image repository](https://quay.io/repository/ascend/vllm-omni?tab=tags);
0.33 or newer images contain the model support.

| Platform | Image |
| --- | --- |
| Atlas 800I A2 | `quay.io/ascend/vllm-omni:minimax-h3-image` |
| Atlas 800T A3 | `quay.io/ascend/vllm-omni:minimax-h3-image-a3` |
| Atlas 350 (A5) | `quay.io/ascend/vllm-omni:minimax-h3-image-a5` |

```bash
export IMAGE=quay.io/ascend/vllm-omni:minimax-h3-image-a3
export CONTAINER_NAME=h3-image

docker run -it -u root --name ${CONTAINER_NAME} \
  --privileged=true \
  --shm-size=2000g \
  --net=host \
  --device /dev/davinci0 \
  --device /dev/davinci1 \
  --device /dev/davinci2 \
  --device /dev/davinci3 \
  --device /dev/davinci4 \
  --device /dev/davinci5 \
  --device /dev/davinci6 \
  --device /dev/davinci7 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /home:/home \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /root/.cache:/root/.cache \
  ${IMAGE} \
  /bin/bash
```

> [!NOTE]
> Atlas 350 has no `/dev/devmm_svm` device; drop that mapping when creating
> the container, and keep only the `/dev/davinciN` entries that exist. Atlas
> 350PR also needs `-v /usr/local/Ascend/driver/topo:/usr/local/Ascend/driver/topo`.

### Install the model support (optional)

If the image predates the model support
([#10120](https://github.com/vllm-project/vllm-omni/pull/10120)), install a
newer source checkout inside the container:

```bash
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
VLLM_OMNI_TARGET_DEVICE=npu pip install -e . --no-build-isolation \
  -i https://mirrors.aliyun.com/pypi/simple
```

## Hardware

### Atlas 800T A3

- Accelerator: Atlas 800T A3 (dual-die card)
- HBM: 128 GB per card (64 GB per die)
- FP16 compute: 752 TFLOPS per card (376 TFLOPS per die)
- Number of dies: 1, 2, 4 or 8 (a card has 2 dies; 8 dies = 4 cards)
- Interconnect: HCCL over the node fabric

### Atlas 800I A2

- Accelerator: Atlas 800I A2
- HBM: 64 GB per card
- FP16 compute: 280 TFLOPS per card
- Number of cards: 1, 2, 4 or 8
- Interconnect: HCCL over the node fabric

### Atlas 350 (A5)

- Accelerator: Atlas 350
- HBM: 112 GB per card
- FP16 compute: 425 TFLOPS per card
- Number of cards: 1, 2, 4 or 8
- Interconnect: HCCL over the node fabric
- Host memory: the DLO profiles keep pinned host copies of the offloaded DiT
  blocks; plan host memory for the DiT shards outside the NPUs

## Command

Set `MODEL` to the checkpoint directory, then run the block for the NPU count
you have. Every block sets the required environment variables first. The
blocks below are the 1- and 8-NPU reference configurations; for 2 or 4 NPUs
use the same shape with the actual counts. Atlas 800T A3 and Atlas 800I A2
share one command set; Atlas 350 has its own section. Atlas 800T A3 counts
dies: its 8-NPU command is 8 dies (4 cards).

```bash
export MODEL=/path/to/MiniMax-H3-Image
```

### Atlas 800T A3 / Atlas 800I A2

#### 1 npu

```bash
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_NPU_SOCKET_PORT_RANGE="auto"

vllm serve "${MODEL}" --omni \
  --host 127.0.0.1 --port 18089 \
  --served-model-name MiniMaxAI/MiniMax-H3-Image \
  --enforce-eager --num-gpus 1 \
  --diffusion-offload-config '{"mode":"layer","components":["dit","text_encoder"]}'
```

#### 8 npus

```bash
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_NPU_SOCKET_PORT_RANGE="auto"

vllm serve "${MODEL}" --omni \
  --host 127.0.0.1 --port 18089 \
  --served-model-name MiniMaxAI/MiniMax-H3-Image \
  --enforce-eager --num-gpus 8 --tensor-parallel-size 8 --usp 1 --text-encoder-tp-size 8
```

### Atlas 350 (A5)

#### 1 npu

```bash
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_NPU_SOCKET_PORT_RANGE="auto"

vllm serve "${MODEL}" --omni \
  --host 127.0.0.1 --port 18089 \
  --served-model-name MiniMaxAI/MiniMax-H3-Image \
  --enforce-eager --num-gpus 1 \
  --diffusion-offload-config '{"mode":"layer","components":["dit"]}'
```

#### 8 npus

```bash
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_NPU_SOCKET_PORT_RANGE="auto"

vllm serve "${MODEL}" --omni \
  --host 127.0.0.1 --port 18089 \
  --served-model-name MiniMaxAI/MiniMax-H3-Image \
  --enforce-eager --num-gpus 8 --usp 8 --text-encoder-tp-size 8
```

## Request scripts (all platforms)

Text-to-image:

```bash
curl -fsS http://127.0.0.1:18089/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"MiniMaxAI/MiniMax-H3-Image",
    "prompt":"Documentary photograph of a lighthouse keeper on a cliff at dawn, Kodak Portra 400",
    "size":"2048x2048",
    "seed":42,
    "response_format":"b64_json"
  }' | jq -er '.data[0].b64_json' | base64 -d > t2i-2048.png
```

Image editing with one reference image:

```bash
ASSETS=/path/to/MiniMax-H3-Image/examples

curl -fsS http://127.0.0.1:18089/v1/images/edits \
  -F 'model=MiniMaxAI/MiniMax-H3-Image' \
  -F "image=@$ASSETS/reference1.png;type=image/png" \
  -F "prompt=<$ASSETS/prompt.edit.txt" \
  -F 'size=4096x4096' \
  -F 'seed=1234' -F 'generator_device=cpu' \
  -F 'response_format=b64_json' -F 'output_format=png' \
  | jq -er '.data[0].b64_json' | base64 -d > edit-4096.png
```

`prompt=<file` reads the prompt from a file with the shell's redirect syntax.
Repeat `-F image=@...` for up to nine references, in the order the prompt
names them.

## Verification

```bash
curl -s http://127.0.0.1:18089/v1/models | jq -r '.data[].id'
# MiniMaxAI/MiniMax-H3-Image
```

A generation response carries the image in `data[0].b64_json`; after base64
decoding, the PNG must have exactly the requested `WIDTHxHEIGHT`.

## Performance

### Atlas 350 (A5) performance

Verified on Atlas 350 (Ascend 950PR, 112 GiB per NPU).

Setup: one Atlas 350 node with an AMD EPYC 9575F host; all profiles used the
same UB group and shared a common NPU. Image:
`quay.io/ascend/vllm-omni:v0.33.0rc1-a5`; vLLM 0.33.0, PyTorch 2.10.0,
torch_npu 2.10.0.post7, Transformers 5.19.0, Diffusers 0.40.0; BF16, eager,
six denoising steps, concurrency 1. All profiles keep the text encoder and
VAE on device.

Client end-to-end time is measured through receipt of the full HTTP response.
The first three columns show the median of three requests after one warm-up
per case; the 4096² edit is one request. T2I uses the lighthouse prompt from
the XPU recipe, seed 42 and the default minimum generation area; the 1024²
output is generated at 2048² and resized. Edits use the released 1024²
reference, structured prompt and CPU generator with seed 1234.

| Profile | T2I 1024² | T2I 2048² | Edit 2048² | Edit 4096² | Peak HBM / NPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 NPUs, Ulysses SP4, text encoder TP4, fully resident | 1.52 s | 1.49 s | 2.47 s | 8.56 s | 85.61 GiB |
| 2 NPUs, Ulysses SP2, text encoder TP2, DiT DLO with AllGather | 6.45 s | 6.45 s | 7.27 s | 18.89 s | 39.65 GiB |
| 1 NPU, DiT layer offload | 7.79 s | 7.76 s | 8.36 s | 31.10 s | 64.20 GiB |

### Atlas 800T A3 performance

Verified on Atlas 800T A3 (128 GiB per card as 2 × 64 GiB dies). The 1- and
2-die profiles ran on one card, 4 dies on two cards, and 8 dies on four
cards. The test case is the same as the Atlas 350 table: BF16, eager, six
denoising steps, concurrency 1. Image:
`quay.io/ascend/vllm-omni:minimax-h3-image-a3` (vLLM-Omni `eb8b991`).

The 8-die profile uses DiT TP8 and text-encoder TP8 with all weights
resident. The 4-die profile uses Ulysses SP4, text-encoder TP4 and DiT
distributed layerwise offload with AllGather: keeping every weight resident
does not fit in 64 GiB per die. The 2-die profile uses Ulysses SP2,
text-encoder TP2 (the encoder stays resident on the dies) and the same DiT
offload. The 1-die profile offloads both DiT and text encoder to host
memory, so it holds much less HBM per die.

The 4- and 8-die runs shared their cards with idle neighbour processes that
held up to 10 GiB per die at 0% AICore. The reported peak excludes that
baseline, and the timings were taken while those processes stayed idle.

| Profile | T2I 1024² | T2I 2048² | Edit 2048² | Edit 4096² | Peak HBM / die |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8 dies (4 cards), DiT TP8, text encoder TP8, fully resident | 2.12 s | 2.10 s | 3.36 s | 9.65 s | 19.45 GiB |
| 4 dies (2 cards), Ulysses SP4, text encoder TP4, DiT DLO with AllGather | 4.48 s | 4.47 s | 5.31 s | 14.87 s | 20.32 GiB |
| 2 dies, Ulysses SP2, text encoder TP2, DiT DLO with AllGather | 6.30 s | 6.10 s | 8.71 s | 28.09 s | 36.20 GiB |
| 1 die, DiT + text encoder layer offload | 10.86 s | 10.79 s | 14.87 s | 54.59 s | 12.26 GiB |

### Atlas 800I A2 performance

The test case is the same as the Atlas 350 table.

| Profile | T2I 1024² | T2I 2048² | Edit 2048² | Edit 4096² | Peak HBM / NPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 NPUs, DiT TP4, fully resident | 5.28 s | 4.98 s | 7.75 s | 24.66 s | 39.10 GiB |
| 2 NPUs, Ulysses SP2, DiT AllGather offload | 16.14 s | 16.09 s | 18.72 s | 55.83 s | 38.44 GiB |
| 1 NPU, DiT + text-encoder layer offload | 24.54 s | 24.86 s | 26.04 s | 98.19 s | 12.69 GiB |

## Supported optimizations

| Optimization | Enabling configuration |
| --- | --- |
| DiT USP parallel | `--num-gpus N --usp N` |
| DiT TP parallel | `--num-gpus N --tensor-parallel-size N --usp 1` |
| Text-encoder TP | `--text-encoder-tp-size N` |
| DiT DLO | `--diffusion-offload-config '{"mode":"layer","components":["dit"]}'` (add `"layer_options":{"dit":{"weight_transfer":"allgather"}}` for multiple NPUs) |
| Text-encoder DLO | `--diffusion-offload-config '{"mode":"layer","components":["text_encoder"]}'` (add `"layer_options":{"text_encoder":{"weight_transfer":"rank-local"}}` for multiple NPUs) |
| VAE tile parallel | On by default with multiple NPUs (`additional_config.vae_tile_parallel`) |
