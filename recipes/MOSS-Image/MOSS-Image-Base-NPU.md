# MOSS-Image-Base on Ascend A3

## Summary

The `MOSSImagePipeline` serves text-to-image requests through the
OpenAI-compatible Images API on vLLM-Omni 0.34.0. Each physical A3 card
has two NPU dies with 64 GiB HBM per die.
[OpenMOSS-Team/MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base)

The profiles below use DiT tensor parallelism. The text encoder and VAE
remain replicated. The general recipe covers offline generation and sampling
controls. [MOSS-Image-Base recipe](MOSS-Image-Base.md)

## Ascend A3 performance

| NPU dies | Physical card capacity used | DiT TP degree | P50 (s) | Peak extra HBM (GiB/die) |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 4 | 8 | 13.28 | 23.62 |
| 4 | 2 | 4 | 15.58 | 24.77 |
| 2 | 1 | 2 | 21.75 | 27.09 |
| 1 | ½ | 1 | 33.90 | 31.24 |

The DiT and text encoder used BF16 in eager mode, and the VAE used
FP32. Requests used 50 denoising steps, CFG 3, flow shift 5, seed 0,
and a CPU random generator.

Measured on an Atlas 900 A3 SuperPoD compute node with an arm64 host,
CANN 9.3.0, Ascend driver 26.1.1, PyTorch 2.10.0, and torch-npu
2.10.0.post7. The vLLM-Omni image was `v0.34.0-a3` with OCI index
`sha256:584e105786b352505427ef81b9080b90267e12114462221f3db41b7988eb044b`.

Each profile served one 1024×1024 PNG per request at concurrency 1.
Two rounds each had one warm-up and five measured requests. The prompts
were `p20` then `p01`–`p05` in round one, and `p19` then `p06`–`p10`
in round two. See the [prompt corpus](../../benchmarks/diffusion/h3-image-prompts.jsonl).

P50 is the median of the ten measured HTTP request-to-full-response
latencies. Every PNG decoded at the requested size. Peak extra HBM is the
largest sampled increase over the pre-launch HBM reading on any selected
die. The sampler paused for three seconds between `npu-smi info` calls.

## Checkpoint

```bash
export MODEL_ROOT=/path/to/models
modelscope download --model openmoss/MOSS-Image-Base \
  --local_dir "${MODEL_ROOT}/MOSS-Image-Base" --max-workers 8
```

The checkpoint includes the Diffusers `model_index.json`, the entry point
for the text encoder, DiT, and VAE components.

## Environment

Use `quay.io/ascend/vllm-omni:v0.34.0-a3` and follow the
[A3 container setup](../Qwen/Qwen-Image-2.1-A3.md#start-the-container).
The command mounts `MODEL_ROOT` at `/models` and sets `NPU_DIES` from
the selected logical IDs. One, two, four, and eight dies use half a
physical card, one card, two cards, and four cards, respectively.

## Command

```bash
export MODEL_ID=OpenMOSS-Team/MOSS-Image-Base
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_TARGET_DEVICE=npu
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_NPU_SOCKET_PORT_RANGE=auto

vllm serve "/models/${MODEL_ID##*/}" --omni \
  --host 127.0.0.1 --port 18089 --served-model-name "${MODEL_ID}" \
  --dtype bfloat16 --enforce-eager --trust-remote-code --max-num-seqs 1 \
  --num-gpus "${NPU_DIES}" --tensor-parallel-size "${NPU_DIES}"
```

## Request and verification

```bash
set -o pipefail
curl -fsS http://127.0.0.1:18089/v1/models | jq -r '.data[].id'

curl -fsS http://127.0.0.1:18089/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"OpenMOSS-Team/MOSS-Image-Base",
    "prompt":"A ceramic teapot on a wooden table",
    "size":"1024x1024",
    "n":1,
    "num_inference_steps":50,
    "guidance_scale":3.0,
    "flow_shift":5.0,
    "seed":0,
    "generator_device":"cpu",
    "response_format":"b64_json",
    "output_format":"png"
  }' | jq -er '.data[0].b64_json' | base64 -d > moss-image.png

python - <<'PYTHON'
from PIL import Image

with Image.open("moss-image.png") as image:
    image.load()
    assert image.format == "PNG" and image.size == (1024, 1024)
PYTHON
```

The response should decode to exactly one 1024×1024 image.
