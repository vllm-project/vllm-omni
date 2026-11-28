# MOSS-Image-Base on Ascend A3

## Summary

`MOSSImagePipeline` serves text-to-image requests through the
OpenAI-compatible Images API. Each A3 card has two NPU dies with 64 GiB
of HBM per die.
[OpenMOSS-Team/MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base)

Tensor parallelism shards the DiT. Each die keeps a copy of the text encoder
and VAE. See the [MOSS-Image-Base recipe](MOSS-Image-Base.md) for offline
generation and sampling options.

## Ascend A3 performance

| NPU dies | Physical cards | DiT TP degree | P50 (s) | Peak extra HBM (GiB/die) |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 4 | 8 | 13.28 | 23.62 |
| 4 | 2 | 4 | 15.58 | 24.77 |
| 2 | 1 | 2 | 21.75 | 27.09 |
| 1 | 1 (one die) | 1 | 33.90 | 31.24 |

The DiT and text encoder used BF16 in eager mode, and the VAE used
FP32. Requests used 50 denoising steps, CFG 3, flow shift 5, seed 0,
and a CPU random generator.

Tests ran on one Atlas 900 A3 SuperPoD node at 1024×1024 and concurrency 1.
The runs followed the same prompt order, warm-ups, and measurement method as the
[Qwen A3 baseline](../Qwen/Qwen-Image-2.1-NPU.md#ascend-a3-performance).

## Checkpoint

```bash
modelscope download --model openmoss/MOSS-Image-Base --local_dir ./models/MOSS-Image-Base
```

## Start the server

Mount the model directory and select the dies as described in the
[A3 container setup](../Qwen/Qwen-Image-2.1-NPU.md#a3-environment).
For 1, 2, or 4 dies, set both `--num-gpus` and `--tensor-parallel-size`
to the selected die count.

```bash
vllm serve /models/MOSS-Image-Base --omni \
  --host 127.0.0.1 --port 18089 --served-model-name OpenMOSS-Team/MOSS-Image-Base \
  --dtype bfloat16 --enforce-eager --trust-remote-code --max-num-seqs 1 \
  --num-gpus 8 --tensor-parallel-size 8
```

## Text-to-image request

```bash
set -o pipefail
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
```
