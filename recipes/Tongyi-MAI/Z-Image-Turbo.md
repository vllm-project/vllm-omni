# Z-Image-Turbo

> Text-to-image generation with vLLM-Omni.

## Summary

* Vendor: Tongyi-MAI
* Model: `Tongyi-MAI/Z-Image-Turbo`
* Task: Text-to-image generation
* Mode: Offline generation and online OpenAI-compatible serving
* Maintainer: Community

## When to use this recipe

Use this recipe as a practical baseline for running `Tongyi-MAI/Z-Image-Turbo` with vLLM-Omni for text-to-image generation. It covers both offline Python inference and online serving through the OpenAI-compatible `/v1/chat/completions` API.

## Validated setup

* OS: Linux
* Python: 3.11
* vLLM version: 0.21.0
* vLLM-Omni version: `v0.21.0rc1`
* GPU: NVIDIA RTX PRO 6000 Blackwell, 95GB
* Modes validated: offline inference and online serving

Set Hugging Face and temporary caches to a disk with enough free space before downloading model weights:

```bash
export HF_HOME=/path/to/large/disk/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TMPDIR=/path/to/large/disk/tmp
```

Optional allocator setting:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Installation

From a local vLLM-Omni checkout:

```bash
uv venv --python 3.11 --seed
source .venv/bin/activate

uv pip install "vllm==0.21.0" --torch-backend=auto
uv pip install -e .
```

Check the installed versions:

```bash
python -c "import vllm; print(vllm.__version__)"
python -c "import vllm_omni; print('vllm_omni ok')"
```

## Offline inference

Use a Python script with a multiprocessing main guard:

```python
from vllm_omni.entrypoints.omni import Omni


def main():
    omni = Omni(model="Tongyi-MAI/Z-Image-Turbo")
    outputs = omni.generate("a cup of coffee on a wooden table")
    image = outputs[0].request_output.images[0]
    image.save("coffee.png")
    print("saved coffee.png")


if __name__ == "__main__":
    main()
```

Expected result:

```text
saved coffee.png
```

## Online serving

Start the server:

```bash
vllm serve Tongyi-MAI/Z-Image-Turbo --omni --port 8000
```

Verify the model endpoint:

```bash
curl http://localhost:8000/v1/models
```

Send a generation request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Tongyi-MAI/Z-Image-Turbo",
    "messages": [
      {
        "role": "user",
        "content": "Generate an image of a cup of coffee on a wooden table."
      }
    ]
  }' > zimage_response.json
```

## Decode the image response

The online response contains a base64 PNG image. One simple way to save it is:

```bash
python - <<'PY'
import base64
import json
import re

with open("zimage_response.json", "r") as f:
    data = json.load(f)


def find_png_base64(obj):
    if isinstance(obj, str):
        match = re.search(r"data:image/png;base64,([A-Za-z0-9+/=]+)", obj)
        if match:
            return match.group(1)
        match = re.search(r"(iVBORw0KGgo[A-Za-z0-9+/=]+)", obj)
        if match:
            return match.group(1)

    if isinstance(obj, dict):
        for key in ("data", "b64_json", "base64"):
            val = obj.get(key)
            if isinstance(val, str) and val.startswith("iVBORw0KGgo"):
                return val
        for val in obj.values():
            found = find_png_base64(val)
            if found:
                return found

    if isinstance(obj, list):
        for val in obj:
            found = find_png_base64(val)
            if found:
                return found

    return None


b64 = find_png_base64(data)
if b64 is None:
    raise RuntimeError("No PNG image data found in response")

with open("online_coffee.png", "wb") as f:
    f.write(base64.b64decode(b64))

print("saved online_coffee.png")
PY
```

Expected result:

```text
saved online_coffee.png
```

## Observed validation result

One online request on the validated setup returned approximately:

```text
queue_wait_ms: 0.36 ms
stage_0_gen_ms: 16.25 s
peak_memory_mb: 24228 MB
```

These numbers are one observed reference point, not a guaranteed benchmark.

## Graceful shutdown check

```bash
ps -ef | grep "vllm serve" | grep -v grep
kill -TERM <SERVER_PID>
sleep 3

ps -ef | grep -E "vllm|omni|python" | grep -v grep
lsof -i :8000
nvidia-smi
```

Expected result:

* no `vllm serve Tongyi-MAI/Z-Image-Turbo` process remains
* port 8000 is released
* GPU memory is released

## Troubleshooting

### vLLM and vLLM-Omni version alignment

Use matching vLLM and vLLM-Omni major/minor versions. A mismatch can lead to compatibility warnings or runtime failures.

## Test plan

* Offline generation produced `coffee.png`.
* Online `/v1/models` returned `Tongyi-MAI/Z-Image-Turbo`.
* Online `/v1/chat/completions` returned base64 PNG image data.
* Base64 decoding produced `online_coffee.png`.
* SIGTERM released the server process, port 8000, and GPU memory.

