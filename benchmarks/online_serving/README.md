# Online Serving Benchmarks

These scripts benchmark vLLM-Omni **online serving** via OpenAI-style HTTP APIs.

## Prerequisites

- A running vLLM-Omni server (example): `vllm serve <model> --omni --port 8000`
- Python 3.10+
- Optional for GPU peak memory monitoring: `pynvml` (`pip install nvidia-ml-py`)

## Minimal Usage (4 scripts)

### 1) Text-to-Image (T2I)

- Script: `benchmarks/online_serving/text_to_image/benchmark_t2i.py`
- Endpoint: `POST /v1/images/generations`

```bash
python benchmarks/online_serving/text_to_image/benchmark_t2i.py \
  --api-base http://localhost:8000/v1 \
  --prompt "a cat wearing sunglasses"
```

### 2) Text-to-Video (T2V)

- Script: `benchmarks/online_serving/text_to_video/benchmark_t2v.py`
- Default endpoint in script: `POST /v1/images/generations` (vLLM-Omni extension-style payload)

```bash
python benchmarks/online_serving/text_to_video/benchmark_t2v.py \
  --api-base http://localhost:8000/v1 \
  --model Wan-AI/Wan2.2-T2V-A14B-Diffusers
```

### 3) Image-to-Image (I2I)

- Script: `benchmarks/online_serving/image_to_image/benchmark_i2i.py`
- Endpoint: `POST /v1/images/edits` (multipart form-data)

```bash
python benchmarks/online_serving/image_to_image/benchmark_i2i.py \
  --api-base http://localhost:8000/v1 \
  --image path/to/input.png
```

### 4) Image-to-Video (I2V)

- Script: `benchmarks/online_serving/image_to_video/benchmark_i2v.py`
- Default endpoint in script: `POST /v1/videos/generations` (endpoint is configurable)

```bash
python benchmarks/online_serving/image_to_video/benchmark_i2v.py \
  --api-base http://localhost:8000/v1 \
  --image path/to/input.png
```
