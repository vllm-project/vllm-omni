# SenseNova-U1: Online serving

## Launch the Server

```bash
vllm serve SenseNova/SenseNova-U1-8B-MoT --omni --port 8091
```

With cache acceleration (`cache_dit` or `tea_cache`):

```bash
vllm serve SenseNova/SenseNova-U1-8B-MoT --omni --port 8091 \
    --cache-backend tea_cache
```

Or use the convenience script:

```bash
cd examples/online_serving/sensenova_u1
bash run_server.sh

# Cache acceleration (cache_dit or tea_cache)
CACHE_BACKEND=tea_cache bash run_server.sh
```

### Tensor Parallelism (TP)

```bash
vllm serve SenseNova/SenseNova-U1-8B-MoT --omni --port 8091 \
    --tensor-parallel-size 2
```

### Request-mode image batching (B2 draft)

This draft is stacked on B1 (#7730), commit `8c971ea1c6ee`.
Keep `step_execution=False` (the default), use `cache_backend="none"`,
and set `max_num_seqs` above one to admit compatible image requests together.
`request_batch_max_wait_ms` can provide a short admission window for arrivals
to coalesce. These are engine settings, not per-request image counts.

The pipeline prepares each request's prefix/think to completion in sequence,
saves its KV, then fuses the denoise forwards across requests. Variable-length
prefixes are padded and masked, with each request retaining its original RoPE
positions and seeded noise. This uses dense denoise KV; it does not introduce
interleaved AR prepare or Scheduler-owned AR KV rows.

Requests must agree on output dimensions, image count, denoise schedule, CFG
settings, LoRA and the scheduler's other compatibility fields. T2I and IT2I
are separate groups. Prompts, seeds and think lengths may differ. Text output
is scheduled as singleton requests. Multi-request batches with cache
acceleration are rejected by the pipeline.

For multiple images per prompt, use `OmniDiffusionSamplingParams`'s
`num_outputs_per_prompt=M`. The legacy `extra_args["batch_size"]` remains an
alias when the standard count is one; conflicting non-default counts are
rejected. Every request returns all its images in order, along with its own
think metadata. With `N` compatible requests, the denoise batch has `N * M`
rows; account for the corresponding activation and padded-KV memory.

B1's step-mode `max_num_seqs=1` restriction is unchanged. Request batching
does not make the shared paged AR decode allocation safe for interleaved
prepare. A single-rank full-checkpoint smoke test covers fused request batches
and `N=2, M=2` output routing. Production-resolution image-quality,
throughput and multi-rank validation remain draft acceptance work.

## Send Requests

```bash
cd examples/online_serving/sensenova_u1
```

### Text to Image (text2img)

**Python client:**

```bash
python openai_chat_client.py \
    --prompt "A beautiful sunset over mountains" \
    --modality text2img \
    --height 2048 --width 2048 --num-steps 50
```

**curl:**

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": [{"type": "text", "text": "A beautiful sunset over mountains"}]}],
    "modalities": ["image"],
    "height": 2048,
    "width": 2048,
    "num_inference_steps": 50,
    "seed": 42
  }'
```

### Image to Image (img2img)

**Python client:**

```bash
python openai_chat_client.py \
    --prompt "Turn this into an oil painting" \
    --modality img2img \
    --image-url /path/to/input.jpg \
    --height 2048 --width 2048
```

**curl:**

```bash
IMAGE_BASE64=$(base64 -w 0 input.jpg)

cat <<EOF > payload.json
{
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Turn this into an oil painting"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,${IMAGE_BASE64}"}}
      ]
    }],
    "modalities": ["image"],
    "height": 2048,
    "width": 2048,
    "num_inference_steps": 50,
    "seed": 42
}
EOF

curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @payload.json
```

### Image to Text (img2text)

**Python client:**

```bash
python openai_chat_client.py \
    --prompt "Describe this image in detail" \
    --modality img2text \
    --image-url /path/to/image.jpg
```

**curl:**

```bash
IMAGE_BASE64=$(base64 -w 0 image.jpg)

cat <<EOF > payload.json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Describe this image in detail"},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,${IMAGE_BASE64}"}}
    ]
  }],
  "modalities": ["text"]
}
EOF

curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @payload.json
```

### Text to Text (text2text)

**Python client:**

```bash
python openai_chat_client.py \
    --prompt "What is the capital of France?" \
    --modality text2text
```

**curl:**

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]}],
    "modalities": ["text"]
  }'
```

## Python Client Arguments

| Argument | Default | Description |
| :------- | :------ | :---------- |
| `--prompt` / `-p` | `A cute cat` | Text prompt |
| `--output` / `-o` | `sensenova_u1_output.png` | Output file path |
| `--server` / `-s` | `http://localhost:8091` | Server URL |
| `--image-url` / `-i` | `None` | Input image URL or local path (img2img/img2text) |
| `--modality` / `-m` | `text2img` | `text2img`, `img2img`, `img2text`, `text2text` |
| `--height` | `2048` | Image height (image generation only) |
| `--width` | `2048` | Image width (image generation only) |
| `--num-steps` | `50` | Number of inference steps (image generation only) |
| `--seed` | `42` | Random seed |
| `--cfg-scale` | `4.0` | CFG scale (image generation only) |
| `--think` | `False` | Enable think mode |
