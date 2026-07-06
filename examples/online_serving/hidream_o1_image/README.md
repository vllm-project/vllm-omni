# HiDream-O1-Image (online serving)

Online serving examples for HiDream-O1-Image via the OpenAI-compatible API.

## Start the server

```bash
# Dev variant (28 steps, no CFG — default)
bash run_server.sh

# Full variant (50 steps, guidance 5.0)
MODEL=HiDream-ai/HiDream-O1-Image bash run_server.sh
```

## Text-to-image via curl

```bash
bash run_curl_hidream_o1.sh
```

Or manually:

```bash
curl -s http://localhost:8095/v1/images/generations \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "A golden retriever in a field of sunflowers",
        "size": "1024x1024",
        "num_inference_steps": 28,
        "seed": 42
    }' | jq -r '.data[0].b64_json' | base64 -d > output.png
```

## Text-to-image via Python client

```bash
python openai_client_hidream_o1.py \
    --prompt "A cinematic mountain landscape at sunrise" \
    --height 1024 --width 1024 \
    --steps 28 \
    --seed 42 \
    --output output.png
```

## Requirements

- Server must be running (`bash run_server.sh`) before running any client.
- `transformers >= 4.57.1` on the server side.
- For the Full variant, set `--guidance-scale 5.0` in the Python client (or `"guidance_scale": 5.0` in the JSON body).
