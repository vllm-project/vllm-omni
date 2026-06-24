# Ming-flash-omni 2.0

## Installation

Please refer to [README.md](../../../README.md)

## Deployment modes

| Mode | Launch command | Output |
|------|---------------|--------|
| Thinker + Talker (omni-speech, default) | `vllm serve ... --omni` | Text + Audio |
| Thinker only (multimodal understanding) | `vllm serve ... --omni --deploy-config vllm_omni/deploy/ming_flash_omni_thinker_only.yaml` | Text |
| Thinker + Imagegen (text-to-image / img2img) | `vllm serve ... --omni --deploy-config vllm_omni/deploy/ming_flash_omni_image.yaml` | Image |

For standalone TTS (talker only), see the [Ming-flash-omni-TTS section in the Text-To-Speech hub](../text_to_speech/README.md#ming-flash-omni-tts).

## Run examples (Ming-flash-omni 2.0)

### Launch the Server

**Thinker + Talker (omni-speech, text + audio output):**
```bash
vllm serve Jonathan1909/Ming-flash-omni-2.0 --omni --port 8091
```

The model registry auto-loads corresponding deploy yaml.

**Thinker-only (text output):**
```bash
vllm serve Jonathan1909/Ming-flash-omni-2.0 --omni --port 8091 \
    --deploy-config vllm_omni/deploy/ming_flash_omni_thinker_only.yaml
```

Pass `--deploy-config /path/to/your_deploy.yaml` to use a custom deploy
config.

### Send Multi-modal Request

Shared Python client (supports `text | use_image | use_audio | use_video |
use_mixed_modalities`; pass `--image-path` / `--audio-path` / `--video-path`
for local files or URLs, `--modalities text` for output, `--help` for the
full flag list):

```bash
python examples/online_serving/openai_chat_completion_client_for_multimodal_generation.py \
    --model Jonathan1909/Ming-flash-omni-2.0 \
    --query-type use_mixed_modalities \
    --port 8091 --host localhost \
    --modalities text
```

Parameterized curl wrapper in this directory:

```bash
bash run_curl_multimodal_generation.sh text
bash run_curl_multimodal_generation.sh use_image
bash run_curl_multimodal_generation.sh use_audio
bash run_curl_multimodal_generation.sh use_video
bash run_curl_multimodal_generation.sh use_mixed_modalities
bash run_curl_multimodal_generation.sh use_image_gen
```

## Image generation (text-to-image)

Ming-flash-omni-2.0 also exposes an image-generation (diffusion) stage.
Launch with the image deploy YAML, which adds an image-gen stage behind
the thinker:

```bash
vllm serve Jonathan1909/Ming-flash-omni-2.0 --omni \
    --deploy-config vllm_omni/deploy/ming_flash_omni_image.yaml \
    --stage-init-timeout 1800 \
    --init-timeout 1800 \
    --port 8091
```

Then request image output by passing `"modalities": ["image"]`:

```bash
curl -s http://127.0.0.1:8091/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Jonathan1909/Ming-flash-omni-2.0",
      "messages": [{"role": "user", "content": "Please draw a cute cat."}],
      "modalities": ["image"]
    }' | jq -r '.choices[0].message.content[0].image_url.url | split(",")[1]' | base64 -d > ming_imagegen.png
```

### Optional knobs

Pass image-gen overrides as flat keys on the diffusion-stage `sampling_params_list[1].extra_args`:

| Key | Default | Description |
| --- | --- | --- |
| `height` / `width` | from config (1024) | Output resolution (multiples of `vae_scale_factor * 2`, currently 16). |
| `steps` | 30 | Number of FlowMatchEuler denoise steps. |
| `cfg` | 2.0 | Classifier-free guidance scale. |
| `seed` | 42 | Per-request RNG seed (deterministic when ByT5 is also seed-stable). |
| `byte5_text` | (auto) | Override the glyph text for ByT5 enhancement; raw strings are auto-wrapped to Ming's `Text "...". ` format. |
| `negative_prompt` | empty | Real CFG negative conditioning (set on **stage-0 thinker** `extra_args` so `expand_cfg_prompts` spawns the companion). |

Example with all the knobs:

```bash
curl http://127.0.0.1:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Jonathan1909/Ming-flash-omni-2.0",
    "modalities": ["image"],
    "sampling_params_list": [
      {
        "temperature": 0.4,
        "top_p": 0.9,
        "top_k": 1,
        "max_tokens": 1,
        "seed": 42,
        "extra_args": {
          "negative_prompt": "ugly, blurry, distorted"
        }
      },
      {
        "seed": 42,
        "extra_args": {
          "steps": 6,
          "cfg": 1.5,
          "height": 512,
          "width": 512,
          "seed": 123,
          "byte5_text": ["理解与生成统一"]
        }
      }
    ],
    "messages": [
      {
        "role": "user",
        "content": "Draw a poster."
      }
    ]
  }' \
  | jq -r '.choices[0].message.content[0].image_url.url | split(",")[1]' \
  | base64 -d > ming_imagegen_knobs.png
```

### img2img (reference image)

Add an `image_url` content part to the user message; the parser routes it
into the diffusion stage as `extra[reference_image]`:

```jsonc
"messages": [{
  "role": "user",
  "content": [
    {"type": "text", "text": "Change the background to a sandy beach at sunset."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,<base64>"}}
  ]
}]
```

### GPU layout

The shipped `ming_flash_omni_image.yaml` allocates the thinker on GPUs 0–3
(TP=4) and the diffusion stage on GPU 4 (TP=1). Copy the YAML and edit
`devices` per stage to relocate; with fewer GPUs available, drop
the thinker TP to 2 and run the diffusion stage on a free card. Image-gen
warmup takes roughly an extra 30–60 s on top of the thinker — set
`--stage-init-timeout 1800` if the default 300 s is too tight.

## Modality control

| `modalities` | Server config | Output |
|-------------|--------------|--------|
| `["text"]` or omitted | Thinker only | Text |
| `["audio"]` | Thinker + Talker | Audio (speech) |
| `["text", "audio"]` | Thinker + Talker | Text + Audio |
| `["image"]` | Thinker + Imagegen (image deploy YAML) | Image (PNG, base64 in `choices[0].message.content`) |

For ready-to-copy curl examples (text / audio / multimodal input, SSE
streaming, reasoning mode), see the recipe at
[`recipes/inclusionAI/Ming-flash-omni-2.0.md`](../../../recipes/inclusionAI/Ming-flash-omni-2.0.md).

## OpenAI Python SDK — streaming

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8091/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="Jonathan1909/Ming-flash-omni-2.0",
    messages=[
        {"role": "system", "content": [{"type": "text", "text": "你是一个友好的AI助手。\n\ndetailed thinking off"}]},
        {"role": "user", "content": "请详细介绍鹦鹉的生活习性。"},
    ],
    modalities=["text"],
    stream=True,
)
for chunk in response:
    for choice in chunk.choices:
        if hasattr(choice, "delta") and choice.delta.content:
            print(choice.delta.content, end="", flush=True)
print()
```

The `--stream` flag on the Python client script above shows the same pattern
driven by the shared multimodal client.
