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


## Image generation (text-to-image / img2img)

Ming-flash-omni-2.0 also exposes an image-generation (diffusion) stage. Launch with the image deploy YAML, which adds an image-gen stage behind the thinker.

The image-generation stage is a standard vLLM-Omni diffusion pipeline (`MingImagePipeline`); its request knobs are declared in `vllm_omni/model_extras/ming_flash_omni.py` and routed through `extra_body`, so they no longer need a bespoke `sampling_params_list` recipe (that form is still available for per-stage thinker sampling — see below).

### Launch

```bash
vllm serve Jonathan1909/Ming-flash-omni-2.0 --omni \
    --deploy-config vllm_omni/deploy/ming_flash_omni_image.yaml \
    --stage-init-timeout 1800 \
    --init-timeout 1800 \
    --port 8091
```


### Text-to-image

Request image output with `"modalities": ["image"]`:

```bash
curl -s http://127.0.0.1:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Jonathan1909/Ming-flash-omni-2.0",
    "messages": [
      {
        "role": "user",
        "content": "Please draw a cute cat."
      }
    ],
    "modalities": ["image"]
  }' \
  | jq -r '.choices[0].message.content[0].image_url.url | split(",")[1]' \
  | base64 -d > ming_imagegen.png
```

Pass generation knobs under a literal `extra_body` object (the OpenAI Python client's `extra_body=` kwarg produces the same request). Keys are filtered against the declared set and routed into every stage's `extra_args`:

```bash
curl http://127.0.0.1:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Jonathan1909/Ming-flash-omni-2.0",
    "modalities": ["image"],
    "messages": [
      {
        "role": "user",
        "content": "Draw a poster."
      }
    ],
    "extra_body": {
      "steps": 6,
      "cfg": 1.5,
      "height": 512,
      "width": 512,
      "seed": 123,
      "byte5_text": ["理解与生成统一"],
      "negative_prompt": "ugly, blurry, distorted"
    }
  }' \
  | jq -r '.choices[0].message.content[0].image_url.url | split(",")[1]' \
  | base64 -d > ming_imagegen_extra_body.png
```

NOTE: `extra_body` does **not** set the thinker's own sampling params. To tune the thinker (stage-0) sampling (`temperature` / `top_p` / `top_k` / `max_tokens`) or place knobs explicitly per stage, use `sampling_params_list` (`[thinker, imagegen]`).
`negative_prompt` must sit on the **stage-0** entry to trigger the real-CFG companion; the imagegen knobs go on **stage-1**:

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

Add an `image_url` content part (a base64 data URL) to the user message; it is routed into the DiT stage as `extra[reference_image]`. base64-encode a local file and stream it through `jq` via stdin - piping `base64 -> jq -> curl` so that avoids the shell `ARG_MAX` limit that inlining a large base64 string in `-d '…'` hits.

```bash
# Reference image: figures/cases/person_gen_05.png from the upstream Ming repo
# Check https://github.com/inclusionAI/Ming/blob/3954fcb880ff5e61ff128bcf7f1ec344d46a6fe3/examples/vllm_demo.py
wget https://raw.githubusercontent.com/inclusionAI/Ming/3954fcb880ff5e61ff128bcf7f1ec344d46a6fe3/figures/cases/person_gen_05.png

base64 -w0 person_gen_05.png \
| jq -Rs --arg prompt "Put a pair of sunglasses on the person." '{
    model: "Jonathan1909/Ming-flash-omni-2.0",
    modalities: ["image"],
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: $prompt },
          { type: "image_url", image_url: { url: ("data:image/png;base64," + (. | rtrimstr("\n"))) } }
        ]
      }
    ]
  }' \
| curl http://127.0.0.1:8091/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d @- \
| jq -r '.choices[0].message.content[0].image_url.url | split(",")[1]' \
| base64 -d > ming_img2img.png
```

The reference image can also be a public URL (`"url": "https://…/photo.jpg"`) or the simplified `{"image": "<base64>"}` content-part form — see the [image-to-image request formats](../image_to_image/README.md#request-format).

### Knobs (declared `extra_body` params)

| Key | Default | Description |
| --- | --- | --- |
| `height` / `width` | 1024 | Output resolution (multiples of `vae_scale_factor * 2`, currently 16). |
| `steps` | 30 | Number of FlowMatchEuler denoise steps. |
| `cfg` | 2.0 | Classifier-free guidance scale. |
| `seed` | 42 | Per-request RNG seed. |
| `byte5_text` | (auto) | Glyph text for ByT5 enhancement; raw strings are auto-wrapped to Ming's `Text "…". ` format. Auto-extracted from quoted spans in the prompt when omitted. |
| `negative_prompt` | (empty) | Real CFG negative conditioning (text-to-image only). |

For the offline `text_to_image.py` / `image_edit.py` scripts and the full knob reference, see the [image-generation section in the recipe](../../../recipes/inclusionAI/Ming-flash-omni-2.0.md#image-generation-text-to-image--img2img).

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
