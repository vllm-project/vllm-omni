# Ming-flash-omni 2.0

> Online serving for multimodal chat + standalone TTS

## Summary

- Vendor: inclusionAI
- Model: `Jonathan1909/Ming-flash-omni-2.0`
- Task: Multimodal chat with text, image, audio, or video input; standalone text-to-speech (TTS);
and image generation
- Mode: Online serving with the OpenAI-compatible API
- Maintainer: Community

## When to use this recipe

Use this recipe when you want a known-good starting point for serving
`Jonathan1909/Ming-flash-omni-2.0` with vLLM-Omni in one of three modes:

- **Thinker only** — multimodal understanding with text output.
- **Thinker + Talker (omni-speech)** — multimodal understanding with text and spoken output.
- **Thinker + Imagegen (DiT)** — text-to-image / img2img
- **Talker only (TTS)** — standalone text-to-speech via the OpenAI `/v1/audio/speech` endpoint.

## References

- Upstream model:
  [`inclusionAI/Ming`](https://github.com/inclusionAI/Ming)
- For offline inference and additional client variants, see the
  multimodal example dirs `examples/offline_inference/ming_flash_omni/` and
  `examples/online_serving/ming_flash_omni/`. The standalone TTS variant
  lives under the consolidated text-to-speech hub at
  `examples/offline_inference/text_to_speech/ming_flash_omni_tts/` and
  `examples/online_serving/text_to_speech/ming_flash_omni_tts/`.


## Hardware Support

This recipe documents reference GPU configurations for the two-stage
omni-speech deployment and the standalone TTS deployment.
Other hardware and configurations are welcome as community validation lands.

## GPU

### 4x H100 80GB — omni-speech/chat (thinker + talker)

The bundled `ming_flash_omni.yaml` runs the thinker with tensor parallel size
4 on GPUs 0–3 and the talker on GPU 3.
Adjust `devices` in the YAML to match your hardware.

#### Environment

- OS: Linux
- Python: 3.10+
- CUDA Driver Version: 590.48.01
- CUDA 13.0
- vLLM version: 0.19.0
- vLLM-Omni version or commit: 0.19.0rc1

#### Command

Thinker + talker (text and/or audio output):

```bash
vllm serve Jonathan1909/Ming-flash-omni-2.0 \
    --omni \
    --port 8091 \
    --log-stats
```

Thinker only (text-only output):

```bash
vllm serve Jonathan1909/Ming-flash-omni-2.0 \
    --omni \
    --deploy-config vllm_omni/deploy/ming_flash_omni_thinker_only.yaml \
    --port 8091
```

`--log-stats` is optional but recommended while validating the deployment.

#### Verification

Text output from a multimodal (image) input:

```bash
curl http://localhost:8091/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Jonathan1909/Ming-flash-omni-2.0",
      "messages": [
        {"role": "system", "content": [{"type": "text", "text": "你是一个友好的AI助手。\n\ndetailed thinking off"}]},
        {"role": "user", "content": [
          {"type": "image_url", "image_url": {"url": "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg"}},
          {"type": "text", "text": "Describe this image in detail."}
        ]}
      ],
      "modalities": ["text"]
    }'
```

Spoken response from a text query (save the WAV bytes):

```bash
curl http://localhost:8091/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Jonathan1909/Ming-flash-omni-2.0",
      "messages": [
        {"role": "system", "content": [{"type": "text", "text": "你是一个友好的AI助手。\n\ndetailed thinking off"}]},
        {"role": "user", "content": "请详细介绍鹦鹉的生活习性。"}
      ],
      "modalities": ["audio"]
    }' | jq -r '.choices[0].message.audio.data' | base64 -d > ming_omni_parrot.wav
```

Text + audio output from an audio input (swap `audio_url` for `video_url`
or `image_url` to exercise the other multimodal input paths):

```bash
curl http://localhost:8091/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Jonathan1909/Ming-flash-omni-2.0",
      "messages": [
        {"role": "system", "content": [{"type": "text", "text": "你是一个友好的AI助手。\n\ndetailed thinking off"}]},
        {"role": "user", "content": [
          {"type": "audio_url", "audio_url": {"url": "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/mary_had_lamb.ogg"}},
          {"type": "text", "text": "Please recognize the language of this speech and transcribe it. Format: oral."}
        ]}
      ],
      "modalities": ["text", "audio"]
    }' | jq -r '.choices[0].message.content'
```

Streaming text output via SSE (set `"stream": true`):

```bash
curl -N http://localhost:8091/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Jonathan1909/Ming-flash-omni-2.0",
      "messages": [
        {"role": "system", "content": [{"type": "text", "text": "你是一个友好的AI助手。\n\ndetailed thinking off"}]},
        {"role": "user", "content": "请详细介绍鹦鹉的生活习性。"}
      ],
      "modalities": ["text"],
      "stream": true
    }'
```

Each SSE event carries a `data:` line with a chat-completion chunk; text
deltas appear at `choices[0].delta.content`.

#### Notes

- Output modality is selected by the request body: `"modalities": ["text"]`,
  `["audio"]`, or `["text", "audio"]`. The two-stage omni-speech server must be launched
  for any request containing `audio`.
- Reasoning mode: flip the system prompt suffix from `detailed thinking off`
  to `detailed thinking on` in any request above.
- Memory usage: size depends on output modalities and multimodal input; leave
  headroom for video frames and audio caches.

## Image generation (text-to-image / img2img)

Ming-flash-omni-2.0 also exposes an image-generation (diffusion) stage. Launch with the image deploy YAML, which adds an image-gen stage behind the thinker.

The image-generation stage is a standard vLLM-Omni diffusion pipeline (`MingImagePipeline`); its request knobs are declared in `vllm_omni/model_extras/ming_flash_omni.py` and routed through `extra_body`, so they no longer need a bespoke `sampling_params_list` recipe (that form is still available for per-stage thinker sampling — see below).

```bash
vllm serve Jonathan1909/Ming-flash-omni-2.0 --omni \
    --deploy-config vllm_omni/deploy/ming_flash_omni_image.yaml \
    --stage-init-timeout 1800 \
    --init-timeout 1800 \
    --port 8091
```

With fewer GPUs, copy the YAML and drop the thinker to TP=2 with the DiT on a
free card.

### Online (text-to-image)

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

**Quick form — `extra_body`** (keys are filtered against the declared set and routed into every stage's `extra_args`). Convenient, but it does **not** let you set the thinker (stage-0) sampling params. Wrap the knobs in a literal `"extra_body"` object for raw curl; the OpenAI Python client's `extra_body=` kwarg produces the same request:

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

**Full control — `sampling_params_list`** (one entry per stage: `[thinker, imagegen]`).
Use this when you need to tune the thinker's own sampling (`temperature` / `top_p` / `top_k` / `max_tokens`), or to place knobs explicitly per stage.
Note `negative_prompt` must sit on the **stage-0 thinker** `extra_args`; the imagegen-stage knobs (`steps` / `cfg` / `height` / `width` / `seed` / `byte5_text`) go on the **stage-1** entry:

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

### Knobs (declared `extra_body` params)

| Key | Default | Description |
| --- | --- | --- |
| `height` / `width` | 1024 | Output resolution (multiples of `vae_scale_factor * 2`, currently 16). |
| `steps` | 30 | Number of FlowMatchEuler denoise steps. |
| `cfg` | 2.0 | Classifier-free guidance scale. |
| `seed` | 42 | Per-request RNG seed. |
| `byte5_text` | (auto) | Glyph text for ByT5 enhancement; raw strings are auto-wrapped to Ming's `Text "…". ` format. Auto-extracted from quoted spans in the prompt when omitted. |
| `negative_prompt` | (empty) | Real CFG negative conditioning. Spawns a CFG-text companion via `expand_cfg_prompts`; **online / text-to-image only** (offline uses Ming's default zero-negative). |

For img2img, add an `image_url` content part to the user message (online) or pass `--image` (offline); the reference image is routed into the DiT stage as `extra[reference_image]`.

### 1x H100 80GB — standalone TTS (talker only)

The bundled `ming_flash_omni_tts.yaml` runs the talker on a single GPU and exposes the OpenAI `/v1/audio/speech` endpoint.

#### Environment

- OS: Linux
- Python: 3.10+
- CUDA Driver Version: 590.48.01
- CUDA 13.0
- vLLM version: 0.19.0
- vLLM-Omni version or commit: 0.19.0rc1

#### Command

```bash
vllm serve Jonathan1909/Ming-flash-omni-2.0 \
    --omni \
    --deploy-config vllm_omni/deploy/ming_flash_omni_tts.yaml \
    --port 8091 \
    --log-stats
```

`--log-stats` is optional but recommended while validating the deployment.

#### Verification

Basic curl:

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Jonathan1909/Ming-flash-omni-2.0",
      "input": "我会一直在这里陪着你。",
      "response_format": "wav"
    }' --output ming_online.wav
```

Speaker selection (e.g. `lingguang`):

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Jonathan1909/Ming-flash-omni-2.0",
      "input": "春天来了，万物复苏，大地一片生机盎然。田野里的油菜花开得金灿灿的，蜜蜂在花丛中忙碌地采蜜。远处的山坡上，桃花和杏花竞相绽放，粉的白的交织在一起，美不胜收。清晨的微风带着泥土的芬芳，轻轻拂过脸颊，让人感到无比惬意。孩子们在田间小路上追逐嬉戏，老人们坐在门前晒太阳，享受着这份宁静与美好。",
      "speaker": "lingguang",
      "response_format": "wav"
    }' --output ming_online_lingguang.wav
```

#### Notes

- The OpenAI `instructions` field is forwarded to the talker as the caption JSON — pass a raw string for `风格` (style) only, or a JSON-encoded object for multiple entries such as `方言` (dialect) and `情感` (emotion).
