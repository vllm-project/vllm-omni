# HunyuanImage-3.0-Instruct

This example runs HunyuanImage-3.0-Instruct offline with the unified deploy
YAMLs under `vllm_omni/deploy/`.

## Deploy Configs

| File | Topology | Default use |
| :--- | :--- | :--- |
| `vllm_omni/deploy/hunyuan_image3.yaml` | AR + DiT | Default for `text2img` and `img2img`. |
| `vllm_omni/deploy/hunyuan_image3_ar.yaml` | AR only | Default for `img2text` and `text2text`. |
| `vllm_omni/deploy/hunyuan_image3_dit.yaml` | DiT only | Standalone diffusion stage. Pass it explicitly with `--deploy-config`. |

The example chooses a deploy config automatically when `--deploy-config` and
`--stage-configs-path` are both omitted:

| `--modality` | `mode` passed to Omni | Default deploy |
| :--- | :--- | :--- |
| `text2img` | `text-to-image` | `hunyuan_image3.yaml` |
| `img2img` | `image-editing` | `hunyuan_image3.yaml` |
| `img2text` | `image-to-text` | `hunyuan_image3_ar.yaml` |
| `text2text` | `text-to-text` | `hunyuan_image3_ar.yaml` |

`--modality` is an offline example convenience flag. It maps to the internal
`mode` argument passed to `Omni(...)` by this script. HunyuanImage3 uses
separate deploy YAMLs for AR + DiT, AR-only, and DiT-only topologies, so the
stage topology is selected by the deploy file rather than by YAML mode
overrides.

Online serving does not expose a `--modality` flag or accept `mode` as an API
request field. Choose the deploy topology when starting the server with
`--deploy-config`, then use the OpenAI-compatible endpoint and request shape for
the scenario. The `modalities` request field is used by the chat completions
path; the image endpoints infer the image task from the endpoint and payload.

| Online scenario | Server deploy | Request |
| :--- | :--- | :--- |
| Text to image | `--deploy-config vllm_omni/deploy/hunyuan_image3.yaml` | `POST /v1/images/generations`, or `POST /v1/chat/completions` with `"modalities": ["image"]`. |
| Image editing | `--deploy-config vllm_omni/deploy/hunyuan_image3.yaml` | `POST /v1/images/edits`. |
| Image/text to text | `--deploy-config vllm_omni/deploy/hunyuan_image3_ar.yaml` | `POST /v1/chat/completions` for text output, for example with `"modalities": ["text"]`. |
| DiT-only image generation | `--deploy-config vllm_omni/deploy/hunyuan_image3_dit.yaml` | `POST /v1/images/generations`. |

## Run Examples

Text to image, using the default AR + DiT deploy:

```bash
python examples/offline_inference/hunyuan_image3/end2end.py \
  --model tencent/HunyuanImage-3.0-Instruct \
  --modality text2img \
  --prompts "A cute cat sitting on a windowsill watching the sunset"
```

Image editing, using the default AR + DiT deploy:

```bash
python examples/offline_inference/hunyuan_image3/end2end.py \
  --model tencent/HunyuanImage-3.0-Instruct \
  --modality img2img \
  --image-path /path/to/image.png \
  --prompts "Make the petals neon pink"
```

Image to text, using the AR-only deploy:

```bash
python examples/offline_inference/hunyuan_image3/end2end.py \
  --model tencent/HunyuanImage-3.0-Instruct \
  --modality img2text \
  --image-path /path/to/image.jpg \
  --prompts "Describe the content of the picture."
```

Text to text, using the AR-only deploy:

```bash
python examples/offline_inference/hunyuan_image3/end2end.py \
  --model tencent/HunyuanImage-3.0-Instruct \
  --modality text2text \
  --prompts "What is the capital of France?"
```

Standalone DiT, using the DiT-only deploy explicitly:

```bash
python examples/offline_inference/hunyuan_image3/end2end.py \
  --model tencent/HunyuanImage-3.0-Instruct \
  --modality text2img \
  --deploy-config vllm_omni/deploy/hunyuan_image3_dit.yaml \
  --prompts "A cinematic portrait of an astronaut in a greenhouse"
```

Override the default full AR + DiT deploy explicitly:

```bash
python examples/offline_inference/hunyuan_image3/end2end.py \
  --model tencent/HunyuanImage-3.0-Instruct \
  --modality text2img \
  --deploy-config vllm_omni/deploy/hunyuan_image3.yaml \
  --prompts "A cute cat"
```

## Additional Config

You can pass diffusion worker `additional_config` from the offline example as a JSON object.
This maps to the upstream vLLM `VllmConfig.additional_config` platform extension field:
https://docs.vllm.ai/en/stable/api/vllm/config/#vllm.config.VllmConfig.additional_config

```bash
python end2end.py --modality text2img \
                  --prompts "A cute cat" \
                  --additional-config '{"torchair_graph_config":{"enabled":true}}'
```


## Key Arguments

| Argument | Description |
| :--- | :--- |
| `--deploy-config` | Preferred config path for unified deploy YAMLs. |
| `--stage-configs-path` | Legacy stage config path, kept only for compatibility. Prefer `--deploy-config`. |
| `--additional-config` | JSON object forwarded to diffusion worker `additional_config`. |
| `--modality` | Offline-only convenience flag. One of `text2img`, `img2img`, `img2text`, `text2text`. It selects prompt formatting, internal `mode`, and default deploy config for this script. Online serving uses `--deploy-config` plus the endpoint and, for chat completions, request `modalities` instead. |
| `--steps` | Number of diffusion inference steps for image generation. |
| `--guidance-scale` | Classifier-free guidance scale for image generation. |
| `--height`, `--width` | Output image size for `text2img`. |
| `--bot-task` | Override prompt mode. `none`, `think`, `recaption`, `think_recaption`, or `vanilla`. |
| `--sys-type` | Override the system prompt type, for example `en_unified` or `en_vanilla`. |
| `--vae-use-tiling` | Enable VAE tiling for memory reduction. |

## Notes

- `hunyuan_image3_ar.yaml` is a 4-card AR-only text/comprehension deploy.
- `hunyuan_image3_dit.yaml` is a single-stage DiT deploy with `stage_id: 0`.
- The old HunyuanImage3 YAMLs under `model_executor/stage_configs/` and `platforms/*/stage_configs/` have been folded into the deploy YAMLs.

## Prompt Format

HunyuanImage-3.0-Instruct uses an instruct chat template:

```text
<|startoftext|>{system_prompt}

User: {<img>?}{user_prompt}

Assistant: {trigger_tag?}
```

- `<img>`: Placeholder for each input image (single token; expanded by the multimodal pipeline).
- Trigger tags: `<think>` for CoT and `<recaption>` for recaptioning, placed after `Assistant: `.
- System prompt: Auto-selected from `task` and `bot_task`.
- `bot_task='vanilla'` with `task='t2i'` uses the bare pretrain template.

The shared `vllm_omni.diffusion.models.hunyuan_image3.prompt_utils.build_prompt_tokens()`
helper handles segment-by-segment tokenization and matches HF `apply_chat_template`.
