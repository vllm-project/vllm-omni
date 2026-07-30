# HunyuanImage-3.0-Instruct

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/hunyuan_image3>.

HunyuanImage-3.0-Instruct now runs entirely through **shared task examples**,
with all model-specific knobs declared centrally in
`vllm_omni/model_extras/hunyuan_image3.py` and routed via `--extra-body` /
`--extra-args`. There is no dedicated script left in this directory.

| Modality | How to run |
| :--- | :--- |
| Text to image (`t2i`) | shared `examples/offline_inference/text_to_image/text_to_image.py` |
| Image editing (`it2i`) | shared `examples/offline_inference/image_to_image/image_edit.py` |
| Image to text (`i2t`) | shared `examples/offline_inference/x_to_text/x_to_text.py --image ...` |
| Text to text (`t2t`) | shared `examples/offline_inference/x_to_text/x_to_text.py` |

All four paths build the AR prefill + stop tokens through the declarative
`vllm_omni.model_extras.hunyuan_image3.build_ar_stage_inputs` seam (t2i/it2i)
or the equivalent `build_x_to_text_prompt` seam (t2t/i2t) — both wrap
`prompt_utils.build_ar_prompt_inputs`, so prompt formatting is identical
across all four offline paths. The OpenAI server's `serving_chat.py` does
not go through this seam yet — it independently calls the same underlying
`build_prompt_tokens`/`resolve_stop_token_ids` primitives, which can diverge
from this seam's `bot_task` resolution (see `build_ar_prompt_inputs`'s
docstring).

## Deploy Configs

| File | Topology | Default use |
| :--- | :--- | :--- |
| `vllm_omni/deploy/hunyuan_image_3_moe.yaml` | AR + DiT | Text-to-image and image-editing. |
| `vllm_omni/deploy/hunyuan_image3_ar.yaml` | AR only | Image-to-text and text-to-text. |
| `vllm_omni/deploy/hunyuan_image3_dit.yaml` | DiT only | Standalone diffusion stage. |

HunyuanImage3 selects its AR + DiT / AR-only / DiT-only topology by deploy file,
so pass the matching `--deploy-config` for the scenario.

## Declared `extra_body` parameters

Declared in `vllm_omni/model_extras/hunyuan_image3.py`
(`HUNYUAN_IMAGE3_EXTRA_BODY_PARAMS`) and filtered automatically for this model:

| Key | Description |
| :--- | :--- |
| `bot_task` | Prompt mode / trigger tag: `think`, `recaption`, `think_recaption`, `vanilla`, or `null` for plain mode. Omitted → each task's default. |
| `use_system_prompt` | System-prompt preset, e.g. `en_unified`, `en_recaption`, `en_vanilla`. |
| `system_prompt` | Custom system prompt (used with `use_system_prompt=custom`). |
| `negative_prompt` | Negative prompt for classifier-free guidance. |

Standard sampling knobs (`--guidance-scale`, `--num-inference-steps`, `--seed`,
`--height`, `--width`) are handled generically by the shared scripts and do not
go through `extra_body`.

## Run examples

### Text to image (shared script)

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model tencent/HunyuanImage-3.0-Instruct \
  --deploy-config vllm_omni/deploy/hunyuan_image_3_moe.yaml \
  --trust-remote-code \
  --prompt "A cute cat sitting on a windowsill watching the sunset" \
  --height 1024 --width 1024 \
  --guidance-scale 5.0 --num-inference-steps 50 --seed 42 \
  --extra-body '{"bot_task": "think", "use_system_prompt": "en_recaption"}' \
  --output hunyuan_t2i.png
```

### Image editing (shared script)

```bash
python examples/offline_inference/image_to_image/image_edit.py \
  --model tencent/HunyuanImage-3.0-Instruct \
  --deploy-config vllm_omni/deploy/hunyuan_image_3_moe.yaml \
  --trust-remote-code \
  --image /path/to/image.png \
  --prompt "Make the petals neon pink" \
  --guidance-scale 5.0 --num-inference-steps 50 --seed 42 \
  --extra-args '{"bot_task": "think"}' \
  --output hunyuan_edit.png
```

`image_edit.py` accepts up to 3 reference images (repeat `--image`) for
HunyuanImage-3.0 multi-image fusion.

### Image to text (shared script)

```bash
python examples/offline_inference/x_to_text/x_to_text.py \
  --model tencent/HunyuanImage-3.0-Instruct \
  --image /path/to/image.jpg \
  --prompt "Describe the content of the picture."
```

### Text to text (shared script)

```bash
python examples/offline_inference/x_to_text/x_to_text.py \
  --model tencent/HunyuanImage-3.0-Instruct \
  --prompt "What is the capital of France?"
```

## Online serving

Online serving selects the topology with `--deploy-config` at startup, then uses
the OpenAI-compatible endpoints (`bot_task` / `use_system_prompt` are accepted as
request / `extra_body` fields):

| Scenario | Server deploy | Request |
| :--- | :--- | :--- |
| Text to image | `hunyuan_image_3_moe.yaml` | `POST /v1/images/generations`, or `/v1/chat/completions` with `"modalities": ["image"]`. |
| Image editing | `hunyuan_image_3_moe.yaml` | `POST /v1/images/edits`. |
| Image/text to text | `hunyuan_image3_ar.yaml` | `POST /v1/chat/completions` with `"modalities": ["text"]`. |
| DiT-only generation | `hunyuan_image3_dit.yaml` | `POST /v1/images/generations`. |

## Prompt format

HunyuanImage-3.0-Instruct uses an instruct chat template:

```text
<|startoftext|>{system_prompt}

User: {<img>?}{user_prompt}

Assistant: {trigger_tag?}
```

- `<img>`: placeholder per input image (single token; expanded by the multimodal pipeline).
- Trigger tags: `<think>` (CoT) / `<recaption>` (recaptioning) after `Assistant: `.
- System prompt: auto-selected from `task` and `bot_task`.
- `bot_task='vanilla'` with `task='t2i'` uses the bare pretrain template.

`vllm_omni.diffusion.models.hunyuan_image3.prompt_utils.build_prompt_tokens()`
does segment-by-segment tokenization matching HF `apply_chat_template`, and
`build_ar_prompt_inputs()` wraps it together with the AR stop-token resolution.

## Example materials

??? abstract "reproduce.sh"
    ``````sh
    --8<-- "examples/offline_inference/hunyuan_image3/reproduce.sh"
    ``````
