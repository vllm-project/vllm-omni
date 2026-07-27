# X-To-Text

Generate text from text or image inputs with vLLM-Omni's shared offline entrypoint.

- `x_to_text.py`: command-line script for text-to-text (T2T) and image-to-text (I2T) inference.

## Supported Models

| Model | T2T | I2T | Default deploy config |
| --- | --- | --- | --- |
| `ByteDance-Seed/BAGEL-7B-MoT` | Yes | Yes | `vllm_omni/deploy/bagel.yaml` |
| `tencent/HunyuanImage-3.0-Instruct` | Yes | Yes | `vllm_omni/deploy/hunyuan_image3_ar.yaml` |
| `bytedance-research/MammothModa2-Preview` | Yes | Yes | `vllm_omni/deploy/mammoth_moda2_ar.yaml` |

The script recognizes these model families from `config.json`, applies the
model-specific prompt format, and selects an AR-only deploy for HunyuanImage-3
and MammothModa2. BAGEL uses its registered default deploy.

## Text-To-Text

### BAGEL

```bash
python examples/offline_inference/x_to_text/x_to_text.py \
  --model ByteDance-Seed/BAGEL-7B-MoT \
  --prompt "Where is the capital of France?"
```

### HunyuanImage-3

```bash
python examples/offline_inference/x_to_text/x_to_text.py \
  --model tencent/HunyuanImage-3.0-Instruct \
  --prompt "Explain multimodal inference in three concise sentences."
```

### MammothModa2

```bash
python examples/offline_inference/x_to_text/x_to_text.py \
  --model bytedance-research/MammothModa2-Preview \
  --prompt "Explain multimodal inference in three concise sentences."
```

## Image-To-Text

Pass one image with `--image`. The default question can be replaced with any
instruction supported by the checkpoint.

```bash
python examples/offline_inference/x_to_text/x_to_text.py \
  --model ByteDance-Seed/BAGEL-7B-MoT \
  --image image.png \
  --prompt "Describe this image in detail."
```

Use the same command with either of the other supported checkpoints:

```bash
python examples/offline_inference/x_to_text/x_to_text.py \
  --model tencent/HunyuanImage-3.0-Instruct \
  --image image.png \
  --prompt "Describe this image in detail."
```

```bash
python examples/offline_inference/x_to_text/x_to_text.py \
  --model bytedance-research/MammothModa2-Preview \
  --image image.png \
  --prompt "Describe this image in detail."
```

## Key Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--model` | Required | Hugging Face model ID or local checkpoint path |
| `--prompt` | Required | User question or instruction |
| `--image` | None | Input image; enables I2T when supplied |
| `--output` | None | Optional text output file |
| `--deploy-config` | Model default | Override the deploy YAML |
| `--max-tokens` | `512` | Maximum generated tokens |
| `--temperature` | `0.0` | Sampling temperature |
| `--top-p` | `1.0` | Nucleus sampling probability |
| `--seed` | `42` | Sampling seed |
| `--trust-remote-code` | Off | Allow checkpoint-provided Python code |
| `--enforce-eager` | Off | Disable graph compilation |

## Hardware Notes

The committed HunyuanImage-3 AR-only default uses four GPUs with TP=4. To use a
different topology, pass a custom AR-only YAML through `--deploy-config`.
BAGEL and MammothModa2 use their repository deploy defaults unless overridden.
