# HunyuanImage-3.0-Instruct

## Set up

Please refer to the [stage configuration documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/stage_configs/) to configure memory allocation appropriately for your hardware setup.

## Run examples

**Note**: These examples work with the default configuration on **8x NVIDIA L40S (48GB)**. For different GPU setups, modify the stage configuration to adjust device allocation and memory utilization.

Get into the hunyuan_image3 folder:

```bash
cd examples/offline_inference/hunyuan_image3
```

### Modality Control

HunyuanImage-3.0-Instruct supports multiple modality modes. You can control the mode using the `--modality` argument:

#### Text to Image (text2img)

- **Pipeline**: Text → AR (CoT + latent tokens) → DiT (denoise) → VAE Decode → Image
- **Stages Used**: Stage 0 (AR) + Stage 1 (DiT)
- **KV Transfer**: AR sends KV cache to DiT for conditioned generation
- **Default Config**: `hunyuan_image3_t2i.yaml`

```bash
python end2end.py --model tencent/HunyuanImage-3.0-Instruct \
                  --modality text2img \
                  --prompts "A cute cat sitting on a windowsill watching the sunset"
```

#### Image to Image (img2img)

- **Pipeline**: Image + Text → AR (CoT + recaption + latent) → DiT → Edited Image
- **Stages Used**: Stage 0 (AR) + Stage 1 (DiT)
- **KV Transfer**: AR sends KV cache to DiT
- **Default Config**: `hunyuan_image3_it2i.yaml`

```bash
python end2end.py --model tencent/HunyuanImage-3.0-Instruct \
                  --modality img2img \
                  --image-path /path/to/image.png \
                  --prompts "Make the petals neon pink"
```

#### Image to Text (img2text)

- **Pipeline**: Image + Question → AR → Text description
- **Stages Used**: Stage 0 (AR) only
- **Default Config**: `hunyuan_image3_i2t.yaml`

```bash
python end2end.py --model tencent/HunyuanImage-3.0-Instruct \
                  --modality img2text \
                  --image-path /path/to/image.jpg \
                  --prompts "Describe the content of the picture."
```

#### Text to Text (text2text)

- **Pipeline**: Text → AR → Text
- **Stages Used**: Stage 0 (AR) only
- **Default Config**: `hunyuan_image3_t2t.yaml`

```bash
python end2end.py --model tencent/HunyuanImage-3.0-Instruct \
                  --modality text2text \
                  --prompts "What is the capital of France?"
```

### Inference Steps & Guidance

Control generation quality for image modalities:

```bash
python end2end.py --modality text2img \
                  --steps 50 \
                  --guidance-scale 5.0 \
                  --height 1024 --width 1024 \
                  --prompts "A photo-realistic sunset over the ocean"
```

### Key Arguments

#### 📌 Command Line Arguments (end2end.py)

| Argument               | Type   | Default                              | Description                                                  |
| :--------------------- | :----- | :----------------------------------- | :----------------------------------------------------------- |
| `--model`              | string | `tencent/HunyuanImage-3.0-Instruct` | Model path or name                                           |
| `--modality`           | choice | `text2img`                           | Modality: `text2img`, `img2img`, `img2text`, `text2text`     |
| `--prompts`            | list   | `None`                               | Input text prompts                                           |
| `--image-path`         | string | `None`                               | Input image path (for `img2img`/`img2text`)                  |
| `--output`             | string | `.`                                  | Output directory for saved images                            |
| `--steps`              | int    | `50`                                 | Number of inference steps                                    |
| `--guidance-scale`     | float  | `5.0`                                | Classifier-free guidance scale                               |
| `--seed`               | int    | `42`                                 | Random seed                                                  |
| `--height`             | int    | `1024`                               | Output image height                                          |
| `--width`              | int    | `1024`                               | Output image width                                           |
| `--bot-task`           | string | auto                                 | Override prompt task (e.g. `it2i_think`, `t2i_recaption`)    |
| `--sys-type`           | string | auto                                 | Override system prompt type (e.g. `en_unified`, `en_vanilla`) |
| `--stage-configs-path` | string | auto                                 | Custom stage config YAML path                                |
| `--enforce-eager`      | flag   | `False`                              | Disable torch.compile                                        |
| `--init-timeout`       | int    | `300`                                | Initialization timeout (seconds)                             |

------

#### ⚙️ Stage Configurations

| Config YAML                         | Modality  | Stages | GPUs   | Description                           |
| :---------------------------------- | :-------- | :----- | :----- | :------------------------------------ |
| `hunyuan_image3_t2i.yaml`           | text2img  | 2      | 8      | T2I with AR→DiT, 4 GPU each          |
| `hunyuan_image3_it2i.yaml`          | img2img   | 2      | 8      | IT2I with AR→DiT, 4 GPU each         |
| `hunyuan_image3_i2t.yaml`           | img2text  | 1      | 4      | I2T (AR only)                         |
| `hunyuan_image3_t2t.yaml`           | text2text | 1      | 4      | T2T (AR only)                         |
| `hunyuan_image3_t2i_2gpu.yaml`      | text2img  | 2      | 2      | T2I for 2-GPU setups                  |
| `hunyuan_image3_moe.yaml`           | text2img  | 2      | 8      | T2I with MoE AR→DiT KV reuse          |
| `hunyuan_image3_moe_dit_2gpu_fp8.yaml` | text2img | 2   | 2      | T2I with FP8 quantization             |

------

## Using MoE Config

The `hunyuan_image3_moe.yaml` config enables AR→DiT KV cache reuse with 8 GPUs (4 for AR + 4 for DiT).

```bash
python end2end.py --model tencent/HunyuanImage-3.0-Instruct \
                  --modality text2img \
                  --stage-configs-path hunyuan_image3_moe.yaml \
                  --prompts "A cute cat"
```

------

## Prompt Format

HunyuanImage-3.0 uses a pretrain template format:

```
<|startoftext|>{system_prompt}{<img>}{trigger_tag}{user_prompt}
```

- `<img>`: Placeholder for each input image (auto-inserted by `prompt_utils.py`)
- Trigger tags: `<think>` (CoT), `<recaption>` (recaptioning)
- System prompt: Auto-selected based on task

The `prompt_utils.build_prompt()` handles this formatting automatically.

------

## FAQ

- **OOM errors**: Decrease `gpu_memory_utilization` in the YAML stage config, or use a smaller `max_num_batched_tokens`.
- **Custom image sizes**: Use `--height` and `--width` flags (multiples of 16 recommended).

| Stage             | VRAM (approx)        |
| :---------------- | :------------------- |
| Stage 0 (AR)      | ~15 GiB + KV Cache   |
| Stage 1 (DiT)     | ~30 GiB              |
| Total (8-GPU)     | ~45 GiB + KV Cache   |
