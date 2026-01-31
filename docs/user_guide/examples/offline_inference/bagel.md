# BAGEL-7B-MoT

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/bagel>.

## Set up

Please refer to the [stage configuration documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/stage_configs/) to configure memory allocation appropriately for your hardware setup.

## Run examples

**Note**: These examples work with the default configuration on an **NVIDIA A100 (80GB)**. We also tested on dual **NVIDIA RTX 5000 Ada (32GB each)**. For dual-GPU setups, please modify the stage configuration to distribute the model across devices.

Get into the bagel folder

```bash
cd examples/offline_inference/bagel
```

### Modality Control

BAGEL-7B-MoT supports multiple modality modes. You can control the mode using the `--modality` argument:

#### Text to Image (text2img)

- **Pipeline**: Text → Thinker  → DiT → VAE Decode → Image
- **Stages Used**: Stage 0 (Thinker) + Stage 1 (DiT)
- **KV Transfer**: Thinker sends KV cache to DiT for conditioned generation

Generate images from text prompts:

```bash
python end2end.py --model ByteDance-Seed/BAGEL-7B-MoT \
                  --modality text2img \
                  --prompts "A cute cat"
```

#### Image to Image (img2img)

- **Pipeline**: Image → VAE Encode → DiT → VAE Decode → New Image
- **Stages Used**: Stage 1 (DiT) only
- **Special**: Bypasses the Thinker stage, direct image-to-image transformation

Transform images based on text prompts:

```bash
python end2end.py --model ByteDance-Seed/BAGEL-7B-MoT \
                  --modality img2img \
                  --image-path /path/to/image.jpg \
                  --prompts "Let the woman wear a blue dress"
```

#### Image to Text (img2text)

- **Pipeline**: Image → ViT + VAE Encode → Thinker → Text Output
- **Stages Used**: Stage 0 (Thinker) only
- **Special**: Uses both VAE latent encoding AND ViT semantic encoding for comprehensive image understanding

Generate text descriptions from images:

```bash
python end2end.py --model ByteDance-Seed/BAGEL-7B-MoT \
                  --modality img2text \
                  --image-path /path/to/image.jpg \
                  --prompts "Describe this image in detail"
```

#### Text to Text (text2text)

- **Pipeline**: Text → Thinker → Text Output
- **Stages Used**: Stage 0 (Thinker) only
- **Special**: No visual components involved, operates as pure language model

Pure text generation:

```bash
python end2end.py --model ByteDance-Seed/BAGEL-7B-MoT \
                  --modality text2text \
                  --prompts "What is the capital of France?"

# You can load prompts from a text file (one prompt per line):  
python end2end.py --model ByteDance-Seed/BAGEL-7B-MoT \
                  --modality text2text \
                  --txt-prompts /path/to/prompts.txt
```

### Inference Steps

Control the number of inference steps for image generation:

```bash
# You can adjust steps to 100 to improve image quality
python end2end.py --model ByteDance-Seed/BAGEL-7B-MoT \
                  --modality text2img \
                  --steps 50 \
                  --prompts "A cute cat"
```

### Key arguments

BAGEL-7B-MoT supports **multiple modality modes** for different use cases.

The default yaml configuration deploys Thinker and DiT on the same GPU. You can use the default configuration file: [`bagel.yaml`](../../../vllm_omni/model_executor/stage_configs/bagel.yaml)

#### 📌 Command Line Arguments (end2end.py)

| Argument               | Type   | Default                       | Description                                                  |
| :--------------------- | :----- | :---------------------------- | :----------------------------------------------------------- |
| `--model`              | string | `ByteDance-Seed/BAGEL-7B-MoT` | Model path or name                                           |
| `--modality`           | choice | `text2img`                    | Modality mode: `text2img`, `img2img`, `img2text`, `text2text` |
| `--prompts`            | list   | `None`                        | Input text prompts directly                                  |
| `--txt-prompts`        | string | `None`                        | Path to txt file with one prompt per line                    |
| `--image-path`         | string | `None`                        | Input image path (for `img2img`/`img2text`)                  |
| `--steps`              | int    | `50`                          | Number of inference steps                                    |
| `--stage-configs-path` | string | `None`                        | Custom stage config file path                                |
| `--worker-backend`     | choice | `process`                     | Worker backend: `process` or `ray`                           |
| `--ray-address`        | string | `None`                        | Ray cluster address                                          |
| `--enable-stats`       | flag   | `False`                       | Enable statistics logging                                    |
| `--init-sleep-seconds` | int    | `20`                          | Initialization sleep time                                    |
| `--batch-timeout`      | int    | `5`                           | Batch timeout                                                |
| `--init-timeout`       | int    | `300`                         | Initialization timeout                                       |

------

#### ⚙️ Stage Configuration Parameters (bagel.yaml)

 **Stage 0 - Thinker (LLM Stage)**

| Parameter                        | Value                           | Description              |
| :------------------------------- | :------------------------------ | :----------------------- |
| `stage_type`                     | `llm`                           | Stage type               |
| `devices`                        | `"0"`                           | GPU device ID            |
| `max_batch_size`                 | `1`                             | Maximum batch size       |
| `model_stage`                    | `thinker`                       | Model stage identifier   |
| `model_arch`                     | `BagelForConditionalGeneration` | Model architecture       |
| `gpu_memory_utilization`         | `0.4`                           | GPU memory utilization   |
| `tensor_parallel_size`           | `1`                             | Tensor parallel size     |
| `max_num_batched_tokens`         | `32768`                         | Maximum batched tokens   |
| `omni_kv_config.need_send_cache` | `true`                          | Whether to send KV cache |

------

**Stage 1 - DiT (Diffusion Stage)**

| Parameter                        | Value       | Description                 |
| :------------------------------- | :---------- | :-------------------------- |
| `stage_type`                     | `diffusion` | Stage type                  |
| `devices`                        | `"0"`       | GPU device ID               |
| `max_batch_size`                 | `1`         | Maximum batch size          |
| `model_stage`                    | `dit`       | Model stage identifier      |
| `gpu_memory_utilization`         | `0.4`       | GPU memory utilization      |
| `omni_kv_config.need_recv_cache` | `true`      | Whether to receive KV cache |
| `engine_input_source`            | `[0]`       | Input source from Stage 0   |

------

#### 🔗 Runtime Configuration

| Parameter             | Value   | Description                      |
| :-------------------- | :------ | :------------------------------- |
| `window_size`         | `-1`    | Window size (-1 means unlimited) |
| `max_inflight`        | `1`     | Maximum inflight requests        |
| `shm_threshold_bytes` | `65536` | Shared memory threshold (64KB)   |

## FAQ

- If you encounter an error about the backend of librosa, try to install ffmpeg with the command below.

```bash
sudo apt update
sudo apt install ffmpeg
```

- If you don’t know how much VRAM is needed for the model or encounter the OOM error, you can try to decrease the max_model_len.

| Stage               | VRAM                         |
| :------------------ | :--------------------------- |
| Stage-0 (Thinker)   | **15.04 GiB** **+ KV Cache** |
| Stage-1 (DiT)       | **26.50 GiB**                |
| Total               | **~42 GiB + KV Cache**       |
