# Qwen3-Omni

## Prerequisites and Dependencies

Before running the Qwen3-Omni examples, ensure you have the required dependencies installed.

### System Requirements

- Python 3.10 or higher
- At least 16GB VRAM for basic inference (24GB+ recommended)
- CUDA 11.8+ or compatible GPU drivers

### Required Python Packages

Install the core vLLM-Omni with development dependencies:

```bash
# From the repository root
pip install -e ".[dev]"
```

This automatically installs all required packages including:
- `librosa` - Audio processing
- `soundfile` - Audio I/O
- `resampy` - Audio resampling
- `imageio[ffmpeg]` - Video/image handling
- `opencv-python` - Computer vision utilities
- `PIL (Pillow)` - Image processing

### System-Level Dependencies

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download ffmpeg from [ffmpeg.org](https://ffmpeg.org/download.html) or use:
```bash
choco install ffmpeg
```

## Setup
Please refer to the [stage configuration documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/stage_configs/) to configure memory allocation appropriately for your hardware setup.

## Run examples

### Multiple Prompts
Get into the example folder
```bash
cd examples/offline_inference/qwen3_omni
```
Then run the command below. Note: for processing large volume data, it uses py_generator mode, which will return a python generator from Omni class.
```bash
bash run_multiple_prompts.sh
```
### Single Prompt
Get into the example folder
```bash
cd examples/offline_inference/qwen3_omni
```
Then run the command below.
```bash
bash run_single_prompt.sh
```
If you have not enough memory, you can set thinker with tensor parallel. Just run the command below.
```bash
bash run_single_prompt_tp.sh
```

### Modality control
If you want to control output modalities, e.g. only output text, you can run the command below:
```bash
python end2end.py --output-wav output_audio \
                  --query-type use_audio \
                  --modalities text
```

#### Using Local Media Files
The `end2end.py` script supports local media files (audio, video, image) via command-line arguments:

```bash
# Use local video file
python end2end.py --query-type use_video --video-path /path/to/video.mp4

# Use local image file
python end2end.py --query-type use_image --image-path /path/to/image.jpg

# Use local audio file
python end2end.py --query-type use_audio --audio-path /path/to/audio.wav

# Combine multiple local media files
python end2end.py --query-type mixed_modalities \
    --video-path /path/to/video.mp4 \
    --image-path /path/to/image.jpg \
    --audio-path /path/to/audio.wav
```

If media file paths are not provided, the script will use default assets. Supported query types:
- `use_video`: Video input
- `use_image`: Image input
- `use_audio`: Audio input
- `text`: Text-only query
- `multi_audios`: Multiple audio inputs
- `mixed_modalities`: Combination of video, image, and audio inputs

### Async-chunk (offline)

For true stage-level concurrency -- where downstream stages (Talker, Code2Wav)
start **before** the upstream stage (Thinker) finishes -- use the async_chunk
example. This requires:

1. A stage config YAML with ``async_chunk: true`` (e.g.
   ``qwen3_omni_moe_async_chunk.yaml``).
2. Hardware that matches the config (e.g. 2x H100 for the default 3-stage
   config).

The async_chunk example uses ``AsyncOmni`` instead of the synchronous ``Omni``
class, which enables the async orchestrator to receive stage-0 intermediate
outputs and trigger downstream stages early. Chunk data flows directly between
stage workers via the in-worker ``OmniChunkTransferAdapter`` / connector,
**not** through the orchestrator.

#### Single prompt
```bash
cd examples/offline_inference/qwen3_omni
bash run_single_prompt_async_chunk.sh
```

#### Multiple prompts with concurrency control
```bash
bash run_multiple_prompts_async_chunk.sh --max-in-flight 4
```

#### Text-only output (skip audio generation)
```bash
python end2end_async_chunk.py --query-type text --modalities text
```

#### Custom stage config
```bash
python end2end_async_chunk.py \
    --query-type use_audio \
    --stage-configs-path /path/to/your_async_chunk.yaml
```

> **Note**: The synchronous ``end2end.py`` (using ``Omni``) is still the
> recommended entry point for non-async-chunk workflows. Only use the
> async_chunk example when you need the stage-level concurrency semantics
> described in PR #962 / #1151.

### FAQ

**Q: I get a `ModuleNotFoundError` when running examples**

A: Ensure you installed vLLM-Omni with development dependencies:
```bash
pip install -e ".[dev]"
```

**Q: Error about backend of librosa or ffmpeg not found**

A: Install ffmpeg (see [System-Level Dependencies](#system-level-dependencies) section) or ensure it's in your PATH.

**Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Q: The script fails with audio processing errors**

A: This typically indicates missing audio dependencies. Install them with:
```bash
pip install librosa soundfile resampy
```

**Q: How do I use my own media files?**

A: Use the `--*-path` arguments to specify local files (see [Using Local Media Files](#using-local-media-files) section).
