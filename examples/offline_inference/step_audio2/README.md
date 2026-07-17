# Step-Audio2: Offline inference

This directory contains examples for running offline inference with Step-Audio2 using vLLM-Omni.

## Model Overview

Step-Audio2 is a two-stage audio model:

- **Stage 0 (Thinker)**: Audio understanding → Text + Audio tokens
  - Input: Audio (16kHz)
  - Output: Text transcription + Audio tokens for synthesis

- **Stage 1 (Token2Wav)**: Audio synthesis
  - Input: Audio tokens + Speaker prompt wav
  - Output: Synthesized audio waveform (24kHz)

## Hardware Requirements

| Mode | GPU Configuration | VRAM Required |
|------|-------------------|---------------|
| ASR (S2T) | 1x GPU | ~20-25GB |
| TTS/S2ST (single GPU) | 1x GPU | ~40-50GB |
| TTS/S2ST (multi GPU) | 2x GPU | GPU0: ~28GB, GPU1: ~22GB |

**Tested on:**
- 1x NVIDIA H100 80GB (single-card S2ST)
- 2x NVIDIA A10 40GB (multi-card S2ST)

**Notes:**
- Single GPU mode requires high VRAM due to both stages sharing memory
- Multi GPU mode separates Stage 0 (Thinker) and Stage 1 (Token2Wav) across GPUs
- VRAM usage can be adjusted via `gpu_memory_utilization` in stage config

## Performance Benchmark

### vLLM-Omni vs Official Step-Audio2

Single request latency comparison between vLLM-Omni and official Step-Audio2 implementation.

| Task | Tokens | vllm-omni | Step-Audio2 | Speedup |
|------|--------|-----------|-------------|---------|
| S2ST | ~85 | 5.45s | 7.36s | **1.35x** |
| S2ST | ~160 | 6.67s | 13.92s | **2.09x** |
| S2ST | ~315 | 9.42s | 31.50s | **3.34x** |
| TTS | ~1024 | 16.65s | ~87s | **~5.2x** |

**Key observations:**
- Speedup increases with sequence length due to vLLM's efficient KV cache management
- TTS (pure generation) shows the largest speedup (~5x)
- S2ST benefits from optimized multi-stage pipeline

**Benchmark environment:**
- GPU: NVIDIA H100 80GB (single card)
- Model: Step-Audio2-mini
- Warmup: 1 run, Measured: 3 runs (averaged)

### Async Chunk Streaming Performance

Comparison between sequential (non-async) and async chunk modes via `/v1/audio/speech` TTS endpoint.

| Mode | Mean TTFP | Mean E2E | Mean RTF | Audio Throughput |
|------|-----------|----------|----------|-----------------|
| Sequential | 4316ms | 4316ms | 0.938 | 1.07x realtime |
| **Async Chunk** | **1437ms** | 4362ms | 0.949 | 1.06x realtime |
| **Improvement** | **-67% (3x faster)** | ~same | ~same | ~same |

**Key observations:**
- Async chunk reduces **time-to-first-audio (TTFP) by 67%** (4.3s → 1.4s)
- E2E latency remains comparable — async chunk overlaps Thinker decode with Token2Wav synthesis
- RTF < 1 in both modes (real-time capable)
- Sequential mode: TTFP ≈ E2E (must wait for all audio tokens before synthesis starts)
- Async chunk mode: Token2Wav starts after first 28 tokens (chunk_size=25 + lookahead=3)

**Benchmark environment:**
- GPU: 4x NVIDIA RTX 3090 24GB (TP=2 for Thinker, 1 GPU for Token2Wav)
- Model: Step-Audio2-mini
- Endpoint: `/v1/audio/speech` (10 prompts, concurrency=1)
- Measured via `bench_tts_serve.py`

## Installation

Make sure you have installed vLLM-Omni and all required dependencies:

```bash
# Install vLLM-Omni
pip install vllm-omni

# Install Step-Audio2 (REQUIRED for Token2Wav stage)
pip install step-audio2
```

## Model Setup

### Option 1: Auto-download from HuggingFace (Recommended)

The script will **automatically download** the model on first run:

```bash
# Just run without specifying --model, it will auto-download stepfun-ai/Step-Audio2-mini
python end2end.py --query-type audio_to_text

# Or explicitly specify the HuggingFace model
python end2end.py --query-type audio_to_text --model stepfun-ai/Step-Audio2-mini
```

Models will be cached in `~/.cache/huggingface/hub/` for future use.

**Available models**:
- `stepfun-ai/Step-Audio2-mini` (smaller, faster)
- `stepfun-ai/Step-Audio2-7B` (larger, better quality)

### Option 2: Manual Download (for offline use)

Download and use locally:

```bash
# Download from HuggingFace
huggingface-cli download stepfun-ai/Step-Audio2-mini --local-dir ./models/Step-Audio2-mini

# Then use the local path
python end2end.py --query-type audio_to_text --model ./models/Step-Audio2-mini
```

Ensure the model directory contains:
```
Step-Audio2-mini/
├── config.json
├── model.safetensors (or pytorch_model.bin)
├── tokenizer.json
├── tokenizer_config.json
└── token2wav/                           # Token2Wav models (REQUIRED)
    ├── speech_tokenizer_v2_25hz.onnx   # Audio tokenizer
    ├── campplus.onnx                    # Speaker encoder
    ├── flow.yaml                        # Flow model config
    ├── flow.pt                          # Flow model weights
    └── hift.pt                          # HiFT vocoder weights
```

## Usage Examples

### 1. Audio to Text (ASR - Speech Recognition)

Transcribe audio to text:

```bash
# Quick start - Using default model and test audio
python end2end.py --query-type audio_to_text

# Using your own audio file (model will auto-download)
python end2end.py --query-type audio_to_text \
    --audio-path /path/to/input.wav

# With specific model
python end2end.py --query-type audio_to_text \
    --audio-path /path/to/input.wav \
    --model stepfun-ai/Step-Audio2-7B

# With custom question
python end2end.py --query-type audio_to_text \
    --audio-path input.wav \
    --question "What is the speaker saying?"
```

**Output**: Text transcription saved to `output_step_audio2/00000_text.txt`

### 2. Text to Audio (TTS - Speech Synthesis)

Convert text to speech:

```bash
# Basic TTS (model auto-downloads)
python end2end.py --query-type text_to_audio \
    --text "Hello, this is a test of Step Audio 2 synthesis."

# With specific model
python end2end.py --query-type text_to_audio \
    --text "Hello, this is a test." \
    --model stepfun-ai/Step-Audio2-7B
```

**Note**: Speaker voice is controlled by the `STEP_AUDIO2_DEFAULT_PROMPT_WAV` environment variable or the default prompt wav bundled with the model.

**Output**:
- Text: `output_step_audio2/00000_text.txt`
- Audio: `output_step_audio2/00000_output.wav` (24kHz)

### 3. Audio to Audio (Voice Conversion)

Process input audio and generate output audio:

```bash
# Basic voice conversion (model auto-downloads)
python end2end.py --query-type audio_to_audio \
    --audio-path /path/to/source_audio.wav

# With specific model
python end2end.py --query-type audio_to_audio \
    --audio-path source.wav \
    --model stepfun-ai/Step-Audio2-7B
```

This mode:
1. Understands the content in `--audio-path` (source)
2. Generates audio output with the default voice

**Note**: To use a custom speaker voice, set the `STEP_AUDIO2_DEFAULT_PROMPT_WAV` environment variable.

### Advanced Options

```bash
# Use custom stage configuration
python end2end.py --query-type audio_to_text \
    --stage-configs-path /path/to/custom_config.yaml

# Multiple prompts (for batch testing)
python end2end.py --query-type audio_to_text \
    --audio-path input.wav \
    --num-prompts 5

# Custom output directory
python end2end.py --query-type text_to_audio \
    --text "Test synthesis" \
    --output-dir ./my_outputs

# Enable detailed logging
python end2end.py --query-type audio_to_text \
    --audio-path input.wav \
    --enable-stats

# Adjust generation parameters
python end2end.py --query-type audio_to_text \
    --audio-path input.wav \
    --max-tokens 2048

# Use custom speaker voice via environment variable
STEP_AUDIO2_DEFAULT_PROMPT_WAV=/path/to/speaker.wav python end2end.py \
    --query-type text_to_audio \
    --text "Hello world"

# Use Ray backend for distributed processing
python end2end.py --query-type text_to_audio \
    --text "Hello world" \
    --worker-backend ray \
    --ray-address "auto"
```

## Configuration

### Stage Configuration

The default configuration (`step_audio_2.yaml`) uses:

- **Stage 0 (Thinker)**: GPU 0, 80% memory
- **Stage 1 (Token2Wav)**: GPU 1, 30% memory

For **single GPU** setup, edit the config to use `devices: "0"` for both stages.

### Sampling Parameters

- **Thinker (Stage 0)**:
  - Temperature: 0.7 (balanced creativity)
  - Top-p: 0.9
  - Max tokens: 1024 (configurable)

- **Token2Wav (Stage 1)**:
  - Temperature: 0.0 (deterministic)
  - Operates in generation mode (not sampling)

## Common Issues

### 1. ImportError: No module named 's3tokenizer'

**Solution**: Install Step-Audio2 package:
```bash
pip install step-audio2
```

### 2. FileNotFoundError: prompt_wav file not found

**Solution**: Set the `STEP_AUDIO2_DEFAULT_PROMPT_WAV` environment variable to a valid audio file:
```bash
export STEP_AUDIO2_DEFAULT_PROMPT_WAV=/path/to/speaker.wav
python end2end.py --query-type text_to_audio --text "Hello"
```
Or ensure the default prompt wav (`default_female.wav`) exists in your model directory.

### 3. FileNotFoundError: token2wav models not found

**Solution**: Ensure your model directory has the complete `token2wav/` subdirectory with all ONNX and PyTorch models.

### 4. CUDA Out of Memory

**Solutions**:
- Use single GPU mode (set both stages to `devices: "0"`)
- Reduce `gpu_memory_utilization` in config
- Reduce `max_num_batched_tokens`
- Process fewer prompts at once

### 5. Model not found in registry

**Solution**: Ensure you're using vLLM-Omni's entry point with `--omni` flag or install vllm-omni properly:
```bash
pip install vllm-omni
```

## Output Files

The script generates files in the output directory (default: `output_step_audio2/`):

```
output_step_audio2/
├── 00000_text.txt        # Text output from Thinker stage
├── 00000_output.wav      # Audio output from Token2Wav stage (24kHz)
├── 00001_text.txt        # (if multiple prompts)
└── 00001_output.wav
```

## Performance Tips

1. **First run is slow**: Stage initialization takes 20-60 seconds
2. **Single GPU**: Set both stages to `devices: "0"` in config
3. **Multiple prompts**: Use `--num-prompts N` for batch testing
4. **Ray backend**: For multi-node or advanced scheduling
5. **Logging**: Use `--enable-stats` to debug performance issues

## Speaker Voice Configuration

The Token2Wav stage requires a speaker prompt wav for voice conditioning. It is automatically resolved in this order:

1. `STEP_AUDIO2_DEFAULT_PROMPT_WAV` environment variable (if set)
2. `{model_dir}/assets/default_female.wav`
3. `{model_dir}/default_female.wav`

If none are found, set the environment variable explicitly:
```bash
export STEP_AUDIO2_DEFAULT_PROMPT_WAV=/path/to/speaker.wav
```

**Guidelines for custom speaker prompt:**

- **Duration**: 3-10 seconds recommended
- **Quality**: Clean audio, minimal background noise
- **Format**: WAV, MP3, FLAC (will be resampled internally)
- **Content**: Clear speech, representative of target voice

## Example Workflow

Complete example from audio to final output:

```bash
# 1. ASR: Transcribe audio
python end2end.py --query-type audio_to_text \
    --audio-path interview.wav \
    --model ./models/Step-Audio2-7B \
    --output-dir ./outputs

# 2. Check the transcription
cat ./outputs/00000_text.txt

# 3. TTS: Synthesize new speech (with custom voice)
STEP_AUDIO2_DEFAULT_PROMPT_WAV=./speaker_samples/female_voice.wav \
python end2end.py --query-type text_to_audio \
    --text "The quick brown fox jumps over the lazy dog" \
    --model ./models/Step-Audio2-7B \
    --output-dir ./outputs

# 4. Listen to the result
# Audio saved to: ./outputs/00000_output.wav
```

## References

- [Step-Audio2 Paper](https://arxiv.org/abs/2507.16632)
- [vLLM-Omni Documentation](https://vllm-omni.readthedocs.io)
- [Model on HuggingFace](https://huggingface.co/stepfun-ai/Step-Audio-2-mini)
