# Voxtral TTS Offline Inference

`end2end.py` runs Voxtral TTS end-to-end offline inference using vLLM. It supports both blocking (`Omni`) and streaming (`AsyncOmni`) generation, batched prompts with configurable concurrency, and voice selection via preset name or reference audio file.

When `mistral_common` has `SpeechRequest` support, prompt token IDs are built via `encode_speech_request`. Otherwise, the script falls back to manual token construction.

## Usage Examples

```bash
# Basic single-prompt with reference audio
python3 examples/offline_inference/voxtral_tts/end2end.py \
    --write-audio \
    --model mistralai/tts-model \
    --text "That eerie silence after the first storm was just the calm before another round of chaos, wasn't it?" \
    --audio-path path/to/reference_audio.wav

# 32 replicate prompts with reference audio
python3 examples/offline_inference/voxtral_tts/end2end.py \
    --num-prompts 32 --write-audio \
    --model mistralai/tts-model \
    --text "That eerie silence after the first storm was just the calm before another round of chaos, wasn't it?" \
    --audio-path path/to/reference_audio.wav

# Short debug prompt with reference audio
python3 examples/offline_inference/voxtral_tts/end2end.py \
    --write-audio \
    --model mistralai/tts-model \
    --text "This is a test message." \
    --audio-path path/to/reference_audio.wav

# Streaming with neutral_female voice preset
python3 examples/offline_inference/voxtral_tts/end2end.py \
    --streaming --write-audio --voice neutral_female \
    --model mistralai/tts-model

# 32 prompts, 8 concurrent requests per wave, streaming with casual_male voice
python3 examples/offline_inference/voxtral_tts/end2end.py \
    --num-prompts 32 --concurrency 8 --streaming --write-audio --voice casual_male \
    --model mistralai/tts-model
```

## Arguments

| Argument | Description |
|---|---|
| `--model PATH` | HuggingFace repo ID or local directory path (default: `mistralai/tts-model`) |
| `--text TEXT` | Text to synthesize (default: `"This is a test message."`) |
| `--audio-path PATH` | Path to reference audio file for voice cloning |
| `--output-dir DIR` | Directory to write output WAV files (default: `output_audio`) |
| `--stage-configs-path PATH` | Path to stage configs YAML (auto-resolved from model if not set) |
| `--num-prompts N` | Number of replicate prompts to run for measuring performance (default: 1) |
| `--streaming` | Use streaming generation via `AsyncOmni` (default: blocking `Omni`) |
| `--concurrency N` | Max concurrent requests per wave (must be used with `--streaming`, must evenly divide `--num-prompts`) |
| `--voice NAME` | Voice preset to use instead of reference audio file (e.g., casual_female, casual_male, cheerful_female, neutral_female, neutral_male) |
| `--write-audio` | Write generated audio to WAV files |
| `--profiling-mode` | Enable profiling mode (reduces max tokens to 50) |
| `--log-stats` | Enable detailed statistics logging |
