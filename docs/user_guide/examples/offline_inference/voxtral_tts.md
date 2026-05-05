# Voxtral TTS Offline Inference

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/voxtral_tts>.


`end2end.py` runs Voxtral TTS end-to-end offline inference using vLLM. It supports both blocking (`Omni`) and streaming (`AsyncOmni`) generation, batched prompts with configurable concurrency, and voice selection via preset name or reference audio file.

When `mistral_common` has `SpeechRequest` support, prompt token IDs are built via `encode_speech_request`. Otherwise, the script falls back to manual token construction.

## Usage Examples


```bash
# Basic single-prompt with cheerful_female voice preset
python3 examples/offline_inference/voxtral_tts/end2end.py \
    --write-audio --voice cheerful_female \
    --model mistralai/Voxtral-4B-TTS-2603 \
    --text "That eerie silence after the first storm was just the calm before another round of chaos, wasn't it?"

# 32 replicate prompts with cheerful_female voice preset
python3 examples/offline_inference/voxtral_tts/end2end.py \
    --num-prompts 32 --write-audio --voice cheerful_female \
    --model mistralai/Voxtral-4B-TTS-2603 \
    --text "That eerie silence after the first storm was just the calm before another round of chaos, wasn't it?"

# Streaming with neutral_female voice preset
python3 examples/offline_inference/voxtral_tts/end2end.py \
    --streaming --write-audio --voice neutral_female \
    --model mistralai/Voxtral-4B-TTS-2603 \
    --text "That eerie silence after the first storm was just the calm before another round of chaos, wasn't it?"

# 32 prompts, 8 concurrent requests per wave, streaming with neutral_female voice
python3 examples/offline_inference/voxtral_tts/end2end.py \
    --num-prompts 32 --concurrency 8 --streaming --write-audio --voice neutral_female \
    --model mistralai/Voxtral-4B-TTS-2603 \
    --text "That eerie silence after the first storm was just the calm before another round of chaos, wasn't it?"

# Short debug prompt with reference audio
python3 examples/offline_inference/voxtral_tts/end2end.py \
    --write-audio \
    --model mistralai/Voxtral-4B-TTS-2603 \
    --text "This is a test message." \
    --audio-path path/to/reference_audio.wav
```

## Arguments

| Argument | Description |
|---|---|
| `--model PATH` | HuggingFace repo ID or local directory path (default: `mistralai/Voxtral-4B-TTS-2603`) |
| `--text TEXT` | Text to synthesize (default: `"This is a test message."`) |
| `--audio-path PATH` | Path to reference audio file for voice cloning |
| `--output-dir DIR` | Directory to write output WAV files (default: `output_audio`) |
| `--deploy-config PATH` | Override the deploy config path. If unset, auto-loads `vllm_omni/deploy/voxtral_tts.yaml` from the HF `model_type`. |
| `--num-prompts N` | Number of replicate prompts to run for measuring performance (default: 1) |
| `--streaming` | Use streaming generation via `AsyncOmni` (default: blocking `Omni`) |
| `--concurrency N` | Max concurrent requests per wave (must be used with `--streaming`, must evenly divide `--num-prompts`) |
| `--voice NAME` | Voice preset to use instead of reference audio file (e.g., casual_female, casual_male, cheerful_female, neutral_female, neutral_male) |
| `--write-audio` | Write generated audio to WAV files |
| `--profiling-mode` | Enable profiling mode (reduces max tokens to 50) |
| `--log-stats` | Enable detailed statistics logging |

## Example materials

??? abstract "end2end.py"
    ``````py
    --8<-- "examples/offline_inference/voxtral_tts/end2end.py"
    ``````
