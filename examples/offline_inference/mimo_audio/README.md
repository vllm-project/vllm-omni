# MiMo-Audio Offline Inference

This directory contains an offline demo for running MiMo-Audio models with vLLM Omni. It builds task-specific inputs and generates WAV files or text outputs locally.

## Model Overview

MiMo-Audio provides multiple task variants for audio understanding and generation:

- **tts_sft**: Basic text-to-speech generation from text input.
- **tts_sft_with_instruct**: TTS generation with explicit voice style instructions.
- **tts_sft_with_audio**: TTS generation with audio reference for voice cloning.
- **tts_sft_with_natural_instruction**: TTS generation from natural language descriptions embedded in text.
- **audio_trancribing_sft**: Transcribe audio to text (speech-to-text).
- **audio_understanding_sft**: Understand and analyze audio content with text queries.
- **audio_understanding_sft_with_thinking**: Audio understanding with reasoning chain.
- **spoken_dialogue_sft_multiturn**: Multi-turn spoken dialogue with audio input/output.
- **speech2text_dialogue_sft_multiturn**: Multi-turn dialogue converting speech to text.
- **text_dialogue_sft_multiturn**: Multi-turn text-only dialogue.

## Setup

Please refer to the [stage configuration documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/stage_configs/) to configure memory allocation appropriately for your hardware setup.

### Environment Variables

The `MIMO_AUDIO_TOKENIZER_PATH` environment variable is mandatory due to the specialized architecture:

```bash
export MIMO_AUDIO_TOKENIZER_PATH="XiaomiMiMo/MiMo-Audio-Tokenizer"
```

## Quick Start

Run a single sample for basic TTS:

```bash
python3 -u end2end.py \
  --stage-configs-path vllm_omni/model_executor/stage_configs/mimo_audio.yaml \
  --model-name XiaomiMiMo/MiMo-Audio-7B-Instruct \
  --query-type tts_sft
```

Run batch samples for basic TTS:

```bash
python3 -u end2end.py \
  --stage-configs-path vllm_omni/model_executor/stage_configs/mimo_audio.yaml \
  --model-name XiaomiMiMo/MiMo-Audio-7B-Instruct \
  --query-type tts_sft \
  --num-prompts {batch_size}
```

When enabling multi-batch processing, if the total number of tokens passed to the next stage exceeds the `max_model_len` value in the `mimo_audio.yaml` configuration file, you must also synchronously update the `max_position_embeddings` value in `MiMo-Audio-7B-Instruct/config.json` to match the modified value.

Generated audio files are saved to `output_audio/` by default. `--num-prompts` also can be used to all tasks below.

## Task Usage

### tts_sft (Basic Text-to-Speech)

Generate speech from text input:

```bash
python3 -u end2end.py \
  --stage-configs-path vllm_omni/model_executor/stage_configs/mimo_audio.yaml \
  --model-name XiaomiMiMo/MiMo-Audio-7B-Instruct \
  --query-type tts_sft \
  --text "The weather is so nice today."
```

### tts_sft_with_instruct (TTS with Voice Instructions)

Generate speech with explicit voice style instructions:

```bash
python3 -u end2end.py \
  --stage-configs-path vllm_omni/model_executor/stage_configs/mimo_audio.yaml \
  --model-name XiaomiMiMo/MiMo-Audio-7B-Instruct \
  --query-type tts_sft_with_instruct \
  --text "The weather is so nice today." \
  --instruct "Speak happily in a child's voice"
```

### tts_sft_with_audio (TTS with Audio Reference)

Generate speech using an audio reference for voice cloning:

```bash
python3 -u end2end.py \
  --stage-configs-path vllm_omni/model_executor/stage_configs/mimo_audio.yaml \
  --model-name XiaomiMiMo/MiMo-Audio-7B-Instruct \
  --query-type tts_sft_with_audio \
  --text "The weather is so nice today." \
  --audio-path "./spoken_dialogue_assistant_turn_1.wav"
```

### tts_sft_with_natural_instruction (Natural Language TTS)

Generate speech from text containing natural voice descriptions:

```bash
python3 -u end2end.py \
  --stage-configs-path vllm_omni/model_executor/stage_configs/mimo_audio.yaml \
  --model-name XiaomiMiMo/MiMo-Audio-7B-Instruct \
  --query-type tts_sft_with_natural_instruction \
  --text "In a panting young male voice, he said: I can't run anymore, wait for me!"
```

### audio_trancribing_sft (Speech-to-Text)

Transcribe audio to text:

```bash
python3 -u end2end.py \
  --stage-configs-path vllm_omni/model_executor/stage_configs/mimo_audio.yaml \
  --model-name XiaomiMiMo/MiMo-Audio-7B-Instruct \
  --query-type audio_trancribing_sft \
  --audio-path "./spoken_dialogue_assistant_turn_1.wav"
```

### audio_understanding_sft (Audio Understanding)

Understand and analyze audio content with text queries:

```bash
python3 -u end2end.py \
  --stage-configs-path vllm_omni/model_executor/stage_configs/mimo_audio.yaml \
  --model-name XiaomiMiMo/MiMo-Audio-7B-Instruct \
  --query-type audio_understanding_sft \
  --text "Summarize the audio." \
  --audio-path "./spoken_dialogue_assistant_turn_1.wav"
```

### audio_understanding_sft_with_thinking (Audio Understanding with Reasoning)

Audio understanding with reasoning chain:

```bash
python3 -u end2end.py \
  --stage-configs-path vllm_omni/model_executor/stage_configs/mimo_audio.yaml \
  --model-name XiaomiMiMo/MiMo-Audio-7B-Instruct \
  --query-type audio_understanding_sft_with_thinking \
  --text "Summarize the audio." \
  --audio-path "./spoken_dialogue_assistant_turn_1.wav"
```

### spoken_dialogue_sft_multiturn (Multi-turn Spoken Dialogue)

Multi-turn dialogue with audio input and output:

```bash
python3 -u end2end.py \
  --stage-configs-path vllm_omni/model_executor/stage_configs/mimo_audio.yaml \
  --model-name XiaomiMiMo/MiMo-Audio-7B-Instruct \
  --query-type spoken_dialogue_sft_multiturn \
  --audio-path "./prompt_speech_zh_m.wav"
```

Note: This task uses hardcoded audio files in the script. The audio files used in examples are available at: https://github.com/XiaomiMiMo/MiMo-Audio/tree/main/examples

### speech2text_dialogue_sft_multiturn (Speech-to-Text Dialogue)

Multi-turn dialogue converting speech to text:

```bash
python3 -u end2end.py \
  --stage-configs-path vllm_omni/model_executor/stage_configs/mimo_audio.yaml \
  --model-name XiaomiMiMo/MiMo-Audio-7B-Instruct \
  --query-type speech2text_dialogue_sft_multiturn
```

Note: This task uses hardcoded audio files and message lists in the script.

### text_dialogue_sft_multiturn (Text Dialogue)

Multi-turn text-only dialogue:

```bash
python3 -u end2end.py \
  --stage-configs-path vllm_omni/model_executor/stage_configs/mimo_audio.yaml \
  --model-name XiaomiMiMo/MiMo-Audio-7B-Instruct \
  --query-type text_dialogue_sft_multiturn
```

Note: This task uses hardcoded message lists in the script.

## Troubleshooting

### Audio dependencies (soundfile, librosa)

This example depends on **soundfile** (read/write WAV) and **librosa** (load audio including MP3). Install the project requirements first:

```bash
pip install -r requirements/common.txt
# or at least: pip install soundfile>=0.13.1 librosa>=0.11.0
```

- **`soundfile` / libsndfile not found**  
  `soundfile` uses the C library **libsndfile**. On Linux, install the system package before pip:
  - Debian/Ubuntu: `sudo apt-get install libsndfile1`
  - For development builds: `sudo apt-get install libsndfile1-dev`
  - Then: `pip install soundfile`

- **`librosa` fails to load MP3 or reports "No backend available"**  
  Loading MP3 (e.g. in `spoken_dialogue_sft_multiturn` with `.mp3` files) uses **ffmpeg** as the backend. Install ffmpeg:
  - Debian/Ubuntu: `sudo apt-get install ffmpeg`
  - macOS: `brew install ffmpeg`

- **`ImportError: No module named 'soundfile'` or `ModuleNotFoundError: ... librosa`**  
  Ensure you are in the same Python environment where vLLM Omni and the example dependencies are installed, and that `requirements/common.txt` (or the packages above) are installed.

### Tokenizer path

- **`MIMO_AUDIO_TOKENIZER_PATH` not set or model fails to find tokenizer**  
  Export the tokenizer path before running:
  ```bash
  export MIMO_AUDIO_TOKENIZER_PATH="XiaomiMiMo/MiMo-Audio-Tokenizer"
  ```
  See [Environment Variables](#environment-variables) in Setup.

### Other

- If the model or stage config fails to load, check [stage configuration documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/stage_configs/) for memory and GPU settings.
- For errors when reading/writing WAV (e.g. unsupported format), ensure input files are standard WAV/MP3 and that `soundfile` is linked to a working libsndfile (see above).

## Notes

- The script uses default model paths and audio files embedded in `end2end.py`. Update them if your local cache path differs.
- Use `--output-dir` to change the output folder (default: `./output_audio`).
- Use `--num-prompts` to generate multiple prompts in one run (default: 1).
- Audio files used in multi-turn dialogue examples are available at: https://github.com/XiaomiMiMo/MiMo-Audio/tree/main/examples
- The script supports various configuration options for initialization timeouts, batch timeouts, and shared memory thresholds. See `--help` for details.
