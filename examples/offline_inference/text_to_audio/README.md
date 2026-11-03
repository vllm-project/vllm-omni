# Text-To-Audio

A unified script for text-to-audio generation. Supported models:

| Model | Tasks | Notes |
| ------- | ------- | ------- |
| `stabilityai/stable-audio-open-1.0` | text-to-audio | gated; uses `--audio-length` |
| `Lightricks/LTX-2` | text-to-audio | use `--model-class-name LTX2TextToAudioPipeline`; defaults to 40 steps |
| `diffusers/LTX-2.3-Diffusers` | text-to-audio | use `--model-class-name LTX2TextToAudioPipeline`; defaults to 30 steps |
| `Lightricks/LTX-2.5-Diffusers` | text-to-audio | use `--model-class-name LTX2TextToAudioPipeline`; defaults to 30 steps |

The `stabilityai/stable-audio-open-1.0` pipeline generates audio from text prompts.

## Prerequisites

If you use a gated model (e.g., `stabilityai/stable-audio-open-1.0`), ensure you have access:

1. **Accept Model License**: Visit the model page on Hugging Face (e.g., [stabilityai/stable-audio-open-1.0]) and accept the user agreement.
2. **Authenticate**: Log in to Hugging Face locally to access the gated model.
   ```bash
   huggingface-cli login
   ```

## Local CLI Usage

For LTX-2 or LTX-2.3:

```bash
python text_to_audio.py \
  --model Lightricks/LTX-2 \
  --model-class-name LTX2TextToAudioPipeline \
  --prompt "A fingerpicked acoustic guitar in a quiet studio" \
  --audio-length 5 \
  --num-inference-steps 40 \
  --output ltx2_audio.wav

python text_to_audio.py \
  --model diffusers/LTX-2.3-Diffusers \
  --model-class-name LTX2TextToAudioPipeline \
  --prompt "A fingerpicked acoustic guitar in a quiet studio" \
  --audio-length 5 \
  --num-inference-steps 30 \
  --output ltx23_audio.wav
```

The LTX output sample rate is taken from the checkpoint vocoder. LTX
text-to-audio currently runs with tensor parallel size 1, sequence parallel
size 1, and no Cache-DiT backend.

For Stable Audio Open:

```bash
python text_to_audio.py \
  --model stabilityai/stable-audio-open-1.0 \
  --prompt "The sound of a hammer hitting a wooden surface" \
  --negative-prompt "Low quality" \
  --seed 42 \
  --guidance-scale 7.0 \
  --audio-length 10.0 \
  --num-inference-steps 100 \
  --cache-backend tea_cache \
  --output stable_audio_output.wav
```

To reduce per-GPU memory for multi-GPU inference, launch with HSDP:

```bash
python text_to_audio.py \
  --model stabilityai/stable-audio-open-1.0 \
  --prompt "The sound of a hammer hitting a wooden surface" \
  --negative-prompt "Low quality" \
  --seed 42 \
  --guidance-scale 7.0 \
  --audio-length 10.0 \
  --num-inference-steps 100 \
  --use-hsdp \
  --hsdp-shard-size 2 \
  --output stable_audio_output.wav
```

Key arguments:

- `--prompt`: text description (string).
- `--audio-start`: audio start offset in seconds (→ `audio_start_in_s`).
- `--audio-length`: audio duration in seconds (audio length for Stable Audio).
- `--extra-body`: JSON dict of model-specific knobs, merged into sampling `extra_args`.
- `--negative-prompt`: negative prompt for classifier-free guidance.
- `--seed`: integer seed for deterministic generation.
- `--guidance-scale`: classifier-free guidance scale.
- `--num-inference-steps`: diffusion sampling steps.(more steps = higher quality, slower).
- `--use-hsdp`: enable HSDP weight sharding for the Stable Audio DiT.
- `--hsdp-shard-size`: number of GPUs used for HSDP sharding.
- `--hsdp-replicate-size`: number of HSDP replica groups.
- `--cache-backend`: cache acceleration backend. Stable Audio currently supports `tea_cache`.
- `--output`: path to save the generated WAV file.
