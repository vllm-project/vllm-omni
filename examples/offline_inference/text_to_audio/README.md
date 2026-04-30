# Text-To-Audio

The `stabilityai/stable-audio-open-1.0` pipeline generates audio from text prompts.

## Prerequisites

If you use a gated model (e.g., `stabilityai/stable-audio-open-1.0`), ensure you have access:

1. **Accept Model License**: Visit the model page on Hugging Face (e.g., [stabilityai/stable-audio-open-1.0]) and accept the user agreement.
2. **Authenticate**: Log in to Hugging Face locally to access the gated model.
   ```bash
   huggingface-cli login
   ```

## Local CLI Usage

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
- `--negative-prompt`: negative prompt for classifier-free guidance.
- `--seed`: integer seed for deterministic generation.
- `--guidance-scale`: classifier-free guidance scale.
- `--audio-length`: audio duration in seconds.
- `--num-inference-steps`: diffusion sampling steps.(more steps = higher quality, slower).
- `--use-hsdp`: enable HSDP weight sharding for the Stable Audio DiT.
- `--hsdp-shard-size`: number of GPUs used for HSDP sharding.
- `--hsdp-replicate-size`: number of HSDP replica groups.
- `--cache-backend`: cache acceleration backend. Stable Audio currently supports `tea_cache`.
- `--output`: path to save the generated WAV file.
