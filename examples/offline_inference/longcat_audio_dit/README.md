# LongCat-AudioDiT

The `meituan-longcat/LongCat-AudioDiT-1B` pipeline generates audio from text prompts.

## Local CLI Usage

```bash
python end2end.py \
  --model meituan-longcat/LongCat-AudioDiT-1B \
  --prompt "A calm ocean wave ambience with soft wind in the background" \
  --negative-prompt "distorted, clipping, noisy" \
  --seed 42 \
  --guidance-scale 4.0 \
  --audio-length 5.0 \
  --num-inference-steps 16 \
  --sample-rate 24000 \
  --output longcat_audio_dit_output.wav
```

Key arguments:

- `--model`: model name or local path.
- `--prompt`: text description.
- `--negative-prompt`: negative prompt for classifier-free guidance.
- `--seed`: integer seed for deterministic generation.
- `--guidance-scale`: classifier-free guidance scale.
- `--audio-length`: audio duration in seconds.
- `--num-inference-steps`: diffusion sampling steps.
- `--sample-rate`: output WAV sample rate. LongCat-AudioDiT uses 24000 Hz.
- `--output`: path to save the generated WAV file.
