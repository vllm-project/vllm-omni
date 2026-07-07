# Text-To-Audio

A unified script for text/video-to-audio generation. Supported models:

| Model | Tasks | Notes |
|-------|-------|-------|
| `stabilityai/stable-audio-open-1.0` | text-to-audio | gated; uses `--audio-length` |
| `zhangj1an/AudioX` | `t2a` / `t2m` / `v2a` / `v2m` / `tv2a` / `tv2m` | pass `--task`; video tasks need `--video` |

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

### AudioX

AudioX supports six tasks. Sampler and reference knobs (declared in
`vllm_omni/model_extras/audiox.py`) are passed via the generic `--extra-body`
JSON flag, routed into sampling `extra_args`.

Text tasks (`t2a` / `t2m`):

```bash
python text_to_audio.py \
  --model zhangj1an/AudioX --task t2a \
  --prompt "Fireworks burst twice, followed by a clock ticking." \
  --num-inference-steps 250 --guidance-scale 6.0 --audio-length 10.0 --seed 42 \
  --extra-body '{"sigma_min": 0.03, "sigma_max": 1000.0}' \
  --output t2a.wav
```

Video-conditioned tasks (`v2a` / `v2m` / `tv2a` / `tv2m`) require `--video`:

```bash
python text_to_audio.py \
  --model zhangj1an/AudioX --task tv2a \
  --prompt "drum beating sound and human talking" \
  --video https://zeyuet.github.io/AudioX/static/samples/V2M/1XeBotOFqHA.mp4 \
  --num-inference-steps 250 --guidance-scale 6.0 --audio-length 10.0 \
  --output tv2a.wav
```

Key arguments:

- `--prompt`: text description (string).
- `--task`: [AudioX] one of `t2a`/`t2m`/`v2a`/`v2m`/`tv2a`/`tv2m`.
- `--video`: [AudioX `v2*`/`tv2*`] video file/URL for conditioning (→ `video_path`).
- `--audio-start`: audio start offset in seconds (→ `audio_start_in_s` for Stable Audio, `seconds_start` for AudioX).
- `--audio-length`: audio duration in seconds (audio length for Stable Audio, `seconds_total` for AudioX).
- `--extra-body`: JSON dict of model-specific knobs (declared in `vllm_omni/model_extras/`), merged into sampling `extra_args`. For AudioX, sampler/reference knobs go here, e.g. `'{"sigma_min": 0.03, "sigma_max": 1000.0, "cfg_rescale": 0.0, "audio_path": "ref.wav"}'`.
- `--negative-prompt`: negative prompt for classifier-free guidance.
- `--seed`: integer seed for deterministic generation.
- `--guidance-scale`: classifier-free guidance scale.
- `--num-inference-steps`: diffusion sampling steps.(more steps = higher quality, slower).
- `--use-hsdp`: enable HSDP weight sharding for the Stable Audio DiT.
- `--hsdp-shard-size`: number of GPUs used for HSDP sharding.
- `--hsdp-replicate-size`: number of HSDP replica groups.
- `--cache-backend`: cache acceleration backend. Stable Audio currently supports `tea_cache`.
- `--output`: path to save the generated WAV file.
