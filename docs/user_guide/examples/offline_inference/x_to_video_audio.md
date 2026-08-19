# X-To-Video-Audio

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/x_to_video_audio>.

MagiHuman is a text → video+audio model with a DiT MoE backbone and a ~9B-param
T5Gemma text encoder. A detailed text prompt is the only required input; an optional
image and/or audio file may be supplied for conditioning. Natively supports Tensor
Parallelism. For an 80GB node, `--tensor-parallel-size 4` is recommended to shard
the MoE weights and the text encoder.

> Install [MagiCompiler](https://github.com/SandAI-org/MagiCompiler) for correct
> attention-kernel behaviour (the pipeline otherwise falls back to stubs).

## Local CLI Usage

Text-only generation:

```bash
python x_to_video_audio.py \
  --model-type magi-human \
  --model /path/to/daVinci-MagiHuman \
  --prompt "A young woman with long, wavy golden blonde hair... <dialogue and background sound>" \
  --tensor-parallel-size 4 \
  --height 256 --width 448 \
  --num-inference-steps 8 \
  --seed 52 \
  --extra-body '{"seconds": 5, "sr_height": 1080, "sr_width": 1920, "sr_num_inference_steps": 5}' \
  --output output_magihuman.mp4
```

With optional image and audio conditioning:

```bash
python x_to_video_audio.py \
  --model-type magi-human \
  --model /path/to/daVinci-MagiHuman \
  --prompt "A young woman..." \
  --tensor-parallel-size 4 \
  --height 256 --width 448 \
  --num-inference-steps 8 \
  --seed 52 \
  --extra-body '{"image_path": "/path/to/ref.jpg", "audio_path": "/path/to/ref.wav", "sr_height": 1080, "sr_width": 1920, "sr_num_inference_steps": 5}' \
  --output output_magihuman.mp4
```

MagiHuman-specific arguments are passed as a JSON dict via `--extra-body` (declared
in `vllm_omni/model_extras/magi_human.py`, routed via `extra_args`):

- `seconds`: output duration in seconds (default 10; ignored when `audio_path` is set,
  because the audio length then determines the number of frames).
- `audio_path`: path to an audio file for audio-to-video conditioning. When provided,
  the audio drives the frame count and is encoded as a latent condition into the DiT
  (`is_a2v` mode). Omit for pure text-to-video+audio generation.
- `image_path`: path to an image file for visual conditioning. Applied at both the
  base-resolution (BR) and super-resolution (SR) stages when SR is enabled.
- `sr_height` / `sr_width`: super-resolution output resolution. SR stage is skipped
  when these are omitted.
- `sr_num_inference_steps`: denoising steps for the SR stage.

## Example materials

??? abstract "x_to_video_audio.py"
    ``````py
    --8<-- "examples/offline_inference/x_to_video_audio/x_to_video_audio.py"
    ``````
