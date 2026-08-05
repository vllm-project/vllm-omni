# X-To-Video-Audio

The `DreamID-Omni` pipeline generates short videos from text, image and video.

## Local CLI Usage
### Download the Model locally
Since DreamID-Omni combine multiple models, and without any config, so we need to download them locally.

```bash
python download_dreamid_omni.py --output-dir ./dreamid_omni
```
After download, the model directory will look like this:

```
dreamid_omni/
├── DreamID-Omni/
│   ├── dreamid_omni.safetensors
├── MMAudio/
│   ├── ext_weights/
│   │   ├── best_netG.pt
│   │   ├── v1-16.pth
├── Wan2.2-TI2V-5B/
│   ├── google/*
│   ├── models_t5_umt5-xxl-enc-bf16.pth
│   ├── Wan2.2_VAE.pth
│
├── model_index.json
└── transformer/
    └── config.json   # create by download_dreamid_omni.py
```

### Run the Inference
```python
python x_to_video_audio.py \
  --model /path/to/dreamid_omni \
  --prompt "Two people walking together and singing happily" \
  --image-path ./example0.png ./example1.png \
  --audio-path ./example0.wav ./example1.wav \
  --video-negative-prompt "jitter, bad hands, blur, distortion" \
  --audio-negative-prompt "robotic, muffled, echo, distorted" \
  --cfg-parallel-size 2 \
  --num-inference-steps 45 \
  --height 704 \
  --width 1280 \
  --output out_dreamid_omni_twoip.mp4
```
In the current test scenario (2 images + 2 audio inputs), the VRAM requirement is 72GB, regardless of whether cfg-parallel is enabled or disabled.
The VRAM usage can be reduced by enabling CPU offload via --enable-cpu-offload.
For multi-GPU memory reduction on the fused DreamID-Omni transformer, you can also enable HSDP:

```python
python x_to_video_audio.py \
  --model /path/to/dreamid_omni \
  --prompt "Two people walking together and singing happily" \
  --image-path ./example0.png ./example1.png \
  --audio-path ./example0.wav ./example1.wav \
  --video-negative-prompt "jitter, bad hands, blur, distortion" \
  --audio-negative-prompt "robotic, muffled, echo, distorted" \
  --cfg-parallel-size 2 \
  --num-inference-steps 45 \
  --height 704 \
  --width 1280 \
  --use-hsdp \
  --hsdp-shard-size 2 \
  --output out_dreamid_omni_twoip.mp4
```


You could take reference images/audios from the test cases in the official repo: https://github.com/Guoxu1233/DreamID-Omni

For example, single IP ref resources can be found under https://github.com/Guoxu1233/DreamID-Omni/tree/main/test_case/oneip, you could download them correspondingly to your local and use them for testing.

```python
# Example usage for oneip, ref media from the official repo DreamID-Omni
python x_to_video_audio.py \
  --model /path/to/dreamid_omni \
  --prompt "<img1>: In the frame, a woman with black long hair is identified as <sub1>.\n**Overall Environment/Scene**: A lively open-kitchen café at night; stove flames flare, steam rises, and warm pendant lights swing slightly as staff move behind her. The shot is an upper-body close-up.\n**Main Characters/Subjects Appearance**: <sub1> is a young woman with thick dark wavy hair and a side part. She wears a fitted black top under a light apron, a thin gold chain necklace, and small stud earrings.\n**Main Characters/Subjects Actions**: <sub1> tastes the sauce with a spoon, then turns her face toward the camera while still holding the spoon, her expression shifting from focused to conflicted.\n<sub1> maintains eye contact, swallows as if choosing her words, and says, <S>I keep telling myself I’m fine,but some nights it feels like I’m just performing calm.<E>" \
  --image-path 9.png \
  --audio-path 9.wav \
  --video-negative-prompt "jitter, bad hands, blur, distortion" \
  --audio-negative-prompt "robotic, muffled, echo, distorted" \
  --cfg-parallel-size 2 \
  --num-inference-steps 45 \
  --height 704 \
  --width 1280 \
  --output out_dreamid_omni_oneip.mp4
```


## MagiHuman (`--model-type magi-human`)

MagiHuman is a text → video+audio model with a DiT MoE backbone and a ~9B-param
T5Gemma text encoder. A detailed text prompt is the only required input; an optional
image and/or audio file may be supplied for conditioning. Natively supports Tensor
Parallelism. For an 80GB node, `--tensor-parallel-size 4` is recommended to shard
the MoE weights and the text encoder.

> Install [MagiCompiler](https://github.com/SandAI-org/MagiCompiler) for correct
> attention-kernel behaviour (the pipeline otherwise falls back to stubs).

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

## DreamID-Omni arguments

Key arguments:
- `--prompt`: text description (string).
- `--model`: path to the model local directory.
- `--height/--width`: output resolution (defaults 704 * 1024).
- `--image-path`: path to the input image list.
- `--audio-path`: path to the input audio list, indicate the timbre of the output video.
- `--cfg-parallel-size`: number of parallel cfg parallel (defaults 1).
- `--use-hsdp`: enable HSDP weight sharding for DreamID-Omni fused blocks.
- `--hsdp-shard-size`: number of GPUs used for HSDP sharding.
- `--hsdp-replicate-size`: number of HSDP replica groups.
- `--num-inference-steps`: number of denoising steps (defaults 45).
- `--video-negative-prompt`: negative prompt for video generation.
- `--audio-negative-prompt`: negative prompt for audio generation.
- `--enable-cpu-offload`: enable CPU offload (defaults False).
- `--cache-backend`: enable `cache_dit` for acceleration.
- `--quantization`: online (dynamic) quantization method — `fp8` or `int8`.
  (VAEs, the T5 text encoder, norms and modulation stay bf16.)
