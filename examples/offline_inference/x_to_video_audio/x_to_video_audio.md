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
```
python x_to_video_audio.py \
  --model /xx/dreamid_omni \
  --prompt "Two people walking together and singing happily" \
  --image-path ./example0.png ./example1.png \
  --audio-path ./example0.wav ./example1.wav \
  --video-negative-prompt "jitter, bad hands, blur, distortion" \
  --audio-negative-prompt "robotic, muffled, echo, distorted" \
  --cfg-parallel-size 2 \
  --num-inference-steps 45 \
  --height 704 \
  --width 1280 \
  --output dreamid_omni.mp4
```
In the current test scenario (2 images + 2 audio inputs), the VRAM requirement is 72GB, regardless of whether cfg-parallel is enabled or disabled.
The VRAM usage can be reduced by enabling CPU offload via --enable-cpu-offload.

Key arguments:
- `--prompt`: text description (string).
- `--model`: path to the model local directory.
- `--height/--width`: output resolution (defaults 704 * 1024).
- `--image-path`: path to the input image list.
- `--audio-path`: path to the input audio list, indicate the timbre of the output video.
- `--cfg-parallel-size`: number of parallel cfg parallel (defaults 1).
- `--num-inference-steps`: number of denoising steps (defaults 45).
- `--video-negative-prompt`: negative prompt for video generation.
- `--audio-negative-prompt`: negative prompt for audio generation.
- `--enable-cpu-offload`: enable CPU offload (defaults False).
