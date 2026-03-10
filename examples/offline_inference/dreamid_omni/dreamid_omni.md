# DreamID-Omni Offline Inference

This example runs DreamID-Omni locally and saves an MP4 with optional audio muxing.


## 1. Checkpoint Layout

Set `--ckpt-dir` to the root containing:

```
<CKPT_DIR>/
  Wan2.2-TI2V-5B/
    Wan2.2_VAE.pth
    models_t5_umt5-xxl-enc-bf16.pth
    google/umt5-xxl/
  MMAudio/
    ext_weights/
      v1-16.pth
      best_netG.pt
  DreamID_Omni/
    model.safetensors
    model_960x960.safetensors
    model_960x960_10s.safetensors
    dreamid_omni_old_twoip.safetensors
```

## 2. Run

```bash
cd examples/offline_inference/dreamid_omni
python dreamid_omni.py \
  --ckpt-dir <CKPT_DIR> \
  --prompt "A person walking in a park with birds singing" \
  --image-path ./ref0.png \
  --audio-path ./ref0.wav \
  --height 720 --width 720 \
  --num-inference-steps 45 \
  --seed 103 \
  --disable-dummy-run \
  --output dreamid_output.mp4
```

Notes:
- `--audio-path` is required by DreamID.
- You can pass up to two `--image-path` and `--audio-path`.
```
