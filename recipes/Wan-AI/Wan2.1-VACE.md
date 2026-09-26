# Wan2.1 VACE Unified Video Generation

## Summary

- Vendor: Wan-AI
- Models: `Wan2.1-VACE-1.3B-diffusers`, `Wan2.1-VACE-14B-diffusers`
- Tasks: T2V, I2V, V2LF, FLF2V, inpainting, and R2V
- Mode: offline inference through the shared task examples

The shared examples construct VACE's pipeline-native conditioning data from the
provided media inputs. They do not require a mode parameter or a model-specific
example script.

## References

- [VACE project](https://github.com/ali-vilab/VACE)
- [Hugging Face 14B checkpoint](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B-diffusers)
- [ModelScope 14B checkpoint](https://modelscope.cn/models/Wan-AI/Wan2.1-VACE-14B-diffusers)
- [Shared text-to-video example](../../examples/offline_inference/text_to_video/text_to_video.md)
- [Shared image-to-video example](../../examples/offline_inference/image_to_video/README.md)
- [Migration RFC #4539](https://github.com/vllm-project/vllm-omni/issues/4539)

## Hardware Support

The 1.3B checkpoint was validated on an RTX 5090; the 81-frame T2V command below
used approximately 20 GiB of peak reserved GPU memory with VAE tiling. The 14B
checkpoint requires approximately 70 GiB of disk space and was validated on a
single 48 GiB L40S with VAE tiling and layerwise offload. Start with the smoke
configuration before attempting an 81-frame run.

## Setup

The public examples use the Hugging Face model ID by default:

```bash
MODEL="${MODEL:-Wan-AI/Wan2.1-VACE-14B-diffusers}"
ASSET_DIR="${ASSET_DIR:-$HOME/datasets/vace}"
OUT="${OUT:-$PWD/vace-results/14b}"
mkdir -p "$ASSET_DIR" "$OUT"
```

For environments where Hugging Face downloads are unavailable, download the
same checkpoint from ModelScope and point `MODEL` at the local directory:

```bash
MODEL="$HOME/models/Wan2.1-VACE-14B-diffusers"
modelscope download --model Wan-AI/Wan2.1-VACE-14B-diffusers --local_dir "$MODEL"
```

Download the public conditioning images and create a mask matching the output
size. White pixels are regenerated and black pixels are preserved.

```bash
wget -O "$ASSET_DIR/astronaut.jpg" \
  https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg
wget -O "$ASSET_DIR/vace_first_frame.png" \
  https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_first_frame.png
wget -O "$ASSET_DIR/vace_last_frame.png" \
  https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_last_frame.png

python - "$ASSET_DIR/vace_center_mask.png" <<'PY'
import sys
from PIL import Image

mask = Image.new("L", (832, 480), 0)
mask.paste(255, (336, 0, 496, 480))
mask.save(sys.argv[1])
PY

ASTRONAUT="$ASSET_DIR/astronaut.jpg"
FIRST="$ASSET_DIR/vace_first_frame.png"
LAST="$ASSET_DIR/vace_last_frame.png"
MASK="$ASSET_DIR/vace_center_mask.png"
```

## Commands

The commands below reproduce the original VACE examples with `81 frames / 30
steps`. Most tasks use `480x832`; FLF2V uses `512x512`. For a smoke test,
replace the frame and step counts with `5 frames / 2 steps`.

```bash
# T2V
python examples/offline_inference/text_to_video/text_to_video.py \
  --model "$MODEL" \
  --prompt "A sleek, humanoid robot stands in a vast warehouse filled with neatly stacked cardboard boxes on industrial shelves." \
  --seed 0 \
  --height 480 --width 832 --num-frames 81 --num-inference-steps 30 \
  --guidance-scale 5.0 --flow-shift 5.0 --vae-use-tiling --enable-layerwise-offload \
  --output "$OUT/t2v.mp4"

# I2V
python examples/offline_inference/image_to_video/image_to_video.py \
  --model "$MODEL" --image "$ASTRONAUT" \
  --prompt "An astronaut emerging from a cracked, otherworldly egg on the surface of the moon" \
  --seed 42 \
  --height 480 --width 832 --num-frames 81 --num-inference-steps 30 \
  --guidance-scale 5.0 --flow-shift 5.0 --vae-use-tiling --enable-layerwise-offload \
  --output "$OUT/i2v.mp4"

# V2LF
python examples/offline_inference/image_to_video/image_to_video.py \
  --model "$MODEL" --last-image "$ASTRONAUT" \
  --prompt "An astronaut emerging from a cracked, otherworldly egg on the surface of the moon" \
  --seed 42 \
  --height 480 --width 832 --num-frames 81 --num-inference-steps 30 \
  --guidance-scale 5.0 --flow-shift 5.0 --vae-use-tiling --enable-layerwise-offload \
  --output "$OUT/v2lf.mp4"

# FLF2V
python examples/offline_inference/image_to_video/image_to_video.py \
  --model "$MODEL" --image "$FIRST" --last-image "$LAST" \
  --prompt "CG animation style, a small blue bird takes off from a branch and lands on another branch" \
  --seed 42 --height 512 --width 512 --num-frames 81 --num-inference-steps 30 \
  --guidance-scale 5.0 --flow-shift 5.0 --vae-use-tiling --enable-layerwise-offload \
  --output "$OUT/flf2v.mp4"

# Inpaint
python examples/offline_inference/image_to_video/image_to_video.py \
  --model "$MODEL" --image "$ASTRONAUT" --mask-image "$MASK" \
  --prompt "Shrek, the ogre, walks out of a building in a happy mood" \
  --seed 42 \
  --height 480 --width 832 --num-frames 81 --num-inference-steps 30 \
  --guidance-scale 5.0 --flow-shift 5.0 --vae-use-tiling --enable-layerwise-offload \
  --output "$OUT/inpaint.mp4"

# R2V; repeat --reference-image to add references
python examples/offline_inference/image_to_video/image_to_video.py \
  --model "$MODEL" --reference-image "$ASTRONAUT" \
  --prompt "Camera slowly zooms out from the character walking in a garden" \
  --seed 42 \
  --height 480 --width 832 --num-frames 81 --num-inference-steps 30 \
  --guidance-scale 5.0 --flow-shift 5.0 --vae-use-tiling --enable-layerwise-offload \
  --output "$OUT/r2v.mp4"
```

## Verification

```bash
for video in "$OUT"/*.mp4; do
  ffprobe -v error -select_streams v:0 \
    -show_entries stream=codec_name,width,height,nb_frames,r_frame_rate,duration \
    -of default=noprint_wrappers=1 "$video"
done
```

For the 1.3B checkpoint, `--enable-layerwise-offload` is generally unnecessary.
