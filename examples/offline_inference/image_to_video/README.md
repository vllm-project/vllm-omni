# Image-To-Video

This shared example generates videos from images with VACE, Wan2.2, LTX2,
HunyuanVideo-1.5, Cosmos3, and other compatible pipelines.

## Local CLI Usage

Download the example image:

```bash
wget https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg
```

### Wan2.2-I2V-A14B-Diffusers (MoE)

```bash
python image_to_video.py \
  --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --image cherry_blossom.jpg \
  --prompt "Cherry blossoms swaying gently in the breeze, petals falling, smooth motion" \
  --negative-prompt "<optional quality filter>" \
  --height 480 \
  --width 832 \
  --num-frames 48 \
  --guidance-scale 5.0 \
  --guidance-scale-high 6.0 \
  --num-inference-steps 40 \
  --boundary-ratio 0.875 \
  --flow-shift 12.0 \
  --fps 16 \
  --output i2v_output.mp4
```

### Wan2.2-TI2V-5B-Diffusers (Unified)

```bash
python image_to_video.py \
  --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --image cherry_blossom.jpg \
  --prompt "Cherry blossoms swaying gently in the breeze, petals falling, smooth motion" \
  --negative-prompt "<optional quality filter>" \
  --height 480 \
  --width 832 \
  --num-frames 48 \
  --guidance-scale 4.0 \
  --num-inference-steps 40 \
  --flow-shift 12.0 \
  --fps 16 \
  --output i2v_output.mp4
```

### Cosmos3

```bash
# Cosmos3 bundles example frames under assets/ (any RGB image works too):
python image_to_video.py \
  --model nvidia/Cosmos3-Nano \
  --image /path/to/Cosmos3-Nano/assets/example_i2v_input.jpg \
  --prompt "The scene comes to life with smooth, natural motion." \
  --negative-prompt "blurry, distorted, low quality" \
  --height 720 --width 1280 --num-frames 189 --fps 24 \
  --num-inference-steps 35 --guidance-scale 6.0 \
  --extra-body '{"flow_shift": 10.0, "max_sequence_length": 4096, "guardrails": false}' \
  --output cosmos3_i2v.mp4
```

Key arguments:

- `--model`: Model ID (I2V-A14B for MoE, TI2V-5B for unified T2V+I2V).
- `--image`: Path to the first-frame or source image.
- `--last-image`: Optional last-frame condition for models such as VACE.
- `--mask-image`: Optional inpainting mask. White pixels are regenerated and black pixels are preserved.
- `--reference-image`: Optional reference image; repeat it to provide multiple references.
- `--extra-body`: JSON object of model-specific generation params, filtered against the model's declared `extra_body_params` (see [`vllm_omni/model_extras`](../../../vllm_omni/model_extras)). Used by Cosmos3.
- `--prompt`: Text description of desired motion/animation.
- `--height/--width`: Output resolution (auto-calculated from image if not set). Dimensions should be multiples of 16.
- `--num-frames`: Number of frames (default: model-specific — Wan 81, LTX2 121, Cosmos3 189).
- `--guidance-scale` and `--guidance-scale-high`: CFG scale (applied to low/high-noise stages for MoE).
- `--negative-prompt`: Optional list of artifacts to suppress.
- `--boundary-ratio`: Boundary split ratio for two-stage MoE models.
- `--flow-shift`: Scheduler flow shift (default: model-specific — Wan/LTX2 5.0, Cosmos3 10.0).
- `--sample-solver`: Wan2.2 sampling solver. Use `unipc` for the default multistep solver, or `euler` for Lightning/Distill checkpoints.
- `--num-inference-steps`: Number of denoising steps (default: model-specific — Wan 50, LTX2 40, Cosmos3 35).
- `--fps`: Frames per second for the saved MP4 (requires `diffusers` export_to_video).
- `--output`: Path to save the generated video.
- `--vae-use-slicing`: Enable VAE slicing for memory optimization.
- `--vae-use-tiling`: Enable VAE tiling for memory optimization.
- `--cfg-parallel-size`: set it to 2 to enable CFG Parallel. See more examples in [`user_guide`](https://github.com/vllm-project/vllm-omni/tree/main/docs/user_guide/diffusion/parallelism/cfg_parallel.md).
- `--tensor-parallel-size`: tensor parallel size (effective for models that support TP, e.g. LTX2).
- `--enable-cpu-offload`: enable CPU offloading for diffusion models.
- `--use-hsdp`: Enable Hybrid Sharded Data Parallel to shard model weights across GPUs.
- `--hsdp-shard-size`: Number of GPUs to shard model weights across within each replica group. -1 (default) auto-calculates as world_size / replicate_size.
- `--hsdp-replicate-size`: Number of replica groups for HSDP. Each replica holds a full sharded copy. Default 1 means pure sharding (no replication).



> ℹ️ If you encounter OOM errors, try using `--vae-use-slicing` and `--vae-use-tiling` to reduce memory usage.

## Wan2.1 VACE Conditional Tasks

The shared script selects the VACE conditioning structure from the media inputs.
No explicit mode parameter is required: the script constructs the source video,
mask, or reference images consumed by the VACE pipeline.
Download the same Hugging Face assets used by the original VACE example:

```bash
wget -O astronaut.jpg https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg
wget -O vace_first_frame.png https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_first_frame.png
wget -O vace_last_frame.png https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/flf2v_input_last_frame.png
```

### Image-to-Video (I2V)

```bash
python image_to_video.py \
  --model Wan-AI/Wan2.1-VACE-1.3B-diffusers \
  --image astronaut.jpg \
  --prompt "An astronaut emerging from a cracked, otherworldly egg on the surface of the moon" \
  --seed 42 --height 480 --width 832 --num-frames 81 \
  --num-inference-steps 30 --guidance-scale 5.0 --flow-shift 5.0 \
  --vae-use-tiling --output vace_i2v_output.mp4
```

### Video-to-Last-Frame (V2LF)

```bash
python image_to_video.py \
  --model Wan-AI/Wan2.1-VACE-1.3B-diffusers \
  --last-image astronaut.jpg \
  --prompt "An astronaut emerging from a cracked, otherworldly egg on the surface of the moon" \
  --seed 42 --height 480 --width 832 --num-frames 81 \
  --num-inference-steps 30 --guidance-scale 5.0 --flow-shift 5.0 \
  --vae-use-tiling --output vace_v2lf_output.mp4
```

### First-Last-Frame-to-Video (FLF2V)

```bash
python image_to_video.py \
  --model Wan-AI/Wan2.1-VACE-1.3B-diffusers \
  --image vace_first_frame.png \
  --last-image vace_last_frame.png \
  --prompt "CG animation style, a small blue bird takes off from a branch and lands on another branch" \
  --seed 42 --height 512 --width 512 --num-frames 81 \
  --num-inference-steps 30 --guidance-scale 5.0 --flow-shift 5.0 \
  --vae-use-tiling --output vace_flf2v_output.mp4
```

### Inpainting

Create a mask matching the original VACE example: a 160-pixel-wide white
vertical stripe marks the region to regenerate.

```bash
python - <<'PY'
from PIL import Image

mask = Image.new("L", (832, 480), 0)
mask.paste(255, (336, 0, 496, 480))
mask.save("vace_center_mask.png")
PY
```

```bash
python image_to_video.py \
  --model Wan-AI/Wan2.1-VACE-1.3B-diffusers \
  --image astronaut.jpg \
  --mask-image vace_center_mask.png \
  --prompt "Shrek, the ogre, walks out of a building in a happy mood" \
  --seed 42 --height 480 --width 832 --num-frames 81 \
  --num-inference-steps 30 --guidance-scale 5.0 --flow-shift 5.0 \
  --vae-use-tiling --output vace_inpaint_output.mp4
```

### Reference-to-Video (R2V)

Repeat `--reference-image` to provide more than one reference image.

```bash
python image_to_video.py \
  --model Wan-AI/Wan2.1-VACE-1.3B-diffusers \
  --reference-image astronaut.jpg \
  --prompt "Camera slowly zooms out from the character walking in a garden" \
  --seed 42 --height 480 --width 832 --num-frames 81 \
  --num-inference-steps 30 --guidance-scale 5.0 --flow-shift 5.0 \
  --vae-use-tiling --output vace_r2v_output.mp4
```

The VACE T2V command is documented in the shared
[`text_to_video.py`](../text_to_video/text_to_video.md#wan21-vace-t2v) example.

For Wan2.2 LightX2V-converted local Diffusers directories and related LoRA
assets, see the [LoRA guide](../../../docs/user_guide/diffusion/lora.md#wan22-lightx2v-offline-assembly).
