# SkyReels-V3 Offline Inference Examples

This directory contains examples for using the SkyReels-V3 multimodal video generation models with vLLM-Omni.

## Models

SkyReels-V3 is a family of multimodal video generation models that support:

- **Image-to-Video (R2V)**: Generate videos from reference images
- **Video-to-Video (V2V)**: Transform existing videos
- **Audio-to-Video (A2V)**: Generate videos guided by audio

### Available Models

- `Skywork/SkyReels-V3-R2V-14B`: Image-to-Video (14B parameters)
- `Skywork/SkyReels-V3-V2V-14B`: Video-to-Video (14B parameters)
- `Skywork/SkyReels-V3-A2V-19B`: Audio-to-Video (19B parameters)

## Installation

Install the required dependencies:

```bash
pip install vllm-omni
pip install imageio imageio-ffmpeg  # For video I/O
```

## Usage

### Image-to-Video (R2V)

Generate a video from a reference image:

```bash
python image_to_video.py \
    --model Skywork/SkyReels-V3-R2V-14B \
    --image path/to/your/image.jpg \
    --prompt "A person walking through a beautiful garden" \
    --height 480 \
    --width 832 \
    --num-frames 81 \
    --num-inference-steps 50 \
    --guidance-scale 7.5 \
    --seed 42 \
    --output-dir ./outputs/skyreels_v3 \
    --output-format mp4
```

### Parameters

- `--model`: Model name or path (default: `Skywork/SkyReels-V3-R2V-14B`)
- `--image`: Path to the reference image (required)
- `--prompt`: Text prompt describing the desired video
- `--negative-prompt`: Negative prompt to avoid certain content (optional)
- `--height`: Video height in pixels (default: 480)
- `--width`: Video width in pixels (default: 832)
- `--num-frames`: Number of frames to generate (default: 81)
- `--num-inference-steps`: Number of denoising steps (default: 50, higher = better quality but slower)
- `--guidance-scale`: Classifier-free guidance scale (default: 7.5, higher = more prompt adherence)
- `--seed`: Random seed for reproducibility (default: 42)
- `--output-dir`: Output directory for generated videos
- `--output-format`: Output format: `mp4`, `gif`, or `frames`

## Examples

### Basic Image-to-Video

```bash
python image_to_video.py \
    --image examples/sample_image.jpg \
    --prompt "A cinematic video of the scene"
```

### High-Quality Generation

```bash
python image_to_video.py \
    --image examples/sample_image.jpg \
    --prompt "A dramatic video with dynamic camera movement" \
    --num-inference-steps 100 \
    --guidance-scale 9.0 \
    --num-frames 121
```

### Generate GIF

```bash
python image_to_video.py \
    --image examples/sample_image.jpg \
    --prompt "A looping animation" \
    --output-format gif \
    --num-frames 49
```

## Tips

1. **Image Quality**: Use high-quality reference images for best results
2. **Aspect Ratio**: The model works best with 16:9 aspect ratio (e.g., 832x480)
3. **Frame Count**: More frames = longer videos but slower generation
4. **Guidance Scale**:
   - Lower (3-5): More creative, less adherence to prompt
   - Medium (7-9): Balanced
   - Higher (10+): Strong prompt adherence, may reduce quality
5. **Inference Steps**: 50 steps is usually sufficient; 100+ for highest quality

## Performance

- **GPU Memory**: ~24GB VRAM required for R2V-14B model
- **Generation Time**: ~2-5 minutes for 81 frames on A100 GPU
- **Batch Size**: Currently supports batch size of 1

## Troubleshooting

### Out of Memory

If you encounter OOM errors:
- Reduce `--num-frames`
- Reduce `--height` and `--width`
- Use a smaller model variant if available

### Poor Quality

If the output quality is poor:
- Increase `--num-inference-steps` (try 75-100)
- Adjust `--guidance-scale` (try 8-10)
- Use a higher quality reference image
- Refine your prompt to be more specific

## Citation

If you use SkyReels-V3 in your research, please cite:

```bibtex
@article{skyreels2025,
  title={SkyReels-V3: Multimodal Video Generation with Unified In-Context Learning},
  author={Skywork Team},
  journal={arXiv preprint},
  year={2025}
}
```

## License

SkyReels-V3 models are released under the Skywork License. Please refer to the model card on Hugging Face for details.
