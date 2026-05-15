# Helios Video Generation via Online API

This example demonstrates how to use the `/v1/videos` API with Helios models using the generic `extra_params` field.

## Overview

The `/v1/videos` API now supports model-specific parameters through the `extra_params` field, which accepts a JSON object containing any model-specific configuration. This allows supporting new models like Helios without modifying the API.

## Helios Model Variants

- **Helios-Base**: Basic T2V/I2V/V2V generation (Stage 1 only)
- **Helios-Mid**: Advanced generation with CFG-Zero* (Stage 2)
- **Helios-Distilled**: Fast generation with DMD (Stage 2)

## API Usage Examples

### 1. T2V (Text-to-Video) - Helios-Base

Basic text-to-video generation:

```bash
curl -X POST http://localhost:8000/v1/videos \
  -F "prompt=A serene lakeside sunrise with mist over the water." \
  -F "model=BestWishYsh/Helios-Base" \
  -F "width=640" \
  -F "height=384" \
  -F "num_frames=99" \
  -F "num_inference_steps=50" \
  -F "guidance_scale=5.0" \
  -F "seed=42"
```

### 2. I2V (Image-to-Video) - Helios-Base

Generate video from an input image:

```bash
curl -X POST http://localhost:8000/v1/videos \
  -F "prompt=The lake water gently ripples as morning mist rises." \
  -F "model=BestWishYsh/Helios-Base" \
  -F "input_reference=@/path/to/image.jpg" \
  -F "width=640" \
  -F "height=384" \
  -F "num_frames=99" \
  -F "guidance_scale=5.0"
```

### 3. Helios-Mid with Stage 2 + CFG-Zero*

Advanced generation with pyramid multi-stage denoising:

```bash
curl -X POST http://localhost:8000/v1/videos \
  -F "prompt=A serene lakeside sunrise with mist over the water." \
  -F "model=BestWishYsh/Helios-Mid" \
  -F "width=640" \
  -F "height=384" \
  -F "guidance_scale=5.0" \
  -F 'extra_params={
    "is_enable_stage2": true,
    "pyramid_num_stages": 3,
    "pyramid_num_inference_steps_list": [20, 20, 20],
    "use_cfg_zero_star": true,
    "use_zero_init": true,
    "zero_steps": 1
  }'
```

### 4. Helios-Distilled with DMD

Fast generation with Distribution Matching Distillation:

```bash
curl -X POST http://localhost:8000/v1/videos \
  -F "prompt=A serene lakeside sunrise with mist over the water." \
  -F "model=BestWishYsh/Helios-Distilled" \
  -F "width=640" \
  -F "height=384" \
  -F "guidance_scale=1.0" \
  -F 'extra_params={
    "is_enable_stage2": true,
    "pyramid_num_stages": 3,
    "pyramid_num_inference_steps_list": [2, 2, 2],
    "is_amplify_first_chunk": true
  }'
```

## Model-Specific Parameters

The `extra_params` field accepts a JSON object with model-specific parameters. For Helios models, supported parameters include:

### Stage 2 Parameters
- `is_enable_stage2` (bool): Enable pyramid multi-stage denoising
- `pyramid_num_stages` (int): Number of pyramid stages (default: 3)
- `pyramid_num_inference_steps_list` (array): Steps per stage, e.g., `[20, 20, 20]`

### CFG Zero Star Parameters (Helios-Mid)
- `use_cfg_zero_star` (bool): Enable CFG Zero Star guidance
- `use_zero_init` (bool): Use zero initialization for first steps
- `zero_steps` (int): Number of initial zero prediction steps (default: 1)

### DMD Parameters (Helios-Distilled)
- `is_amplify_first_chunk` (bool): Enable DMD amplification for first chunk

### Video Input (V2V mode)
For video-to-video generation, upload a video file via `input_reference`:

```bash
curl -X POST http://localhost:8000/v1/videos \
  -F "prompt=Transform the video into a watercolor painting style." \
  -F "model=BestWishYsh/Helios-Base" \
  -F "input_reference=@/path/to/video.mp4" \
  -F "guidance_scale=5.0"
```

## Python Example

```python
import requests
import json

url = "http://localhost:8000/v1/videos"

# Helios-Mid with Stage 2
data = {
    "prompt": "A serene lakeside sunrise with mist over the water.",
    "model": "BestWishYsh/Helios-Mid",
    "width": 640,
    "height": 384,
    "guidance_scale": 5.0,
    "extra_params": json.dumps({
        "is_enable_stage2": True,
        "pyramid_num_stages": 3,
        "pyramid_num_inference_steps_list": [20, 20, 20],
        "use_cfg_zero_star": True,
        "use_zero_init": True,
        "zero_steps": 1
    })
}

response = requests.post(url, data=data)
video_job = response.json()
print(f"Video job created: {video_job['id']}")

# Poll for completion
import time
while True:
    status_response = requests.get(f"{url}/{video_job['id']}")
    status = status_response.json()

    if status['status'] == 'completed':
        print(f"Video generated: {status['file_name']}")
        break
    elif status['status'] == 'failed':
        print(f"Generation failed: {status.get('error')}")
        break

    print(f"Progress: {status['progress']}%")
    time.sleep(2)
```

## Benefits of This Approach

1. **No API Changes**: New models can be supported without modifying the API
2. **Backward Compatible**: Existing clients continue to work
3. **Flexible**: Any model-specific parameter can be passed
4. **Type-Safe**: Parameters are validated at the model level
5. **Future-Proof**: Supports models that don't exist yet

## Notes

- The `extra_params` field must be a valid JSON object
- Parameters are passed directly to the model's `extra_args`
- Invalid parameters will be caught by the model implementation
- Tensor inputs (images/videos) should use `input_reference`, not `extra_params`
