# Z-Image Offline Inference

Z-Image support was added in PR [#149](https://github.com/vllm-project/vllm-omni/pull/149). The model ID is `Tongyi-MAI/Z-Image-Turbo`.

## Quickstart (Python)

```python
import torch
from vllm_omni import Omni
from vllm_omni.utils.platform_utils import detect_device_type

device = detect_device_type()
generator = torch.Generator(device=device).manual_seed(42)

omni = Omni(model="Tongyi-MAI/Z-Image-Turbo")
images = omni.generate(
    "a photo of a cat sitting on a laptop keyboard",
    height=1024,
    width=1024,
    num_inference_steps=9,
    guidance_scale=5.0,  # important for good results
    generator=generator,
)
images[0].save("z_image_output.png")
```

Notes:
- Use `guidance_scale` (around `5.0`) for best quality; `num_inference_steps=9` was used in the PR test.
- Keep `height/width` multiples of 16 (VAE downsample factor); 1024x1024 works well.
- `generator` is optional but keeps outputs reproducible.
