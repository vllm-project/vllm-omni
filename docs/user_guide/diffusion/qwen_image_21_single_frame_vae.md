# Qwen-Image-2.1 single-frame VAE cache

The optional `qwen21_skip_single_frame_vae_cache` setting avoids materializing
temporal feature-cache copies during untiled, single-frame VAE decoding.
Those copies do not contribute to the decoded frame. The goal is lower
allocated memory; this option does not promise a latency improvement or an
equal decrease in allocator-reserved memory.

```python
from vllm_omni.entrypoints.omni import Omni

omni = Omni(
    model="Qwen/Qwen-Image-2.1",
    additional_config={"qwen21_skip_single_frame_vae_cache": True},
)
```

The setting defaults to `false`, requires a Boolean, and is passed to the
loaded VAE by the pipeline. It is independent of Transformer optimizations
and MindIE-SD. Tiled decoding keeps its existing cache path. This image VAE
does not gain video/multiframe support.

CPU tests verify exact single-frame output across different inputs and repeated
calls, unchanged tiled output, and unchanged multiframe rejection:

```bash
python -m pytest -q \
  tests/diffusion/models/qwen_image_21/test_single_frame_cache.py \
  tests/diffusion/models/qwen_image_21/test_vae.py
```

Full-model memory validation of the new configuration wiring, group offload,
distributed VAE tiling, and Turbo-checkpoint validation remain pending.
Set the option to `false` in a fresh engine to restore the original path.
