# Hy-Nomas model sources

The inference implementations are adapted from Tencent-Hunyuan/Hy-Nomas
at `911cea6458a89c2663513ff7099561ec190dd491` (Apache-2.0;
Copyright 2026 Tencent). Rolling PixelUMM state follows the opt-in extension
in `wtz2333/Hy-Nomas@16a4c228fd6f4770ed4a111b360031f7686c4a19`.
These source revisions are independent test oracles, not runtime dependencies.

- `dit.py`, `layers.py`, `attention.py`, `text_encoder.py`, `camera.py` and the
  fixed trajectory matrices retain the released inference geometry, parameter
  names and conditioning conventions.
- `pixelumm_model.py`, `pixelumm.py` and `pixelumm_streaming.py` retain the
  PixelUMM/DCAE-V decoder and window-blending math. The PixelUMM model code is
  derived from `nv-tlabs/PixelUMM` (Apache-2.0). The upstream source distinguishes
  this code license from the NVIDIA One-Way Noncommercial license of released
  PixelUMM weights and derivatives; checkpoint licenses remain applicable.
- `scheduler.py` retains the Euler algorithm adapted by the upstream repository
  from Diffusers. Copyright 2024 Stability AI, Katherine Crowson and the
  HuggingFace Team, Apache-2.0.

Training code, alternative checkpoint loaders, sparse attention and quantized
implementations are not included. Omni owns checkpoint loading, process groups,
AR state, offload and output delivery.
