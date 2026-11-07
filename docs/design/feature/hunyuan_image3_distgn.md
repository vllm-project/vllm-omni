# HunyuanImage3 VAE DistGN

NPU layers are selected by `hunyuan_image3/layers/__init__.py`, following the
existing CUDA layer dispatch. Independent tiles retain native GroupNorm and
SiLU, including the caller's autocast precision. DistGN is selected only inside
the explicit spatial decode context. No model environment variable controls
fusion.

For experimental spatial decode, use the existing parallel
configuration rather than an environment variable:

```yaml
vae_use_tiling: true
parallel_config:
  vae_patch_parallel_size: 4
  vae_parallel_mode: spatial_shard_height
```

The VAE parallel size must match the existing DiT device group. For HunyuanImage3
on NPU this mode applies to decode only. Conditional-image encoding retains the
established tile path, including its normalization domain and autocast precision.
VAE encode is sensitive to numerical changes in fused operators: even local
GN fusion with the same tile boundaries introduced small posterior differences
that were amplified during the tested MXFP8 TI2I inference. Changing encode to
full-image normalization introduced an additional, independent difference in
conditioning. For this reason, encode retains native GN and SiLU as well as the
original tiling, normalization domain, and precision; neither local fusion nor
spatial DistGN is enabled for encode. These observations concern pixel
consistency and do not establish that every numerical change affects semantics.

Decode latent height must divide by the group size. Unsupported
inputs/configurations retain the tile path. The default `tile` mode retains
independent-tile statistics and never activates cross-rank GN. Requesting spatial
decode on a non-NPU backend logs a warning and retains tile execution; unsupported
NPU shapes or execution modes also log their fallback reason.

Spatial execution uses BF16 activations and convolutions with FP32 global GN
statistics; precision and normalization domains differ from FP16 independent
tiles. Final-image acceptance remains pending. It is therefore an explicit
parallel-mode choice rather than an automatic replacement of the default path.

The generic NPU operator `dist_group_norm` accepts `layout="NCTHW"` or
`layout="NHWC"`; HunyuanImage3 explicitly selects NCTHW. No implicit layout
conversion or model-specific operator implementation is used.

Use the existing diffusion performance runner and these CPU tests:

```bash
pytest -q tests/diffusion/models/hunyuan_image3/test_vae_distgn.py \
  tests/diffusion/models/hunyuan_image3/test_vae_spatial_layers.py
```

Keep two warmups separate from timing samples and disable profiling call stacks.
