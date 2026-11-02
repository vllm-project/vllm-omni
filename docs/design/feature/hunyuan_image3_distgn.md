# HunyuanImage3 VAE DistGN

NPU layers are selected by `hunyuan_image3/layers/__init__.py`, following the
existing CUDA layer dispatch. Supported local GN calls select fusion by device,
dtype, shape and execution capability; unsupported calls retain native behavior.
No model environment variable controls fusion. Local fusion preserves eager
GroupNorm's FP32 output under NPU autocast; it does not fuse a lower-precision
consumer cast by default.

For experimental full-feature spatial encode/decode, use the existing parallel
configuration rather than an environment variable:

```yaml
vae_use_tiling: true
parallel_config:
  vae_patch_parallel_size: 4
  vae_parallel_mode: spatial_shard_height
```

The VAE parallel size must match the existing DiT device group. For HunyuanImage3
on NPU this mode applies to both encode and decode. Height must divide by group
size times `ffactor_spatial` for encode, or group size for decode. Unsupported
inputs/configurations retain the tile path. The default `tile` mode retains
independent-tile statistics and never activates cross-rank GN.

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
