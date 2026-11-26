# MiniMax-H3 optimized operators

This package contains eager optimizations whose contracts are specific to the
MiniMax-H3 model implementation. Keeping them beside the model is intentional,
not an assumption that their underlying operations can never be shared.

## Why model-owned

The optimized path must preserve the H3 VAE's exact operation order, dtype and
rounding behavior, tensor layout, execution mode, and remote-code contract. A
kernel that implements a mathematically similar expression may still change the
decoded video, and a kernel that is beneficial on one platform may regress on
another. The model package therefore owns the capability checks, per-operation
input guards, reference fallbacks, numerical evidence, and performance evidence.

This placement is one point in the design space being discussed in the
[diffusion operator-boundary RFC](https://github.com/vllm-project/vllm-omni/issues/6305).
If that RFC establishes a suitable shared tier for any of these operations,
migration can be handled separately without weakening the current contract.

## Current boundary

The VAE installer validates the official model structure before making any
change. It then binds the selected operator set, reuses Omni's existing
`SiluAndMul`, and materializes only the decoder-block Linear weights in the FP16
dtype already selected by decode autocast. Unsupported model contracts, tensor
inputs, execution modes, and devices retain the original implementation.

Hardware dispatch is an explicit allowlist. SM89, SM90, SM100, SM103, and SM120 are enabled;
other capabilities fall back to the reference path. Bit-exact full-decode and
operator evidence has been collected on SM89, SM90, and SM120, with independent full-decode and
stress validation on SM103. Enabling a target and claiming it as validated are
kept separate so the evidence remains clear.

## Extending platform support

### Experimental NPU video encoder empty spatial padding

`VLLM_OMNI_H3_VAE_SKIP_NOOP_PAD=1` opts into removal of the encoder's
all-zero spatial `F.pad` for convolutions whose padding is `(1, 0, 0)`.
The temporal pad is still required: the convolution's complete padding
configuration is not zero. Downsampling's outer nonzero right/bottom spatial
pad and the original Conv3D forward also remain in place.

The switch is disabled by default and read at NPU video VAE construction;
`true`/`on` also enable it, `0`/`false`/`off` or an empty value disable it,
and any other value raises `ValueError`. Restart workers when changing it.
It is classified as deprecate/remove: remove the temporary switch or move
validated behavior into normal dispatch after public device qualification.

Installation binds only the loaded `EncoderFCN3D`'s concrete
`SpatialParallelConv3d`, after checking its padding on CPU. Eager,
contiguous NCDHW FP32 NPU video tensors use the path. T=1 images retain the
checkpoint path, avoiding overlap with the separate NPU image convolution
optimization. Other devices, nonzero spatial padding, changed checkpoint
contracts, spatial parallelism, strided tensors, training, gradients,
autocast, hooks, and compilation retain the original method. Decoder and
quant_conv modules are untouched.

CPU tests cover actual padding/convolution results, the removal of the empty
pad call, the retained outer spatial and temporal pads, and fallbacks. Public
NPU bitwise accuracy, complete encoding latency, and end-to-end media checks
have not been run, so the default remains disabled.

### Platform qualification

Validation and improvements for other platforms are welcome. A contribution
should keep the model-facing installation path unchanged and add a complete
operator-set entry with conservative fallback behavior. Please include:

- the GPU product, compute capability, driver, CUDA, PyTorch, and Triton versions;
- representative production shapes plus edge-shape and unsupported-input tests;
- complete decoded-output comparison against the reference path;
- direct numerical checks for every optimized operation;
- warmed latency measurements for both complete decode and individual operations;
- tests proving that unsupported targets and contracts still use the reference path.

Bitwise equality is the current numerical contract. If a platform cannot meet
it, please discuss the proposed quality contract and validation criteria before
changing the default path.
