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

Hardware dispatch is an explicit allowlist. SM90, SM100, and SM103 are enabled;
other capabilities fall back to the reference path. Bit-exact full-decode and
operator evidence has been collected on SM90, with independent full-decode and
stress validation on SM103. Enabling a target and claiming it as validated are
kept separate so the evidence remains clear.

## Extending platform support

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
