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

Hardware dispatch is an explicit allowlist. SM90, SM100, SM103, and SM120 are enabled;
other capabilities fall back to the reference path. Bit-exact full-decode and
operator evidence has been collected on SM90, with independent full-decode and
stress validation on SM103. Enabling a target and claiming it as validated are
kept separate so the evidence remains clear.

SM120 was validated on one RTX 5090 D v2 (24 GB), driver 580.126.20,
PyTorch 2.13.0+cu130, Triton 3.7.1, and vLLM 0.30.0. With official H3 VAE
weights, FP16 decode autocast, and the tiled eager `decode_latent` path, a sampled `[1, 24, 37, 32, 48]` latent decoded
to 124 frames at 512x768. Warmed ABBA median decode latency decreased from
6202.364 to 4631.699 ms (25.32%). RGB8 and pre-conversion FP32 output were
bit-exact; three additional cropped, zero, and noise latent inputs also passed
full FP32 equality. The small 195-row scaled-residual operator was slower in
isolation; the complete decode improved. These are video-VAE measurements,
not end-to-end generation or serving throughput. Compile, untiled decode,
DLO residency staging, and multi-GPU patch-parallel execution were not
measured. Calls traced inside a compiled VAE region retain the reference path;
regional DiT compilation alone leaves VAE decode eager and does not disable
these VAE operators. The capability `== 120`
allowlist also applies to the in-tree `MiniMax-H3-5090.md`,
`MiniMax-H3-RTX-PRO-5000.md`, and `MiniMax-H3-RTX-PRO-6000.md` recipes,
including configurations with `--vae-patch-parallel-size` greater than one.
The RTX PRO 5000/6000 VAE timing tables and multi-GPU configurations were not
remeasured here; the 5090 recipe has no VAE timing table. SM121 remains on the
reference path. The mocked dispatch test checks capability routing only;
SM120 kernel equality is supported by the real 5090 D v2 measurements above.

The `MiniMax-H3-Spark-GB10.md` recipe was not revalidated here. Its historical
VAE timings are not evidence for this operator path. Dispatch uses the reported
compute capability, not the product name: `120` selects the operators and FP16
decoder Linear precast; `121` retains the reference path.

The [reproduction commands and scripts](https://github.com/Tokha233/ComfyUI-H3-SpeedKit/tree/d6d6da7/experiments/omni-migration-1002)
and [raw measurements](https://github.com/Tokha233/ComfyUI-H3-SpeedKit/tree/d6d6da7/evidence/omni-migration-1002)
pin the baseline, official model revision, autocast, warmup, and timing scope.

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
