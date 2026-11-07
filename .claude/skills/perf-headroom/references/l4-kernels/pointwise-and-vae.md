# Norm, modulation, RoPE, layout and media decode

## Norm / modulation / RoPE / pointwise

Inventory reduction dimensions, tensor size/strides, broadcast inputs, dtype,
call frequency and adjacent producers/consumers. Compare per-call and aggregate
cost: short kernels can matter through repetition even when no single launch is
large. Inspect current compile/backend fusion before suggesting another fusion.

1. Determine whether the operation is a view, a materializing kernel or already
   part of a larger fused kernel.
2. Build logical read/write and reduction work models. Compare measured DRAM/L2
   traffic, not logical bytes alone, using the memory guide.
3. Distinguish launch-exposed cost, insufficient grid parallelism, reduction
   dependencies, transaction inefficiency and actual bandwidth saturation.
4. Trace modulation/position inputs across steps/branches to test invariance;
   a recurring shape does not imply recurring values.
5. Compare a complete region with any proposed fusion/packing alternative. Include
   precision, reduction order, broadcasting, register use and memory lifetimes.

If a kernel is bandwidth-efficient but writes a temporary consumed immediately,
record mature execution and a separate L3 structural candidate. If it is slow
relative to peak but negligible on the critical path, report low current leverage,
not "bad kernel" as the top priority.

## VAE / convolution / upsample / decoder

Treat media decode as its own phase, not a residual folded into DiT. Record the
actual convolution dimensions, padding, groups, temporal dimensions, layout,
dtype/accumulation, selected algorithm, tile sizes/overlap and memory peaks.

1. Map decode through conv, norm, activation, upsample, tile assembly and output
   conversion; separate GPU decode from CPU media encoding and file/network work.
2. Check shape-weighted conv cost and algorithm/layout compatibility. A faster
   convolution that requires repeated layout conversions may lose overall.
3. Inspect repeated tile borders and assembly. Some overlap is required for
   receptive fields or temporal continuity; do not label it redundant without
   proving the alternative preserves output quality.
4. Inspect peak live memory and why tiling/offload exists. A full decode reference
   that exceeds available memory is not an admissible replacement.
5. Use the compute/memory/scheduling branches for important conv and non-conv
   kernels. Do not infer VAE efficiency from DiT attention or linear counters.
6. With patch parallelism, follow rank dependencies and boundary exchange in L5.

## Recommendation and acceptance

Possible directions include compatible conv algorithm/layout, decoder-region
fusion, lifetime/tiling changes or removal of duplicate output materialization.
Each requires source/runtime evidence and a complete-region reference. Do not
recommend a menu of all options for every decoder.

Acceptance proposals include full decode time, peak memory, image seams, temporal
consistency, output dimensions and declared quality metrics. Changed VAE precision
or approximate tiling requires the relevant quality permission. Unprofiled VAE
work remains UNKNOWN even when the denoiser has extensive NCU coverage.
