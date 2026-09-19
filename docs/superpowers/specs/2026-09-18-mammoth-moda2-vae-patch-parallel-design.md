# MammothModa2 VAE Patch-Parallel Decode Design

## Summary

Enable MammothModa2's `gen_vae` to use the existing distributed VAE executor for patch/tile-parallel decode. The implementation remains narrowly scoped to distributed decode and does not take ownership of ordinary single-rank VAE slicing or tiling from #7435.

The work is based on the shared diffusion runtime introduced by #7134. Until #7134 merges, development and tests live on a branch stacked on its latest head; the eventual PR should be rebased onto `main` after #7134 lands.

## Goals

- Make `vae_patch_parallel_size > 1` reach MammothModa2's actual `gen_vae` component.
- Reuse `DistributedAutoencoderKL` and `DistributedVaeExecutor`; do not add a Mammoth-specific distributed decoder.
- Preserve checkpoint keys under `gen_vae.*`, latent scaling/shift, decoded shape/dtype, and output postprocessing.
- Compare PP=1 and PP=2 for correctness, per-rank peak memory, and VAE decode latency.
- Keep the expensive validation path economical by testing the VAE independently before one final end-to-end run.

## Non-Goals

- Enabling or benchmarking ordinary single-rank `vae_use_slicing` or `vae_use_tiling`; that is #7435.
- Changing VAE kernels, halo geometry, tile assignment, gather/stitch algorithms, or process-group construction.
- Adding DiT tensor/sequence/CFG parallelism, request batching, step execution, caching, offload, or quantization.
- Claiming a latency improvement when the result is only a peak-memory or capacity improvement.
- Supporting `vae_patch_parallel_size` larger than the existing diffusion worker world.

## Considered Approaches

### A. Add `vae` as an alias for `gen_vae`

This is the smallest patch and makes the existing registry path recognize MammothModa2. It is rejected because the same alias also causes generic slicing/tiling configuration to reach `gen_vae`, overlapping #7435 and coupling two independently owned features.

### B. Configure declared distributed VAE components for patch parallelism

This is the selected approach. The registry discovers VAE components declared through `_vae_modules`, selects components implementing `DistributedVaeMixin`, and applies only patch-parallel configuration. MammothModa2 changes its VAE implementation from Diffusers `AutoencoderKL` to the API-compatible `DistributedAutoencoderKL` while retaining the `gen_vae` attribute and checkpoint prefix.

This approach improves the shared runtime rather than adding a model-name special case. It also avoids changing ordinary slicing/tiling behavior when patch parallelism is disabled.

### C. Install a Mammoth-specific decode wrapper

This would wrap `gen_vae.decode` inside the Mammoth pipeline or call the legacy patch-parallel helper directly. It is rejected because it duplicates registry policy and creates a second distributed VAE integration path that future models would have to copy.

## Architecture

### Patch-parallel component configuration

Add a focused registry helper that receives the initialized pipeline and `OmniDiffusionConfig`:

1. Read `vae_patch_parallel_size`.
2. If it is at most one, return without changing VAE behavior.
3. Preserve the current `model.vae` selection when that attribute implements `DistributedVaeMixin`.
4. Otherwise resolve the attributes declared by `_vae_modules` and filter them to `DistributedVaeMixin` instances.
5. Configure the declared component only when exactly one compatible VAE is found. If none is found, emit the existing unsupported-feature warning once; if several are found, warn that the target is ambiguous and leave them unchanged.
6. Enable the selected component's required `use_tiling` flag and call `set_parallel_size(size, mode=vae_parallel_mode)`.

The helper replaces only the current patch-parallel selection/configuration block. The existing memory-optimization block for conventional `model.vae` remains separate. The helper must not apply `vae_use_slicing` or user-requested single-rank `vae_use_tiling`; those remain in the existing configuration path and #7435's scope. Enabling `use_tiling` when PP>1 is an intrinsic precondition of `DistributedVaeMixin.is_distributed_enabled()`, not a claim of standalone tiling support.

### MammothModa2 VAE type

Construct `gen_vae` with `DistributedAutoencoderKL.from_config` instead of Diffusers `AutoencoderKL.from_config`. The distributed class subclasses the same Diffusers implementation, so existing configuration, state-dict layout, weight loading, scaling/shift, and `decode(..., return_dict=False)` usage remain unchanged.

No `vae` alias is added. The module remains registered only as `gen_vae`, preventing duplicate module naming and keeping `gen_vae.*` checkpoint keys stable.

### Decode data flow

1. MammothModa2 finishes denoising and applies the existing scaling and shift transformations.
2. `gen_vae.decode` selects its existing distributed strategy:
   - tile distribution when the latent exceeds the configured tile threshold;
   - patch distribution with latent halos for smaller inputs.
3. Each active rank decodes its assigned work through the unchanged VAE decoder.
4. The existing executor gathers decoded regions and reconstructs the result on rank 0.
5. MammothModa2 returns the same `DiffusionOutput`; the shared postprocessor continues producing RGB images.

The first PR does not change `broadcast_result=False` or output ownership. All ranks must still enter the decode collectives in the same order.

## Failure and Fallback Behavior

- PP=1 follows the existing non-distributed decode path.
- PP>1 with a non-distributed VAE logs one clear warning and preserves the existing decode path.
- A configured size larger than the process-group world retains the executor's existing clamping/warning behavior.
- Unsupported or missing declared VAE attributes are reported without silently configuring another module.
- Distributed collective, decode, or stitching failures propagate; the feature must not silently retry a single-rank decode after ranks have entered collectives.

## Testing

### CPU tests

- Registry configuration discovers a declared `gen_vae` implementing `DistributedVaeMixin`.
- PP=1 leaves the component unchanged.
- PP=2 enables the tiling prerequisite and records the requested parallel size/mode.
- PP=2 with no compatible distributed VAE emits the unsupported warning.
- MammothModa2 constructs `gen_vae` as `DistributedAutoencoderKL` while retaining `_vae_modules == ["gen_vae"]`.
- The existing Mammoth request parsing, VAE scaling/shift, output contract, and postprocessing tests remain green.

Tests are written before production changes and must first fail for the missing discovery/type behavior.

### Two-GPU VAE-only experiment

Run a small standalone `torchrun --nproc-per-node=2` harness that creates the Mammoth VAE architecture and feeds a fixed latent. Full AR and DiT weights are not required for the initial experiment; identical architecture and shapes are sufficient for execution time and activation-memory comparisons.

For PP=1 and PP=2, record:

- warmup count and measured iteration count;
- per-rank `torch.cuda.max_memory_allocated()` and `max_memory_reserved()`;
- CUDA-event VAE decode latency, median and p95;
- output shape, dtype, finite-value check, maximum absolute error, mean absolute error, relative L2, and PSNR where the output range permits it;
- GPU model, topology, software versions, latent shape/dtype, tile geometry, and exact commit.

The first matrix uses BF16 and the 1024x1024 generation latent shape, followed by one larger shape that materially exercises the memory benefit. The harness must synchronize around measured regions and exclude construction/warmup time.

### Final integrated validation

After #7134 merges, run one matched MammothModa2-Preview text-to-image comparison on two GPUs:

- fixed prompt, seed, resolution, inference steps, guidance, checkpoint, and dependency revisions;
- PP=1 versus PP=2;
- successful final RGB image with the requested dimensions;
- VAE latency, end-to-end latency, and per-rank peak memory;
- numerical/image comparison with seam inspection;
- repeated unprofiled samples with variability reported separately from diagnostic profiling.

## Acceptance Criteria

- CPU coverage proves the shared registry configures `gen_vae` without adding a model-specific registry branch.
- PP=1 preserves the existing MammothModa2 path.
- PP=2 executes distributed VAE decode on two ranks and produces the expected output shape/dtype with finite values.
- The PP=1/PP=2 comparison publishes numerical error and visible seam evidence rather than claiming bit identity.
- The result reports a measurable per-rank peak-memory reduction at a shape that exercises parallel decode, or explicitly reports that the existing executor does not provide one for the tested shape.
- No ordinary slicing/tiling implementation or benchmark is included in the PR.

## Delivery

The implementation should be split into focused commits:

1. Shared registry discovery/configuration tests and helper.
2. MammothModa2 distributed VAE type and model tests.
3. VAE-only benchmark harness and recorded run instructions.
4. Documentation/evidence updates after GPU validation.

The PR will use `Refs #7075`, reference the focused VAE patch-parallel issue, list #7134 as its runtime dependency, and explicitly coordinate the shared `gen_vae` boundary with #7435.
