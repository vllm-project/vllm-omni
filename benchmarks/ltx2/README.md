# LTX-2.5 Stage-2 TDP A/B benchmark

This manual benchmark compares two four-GPU, two-stage LTX-2.5 runs:

- **Global SP:** Ulysses sequence parallelism and global attention in both stages.
- **Stage-2 TDP:** global SP in Stage 1, then a 2x2 overlapping spatial tile grid
  with local attention in Stage 2.

Both runs use the same prompt, seed, frame count, internal latent resolution,
and the official two-step tiled Stage-2 sigma schedule (`0.625, 0.4, 0.0`).
Matching the schedule isolates the attention strategy; comparing against the
ordinary three-step Stage-2 schedule would mix a sampling change into the
quality and latency result.

Run from the repository root when four otherwise-idle GPUs are available:

```bash
python benchmarks/ltx2/tiled_data_parallel_ab.py \
  --devices 0,1,2,3 \
  --model Lightricks/LTX-2.5-Diffusers \
  --output-dir /tmp/ltx25_tdp_ab
```

For a 3840x2160 target, both runs compute a 3840x2176 internal shape. The
ordinary global-SP run writes all 2176 rows, while the TDP runtime crops its
decoded result to 2160 rows. The comparison filters crop the global-SP video
to the same target without an additional encode.

Outputs:

- `global_sp_internal.mp4` and `stage2_tdp.mp4`: source clips;
- `side_by_side.mp4`: labeled, downscaled visual comparison;
- `global_sp.log` and `stage2_tdp.log`: complete generation logs;
- `comparison.json`: request latency, peak memory, profiler events, full-frame
  SSIM/PSNR, and SSIM/PSNR over the horizontal and vertical overlap bands.

Use `--dry-run` to inspect the exact commands, or `--analyze-only` to rebuild
the metrics and side-by-side video from existing source clips. Existing logs
are reused when present. Video metrics do not score audio: the TDP path
deliberately returns the full-context Stage-1 audio, matching the upstream LTX
multi-GPU behavior.
