# MiniMax H3 forward-input preparation

`benchmark_timestep_preparation.py` compares the CUDA time observed by the CPU
for preparing one H3 DiT call: packed video/audio tensors, sorted timestep
classes and row-to-class indices. It uses the real reference-block packer and
four sequence geometries; checkpoint weights are not required.

## Run

Use an installed vLLM-Omni CUDA environment with the candidate checkout on
`PYTHONPATH`. Export the baseline module from the revision being compared:

```bash
git show BASELINE_SHA:vllm_omni/diffusion/models/minimax_h3/denoise_loop.py > /tmp/h3_baseline_denoise.py
PYTHONPATH=. python benchmarks/minimax_h3/benchmark_timestep_preparation.py \
  --baseline /tmp/h3_baseline_denoise.py \
  --output /tmp/h3_preparation.json
```

Record the baseline and candidate SHAs beside the result. The baseline module
must be compatible with the checkout's adjacent H3 modules.

## Measurements

The script checks exact `x`, `audio_x`, `unique_timesteps` and `inverse_indices`
before timing. Each A/B/B/A arm performs 20 warmup calls and 30 measurements of
eight calls. CUDA synchronization brackets each measurement. The JSON includes
raw samples, pooled medians, sample counts, GPU name and PyTorch version.

A separate profiler pass counts `aten::nonzero` and synchronization events
inside `forward_kwargs`. Model execution, checkpoint loading, batching and
serving latency are outside this measurement. Use the
[diffusion serving benchmark](../diffusion/README.md) for request throughput
and latency percentiles.
