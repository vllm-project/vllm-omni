# Wan TeaCache validation

These scripts are an experimental calibration harness, not a production-quality
claim. The traced cache path additionally clamps negative polynomial predictions
and forces full computation above the largest calibrated input distance. Until those
policies and stage-specific profiles are integrated into the production backend and validated,
a passing harness run alone does not qualify the production backend.

Generated outputs belong under an external `WAN_VALIDATION_ROOT`, outside the
source checkout. See the [evidence summary](results/README.md) for the archived
runs and remaining production-validation requirements.

The coefficient estimator collects each stamped CFG branch as a separate
trajectory on its local transformer/PP stage. Its CPU regression exercises the
collector and fitter; it does not qualify a production cache profile or provide
distributed orchestration for the standalone estimator.

## Inputs

Use the model and exact revision in `model.json`. Download its Diffusers snapshot
to `$WAN_VALIDATION_ROOT/model`; copy `prompts.json` into that directory. The
calibration split has 24 prompts with seeds 17 and 29. The held-out split has 12
separate prompts with seeds 101, 202 and 303. Do not tune on the held-out split.
Full runs use BF16, 512x512, 17 frames, 50 steps and guidance scale 5.0.

Install the repository's dependencies with vLLM 0.31.0, then pytest, numpy,
scikit-image, imageio and imageio-ffmpeg. Preserve an exact `pip freeze`, Python,
Torch/CUDA versions, GPU topology, source SHA, commands and scheduler job ID for
each run. The validation environment used Torch 2.13.0+cu130 and Python 3.12.13.

```bash
export WAN_VALIDATION_ROOT=/shared/path/to/validation
export PATH=/shared/path/to/environment/bin:$PATH
export PYTHONPATH="$PWD/tools/wan_teacache_validation:$PWD"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_TARGET_DEVICE=cuda
# Preserve the scheduler's CUDA_VISIBLE_DEVICES.
python tools/wan_teacache_validation/campaign.py --phase calibrate --pp 1 --cfg 1 --root "$WAN_VALIDATION_ROOT/pp1-calibration"
python tools/wan_teacache_validation/campaign.py --phase calibrate --pp 2 --cfg 1 --root "$WAN_VALIDATION_ROOT/pp2-calibration"
```

Each calibration first records full-compute, branch-local input/residual changes,
then fits one quartic per PP stage. Threshold candidates use calibration prompts
only. `selection-ppN.json` records the chosen threshold. Run held-out comparisons
for each combination of PP and CFG in {1, 2} after selecting thresholds:

```bash
python tools/wan_teacache_validation/campaign.py --phase validate --pp 2 --cfg 2 --root "$WAN_VALIDATION_ROOT/pp2-cfg2-validation"
python tools/wan_teacache_validation/audit.py "$WAN_VALIDATION_ROOT/pp2-cfg2-validation/held-out" --pp 2 --cfg 2
```

The generator alternates native and cached requests in a single loaded model,
warms up both paths, and excludes loading, warmup and output serialization from
request timing. It saves float output arrays, MP4, first/middle/last frames, final
latents and real-hook per-rank decisions. `audit.py` checks first-call computation,
CFG rank/branch mapping, finite paired latents and actual skips. Rank-trace I/O and final-latent capture remain
inside request timing. Inspect videos as well as automated measurements.

`comparison.json` records mean frame SSIM, worst video mean SSIM, mean temporal
change error relative to native temporal change, and cached/native latency ratio.
Acceptance requires respectively >=0.95, >=0.90, <=0.10 and <=0.90. A failed gate
must remain visible; unsuccessful measurements must not be described as production
acceptance. The `smoke` split uses 5 frames at 256x256 and is only a startup test.

## Selection and independent evaluation

`refine.py` evaluates additional thresholds and optional early steps of full
computation on calibration prompts. For example:

```bash
python tools/wan_teacache_validation/refine.py --pp 1 --root "$WAN_VALIDATION_ROOT/pp1-refine" --thresholds .07 .075 .08 .085 .09
python tools/wan_teacache_validation/refine.py --pp 2 --root "$WAN_VALIDATION_ROOT/pp2-warmup30" --thresholds .1 --warmup-steps 30
```

After all calibration candidates finish, use `freeze.py` with the calibration
campaign directories to copy the chosen coefficients, input range, threshold,
warmup policy, prompt/model manifests and checksums into a new directory. Pass a
separate, clean checkout through `--source-checkout` and use that checkout for
evaluation. Point `WAN_VALIDATION_ROOT` at the frozen directory for those runs.
The selector refuses held-out inputs and candidates without real skips in every
stage/branch. It selects the fastest qualified candidate. If none qualifies, it
selects the candidate with the smallest worst normalized gate violation and marks
it `diagnostic_only: true`. Such a run remains a failed production candidate;
independent evaluation must not be used to tune it further.

For Slurm, use one node, four B200 GPUs, 32 CPUs and 256 GB per job. The validation
campaign used partition `overflow`, account `wrd` and job name `test`. Short
20–45 minute allocations were used for individual evaluations/refinements; no
allocation exceeded eight hours. Preserve Slurm's GPU visibility. Capture the
job ID, `nvidia-smi topo -m`, `pip freeze`, source SHA and command with each job.

## Metric definitions and limitations

SSIM uses decoded floating-point RGB frames before MP4 compression, Gaussian
weights (sigma 1.5), population covariance and data range 1. The reported video
SSIM is the mean over its 17 frames. Dataset SSIM averages those video means;
the minimum gate uses the lowest video mean.

The temporal metric is deliberately a comparison of motion changes, not just a
ratio of overall motion magnitudes. For each video, with native frames B and
cached frames C, it is:

```text
mean(abs(diff(C, time) - diff(B, time))) / (mean(abs(diff(B, time))) + 1e-8)
```

The dataset metric averages these per-video ratios. A value of 0.10 is the
acceptance limit. This metric can detect differences even when both videos look
smooth. No claim of visible flicker should be inferred solely from it.

Latency is the ratio of total cached to total native request time over equal
request sets, including tracing overhead. Loading, warmup, image/video export and
metric computation are excluded. `hits` counts reuse of the entire local block
stack, not individual layers. The audit verifies that each new request starts at
step zero with full computation, validates CFG branch mapping, and reports paired
latent max absolute error, RMSE and relative L2 error.

Use `quality_junit.py comparison.json quality.xml` to export all four measured
gates, including failures. A completed Slurm process means artifacts were
produced; production acceptance additionally requires every numerical, quality,
real-cache-use and performance check to pass.
