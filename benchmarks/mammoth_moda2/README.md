# MammothModa2 fixed-conditioning qualification

These are regression/benchmark tools, not alternative inference entrypoints.
They use the native model, loader, scheduler, CFG, collectives and VAE. Run
from the repository root with the normal vLLM-Omni dependencies installed.
Keep the model revision, source revision, hardware, dtype, backend and all
sampling parameters with the resulting artifacts. Never replace a golden
reference with a candidate output to make a comparison pass.

## Capture one real AR payload

Use the shared text-to-image example through an observation-only wrapper.
`mammoth-sp2.yaml` is the opt-in deployment configuration in the
[model recipe](../../recipes/MammothModa2/MammothModa2.md). For a BF16 run,
set stage 1's `engine_extras.dtype` to `bfloat16` explicitly; the original
FP32 control uses `float32`. The capture directory must not already exist.

```bash
export mammoth_model=/absolute/path/to/MammothModa2-Preview
export MAMMOTH_CONDITIONING_DIR=/absolute/path/to/qualification/conditioning
CUDA_VISIBLE_DEVICES=0,1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
python benchmarks/mammoth_moda2/capture_conditioning.py \
  --model "$mammoth_model" --deploy-config ./mammoth-sp2.yaml \
  --ulysses-degree 2 --ulysses-mode advanced_uaa --enforce-eager \
  --prompt 'A red ceramic teapot on a wooden table beside a small green plant, soft morning light, detailed product photograph.' \
  --height 1024 --width 1024 --seed 42 --num-inference-steps 50 \
  --guidance-scale 4 --extra-body '{"cfg_range":[0.0,1.0]}' \
  --output ./mammoth-sp2.png
```

The wrapper calls the original stage input processor unchanged and records
its complete hidden states, token IDs, answer boundary and image dimensions.
The capture is an input artifact: reuse this same directory for every paired
DiT run. Generating a different AR payload for each SP configuration is not
a fixed-conditioning comparison.

## Native DiT replay and measurement

Run configurations sequentially on otherwise idle GPUs. Each process executes
one observed warmup and three measured requests. Observer hooks are removed
before measurement, and memory-counter RPCs and image serialization are
outside the timed region. The measured interval includes the native diffusion
engine request and output transfer, but excludes engine startup and AR.
Outputs include every raw decoded tensor and PNG, per-rank peak allocated and
reserved bytes, actual component parameter dtypes, first-step noise and
conditioning, selected denoiser predictions, and the VAE decoder input.

```bash
PYTHONPATH="$PWD:$PWD/benchmarks/mammoth_moda2" \
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
python benchmarks/mammoth_moda2/replay_checkpoint.py \
  --model "$mammoth_model" --conditioning "$MAMMOTH_CONDITIONING_DIR" \
  --output ./qualification-sp1 --degree 1 --backend TORCH_SDPA \
  --dtype bfloat16 --steps 50 --seed 42 --guidance 4

PYTHONPATH="$PWD:$PWD/benchmarks/mammoth_moda2" \
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 \
python benchmarks/mammoth_moda2/replay_checkpoint.py \
  --model "$mammoth_model" --conditioning "$MAMMOTH_CONDITIONING_DIR" \
  --output ./qualification-sp2 --degree 2 --backend TORCH_SDPA \
  --dtype bfloat16 --steps 50 --seed 42 --guidance 4

python benchmarks/mammoth_moda2/compare_replays.py \
  ./qualification-sp1 ./qualification-sp2 --output ./comparison.json
```

Repeat the pair with `--backend FLASH_ATTN` and new output directories.
Do not treat a backend change and an SP change as the same experiment.
The comparison reports self-repeat and cross-rank equality, raw tensor error,
normalized RGB error, PSNR and 11x11 Gaussian-window SSIM. Inspect the saved
images too. It produces measurements, not an automatic broad quality verdict.
Empty `stage_durations` means native substage timing is unavailable, not zero.

`gpu_ms_per_image` is allocated DiT GPU count times request latency. SP=2
replicates model weights, refiners and VAE; reduced latency does not imply
lower aggregate memory or GPU cost. The warmup is excluded from the summary.
When comparing source revisions, put the selected checkout first in
`PYTHONPATH` and retain the benchmark directory so the worker extension can
be imported. An old attention implementation inside the shared diffusion
runtime is an attention-migration control, **not** a legacy-runtime baseline.

## Preview and Dev understanding

The understanding harness uses the shared Omni API and prompt builder,
asserts a single AR-only stage, and executes a greedy A/B/A sequence. It saves
exact text, token IDs and stop reasons and rejects empty/truncated answers or
non-identical A/B/A replay. Use the same reference image on both revisions.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
python benchmarks/mammoth_moda2/understanding_regression.py \
  --model "$mammoth_model" \
  --deploy-config benchmarks/mammoth_moda2/understanding-ar.yaml \
  --image ./mammoth-sp2.png --mode text-to-text \
  --output ./understanding-text.json

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
python benchmarks/mammoth_moda2/understanding_regression.py \
  --model "$mammoth_model" \
  --deploy-config benchmarks/mammoth_moda2/understanding-ar.yaml \
  --image ./mammoth-sp2.png --mode image-to-text \
  --output ./understanding-image.json
```

Repeat with the pinned Dev checkpoint and both source revisions. Compare
the `prompt_index` and `completions` fields, not wall-clock durations. These
few requests are targeted non-regression evidence, not an understanding
accuracy benchmark or validation of Dev text-to-image.
