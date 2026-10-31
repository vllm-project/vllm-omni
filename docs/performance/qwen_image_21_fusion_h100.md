# Qwen-Image 2.1 fused operations on H100

This is the A1 fusion work in [#8586](https://github.com/vllm-project/vllm-omni/issues/8586),
measured on a B3 four-replica workload. A full TP/CFG/SP topology comparison remains separate.

## Measured configuration

Baseline Omni: `3dc35694b3d458fc1461a8fe71832ac12e8c8cea`.
Fusion source: `287d9483a5c4ff3dde9870bba91126917d86d6ad`.
Model: `Qwen/Qwen-Image-2.1@b3179ad355be050328e483a9dfdd9e60cd62adfa`; all 27 files SHA256-verified.

Four independent TP1/SP1/CFG1 H100 servers, actual transformer B1; BF16,
1024x1024, 50 steps, true CFG 1, FLASH_ATTN, eager, no decode graph or offload,
VAE tiling. Both arms have the same four GPUs and CPU/memory allocation.
Devices report 65,247 MiB each; NVIDIA driver 595.91.07.
PyTorch 2.13.0+cu130, vLLM 0.31.0, Transformers 5.14.1, Diffusers 0.40.0,
FlashInfer 0.7.0.post1; CUDA build 13.0.

Fresh servers run A/B/B/A. Each start has four first-validation, eight warmup and
16 qualification requests outside timing, followed by 128 measured requests at C64.
C64 means client concurrency. Per-second monitoring found no foreign GPU processes.

| Round | Arm | Success | Images/s | P50 seconds | P95 seconds | P99 seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Baseline | 128/128 | 0.427811 | 148.605 | 149.579 | 149.861 |
| 1 | Fusion | 128/128 | 0.618699 | 101.857 | 103.159 | 103.217 |
| 2 | Baseline | 128/128 | 0.427446 | 148.933 | 149.690 | 149.778 |
| 2 | Fusion | 128/128 | 0.618552 | 101.666 | 103.194 | 103.279 |

Mean throughput: 0.427628 -> 0.618625 images/s, **+44.6643%**.
Mean P95 reduction: **31.0477%**. This describes two paired rounds at this fixed load.
All **256 paired timed images have identical decoded RGB pixels**; 59 fusion GPU tests pass.
Sampled maximum single-device memory: 33.506 GiB baseline, 33.494 GiB fusion.
One-second NVML sampling can miss instantaneous peaks. Generated images are model output,
not ground truth.

Older shared-node throughput numbers are superseded. Their output comparisons remain historical evidence.
This PR contains fusion only. The original six-resident-layer offload implementation is preserved
on a separate legacy branch and is excluded from this comparison.

The current PR adds stricter row validation and eager fallback for strided or differently
typed/device sublayers. The measured contiguous BF16 path uses the same GPU kernels.
The table above belongs to the pinned measured source, not a rerun of these extra guards.
The four newly added CUDA fallback regressions have not run on the macOS submission host.

## Reproduce a paired comparison

Use the pinned server sources above when reproducing the table; subsequent main changes to
sampling defaults are outside this recorded cohort. On an already allocated four-GPU machine,
start four independent services on ports 8000-8003,
each assigned exactly one GPU. Set MODEL to the pinned local checkpoint; for each server,
set its device and port:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
vllm serve "$MODEL" --omni --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --num-gpus 1 --tensor-parallel-size 1 \
  --ulysses-degree 1 --cfg-parallel-size 1 --diffusion-attention-backend FLASH_ATTN \
  --stage-overrides '{"0":{"max_num_seqs":1,"enforce_eager":true,"step_execution":true,"enable_cuda_graph_decode":false}}' \
  --vae-use-tiling --no-enable-cuda-graph-decode --enforce-eager --step-execution
```

Run the checked-in HTTP client with the same prompt and seeds in both arms.
The CLI creates a reproducible fixed-shape cohort; keep its emitted request records.

```bash
python benchmarks/diffusion/benchmark_image_replicas.py \
  --endpoint http://127.0.0.1:8000 --endpoint http://127.0.0.1:8001 \
  --endpoint http://127.0.0.1:8002 --endpoint http://127.0.0.1:8003 \
  --model "$MODEL" --prompt "$PROMPT" --seed 42 \
  --concurrency 64 --requests 128 --warmup 8 \
  --width 1024 --height 1024 --steps 50 --cfg 1 --output "$RESULT_DIR"
```

The client retains failures and validates complete decoded images after timing.
Repeat with fresh services in A/B/B/A order. Compare per-request RGB hashes between arms.
Its local HTTP fixture tests concurrency, failed-response accounting and natural drain;
that fixture is not model/GPU validation.

```bash
python -m pytest -q \
  tests/diffusion/models/qwen_image_21/test_fused_kernels.py \
  tests/diffusion/models/qwen_image_21/test_modulation_fusion.py \
  tests/diffusion/models/qwen_image_21/test_gated_norm_fusion.py
```
