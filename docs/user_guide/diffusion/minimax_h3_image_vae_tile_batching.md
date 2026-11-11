# MiniMax-H3-Image VAE tile batching

MiniMax-H3-Image can batch equal-shaped VAE tiles before CUDA graph replay.
This reduces small-kernel overhead without changing the tile grid, overlap,
blending, checkpoint precision, attention backend, or six-step PDD schedule.
It batches tiles within one request, not requests or images.

## Configuration

The opt-in `additional_config.vae_tile_batch_size` accepts integers from 1
through 32. The default is 1 and preserves single-tile graph replay.
Values greater than 1 require VAE CUDA graphs; do not combine them with
`--enforce-eager` or `--no-enable-cuda-graph-decode`. Non-CUDA devices reject
batching rather than silently ignoring it.

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve MiniMaxAI/MiniMax-H3-Image --omni \
  --num-gpus 1 --usp 1 --tensor-parallel-size 1 --text-encoder-tp-size 1 \
  --diffusion-attention-backend TRTLLM_ATTN \
  --additional-config '{"vae_tile_batch_size":32}' \
  --host 127.0.0.1 --port 8000
```

Choose a smaller value when memory is constrained. Larger batches can use
more graph and activation memory and need not be faster on every accelerator.
Only single-B300 text-to-image performance is measured here; image editing,
offload, multi-GPU layouts and other accelerators are not qualified by these
measurements. Existing per-platform graph replica limits are unchanged.

Full batches and the remaining tiles use separate captured graph shapes.
Outputs return to their original order before the existing blending path.
Changing the tile batch size rebuilds the graph cache. Decoder weight or
precision changes continue to invalidate captured graphs.

## Numerical Behavior

Batching changes floating-point kernel selection. Outputs are **not promised
to be pixel-identical** to batch size 1, even at the same model precision.
Use the default when exact compatibility with unbatched decoding is required.
Repeated outputs within one configuration and comparisons against batch size
1 should be tested separately.

## Reproduce the Comparison

Use the same checkpoint, runtime, single GPU, attention backend and prompt
corpus for both servers. For the measured checkpoint, pin
`MiniMaxAI/MiniMax-H3-Image` to revision
`492fd8c4eca666f1085f57e30738c1e0d86b3439`. All weights are unchanged; the
measurement used a local model index selecting the bundled text encoder
instead of making another Hub request.

Run the server above first with `vae_tile_batch_size` set to 1, then 32,
restarting between configurations. After each server becomes healthy, run
the existing benchmark from the repository root:

```bash
BATCH=1  # Set to 32 for the optimized server.
for SIZE in 2048 4096; do
  python benchmarks/diffusion/h3_image_bench.py \
    --base-url http://127.0.0.1:8000 --model MiniMaxAI/MiniMax-H3-Image \
    --corpus benchmarks/diffusion/h3-image-tile-batching-prompts.jsonl \
    --output-dir "results/batch-${BATCH}-${SIZE}" \
    --phases c1-b1 c1-b2 c1-b3 --num-prompts 3 --warmups 2 \
    --width "$SIZE" --height "$SIZE" --steps 6
done
```

If serving a local checkpoint, use that same path in both server and benchmark
commands. Each size has three phases with two warmups and three measured
requests per phase. Requests use fixed seeds, a CPU noise generator and PNG
output. The benchmark validates every PNG and records request success.
Repeat the A/B pair with fresh processes and output directories.

Latency runs from POST submission through complete HTTP response receipt,
including generation, PNG encoding and base64 transfer, but excluding client
JSON/base64 decoding and file persistence. Do not mix profiler requests into
these latency measurements. This small corpus does not establish production
percentiles or concurrent throughput.

Compare RGB PNG images with matching prompt IDs and sizes. PSNR and mean absolute
RGB difference quantify numerical changes relative to batch size 1, not
quality against ground truth. Visual inspection is supplementary; no FID,
CLIP score or general perceptual-quality claim follows from this comparison.

## Regression Tests

The focused tests need the repository development dependencies and one CUDA
GPU; they use random small weights and do not download the checkpoint:

```bash
python -m pytest -o addopts= -q \
  tests/diffusion/models/minimax_h3/test_minimax_h3_image.py \
  tests/diffusion/models/minimax_h3/test_minimax_h3_image_vae_cuda.py
```

They cover invalid settings, tile order, full and partial batches, input
batch size greater than one, mixed shapes, output ownership across graph
replays and graph-cache switching. The original eager/default-graph exact
output tests remain in place.
