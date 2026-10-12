# LingBot-Video Benchmarks

These manual benchmarks compare the upstream LingBot-Video implementation with
vLLM-Omni. Run all commands from the repository root.

| Benchmark | Script | Purpose |
|-----------|--------|---------|
| Dense pipeline parity | `dense_pipeline_parity.py` | Compare generated videos and report MAE, MSE, PSNR, latency, and optional steady-state timings |
| MoE numerical parity | `moe_transformer_parity.py` | Check bitwise parity for the sparse MoE block and the full MoE transformer |

Both benchmarks require a local LingBot-Video checkout and local model files.
They do not run in CI.

## Dense pipeline parity

The dense benchmark launches the upstream Diffusers inference script and the
vLLM-Omni offline example with identical prompts and sampling parameters. It
then compares the decoded MP4 frames.

```bash
CUDA_VISIBLE_DEVICES=0 \
python benchmarks/lingbot_video/dense_pipeline_parity.py \
  --model /path/to/lingbot-video-dense-1.3b \
  --official-repo /path/to/lingbot-video \
  --output-dir /tmp/lingbot_dense_parity
```

Use `--runs N` to additionally measure repeated in-process vLLM-Omni
requests. The video comparison reports MAE, MSE, and PSNR; it is an end-to-end
pipeline diagnostic rather than a bitwise transformer check.

## MoE numerical parity

The lightweight block comparison uses deterministic weights and inputs to
cover correction-bias routing, group-limited top-k, routed experts, FP32
scatter-weighted restore, padding masks, and the shared expert.

```bash
CUDA_VISIBLE_DEVICES=0 \
python benchmarks/lingbot_video/moe_transformer_parity.py \
  --scope block \
  --official-repo /path/to/lingbot-video \
  --output-json /tmp/lingbot_moe_block_parity.json
```

The full comparison loads the upstream and vLLM-Omni 30B transformers
sequentially and compares their output tensors. It requires a CUDA device with
`torch._grouped_mm` and enough memory for one 30B BF16 transformer.

```bash
CUDA_VISIBLE_DEVICES=0 \
python benchmarks/lingbot_video/moe_transformer_parity.py \
  --scope transformer \
  --official-repo /path/to/lingbot-video \
  --model /path/to/lingbot-video-moe-30b-a3b \
  --output-json /tmp/lingbot_moe_transformer_parity.json
```

For the correctness oracle, the upstream transformer uses
`diffusers:_native_math` and vLLM-Omni uses
`TORCH_SDPA + SDPBackend.MATH`. The command exits with a nonzero status unless
the selected comparisons are bitwise equal.
