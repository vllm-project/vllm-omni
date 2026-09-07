# LingBot-Video Benchmarks

These manual benchmarks compare the upstream LingBot-Video implementation with
vLLM-Omni. Run all commands from the repository root.

| Benchmark | Script | Purpose |
|-----------|--------|---------|
| MoE block performance | `moe_block_benchmark.py` | Compare the legacy grouped-MM path and common FusedMoE at the measured production token shape |
| Dense pipeline parity | `dense_pipeline_parity.py` | Compare generated videos and report MAE, MSE, PSNR, latency, and optional steady-state timings |
| MoE numerical parity | `moe_transformer_parity.py` | Check strict numerical parity and separately report bitwise equality for the sparse MoE block and full transformer |

The parity benchmarks require a local LingBot-Video checkout and local model files.
They do not run in CI.

## MoE block performance

The default shape matches the nominal trace: 49,145 joint tokens, 128 experts,
top-k 8, hidden size 2,048, and expert intermediate size 768. Run the same
command from clean baseline and candidate worktrees; the JSON records the
runner classes as fast-path evidence.

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m benchmarks.lingbot_video.moe_block_benchmark \
  --warmups 5 \
  --repeats 20 \
  --output-json /tmp/lingbot_moe_block.json
```

Use `--padding-tokens N` as a separate padding stress arm. Do not mix its
latency with the unpadded production comparison. For a production-shape block
quality comparison, add `--output-tensor /tmp/<arm>.pt` to one baseline and
one candidate run with identical arguments, then compare the tensors
elementwise. Add `--routing-artifact /tmp/<arm>-routing.pt` when a mismatch
must be split into top-k expert-ID, router-weight, and expert-execution causes.

## Dense pipeline parity

The dense benchmark launches the upstream Diffusers inference script and the
vLLM-Omni offline example with identical prompts and sampling parameters. It
then compares the decoded MP4 frames.

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m benchmarks.lingbot_video.dense_pipeline_parity \
  --model /path/to/lingbot-video-dense-1.3b \
  --official-repo /path/to/lingbot-video \
  --output-dir /tmp/lingbot_dense_parity
```

Use `--runs N` to additionally measure repeated in-process vLLM-Omni
requests. The video comparison reports MAE, MSE, and PSNR; it is an end-to-end
pipeline diagnostic rather than a bitwise transformer check.

## MoE numerical parity

The lightweight block comparison uses deterministic weights and inputs to
cover correction-bias routing, group-limited top-k, routed experts, padding
compaction and restore, and the shared expert. It reports bitwise equality
separately because the common FusedMoE kernel numerically reorders expert
execution. The block gate requires identical routed expert IDs, router weights
aligned by expert ID within `rtol=3e-3, atol=2e-4`, and final output with
relative L2 at most 0.5%, cosine at least 0.99999, and max absolute error at
most 0.02. The full-transformer gate uses the same output thresholds.

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m benchmarks.lingbot_video.moe_transformer_parity \
  --scope block \
  --official-repo /path/to/lingbot-video \
  --output-json /tmp/lingbot_moe_block_parity.json
```

The full comparison loads the upstream and vLLM-Omni 30B transformers
sequentially and compares their output tensors. It requires a CUDA device with
`torch._grouped_mm` and enough memory for one 30B BF16 transformer.

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m benchmarks.lingbot_video.moe_transformer_parity \
  --scope transformer \
  --official-repo /path/to/lingbot-video \
  --model /path/to/lingbot-video-moe-30b-a3b \
  --output-json /tmp/lingbot_moe_transformer_parity.json
```

For the correctness oracle, the upstream transformer uses
`diffusers:_native_math` and vLLM-Omni uses
`TORCH_SDPA + SDPBackend.MATH`. The command exits with a nonzero status unless
the selected comparisons pass the strict numerical gate. The JSON retains an
independent `exact` field.
