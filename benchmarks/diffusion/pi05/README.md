# Pi0.5 step-batching benchmark

Compare full-forward (capacity S=1) and opt-in step execution at the same client
concurrency C, precision, weights, seeds and observations. See
[execution modes](../../../docs/user_guide/diffusion/execution_modes.md#pi05-robot-policy)
for supported behavior. This is a synthetic OpenPI workload, not a robot-task
success evaluation.

## Reproduce an A/B pair

Requires a CUDA GPU, the Pi0.5 checkpoint, a local PaliGemma tokenizer, and the
installed repository environment. Set `PI05_MODEL` and `PI05_TOKENIZER` to the
local asset directories. Install the official client:

```bash
pip install --no-deps "openpi-client @ git+https://github.com/Physical-Intelligence/openpi.git@215abfb217dbac7d5f1273282331b9b1866c0479#subdirectory=packages/openpi-client"
```

Run from the repository root. Example: C=4, S=4 for step, S=1 for full.
Change both C and step S to 1 or 8 for the other minimal comparison points.
Use a fresh result directory for every repetition.

```bash
python -m benchmarks.diffusion.pi05.prepare_deploy \
  --backend full --dtype bfloat16 --tokenizer "$PI05_TOKENIZER" \
  --output results/r1/full.yaml
python -m benchmarks.diffusion.pi05.prepare_deploy \
  --backend step --dtype bfloat16 --tokenizer "$PI05_TOKENIZER" \
  --max-num-seqs 4 --output results/r1/step.yaml
```

Start one server at a time. For the second half of the pair, replace
`full.yaml` with `step.yaml`, including in `--server-command` below.

```bash
vllm serve "$PI05_MODEL" --omni --host 127.0.0.1 --port 8093 \
  --deploy-config results/r1/full.yaml --enforce-eager --disable-log-stats
```

In another terminal:

```bash
python -m benchmarks.diffusion.pi05.benchmark_openpi \
  --backend full \
  --server-command "vllm serve $PI05_MODEL --omni --host 127.0.0.1 --port 8093 --deploy-config results/r1/full.yaml --enforce-eager --disable-log-stats" \
  --views 3 --concurrency 4 --arrivals burst --seed 20042 \
  --requests 1024 --warmup 20 --output results/r1/full.jsonl
```

Repeat the client with `--backend step`, the matching server command, and
`--output results/r1/step.jsonl`. Labels do not switch the running server.
For a separate quality run, add `--save-actions --requests 32` and use fresh
outputs; compare corresponding seeds and report max/mean absolute error.
FP32 parity uses `atol=1e-4, rtol=1e-5`; BF16 needs separate qualification.

## Measurement contract

Each wave drains before the next. Request latency includes packing, send,
inference, receive and unpack, but excludes connection setup. Throughput uses
the sum of wave durations, excluding observation creation and artifact writes.
Counts round up to full waves (e.g. 20 warmups become 24 at C=8). JSONL records
seeds, latency, failures and action hashes; any failure stops the run.
Use identical arguments across backends. For staggered waves, explicitly set
`--arrivals staggered --stagger-ms N` with the same positive N for both.

Record GPU/driver, package versions, checkpoint/tokenizer hashes and the exact
code revision/patch. Run without profiling and retain per-round results; at
least two A/B rounds are needed to report variation. A single pair is screening
evidence only. Collect telemetry in another terminal and stop it after the run:

```bash
nvidia-smi --query-gpu=timestamp,uuid,memory.used,utilization.gpu,power.draw \
  --format=csv --loop-ms=100 -f results/r1/full-gpu.csv
```

Sampled device memory is not the allocator peak. Never compare BF16 step results
against an FP32 baseline and attribute the difference solely to scheduling.
