# InternVLA-A1

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/internvla_a1>.


## Setup
This example is adapted from: https://github.com/InternRobotics/InternVLA-A1/blob/master/tests/policies/internvla_a1_3b/open_loop_genie1_real.ipynb

This example runs the single-path vLLM offline inference workflow for InternVLA-A1 open-loop action prediction.

Before running the script, export the required local paths:

```bash
# hf Jia-Zeng/InternVLA-A1-3B-FineTuned-Place_Markpen
export INTERNVLA_A1_MODEL_DIR=/path/to/InternVLA-A1-3B-ft-pen
# hf download InternRobotics/InternData-A1 real_lerobotv30/genie1/Genie1-Place_Markpen.tar.gz --repo-type dataset --local-dir /path/to/Genie1-Place_Markpen
export INTERNVLA_A1_DATASET_DIR=/path/to/Genie1-Place_Markpen
export INTERNVLA_A1_PROCESSOR_DIR=/path/to/Qwen3-VL-2B-Instruct
# hf tenstep/Cosmos-Tokenizer-CI8x8-SafeTensors
export INTERNVLA_A1_COSMOS_DIR=/path/to/Cosmos-Tokenizer-CI8x8-SafeTensor
```

The shell entrypoint also accepts these variables as defaults, so you can keep the command line itself short.

`INTERNVLA_A1_COSMOS_DIR` is expected to contain:

- `encoder.safetensors`
- `decoder.safetensors`

Reference Hugging Face repo: `tenstep/Cosmos-Tokenizer-CI8x8-SafeTensors`

## Run examples

Get into the example folder:

```bash
cd examples/offline_inference/internvla_a1
```

### Run one sample

This runs a single sample through the maintained vLLM registry path.

```bash
bash run.sh --num-samples 1 --num-episodes 0
```

### Run open-loop evaluation against GT

This runs one episode, writes evaluation metrics, and saves prediction-vs-GT plots when `matplotlib` is available.

```bash
bash run.sh --num-episodes 1
```

Outputs are written under:

```bash
outputs/internvla_a1/vllm_infer/
```

Typical files:

- `summary.json`: top-level run summary
- `registry/log.json`: per-episode GT comparison metrics
- `registry/plots/*.jpg`: prediction-vs-GT figures

### Eager float32 validation

Use eager attention and float32 when you want the most stable numerical comparison baseline.

```bash
bash run.sh \
  --num-episodes 1 \
  --attn-implementation eager \
  --dtype float32
```

### Optional runtime switches

- `--dtype {bfloat16,float32}`: choose inference dtype
- `--attn-implementation {eager,sdpa}`: switch attention backend
- `--enable-regional-compile`: enable regional `torch.compile`
- `--enable-warmup`: run pipeline warmup in initialization
- `--skip-plots`: skip plot generation even if `matplotlib` is installed

### Collect results and performance logs

Use the helper below when you want to collect one-sample output, forward latency, GT comparison, plots, timing logs, and optional pytest output into a single result directory:

```bash
bash collect_results.sh
```

The script writes a timestamped directory under:

```bash
outputs/internvla_a1/collected_results/
```

Typical result files:

- `env_summary.txt`: environment and path summary
- `sample_run.log`: one-sample inference log
- `forward_benchmark/forward_latency.json`: isolated `pipeline.forward` latency summary
- `eval_run.log`: open-loop GT evaluation log
- `eval_outputs/summary.json`: top-level output summary
- `eval_outputs/registry/log.json`: GT comparison metrics
- `eval_outputs/registry/plots/*.jpg`: prediction-vs-GT figures
- `*_time.txt`: timing output
- `*_gpu.csv`: sampled GPU usage

### Benchmark forward latency

Use the dedicated mode below when you want the isolated `pipeline.forward` latency instead of end-to-end script time:

```bash
python end2end.py \
  --model-dir "$INTERNVLA_A1_MODEL_DIR" \
  --dataset-dir "$INTERNVLA_A1_DATASET_DIR" \
  --benchmark-forward \
  --dtype bfloat16 \
  --attn-implementation eager \
  --warmup-iters 3 \
  --benchmark-iters 10 \
  --output-dir outputs/internvla_a1/forward_benchmark
```

The output JSON is written to `outputs/internvla_a1/forward_benchmark/forward_latency.json` and contains:

- `cold_start_ms`
- `warmup_summary`
- `benchmark_summary`
- `benchmark_samples_ms`

### Reference results

Reference run collected on `1x NVIDIA H200`, `bfloat16`, `eager`:

- one-sample end-to-end run: `38s`
- one-episode GT evaluation run: `45s`
- `average_mse = 1.7173260857816786e-05`
- `average_mae = 0.0011860118247568607`
- `average_mse_joint = 7.42028441891307e-06`
- `average_mae_joint = 0.0010777723509818316`
- `average_mse_gripper = 8.544408774469048e-05`
- `average_mae_gripper = 0.0019436875591054559`

The reference plot filename is:

- `eval_outputs/registry/plots/vllm_registry_open_loop_ep0.jpg`

## Measure latency and VRAM

You can collect end-to-end runtime and peak host memory with:

```bash
/usr/bin/time -v bash run.sh --num-episodes 1
```

For GPU memory snapshots, use:

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
```

Run `nvidia-smi` before and during inference to record the peak device memory for your PR report.

## FAQ

If `matplotlib` is missing, evaluation still runs and only plot generation is skipped. To enable plots:

```bash
pip install matplotlib
```

## Example materials

- `end2end.py`: main offline inference and GT evaluation entrypoint
- `run.sh`: shell wrapper with local path env vars
- `collect_results.sh`: helper to gather result summaries and performance logs into one directory
- `internvla_a1_common.py`: dataset, evaluation, and plotting helpers

## Embedded source listings

??? abstract "collect_results.sh"
    ``````sh
    --8<-- "examples/offline_inference/internvla_a1/collect_results.sh"
    ``````
??? abstract "end2end.py"
    ``````py
    --8<-- "examples/offline_inference/internvla_a1/end2end.py"
    ``````
??? abstract "internvla_a1_common.py"
    ``````py
    --8<-- "examples/offline_inference/internvla_a1/internvla_a1_common.py"
    ``````
??? abstract "run.sh"
    ``````sh
    --8<-- "examples/offline_inference/internvla_a1/run.sh"
    ``````
??? abstract "standalone/outputs/qwena1/standalone_infer/log.json"
    ``````json
    --8<-- "examples/offline_inference/internvla_a1/standalone/outputs/qwena1/standalone_infer/log.json"
    ``````
