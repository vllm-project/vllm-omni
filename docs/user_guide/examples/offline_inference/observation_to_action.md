# Observation-to-Action

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/observation_to_action>.

## Overview

This is the **standard offline task example** for robot / VLA policies that map
observations (cameras, proprioceptive state, language) to action chunks.

Model-specific logic is registered in `vllm_omni/model_extras/`:

- `eval_context_loader` — load dataset + config
- `observation_builder` — build `batch_inputs` + `noise`
- `action_processor` — truncate / summarize predicted actions
- `open_loop_runner` — optional open-loop GT evaluation
- `EXTRA_BODY_PARAMS` / `EXTRA_OUTPUT_PARAMS` — request / response knobs

The first consumer is **InternVLA-A1** (`InternVLAA1Pipeline`).

Recipe: [`recipes/InternRobotics/InternVLA-A1-3B.md`](../../../../recipes/InternRobotics/InternVLA-A1-3B.md)

## Setup (InternVLA-A1)

This example is adapted from:
https://github.com/InternRobotics/InternVLA-A1/blob/master/tests/policies/internvla_a1_3b/open_loop_genie1_real.ipynb

Export the required local paths:

```bash
# hf Jia-Zeng/InternVLA-A1-3B-FineTuned-Place_Markpen
export INTERNVLA_A1_MODEL_DIR=/path/to/InternVLA-A1-3B-ft-pen
# hf download InternRobotics/InternData-A1 real_lerobotv30/genie1/Genie1-Place_Markpen.tar.gz --repo-type dataset --local-dir /path/to/Genie1-Place_Markpen
export INTERNVLA_A1_DATASET_DIR=/path/to/Genie1-Place_Markpen
export INTERNVLA_A1_PROCESSOR_DIR=/path/to/Qwen3-VL-2B-Instruct
# hf tenstep/Cosmos-Tokenizer-CI8x8-SafeTensors
export INTERNVLA_A1_COSMOS_DIR=/path/to/Cosmos-Tokenizer-CI8x8-SafeTensor
```

`INTERNVLA_A1_COSMOS_DIR` is expected to contain:

- `encoder.safetensors`
- `decoder.safetensors`

## Run examples

```bash
cd examples/offline_inference/observation_to_action
```

### Run one sample

```bash
bash run.sh --num-samples 1 --num-episodes 0
```

Or directly:

```bash
python observation_to_action.py \
  --model-class-name InternVLAA1Pipeline \
  --model-dir "$INTERNVLA_A1_MODEL_DIR" \
  --dataset-dir "$INTERNVLA_A1_DATASET_DIR" \
  --num-samples 1 \
  --num-episodes 0 \
  --extra-body '{"num_steps": 2}'
```

### Run open-loop evaluation against GT

```bash
bash run.sh --num-episodes 1
```

Outputs are written under:

```bash
outputs/observation_to_action/vllm_infer/
```

Typical files:

- `summary.json`: top-level run summary
- `registry/log.json`: per-episode GT comparison metrics
- `registry/plots/*.jpg`: prediction-vs-GT figures

### Optional runtime switches

- `--dtype {bfloat16,float32}`: choose inference dtype
- `--attn-implementation {eager,sdpa}`: switch attention backend
- `--enable-regional-compile`: enable regional `torch.compile`
- `--enable-warmup`: run pipeline warmup in initialization
- `--num-steps N`: override `config.num_inference_steps` for this request
- `--decode-image`: request decoded image features in `custom_output["decoded"]`
- `--extra-body '{"num_steps": 2, "decode_image": true}'`: declared extras path (same knobs)
- `--skip-plots`: skip plot generation even if `matplotlib` is installed

InternVLA-A1 declares the request knobs `num_steps` and `decode_image` in
`vllm_omni/model_extras/internvla_a1.py`. Robot tensor payloads
(`batch_inputs`, `noise`) stay in `sampling_params.extra_args`; declared knobs
are routed through `apply_declared_extra_args`.

### Collect results and performance logs

```bash
bash collect_results.sh
```

### Benchmark forward latency

```bash
python observation_to_action.py \
  --model-class-name InternVLAA1Pipeline \
  --model-dir "$INTERNVLA_A1_MODEL_DIR" \
  --dataset-dir "$INTERNVLA_A1_DATASET_DIR" \
  --benchmark-forward \
  --dtype bfloat16 \
  --attn-implementation eager \
  --warmup-iters 3 \
  --benchmark-iters 10 \
  --output-dir outputs/observation_to_action/forward_benchmark
```

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

## FAQ

If `matplotlib` is missing, evaluation still runs and only plot generation is skipped:

```bash
pip install matplotlib
```

## Example materials

- `observation_to_action.py`: shared offline inference and GT evaluation entrypoint
- `run.sh`: shell wrapper with InternVLA local path env vars
- `collect_results.sh`: helper to gather result summaries and performance logs

The gated e2e test lives at
`tests/examples/offline_inference/test_internvla_a1.py`.

## Embedded source listings

??? abstract "collect_results.sh"
    ``````sh
    --8<-- "examples/offline_inference/observation_to_action/collect_results.sh"
    ``````
??? abstract "observation_to_action.py"
    ``````py
    --8<-- "examples/offline_inference/observation_to_action/observation_to_action.py"
    ``````
??? abstract "run.sh"
    ``````sh
    --8<-- "examples/offline_inference/observation_to_action/run.sh"
    ``````
