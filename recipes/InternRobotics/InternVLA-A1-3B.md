# InternVLA-A1-3B

> InternRobotics InternVLA-A1 robot VLA policy for offline open-loop action prediction

## Summary

- Vendor: InternRobotics
- Model: `Jia-Zeng/InternVLA-A1-3B-FineTuned-Place_Markpen`
- Task: Vision-Language-Action (VLA) robot action prediction
- Mode: Offline inference through the shared `observation_to_action` example
- Maintainer: Community

## When to use this recipe

Use this recipe when you need to run InternVLA-A1 on local robot observation
datasets and compare predicted action chunks against ground truth trajectories.
InternVLA-A1 inputs are pre-built robot observations (not text/image/video
prompts), so it uses the standard
[`observation_to_action`](../../examples/offline_inference/observation_to_action)
task example with model-specific hooks in `vllm_omni/model_extras/internvla_a1.py`.

## References

- Upstream notebook: <https://github.com/InternRobotics/InternVLA-A1/blob/master/tests/policies/internvla_a1_3b/open_loop_genie1_real.ipynb>
- Model checkpoint: <https://huggingface.co/Jia-Zeng/InternVLA-A1-3B-FineTuned-Place_Markpen>
- Dataset: <https://huggingface.co/datasets/InternRobotics/InternData-A1>
- Cosmos tokenizer checkpoints: <https://huggingface.co/tenstep/Cosmos-Tokenizer-CI8x8-SafeTensors>
- Offline example: [`examples/offline_inference/observation_to_action`](../../examples/offline_inference/observation_to_action)
- Pipeline: `vllm_omni.diffusion.models.internvla_a1.pipeline_internvla_a1.InternVLAA1Pipeline`
- Param contract: `vllm_omni/model_extras/internvla_a1.py`
- E2E test: [`tests/examples/offline_inference/test_internvla_a1.py`](../../tests/examples/offline_inference/test_internvla_a1.py)

## Hardware Support

## GPU

### 1x H200 141GB

#### Environment

- OS: Linux
- Python: 3.11+
- Driver / runtime: NVIDIA CUDA
- vLLM-Omni version or commit: use versions from your current checkout
- Checkpoints and dataset: local paths are required

#### Command

Download or prepare the model, processor, dataset, and Cosmos tokenizer files,
then export the local paths:

```bash
export INTERNVLA_A1_MODEL_DIR=/path/to/InternVLA-A1-3B-ft-pen
export INTERNVLA_A1_DATASET_DIR=/path/to/Genie1-Place_Markpen
export INTERNVLA_A1_PROCESSOR_DIR=/path/to/Qwen3-VL-2B-Instruct
export INTERNVLA_A1_COSMOS_DIR=/path/to/Cosmos-Tokenizer-CI8x8-SafeTensor
```

`INTERNVLA_A1_COSMOS_DIR` must contain `encoder.safetensors` and
`decoder.safetensors`.

Run one sample:

```bash
bash examples/offline_inference/observation_to_action/run.sh \
  --num-samples 1 \
  --num-episodes 0 \
  --dtype bfloat16 \
  --attn-implementation eager
```

Or call the shared script directly with `--extra-body`:

```bash
python examples/offline_inference/observation_to_action/observation_to_action.py \
  --model-class-name InternVLAA1Pipeline \
  --model-dir "$INTERNVLA_A1_MODEL_DIR" \
  --dataset-dir "$INTERNVLA_A1_DATASET_DIR" \
  --num-samples 1 \
  --num-episodes 0 \
  --extra-body '{"num_steps": 2}' \
  --dtype bfloat16 \
  --attn-implementation eager
```

Run a one-episode open-loop evaluation:

```bash
bash examples/offline_inference/observation_to_action/run.sh \
  --num-episodes 1 \
  --dtype bfloat16 \
  --attn-implementation eager
```

#### Verification

```bash
python -m pytest tests/examples/offline_inference/test_internvla_a1.py --collect-only
```

With local checkpoints and dataset available, run the gated e2e test:

```bash
python -m pytest -sv tests/examples/offline_inference/test_internvla_a1.py -m advanced_model
```

Expected artifacts include `summary.json`, `registry/log.json`, and optional
`registry/plots/*.jpg` under the selected output directory.

#### Notes

- Memory usage: reference collection used 1x H200 with `bfloat16` and eager attention.
- The shared example keeps tensor payloads (`batch_inputs`, `noise`) in
  `sampling_params.extra_args`; request-time knobs are routed through declared
  `extra_body` params.
- Declared knobs:
  - `num_steps`: optional request-time override for `config.num_inference_steps`
  - `decode_image`: returns decoded image features in `custom_output["decoded"]`
- Declared output:
  - `decoded`: present only when `decode_image` is enabled
- The full e2e path is gated behind `advanced_model` because it requires real
  local checkpoints, processor files, Cosmos tokenizer files, and dataset.
