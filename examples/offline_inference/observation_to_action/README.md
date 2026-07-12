# Observation-to-Action

Full usage and result-reporting guidance lives in
[docs/user_guide/examples/offline_inference/observation_to_action.md](../../../docs/user_guide/examples/offline_inference/observation_to_action.md).

InternVLA-A1 recipe:
[recipes/InternRobotics/InternVLA-A1-3B.md](../../../recipes/InternRobotics/InternVLA-A1-3B.md)

Quick start (InternVLA-A1):

```bash
export INTERNVLA_A1_MODEL_DIR=/path/to/InternVLA-A1-3B-ft-pen
export INTERNVLA_A1_DATASET_DIR=/path/to/Genie1-Place_Markpen
export INTERNVLA_A1_PROCESSOR_DIR=/path/to/Qwen3-VL-2B-Instruct
# hf tenstep/Cosmos-Tokenizer-CI8x8-SafeTensors
export INTERNVLA_A1_COSMOS_DIR=/path/to/Cosmos-Tokenizer-CI8x8-SafeTensor

bash run.sh --num-samples 1 --num-episodes 0
bash run.sh --num-episodes 1
bash collect_results.sh
```

Request-time knobs for InternVLA-A1 (`vllm_omni/model_extras/internvla_a1.py`):

- `--num-steps N` or `--extra-body '{"num_steps": N}'`
- `--decode-image` or `--extra-body '{"decode_image": true}'`

Key entrypoints:

- `observation_to_action.py`: shared offline runner
- `run.sh`: env-gated wrapper for InternVLA-A1
- `collect_results.sh`: collect sample output, latency, metrics, plots, and logs

Gated e2e test:

```bash
python -m pytest -sv tests/examples/offline_inference/test_internvla_a1.py -m advanced_model
```
