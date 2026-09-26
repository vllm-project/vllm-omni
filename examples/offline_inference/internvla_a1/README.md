# InternVLA-A1

Full usage and result-reporting guidance lives in [docs/user_guide/examples/offline_inference/internvla_a1.md](../../../docs/user_guide/examples/offline_inference/internvla_a1.md).

Quick start:

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

Expected files under `INTERNVLA_A1_COSMOS_DIR`:

- `encoder.safetensors`
- `decoder.safetensors`

Reference Hugging Face repo: `tenstep/Cosmos-Tokenizer-CI8x8-SafeTensors`

Key entrypoints:

- `run.sh`: wrapper for offline inference and GT evaluation
- `collect_results.sh`: collect sample output, latency, metrics, plots, and logs
- `end2end.py`: underlying Python entrypoint
