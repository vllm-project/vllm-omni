# Apertus Offline End-to-End Example

This example runs text+image inference with vLLM-Omni for `ApertusForCausalLM` using the Omni pipeline.

## Setup

From repo root:

```bash
cd vllm-omni
python3 -m pip install -e . --no-build-isolation --no-deps
```

## Run End-to-End

From repo root:

```bash
python3 examples/offline_inference/apertus/end2end.py \
  --model /capstor/store/cscs/swissai/infra01/MLLM/ablations/apertus-8b-img-SFT-32nodes-gbs512-mbs1-steps8030-img-text-seqlen8192-s2onlytxtloss/HF \
  --stage-configs-path vllm_omni/model_executor/stage_configs/apertus.yaml \
  --prompt "Describe the image briefly: <|image|>" \
  --max-tokens 64 \
  --temperature 0.0 \
  --top-p 1.0 \
  --top-k -1 \
  --emu-checkpoint BAAI/Emu3.5-VisionTokenizer \
  --emu-device cuda:0 \
  --emu-dtype bfloat16
```

If `--image-path` is not provided, the script uses a synthetic image for smoke testing.

To use a real image:

```bash
python3 examples/offline_inference/apertus/end2end.py \
  --model /capstor/store/cscs/swissai/infra01/MLLM/ablations/apertus-8b-img-SFT-32nodes-gbs512-mbs1-steps8030-img-text-seqlen8192-s2onlytxtloss/HF \
  --stage-configs-path vllm_omni/model_executor/stage_configs/apertus.yaml \
  --image-path /path/to/image.jpg \
  --prompt "What is in this image? <|image|>" \
  --emu-checkpoint BAAI/Emu3.5-VisionTokenizer \
  --emu-device cuda:0 \
  --emu-dtype bfloat16
```

## Run Tests

From repo root:

```bash
python3 -m pytest -q -o addopts='' \
  tests/entrypoints/test_apertus_input_preprocessor.py \
  tests/entrypoints/test_apertus_stage_adapter.py
```

If your environment has the coverage plugins configured in `pyproject.toml`,
you can run without `-o addopts=''`.

## Notes

- `vllm_omni/model_executor/stage_configs/apertus.yaml` uses `model_stage: generation` for readability.
- The stage config uses `trust_remote_code: false` to keep inference on the native vLLM Apertus path.
- EMU tokenizer runs on GPU when `--emu-device cuda:0` is set.
