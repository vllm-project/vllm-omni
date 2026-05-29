# Dynin-Omni Offline End2End Example

This folder contains a unified offline inference entrypoint:

- `end2end.py`

## 1. Environment Setup

Run from repository root:

```bash
cd <REPO_ROOT>
```

If needed, install this repo in editable mode:

```bash
pip install -e .
```

## 2. Extra Dependencies (EMOVA)

Install the following packages for EMOVA-related components:

```bash
pip install \
  "phonemizer==3.3.0" \
  "Unidecode==1.4.0" \
  "hydra-core==1.3.2" \
  "pytorch-lightning==1.1.0" \
  "wget==3.2" \
  "wrapt==2.1.1" \
  "onnx==1.20.1" \
  "frozendict==2.4.7" \
  "inflect==7.5.0" \
  "braceexpand==0.1.7" \
  "webdataset==1.0.2" \
  "torch-stft==0.1.4" \
  "editdistance==0.8.1"
```

## 3. Hardware and VRAM Requirements

This example uses a 3-stage pipeline on one GPU by default
([`dynin_omni.yaml`](../../../vllm_omni/model_executor/stage_configs/dynin_omni.yaml)):

- Stage-0 (`token2text`): `gpu_memory_utilization: 0.5`
- Stage-1 (`token2image`): `gpu_memory_utilization: 0.1`
- Stage-2 (`token2audio`): `gpu_memory_utilization: 0.1`

### Requested GPU Memory Budget from `gpu_memory_utilization`

| Stage | Utilization | A100 80GB | H200 141GB |
| :-- | :-- | :-- | :-- |
| Stage-0 (token2text) | 0.5 | ~40.0 GB | ~70.5 GB |
| Stage-1 (token2image) | 0.1 | ~8.0 GB | ~14.1 GB |
| Stage-2 (token2audio) | 0.1 | ~8.0 GB | ~14.1 GB |
| Total requested budget | 0.7 | ~56.0 GB | ~98.7 GB |

### Observed Runtime Signal (from your log)

- Stage-0 reported: `Model loading took 15.12 GiB memory` (weights footprint signal).
- Stages 1/2 can still add runtime memory depending on task path and backend allocations.
- Keep extra headroom for CUDA/PyTorch overhead and temporary allocations.

### GPU Compatibility

- Confirmed target GPUs for this setup: **NVIDIA H200**, **NVIDIA A100**.
- CI/e2e coverage in this repo also includes CUDA **L4** markers for Dynin tests.

## 4. End2End Run Examples

```bash
# t2t
python <REPO_ROOT>/examples/offline_inference/dynin_omni/end2end.py \
  --task t2t --model snu-aidas/Dynin-Omni --text <INSTRUCTION_TEXT>

# i2t
python <REPO_ROOT>/examples/offline_inference/dynin_omni/end2end.py \
  --task i2t --model snu-aidas/Dynin-Omni --image <IMAGE_PATH> --text "Please describe this image in detail."

# s2t
python <REPO_ROOT>/examples/offline_inference/dynin_omni/end2end.py \
  --task s2t --model snu-aidas/Dynin-Omni --audio <AUDIO_PATH> --text "Transcribe the given audio."

# t2i
python <REPO_ROOT>/examples/offline_inference/dynin_omni/end2end.py \
  --task t2i --model snu-aidas/Dynin-Omni --text <INSTRUCTION_TEXT>

# v2t
python <REPO_ROOT>/examples/offline_inference/dynin_omni/end2end.py \
  --task v2t --model snu-aidas/Dynin-Omni --video <VIDEO_PATH> --text "Describe this video in detail."

# i2i
python <REPO_ROOT>/examples/offline_inference/dynin_omni/end2end.py \
  --task i2i --model snu-aidas/Dynin-Omni --image <IMAGE_PATH> --text <INSTRUCTION_TEXT>

# t2s
python <REPO_ROOT>/examples/offline_inference/dynin_omni/end2end.py \
  --task t2s --model snu-aidas/Dynin-Omni --text <INSTRUCTION_TEXT>
```

## 5. Notes

- Outputs are saved under task-specific directories in `/tmp` by default.
- You can override output path with `--output-dir`.
- If you want to force local config resolution, pass `--dynin-config-path <PATH_TO_DYNIN_OMNI_YAML>`.
- If you see the warning
  `max_num_batched_tokens (32768) exceeds max_num_seqs * max_model_len (4096)`,
  reduce `max_num_batched_tokens` in stage config (for example, `4096` in CI config).
