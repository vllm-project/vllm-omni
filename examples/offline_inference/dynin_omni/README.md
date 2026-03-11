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

## 3. End2End Run Examples

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

## 4. Notes

- Outputs are saved under task-specific directories in `/tmp` by default.
- You can override output path with `--output-dir`.
- If you want to force local config resolution, pass `--dynin-config-path <PATH_TO_DYNIN_OMNI_YAML>`.
