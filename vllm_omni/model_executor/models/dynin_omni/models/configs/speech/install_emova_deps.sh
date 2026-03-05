#!/usr/bin/env bash
set -euo pipefail

# Optional override:
#   PYTHON_BIN=/path/to/python ./install_emova_deps.sh
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m pip install \
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
  "editdistance==0.8.1"\
