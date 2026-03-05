# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Keep model package import lightweight for EMOVA speech tokenizer runtime.
# Importing the full NeMo model zoo here triggers optional dependency chains
# that are not required for the spec2vec tokenizer path.
__all__ = [
    "asr_model",
    "classification_models",
    "ctc_bpe_models",
    "ctc_models",
    "label_models",
    "rnnt_bpe_models",
    "rnnt_models",
    "spec2vec",
    "st2vec",
    "wav2vec",
]
