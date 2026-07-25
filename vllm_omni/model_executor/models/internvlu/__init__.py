# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Model classes are loaded lazily through OmniModelRegistry.  Keeping this
# package initializer empty avoids importing CUDA-heavy dependencies during
# model discovery.
