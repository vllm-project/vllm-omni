# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Keep this package lightweight. The pipeline registry imports pipeline.py
# directly, while diffusion model classes are loaded lazily from their registry.
