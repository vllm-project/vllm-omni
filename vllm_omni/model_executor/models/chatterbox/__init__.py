# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# NOTE: Do not import model classes in this file. Importing any
# submodule in this package (the pipeline registry imports ``pipeline``)
# executes it, and the model registry resolves the class from its own
# module, so a re-export here would only load both backbones, S3Gen and
# the samplers whenever another model is served.
