# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# NOTE: Do not import model classes in this file. Importing any
# submodule in this package triggers __init__.py execution, and
# both the model registry and pipeline registry import submodules
# directly — heavy imports here would be loaded as a side effect
# even though nothing depends on these re-exports.
