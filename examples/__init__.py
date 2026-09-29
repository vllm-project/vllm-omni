# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""vLLM-Omni example scripts, importable as a package.

Keep this ``__init__.py``: with the upstream vLLM installed editable (which also
places an ``examples`` package on ``sys.path``), a regular package at this repo
root wins over the upstream ``examples`` namespace during ``import examples``.
"""
