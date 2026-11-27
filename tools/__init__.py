# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Repository-local developer and CI tools.

Making this directory a regular package prevents a dependency's top-level
``tools`` package from shadowing these modules when the repository root appears
first on ``sys.path``.
"""
