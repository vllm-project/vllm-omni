# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-neutral helpers shared between model packages.

``audio`` and ``request_outputs`` carry no duplex vocabulary and may be used
from any model. ``duplex`` builds on the duplex framework contracts and is
imported only from a model's ``duplex`` package.
"""
