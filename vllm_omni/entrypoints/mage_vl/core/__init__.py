# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Internal session scaffold for the standalone Mage-VL service.

Mage-VL is the only production consumer of these runtime, adapter, session,
and protocol helpers. Shared engine-resident duplex integrations use
``vllm_omni.engine.duplex``.
"""
