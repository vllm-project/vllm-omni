# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Components shared by the Ming-family model packages.

Naming: The flow-matching pieces all use the cfm_ prefix,
because "flow matching", "fm" and "cfm" were three names for one concept here:

* dit_blocks    - DiT building blocks (DiTBlock, FinalLayer, CondEmbedder)
* cfm_dit       - the DiT used as the CFM velocity model
* cfm_solver    - timestep schedules and the ODE/SDE integrator
* cfm_head      - inference-only CFM wrapper preserving cfm.model.* names
* cfm_cudagraph - batch-1 CUDA-graph executor for the sampling loop
"""
