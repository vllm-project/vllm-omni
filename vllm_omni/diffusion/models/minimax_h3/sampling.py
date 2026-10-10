# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compatibility aliases for the shared diffusion sampler."""

from vllm_omni.diffusion.sampler import ResMultistepSampler, Sampler, resolve_sampler_name

H3SampleSolver = Sampler
create_h3_sample_solver = Sampler
normalize_h3_sampler = resolve_sampler_name
res_multistep_coeffs = ResMultistepSampler.compute_coefficients

__all__ = ["H3SampleSolver", "create_h3_sample_solver", "normalize_h3_sampler", "res_multistep_coeffs"]
