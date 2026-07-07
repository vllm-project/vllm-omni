# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer B upstream-parity test: FlowUniPCMultistepScheduler in-tree vs upstream.

Requires _test_dev/_hidream_o1_upstream_fm_solvers_unipc.py (vendor from
HiDream-ai/HiDream-O1-Image @21bcd30471ac / models/fm_solvers_unipc.py).

Usage:
    python _test_dev/_test_hidream_o1_scheduler_upstream_parity.py
"""
from __future__ import annotations

import importlib.util
import os

import torch

from vllm_omni.diffusion.models.schedulers import (
    FlowUniPCMultistepScheduler as InTreeScheduler,
)

# load upstream reference by absolute path (robust: no sys.path / namespace package needed)
_here = os.path.dirname(os.path.abspath(__file__))
_upstream_file = os.path.join(_here, '_hidream_o1_upstream_fm_solvers_unipc.py')
if not os.path.isfile(_upstream_file):
    raise FileNotFoundError(
        f'upstream reference missing at {_upstream_file!r}: refetch models/fm_solvers_unipc.py '
        f'from https://github.com/HiDream-ai/HiDream-O1-Image (SHA 21bcd30471ac) into that path'
    )
_spec = importlib.util.spec_from_file_location('_hidream_o1_upstream_fm_solvers_unipc', _upstream_file)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
UpstreamScheduler = _mod.FlowUniPCMultistepScheduler
# --- end upstream load ---

UPSTREAM_SHA = '21bcd30471ac69d6a7134db293eefbc10e795c3f'

# HiDream-O1 config (upstream models/pipeline.py::build_scheduler default branch)
NUM_INFERENCE_STEPS, SHIFT, NUM_TRAIN_TIMESTEPS, PREDICTION_TYPE = 50, 3.0, 1000, 'flow_prediction'

# B2/B3 test tensor shape (small enough for fast run, exercises multistep state)
BATCH, SEQ_LEN, PATCH_DIM = 1, 100, 3 * 32 * 32
SEED = 0

# predeclared engineering tolerances
TOL_B1B, TOL_B2, TOL_B3 = 1e-7, 1e-6, 1e-4


def build_pair(device):
    up = UpstreamScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS, shift=SHIFT, use_dynamic_shifting=False, prediction_type=PREDICTION_TYPE)
    it = InTreeScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS, shift=SHIFT, use_dynamic_shifting=False, prediction_type=PREDICTION_TYPE)
    up.set_timesteps(NUM_INFERENCE_STEPS, device=device)
    it.set_timesteps(NUM_INFERENCE_STEPS, device=device)
    return up, it


def diff_stats(a, b):
    d = (a - b).abs()
    return d.max().item(), d.mean().item(), max(a.abs().max().item(), b.abs().max().item())


b2_b3_device = 'cuda' if torch.cuda.is_available() else 'cpu'

# B1a: config semantics — timesteps exact integer equality (shape + dtype + torch.equal three-item combo; torch.equal alone is not a bit-pattern check because it treats +0.0 == -0.0)
up, it = build_pair(device='cpu')
shape_eq = up.timesteps.shape == it.timesteps.shape
dtype_eq = up.timesteps.dtype == it.timesteps.dtype == torch.int64
equal = torch.equal(up.timesteps, it.timesteps)
print(f'B1a[timesteps]           : shape={tuple(up.timesteps.shape)} dtype={up.timesteps.dtype} shape_eq={shape_eq} dtype_eq={dtype_eq} torch_equal={equal}')
assert shape_eq and dtype_eq and equal, 'B1a: timesteps not exactly integer-equal (upstream vs in-tree)'

# B1b: config semantics — sigmas float within TOL_B1B (cast to fp64 for measurement; fp32 machine epsilon ~1.2e-7 near tolerance)
up, it = build_pair(device='cpu')
b1b_max, b1b_mean, b1b_out = diff_stats(up.sigmas.to(torch.float64), it.sigmas.to(torch.float64))
print(f'B1b[sigmas]              : shape={tuple(up.sigmas.shape)} max_abs_diff={b1b_max:.3e} mean_abs_diff={b1b_mean:.3e} output_abs_max={b1b_out:.3e} (tol {TOL_B1B:.0e})')
assert b1b_max < TOL_B1B, f'B1b: sigmas max_abs_diff={b1b_max:.3e} >= tol {TOL_B1B:.0e}'

# B2: single-step step() — one fresh pair, one step, same random inputs
# (do NOT manually call _init_step_index; upstream fm_solvers_unipc.py:685 and
#  in-tree scheduling_flow_unipc_multistep.py:635 both auto-init inside step())
up, it = build_pair(device=b2_b3_device)
g = torch.Generator(device='cpu').manual_seed(SEED)
z0 = torch.randn(BATCH, SEQ_LEN, PATCH_DIM, generator=g, dtype=torch.float32).to(b2_b3_device)
m0 = torch.randn(BATCH, SEQ_LEN, PATCH_DIM, generator=g, dtype=torch.float32).to(b2_b3_device)
z_up = up.step(m0.clone(), up.timesteps[0], z0.clone(), return_dict=False)[0]
z_it = it.step(m0.clone(), it.timesteps[0], z0.clone(), return_dict=False)[0]
b2_max, b2_mean, b2_out = diff_stats(z_up, z_it)
print(f'B2 [single-step step()]  : device={b2_b3_device} max_abs_diff={b2_max:.3e} mean_abs_diff={b2_mean:.3e} output_abs_max={b2_out:.3e} (tol {TOL_B2:.0e})')
assert b2_max < TOL_B2, f'B2: step() max_abs_diff={b2_max:.3e} >= tol {TOL_B2:.0e}'

# B3: 50-step stateful rollout — ONE fresh pair AT START (not per step), pre-generated identical
# model_outputs shared by both branches; rollout_max_abs_diff = max_i step_max (worst step, not last)
up, it = build_pair(device=b2_b3_device)
g = torch.Generator(device='cpu').manual_seed(SEED)
z0 = torch.randn(BATCH, SEQ_LEN, PATCH_DIM, generator=g, dtype=torch.float32).to(b2_b3_device)
model_outputs = [torch.randn(BATCH, SEQ_LEN, PATCH_DIM, generator=g, dtype=torch.float32).to(b2_b3_device) for _ in range(NUM_INFERENCE_STEPS)]
z_up, z_it = z0.clone(), z0.clone()
b3_max, b3_mean_sum, b3_out = 0.0, 0.0, 0.0
for i in range(NUM_INFERENCE_STEPS):
    m = model_outputs[i]
    z_up = up.step(m.clone(), up.timesteps[i], z_up, return_dict=False)[0]
    z_it = it.step(m.clone(), it.timesteps[i], z_it, return_dict=False)[0]
    step_max, step_mean, step_out = diff_stats(z_up, z_it)
    if step_max > b3_max: b3_max = step_max
    b3_mean_sum += step_mean
    if step_out > b3_out: b3_out = step_out
b3_mean = b3_mean_sum / NUM_INFERENCE_STEPS
print(f'B3 [50-step rollout]     : device={b2_b3_device} rollout_max_abs_diff={b3_max:.3e} rollout_mean_abs_diff={b3_mean:.3e} rollout_output_abs_max={b3_out:.3e} (tol {TOL_B3:.0e})')
assert b3_max < TOL_B3, f'B3: rollout_max_abs_diff={b3_max:.3e} >= tol {TOL_B3:.0e}'

if b2_b3_device == 'cpu':
    print('warn                     : CUDA unavailable — B2/B3 ran on CPU, .to(device) engineering difference not stress-tested')

print(f'pass (upstream @{UPSTREAM_SHA[:12]} vs in-tree numerically agree within predeclared tolerances on the declared test scope: B1a integer-equal, B1b sigmas < 1e-7, B2 single step < 1e-6, B3 50-step rollout < 1e-4)')


# output:
# B1a[timesteps]           : shape=(50,) dtype=torch.int64 shape_eq=True dtype_eq=True torch_equal=True
# B1b[sigmas]              : shape=(51,) max_abs_diff=0.000e+00 mean_abs_diff=0.000e+00 output_abs_max=9.999e-01 (tol 1e-07)
# B2 [single-step step()]  : device=cuda max_abs_diff=0.000e+00 mean_abs_diff=0.000e+00 output_abs_max=4.666e+00 (tol 1e-06)
# B3 [50-step rollout]     : device=cuda rollout_max_abs_diff=5.960e-06 rollout_mean_abs_diff=4.533e-07 rollout_output_abs_max=4.820e+00 (tol 1e-04)
# pass (upstream @21bcd30471ac vs in-tree numerically agree within predeclared tolerances on the declared test scope: B1a integer-equal, B1b sigmas < 1e-7, B2 single step < 1e-6, B3 50-step rollout < 1e-4)
