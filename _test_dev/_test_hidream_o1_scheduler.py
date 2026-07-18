# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer A test: build_hidream_o1_scheduler config semantics.

Class-vs-class parity lives in _test_hidream_o1_scheduler_upstream_parity.py.

Usage:
    python _test_dev/_test_hidream_o1_scheduler.py
"""
from __future__ import annotations

import torch

from vllm_omni.diffusion.models.hidream_o1_image.pipeline_hidream_o1_image import (
    build_hidream_o1_scheduler,
)


# A1: default recipe config (canonical: shift=3.0, num_inference_steps=50)
sched = build_hidream_o1_scheduler(num_inference_steps=50, shift=3.0, device='cpu')
type_ok = type(sched).__name__ == 'FlowUniPCMultistepScheduler'
shift_ok = sched.config.shift == 3.0
n_train_ok = sched.config.num_train_timesteps == 1000
pred_ok = sched.config.prediction_type == 'flow_prediction'
dyn_ok = sched.config.use_dynamic_shifting is False
predict_x0_ok = sched.config.predict_x0 is True
solver_order_ok = sched.config.solver_order == 2
n_inf_ok = sched.num_inference_steps == 50
t_dtype_ok = sched.timesteps.dtype == torch.int64
t_shape_ok = sched.timesteps.shape == (50,)
s_shape_ok = sched.sigmas.shape == (51,)
s_final_ok = sched.sigmas[-1].item() == 0.0
print(f'A1[default recipe config]      : type_ok={type_ok} shift_ok={shift_ok} n_train_ok={n_train_ok} pred_ok={pred_ok} dyn_ok={dyn_ok} predict_x0_ok={predict_x0_ok} solver_order_ok={solver_order_ok} n_inf_ok={n_inf_ok} timesteps.dtype={sched.timesteps.dtype} timesteps.shape={tuple(sched.timesteps.shape)} sigmas.shape={tuple(sched.sigmas.shape)} sigmas[-1]={sched.sigmas[-1].item()}')
assert type_ok and shift_ok and n_train_ok and pred_ok and dyn_ok and predict_x0_ok and solver_order_ok and n_inf_ok, 'A1: config semantics mismatch'
assert t_dtype_ok and t_shape_ok and s_shape_ok and s_final_ok, 'A1: timesteps/sigmas shape or dtype mismatch'

# A2a: scheduler_name='flash' raises NotImplementedError with informative message
raised_flash = False
msg_flash = ''
try:
    build_hidream_o1_scheduler(scheduler_name='flash', num_inference_steps=28, shift=1.0)
except NotImplementedError as e:
    raised_flash = True
    msg_flash = str(e)
msg_flash_ok = msg_flash == "unsupported scheduler_name='flash'"
print(f"A2a[scheduler_name='flash' guard] : raised={raised_flash} msg_ok={msg_flash_ok} msg={msg_flash!r}")
assert raised_flash, "A2a: scheduler_name='flash' should raise NotImplementedError"
assert msg_flash_ok, f"A2a: unexpected NotImplementedError message {msg_flash!r}"

# A2b: scheduler_name='flow_match' raises NotImplementedError
raised_fm = False
try:
    build_hidream_o1_scheduler(scheduler_name='flow_match', num_inference_steps=28, shift=1.0)
except NotImplementedError:
    raised_fm = True
print(f"A2b[scheduler_name='flow_match' guard] : raised={raised_fm}")
assert raised_fm, "A2b: scheduler_name='flow_match' should raise NotImplementedError"

# A3: timesteps_list override (verbatim from upstream pipeline.py 28-step list)
DEFAULT_TIMESTEPS_28 = [999, 987, 974, 960, 945, 929, 913, 895, 877, 857,
                       836, 814, 790, 764, 737, 707, 675, 640, 602, 560,
                       515, 464, 409, 347, 278, 199, 110, 8]
sched = build_hidream_o1_scheduler(num_inference_steps=28, shift=1.0, timesteps_list=DEFAULT_TIMESTEPS_28)
expected_t = torch.tensor(DEFAULT_TIMESTEPS_28, dtype=torch.long)
t_shape_eq = sched.timesteps.shape == expected_t.shape
t_dtype_eq = sched.timesteps.dtype == expected_t.dtype == torch.int64
t_equal = torch.equal(sched.timesteps, expected_t)
s_shape_ok = sched.sigmas.shape == (29,)
s_final_ok = sched.sigmas[-1].item() == 0.0
sigma_max_diff = max(abs(sched.sigmas[i].item() - t / 1000.0) for i, t in enumerate(DEFAULT_TIMESTEPS_28))
print(f'A3[timesteps_list override]    : shape_eq={t_shape_eq} dtype_eq={t_dtype_eq} torch_equal={t_equal} sigmas.shape={tuple(sched.sigmas.shape)} sigmas[-1]={sched.sigmas[-1].item()} sigmas_max_diff_vs_t/1000={sigma_max_diff:.3e} (tol 1e-06)')
assert t_shape_eq and t_dtype_eq and t_equal, 'A3: timesteps override not exactly integer-equal to the list'
assert s_shape_ok and s_final_ok, 'A3: sigmas shape or final-zero convention broken'
assert sigma_max_diff < 1e-6, f'A3: sigmas[i] != timesteps_list[i]/1000 within 1e-6 (max diff {sigma_max_diff:.3e})'

# A4: 4-step scheduler wired correctly: set_timesteps + step() round-trip smoke (no NaN/Inf, shape preserved)
sched = build_hidream_o1_scheduler(num_inference_steps=4, shift=3.0, device='cpu')
torch.manual_seed(0)
PATCH_DIM, SEQ_LEN = 3 * 32 * 32, 100
z = torch.randn(1, SEQ_LEN, PATCH_DIM, dtype=torch.float32)
n_iter = 0
for step_t in sched.timesteps:
    model_output = torch.randn_like(z)
    z_next = sched.step(model_output, step_t, z, return_dict=False)[0]
    assert z_next.shape == z.shape, f'A4: step {n_iter} shape drift {z_next.shape} != {z.shape}'
    assert not torch.isnan(z_next).any(), f'A4: step {n_iter} produced NaN'
    assert not torch.isinf(z_next).any(), f'A4: step {n_iter} produced Inf'
    z = z_next
    n_iter += 1
z_abs_max = z.abs().max().item()
print(f'A4[set_timesteps + step smoke]  : ran={n_iter} steps shape={tuple(z.shape)} final_abs_max={z_abs_max:.3e} finite=True')
assert n_iter == 4, f'A4: expected 4 iterations, got {n_iter}'

print('pass (build_hidream_o1_scheduler config semantics match HiDream-O1 upstream default recipe branch; unsupported alternate scheduler names raise; timesteps_list override + step smoke correct)')


# output:
# A1[default recipe config]      : type_ok=True shift_ok=True n_train_ok=True pred_ok=True dyn_ok=True predict_x0_ok=True solver_order_ok=True n_inf_ok=True timesteps.dtype=torch.int64 timesteps.shape=(50,) sigmas.shape=(51,) sigmas[-1]=0.0
# A2a[scheduler_name='flash' guard] : raised=True msg_ok=True msg="unsupported scheduler_name='flash'"
# A2b[scheduler_name='flow_match' guard] : raised=True
# A3[timesteps_list override]    : shape_eq=True dtype_eq=True torch_equal=True sigmas.shape=(29,) sigmas[-1]=0.0 sigmas_max_diff_vs_t/1000=2.575e-08 (tol 1e-06)
# A4[set_timesteps + step smoke]  : ran=4 steps shape=(1, 100, 3072) final_abs_max=5.848e+00 finite=True
# pass (build_hidream_o1_scheduler config semantics match HiDream-O1 upstream default recipe branch; unsupported alternate scheduler names raise; timesteps_list override + step smoke correct)
