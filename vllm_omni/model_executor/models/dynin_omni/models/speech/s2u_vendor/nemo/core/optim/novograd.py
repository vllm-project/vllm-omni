# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.optim.optimizer import Optimizer

__all__ = ['Novograd']


def _check_valid_opt_params(lr, eps, betas):
    if lr < 0:
        raise ValueError(f"Invalid learning rate: {lr}")
    if eps < 0:
        raise ValueError(f"Invalid epsilon value: {eps}")
    if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
        raise ValueError(f"Betas have to be between 0 and 1: {betas}")


class Novograd(Optimizer):
    """Implements Novograd algorithm.
    It has been proposed  in "Stochastic Gradient Methods with Layer-wise
    Adaptive Moments for Training of Deep Networks"
    (https://arxiv.org/abs/1905.11286)
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and Beyond"
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.95, 0.98),
        eps=1e-8,
        eps_in_sqrt=False,
        weight_decay=0,
        weight_decay_ema=True,
        grad_averaging=False,
        amsgrad=False,
        luc=False,
        luc_grad_trust=0.0,
        luc_grad_trust_rel=False,
        luc_trust=1e-3,
        luc_trust_min=0.0,
        luc_eps=1e-8,
        luc_update_min=1e-7,
        luc_update_max=1.0,
    ):
        _check_valid_opt_params(lr, eps, betas)
        assert isinstance(eps_in_sqrt, bool)
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, grad_averaging=grad_averaging, amsgrad=amsgrad,
            eps_in_sqrt=eps_in_sqrt,
            luc=luc,
            luc_grad_trust=luc_grad_trust,
            luc_grad_trust_rel=luc_grad_trust_rel,
            luc_trust=luc_trust,
            luc_trust_min=luc_trust_min,
            luc_eps=luc_eps,
            luc_update_min=luc_update_min,
            luc_update_max=luc_update_max,
            weight_decay_ema=weight_decay_ema
        )
        super(Novograd, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Novograd, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Sparse gradients are not supported.")

                amsgrad = group["amsgrad"]
                state = self.state[p]

                # State initialization
                if not state:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros([]).to(state["exp_avg"].device)
                    if amsgrad:
                        # Maintains max of all exp moving avg of squared grad
                        state["max_exp_avg_sq"] = torch.zeros([]).to(state["exp_avg"].device)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                if group['luc'] and group['luc_grad_trust'] > 0:
                    if not group['luc_grad_trust_rel']:
                        # Clip grad so that grad are less than eta*weights
                        luc_factor = get_luc_factor(p.data, grad, luc_trust=group['luc_grad_trust'], luc_trust_min=0.0)
                        grad.mul_(luc_factor)
                    else:
                        if exp_avg_sq != 0:
                            luc_factor = get_luc_factor(exp_avg_sq.sqrt(), grad, luc_trust=group['luc_grad_trust'], luc_trust_min=0.0)
                            grad.mul_(luc_factor)

                norm = grad.norm().pow(2)

                if exp_avg_sq == 0:
                    exp_avg_sq.copy_(norm)
                else:
                    exp_avg_sq.mul_(beta2).add_(norm, alpha=1.0 - beta2)

                if amsgrad:
                    # Maintains max of all 2nd moment running avg till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max for normalizing running avg. of gradient
                    if not group['eps_in_sqrt']:
                        denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                    else:
                        denom = max_exp_avg_sq.add_(group["eps"]).sqrt()
                else:
                    if not group['eps_in_sqrt']:
                        denom = exp_avg_sq.sqrt().add_(group["eps"])
                    else:
                        denom = exp_avg_sq.add_(group["eps"]).sqrt()

                grad.div_(denom)
                if group["weight_decay"] != 0 and group['weight_decay_ema']:
                    grad.add_(p.data, alpha=group["weight_decay"])
                if group["grad_averaging"]:
                    grad.mul_(1 - beta1)
                exp_avg.mul_(beta1).add_(grad)

                update = exp_avg
                if group["weight_decay"] != 0 and not group['weight_decay_ema']:
                    update = update.add(p.data, alpha=group["weight_decay"])

                lr = group["lr"]
                if group['luc'] and group['luc_trust'] > 0:
                    # Clip lr so that updates are less than eta*weights
                    luc_factor = get_luc_factor(p.data, update.data,  luc_trust=group['luc_trust'], luc_trust_min=group['luc_trust_min'])
                    lr = luc_factor * lr

                p.data.add_(update, alpha=-lr)

        return loss


def get_luc_factor(param, grad, *, luc_trust, luc_trust_min):
    param_norm = torch.norm(param)
    param_norm = max(param_norm, 1e-3)
    grad_norm = torch.norm(grad)
    if grad_norm == 0:
        return 1.0

    max_grad_norm = param_norm * luc_trust
    min_grad_norm = param_norm * luc_trust_min
    if grad_norm > max_grad_norm:
        luc_factor = max_grad_norm / grad_norm
    elif grad_norm < min_grad_norm:
        luc_factor = min_grad_norm / grad_norm
    else:
        luc_factor = 1.0

    return luc_factor
