# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Validation-only tracing; no production defaults or unconditional cache hits."""

import json
import os
from pathlib import Path

import numpy as np
import torch

from vllm_omni.diffusion.cache.teacache import backend
from vllm_omni.diffusion.cache.teacache.hook import TeaCacheHook
from vllm_omni.diffusion.distributed.parallel_state import get_classifier_free_guidance_rank, get_pp_group
from vllm_omni.diffusion.hooks import HookRegistry
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import Wan22Pipeline


class TracedHook(TeaCacheHook):
    def initialize_hook(self, module):
        result = super().initialize_hook(module)
        self.previous = {}
        self.branch_states = {}
        self.request_index = 0
        self.pp_rank = get_pp_group().rank_in_group
        self.cfg_rank = get_classifier_free_guidance_rank()
        self.mode = os.environ["WAN_TRACE_MODE"]
        self.root = Path(os.environ["WAN_TRACE_DIR"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.log = self.root / f"rank-{torch.distributed.get_rank()}.jsonl"
        if self.mode == "cache":
            coefficients = json.loads(Path(os.environ["WAN_COEFFICIENTS"]).read_text())
            polynomial = np.poly1d(coefficients[str(self.pp_rank)])
            fit = json.loads(Path(os.environ["WAN_COEFFICIENTS"]).with_suffix(".fit.json").read_text())
            upper = fit[str(self.pp_rank)]["x_max"]
            self.rescale_func = lambda distance: (
                self.config.rel_l1_thresh + 1 if distance > upper else max(0.0, float(polynomial(distance)))
            )
        extractor = self.extractor_fn

        def extract(module, *args, **kwargs):
            ctx = extractor(module, *args, **kwargs)
            branch = self._explicit_cfg_branch(module, ctx)
            if branch is None:
                raise AssertionError("Wan trace requires explicit CFG branch")
            run = ctx.run_transformer_blocks

            def measured():
                output = run()
                if self.mode == "collect":
                    residual = output[0] - ctx.hidden_states
                    previous = self.previous.get(branch)
                    row = {
                        "branch": branch,
                        "pp_rank": self.pp_rank,
                        "cfg_rank": self.cfg_rank,
                        "request": self.request_index,
                        "kind": "calibration",
                    }
                    if previous is not None:
                        mod, res = previous
                        row["input_distance"] = (
                            (ctx.modulated_input.float() - mod.float()).abs().mean() / (mod.float().abs().mean() + 1e-8)
                        ).item()
                        row["residual_distance"] = (
                            (residual.float() - res.float()).abs().mean() / (res.float().abs().mean() + 1e-8)
                        ).item()
                        self.write(row)
                    self.previous[branch] = (ctx.modulated_input.detach().clone(), residual.detach().clone())
                return output

            ctx.run_transformer_blocks = measured
            return ctx

        self.extractor_fn = extract
        return result

    def new_forward(self, module, *args, **kwargs):
        if self.mode == "none":
            return module._omni_original_forward(*args, **kwargs)
        return super().new_forward(module, *args, **kwargs)

    def write(self, row):
        row["request_name"] = getattr(self, "request_name", "engine-warmup")
        with self.log.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def _should_compute_full_transformer(self, state, modulated):
        branch = self.state_manager._current_context
        assert all(other is not state for name, other in self.branch_states.items() if name != branch)
        self.branch_states[branch] = state
        if state.cnt == 0:
            assert state.previous_residual is None, "Residual leaked across requests"
        if state.cnt < getattr(self, "cache_warmup_steps", 0):
            state.accumulated_rel_l1_distance = 0.0
            compute = True
        else:
            compute = (
                True if self.mode in ("collect", "full") else super()._should_compute_full_transformer(state, modulated)
            )
        self.write(
            {
                "kind": "decision",
                "request": self.request_index,
                "step": state.cnt,
                "compute": compute,
                "pp_rank": self.pp_rank,
                "cfg_rank": self.cfg_rank,
                "context": self.state_manager._current_context,
            }
        )
        return compute

    def reset_state(self, module):
        super().reset_state(module)
        self.previous.clear()
        self.branch_states.clear()
        self.request_index += 1
        return module


def apply(module, config):
    if config.transformer_type == "WanTransformer3DModel":
        HookRegistry.get_or_create(module).register_hook("teacache", TracedHook(config))
    else:
        original_apply(module, config)


original_apply = backend.apply_teacache_hook
backend.apply_teacache_hook = apply

# Capture final denoising latents before VAE scaling/decoding for numerical audit.

_original_diffuse = Wan22Pipeline.diffuse


def traced_diffuse(self, *args, **kwargs):
    control = None
    if os.environ.get("WAN_CONTROL"):
        control_path = Path(os.environ["WAN_CONTROL"])
        control = (
            json.loads(control_path.read_text())
            if control_path.exists()
            else {"mode": os.environ["WAN_TRACE_MODE"], "name": "engine-warmup"}
        )
        hook = HookRegistry.get_or_create(self.transformer).get_hook("teacache")
        hook.mode = control["mode"]
        hook.cache_warmup_steps = control.get("cache_warmup_steps", 0)
        hook.request_name = control["name"]
        hook.log = Path(os.environ["WAN_TRACE_DIR"]) / control["mode"] / f"rank-{torch.distributed.get_rank()}.jsonl"
        hook.log.parent.mkdir(parents=True, exist_ok=True)
    latent = _original_diffuse(self, *args, **kwargs)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank == 0:
        if not isinstance(latent, torch.Tensor) and hasattr(latent, "resolve"):
            latent = latent.resolve()
        if not isinstance(latent, torch.Tensor):
            raise AssertionError(f"Expected final latent tensor, got {type(latent)}")
        root = Path(os.environ["WAN_TRACE_DIR"])
        root.mkdir(parents=True, exist_ok=True)
        idx = getattr(self, "_validation_request_index", 0)
        target = (
            root / f"latent-{idx:03d}.pt" if control is None else root / control["mode"] / (control["name"] + ".pt")
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(latent.detach().cpu(), target)
        self._validation_request_index = idx + 1
    return latent


Wan22Pipeline.diffuse = traced_diffuse
