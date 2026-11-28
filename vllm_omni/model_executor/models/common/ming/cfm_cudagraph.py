# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDAGraph-accelerated CFM (flow-matching) diffusion head for Ming-TTS.

The per-AR-step flow head runs ``steps`` ODE iterations, each a tiny DiT forward
(history+patch tokens, CFG cond+null batched). In eager mode these are dozens of
micro kernels per step and the wall time is dominated by kernel-launch / Python
overhead, not GPU math. Capturing the whole sampling loop + Aggregator + stop
head into one CUDA graph removes that overhead.

Each executor is captured for one batch bucket and cached by the Stage-0 model.
Smaller live batches are zero-padded before replay and sliced back afterwards.
Unlike a streaming codec, this CFM head has no persistent per-row state, so
padding rows do not need state slots or a validity mask.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm.platforms import current_platform

from vllm_omni.model_executor.models.common.ming.cfm_dit import DiT
from vllm_omni.model_executor.models.common.ming.cfm_kernels import (
    cfm_triton_supported,
    fused_cfm_euler_sde_step,
)
from vllm_omni.model_executor.models.common.ming.cfm_solver import DEFAULT_CFM_STEPS, build_timesteps


class CFMSampler(nn.Module):
    """Unrolled CFM ODE solver, graph-capturable (no ``torch.randn``/``.item()``).

    All randomness (``y0``, ``sde_rnd``) and the timestep schedule ``t`` are
    explicit inputs. Integration follows the dtype of the checkpoint's noise.
    """

    def __init__(self, dit_model: nn.Module, steps: int = DEFAULT_CFM_STEPS, *, include_final_noise: bool = False):
        super().__init__()
        self.model = dit_model
        self.steps = steps
        self.include_final_noise = include_final_noise

    def supports_fast_path_for(self, llm_cond: torch.Tensor) -> bool:
        """Return True when this sampler's DiT model can run the CUDA Triton fast path."""
        return (
            isinstance(self.model, DiT)
            and not self.model.training
            and llm_cond.dtype in (torch.bfloat16, torch.float16)
            and cfm_triton_supported(llm_cond)
            and not any(block.attn.native_attention for block in self.model.blocks)
        )

    def _can_use_fast_path(self, llm_cond: torch.Tensor, lat_cond: torch.Tensor, y0: torch.Tensor) -> bool:
        return (
            self.supports_fast_path_for(llm_cond)
            and llm_cond.ndim == 3
            and lat_cond.ndim == 3
            and y0.ndim == 3
            and llm_cond.shape[0] == lat_cond.shape[0] == y0.shape[0]
            and y0.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and cfm_triton_supported(y0)
        )

    @torch.no_grad()
    def forward_native(self, llm_cond, lat_cond, y0, t, sde_args, sde_rnd):
        model_dtype = llm_cond.dtype
        for step in range(self.steps):
            dt = t[step + 1] - t[step]
            pred_cfg = self.model.forward_with_cfg(
                x=y0.to(model_dtype),
                t=t[step].to(model_dtype),
                c=llm_cond,
                latent_history=lat_cond,
                cfg_scale=sde_args[0],
                patch_size=y0.shape[1],
            )
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            flow = (pred + (pred - null_pred) * sde_args[0]).to(y0.dtype)
            y0 = y0 + flow * dt
            if self.include_final_noise or step < self.steps - 1:
                y0 = y0 + sde_args[1] * (sde_args[2] ** 0.5) * (dt.abs() ** 0.5) * sde_rnd[step]
        return y0

    @torch.no_grad()
    def _forward_fast(self, llm_cond, lat_cond, y0, t, sde_args, sde_rnd):
        model = self.model
        assert isinstance(model, DiT)
        model_dtype = llm_cond.dtype
        patch_size = y0.shape[1]

        # Hoist loop-invariant projections out of the ODE loop: conditioning,
        # latent history, timestep embeddings, and rotary frequencies do not
        # depend on the evolving state ``y0``.
        c_cond = model.c_embedder(llm_cond)
        c_bias = model.c_embedder.cond_embedder.bias
        c_null = c_bias.view(1, 1, -1).expand_as(c_cond) if c_bias is not None else torch.zeros_like(c_cond)
        c_2b = torch.cat((c_cond, c_null), dim=0)

        h_1b = model.x_embedder(lat_cond)
        h_2b = torch.cat((h_1b, h_1b), dim=0)

        seq_len = c_2b.shape[1] + h_2b.shape[1] + patch_size
        rope = model.rotary_embed.forward_from_seq_len(seq_len)
        time_embs = model.t_embedder(t[: self.steps].to(model_dtype)).unsqueeze(1).unsqueeze(2)

        y_model = y0.to(model_dtype)
        for step in range(self.steps):
            x_now_1b = model.x_embedder(y_model)
            x = torch.cat((time_embs[step] + c_2b, h_2b, torch.cat((x_now_1b, x_now_1b), dim=0)), dim=1)
            pred_cfg = model.forward_blocks_and_tail(x, rope, patch_size, use_fused_norm=True)
            add_noise = self.include_final_noise or step < self.steps - 1
            y0, y_model = fused_cfm_euler_sde_step(
                pred_cfg,
                y0,
                sde_rnd[step],
                t,
                sde_args,
                step,
                add_noise=add_noise,
                model_dtype=model_dtype,
            )
        return y0

    @torch.no_grad()
    def forward(self, llm_cond, lat_cond, y0, t, sde_args, sde_rnd):
        if self._can_use_fast_path(llm_cond, lat_cond, y0):
            return self._forward_fast(llm_cond, lat_cond, y0, t, sde_args, sde_rnd)
        return self.forward_native(llm_cond, lat_cond, y0, t, sde_args, sde_rnd)


class CFMGraphExecutor:
    """Capture CFM sampling + Aggregator + stop head into one CUDA graph.

    First ``execute`` captures the graph; later calls copy inputs into the
    static placeholders and ``replay()``. Runtime controls (cfg/sigma/temp) are
    fed via the ``sde_args`` placeholder, so they are tunable without recapture.
    """

    def __init__(
        self,
        cfm_sampler: CFMSampler,
        aggregator: nn.Module,
        stop_head: nn.Module,
        *,
        patch_size: int,
        latent_dim: int,
        batch_size: int,
        steps: int = DEFAULT_CFM_STEPS,
        num_warmups: int = 1,
    ):
        self.cfm_sampler = cfm_sampler
        self.aggregator = aggregator
        self.stop_head = stop_head
        self.patch_size = int(patch_size)
        self.latent_dim = int(latent_dim)
        self.batch_size = int(batch_size)
        self.steps = int(steps)
        self.num_warmups = max(1, int(num_warmups))
        self.initialized = False
        self.graph: torch.cuda.CUDAGraph | None = None
        self._timesteps_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
        self._last_sde_args: tuple[float, float, float] | None = None

    def _timesteps(self, device, dtype=torch.float32):
        # Raw EPSS schedule + sway in the checkpoint solver dtype.
        # The schedule is constant; building it from a Python list is a
        # pageable H2D copy that blocks the host, so build it once per device.
        key = (device, dtype)
        timesteps = self._timesteps_cache.get(key)
        if timesteps is None:
            timesteps = build_timesteps(self.steps, device=device, dtype=dtype, use_epss=True, sway_sampling_coef=-1.0)
            self._timesteps_cache[key] = timesteps
        return timesteps

    def _pad_batch(self, tensor: torch.Tensor, *, dim: int = 0) -> torch.Tensor:
        actual_batch_size = int(tensor.shape[dim])
        if actual_batch_size > self.batch_size:
            raise ValueError(f"CFM graph bucket {self.batch_size} cannot run batch {actual_batch_size}.")
        if actual_batch_size == self.batch_size:
            return tensor
        padding_shape = list(tensor.shape)
        padding_shape[dim] = self.batch_size - actual_batch_size
        return torch.cat((tensor, tensor.new_zeros(padding_shape)), dim=dim)

    @torch.no_grad()
    def execute(
        self,
        llm_cond,
        his_lat,
        cfg_strength,
        sigma,
        temperature,
        generator=None,
        initial_noise=None,
        solver_noise=None,
    ):
        if torch.cuda.is_current_stream_capturing():
            return None
        actual_batch_size = int(llm_cond.shape[0])
        device = llm_cond.device
        # Legacy checkpoints integrate in float32.
        y0 = (
            initial_noise
            if initial_noise is not None
            else torch.randn(
                (actual_batch_size, self.patch_size, self.latent_dim),
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
        )
        t = self._timesteps(device, y0.dtype)
        sde_rnd = (
            solver_noise
            if solver_noise is not None
            else torch.randn(
                (self.steps, actual_batch_size, self.patch_size, self.latent_dim),
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
        )

        if not self.initialized:
            self._initialize_graph(
                self._pad_batch(llm_cond),
                self._pad_batch(his_lat),
                self._pad_batch(y0),
                t,
                self._pad_batch(sde_rnd, dim=1),
            )

        # As in the MOSS bucket wrapper, keep address-stable full-bucket
        # buffers and replace only the live rows. CFM has no persistent row
        # state, so zero-filled padding is sufficient; full buckets skip it.
        padded = actual_batch_size < self.batch_size
        if padded:
            self._llm_cond_ph[actual_batch_size:].zero_()
            self._his_lat_ph[actual_batch_size:].zero_()
            self._y0_ph[actual_batch_size:].zero_()
            self._sde_rnd_ph[:, actual_batch_size:].zero_()
        self._llm_cond_ph[:actual_batch_size].copy_(llm_cond)
        self._his_lat_ph[:actual_batch_size].copy_(his_lat)
        self._y0_ph[:actual_batch_size].copy_(y0)
        # ``_t_ph`` holds the constant schedule copied in at capture time.
        sde_args = (float(cfg_strength), float(sigma), float(temperature))
        if sde_args != self._last_sde_args:
            self._sde_args_ph[0], self._sde_args_ph[1], self._sde_args_ph[2] = sde_args
            self._last_sde_args = sde_args
        self._sde_rnd_ph[:, :actual_batch_size].copy_(sde_rnd)

        assert self.graph is not None
        self.graph.replay()

        return (
            self._gen_lat_ph[:actual_batch_size].clone(),
            self._embeds_ph[:actual_batch_size].clone(),
            self._stop_ph[:actual_batch_size].clone(),
        )

    def _initialize_graph(self, llm_cond, his_lat, y0, t, sde_rnd):
        model_dtype = llm_cond.dtype
        self._llm_cond_ph = torch.empty_like(llm_cond)
        self._his_lat_ph = torch.empty_like(his_lat)
        self._y0_ph = torch.empty_like(y0)
        self._t_ph = torch.empty_like(t)
        self._sde_args_ph = torch.empty(3, device=llm_cond.device, dtype=y0.dtype)
        self._sde_rnd_ph = torch.empty_like(sde_rnd)

        self._llm_cond_ph.copy_(llm_cond)
        self._his_lat_ph.copy_(his_lat)
        self._y0_ph.copy_(y0)
        self._t_ph.copy_(t)
        self._sde_args_ph[0], self._sde_args_ph[1], self._sde_args_ph[2] = 2.0, 0.25, 0.0
        self._sde_rnd_ph.copy_(sde_rnd)

        def _run():
            # Cast the final latent to the model dtype
            # for the Aggregator / stop head (matches eager ``.to(dtype)``).
            gen_lat = self.cfm_sampler(
                self._llm_cond_ph, self._his_lat_ph, self._y0_ph, self._t_ph, self._sde_args_ph, self._sde_rnd_ph
            ).to(model_dtype)
            embeds = self.aggregator(gen_lat)
            stop = self.stop_head(self._llm_cond_ph[:, -1, :]).softmax(dim=-1)
            return gen_lat, embeds, stop

        # Warm up on a side stream to stabilize allocations before capture.
        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            for _ in range(self.num_warmups):
                _run()
        torch.cuda.current_stream().wait_stream(warmup)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            self.graph,
            pool=current_platform.get_global_graph_pool(),
            capture_error_mode="thread_local",
        ):
            self._gen_lat_ph, self._embeds_ph, self._stop_ph = _run()

        self.initialized = True
