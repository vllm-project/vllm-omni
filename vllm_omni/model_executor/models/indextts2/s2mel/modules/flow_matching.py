# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC
from collections import OrderedDict
from dataclasses import dataclass

import torch

from vllm_omni.model_executor.models.indextts2.s2mel.modules.diffusion_transformer import DiT


@dataclass
class EulerRequestState:
    """Persistent per-request state for resumable CFM inference."""

    request_id: str
    x: torch.Tensor
    x_lens: torch.Tensor
    prompt_x: torch.Tensor
    style: torch.Tensor
    mu: torch.Tensor
    prefix_mask: torch.Tensor
    pre_mask: tuple[torch.Tensor, torch.Tensor | None]
    t_span: torch.Tensor
    t: torch.Tensor
    step_index: int
    inference_cfg_rate: float
    cfg_prompt_x: torch.Tensor | None
    cfg_style: torch.Tensor | None
    cfg_mu: torch.Tensor | None
    cfg_pre_mask: tuple[torch.Tensor, torch.Tensor | None] | None

    @property
    def finished(self) -> bool:
        return self.step_index >= int(self.t_span.numel()) - 1


@dataclass
class _EulerGroupCacheEntry:
    logical_lengths: tuple[int, ...]
    x_lens: torch.Tensor
    prefix_mask: torch.Tensor
    estimator_x: torch.Tensor
    estimator_prompt: torch.Tensor
    estimator_style: torch.Tensor
    estimator_mu: torch.Tensor
    estimator_t: torch.Tensor
    estimator_mask: tuple[torch.Tensor, torch.Tensor | None]
    unpad_data: tuple[torch.Tensor, torch.Tensor] | None
    dt: torch.Tensor
    updated_x: torch.Tensor


def build_cfm_unpad_data(
    lengths: list[int],
    *,
    padded_length: int,
    token_offset: int,
    repeat_for_cfg: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build row-major FlashAttention unpadding metadata for one CFM group."""
    if not lengths:
        raise ValueError("CFM unpad data requires at least one sequence length")
    if token_offset < 0:
        raise ValueError(f"CFM unpad token offset must be non-negative: {token_offset}")
    if any(length < 0 or length > padded_length for length in lengths):
        raise ValueError(f"CFM unpad lengths out of range: lengths={lengths} padded={padded_length}")
    if max(lengths) != padded_length:
        raise ValueError(f"CFM unpad group must contain its padded max: lengths={lengths} padded={padded_length}")

    valid_lengths = [length + token_offset for length in lengths]
    if repeat_for_cfg:
        valid_lengths = [*valid_lengths, *valid_lengths]
    padded_stride = padded_length + token_offset
    indices = torch.cat(
        [
            torch.arange(valid_length, device=device, dtype=torch.int64) + row * padded_stride
            for row, valid_length in enumerate(valid_lengths)
        ]
    )
    cumulative_lengths = [0]
    for valid_length in valid_lengths:
        cumulative_lengths.append(cumulative_lengths[-1] + valid_length)
    cu_seqlens = torch.tensor(cumulative_lengths, device=device, dtype=torch.int32)
    return indices, cu_seqlens


class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        self.sigma_min = 1e-6

        self.estimator = None

        self.in_channels = args.DiT.in_channels

        if hasattr(args.DiT, "zero_prompt_speech_token"):
            self.zero_prompt_speech_token = args.DiT.zero_prompt_speech_token
        else:
            self.zero_prompt_speech_token = False

    def init_euler_state(
        self,
        *,
        request_id: str,
        mu: torch.Tensor,
        x_lens: torch.Tensor,
        prompt: torch.Tensor,
        style: torch.Tensor,
        n_timesteps: int,
        inference_cfg_rate: float,
        initial_noise: torch.Tensor | None = None,
        prompt_lens: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> EulerRequestState:
        """Initialize one request without entering the recurrent Euler loop."""
        if mu.shape[0] != 1:
            raise ValueError(f"Resumable CFM state expects batch=1, got {mu.shape[0]}")
        if n_timesteps < 1:
            raise ValueError(f"Resumable CFM requires at least one step, got {n_timesteps}")

        expected_noise_shape = (1, self.in_channels, int(mu.shape[1]))
        if initial_noise is None:
            x = torch.randn(expected_noise_shape, device=mu.device, dtype=mu.dtype)
        else:
            if tuple(initial_noise.shape) != expected_noise_shape:
                raise ValueError(
                    "CFM initial noise shape mismatch: "
                    f"expected={expected_noise_shape}, got={tuple(initial_noise.shape)}"
                )
            x = initial_noise.to(device=mu.device, dtype=mu.dtype).clone()
        x.mul_(temperature)

        if prompt_lens is None:
            prompt_lens = torch.tensor([prompt.size(-1)], device=x.device, dtype=torch.long)
        else:
            prompt_lens = prompt_lens.to(device=x.device, dtype=torch.long).reshape(-1)
        if prompt_lens.shape != (1,):
            raise ValueError(f"Resumable CFM prompt length must have one row, got {tuple(prompt_lens.shape)}")

        prompt_positions = torch.arange(prompt.size(-1), device=x.device)
        prompt_mask = prompt_positions.unsqueeze(0) < prompt_lens.unsqueeze(1)
        prefix_positions = torch.arange(x.size(-1), device=x.device)
        prefix_mask = prefix_positions.unsqueeze(0) < prompt_lens.unsqueeze(1)
        prompt_x = torch.zeros_like(x)
        prompt_x[..., : prompt.size(-1)] = prompt * prompt_mask.unsqueeze(1).to(prompt.dtype)
        x.masked_fill_(prefix_mask.unsqueeze(1), 0)
        if self.zero_prompt_speech_token:
            mu = mu.masked_fill(prefix_mask.unsqueeze(-1), 0)

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        state_x_lens = x_lens.to(device=x.device, dtype=torch.long).reshape(1)
        pre_mask = self._precompute_mask(x, state_x_lens)
        cfg_prompt_x = None
        cfg_style = None
        cfg_mu = None
        cfg_pre_mask = None
        if inference_cfg_rate > 0:
            cfg_prompt_x = prompt_x.new_zeros((2, *prompt_x.shape[1:]))
            cfg_prompt_x[:1].copy_(prompt_x)
            cfg_style = style.new_zeros((2, *style.shape[1:]))
            cfg_style[:1].copy_(style)
            cfg_mu = mu.new_zeros((2, *mu.shape[1:]))
            cfg_mu[:1].copy_(mu)
            cfg_pre_mask = self._repeat_pre_mask_for_cfg(pre_mask)
        return EulerRequestState(
            request_id=request_id,
            x=x,
            x_lens=state_x_lens,
            prompt_x=prompt_x,
            style=style,
            mu=mu,
            prefix_mask=prefix_mask,
            pre_mask=pre_mask,
            t_span=t_span,
            t=t_span[0].clone(),
            step_index=0,
            inference_cfg_rate=float(inference_cfg_rate),
            cfg_prompt_x=cfg_prompt_x,
            cfg_style=cfg_style,
            cfg_mu=cfg_mu,
            cfg_pre_mask=cfg_pre_mask,
        )

    @staticmethod
    def _stack_state_masks(
        states: list[EulerRequestState],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        first_masks = [state.pre_mask[0] for state in states]
        second_masks = [state.pre_mask[1] for state in states]
        first = torch.cat(first_masks, 0)
        if second_masks[0] is None:
            return first, None
        if any(mask is None for mask in second_masks):
            raise ValueError("CFM group mixes causal and non-causal masks")
        query_length = second_masks[0].shape[-2]
        second = first[:, None, :].expand(-1, -1, query_length, -1)
        return first, second

    @staticmethod
    def _euler_group_cache_key(
        states: list[EulerRequestState],
        cfg_rate: float,
    ) -> tuple:
        reference = states[0]
        return (
            tuple((state.request_id, id(state), int(state.x.shape[-1])) for state in states),
            reference.x.device.type,
            reference.x.device.index,
            reference.x.dtype,
            float(cfg_rate),
        )

    def _build_euler_group_cache_entry(
        self,
        states: list[EulerRequestState],
        cfg_rate: float,
    ) -> _EulerGroupCacheEntry:
        reference = states[0]
        batch = len(states)
        logical_lengths = tuple(int(state.x.shape[-1]) for state in states)
        padded_length = max(logical_lengths)
        estimator_batch = 2 * batch if cfg_rate > 0 else batch
        estimator_x = reference.x.new_zeros((estimator_batch, reference.x.shape[1], padded_length))

        def pad_last(tensor: torch.Tensor) -> torch.Tensor:
            padded = tensor.new_zeros((tensor.shape[0], tensor.shape[1], padded_length))
            padded[..., : tensor.shape[-1]].copy_(tensor)
            return padded

        def pad_time(tensor: torch.Tensor) -> torch.Tensor:
            padded = tensor.new_zeros((tensor.shape[0], padded_length, tensor.shape[2]))
            padded[:, : tensor.shape[1]].copy_(tensor)
            return padded

        def pad_mask(tensor: torch.Tensor) -> torch.Tensor:
            padded = tensor.new_zeros((tensor.shape[0], padded_length))
            padded[:, : tensor.shape[-1]].copy_(tensor)
            return padded

        if batch == 1:
            x_lens = reference.x_lens
            prompt_x = reference.prompt_x
            style = reference.style
            mu = reference.mu
            prefix_mask = reference.prefix_mask
            pre_mask = reference.pre_mask
        else:
            x_lens = torch.cat([state.x_lens for state in states], dim=0)
            prompt_x = torch.cat([pad_last(state.prompt_x) for state in states], dim=0)
            style = torch.cat([state.style for state in states], dim=0)
            mu = torch.cat([pad_time(state.mu) for state in states], dim=0)
            prefix_mask = torch.cat([pad_mask(state.prefix_mask) for state in states], dim=0)
            if len(set(logical_lengths)) > 1:
                pre_mask = self._precompute_mask(estimator_x[:batch], x_lens)
            else:
                pre_mask = self._stack_state_masks(states)

        unpad_data = None
        if len(set(logical_lengths)) > 1:
            estimator = getattr(self, "_eager_estimator", self.estimator)
            token_offset = int(getattr(estimator, "style_as_token", False)) + int(
                getattr(estimator, "time_as_token", False)
            )
            unpad_data = build_cfm_unpad_data(
                list(logical_lengths),
                padded_length=padded_length,
                token_offset=token_offset,
                repeat_for_cfg=cfg_rate > 0,
                device=reference.x.device,
            )

        if cfg_rate > 0:
            if batch == 1:
                if (
                    reference.cfg_prompt_x is None
                    or reference.cfg_style is None
                    or reference.cfg_mu is None
                    or reference.cfg_pre_mask is None
                ):
                    raise RuntimeError("Single-request CFG conditioning cache was not initialized")
                estimator_prompt = reference.cfg_prompt_x
                estimator_style = reference.cfg_style
                estimator_mu = reference.cfg_mu
                estimator_mask = reference.cfg_pre_mask
            else:
                estimator_prompt = prompt_x.new_zeros((2 * batch, *prompt_x.shape[1:]))
                estimator_prompt[:batch].copy_(prompt_x)
                estimator_style = style.new_zeros((2 * batch, *style.shape[1:]))
                estimator_style[:batch].copy_(style)
                estimator_mu = mu.new_zeros((2 * batch, *mu.shape[1:]))
                estimator_mu[:batch].copy_(mu)
                estimator_mask = self._repeat_pre_mask_for_cfg(pre_mask)
        else:
            estimator_prompt = prompt_x
            estimator_style = style
            estimator_mu = mu
            estimator_mask = pre_mask
        return _EulerGroupCacheEntry(
            logical_lengths=logical_lengths,
            x_lens=x_lens,
            prefix_mask=prefix_mask,
            estimator_x=estimator_x,
            estimator_prompt=estimator_prompt,
            estimator_style=estimator_style,
            estimator_mu=estimator_mu,
            estimator_t=reference.t.new_empty((estimator_batch,)),
            estimator_mask=estimator_mask,
            unpad_data=unpad_data,
            dt=reference.t.new_empty((batch,)),
            updated_x=reference.x.new_empty((batch, reference.x.shape[1], padded_length)),
        )

    def _get_euler_group_cache_entry(
        self,
        states: list[EulerRequestState],
        cfg_rate: float,
    ) -> _EulerGroupCacheEntry:
        cache = getattr(self, "_euler_group_cache", None)
        if cache is None:
            cache = OrderedDict()
            self._euler_group_cache = cache
        key = self._euler_group_cache_key(states, cfg_rate)
        entry = cache.pop(key, None)
        if entry is None:
            entry = self._build_euler_group_cache_entry(states, cfg_rate)
        cache[key] = entry
        while len(cache) > 8:
            cache.popitem(last=False)
        return entry

    def discard_euler_group_cache(self, request_ids: set[str]) -> None:
        """Drop cached workspaces that retain completed request tensors."""
        cache = getattr(self, "_euler_group_cache", None)
        if not cache or not request_ids:
            return
        stale_keys = [
            key for key in cache if any(request_id in request_ids for request_id, _state_id, _length in key[0])
        ]
        for key in stale_keys:
            cache.pop(key, None)

    @staticmethod
    def _refresh_euler_group_workspaces(
        entry: _EulerGroupCacheEntry,
        states: list[EulerRequestState],
        cfg_rate: float,
    ) -> None:
        batch = len(states)
        conditional_x = entry.estimator_x[:batch]
        for row, state in enumerate(states):
            length = entry.logical_lengths[row]
            resident_x = conditional_x[row : row + 1, :, :length]
            if resident_x.data_ptr() != state.x.data_ptr():
                resident_x.copy_(state.x)
                state.x = resident_x
        torch.stack([state.t for state in states], out=entry.estimator_t[:batch])
        if cfg_rate > 0:
            entry.estimator_x[batch:].copy_(conditional_x)
            entry.estimator_t[batch:].copy_(entry.estimator_t[:batch])

    @torch.inference_mode()
    def run_euler_step(self, states: list[EulerRequestState]) -> None:
        """Advance one independently timed request group by one Euler step."""
        if not states:
            return
        if any(state.finished for state in states):
            raise ValueError("Cannot advance an already-finished resumable CFM state")
        expected_cfg = states[0].inference_cfg_rate
        if any(state.inference_cfg_rate != expected_cfg for state in states):
            raise ValueError("Resumable CFM step requires one CFG rate per group")

        entry = self._get_euler_group_cache_entry(states, expected_cfg)
        self._refresh_euler_group_workspaces(entry, states, expected_cfg)
        batch = len(states)
        x = entry.estimator_x[:batch]
        autocast_dtype = getattr(self, "estimator_autocast_dtype", None)
        estimator_args = (
            entry.estimator_x,
            entry.estimator_prompt,
            entry.x_lens,
            entry.estimator_t,
            entry.estimator_style,
            entry.estimator_mu,
        )
        estimator_kwargs = {
            "pre_mask": entry.estimator_mask,
            "unpad_data": entry.unpad_data,
        }
        if autocast_dtype is None:
            derivative = self.estimator(*estimator_args, **estimator_kwargs)
        else:
            with torch.autocast(x.device.type, dtype=autocast_dtype):
                derivative = self.estimator(*estimator_args, **estimator_kwargs)
            derivative = derivative.float()

        torch.stack(
            [state.t_span[state.step_index + 1] - state.t_span[state.step_index] for state in states],
            out=entry.dt,
        )
        if expected_cfg > 0:
            conditional, unconditional = derivative.chunk(2, dim=0)
            derivative = (1.0 + expected_cfg) * conditional - expected_cfg * unconditional
        torch.mul(derivative, entry.dt[:, None, None], out=entry.updated_x)
        x.add_(entry.updated_x)
        x.masked_fill_(entry.prefix_mask.unsqueeze(1), 0)
        for row, state in enumerate(states):
            state.t.add_(entry.dt[row])
            state.step_index += 1

    @staticmethod
    def finalize_euler_state(state: EulerRequestState) -> torch.Tensor:
        if not state.finished:
            raise ValueError(f"CFM request {state.request_id} is not finished")
        return state.x

    @torch.inference_mode()
    def inference(
        self,
        mu,
        x_lens,
        prompt,
        style,
        f0,
        n_timesteps,
        temperature=1.0,
        inference_cfg_rate=0.5,
        initial_noise=None,
        prompt_lens=None,
        unpad_data: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        """Forward diffusion

        Args:
            mu (torch.Tensor): semantic info of reference audio and altered audio
                shape: (batch_size, mel_timesteps(795+1069), 512)
            x_lens (torch.Tensor): mel frames output
                shape: (batch_size, mel_timesteps)
            prompt (torch.Tensor): reference mel
                shape: (batch_size, 80, 795)
            style (torch.Tensor): reference global style
                shape: (batch_size, 192)
            f0 (None): unused, reserved for f0 conditioning
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.

        Returns:
            sample (torch.Tensor): generated mel-spectrogram
                shape: (batch_size, 80, mel_timesteps)
        """
        B, T = mu.size(0), mu.size(1)
        expected_noise_shape = (B, self.in_channels, T)
        if initial_noise is None:
            z = torch.randn(
                expected_noise_shape,
                device=mu.device,
                dtype=mu.dtype,
            )
        else:
            if tuple(initial_noise.shape) != expected_noise_shape:
                raise ValueError(
                    "CFM initial noise shape mismatch: "
                    f"expected={expected_noise_shape}, got={tuple(initial_noise.shape)}"
                )
            z = initial_noise.to(device=mu.device, dtype=mu.dtype)
        z = z * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        return self.solve_euler(
            z,
            x_lens,
            prompt,
            mu,
            style,
            f0,
            t_span,
            inference_cfg_rate,
            prompt_lens=prompt_lens,
            unpad_data=unpad_data,
        )

    def solve_euler(
        self,
        x,
        x_lens,
        prompt,
        mu,
        style,
        f0,
        t_span,
        inference_cfg_rate=0.5,
        prompt_lens=None,
        unpad_data: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): semantic info of reference audio and altered audio
                shape: (batch_size, mel_timesteps(795+1069), 512)
            x_lens (torch.Tensor): mel frames output
                shape: (batch_size, mel_timesteps)
            prompt (torch.Tensor): reference mel
                shape: (batch_size, 80, 795)
            style (torch.Tensor): reference global style
                shape: (batch_size, 192)
        """
        t = t_span[0]

        # Optional bf16 autocast for the DiT estimator (set by the decoder via
        # `estimator_autocast_dtype`). The Euler solver state stays float32 —
        # only the estimator forward runs in reduced precision, which restores
        # flash-attention eligibility and halves GEMM cost.
        autocast_dtype = getattr(self, "estimator_autocast_dtype", None)

        def _run_estimator(*args, **kw):
            if autocast_dtype is None:
                return self.estimator(*args, **kw)
            with torch.autocast(x.device.type, dtype=autocast_dtype):
                out = self.estimator(*args, **kw)
            return out.float()

        batch = x.shape[0]
        if prompt_lens is None:
            prompt_lens = torch.full(
                (batch,),
                prompt.size(-1),
                device=x.device,
                dtype=torch.long,
            )
        else:
            prompt_lens = prompt_lens.to(device=x.device, dtype=torch.long).reshape(-1)
            if prompt_lens.shape != (batch,):
                raise ValueError(f"CFM prompt length/batch mismatch: lengths={prompt_lens.numel()} batch={batch}")

        prompt_positions = torch.arange(prompt.size(-1), device=x.device)
        prompt_mask = prompt_positions.unsqueeze(0) < prompt_lens.unsqueeze(1)
        x_positions = torch.arange(x.size(-1), device=x.device)
        noise_prefix_mask = x_positions.unsqueeze(0) < prompt_lens.unsqueeze(1)

        prompt_x = torch.zeros_like(x)
        prompt_x[..., : prompt.size(-1)] = prompt * prompt_mask.unsqueeze(1).to(prompt.dtype)
        x.masked_fill_(noise_prefix_mask.unsqueeze(1), 0)
        if self.zero_prompt_speech_token:
            mu = mu.masked_fill(noise_prefix_mask.unsqueeze(-1), 0)

        # Precompute mask once — T and x_lens are constant across all ODE steps.
        # x_mask [B, 1, T] broadcasts to [2B, 1, T] when CFG doubles batch.
        pre_mask = self._precompute_mask(x, x_lens)

        if inference_cfg_rate > 0:
            cfg_prompt_x = prompt_x.new_zeros((2 * batch, *prompt_x.shape[1:]))
            cfg_prompt_x[:batch].copy_(prompt_x)
            cfg_style = style.new_zeros((2 * batch, *style.shape[1:]))
            cfg_style[:batch].copy_(style)
            cfg_mu = mu.new_zeros((2 * batch, *mu.shape[1:]))
            cfg_mu[:batch].copy_(mu)
            cfg_x = x.new_empty((2 * batch, *x.shape[1:]))
            cfg_t = x.new_empty((2 * batch,))
            cfg_pre_mask = self._repeat_pre_mask_for_cfg(pre_mask)

        for step in range(1, len(t_span)):
            dt = t_span[step] - t_span[step - 1]
            if inference_cfg_rate > 0:
                cfg_x[:batch].copy_(x)
                cfg_x[batch:].copy_(x)
                cfg_t.fill_(t)

                stacked_dphi_dt = _run_estimator(
                    cfg_x,
                    cfg_prompt_x,
                    x_lens,
                    cfg_t,
                    cfg_style,
                    cfg_mu,
                    pre_mask=cfg_pre_mask,
                    unpad_data=unpad_data,
                )

                dphi_dt, cfg_dphi_dt = stacked_dphi_dt.chunk(2, dim=0)
                dphi_dt = (1.0 + inference_cfg_rate) * dphi_dt - inference_cfg_rate * cfg_dphi_dt
            else:
                dphi_dt = _run_estimator(
                    x,
                    prompt_x,
                    x_lens,
                    t.unsqueeze(0),
                    style,
                    mu,
                    pre_mask=pre_mask,
                    unpad_data=unpad_data,
                )

            x = x + dt * dphi_dt
            t = t + dt
            x.masked_fill_(noise_prefix_mask.unsqueeze(1), 0)

        return x

    @staticmethod
    def _repeat_pre_mask_for_cfg(pre_mask):
        """Duplicate precomputed masks when CFG doubles the batch.

        The original IndexTTS2 inference path used B=1, where a two-element
        timestep tensor happened to match CFG. Stage-1 batching makes the
        estimator input batch 2B, so all batch-indexed masks must be repeated
        consistently with ``torch.cat([cond, uncond], dim=0)``.
        """
        if not isinstance(pre_mask, tuple):
            return pre_mask
        repeated = []
        for mask in pre_mask:
            if isinstance(mask, torch.Tensor):
                repeated.append(torch.cat([mask, mask], dim=0))
            else:
                repeated.append(mask)
        return tuple(repeated)

    def _precompute_mask(self, x: torch.Tensor, x_lens: torch.Tensor):
        """Precompute padding mask for DiT — reused across all ODE steps."""
        from .commons import sequence_mask

        estimator = self._eager_estimator if hasattr(self, "_eager_estimator") else self.estimator
        T = x.size(2)
        style_offset = getattr(estimator, "style_as_token", False)
        time_offset = getattr(estimator, "time_as_token", False)
        T_in = T + int(style_offset) + int(time_offset)
        is_causal = getattr(estimator, "is_causal", False)

        x_mask = sequence_mask(x_lens + int(style_offset) + int(time_offset), max_length=T_in).to(x.device).unsqueeze(1)
        if is_causal:
            return (x_mask, None)
        x_mask_expanded = x_mask[:, None, :].expand(-1, -1, T_in, -1)
        return (x_mask, x_mask_expanded)


class CFM(BASECFM):
    def __init__(self, args):
        super().__init__(args)
        if args.dit_type == "DiT":
            self.estimator = DiT(args)
            object.__setattr__(self, "_eager_estimator", self.estimator)
        else:
            raise NotImplementedError(f"Unknown diffusion type {args.dit_type}")
        self._compiled = False

    def enable_torch_compile(self, mode: str = "default"):
        """Enable torch.compile optimization for the estimator model.

        This method applies torch.compile to the estimator (DiT model) for significant
        performance improvements during inference. It also configures distributed
        training optimizations if applicable.
        """
        if torch.distributed.is_initialized():
            torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.estimator = torch.compile(
            self._eager_estimator,
            mode=mode,
            fullgraph=True,
            dynamic=True,
        )
        self._compiled = True

    def disable_torch_compile(self):
        self.estimator = self._eager_estimator
        self._compiled = False
