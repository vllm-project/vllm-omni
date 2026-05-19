# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diffusion input-batch structures following the MRV2-style vLLM layout.

Request states remain the only persistent source of truth. Static tensors are
normalized/padded onto the request state once, while :class:`InputBatch`
assembles an ephemeral step-local view. Dynamic tensors are re-gathered every
step, and step outputs are scattered back into request states by
``scatter_latents()`` using ``idx_mapping``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from vllm_omni.diffusion.worker.utils import DiffusionRequestState


def _normalize_prompt_embeds(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x.unsqueeze(0)
    if x.ndim == 3:
        return x
    raise ValueError(f"prompt_embeds must be 2D or 3D, got shape={tuple(x.shape)}")


def _normalize_mask(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    elif x.ndim != 2:
        raise ValueError(f"prompt mask must be 1D or 2D, got shape={tuple(x.shape)}")
    if x.dtype != torch.bool:
        x = x != 0
    return x


def _pad_prompt_embeds(x: torch.Tensor, target_seq_len: int) -> torch.Tensor:
    x = _normalize_prompt_embeds(x)
    bsz, seq_len, hidden = x.shape
    if seq_len == target_seq_len:
        return x
    out = x.new_zeros((bsz, target_seq_len, hidden))
    out[:, :seq_len] = x
    return out


def _pad_mask(x: torch.Tensor, target_seq_len: int) -> torch.Tensor:
    x = _normalize_mask(x)
    bsz, seq_len = x.shape
    if seq_len == target_seq_len:
        return x
    out = torch.zeros((bsz, target_seq_len), dtype=torch.bool, device=x.device)
    out[:, :seq_len] = x
    return out


def _select_states(
    states: Sequence[DiffusionRequestState],
    idx_mapping: torch.Tensor | None,
) -> tuple[list[DiffusionRequestState], torch.Tensor, np.ndarray]:
    if not states:
        raise ValueError("Cannot build InputBatch from empty states.")

    if idx_mapping is None:
        device = states[0].latents.device if states[0].latents is not None else None
        idx_mapping = torch.arange(len(states), dtype=torch.int32, device=device)
    else:
        if idx_mapping.ndim != 1:
            raise ValueError("idx_mapping must be a 1D tensor.")
        idx_mapping = idx_mapping.to(dtype=torch.int32)

    selected_states: list[DiffusionRequestState] = []
    for batch_idx, state_idx in enumerate(idx_mapping.tolist()):
        if state_idx < 0 or state_idx >= len(states):
            raise ValueError(f"idx_mapping[{batch_idx}]={state_idx} is out of range for states.")
        selected_states.append(states[state_idx])
    return selected_states, idx_mapping, idx_mapping.detach().cpu().numpy()


def _prepare_req_ids(states: Sequence[DiffusionRequestState]) -> list[str]:
    return [state.req_id for state in states]


def _prepare_prompt_field_on_state(
    state: DiffusionRequestState,
    *,
    embeds_attr: str,
    mask_attr: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    embeds = getattr(state, embeds_attr)
    if embeds is None:
        return None, None

    normalized_embeds = _normalize_prompt_embeds(embeds)
    if normalized_embeds is not embeds:
        setattr(state, embeds_attr, normalized_embeds)
    embeds = normalized_embeds

    mask = getattr(state, mask_attr)
    if mask is None:
        return embeds, None

    normalized_mask = _normalize_mask(mask)
    if normalized_mask is not mask:
        setattr(state, mask_attr, normalized_mask)
    return embeds, normalized_mask


def _prepare_reused_buffer(
    current: torch.Tensor | None,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if current is not None and tuple(current.shape) == shape and current.dtype == dtype and current.device == device:
        return current
    return torch.empty(shape, dtype=dtype, device=device)


def _validate_gather_tensors(
    values: Sequence[torch.Tensor],
    *,
    field_name: str,
) -> tuple[torch.dtype, torch.device, tuple[int, ...], int]:
    if not values:
        raise ValueError(f"Cannot gather empty tensor list for {field_name}.")

    first = values[0]
    dtype = first.dtype
    device = first.device
    suffix_shape = tuple(first.shape[1:])
    total_rows = 0

    for value in values:
        if value.dtype != dtype:
            raise ValueError(f"Mixed dtypes in {field_name} batch.")
        if value.device != device:
            raise ValueError(f"Mixed devices in {field_name} batch.")
        if tuple(value.shape[1:]) != suffix_shape:
            raise ValueError(
                f"Mixed trailing shapes in {field_name} batch: expected {suffix_shape}, got {tuple(value.shape[1:])}."
            )
        total_rows += int(value.shape[0])

    return dtype, device, suffix_shape, total_rows


def _gather_tensor_rows(
    values: Sequence[torch.Tensor],
    *,
    field_name: str,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    dtype, device, suffix_shape, total_rows = _validate_gather_tensors(
        values,
        field_name=field_name,
    )
    gathered = _prepare_reused_buffer(
        out,
        shape=(total_rows, *suffix_shape),
        dtype=dtype,
        device=device,
    )

    row_offset = 0
    for value in values:
        next_row_offset = row_offset + int(value.shape[0])
        gathered[row_offset:next_row_offset].copy_(value)
        row_offset = next_row_offset
    return gathered


def _get_seq_lens_from_mask(mask: torch.Tensor) -> list[int]:
    return mask.sum(dim=1, dtype=torch.int32).tolist()


def _get_request_prompt_seq_lens(
    state: DiffusionRequestState,
    *,
    embeds_attr: str,
    mask_attr: str,
    seq_lens_attr: str,
) -> list[int]:
    embeds, mask = _prepare_prompt_field_on_state(
        state,
        embeds_attr=embeds_attr,
        mask_attr=mask_attr,
    )
    if embeds is None:
        raise ValueError(f"{embeds_attr} is not initialized on request {state.req_id}.")

    if mask is not None:
        if mask.shape[0] != embeds.shape[0]:
            raise ValueError(f"{mask_attr} batch dimension does not match {embeds_attr} for request {state.req_id}.")
        return _get_seq_lens_from_mask(mask)

    seq_lens = getattr(state, seq_lens_attr)
    if seq_lens is not None:
        return [int(value) for value in seq_lens]

    return [int(embeds.shape[1])] * int(embeds.shape[0])


def _prepare_request_prompt_field(
    state: DiffusionRequestState,
    *,
    embeds_attr: str,
    mask_attr: str,
    seq_lens_attr: str,
    target_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    embeds, mask = _prepare_prompt_field_on_state(
        state,
        embeds_attr=embeds_attr,
        mask_attr=mask_attr,
    )
    if embeds is None:
        raise ValueError(f"{embeds_attr} is not initialized on request {state.req_id}.")

    actual_seq_lens = _get_request_prompt_seq_lens(
        state,
        embeds_attr=embeds_attr,
        mask_attr=mask_attr,
        seq_lens_attr=seq_lens_attr,
    )
    max_actual_seq_len = max(actual_seq_lens) if actual_seq_lens else 0

    if mask is None and max_actual_seq_len != target_seq_len:
        raise ValueError(
            f"Variable-length {embeds_attr} in batch but {mask_attr} is None. "
            f"Provide masks or ensure {embeds_attr} have the same seq_len."
        )

    current_seq_len = int(embeds.shape[1])
    if current_seq_len < target_seq_len:
        embeds = _pad_prompt_embeds(embeds, target_seq_len)
        setattr(state, embeds_attr, embeds)
        if mask is not None:
            mask = _pad_mask(mask, target_seq_len)
            setattr(state, mask_attr, mask)
        current_seq_len = target_seq_len

    if current_seq_len > target_seq_len:
        if max_actual_seq_len > target_seq_len:
            raise ValueError(
                f"{embeds_attr} for request {state.req_id} requires seq_len "
                f"{max_actual_seq_len}, got target {target_seq_len}."
            )
        return embeds[:, :target_seq_len], None if mask is None else mask[:, :target_seq_len]

    return embeds, mask


def _prepare_padded_prompt_fields(
    states: Sequence[DiffusionRequestState],
    *,
    embeds_attr: str,
    mask_attr: str,
    seq_lens_attr: str,
    embeds_out: torch.Tensor | None = None,
    mask_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    prepared_fields = [
        _prepare_prompt_field_on_state(
            state,
            embeds_attr=embeds_attr,
            mask_attr=mask_attr,
        )
        for state in states
    ]
    embeds_values = [embeds for embeds, _ in prepared_fields]
    if not any(embeds is not None for embeds in embeds_values):
        return None, None
    if not all(embeds is not None for embeds in embeds_values):
        raise ValueError(f"Mixed {embeds_attr} in batch.")

    mask_values = [mask for _, mask in prepared_fields]
    if any(mask is None for mask in mask_values):
        if any(mask is not None for mask in mask_values):
            raise ValueError(f"Mixed {mask_attr} in batch.")

    target_seq_len = max(
        max(
            _get_request_prompt_seq_lens(
                state,
                embeds_attr=embeds_attr,
                mask_attr=mask_attr,
                seq_lens_attr=seq_lens_attr,
            )
        )
        for state in states
    )

    request_embeds: list[torch.Tensor] = []
    request_masks: list[torch.Tensor] = []
    for state in states:
        prepared_embeds, prepared_mask = _prepare_request_prompt_field(
            state,
            embeds_attr=embeds_attr,
            mask_attr=mask_attr,
            seq_lens_attr=seq_lens_attr,
            target_seq_len=target_seq_len,
        )
        request_embeds.append(prepared_embeds)
        if prepared_mask is not None:
            request_masks.append(prepared_mask)

    gathered_embeds = _gather_tensor_rows(
        request_embeds,
        field_name=embeds_attr,
        out=embeds_out,
    )

    if not request_masks:
        return gathered_embeds, None

    gathered_masks = _gather_tensor_rows(
        request_masks,
        field_name=mask_attr,
        out=mask_out,
    )
    return gathered_embeds, gathered_masks


def _prepare_latents(
    states: Sequence[DiffusionRequestState],
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    latents_values = [state.latents for state in states]
    if any(latents is None for latents in latents_values):
        raise ValueError("All requests must have `latents` initialized.")
    return _gather_tensor_rows(
        [latents for latents in latents_values if latents is not None],
        field_name="latents",
        out=out,
    )


def _require_state_latents(
    state: DiffusionRequestState,
    *,
    for_field: str,
) -> torch.Tensor:
    latents = state.latents
    if latents is None:
        raise ValueError(f"Request {state.req_id} has no latents while preparing {for_field}.")
    return latents


def _expand_scalar_or_vector(
    value: torch.Tensor,
    *,
    num_rows: int,
    field_name: str,
) -> torch.Tensor:
    if value.ndim == 0:
        return value.reshape(1).expand(num_rows)
    if value.ndim != 1:
        raise ValueError(f"{field_name} must be scalar or 1D, got ndim={value.ndim}.")
    if value.shape[0] == num_rows:
        return value
    if value.shape[0] == 1:
        return value.expand(num_rows)
    raise ValueError(
        f"Per-request {field_name} must have either 1 element or {num_rows} elements; got {value.shape[0]}."
    )


def _prepare_timesteps(
    states: Sequence[DiffusionRequestState],
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    timestep_values: list[torch.Tensor] = []
    for state in states:
        timestep = state.current_timestep
        if timestep is None:
            raise ValueError("All requests must have a current timestep initialized.")
        if not torch.is_tensor(timestep):
            raise ValueError("InputBatch expects tensor timesteps; normalize them before batching.")
        latents = _require_state_latents(state, for_field="timesteps")
        timestep_values.append(
            _expand_scalar_or_vector(
                timestep,
                num_rows=int(latents.shape[0]),
                field_name="timestep tensor",
            )
        )

    return _gather_tensor_rows(
        timestep_values,
        field_name="timesteps",
        out=out,
    )


def _prepare_cfg_scalars(
    states: Sequence[DiffusionRequestState],
) -> tuple[bool, float, bool]:
    def _cfg_scalars(state: DiffusionRequestState) -> tuple[bool, float, bool]:
        true_cfg_scale = getattr(state.sampling, "true_cfg_scale", None) or 4.0
        cfg_normalize = bool(getattr(state.sampling, "cfg_normalize", False))
        return state.do_true_cfg, true_cfg_scale, cfg_normalize

    scalars = _cfg_scalars(states[0])
    for state in states[1:]:
        if _cfg_scalars(state) != scalars:
            raise ValueError("Mixed CFG settings in one diffusion batch are not supported.")
    return scalars


def _prepare_guidance(
    states: Sequence[DiffusionRequestState],
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor | None:
    guidance_values = [state.guidance for state in states]
    if all(guidance is None for guidance in guidance_values):
        return None
    if any(guidance is None for guidance in guidance_values):
        raise ValueError("Mixed guidance in one diffusion batch are not supported.")

    gathered_guidance: list[torch.Tensor] = []
    for state in states:
        latents = _require_state_latents(state, for_field="guidance")
        guidance = state.guidance
        assert guidance is not None
        guidance_tensor = torch.as_tensor(
            guidance,
            device=latents.device,
            dtype=latents.dtype,
        )
        gathered_guidance.append(
            _expand_scalar_or_vector(
                guidance_tensor,
                num_rows=int(latents.shape[0]),
                field_name="guidance tensor",
            )
        )

    return _gather_tensor_rows(
        gathered_guidance,
        field_name="guidance",
        out=out,
    )


def _prepare_image_latents(
    states: Sequence[DiffusionRequestState],
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor | None:
    image_latents = [getattr(state.sampling, "image_latent", None) for state in states]
    if all(image_latent is None for image_latent in image_latents):
        return None
    if any(image_latent is None for image_latent in image_latents):
        raise ValueError("Mixed image_latent presence in one diffusion batch is not supported.")
    return _gather_tensor_rows(
        [image_latent for image_latent in image_latents if image_latent is not None],
        field_name="image_latents",
        out=out,
    )


def _prepare_seq_lens(
    states: Sequence[DiffusionRequestState],
    attr_name: str,
) -> list[int] | None:
    values = [getattr(state, attr_name) for state in states]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"Mixed {attr_name} in batch.")
    return [int(value[0]) for value in values if value is not None]


def _prepare_img_shapes(states: Sequence[DiffusionRequestState]) -> list | None:
    values = [state.img_shapes for state in states]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("Mixed img_shapes in batch.")
    return [value[0] if value else [] for value in values if value is not None]


def _prepare_prompt_embeds(
    states: Sequence[DiffusionRequestState],
    *,
    embeds_out: torch.Tensor | None = None,
    mask_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    prompt_embeds, prompt_embeds_mask = _prepare_padded_prompt_fields(
        states,
        embeds_attr="prompt_embeds",
        mask_attr="prompt_embeds_mask",
        seq_lens_attr="txt_seq_lens",
        embeds_out=embeds_out,
        mask_out=mask_out,
    )
    if prompt_embeds is None:
        raise ValueError("All requests must have `prompt_embeds` initialized.")
    return prompt_embeds, prompt_embeds_mask


def _prepare_negative_prompt_embeds(
    states: Sequence[DiffusionRequestState],
    *,
    embeds_out: torch.Tensor | None = None,
    mask_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    return _prepare_padded_prompt_fields(
        states,
        embeds_attr="negative_prompt_embeds",
        mask_attr="negative_prompt_embeds_mask",
        seq_lens_attr="negative_txt_seq_lens",
        embeds_out=embeds_out,
        mask_out=mask_out,
    )


def _same_composition(
    cached_batch: InputBatch | None,
    req_ids: list[str],
    idx_mapping_np: np.ndarray,
) -> bool:
    if cached_batch is None:
        return False
    if cached_batch.req_ids != req_ids:
        return False
    return np.array_equal(cached_batch.idx_mapping_np, idx_mapping_np)


def _scatter_batch_tensor_by_mapping(
    states: Sequence[DiffusionRequestState],
    idx_mapping_np: np.ndarray,
    *,
    attr_name: str,
    value: torch.Tensor,
) -> None:
    row_offset = 0
    for batch_idx, state_idx in enumerate(idx_mapping_np.tolist()):
        if state_idx < 0 or state_idx >= len(states):
            raise ValueError(f"idx_mapping[{batch_idx}]={state_idx} is out of range for states.")
        state = states[state_idx]
        state_value = getattr(state, attr_name)
        num_rows = 1 if state_value is None else int(state_value.shape[0])
        next_row_offset = row_offset + num_rows
        value_slice = value[row_offset:next_row_offset]
        if state_value is None:
            setattr(state, attr_name, value_slice.clone())
        elif (
            tuple(state_value.shape) != tuple(value_slice.shape)
            or state_value.dtype != value_slice.dtype
            or state_value.device != value_slice.device
        ):
            setattr(state, attr_name, value_slice.clone())
        else:
            state_value.copy_(value_slice)
        row_offset = next_row_offset

    if row_offset != int(value.shape[0]):
        raise ValueError(
            f"Scatter for {attr_name} consumed {row_offset} rows, but batch has {int(value.shape[0])} rows."
        )


@dataclass
class InputBatch:
    """Ephemeral step-level batch view.

    Static request-local tensors are normalized and padded onto
    ``DiffusionRequestState`` itself, making the request state the persistent
    source of truth. ``InputBatch`` only assembles a contiguous view for the
    current step and refreshes dynamic fields in-place when composition is
    unchanged.
    """

    req_ids: list[str]
    num_reqs: int
    num_reqs_after_padding: int
    idx_mapping: torch.Tensor
    idx_mapping_np: np.ndarray

    latents: torch.Tensor
    timesteps: torch.Tensor
    prompt_embeds: torch.Tensor
    prompt_embeds_mask: torch.Tensor | None
    negative_prompt_embeds: torch.Tensor | None
    negative_prompt_embeds_mask: torch.Tensor | None
    guidance: torch.Tensor | None = None
    do_true_cfg: bool = False
    true_cfg_scale: float = 4.0
    cfg_normalize: bool = False
    image_latents: torch.Tensor | None = None

    img_shapes: list | None = None
    txt_seq_lens: list[int] | None = None
    negative_txt_seq_lens: list[int] | None = None

    def __post_init__(self) -> None:
        if len(self.req_ids) != int(self.idx_mapping.numel()):
            raise ValueError("`req_ids` and `idx_mapping` must have the same length.")
        if self.num_reqs != len(self.req_ids):
            raise ValueError("`num_reqs` must match the number of request ids.")
        if self.num_reqs_after_padding < self.num_reqs:
            raise ValueError("`num_reqs_after_padding` must be >= `num_reqs`.")

    def _refresh_dynamic_fields(
        self,
        selected_states: Sequence[DiffusionRequestState],
    ) -> None:
        self.latents = _prepare_latents(selected_states, out=self.latents)
        self.timesteps = _prepare_timesteps(selected_states, out=self.timesteps)

    def _refresh_static_fields(
        self,
        states: Sequence[DiffusionRequestState],
    ) -> None:
        self.do_true_cfg, self.true_cfg_scale, self.cfg_normalize = _prepare_cfg_scalars(states)
        self.guidance = _prepare_guidance(states, out=self.guidance)
        self.image_latents = _prepare_image_latents(states, out=self.image_latents)
        self.prompt_embeds, self.prompt_embeds_mask = _prepare_prompt_embeds(
            states,
            embeds_out=self.prompt_embeds,
            mask_out=self.prompt_embeds_mask,
        )
        (
            self.negative_prompt_embeds,
            self.negative_prompt_embeds_mask,
        ) = _prepare_negative_prompt_embeds(
            states,
            embeds_out=self.negative_prompt_embeds,
            mask_out=self.negative_prompt_embeds_mask,
        )
        self.img_shapes = _prepare_img_shapes(states)
        self.txt_seq_lens = _prepare_seq_lens(states, "txt_seq_lens")
        self.negative_txt_seq_lens = _prepare_seq_lens(states, "negative_txt_seq_lens")

    def _repack_dynamic_fields(
        self,
        selected_states: Sequence[DiffusionRequestState],
    ) -> None:
        self._refresh_dynamic_fields(selected_states)

    def _rebuild(
        self,
        selected_states: Sequence[DiffusionRequestState],
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        req_ids: list[str],
    ) -> InputBatch:
        self.req_ids = req_ids
        self.num_reqs = len(req_ids)
        self.num_reqs_after_padding = len(req_ids)
        self.idx_mapping = idx_mapping
        self.idx_mapping_np = idx_mapping_np
        self.latents = _prepare_latents(selected_states, out=self.latents)
        self.timesteps = _prepare_timesteps(selected_states, out=self.timesteps)
        self._refresh_static_fields(selected_states)
        self.__post_init__()
        return self

    @classmethod
    def make_batch(
        cls,
        states: Sequence[DiffusionRequestState],
        idx_mapping: torch.Tensor | None = None,
        cached_batch: InputBatch | None = None,
    ) -> InputBatch:
        """Build a temporary step-local batch view from request states."""
        selected_states, idx_mapping, idx_mapping_np = _select_states(states, idx_mapping)
        req_ids = _prepare_req_ids(selected_states)

        if _same_composition(cached_batch, req_ids, idx_mapping_np):
            assert cached_batch is not None
            cached_batch._repack_dynamic_fields(selected_states)
            return cached_batch

        if cached_batch is not None:
            return cached_batch._rebuild(
                selected_states,
                idx_mapping,
                idx_mapping_np,
                req_ids,
            )

        prompt_embeds, prompt_embeds_mask = _prepare_prompt_embeds(selected_states)
        negative_prompt_embeds, negative_prompt_embeds_mask = _prepare_negative_prompt_embeds(selected_states)
        do_true_cfg, true_cfg_scale, cfg_normalize = _prepare_cfg_scalars(selected_states)
        return cls(
            req_ids=req_ids,
            num_reqs=len(selected_states),
            num_reqs_after_padding=len(selected_states),
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            latents=_prepare_latents(selected_states),
            timesteps=_prepare_timesteps(selected_states),
            guidance=_prepare_guidance(selected_states),
            do_true_cfg=do_true_cfg,
            true_cfg_scale=true_cfg_scale,
            cfg_normalize=cfg_normalize,
            image_latents=_prepare_image_latents(selected_states),
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            img_shapes=_prepare_img_shapes(selected_states),
            txt_seq_lens=_prepare_seq_lens(selected_states, "txt_seq_lens"),
            negative_txt_seq_lens=_prepare_seq_lens(
                selected_states,
                "negative_txt_seq_lens",
            ),
        )


def scatter_latents(
    states: Sequence[DiffusionRequestState],
    input_batch: InputBatch,
) -> None:
    """Scatter the step-updated latents back into persistent request states.

    This is the CPU fallback of the vLLM-style post-update path. The mapping is
    driven entirely by ``input_batch.idx_mapping_np`` so the runner remains free
    to keep request states in its own persistent storage layout.
    """
    _scatter_batch_tensor_by_mapping(
        states,
        input_batch.idx_mapping_np,
        attr_name="latents",
        value=input_batch.latents,
    )


DiffusionInputBatch = InputBatch
