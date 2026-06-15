# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step-local diffusion input batches.

``DiffusionRequestState`` remains the durable source of truth. ``InputBatch``
is a per-step view that either exposes the existing dense tensors or, for
Qwen-Image step batching, a dynamic request-local view with no latent/text
padding in attention.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from vllm_omni.diffusion.worker.utils import DiffusionRequestState


@dataclass(frozen=True)
class RequestSpan:
    request_id: str
    request_index: int
    row_start: int
    row_count: int
    token_start: int
    token_count: int

    @property
    def row_end(self) -> int:
        return self.row_start + self.row_count

    @property
    def token_end(self) -> int:
        return self.token_start + self.token_count


@dataclass(frozen=True)
class PromptBatch:
    embeds: torch.Tensor | None
    mask: torch.Tensor | None
    seq_lens: list[int] | None


def _normalize_embeds(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 2:
        return value.unsqueeze(0)
    if value.ndim == 3:
        return value
    raise ValueError(f"prompt_embeds must be 2D or 3D, got shape={tuple(value.shape)}")


def _normalize_mask(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 1:
        value = value.unsqueeze(0)
    elif value.ndim != 2:
        raise ValueError(f"prompt mask must be 1D or 2D, got shape={tuple(value.shape)}")
    return value if value.dtype == torch.bool else value != 0


def _buffer_like(
    current: torch.Tensor | None,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if current is not None and tuple(current.shape) == shape and current.dtype == dtype and current.device == device:
        return current
    return torch.empty(shape, dtype=dtype, device=device)


def _select_states(
    states: Sequence[DiffusionRequestState],
    idx_mapping: torch.Tensor | None,
) -> tuple[list[DiffusionRequestState], torch.Tensor, np.ndarray]:
    if not states:
        raise ValueError("Cannot build InputBatch from empty states.")
    if idx_mapping is None:
        latents = states[0].latents
        device = latents.device if latents is not None else None
        idx_mapping = torch.arange(len(states), dtype=torch.int32, device=device)
    elif idx_mapping.ndim != 1:
        raise ValueError("idx_mapping must be a 1D tensor.")
    else:
        idx_mapping = idx_mapping.to(dtype=torch.int32)

    selected: list[DiffusionRequestState] = []
    for batch_idx, state_idx in enumerate(idx_mapping.tolist()):
        if state_idx < 0 or state_idx >= len(states):
            raise ValueError(f"idx_mapping[{batch_idx}]={state_idx} is out of range for states.")
        selected.append(states[state_idx])
    return selected, idx_mapping, idx_mapping.detach().cpu().numpy()


def _state_latents(state: DiffusionRequestState) -> torch.Tensor:
    if state.latents is None:
        raise ValueError(f"Request {state.request_id} has no latents.")
    return state.latents


def _gather_rows(
    values: Sequence[torch.Tensor],
    *,
    field_name: str,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if not values:
        raise ValueError(f"Cannot gather empty tensor list for {field_name}.")
    dtype = values[0].dtype
    device = values[0].device
    suffix = tuple(values[0].shape[1:])
    total_rows = 0
    for value in values:
        if value.dtype != dtype:
            raise ValueError(f"Mixed dtypes in {field_name} batch.")
        if value.device != device:
            raise ValueError(f"Mixed devices in {field_name} batch.")
        if tuple(value.shape[1:]) != suffix:
            raise ValueError(
                f"Mixed trailing shapes in {field_name} batch: expected {suffix}, got {tuple(value.shape[1:])}."
            )
        total_rows += int(value.shape[0])

    output = _buffer_like(out, (total_rows, *suffix), dtype=dtype, device=device)
    offset = 0
    for value in values:
        next_offset = offset + int(value.shape[0])
        output[offset:next_offset].copy_(value)
        offset = next_offset
    return output


def _expand_rows(value: torch.Tensor, row_count: int, *, field_name: str) -> torch.Tensor:
    if value.ndim == 0:
        return value.reshape(1).expand(row_count)
    if value.ndim != 1:
        raise ValueError(f"{field_name} must be scalar or 1D, got ndim={value.ndim}.")
    if value.shape[0] == row_count:
        return value
    if value.shape[0] == 1:
        return value.expand(row_count)
    raise ValueError(f"{field_name} must have either 1 or {row_count} elements, got {value.shape[0]}.")


def _prompt_field(
    state: DiffusionRequestState,
    *,
    embeds_attr: str,
    mask_attr: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    embeds = getattr(state, embeds_attr)
    if embeds is None:
        return None, None
    embeds = _normalize_embeds(embeds)
    setattr(state, embeds_attr, embeds)

    mask = getattr(state, mask_attr)
    if mask is None:
        return embeds, None
    mask = _normalize_mask(mask)
    setattr(state, mask_attr, mask)
    if mask.shape[0] != embeds.shape[0]:
        raise ValueError(f"{mask_attr} batch dimension does not match {embeds_attr} for request {state.request_id}.")
    return embeds, mask


def _seq_lens(
    state: DiffusionRequestState,
    *,
    embeds_attr: str,
    mask_attr: str,
    seq_lens_attr: str,
) -> list[int]:
    embeds, mask = _prompt_field(state, embeds_attr=embeds_attr, mask_attr=mask_attr)
    if embeds is None:
        raise ValueError(f"{embeds_attr} is not initialized on request {state.request_id}.")
    if mask is not None:
        return mask.sum(dim=1, dtype=torch.int32).tolist()
    seq_lens = getattr(state, seq_lens_attr)
    if seq_lens is not None:
        return [int(value) for value in seq_lens]
    return [int(embeds.shape[1])] * int(embeds.shape[0])


def _pad_prompt(value: torch.Tensor, target_len: int) -> torch.Tensor:
    value = _normalize_embeds(value)
    if value.shape[1] == target_len:
        return value
    out = value.new_zeros((value.shape[0], target_len, value.shape[2]))
    out[:, : value.shape[1]].copy_(value)
    return out


def _pad_mask(value: torch.Tensor, target_len: int) -> torch.Tensor:
    value = _normalize_mask(value)
    if value.shape[1] == target_len:
        return value
    out = torch.zeros((value.shape[0], target_len), dtype=torch.bool, device=value.device)
    out[:, : value.shape[1]].copy_(value)
    return out


def _build_prompt_batch(
    states: Sequence[DiffusionRequestState],
    *,
    embeds_attr: str,
    mask_attr: str,
    seq_lens_attr: str,
    required: bool,
    embeds_out: torch.Tensor | None = None,
    mask_out: torch.Tensor | None = None,
) -> PromptBatch:
    fields = [_prompt_field(state, embeds_attr=embeds_attr, mask_attr=mask_attr) for state in states]
    if not any(embeds is not None for embeds, _ in fields):
        if required:
            raise ValueError(f"All requests must have `{embeds_attr}` initialized.")
        return PromptBatch(None, None, None)
    if not all(embeds is not None for embeds, _ in fields):
        raise ValueError(f"Mixed {embeds_attr} in batch.")

    seq_lens_by_state = [
        _seq_lens(state, embeds_attr=embeds_attr, mask_attr=mask_attr, seq_lens_attr=seq_lens_attr) for state in states
    ]
    target_len = max(max(lengths) for lengths in seq_lens_by_state)

    embeds_values: list[torch.Tensor] = []
    mask_values: list[torch.Tensor] = []
    any_mask = any(mask is not None for _, mask in fields)
    if any_mask and not all(mask is not None for _, mask in fields):
        raise ValueError(f"Mixed {mask_attr} in batch.")

    for state, (embeds, mask), lengths in zip(states, fields, seq_lens_by_state, strict=True):
        assert embeds is not None
        if mask is None and max(lengths) != target_len:
            raise ValueError(
                f"Variable-length {embeds_attr} in batch but {mask_attr} is None for request {state.request_id}."
            )
        embeds = _pad_prompt(embeds, target_len)
        setattr(state, embeds_attr, embeds)
        embeds_values.append(embeds)
        if mask is not None:
            mask = _pad_mask(mask, target_len)
            setattr(state, mask_attr, mask)
            mask_values.append(mask)

    embeds = _gather_rows(embeds_values, field_name=embeds_attr, out=embeds_out)
    mask = _gather_rows(mask_values, field_name=mask_attr, out=mask_out) if mask_values else None
    return PromptBatch(
        embeds,
        mask,
        [int(length) for lengths in seq_lens_by_state for length in lengths],
    )


def _metadata_list(states: Sequence[DiffusionRequestState], attr_name: str) -> list | None:
    values = [getattr(state, attr_name) for state in states]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"Mixed {attr_name} in batch.")
    if attr_name == "img_shapes":
        return [value[0] if value else [] for value in values]
    return [int(value[0]) for value in values]


def _cfg_values(states: Sequence[DiffusionRequestState]) -> tuple[bool, float, bool, list[float], list[bool]]:
    do_cfg = bool(states[0].do_true_cfg)
    scales: list[float] = []
    normalizes: list[bool] = []
    for state in states:
        if bool(state.do_true_cfg) != do_cfg:
            raise ValueError("Mixed CFG execution shapes in one diffusion batch are not supported.")
        scale = getattr(state.sampling, "true_cfg_scale", None)
        scales.append(4.0 if scale is None else float(scale))
        normalizes.append(bool(getattr(state.sampling, "cfg_normalize", False)))
    return do_cfg, scales[0], normalizes[0], scales, normalizes


def _guidance(states: Sequence[DiffusionRequestState], out: torch.Tensor | None = None) -> torch.Tensor | None:
    values = [state.guidance for state in states]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("Mixed guidance in one diffusion batch is not supported.")
    tensors: list[torch.Tensor] = []
    for state in states:
        latents = _state_latents(state)
        guidance = torch.as_tensor(state.guidance, device=latents.device, dtype=latents.dtype)
        tensors.append(_expand_rows(guidance, int(latents.shape[0]), field_name="guidance tensor"))
    return _gather_rows(tensors, field_name="guidance", out=out)


def _timesteps(states: Sequence[DiffusionRequestState], out: torch.Tensor | None = None) -> torch.Tensor:
    tensors: list[torch.Tensor] = []
    for state in states:
        timestep = state.current_timestep
        if timestep is None or not torch.is_tensor(timestep):
            raise ValueError("All requests must have tensor current timesteps.")
        tensors.append(_expand_rows(timestep, int(_state_latents(state).shape[0]), field_name="timestep tensor"))
    return _gather_rows(tensors, field_name="timesteps", out=out)


def _image_latents(states: Sequence[DiffusionRequestState], out: torch.Tensor | None = None) -> torch.Tensor | None:
    values = [getattr(state.sampling, "image_latent", None) for state in states]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("Mixed image_latent presence in one diffusion batch is not supported.")
    return _gather_rows([value for value in values if value is not None], field_name="image_latents", out=out)


def _request_spans(states: Sequence[DiffusionRequestState]) -> list[RequestSpan]:
    spans: list[RequestSpan] = []
    row_start = 0
    token_start = 0
    for request_index, state in enumerate(states):
        latents = _state_latents(state)
        row_count = int(latents.shape[0])
        token_count = int(latents.shape[1])
        spans.append(
            RequestSpan(
                request_id=state.request_id,
                request_index=request_index,
                row_start=row_start,
                row_count=row_count,
                token_start=token_start,
                token_count=token_count,
            )
        )
        row_start += row_count
        token_start += token_count
    return spans


def _dense_latents(
    states: Sequence[DiffusionRequestState],
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    return _gather_rows([_state_latents(state) for state in states], field_name="latents", out=out)


def _needs_dynamic_batch(
    states: Sequence[DiffusionRequestState],
    *,
    positive_lens: list[int] | None,
    negative_lens: list[int] | None,
    cfg_scales: list[float],
    cfg_normalize: list[bool],
) -> bool:
    latents = [_state_latents(state) for state in states]
    latent_shapes = {tuple(value.shape[1:]) for value in latents}
    row_counts = {int(value.shape[0]) for value in latents}
    dynamic = False
    if len(latent_shapes) > 1:
        if len({value.dtype for value in latents}) > 1 or len({value.device for value in latents}) > 1:
            raise ValueError("Dynamic latents require matching dtype and device.")
        if len({tuple(value.shape[2:]) for value in latents}) > 1:
            raise ValueError("Dynamic latents only support varying sequence length.")
        dynamic = True
    if positive_lens is not None and len(set(positive_lens)) > 1:
        dynamic = True
    if negative_lens is not None and len(set(negative_lens)) > 1:
        dynamic = True
    if len(set(cfg_scales)) > 1 or len(set(cfg_normalize)) > 1:
        dynamic = True
    if dynamic and row_counts != {1}:
        raise ValueError("Dynamic Qwen-Image batching currently supports one output row per request.")
    return dynamic


def _can_use_dynamic_request_view(states: Sequence[DiffusionRequestState]) -> bool:
    return {int(_state_latents(state).shape[0]) for state in states} == {1}


def _should_use_dynamic_batch(
    states: Sequence[DiffusionRequestState],
    *,
    allow_dynamic: bool,
    positive_lens: list[int] | None,
    negative_lens: list[int] | None,
    cfg_scales: list[float],
    cfg_normalize: list[bool],
) -> bool:
    if not allow_dynamic:
        return False
    return _needs_dynamic_batch(
        states,
        positive_lens=positive_lens,
        negative_lens=negative_lens,
        cfg_scales=cfg_scales,
        cfg_normalize=cfg_normalize,
    ) or (len(states) == 1 and _can_use_dynamic_request_view(states))


@dataclass
class InputBatch:
    request_ids: list[str]
    num_reqs: int
    num_reqs_after_padding: int
    idx_mapping: torch.Tensor
    idx_mapping_np: np.ndarray

    latents: torch.Tensor | None
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

    is_dynamic: bool = False
    dynamic_latents: list[torch.Tensor] | None = None
    request_spans: list[RequestSpan] = field(default_factory=list)
    true_cfg_scales: list[float] = field(default_factory=list)
    cfg_normalize_flags: list[bool] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.request_ids) != int(self.idx_mapping.numel()):
            raise ValueError("`request_ids` and `idx_mapping` must have the same length.")
        if self.num_reqs != len(self.request_ids):
            raise ValueError("`num_reqs` must match request_ids.")
        if self.num_reqs_after_padding < self.num_reqs:
            raise ValueError("`num_reqs_after_padding` must be >= `num_reqs`.")
        if self.is_dynamic and not self.dynamic_latents:
            raise ValueError("Dynamic InputBatch requires dynamic_latents.")
        if not self.is_dynamic and self.latents is None:
            raise ValueError("Dense InputBatch requires latents.")

    def _refresh(
        self,
        states: Sequence[DiffusionRequestState],
        *,
        allow_dynamic: bool,
        static: bool,
    ) -> InputBatch:
        timesteps = _timesteps(states, out=self.timesteps)
        request_spans = _request_spans(states)
        latents_out = self.latents if not self.is_dynamic else None

        prompt = (
            _build_prompt_batch(
                states,
                embeds_attr="prompt_embeds",
                mask_attr="prompt_embeds_mask",
                seq_lens_attr="txt_seq_lens",
                required=True,
                embeds_out=self.prompt_embeds if static else None,
                mask_out=self.prompt_embeds_mask if static else None,
            )
            if static
            else PromptBatch(self.prompt_embeds, self.prompt_embeds_mask, self.txt_seq_lens)
        )
        negative = (
            _build_prompt_batch(
                states,
                embeds_attr="negative_prompt_embeds",
                mask_attr="negative_prompt_embeds_mask",
                seq_lens_attr="negative_txt_seq_lens",
                required=False,
                embeds_out=self.negative_prompt_embeds if static else None,
                mask_out=self.negative_prompt_embeds_mask if static else None,
            )
            if static
            else PromptBatch(self.negative_prompt_embeds, self.negative_prompt_embeds_mask, self.negative_txt_seq_lens)
        )
        do_cfg, cfg_scale, cfg_norm, cfg_scales, cfg_norms = _cfg_values(states)
        is_dynamic = _should_use_dynamic_batch(
            states,
            allow_dynamic=allow_dynamic,
            positive_lens=prompt.seq_lens,
            negative_lens=negative.seq_lens,
            cfg_scales=cfg_scales,
            cfg_normalize=cfg_norms,
        )

        self.timesteps = timesteps
        self.request_spans = request_spans
        self.do_true_cfg = do_cfg
        self.true_cfg_scale = cfg_scale
        self.cfg_normalize = cfg_norm
        self.true_cfg_scales = cfg_scales
        self.cfg_normalize_flags = cfg_norms
        self.guidance = _guidance(states, out=self.guidance)

        if static:
            self.prompt_embeds = prompt.embeds
            self.prompt_embeds_mask = prompt.mask
            self.negative_prompt_embeds = negative.embeds
            self.negative_prompt_embeds_mask = negative.mask
            self.img_shapes = _metadata_list(states, "img_shapes")
            self.txt_seq_lens = prompt.seq_lens
            self.negative_txt_seq_lens = negative.seq_lens
            self.image_latents = _image_latents(states, out=self.image_latents)

        self.is_dynamic = is_dynamic
        if is_dynamic:
            if self.image_latents is not None:
                raise ValueError("Dynamic Qwen-Image step batching does not support image_latents yet.")
            self.latents = None
            self.dynamic_latents = [_state_latents(state) for state in states]
        else:
            self.latents = _dense_latents(states, out=latents_out)
            self.dynamic_latents = None
        self.__post_init__()
        return self

    @classmethod
    def make_batch(
        cls,
        states: Sequence[DiffusionRequestState],
        idx_mapping: torch.Tensor | None = None,
        cached_batch: InputBatch | None = None,
        allow_dynamic: bool = False,
    ) -> InputBatch:
        selected, idx_mapping, idx_mapping_np = _select_states(states, idx_mapping)
        request_ids = [state.request_id for state in selected]

        same = (
            cached_batch is not None
            and cached_batch.request_ids == request_ids
            and np.array_equal(cached_batch.idx_mapping_np, idx_mapping_np)
        )
        if same:
            assert cached_batch is not None
            return cached_batch._refresh(selected, allow_dynamic=allow_dynamic, static=False)

        prompt = _build_prompt_batch(
            selected,
            embeds_attr="prompt_embeds",
            mask_attr="prompt_embeds_mask",
            seq_lens_attr="txt_seq_lens",
            required=True,
        )
        negative = _build_prompt_batch(
            selected,
            embeds_attr="negative_prompt_embeds",
            mask_attr="negative_prompt_embeds_mask",
            seq_lens_attr="negative_txt_seq_lens",
            required=False,
        )
        do_cfg, cfg_scale, cfg_norm, cfg_scales, cfg_norms = _cfg_values(selected)
        is_dynamic = _should_use_dynamic_batch(
            selected,
            allow_dynamic=allow_dynamic,
            positive_lens=prompt.seq_lens,
            negative_lens=negative.seq_lens,
            cfg_scales=cfg_scales,
            cfg_normalize=cfg_norms,
        )
        image_latents = _image_latents(selected)
        if is_dynamic and image_latents is not None:
            raise ValueError("Dynamic Qwen-Image step batching does not support image_latents yet.")

        batch = cls(
            request_ids=request_ids,
            num_reqs=len(selected),
            num_reqs_after_padding=len(selected),
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            latents=None if is_dynamic else _dense_latents(selected),
            timesteps=_timesteps(selected),
            guidance=_guidance(selected),
            do_true_cfg=do_cfg,
            true_cfg_scale=cfg_scale,
            cfg_normalize=cfg_norm,
            image_latents=image_latents,
            prompt_embeds=prompt.embeds,
            prompt_embeds_mask=prompt.mask,
            negative_prompt_embeds=negative.embeds,
            negative_prompt_embeds_mask=negative.mask,
            img_shapes=_metadata_list(selected, "img_shapes"),
            txt_seq_lens=prompt.seq_lens,
            negative_txt_seq_lens=negative.seq_lens,
            is_dynamic=is_dynamic,
            dynamic_latents=[_state_latents(state) for state in selected] if is_dynamic else None,
            request_spans=_request_spans(selected),
            true_cfg_scales=cfg_scales,
            cfg_normalize_flags=cfg_norms,
        )
        return batch


def scatter_latents(states: Sequence[DiffusionRequestState], input_batch: InputBatch) -> None:
    if input_batch.is_dynamic:
        return
    if input_batch.latents is None:
        raise ValueError("Dense scatter requires input_batch.latents.")

    offset = 0
    for state_idx in input_batch.idx_mapping_np.tolist():
        if state_idx < 0 or state_idx >= len(states):
            raise ValueError(f"idx_mapping contains out-of-range state index {state_idx}.")
        state = states[state_idx]
        row_count = int(_state_latents(state).shape[0])
        value = input_batch.latents[offset : offset + row_count]
        if tuple(state.latents.shape) == tuple(value.shape) and state.latents.dtype == value.dtype:
            state.latents.copy_(value)
        else:
            state.latents = value.clone()
        offset += row_count
    if offset != int(input_batch.latents.shape[0]):
        raise ValueError(f"Scatter consumed {offset} rows, but batch has {input_batch.latents.shape[0]} rows.")


DiffusionInputBatch = InputBatch
