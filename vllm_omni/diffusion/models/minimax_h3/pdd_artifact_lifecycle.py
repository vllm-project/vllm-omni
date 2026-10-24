# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PDD adapter lifecycle and request validation for the MiniMax-H3 pipeline."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from vllm_omni.errors import OmniClientError

from .pdd_artifact import PDDAdapter, PDDConfig, load_minimax_h3_pdd_lora
from .sampling import normalize_h3_sampler

if TYPE_CHECKING:
    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.peft_helper import PEFTHelper

    from vllm_omni.lora.request import LoRARequest


class MiniMaxH3PDDLifecycleMixin:
    """Own PDD banks, head activation, eviction and adapter contracts.

    The pipeline supplies its partition, DiT components and sampling defaults.
    Denoising stays in the pipeline and calls these lifecycle hooks. Keeping
    the registries on the pipeline avoids registering an extra nn.Module or
    retaining a second reference to its model components.
    """

    def _ensure_pdd_bookkeeping(self) -> None:
        """Create the PDD adapter registries if ``__init__`` has not run.

        ``__init__`` always populates these, but the LoRA entry points are also
        exercised on partially constructed pipelines, where an AttributeError
        raised from PDD bookkeeping would mask the turbo / generic PEFT path
        under test (same reason ``_base_schedule_by_partition`` has a default).
        Writing through ``__dict__`` keeps nn.Module's ``__setattr__`` out of it.
        """
        state = self.__dict__
        state.setdefault("_pdd_adapters", {})
        state.setdefault("_pdd_adapter_ids", set())
        state.setdefault("_pdd_active_adapters", {})

    def _load_pdd_lora_adapter(
        self,
        *,
        lora_request: LoRARequest,
        lora_path: str | Path,
        dtype: torch.dtype,
        unsupported_offload_mode: str | None,
    ) -> tuple[LoRAModel, PEFTHelper] | None:
        self._ensure_pdd_bookkeeping()
        self._remove_diffusion_lora_adapter(lora_request.lora_int_id)
        # Try PDD first: the PDD loader raises if the path names the PDD
        # artifact but its contents are malformed, and returns None for a
        # non-PDD path so turbo / generic PEFT get their chance.
        pdd_loaded = load_minimax_h3_pdd_lora(
            partition=self.partition,
            lora_request=lora_request,
            lora_path=lora_path,
            dtype=dtype,
            unsupported_offload_mode=unsupported_offload_mode,
        )
        if pdd_loaded is not None:
            lora_model, peft_helper, pdd_cfg, head_weights, head_biases = pdd_loaded
            self._pdd_adapters[lora_request.lora_int_id] = {
                "cfg": pdd_cfg,
                "head_weights": head_weights,
                "head_biases": head_biases,
            }
            self._pdd_adapter_ids.add(lora_request.lora_int_id)
            return lora_model, peft_helper
        return None

    def _validate_pdd_lora_binding(self, *, lora_model: LoRAModel, bound_lora_names: frozenset[str]) -> bool:
        """Validate PDD trunk targets; return whether this is a PDD adapter."""
        # Head banks are installed separately via _ensure_pdd_heads().
        # Every targeted trunk module must still be bound.
        self._ensure_pdd_bookkeeping()
        if lora_model.id in self._pdd_adapter_ids:
            missing = sorted(
                name
                for name in lora_model.loras
                if name not in bound_lora_names
                # Final-layer adaln is not targeted by PDD (artifact contains
                # only block-level adaln); don't require it.
                and ".final_layer." not in name
            )
            if missing:
                raise ValueError(
                    "MiniMax-H3 PDD LoRA trunk binding is incomplete: "
                    f"bound={len(bound_lora_names)}/{len(lora_model.loras)}, missing={missing[:5]}"
                )
            return True
        return False

    def _has_active_pdd_lora(self, sampling: Any) -> bool:
        lora_request = sampling.lora_request
        self._ensure_pdd_bookkeeping()
        return (
            lora_request is not None
            and not math.isclose(0.0, float(sampling.lora_scale))
            and lora_request.lora_int_id in self._pdd_adapter_ids
        )

    def _validate_pdd_sampling(self, sampling: Any, task: str | None = None) -> PDDConfig:
        extra = sampling.extra_args or {}
        cfg = self._pdd_adapters[sampling.lora_request.lora_int_id]["cfg"]
        try:
            sampler = normalize_h3_sampler(extra.get("sampler"))
        except ValueError as exc:
            raise OmniClientError(str(exc)) from exc
        # The parallel head bank fuses four trained Euler updates per call.
        # A multistep solver would reinterpret this fused velocity using a
        # denoised history the artifact was never trained against.
        if sampler != "euler":
            raise OmniClientError(f"MiniMax-H3 PDD {cfg.variant} requires sampler='euler', got {sampler!r}")
        # The head bank is swapped in at full strength (load_head_bank has no
        # scale knob); only the trunk LoRA delta honors lora_scale. A
        # fractional scale would silently blend a scaled trunk with an
        # unscaled distilled head, which isn't the trained PDD model and
        # isn't the requested scale either. Require full strength.
        lora_scale = float(sampling.lora_scale)
        if not math.isclose(lora_scale, 1.0):
            raise OmniClientError(
                f"MiniMax-H3 PDD {cfg.variant} 8-step artifact only supports lora_scale=1.0 "
                f"(the fused head bank is installed at full strength regardless of trunk scale), "
                f"got {lora_scale:g}"
            )
        # Each release is distilled against one DiT: Ref2VA against
        # ``transformers_ref``, FL2VA against ``transformer`` (which also serves
        # t2va). Accepting the wrong one would bind the trunk delta and head
        # bank to a DiT that never saw them, on a schedule pinned to 8 evaluations --
        # a silently bad video rather than an error. Refuse instead.
        if task is not None and task not in cfg.tasks:
            raise OmniClientError(
                f"MiniMax-H3 PDD {cfg.variant} 8-step artifact ({cfg.filename}) serves "
                f"{sorted(cfg.tasks)}, got task={task!r}; use the {task} artifact or drop the "
                "lora field to fall back to the undistilled schedule"
            )
        num_steps = sampling.num_inference_steps
        if num_steps != cfg.nfe:
            raise OmniClientError(
                f"MiniMax-H3 PDD {cfg.variant} {cfg.nfe}-step requires "
                f"num_inference_steps={cfg.nfe} "
                f"({cfg.nfe} denoiser evaluations, {cfg.sigma_points} sigma points), got {num_steps}"
            )
        try:
            video_shift = float(extra.get("flow_shift", self.default_video_shift))
        except (TypeError, ValueError) as exc:
            raise OmniClientError(f"MiniMax-H3 PDD requires flow_shift={cfg.video_shift:g}") from exc
        if not math.isclose(video_shift, cfg.video_shift):
            raise OmniClientError(f"MiniMax-H3 PDD requires flow_shift={cfg.video_shift:g}, got {video_shift:g}")
        try:
            audio_shift = float(extra.get("audio_flow_shift", self.default_audio_shift))
        except (TypeError, ValueError) as exc:
            raise OmniClientError(f"MiniMax-H3 PDD requires audio_flow_shift={cfg.audio_shift:g}") from exc
        if not math.isclose(audio_shift, cfg.audio_shift):
            raise OmniClientError(f"MiniMax-H3 PDD requires audio_flow_shift={cfg.audio_shift:g}, got {audio_shift:g}")
        return cfg

    def _ensure_pdd_heads(self, lora_id: int) -> PDDAdapter:
        """Install PDD parallel heads on the artifact's own DiT.

        The LoRA manager has already wrapped the trunk linear layers by this
        point; we now swap final_layer.video_out / audio_out for PDDParallelHead
        modules (fp32, TP-sharded) and copy the artifact's 32-copy bank into
        them. Idempotent across activations.

        Only ``cfg.dit_component`` is touched. In a combined deployment the
        other DiT serves the other task family and has its own release, so
        installing a Ref2VA bank on the FL2VA DiT (as an earlier version did)
        would corrupt every fl2va/t2va request.
        """
        adapter = self._pdd_active_adapters.get(lora_id)
        info = self._pdd_adapters[lora_id]
        cfg: PDDConfig = info["cfg"]
        if adapter is None:
            v_plans, a_plans = cfg.plans()
            adapter = PDDAdapter(
                config=cfg,
                lora_id=lora_id,
                video_plans=v_plans,
                audio_plans=a_plans,
            )
        dit = getattr(self, cfg.dit_component, None)
        if dit is None:
            raise OmniClientError(
                f"MiniMax-H3 PDD {cfg.variant} artifact targets {cfg.dit_component!r}, "
                f"absent from a {self.partition!r} deployment"
            )
        if lora_id in self._pdd_active_adapters:
            return adapter
        adapter.install_heads(dit)
        adapter.load_head_bank(dit, info["head_weights"], info["head_biases"])
        self._pdd_active_adapters[lora_id] = adapter
        return adapter

    def _remove_diffusion_lora_adapter(self, adapter_id: int) -> None:
        """Release only this ID's model-owned banks on removal or eviction."""
        self._ensure_pdd_bookkeeping()
        adapter = self._pdd_active_adapters.pop(adapter_id, None)
        if adapter is not None:
            dit = getattr(self, adapter.config.dit_component, None)
            if dit is not None:
                adapter.remove_heads(dit)
        self._pdd_adapters.pop(adapter_id, None)
        self._pdd_adapter_ids.discard(adapter_id)
