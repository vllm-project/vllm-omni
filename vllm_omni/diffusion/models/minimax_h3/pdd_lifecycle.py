# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PDD adapter lifecycle and request validation for the MiniMax-H3 pipeline."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from vllm_omni.errors import OmniClientError

from .pdd import PDDAdapter, PDDConfig, PDDParallelHead, load_minimax_h3_pdd_lora

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
        self._pdd_adapter_ids.discard(lora_request.lora_int_id)
        # Drop any cached PDD head install handle from a prior load with the
        # same id (eviction + re-add). The installed-adapter handle must go too:
        # with two PDD releases in play, a reused id could otherwise keep a
        # handle pointing at the other variant's bank and DiT.
        self._pdd_adapters.pop(lora_request.lora_int_id, None)
        self._pdd_active_adapters.pop(lora_request.lora_int_id, None)
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
        # bank to a DiT that never saw them, on a schedule pinned to 9 steps --
        # a silently bad video rather than an error. Refuse instead.
        if task is not None and task not in cfg.tasks:
            raise OmniClientError(
                f"MiniMax-H3 PDD {cfg.variant} 8-step artifact ({cfg.filename}) serves "
                f"{sorted(cfg.tasks)}, got task={task!r}; use the {task} artifact or drop the "
                "lora field to fall back to the undistilled schedule"
            )
        sigma_points = sampling.num_inference_steps
        if sigma_points != cfg.sigma_points:
            raise OmniClientError(
                f"MiniMax-H3 PDD {cfg.variant} {cfg.nfe}-step requires "
                f"num_inference_steps={cfg.sigma_points} "
                f"({cfg.nfe} NFE + terminal zero), got {sigma_points}"
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
        if getattr(dit, "_pdd_adapter", None) is adapter:
            return adapter
        # Only install if this DiT hasn't had heads replaced by this adapter
        # already (possible on re-activation after deactivation).
        if not isinstance(dit.final_layer.video_out, PDDParallelHead):
            adapter.install_heads(dit)
        adapter.load_head_bank(dit, info["head_weights"], info["head_biases"])
        dit._pdd_adapter = adapter
        self._pdd_active_adapters[lora_id] = adapter
        return adapter

    def _deactivate_pdd_heads(self, lora_id: int | None = None) -> None:
        """Release references to installed PDD adapters and disarm their heads.

        We do NOT swap the heads back to plain ColumnParallelLinear here
        because that would require saving originals and breaks fp8/fp32
        state. Instead each adapter's ``disarm`` resets its heads' plan back
        to the saved original base weights, so a later request that
        reuses the same DiT without this adapter (no-LoRA, Turbo, or a
        different PDD artifact) does not keep running through this
        adapter's last-armed per-step plan."""
        if lora_id is None:
            adapters = list(self._pdd_active_adapters.values())
            self._pdd_active_adapters.clear()
        else:
            adapter = self._pdd_active_adapters.pop(lora_id, None)
            adapters = [adapter] if adapter is not None else []
        for adapter in adapters:
            dit = getattr(self, adapter.config.dit_component, None)
            if dit is not None and getattr(dit, "_pdd_adapter", None) is adapter:
                adapter.disarm(dit)
                dit._pdd_adapter = None

    @staticmethod
    def _reset_pdd_heads(transformer: nn.Module) -> None:
        final_layer = getattr(transformer, "final_layer", None)
        for name in ("video_out", "audio_out"):
            head = getattr(final_layer, name, None)
            if isinstance(head, PDDParallelHead):
                head.reset_plan()

    def _remove_diffusion_lora_adapter(self, adapter_id: int) -> None:
        """Release model-owned banks when the manager removes or evicts an ID."""
        self._ensure_pdd_bookkeeping()
        self._deactivate_pdd_heads(adapter_id)
        self._pdd_adapters.pop(adapter_id, None)
        self._pdd_adapter_ids.discard(adapter_id)
