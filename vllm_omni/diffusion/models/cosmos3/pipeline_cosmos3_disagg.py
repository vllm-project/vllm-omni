# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tower-split Cosmos3 T2I pipelines: reasoner (UND tower) and generator (GEN tower + VAE).

The reasoner runs the UND tower once per CFG branch and emits its per-layer text
K/V, keyed by a hash of the branch's token ids. The generator replaces the UND
tower with ``_ReplayLanguageModel``, so the stock ``forward`` gets that K/V back
whenever it would have run the tower. The topology is in ``cosmos3_pipeline_config.py``.
"""

from __future__ import annotations

import hashlib
from typing import Any, ClassVar

import torch
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.distributed.parallel_state import model_parallel_is_initialized

from vllm_omni.diffusion.models.cosmos3_pipeline_config import COSMOS3_UND_KV_KEY as KV_KEY

# The get_cosmos3_* process funcs are re-exported: they are resolved from the module the arch maps to.
from .pipeline_cosmos3 import (  # noqa: F401
    COSMOS3_DEFAULT_MAX_SEQUENCE_LENGTH,
    COSMOS3_T2I_DEFAULT_GUIDANCE_SCALE,
    COSMOS3_T2I_DEFAULT_HEIGHT,
    COSMOS3_T2I_DEFAULT_WIDTH,
    Cosmos3EdgeVFMTransformer,
    Cosmos3OmniDiffusersPipeline,
    get_cosmos3_ir_op_priority_func,
    get_cosmos3_post_process_func,
    get_cosmos3_pre_process_func,
    resolve_cosmos3_transformer_cls,
)


def fingerprint_text_ids(text_ids: torch.Tensor) -> str:
    """Content hash of a token-id tensor, used as the CFG branch key."""
    flat = text_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1).numpy()
    return hashlib.sha256(flat.tobytes()).hexdigest()[:32]


def _tp_world_size() -> int:
    return get_tensor_model_parallel_world_size() if model_parallel_is_initialized() else 1


def _tp_rank() -> int:
    return get_tensor_model_parallel_rank() if model_parallel_is_initialized() else 0


class _ReplayLanguageModel(torch.nn.Module):
    """Generator-side stand-in for the UND tower that returns the reasoner's K/V."""

    def __init__(self, rotary_emb: torch.nn.Module, kv_head_offset: int, num_kv_heads_local: int) -> None:
        super().__init__()
        # Still used by ``_compute_rope_freqs`` for the GEN frequencies.
        self.rotary_emb = rotary_emb
        self.layers = torch.nn.ModuleList()
        self.kv_heads = slice(kv_head_offset, kv_head_offset + num_kv_heads_local)
        self.table: dict[str, list] = {}
        self.dtype: torch.dtype | None = None

    def forward(self, text_ids: torch.Tensor, freqs: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
        key = fingerprint_text_ids(text_ids)
        if key not in self.table:
            raise RuntimeError(
                f"Cosmos3 generator stage has no reasoner K/V for prompt branch {key}; "
                "both stages must tokenize the prompt identically."
            )
        # Each TP rank takes its own slice of the gathered KV heads.
        return [
            (
                k[..., self.kv_heads, :].to(text_ids.device, self.dtype),
                v[..., self.kv_heads, :].to(text_ids.device, self.dtype),
            )
            for k, v in self.table[key]
        ]


def _reject_edge(od_config: Any) -> None:
    if resolve_cosmos3_transformer_cls(od_config.tf_model_config) is Cosmos3EdgeVFMTransformer:
        raise ValueError("Cosmos3 tower disaggregation does not support Cosmos3 Edge models.")


class Cosmos3ReasonerPipeline(Cosmos3OmniDiffusersPipeline):
    """Stage 0: runs only the UND tower and emits per-layer text K/V."""

    cosmos3_owned_towers: ClassVar[tuple[str, ...]] = ("reasoner",)
    cosmos3_loads_vae: ClassVar[bool] = False
    _vae_modules: ClassVar[list[str]] = []
    # The engine's warmup request is not T2I.
    dummy_run_num_frames: ClassVar[int] = 0

    def __init__(self, *, od_config: Any, prefix: str = "") -> None:
        _reject_edge(od_config)
        super().__init__(od_config=od_config, prefix=prefix)

    @torch.no_grad()
    def forward(self, req: Any) -> Any:  # type: ignore[override]
        from vllm_omni.diffusion.data import DiffusionOutput

        if not self._is_t2i_request(req):
            raise ValueError("Cosmos3 tower disaggregation supports text-to-image only (modalities=['image']).")
        if len(req.prompts) > 1:
            raise ValueError("Cosmos3 tower disaggregation supports a single prompt per request.")
        prompt_data = req.prompts[0] if req.prompts else ""
        if isinstance(prompt_data, str):
            prompt, negative_prompt = prompt_data, ""
        else:
            prompt = prompt_data.get("prompt", "")
            negative_prompt = prompt_data.get("negative_prompt") or ""

        # Must resolve exactly as the stock T2I ``forward`` does on the generator.
        sp = req.sampling_params
        height = sp.height or COSMOS3_T2I_DEFAULT_HEIGHT
        width = sp.width or COSMOS3_T2I_DEFAULT_WIDTH
        frame_rate = self._get_sp_param(sp, "resolved_frame_rate") or self._get_sp_param(sp, "frame_rate") or 24.0
        max_sequence_length = (
            self._get_sp_param(sp, "max_sequence_length", COSMOS3_DEFAULT_MAX_SEQUENCE_LENGTH)
            or COSMOS3_DEFAULT_MAX_SEQUENCE_LENGTH
        )
        use_system_prompt = bool(self._get_sp_param(sp, "use_system_prompt", False))
        guidance_scale = self._resolve_guidance_scale(sp, COSMOS3_T2I_DEFAULT_GUIDANCE_SCALE)

        cond_ids, cond_mask, uncond_ids, uncond_mask = self._format_and_tokenize_prompts(
            prompt,
            negative_prompt,
            1,
            frame_rate,
            height,
            width,
            max_sequence_length,
            sp,
            use_system_prompt,
            is_t2i=True,
        )
        branches = [(cond_ids, cond_mask)]
        if guidance_scale > 1.0 and uncond_ids is not None:
            branches.append((uncond_ids, uncond_mask))

        transformer = self.transformer
        hp, wp, _, _ = transformer._pad_to_patch_size(
            height // self.vae_scale_factor_spatial, width // self.vae_scale_factor_spatial
        )
        dtype = self.dtype
        gather = _tp_world_size() > 1

        kv: dict[str, list] = {}
        for text_ids, text_mask in branches:
            text_ids, text_mask = text_ids.to(self.device), text_mask.to(self.device)
            max_real_len = int(text_mask.sum(dim=1).max().item())
            freqs_und, _ = transformer._compute_rope_freqs(text_mask, 1, hp, wp, None, self.device, dtype)
            with transformer._offload_context("reasoner"):
                und_kv = transformer.language_model(text_ids, freqs_und)
            layers_kv = []
            for k, v in und_kv:
                # Trim padding as the co-located forward does; ship the full KV head set.
                k, v = k[:, :max_real_len].contiguous(), v[:, :max_real_len].contiguous()
                if gather:
                    k = tensor_model_parallel_all_gather(k, dim=-2)
                    v = tensor_model_parallel_all_gather(v, dim=-2)
                layers_kv.append((k.cpu(), v.cpu()))
            kv[fingerprint_text_ids(text_ids)] = layers_kv
        return DiffusionOutput(output={KV_KEY: kv})


class Cosmos3GeneratorPipeline(Cosmos3OmniDiffusersPipeline):
    """Stage 1: runs only the GEN tower, replaying the reasoner's K/V."""

    cosmos3_owned_towers: ClassVar[tuple[str, ...]] = ("generator",)
    # The engine's warmup request carries no reasoner K/V.
    dummy_run_num_frames: ClassVar[int] = 0

    def __init__(self, *, od_config: Any, prefix: str = "") -> None:
        _reject_edge(od_config)
        super().__init__(od_config=od_config, prefix=prefix)
        cross_attention = self.transformer.gen_layers[0].cross_attention
        self.transformer.language_model = _ReplayLanguageModel(  # type: ignore[assignment]
            self.transformer.language_model.rotary_emb,
            kv_head_offset=_tp_rank() * cross_attention.num_kv_heads_local,
            num_kv_heads_local=cross_attention.num_kv_heads_local,
        )

    def forward(self, req: Any) -> Any:  # type: ignore[override]
        prompt_data = req.prompts[0] if req.prompts else None
        # Pop the K/V so the prompt echoed on the final output does not carry it.
        extra = prompt_data.pop("extra", None) if isinstance(prompt_data, dict) else None
        if not extra or KV_KEY not in extra:
            raise ValueError("Cosmos3 generator stage needs reasoner K/V in prompt['extra']; route through stage 0.")
        replay = self.transformer.language_model
        replay.table, replay.dtype = extra[KV_KEY], self.dtype
        try:
            return super().forward(req)
        finally:
            replay.table = {}


def get_cosmos3_reasoner_post_process_func(od_config: Any):
    """Pass the reasoner K/V through under ``trajectory``, which skips metadata validation."""
    del od_config

    def post_process_func(output: Any, output_type: str = "np", sampling_params: Any = None) -> Any:
        del output_type, sampling_params
        return {"payload": {"trajectory": output}}

    return post_process_func
