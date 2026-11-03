# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tower-split Cosmos3 pipelines: reasoner (UND) and generator (GEN) stages.

Both subclass ``Cosmos3OmniDiffusersPipeline``; only the tower boundary differs.
The topology wiring the two stages lives in ``cosmos3_pipeline_config.py``.

* The reasoner builds only the UND tower and no VAE. It runs the UND tower once
  per CFG branch, all-gathers the TP-sharded KV heads, and ships per-layer K/V as
  a ``Cosmos3TextConditioning`` payload.
* The generator builds only the GEN tower and replaces ``language_model`` with
  ``_ReplayLanguageModel``. The stock ``Cosmos3VFMTransformer.forward`` calls it
  once per branch, as it would call the UND tower. Replacing the tower, rather
  than pre-setting ``cached_kv``, is needed because the T2I denoise loop resets
  that cache. Each branch is looked up by a fingerprint of its ``text_ids``, and
  each TP rank slices its own KV-head range out of the gathered payload, so the
  two stages may run different TP sizes.

Both stages still read the whole checkpoint. ``load_weights`` drops the unowned
tower's tensors after reading them, so the split saves device memory, not
startup I/O.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.distributed.parallel_state import model_parallel_is_initialized
from vllm.logger import init_logger

from vllm_omni.diffusion.models.cosmos3_pipeline_config import (
    COSMOS3_UND_KV_KEY as KV_KEY,
)
from vllm_omni.diffusion.models.cosmos3_pipeline_config import (
    COSMOS3_UND_META_KEY as META_KEY,
)
from vllm_omni.diffusion.models.cosmos3_pipeline_config import (
    COSMOS3_UND_PAYLOAD_WARN_MIB,
    COSMOS3_UND_SCHEMA,
)

from .pipeline_cosmos3 import (
    COSMOS3_T2I_DEFAULT_GUIDANCE_SCALE,
    Cosmos3OmniDiffusersPipeline,
)

# Re-exported because ``_load_process_func`` resolves process funcs from the
# module ``_DIFFUSION_MODELS`` maps both tower arch names to.
from .pipeline_cosmos3 import (  # noqa: F401  (re-export)
    get_cosmos3_ir_op_priority_func as get_cosmos3_ir_op_priority_func,
)
from .pipeline_cosmos3 import (  # noqa: F401  (re-export)
    get_cosmos3_post_process_func as get_cosmos3_post_process_func,
)
from .pipeline_cosmos3 import (  # noqa: F401  (re-export)
    get_cosmos3_pre_process_func as get_cosmos3_pre_process_func,
)

logger = init_logger(__name__)

#: Per-layer ``(K, V)`` for one branch. Serializers may turn the pairs into lists.
KVBranch = list[tuple[torch.Tensor, torch.Tensor]]


def fingerprint_text_ids(text_ids: torch.Tensor) -> str:
    """Content hash of a token-id tensor, used as the CFG branch key."""
    flat = text_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1).numpy()
    return hashlib.sha256(flat.tobytes()).hexdigest()[:32]


def _tp_world_size() -> int:
    return get_tensor_model_parallel_world_size() if model_parallel_is_initialized() else 1


def _tp_rank() -> int:
    return get_tensor_model_parallel_rank() if model_parallel_is_initialized() else 0


def _gather_kv_heads(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather ``[B, S, H_kv_local, D]`` into the full head set.

    Only rank 0's stage output leaves the reasoner, so it must carry every head.
    ``all_gather`` concatenates in rank order, which matches the head order
    ``ColumnParallelLinear`` sharded them in.
    """
    if _tp_world_size() == 1:
        return tensor
    return tensor_model_parallel_all_gather(tensor.contiguous(), dim=-2)


@dataclass(frozen=True)
class Cosmos3TextConditioning:
    """The reasoner -> generator conditioning contract.

    ``branches`` maps a token-id fingerprint to that branch's per-layer
    ``(K, V)``, each ``[B, S_und, num_kv_heads, head_dim]`` with the full,
    gathered head set. Construction validates the declared layout against the
    tensors, so a corrupted payload fails here rather than inside attention.
    """

    branches: dict[str, KVBranch]
    num_layers: int
    num_kv_heads: int
    head_dim: int
    height: int
    width: int
    max_sequence_length: int
    use_system_prompt: bool
    reasoner_tp_size: int
    payload_mib: float
    schema: str = COSMOS3_UND_SCHEMA

    _WIRE_FIELDS: ClassVar[tuple[str, ...]] = (
        "num_layers",
        "num_kv_heads",
        "head_dim",
        "height",
        "width",
        "max_sequence_length",
        "use_system_prompt",
        "reasoner_tp_size",
        "payload_mib",
    )

    def __post_init__(self) -> None:
        _check_schema(self.schema)
        if not self.branches:
            raise ValueError("Cosmos3 text conditioning carries no prompt branches.")
        for field in ("num_layers", "num_kv_heads", "head_dim"):
            if getattr(self, field) <= 0:
                raise ValueError(f"Cosmos3 text conditioning {field}={getattr(self, field)} must be positive.")
        # Batch size and dtype must be uniform across the payload; the token
        # count only within a branch, since each branch is trimmed to its own
        # prompt length.
        batch_size: int | None = None
        dtype: torch.dtype | None = None
        for key, entry in self.branches.items():
            if len(entry) != self.num_layers:
                raise ValueError(
                    f"Cosmos3 text conditioning branch {key} has {len(entry)} layer(s) but "
                    f"declares num_layers={self.num_layers}."
                )
            seq_len: int | None = None
            for layer_idx, (k, v) in enumerate(entry):
                for label, tensor in (("K", k), ("V", v)):
                    where = f"Cosmos3 text conditioning {label} for branch {key} layer {layer_idx}"
                    if not isinstance(tensor, torch.Tensor):
                        raise ValueError(f"{where} is {type(tensor).__name__}, not a tensor.")
                    if tensor.ndim != 4 or tuple(tensor.shape[-2:]) != (self.num_kv_heads, self.head_dim):
                        raise ValueError(
                            f"{where} is shaped {tuple(tensor.shape)}, but the contract declares "
                            f"[B, S_und, {self.num_kv_heads}, {self.head_dim}]."
                        )
                    batch_size = batch_size if batch_size is not None else int(tensor.shape[0])
                    seq_len = seq_len if seq_len is not None else int(tensor.shape[1])
                    dtype = dtype if dtype is not None else tensor.dtype
                    if (int(tensor.shape[0]), int(tensor.shape[1]), tensor.dtype) != (batch_size, seq_len, dtype):
                        raise ValueError(
                            f"{where} has batch={int(tensor.shape[0])}, tokens={int(tensor.shape[1])}, "
                            f"dtype={tensor.dtype}; expected batch={batch_size}, tokens={seq_len}, dtype={dtype}."
                        )

    @property
    def num_branches(self) -> int:
        return len(self.branches)

    @property
    def conditioning_length(self) -> int:
        """Longest trimmed UND sequence length across branches, for logging."""
        return max(int(k.shape[1]) for entry in self.branches.values() for k, _v in entry)

    def to_payload(self) -> dict[str, Any]:
        """Flatten to the plain two-key dict the stage connectors serialize."""
        meta: dict[str, Any] = {field: getattr(self, field) for field in self._WIRE_FIELDS}
        meta["schema"] = self.schema
        meta["num_branches"] = self.num_branches
        # Copy the mapping so callers cannot add branches after validation.
        return {KV_KEY: dict(self.branches), META_KEY: meta}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Cosmos3TextConditioning:
        """Rebuild and validate the contract from what came off the wire."""
        branches = payload.get(KV_KEY)
        if not isinstance(branches, dict) or not branches:
            raise ValueError(f"Cosmos3 text conditioning payload has no {KV_KEY!r} branches.")
        meta = payload.get(META_KEY)
        if not isinstance(meta, dict):
            raise ValueError(f"Cosmos3 text conditioning payload carries no {META_KEY!r} metadata.")
        _check_schema(meta.get("schema"))
        missing = [field for field in cls._WIRE_FIELDS if field not in meta]
        if missing:
            raise ValueError(f"Cosmos3 text conditioning metadata is missing {', '.join(missing)}.")
        return cls(branches=branches, **{field: meta[field] for field in cls._WIRE_FIELDS})


def _check_schema(schema: Any) -> None:
    if schema != COSMOS3_UND_SCHEMA:
        raise ValueError(
            f"Cosmos3 text conditioning declares schema {schema!r}, but this stage implements "
            f"{COSMOS3_UND_SCHEMA!r}. Both stages must run the same vLLM-Omni version."
        )


class _ReplayLanguageModel(torch.nn.Module):
    """Generator-stage stand-in for the UND tower that replays reasoner K/V.

    Matches ``Cosmos3LanguageModel.forward(text_ids, freqs)``. It keeps the real
    ``rotary_emb``, which ``_compute_rope_freqs`` uses for the GEN frequencies,
    and an empty ``layers``, which the offloaders inspect.
    """

    _layerwise_offload_blocks_attrs: ClassVar[list[str]] = ["layers"]

    def __init__(
        self,
        num_hidden_layers: int,
        rotary_emb: torch.nn.Module,
        *,
        num_kv_heads: int,
        num_kv_heads_local: int,
        kv_head_offset: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.rotary_emb = rotary_emb
        self.layers = torch.nn.ModuleList()
        self.num_kv_heads = num_kv_heads
        self.num_kv_heads_local = num_kv_heads_local
        self.kv_head_offset = kv_head_offset
        self.head_dim = head_dim
        self._check_head_range()
        self._table: dict[str, KVBranch] = {}
        self._dtype: torch.dtype | None = None
        self._reasoner_settings: str | None = None

    def _check_head_range(self) -> None:
        """Reject a head range that is not a slice of the gathered head set.

        Slicing clamps instead of raising, so a bad range would silently replay
        too few heads.
        """
        stop = self.kv_head_offset + self.num_kv_heads_local
        if self.num_kv_heads_local > 0 and self.kv_head_offset >= 0 and stop <= self.num_kv_heads:
            return
        raise ValueError(
            f"Cosmos3 generator stage would replay KV heads [{self.kv_head_offset}, {stop}) of "
            f"{self.num_kv_heads}, which is not a valid shard: tensor_parallel_size={_tp_world_size()} "
            f"(rank {_tp_rank()}) must divide the {self.num_kv_heads} KV heads."
        )

    def install(self, conditioning: Cosmos3TextConditioning, dtype: torch.dtype | None = None) -> None:
        """Hold this rank's KV-head shard of the payload for one request."""
        self._check_layout(conditioning)
        start, stop = self.kv_head_offset, self.kv_head_offset + self.num_kv_heads_local
        self._table = {
            key: [(k[..., start:stop, :].contiguous(), v[..., start:stop, :].contiguous()) for k, v in entry]
            for key, entry in conditioning.branches.items()
        }
        self._dtype = dtype
        # Reported on a replay miss so the operator can compare settings.
        self._reasoner_settings = (
            f"height={conditioning.height}, width={conditioning.width}, "
            f"max_sequence_length={conditioning.max_sequence_length}, "
            f"use_system_prompt={conditioning.use_system_prompt}"
        )

    def _check_layout(self, conditioning: Cosmos3TextConditioning) -> None:
        expected = {
            "num_layers": self.num_hidden_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
        }
        mismatched = [
            f"{field}={getattr(conditioning, field)} from reasoner, {want} here"
            for field, want in sorted(expected.items())
            if getattr(conditioning, field) != want
        ]
        if mismatched:
            raise RuntimeError(
                "Cosmos3 reasoner and generator stages disagree on the UND K/V layout: "
                f"{', '.join(mismatched)}. Both stages must load the same model."
            )

    def clear(self) -> None:
        self._table = {}
        self._dtype = None
        self._reasoner_settings = None

    def forward(
        self,
        text_ids: torch.Tensor,
        freqs: tuple[torch.Tensor, torch.Tensor],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        del freqs  # UND RoPE was already applied on the reasoner stage.
        key = fingerprint_text_ids(text_ids)
        entry = self._table.get(key)
        if entry is None:
            raise RuntimeError(
                f"Cosmos3 generator stage has no reasoner K/V for this prompt branch (fingerprint={key}, "
                f"known={sorted(self._table)}). Both stages must tokenize identically; "
                f"the reasoner resolved {self._reasoner_settings}."
            )
        device = text_ids.device
        return [
            (
                k.to(device=device, dtype=self._dtype, non_blocking=True),
                v.to(device=device, dtype=self._dtype, non_blocking=True),
            )
            for k, v in entry
        ]


class Cosmos3ReasonerPipeline(Cosmos3OmniDiffusersPipeline):
    """Stage 0: runs only the UND tower and emits per-layer text K/V."""

    cosmos3_owned_towers: ClassVar[tuple[str, ...]] = ("reasoner",)
    # Nothing here decodes pixels: read only the VAE config, for its scale factors.
    cosmos3_loads_vae: ClassVar[bool] = False
    _vae_modules: ClassVar[list[str]] = []
    # The engine's warmup request has no modalities (read as video), which the
    # T2I guard in ``forward`` would reject.
    dummy_run_num_frames: ClassVar[int] = 0

    def forward(self, req: Any) -> Any:  # type: ignore[override]
        from vllm_omni.diffusion.data import DiffusionOutput

        if not self._is_t2i_request(req):
            raise ValueError(
                "Cosmos3 disagg currently splits the towers for text-to-image only. "
                "Request prompt['modalities'] must be ['image']."
            )

        prompt_data = req.prompts[0] if req.prompts else ""
        if isinstance(prompt_data, str):
            prompt, negative_prompt = prompt_data, ""
        else:
            prompt = prompt_data.get("prompt", "")
            negative_prompt = prompt_data.get("negative_prompt") or ""

        conditioning = self.encode_text_conditioning(prompt, negative_prompt, req.sampling_params)
        return DiffusionOutput(output=conditioning.to_payload())

    def encode_text_conditioning(self, prompt: str, negative_prompt: str, sp: Any) -> Cosmos3TextConditioning:
        """Run the UND tower and build the reasoner -> generator contract.

        Resolves geometry and tokenizer inputs through the same helpers as the
        stock T2I ``forward``, so the generator's fingerprints match.
        """
        height, width = self._resolve_t2i_geometry(sp)
        max_sequence_length, use_system_prompt, frame_rate = self._resolve_text_encode_params(
            sp,
            default_use_system_prompt=False,
        )
        guidance_scale = self._resolve_guidance_scale(sp, COSMOS3_T2I_DEFAULT_GUIDANCE_SCALE)

        cond_ids, cond_mask, uncond_ids, uncond_mask = self._format_and_tokenize_prompts(
            prompt,
            negative_prompt,
            1,  # T2I is a single frame.
            frame_rate,
            height,
            width,
            max_sequence_length,
            sp,
            use_system_prompt,
            is_t2i=True,
        )

        transformer = self.transformer
        t = 1
        h = height // self.vae_scale_factor_spatial
        w = width // self.vae_scale_factor_spatial
        hp, wp, _, _ = transformer._pad_to_patch_size(h, w)
        dtype = transformer.proj_in.weight.dtype

        # The unconditional branch is only replayed under CFG.
        do_cfg = guidance_scale > 1.0
        branches_to_encode = [(cond_ids, cond_mask)]
        if do_cfg and uncond_ids is not None:
            branches_to_encode.append((uncond_ids, uncond_mask))

        # Calling language_model directly skips FSDP2's root pre-forward hook,
        # leaving root-managed params (embed_tokens, norm) sharded under HSDP.
        fsdp_root: Any = transformer if hasattr(transformer, "unshard") and hasattr(transformer, "reshard") else None
        if fsdp_root is not None:
            fsdp_root.unshard()
        try:
            branches: dict[str, KVBranch] = {}
            for text_ids, text_mask in branches_to_encode:
                text_ids = text_ids.to(self.device)
                text_mask = text_mask.to(self.device)
                max_real_len = int(text_mask.sum(dim=1).max().item())

                freqs_und, _freqs_gen = transformer._compute_rope_freqs(text_mask, t, hp, wp, None, self.device, dtype)
                with transformer._offload_context("reasoner"):
                    cached_kv_full = transformer.language_model(text_ids, freqs_und)

                # Trim padding as the co-located forward does, then gather heads.
                branches[fingerprint_text_ids(text_ids)] = [
                    (
                        _gather_kv_heads(k[:, :max_real_len]).contiguous().cpu(),
                        _gather_kv_heads(v[:, :max_real_len]).contiguous().cpu(),
                    )
                    for k, v in cached_kv_full
                ]
        finally:
            if fsdp_root is not None:
                fsdp_root.reshard()

        payload_mib = (
            sum(
                k.numel() * k.element_size() + v.numel() * v.element_size()
                for entry in branches.values()
                for k, v in entry
            )
            / 2**20
        )
        sample_k = next(iter(branches.values()))[0][0]
        conditioning = Cosmos3TextConditioning(
            branches=branches,
            num_layers=len(next(iter(branches.values()))),
            num_kv_heads=int(sample_k.shape[-2]),
            head_dim=int(sample_k.shape[-1]),
            height=height,
            width=width,
            max_sequence_length=max_sequence_length,
            use_system_prompt=use_system_prompt,
            reasoner_tp_size=_tp_world_size(),
            payload_mib=round(payload_mib, 1),
        )
        logger.info(
            "Cosmos3 reasoner: %d branch(es) (cfg=%s), K/V payload=%.1f MiB, target=%dx%d",
            conditioning.num_branches,
            do_cfg,
            payload_mib,
            height,
            width,
        )
        if payload_mib > COSMOS3_UND_PAYLOAD_WARN_MIB:
            logger.warning(
                "Cosmos3 reasoner: K/V payload is %.1f MiB (> %.1f MiB) for a %d-token "
                "conditioning length; consider lowering max_sequence_length (currently %d).",
                payload_mib,
                COSMOS3_UND_PAYLOAD_WARN_MIB,
                conditioning.conditioning_length,
                max_sequence_length,
            )
        return conditioning


class Cosmos3GeneratorPipeline(Cosmos3OmniDiffusersPipeline):
    """Stage 1: runs only the GEN tower, replaying the reasoner's K/V."""

    cosmos3_owned_towers: ClassVar[tuple[str, ...]] = ("generator",)
    # The engine's warmup request carries no reasoner K/V to replay.
    dummy_run_num_frames: ClassVar[int] = 0

    def __init__(self, *, od_config: Any, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        self._bind_owned_tower()

    def _bind_owned_tower(self) -> None:
        """Replace the rope-only UND tower with the replay stub.

        The expected head layout comes from the cross-attention that consumes it.
        """
        transformer = self.transformer
        consumer = transformer.gen_layers[0].cross_attention
        transformer.language_model = _ReplayLanguageModel(  # type: ignore[assignment]
            transformer.num_hidden_layers,
            transformer.language_model.rotary_emb,
            num_kv_heads=consumer.num_kv_heads,
            num_kv_heads_local=consumer.num_kv_heads_local,
            kv_head_offset=_tp_rank() * consumer.num_kv_heads_local,
            head_dim=consumer.head_dim,
        )

    def forward(self, req: Any) -> Any:  # type: ignore[override]
        """Install the reasoner K/V for this request, then run the stock path."""
        self.install_text_conditioning(self._extract_und_payload(req))
        try:
            return super().forward(req)
        finally:
            self.transformer.language_model.clear()

    @staticmethod
    def _extract_und_payload(req: Any) -> dict[str, Any]:
        """Return the reasoner payload that ``reasoner2generator`` put in ``prompt["extra"]``."""
        prompt_data = req.prompts[0] if req.prompts else ""
        if isinstance(prompt_data, dict):
            extra = prompt_data.get("extra") or {}
            if KV_KEY in extra:
                return extra
        raise ValueError(
            f"Cosmos3 generator stage received a request without reasoner K/V in prompt['extra'][{KV_KEY!r}]; "
            "route requests through the reasoner stage."
        )

    def install_text_conditioning(self, payload: dict[str, Any]) -> Cosmos3TextConditioning:
        conditioning = Cosmos3TextConditioning.from_payload(payload)
        stub = self.transformer.language_model
        stub.install(conditioning, dtype=self.transformer.proj_in.weight.dtype)
        logger.info(
            "Cosmos3 generator: installed reasoner K/V for %d branch(es) (%.1f MiB, heads [%d, %d) of %d)",
            conditioning.num_branches,
            conditioning.payload_mib,
            stub.kv_head_offset,
            stub.kv_head_offset + stub.num_kv_heads_local,
            conditioning.num_kv_heads,
        )
        return conditioning


def get_cosmos3_reasoner_post_process_func(od_config: Any):
    """Postprocessor for the reasoner: pass the K/V through under ``trajectory``.

    ``trajectory`` is the payload key the output formatter copies into
    ``multimodal_output`` without metadata validation. Its reserved sub-keys
    (``latents``, ``timesteps``, ``log_probs``, ``decoded``) are not used.
    """
    del od_config

    def post_process_func(
        output: Any,
        output_type: str = "np",
        sampling_params: Any = None,
    ) -> Any:
        del sampling_params
        if output_type == "latent":
            return output
        if not isinstance(output, dict) or KV_KEY not in output:
            raise ValueError(
                f"Cosmos3 reasoner postprocess expected a dict payload containing {KV_KEY!r}, "
                f"got {type(output).__name__}."
            )
        meta = output.get(META_KEY) or {}
        return {
            "payload": {"trajectory": {KV_KEY: output[KV_KEY], META_KEY: meta}},
            # Unknown metadata groups are tolerated; never use the reserved "internal".
            "metadata": {"cosmos3_und": dict(meta)},
        }

    return post_process_func
