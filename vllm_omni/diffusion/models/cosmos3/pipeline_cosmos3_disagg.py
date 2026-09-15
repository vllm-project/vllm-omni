# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tower-split Cosmos3 pipelines: reasoner (UND) and generator (GEN) stages.

Both classes subclass ``Cosmos3OmniDiffusersPipeline`` so prompt formatting,
tokenization, mRoPE construction, scheduler setup, VAE decode and the checkpoint
key remap are inherited verbatim. Only the tower boundary is overridden -- no
Cosmos3 math is reimplemented here. The topology that wires the two stages
together lives in ``vllm_omni/diffusion/models/cosmos3_pipeline_config.py``.

WHERE THE SPLIT IS MADE
-----------------------
``Cosmos3VFMTransformer.forward`` calls the UND tower exactly once per branch::

    if self.cached_kv is None:
        freqs_und, freqs_gen = self._compute_rope_freqs(...)
        self.cached_freqs_gen = freqs_gen
        if need_kv:
            with self._offload_context("reasoner"):
                cached_kv_full = self.language_model(text_ids, freqs_und)   # <-- seam
            self.cached_kv = [(k[:, :max_real_len], v[:, :max_real_len]) ...]

The generator stage swaps ``language_model`` for a stub that *replays* the K/V
computed on the reasoner stage instead of running 31.2 B parameters of UND
weights. Intercepting at this call -- rather than pre-setting ``cached_kv`` --
matters because the T2I denoise loop calls ``transformer.reset_cache()`` and
then drives CFG through *local* ``cond_cache`` / ``uncond_cache`` variables
seeded to ``(None, None)``; anything written to ``cached_kv`` beforehand is
discarded. Replacing the tower is the one interception point every path
(CFG-parallel, sequential CFG, and no-CFG) funnels through.

It also keeps ``_compute_rope_freqs`` running locally on the generator stage, so
the GEN mRoPE frequencies are derived from the true latent geometry that stage
actually allocated, rather than being shipped from a stage that would have to
predict it. Only K/V crosses the wire.

THE CONDITIONING CONTRACT
-------------------------
``Cosmos3TextConditioning`` is what crosses the edge, and it is deliberately
model-specific: Cosmos3 UND per-layer text K/V in a Cosmos3 layout, tagged with
the code-owned ``cosmos3.text_conditioning/v1`` schema identifier. It makes the
four things that can silently go wrong explicit --

* **branch identity** -- keyed by a fingerprint of ``text_ids``, so the
  conditional and unconditional CFG branches cannot be confused for one another
  even if the denoise loop reorders them;
* **tensor layout** -- ``num_layers`` / ``num_kv_heads`` / ``head_dim``, checked
  against the tensors themselves *and* against the consuming stage's config;
* **sharding** -- the payload always carries the *full, unsharded* KV-head set
  (see below), so it does not depend on either stage's TP size;
* **lifetime** -- one payload belongs to exactly one request; the generator
  installs it for the duration of ``forward`` and clears it afterwards.

TP SHARDING IS RESOLVED ON THE EDGE, NOT ASSUMED AWAY
-----------------------------------------------------
The UND tower's K/V is TP-sharded at birth: ``Cosmos3CausalAttention`` builds K
and V through ``ColumnParallelLinear(..., gather_output=False)``, so rank *r*
holds KV heads ``[r * H_local, (r + 1) * H_local)`` and no rank holds the rest.
Only rank 0's output leaves a stage (``MultiprocExecutor.execute_request`` passes
``unique_reply_rank=0``), so shipping TP-local shards would hand *every*
generator rank rank 0's heads -- and with matching TP sizes the shapes agree, so
generator rank 1 would silently condition on the wrong heads.

The reasoner therefore all-gathers the KV-head dimension before the payload
leaves the tower (``_gather_kv_heads``), and each generator rank slices the range
its own cross-attention expects (``_ReplayLanguageModel.install``). The wire
format is TP-independent: the two stages may run different
``tensor_parallel_size`` values, and the gathered payload is the same size the
co-located tower would produce.

CFG BRANCHES
------------
Guidance runs the UND tower twice -- once for the conditional prompt and once
for the unconditional/negative one -- with different token streams and hence
different K/V. Both stages tokenize with the same inherited code path and the
same geometry, so the fingerprints agree by construction.

WHY EACH STAGE BUILDS ONLY ITS OWN TOWER
----------------------------------------
``DiffusionModelRunner.load_model`` selects the device and
``DiffusersPipelineLoader._init_from_load_format`` constructs the pipeline inside
that device context, so a tower allocated in ``__init__`` is allocated *on the
card*. Building both towers and pruning one afterwards would therefore still pay
the full ~120 GiB peak and fail to start on an 80 GB card -- exactly the
two-smaller-cards case the split exists for. Instead each stage declares the
tower it owns (``cosmos3_owned_towers``), ``Cosmos3VFMTransformer`` constructs
only that one, and the unowned tower's checkpoint tensors match no live parameter
and are filtered out by the inherited ``load_weights``. ``_require_unowned_absent``
then verifies the ownership request was honoured, so a transformer that ignores it
fails loudly instead of quietly doubling startup memory.

WHAT THE SPLIT DOES *NOT* SAVE: CHECKPOINT READS
-----------------------------------------------
Both stages still stream the whole checkpoint. ``Cosmos3.load_weights`` filters
by name *after* ``safetensors_weights_iterator`` has already materialized each
tensor, so the unowned tower's tensors are read from disk and discarded rather
than skipped (the "kept N/M tensors" line it logs is the filter, not the read).
Splitting the towers therefore roughly doubles aggregate startup read I/O across
the two stages instead of halving it. Fixing that means teaching the loader to
skip tensors by name before materializing them, which is a loader change and not
in scope here.
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

# Re-exported so ``_load_process_func`` resolves them from this module, which is
# what ``_DIFFUSION_MODELS`` maps both tower arch names to. The generator reuses
# Cosmos3's stock funcs verbatim: its output *is* an image, so the stock
# postprocessor is exactly right, and the preprocessor is a near no-op for T2I.
# The IR-op override (native rms_norm) is a property of the Cosmos3 kernels, not
# of either tower, so both stages want it.
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

#: One branch of conditioning: per-layer ``(K, V)``. Stage serializers turn the
#: pairs into 2-lists on the way across the edge, so nothing may require tuples.
KVBranch = list[tuple[torch.Tensor, torch.Tensor]]


def fingerprint_text_ids(text_ids: torch.Tensor) -> str:
    """Stable content hash of a token-id tensor, used as the branch key.

    Hashing the ids (rather than trusting call order) keeps the conditional and
    unconditional CFG branches from being confused for one another even if the
    denoise loop changes the order in which it evaluates them.
    """
    flat = text_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1).numpy()
    return hashlib.sha256(flat.tobytes()).hexdigest()[:32]


def _tp_world_size() -> int:
    """This process's TP world size, or 1 when no model-parallel group exists.

    A process with no TP group is at TP 1 by definition, which is also what makes
    this callable from a single-process test.
    """
    return get_tensor_model_parallel_world_size() if model_parallel_is_initialized() else 1


def _tp_rank() -> int:
    """This process's rank inside its TP group, or 0 when there is no group.

    This is the rank whose KV-head range the generator stage must slice out of the
    gathered payload: ``ColumnParallelLinear`` hands rank *r* the contiguous output
    range ``[r * H_local, (r + 1) * H_local)``.
    """
    return get_tensor_model_parallel_rank() if model_parallel_is_initialized() else 0


def _gather_kv_heads(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather a ``[B, S, H_kv_local, D]`` tensor into the full head set.

    UND K/V is born TP-sharded and only rank 0's stage output is kept, so the
    payload has to carry every head or the generator's other ranks receive
    conditioning that is not theirs. ``all_gather`` concatenates in rank order,
    which is exactly the order ``ColumnParallelLinear`` sharded the heads in, so
    the result is the head layout a TP-1 tower would have produced.

    Called on activations (the tower's params are unsharded around the call), so
    there is no DTensor to redistribute here. At TP 1 this is a no-op and no
    collective is entered.
    """
    if _tp_world_size() == 1:
        return tensor
    return tensor_model_parallel_all_gather(tensor.contiguous(), dim=-2)


def _require_unowned_absent(module_list: torch.nn.ModuleList, label: str) -> None:
    """Verify the tower this stage does not own was never constructed.

    ``Cosmos3VFMTransformer`` skips it when ``owned_towers`` says so, and that is
    the whole memory saving -- so a non-empty container here means the ownership
    request did not reach the transformer and this stage is about to hold two
    towers' worth of weights. Fail on it rather than pruning after the fact: by
    this point the allocation has already happened on the device.
    """
    if len(module_list):
        raise RuntimeError(
            f"Cosmos3 tower-split stage built {len(module_list)} {label} block(s) it does not own. "
            "The transformer ignored owned_towers, so this stage would allocate both towers; "
            "check that the pipeline's cosmos3_owned_towers reaches Cosmos3VFMTransformer.__init__."
        )


@dataclass(frozen=True)
class Cosmos3TextConditioning:
    """The reasoner -> generator conditioning contract.

    Model-specific by design (see the module docstring): this describes Cosmos3
    UND per-layer text K/V, tagged with the code-owned
    ``cosmos3.text_conditioning/v1`` schema identifier.

    ``branches`` maps a token-id fingerprint to that branch's per-layer
    ``(K, V)``, each shaped ``[B, S_und, num_kv_heads, head_dim]`` with the
    **full, unsharded** head set -- the reasoner gathers its TP shards before
    building this, so the contract does not depend on either stage's TP size.
    ``reasoner_tp_size`` is retained for diagnostics only.

    Constructing one validates it: every declared field is cross-checked against
    the tensors, so a payload cannot describe a layout it does not carry. The
    properties the wire format does *not* declare are checked for internal
    consistency instead -- batch size and dtype across the whole payload, UND
    token count within each branch -- so a partially corrupted payload is rejected
    here rather than deep inside cross-attention.
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

    #: Metadata fields the wire format must carry. ``schema`` is checked first and
    #: separately, so a mismatch is reported as a schema error rather than as a
    #: missing field.
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
        if self.schema != COSMOS3_UND_SCHEMA:
            raise ValueError(
                f"Cosmos3 text conditioning declares schema {self.schema!r}, but this "
                f"stage implements {COSMOS3_UND_SCHEMA!r}. The two stages must run the "
                "same vLLM-Omni version."
            )
        if not self.branches:
            raise ValueError("Cosmos3 text conditioning carries no prompt branches.")
        # A zero here would pass every check below vacuously -- ``num_layers=0``
        # matches an empty branch list -- and then fail as a bare ``max()`` error
        # from ``conditioning_length``, or replay nothing at all.
        for field in ("num_layers", "num_kv_heads", "head_dim"):
            if getattr(self, field) <= 0:
                raise ValueError(
                    f"Cosmos3 text conditioning declares {field}={getattr(self, field)}; the layout "
                    "fields describe real tensors, so all three must be positive."
                )
        # Every branch replays into the same GEN forward, so one batch size has to
        # hold across all of them. Taken from the first tensor seen rather than
        # declared, because the reasoner does not put it on the wire.
        batch_size: int | None = None
        # One UND forward per branch produces every layer, so one dtype holds
        # across the whole payload. Branch *lengths* differ, though: each branch is
        # trimmed to its own real prompt length, so S_und is compared per branch.
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
                    if not isinstance(tensor, torch.Tensor):
                        # Checked before anything reads ``.ndim``, so a corrupted or
                        # hand-built payload is a contract error naming the member
                        # rather than an AttributeError from inside validation.
                        raise ValueError(
                            f"Cosmos3 text conditioning {label} for branch {key} layer {layer_idx} is "
                            f"{type(tensor).__name__}, not a tensor. The wire format carries K/V as "
                            "tensors; only the (K, V) pairing itself may decode as a list."
                        )
                    if tensor.ndim != 4:
                        raise ValueError(
                            f"Cosmos3 text conditioning {label} for branch {key} layer {layer_idx} "
                            f"has {tensor.ndim} dim(s), expected 4 ([B, S_und, num_kv_heads, head_dim])."
                        )
                    if tuple(tensor.shape[-2:]) != (self.num_kv_heads, self.head_dim):
                        raise ValueError(
                            f"Cosmos3 text conditioning {label} for branch {key} layer {layer_idx} is "
                            f"shaped {tuple(tensor.shape)}, but the contract declares "
                            f"[B, S_und, {self.num_kv_heads}, {self.head_dim}]."
                        )
                    if batch_size is None:
                        batch_size = int(tensor.shape[0])
                    elif int(tensor.shape[0]) != batch_size:
                        raise ValueError(
                            f"Cosmos3 text conditioning {label} for branch {key} layer {layer_idx} has "
                            f"batch size {int(tensor.shape[0])}, but the rest of the payload carries "
                            f"{batch_size}. All branches replay into the same GEN forward, so the batch "
                            "size has to be uniform."
                        )
                    if seq_len is None:
                        seq_len = int(tensor.shape[1])
                    elif int(tensor.shape[1]) != seq_len:
                        raise ValueError(
                            f"Cosmos3 text conditioning {label} for branch {key} layer {layer_idx} covers "
                            f"{int(tensor.shape[1])} UND token(s), but the rest of the branch covers "
                            f"{seq_len}. Every layer of a branch comes from one UND forward over one "
                            "prompt, so the token count cannot vary within it."
                        )
                    if dtype is None:
                        dtype = tensor.dtype
                    elif tensor.dtype != dtype:
                        raise ValueError(
                            f"Cosmos3 text conditioning {label} for branch {key} layer {layer_idx} is "
                            f"{tensor.dtype}, but the rest of the payload is {dtype}. The reasoner emits "
                            "one dtype for the whole payload; a mixed one means the tensors did not "
                            "survive the stage edge intact."
                        )
                # No separate ``k.shape != v.shape`` check: the loop above pins every
                # dim of *both* members -- ndim, the trailing (num_kv_heads, head_dim),
                # the payload-wide batch size and the branch-wide token count -- so a
                # K/V pair that disagreed on any dim has already been rejected, in
                # terms of the dim that disagrees.

    @property
    def num_branches(self) -> int:
        return len(self.branches)

    @property
    def conditioning_length(self) -> int:
        """Longest trimmed UND sequence length across branches, for logging."""
        return max(int(k.shape[1]) for entry in self.branches.values() for k, _v in entry)

    def to_payload(self) -> dict[str, Any]:
        """Flatten to the two-key dict shape that crosses the stage edge.

        Kept a plain dict of tensors on purpose: the connector serde handles
        nested dicts/lists/tensors, and the typed object is the producer- and
        consumer-side contract rather than the wire encoding.
        """
        meta: dict[str, Any] = {field: getattr(self, field) for field in self._WIRE_FIELDS}
        meta["schema"] = self.schema
        meta["num_branches"] = self.num_branches
        # A shallow copy of the branch mapping, not the mapping itself:
        # ``frozen=True`` does not freeze what the fields point at, so handing out
        # the live dict would let a caller add a branch after validation ran. The
        # tensors are shared, so this costs nothing that matters.
        return {KV_KEY: dict(self.branches), META_KEY: meta}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Cosmos3TextConditioning:
        """Rebuild and validate the contract from what came off the wire."""
        branches = payload.get(KV_KEY)
        if not isinstance(branches, dict) or not branches:
            raise ValueError(
                f"Cosmos3 text conditioning payload has no {KV_KEY!r} branches "
                f"(keys={sorted(payload) if isinstance(payload, dict) else '<n/a>'})."
            )
        meta = payload.get(META_KEY)
        if not isinstance(meta, dict):
            raise ValueError(
                f"Cosmos3 text conditioning payload carries no {META_KEY!r} metadata, so its layout "
                "and schema are undeclared. Both stages must run the same vLLM-Omni version."
            )
        schema = meta.get("schema")
        if schema != COSMOS3_UND_SCHEMA:
            raise ValueError(
                f"Cosmos3 text conditioning declares schema {schema!r}, but this stage implements "
                f"{COSMOS3_UND_SCHEMA!r}. The two stages must run the same vLLM-Omni version."
            )
        missing = [field for field in cls._WIRE_FIELDS if field not in meta]
        if missing:
            raise ValueError(
                f"Cosmos3 text conditioning metadata is missing {', '.join(missing)}; "
                f"schema {COSMOS3_UND_SCHEMA!r} requires every layout field to be declared."
            )
        return cls(branches=branches, **{field: meta[field] for field in cls._WIRE_FIELDS})


class _ReplayLanguageModel(torch.nn.Module):
    """Stands in for the UND tower on the generator stage.

    Returns per-layer ``(K, V)`` produced by the reasoner stage. The signature
    matches ``Cosmos3LanguageModel.forward(text_ids, freqs)`` so the unmodified
    transformer forward path calls it transparently.

    THE STUB'S ATTRIBUTE SURFACE IS NOT OPTIONAL
    --------------------------------------------
    The rest of ``Cosmos3VFMTransformer`` reaches into ``language_model`` in two
    places, and both run on the generator stage:

    * ``_compute_rope_freqs`` ends with ``rotary_emb = self.language_model
      .rotary_emb`` and uses it to build *both* the UND and the GEN frequencies.
      The GEN ones drive every denoising step, so the real
      ``Qwen3VLTextRotaryEmbedding`` must be carried over. It holds no
      parameters -- only a non-persistent ``inv_freq`` buffer -- so keeping it
      costs nothing and, being non-persistent, it never appears in
      ``state_dict()`` and so is never expected by the weight loader.
    * The offload rings introspect ``layers`` -- both the layerwise ring, via
      ``_layerwise_offload_blocks_attrs`` below, and ``ModuleDiscovery`` -- so it
      must exist. It is an empty ``ModuleList`` here: there is nothing to swap in
      or out, which is also why ``_model_cpu_offload_components`` on a
      generator-only transformer advertises no ``"reasoner"`` component at all
      (see ``Cosmos3VFMTransformer._offload_context``).

    The stub also has to be a real ``nn.Module`` because
    ``transformer.language_model`` is named in ``_dit_modules`` and
    ``ModuleDiscovery`` warns-and-skips anything that is not one.
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
        # The full head set the payload carries, and the slice of it this rank's
        # cross-attention consumes -- see ``install``.
        self.num_kv_heads = num_kv_heads
        self.num_kv_heads_local = num_kv_heads_local
        self.kv_head_offset = kv_head_offset
        self.head_dim = head_dim
        self._check_head_range()
        self._table: dict[str, KVBranch] = {}
        self._dtype: torch.dtype | None = None
        self._reasoner_settings: str | None = None

    def _check_head_range(self) -> None:
        """Reject a head range that cannot be a slice of the gathered head set.

        ``install`` slices ``[offset, offset + local)`` out of the payload's full
        head set. Python slicing clamps instead of raising, so an out-of-range
        range would silently replay *fewer* heads than the consumer expects, or
        none at all -- ``num_kv_heads_local == 0`` (a generator TP size larger than
        the KV-head count) yields an empty tensor that only fails much later, deep
        in cross-attention. Fail at bind time, in terms of the two stages' TP
        sizes, rather than per request in terms of a shape.
        """
        stop = self.kv_head_offset + self.num_kv_heads_local
        if self.num_kv_heads_local > 0 and self.kv_head_offset >= 0 and stop <= self.num_kv_heads:
            return
        raise ValueError(
            f"Cosmos3 generator stage would replay KV heads [{self.kv_head_offset}, {stop}) of the "
            f"{self.num_kv_heads} the reasoner ships, which is not a valid shard: this stage runs at "
            f"tensor_parallel_size={_tp_world_size()} (rank {_tp_rank()}) with "
            f"{self.num_kv_heads_local} local KV head(s). The K/V handoff is TP-independent, but the "
            f"stage's own tensor_parallel_size must still divide its {self.num_kv_heads} KV heads."
        )

    def install(self, conditioning: Cosmos3TextConditioning, dtype: torch.dtype | None = None) -> None:
        """Take this rank's KV-head shard of a validated payload, then hold it.

        WHY THE SHARD IS TAKEN HERE
        ---------------------------
        The payload carries every KV head (the reasoner gathers its TP shards),
        while ``Cosmos3CrossAttention`` on this rank consumes exactly
        ``[B, S_und, num_kv_heads // tp_size, head_dim]`` -- the contiguous head
        range ``ColumnParallelLinear`` assigned to *this* TP rank. Slicing it out
        once per request, on the host tensors before they reach the device, is
        both cheaper than doing it per denoising step and smaller to copy.

        The layout is checked first so a stage-configuration mistake is reported
        in terms of the two stages' settings rather than as a shape error from
        inside attention.
        """
        self._check_layout(conditioning)
        start, stop = self.kv_head_offset, self.kv_head_offset + self.num_kv_heads_local
        # ``entry`` is a sequence of 2-sequences; deliberately not required to be a
        # list of *tuples*, because stage serializers turn tuples into lists on the
        # way across the stage edge.
        self._table = {
            key: [(k[..., start:stop, :].contiguous(), v[..., start:stop, :].contiguous()) for k, v in entry]
            for key, entry in conditioning.branches.items()
        }
        self._dtype = dtype
        # Kept for the replay-miss message only. The settings that feed the
        # fingerprint are not *validated* against this stage (the fingerprint is
        # the check), but reporting what the reasoner resolved turns an opaque hash
        # mismatch into the two values an operator has to compare.
        self._reasoner_settings = (
            f"height={conditioning.height}, width={conditioning.width}, "
            f"max_sequence_length={conditioning.max_sequence_length}, "
            f"use_system_prompt={conditioning.use_system_prompt}"
        )

    def _check_layout(self, conditioning: Cosmos3TextConditioning) -> None:
        """Compare the reasoner's declared layout with what this stage consumes.

        The contract already validated itself against its own tensors, so this is
        the *cross-stage* check: same checkpoint, same transformer config. TP size
        is deliberately not part of it -- the payload is unsharded, so the two
        stages are free to differ -- but both sizes are reported, because they are
        the context an operator needs when the head counts do not line up.
        """
        expected = {
            "num_layers": self.num_hidden_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
        }
        mismatched = {
            field: (getattr(conditioning, field), want)
            for field, want in expected.items()
            if getattr(conditioning, field) != want
        }
        if not mismatched:
            return
        detail = ", ".join(
            f"{field}={got} from reasoner, {want} here" for field, (got, want) in sorted(mismatched.items())
        )
        raise RuntimeError(
            f"Cosmos3 reasoner and generator stages disagree on the UND K/V layout: {detail}. "
            f"Reasoner ran at tensor_parallel_size={conditioning.reasoner_tp_size}, this stage at "
            f"{_tp_world_size()}. The K/V handoff is TP-independent, so this is a checkpoint or "
            "transformer-config mismatch: both stages must load the same model."
        )

    def clear(self) -> None:
        """Drop the installed payload.

        The stub is long-lived pipeline state while a payload belongs to exactly
        one request, so the generator clears it once the request is done. That
        keeps a stale branch from ever being replayable for a later request and
        releases the host tensors instead of holding the last request's K/V until
        the next one arrives.
        """
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
                "Cosmos3 generator stage has no reasoner K/V for this prompt "
                f"branch (fingerprint={key}, known={sorted(self._table)}). The "
                "reasoner and generator stages must tokenize identically: check "
                "that max_sequence_length/use_system_prompt, the geometry and the "
                "negative prompt reach both stages unchanged in sampling_params. "
                f"The reasoner resolved {self._reasoner_settings}."
            )
        device = text_ids.device
        dtype = self._dtype
        return [
            (
                k.to(device=device, dtype=dtype, non_blocking=True),
                v.to(device=device, dtype=dtype, non_blocking=True),
            )
            for k, v in entry
        ]


class _Cosmos3TowerPipeline(Cosmos3OmniDiffusersPipeline):
    """Shared plumbing for the two single-tower pipelines.

    ``_remap_ckpt_key`` is inherited unchanged, so both stages read the *same*
    checkpoint files. The checkpoint interleaves the towers inside every
    ``layers.{i}`` entry (``mlp`` vs ``mlp_moe_gen``, ``self_attn.to_q`` vs
    ``self_attn.add_q_proj``) and the inherited remap already routes those to
    ``language_model.layers.*`` vs ``gen_layers.*``. Keys belonging to the tower
    this stage does not own therefore match no live parameter and are filtered out
    by the inherited ``load_weights`` -- which is what lets each stage load half a
    model without a separately prepared checkpoint. The filter runs after the
    tensor has been read, so this saves device memory, not startup I/O; see the
    module docstring.
    """

    def __init__(self, *, od_config: Any, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        # ``super().__init__`` built only the owned tower, because
        # ``cosmos3_owned_towers`` reached the transformer. Verify that, then bind
        # whatever stands in for the tower this stage does not run.
        self._bind_owned_tower()

    def _bind_owned_tower(self) -> None:
        raise NotImplementedError


class Cosmos3ReasonerPipeline(_Cosmos3TowerPipeline):
    """Stage 0 -- the UND / autoregressive tower only.

    Runs the language-model tower over the formatted prompt (and the negative
    prompt, for the unconditional CFG branch) and returns per-layer K/V. This
    stage never allocates latents, never denoises and never touches the VAE.
    """

    #: Only the UND tower is constructed here: no GEN blocks are allocated at all.
    cosmos3_owned_towers: ClassVar[tuple[str, ...]] = ("reasoner",)

    # Skip the engine's synthetic warmup run. Same mechanism as the generator
    # (see that class), different trigger: ``_dummy_run`` builds
    # ``{"prompt": "dummy run"}`` with no ``modalities`` key, and stock Cosmos3
    # semantics read absent modalities as *video*, not image -- so the T2I guard
    # in ``forward`` below correctly refuses it, which kills every worker in this
    # stage during startup. A UND-only forward has little to warm up regardless:
    # no latents, no denoise loop, no VAE.
    dummy_run_num_frames: ClassVar[int] = 0

    def _bind_owned_tower(self) -> None:
        _require_unowned_absent(self.transformer.gen_layers, "GEN (generator)")

    def forward(self, req: Any) -> Any:  # type: ignore[override]
        """Engine entry point: emit K/V instead of pixels.

        The returned ``DiffusionOutput`` payload is the flat handoff dict that
        ``get_cosmos3_reasoner_post_process_func`` wraps into the
        payload/metadata envelope. This stage is ``final_output=False``, so it
        never reaches the client -- ``reasoner2generator`` consumes it.
        """
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
            # Matches the stock forward, which normalizes a missing negative
            # prompt to "" before tokenizing the unconditional branch.
            negative_prompt = prompt_data.get("negative_prompt") or ""

        conditioning = self.encode_text_conditioning(prompt, negative_prompt, req.sampling_params)
        return DiffusionOutput(output=conditioning.to_payload())

    def encode_text_conditioning(self, prompt: str, negative_prompt: str, sp: Any) -> Cosmos3TextConditioning:
        """Run the UND tower and build the reasoner -> generator contract.

        Every geometry/tokenization value is resolved exactly the way the stock
        T2I ``forward`` resolves it, because the generator stage re-derives the
        same values and any divergence shows up as a replay-table miss.

        Deliberately *not* decorated with ``torch.inference_mode()``:
        ``DiffusionModelRunner._execute_request_list`` already picks the right
        grad context for the configuration, and deliberately selects plain
        ``no_grad`` when HSDP or distributed layerwise offload is on.
        """
        # Resolved through the *same* helpers the stock T2I ``forward`` uses, so
        # the two paths cannot drift apart and make the fingerprints disagree.
        # ``default_use_system_prompt=False`` is what the stock path resolves to
        # here, because its ``is_v2v`` is always False for T2I.
        height, width = self._resolve_t2i_geometry(sp)
        max_sequence_length, use_system_prompt, frame_rate = self._resolve_text_encode_params(
            sp,
            default_use_system_prompt=False,
        )
        guidance_scale = self._resolve_guidance_scale(sp, COSMOS3_T2I_DEFAULT_GUIDANCE_SCALE)

        # Inherited formatter/tokenizer: the generator stage runs the identical
        # call, which is what makes the fingerprints line up.
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
        # GEN latent geometry. ``freqs_und`` does not depend on it -- the UND
        # frequencies are a function of ``text_mask`` alone -- but passing the
        # real shape keeps this call identical to the co-located one.
        t = 1
        h = height // self.vae_scale_factor_spatial
        w = width // self.vae_scale_factor_spatial
        hp, wp, _, _ = transformer._pad_to_patch_size(h, w)
        dtype = transformer.proj_in.weight.dtype

        # ``_format_and_tokenize_prompts`` always returns an unconditional
        # branch, but ``diffuse`` only evaluates it when ``do_cfg``. Skipping it
        # here saves a full 31 B-parameter UND forward and halves the payload.
        # Both stages resolve guidance from the same sampling params, so they
        # agree on this; if they ever did not, the generator's replay lookup
        # would raise rather than silently produce a wrong image.
        do_cfg = guidance_scale > 1.0
        branches_to_encode = [(cond_ids, cond_mask)]
        if do_cfg and uncond_ids is not None:
            branches_to_encode.append((uncond_ids, uncond_mask))

        # UNSHARDING IS MANDATORY UNDER HSDP
        # ----------------------------------
        # We invoke ``transformer.language_model(...)`` directly, so FSDP2's
        # pre-forward hook on the *root* module (``transformer``) never fires.
        # ``_hsdp_shard_conditions`` wraps only the numbered blocks
        # (``language_model.layers.{i}`` / ``gen_layers.{i}``), which means
        # ``language_model.embed_tokens`` and ``.norm`` are root-managed: their
        # params are still sharded DTensors until the root unshards them.
        # Without this, the very first op fails with "aten.embedding.default got
        # mixed torch.Tensor and DTensor" -- observed, not hypothetical.
        # ``wan2_2/wan2_2_s2v_transformer.py`` (``encode_audio``) does exactly
        # this for the same reason. The guard keeps the non-HSDP path working,
        # where these methods do not exist.
        is_fsdp = hasattr(transformer, "unshard") and hasattr(transformer, "reshard")
        if is_fsdp:
            transformer.unshard()
        try:
            branches: dict[str, KVBranch] = {}
            for text_ids, text_mask in branches_to_encode:
                text_ids = text_ids.to(self.device)
                text_mask = text_mask.to(self.device)
                max_real_len = int(text_mask.sum(dim=1).max().item())

                freqs_und, _freqs_gen = transformer._compute_rope_freqs(text_mask, t, hp, wp, None, self.device, dtype)
                with transformer._offload_context("reasoner"):
                    cached_kv_full = transformer.language_model(text_ids, freqs_und)

                # Trim padding exactly as the co-located forward does, *then*
                # gather the TP-local heads: every rank derives ``max_real_len``
                # from the same mask, so the trimmed shapes agree and the
                # collective moves only real tokens. Shipping already-trimmed K/V
                # makes the generator's own trim a no-op and keeps the payload
                # proportional to the real prompt length. ``.cpu()`` also
                # materializes any DTensor-backed result into a plain tensor,
                # which is what has to cross the stage boundary.
                branches[fingerprint_text_ids(text_ids)] = [
                    (
                        _gather_kv_heads(k[:, :max_real_len]).contiguous().cpu(),
                        _gather_kv_heads(v[:, :max_real_len]).contiguous().cpu(),
                    )
                    for k, v in cached_kv_full
                ]
        finally:
            if is_fsdp:
                transformer.reshard()

        payload_mib = (
            sum(
                k.numel() * k.element_size() + v.numel() * v.element_size()
                for entry in branches.values()
                for k, v in entry
            )
            / 2**20
        )
        # Every branch has the same layout -- same tower, same config -- so one
        # tensor describes all of them. Read off the tensors that were actually
        # produced rather than recomputed from the config, so the contract cannot
        # describe a payload this stage did not emit.
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
            # Not an error: an oversized payload is still correct, just expensive
            # to serialize and ship. Almost always a symptom of a
            # ``max_sequence_length`` far larger than the prompt needs, since the
            # payload is trimmed to the real text length.
            logger.warning(
                "Cosmos3 reasoner: K/V payload is %.1f MiB (> %.1f MiB) for a %d-token "
                "conditioning length; every byte crosses the stage edge once per request. "
                "Consider lowering max_sequence_length (currently %d).",
                payload_mib,
                COSMOS3_UND_PAYLOAD_WARN_MIB,
                conditioning.conditioning_length,
                max_sequence_length,
            )
        return conditioning


class Cosmos3GeneratorPipeline(_Cosmos3TowerPipeline):
    """Stage 1 -- the GEN / diffusion tower only.

    The UND tower is never constructed here; ``language_model`` is a replay stub
    fed by the reasoner payload, and the stock denoise loop and VAE decode run
    unmodified.
    """

    #: Only the GEN tower is constructed here. The transformer still builds the
    #: mRoPE embedding that lives on ``language_model`` (the GEN frequencies need
    #: it every step) but none of the UND embedding table, blocks or norm.
    cosmos3_owned_towers: ClassVar[tuple[str, ...]] = ("generator",)

    #: Skip the engine's warmup run. ``DiffusionEngine._dummy_run`` returns
    #: before it even builds a request when this is <= 0, which is what we want:
    #: the synthetic warmup request carries no reasoner K/V, so the replay stub
    #: would have nothing to look up. There is no meaningful loss -- the real
    #: first request warms the same GEN kernels.
    dummy_run_num_frames: ClassVar[int] = 0

    def _bind_owned_tower(self) -> None:
        transformer = self.transformer
        language_model = transformer.language_model
        _require_unowned_absent(language_model.layers, "UND (reasoner)")
        # Swap the (block-less, rope-only) tower for the replay stub, carrying the
        # real rotary embedding across -- see _ReplayLanguageModel's docstring.
        # Both towers are built from ``transformer.num_hidden_layers``, so that is
        # also the per-layer K/V count the reasoner will ship.
        #
        # The expected K/V shape is read from the module that will actually receive
        # the replayed tensors, rather than recomputed from the config and the TP
        # world size. Cosmos3CrossAttention already resolved
        # ``num_kv_heads // tp_size`` for itself at construction time, so taking it
        # from there cannot disagree with the consumer.
        consumer = self._kv_consumer()
        transformer.language_model = _ReplayLanguageModel(
            transformer.num_hidden_layers,
            language_model.rotary_emb,
            num_kv_heads=consumer.num_kv_heads,
            num_kv_heads_local=consumer.num_kv_heads_local,
            kv_head_offset=_tp_rank() * consumer.num_kv_heads_local,
            head_dim=consumer.head_dim,
        )

    def _kv_consumer(self) -> torch.nn.Module:
        """The ``Cosmos3CrossAttention`` that consumes the replayed UND K/V.

        Every GEN block has one and they are all built from the same config, so the
        first block speaks for all of them.
        """
        gen_layers = self.transformer.gen_layers
        if not len(gen_layers):
            raise RuntimeError(
                "Cosmos3 generator stage has no GEN blocks, so there is nothing to "
                "replay reasoner K/V into. The stage did not build the tower it owns."
            )
        return gen_layers[0].cross_attention

    def forward(self, req: Any) -> Any:  # type: ignore[override]
        """Install the reasoner K/V, then run the stock denoise + decode path.

        The payload is installed for the duration of this request only. The stub
        outlives the request, so leaving a table behind would let a later request
        replay another request's conditioning if its fingerprints happened to
        match, and would pin the host K/V until the next request overwrote it.
        """
        payload = self._extract_und_payload(req)
        self.install_text_conditioning(payload)
        try:
            return super().forward(req)
        finally:
            self.transformer.language_model.clear()

    @staticmethod
    def _extract_und_payload(req: Any) -> dict[str, Any]:
        """Find the reasoner payload on the incoming request.

        ``prompt["extra"]`` is where ``reasoner2generator`` puts it. The
        ``sampling_params.extra_args`` fallback mirrors GLM-Image's DiT stage,
        which accepts ``prior_token_ids`` from either place -- handy for driving
        this stage directly in a single-process test.
        """
        prompt_data = req.prompts[0] if req.prompts else ""
        if isinstance(prompt_data, dict):
            extra = prompt_data.get("extra") or {}
            if KV_KEY in extra:
                return extra

        sp = getattr(req, "sampling_params", None)
        extra_args = getattr(sp, "extra_args", None) or {}
        if KV_KEY in extra_args:
            return extra_args

        raise ValueError(
            "Cosmos3 generator stage received a request without reasoner K/V in "
            f"prompt['extra'][{KV_KEY!r}] or sampling_params.extra_args[{KV_KEY!r}]. "
            "This stage cannot run standalone: route requests through stage 0 "
            "(reasoner) via the stage router."
        )

    def install_text_conditioning(self, payload: dict[str, Any]) -> Cosmos3TextConditioning:
        """Load this rank's shard of the reasoner's conditioning into the stub."""
        conditioning = Cosmos3TextConditioning.from_payload(payload)
        stub = self.transformer.language_model
        if not isinstance(stub, _ReplayLanguageModel):
            raise RuntimeError(
                "Cosmos3 generator stage is not running the replay UND stub; "
                "the pipeline was not built by Cosmos3GeneratorPipeline."
            )
        stub.install(conditioning, dtype=self.transformer.proj_in.weight.dtype)
        logger.info(
            "Cosmos3 generator: installed reasoner K/V for %d branch(es) (%.1f MiB, "
            "heads [%d, %d) of %d); UND tower not loaded on this stage",
            conditioning.num_branches,
            conditioning.payload_mib,
            stub.kv_head_offset,
            stub.kv_head_offset + stub.num_kv_heads_local,
            conditioning.num_kv_heads,
        )
        return conditioning


def get_cosmos3_reasoner_post_process_func(od_config: Any):
    """Postprocessor for the reasoner stage: pass the UND K/V through intact.

    The stock Cosmos3 postprocessor rejects anything that is not an image or
    video payload, so the reasoner needs its own. It emits the payload/metadata
    envelope shape that ``normalize_diffusion_postprocess_output`` understands,
    and parks the K/V under the ``trajectory`` payload key.

    ``trajectory`` is the one payload key that survives the output formatter
    unmodified: ``_build_multimodal_output`` copies only ``audio``, ``actions``
    and ``trajectory`` into ``multimodal_output``, and ``trajectory`` (unlike
    ``actions``) carries no metadata-validation rules. Its ``latents``,
    ``timesteps``, ``log_probs`` and ``decoded`` sub-keys are reserved -- the
    formatter siphons those into dedicated ``OmniRequestOutput`` fields -- so
    this payload deliberately uses only the two Cosmos3 K/V keys.

    Because the primary-key inference maps a ``{"trajectory": ...}``-only payload
    to ``None``, the stage reports zero images and the K/V rides out on
    ``multimodal_output``, which is what the stage connectors preserve across the
    stage edge.
    """
    del od_config  # No per-engine state: this is a pure repackaging step.

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
                "Cosmos3 reasoner postprocess expected a dict payload containing "
                f"{KV_KEY!r}, got {type(output).__name__} with keys "
                f"{sorted(output) if isinstance(output, dict) else '<n/a>'}."
            )
        meta = output.get(META_KEY) or {}
        return {
            "payload": {
                "trajectory": {
                    KV_KEY: output[KV_KEY],
                    META_KEY: meta,
                },
            },
            # Unknown metadata groups are explicitly tolerated by
            # ``validate_diffusion_metadata``; never use the reserved
            # ``internal`` group, which must not escape public formatting.
            "metadata": {"cosmos3_und": dict(meta)},
        }

    return post_process_func
