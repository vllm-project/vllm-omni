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

CFG BRANCHES
------------
Guidance runs the UND tower twice -- once for the conditional prompt and once
for the unconditional/negative one -- with different token streams and hence
different K/V. The stub keys its replay table by a fingerprint of ``text_ids``,
so each branch gets its own entry and the lookup cannot cross-wire them. Both
stages tokenize with the same inherited code path and the same geometry, so the
fingerprints agree by construction.

WHY THE TOWER IS DROPPED IN ``__init__`` AND NOT IN ``load_weights``
-------------------------------------------------------------------
``DiffusersPipelineLoader.load_weights`` snapshots the set of parameters it
expects to fill *before* delegating to the model::

    weights_to_load = self._get_expected_parameter_names(model)   # snapshot
    loaded_weights = model.load_weights(self.get_all_weights(model))
    ...
    _check_unloaded_weights(weights_to_load - loaded_weights, ...)

With ``quant_config is None`` that last check raises for any expected parameter
that no checkpoint tensor filled. Dropping a tower from inside ``load_weights``
is therefore too late -- the snapshot already contains the dropped tower's
parameters and every one of them shows up as unloaded. Dropping in ``__init__``
means those parameters never exist, so they are never snapshotted and never
allocated. That is where the win is: ~58 GiB of device memory per stage.

WHAT THE SPLIT DOES *NOT* SAVE: CHECKPOINT READS
-----------------------------------------------
Both stages still stream the whole checkpoint. ``Cosmos3.load_weights`` filters
by name *after* ``safetensors_weights_iterator`` has already materialized each
tensor, so the dropped tower's tensors are read from disk and discarded rather
than skipped (the "kept N/M tensors" line it logs is the filter, not the read).
Splitting the towers therefore roughly doubles aggregate startup read I/O across
the two stages instead of halving it. Fixing that means teaching the loader to
skip tensors by name before materializing them, which is a loader change and not
in scope here.
"""

from __future__ import annotations

import hashlib
from typing import Any, ClassVar

import torch
from vllm.distributed import get_tensor_model_parallel_world_size
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


def fingerprint_text_ids(text_ids: torch.Tensor) -> str:
    """Stable content hash of a token-id tensor, used as the replay-table key.

    Hashing the ids (rather than trusting call order) keeps the conditional and
    unconditional CFG branches from being confused for one another even if the
    denoise loop changes the order in which it evaluates them.
    """
    flat = text_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1).numpy()
    return hashlib.sha256(flat.tobytes()).hexdigest()[:32]


def _tp_world_size() -> int:
    """This process's TP world size, or 1 when no model-parallel group exists.

    Reported in the handoff metadata so a stage-configuration mismatch can name
    both stages' TP sizes. Purely diagnostic -- the K/V layout that actually gets
    validated is read off the tensors and off the consuming attention module, not
    recomputed from this. A process with no TP group is at TP 1 by definition,
    which is also what makes this callable from a single-process test.
    """
    return get_tensor_model_parallel_world_size() if model_parallel_is_initialized() else 1


def _drop_blocks(module_list: torch.nn.ModuleList, label: str) -> None:
    """Empty a tower's block container in place, before weights are loaded.

    The container object is kept (rather than replaced with ``None``) because the
    offload rings, cache-dit adapter and ``_model_cpu_offload_components`` all
    introspect it. ``get_blocks_from_dit`` raises only when a declared attribute
    is *missing*; an empty container merely logs a "no blocks found, skipping"
    warning, which is the correct outcome here.
    """
    logger.info("Cosmos3 disagg: dropped %d %s block(s); their weights will not be loaded", len(module_list), label)
    del module_list[:]


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
    * ``_model_cpu_offload_components`` returns ``{"reasoner":
      [self.language_model.layers], ...}``, so ``layers`` must exist. It is an
      empty ``ModuleList`` here: there is nothing to swap in or out.

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
        num_kv_heads_local: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.rotary_emb = rotary_emb
        self.layers = torch.nn.ModuleList()
        # The per-layer K/V shape this stage's own cross-attention will consume,
        # derived from this stage's config and TP world size -- see ``install``.
        self.num_kv_heads_local = num_kv_heads_local
        self.head_dim = head_dim
        self._table: dict[str, Any] = {}
        self._dtype: torch.dtype | None = None

    def install(self, table: dict[str, Any], dtype: torch.dtype | None = None) -> None:
        """Validate a reasoner payload against this stage's layout, then hold it.

        WHY THE SHAPES ARE CHECKED HERE AND NOT IN ``forward``
        -----------------------------------------------------
        UND K/V is TP-sharded: ``Cosmos3SelfAttention`` produces
        ``[B, S_und, num_kv_heads // tp_size, head_dim]`` and
        ``Cosmos3CrossAttention`` consumes exactly that. The reasoner stage runs
        with its *own* ``tensor_parallel_size``, and nothing in the stage
        plumbing requires the two stages to agree on it -- so a payload built at
        TP 1 and replayed on a TP 2 generator carries twice the KV heads this
        stage expects. Left unchecked that surfaces either as a shape error deep
        inside attention or, worse, as silently wrong conditioning.

        Validating at install time costs ``2 * num_hidden_layers`` shape reads
        once per request, rather than once per denoising step, and reports the
        mismatch in terms of the two stages' configurations.
        """
        for key, entry in table.items():
            if len(entry) != self.num_hidden_layers:
                raise RuntimeError(
                    f"Cosmos3 reasoner K/V for branch {key} has {len(entry)} layer(s), "
                    f"generator expects {self.num_hidden_layers}. The two stages must load "
                    "the same checkpoint and the same transformer config."
                )
            for layer_idx, (k, v) in enumerate(entry):
                for label, tensor in (("K", k), ("V", v)):
                    if tensor.ndim != 4:
                        raise RuntimeError(
                            f"Cosmos3 reasoner {label} for branch {key} layer {layer_idx} has "
                            f"{tensor.ndim} dim(s), expected 4 ([B, S_und, num_kv_heads_local, head_dim])."
                        )
                    if tensor.shape[-2:] != (self.num_kv_heads_local, self.head_dim):
                        raise RuntimeError(
                            f"Cosmos3 reasoner {label} for branch {key} layer {layer_idx} is shaped "
                            f"{tuple(tensor.shape)}, but this generator stage consumes "
                            f"[B, S_und, {self.num_kv_heads_local}, {self.head_dim}]. UND K/V is "
                            "TP-sharded, so the reasoner and generator stages must run with the "
                            "same tensor_parallel_size and the same transformer config."
                        )
                if k.shape != v.shape:
                    raise RuntimeError(
                        f"Cosmos3 reasoner K/V for branch {key} layer {layer_idx} disagree on shape: "
                        f"K={tuple(k.shape)}, V={tuple(v.shape)}."
                    )
        self._table = table
        self._dtype = dtype

    def clear(self) -> None:
        """Drop the installed payload.

        The stub is long-lived pipeline state while a payload belongs to exactly
        one request, so the generator clears it once the request is done. That
        keeps a stale branch from ever being replayable for a later request and
        releases the pinned host tensors instead of holding the last request's
        K/V until the next one arrives.
        """
        self._table = {}
        self._dtype = None

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
                "negative prompt reach both stages unchanged in sampling_params."
            )
        device = text_ids.device
        dtype = self._dtype
        # ``entry`` is a sequence of 2-sequences; it is deliberately not required
        # to be a list of *tuples*, because stage serializers turn tuples into
        # lists on the way across the stage edge.
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
    ``language_model.layers.*`` vs ``gen_layers.*``. Keys belonging to the
    dropped tower therefore match no live parameter and are filtered out by the
    inherited ``load_weights`` -- which is what lets each stage load half a model
    without a separately prepared checkpoint. The filter runs after the tensor
    has been read, so this saves device memory, not startup I/O; see the module
    docstring.
    """

    def __init__(self, *, od_config: Any, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        # Must happen here, not in load_weights -- see the module docstring.
        # ``weights_sources`` is assigned by the base __init__, so the loader
        # still finds the checkpoint after this returns.
        self._drop_unused_tower()

    def _drop_unused_tower(self) -> None:
        raise NotImplementedError


class Cosmos3ReasonerPipeline(_Cosmos3TowerPipeline):
    """Stage 0 -- the UND / autoregressive tower only.

    Runs the language-model tower over the formatted prompt (and the negative
    prompt, for the unconditional CFG branch) and returns per-layer K/V. This
    stage never allocates latents, never denoises and never touches the VAE.
    """

    # Skip the engine's synthetic warmup run. Same mechanism as the generator
    # (see that class), different trigger: ``_dummy_run`` builds
    # ``{"prompt": "dummy run"}`` with no ``modalities`` key, and stock Cosmos3
    # semantics read absent modalities as *video*, not image -- so the T2I guard
    # in ``forward`` below correctly refuses it, which kills every worker in this
    # stage during startup. A UND-only forward has little to warm up regardless:
    # no latents, no denoise loop, no VAE.
    dummy_run_num_frames: ClassVar[int] = 0

    def _drop_unused_tower(self) -> None:
        _drop_blocks(self.transformer.gen_layers, "GEN (generator)")

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

        payload = self.encode_prompt_to_kv(prompt, negative_prompt, req.sampling_params)
        return DiffusionOutput(output=payload)

    def encode_prompt_to_kv(self, prompt: str, negative_prompt: str, sp: Any) -> dict[str, Any]:
        """Run the UND tower and build the reasoner -> generator payload.

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
        branches = [(cond_ids, cond_mask)]
        if do_cfg and uncond_ids is not None:
            branches.append((uncond_ids, uncond_mask))

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
            table: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
            for text_ids, text_mask in branches:
                text_ids = text_ids.to(self.device)
                text_mask = text_mask.to(self.device)
                max_real_len = int(text_mask.sum(dim=1).max().item())

                freqs_und, _freqs_gen = transformer._compute_rope_freqs(text_mask, t, hp, wp, None, self.device, dtype)
                with transformer._offload_context("reasoner"):
                    cached_kv_full = transformer.language_model(text_ids, freqs_und)

                # Trim padding exactly as the co-located forward does. Shipping
                # already-trimmed K/V makes the generator's own trim a no-op and
                # keeps the payload proportional to the real prompt length.
                # ``.cpu()`` also materializes any DTensor-backed result into a
                # plain tensor, which is what has to cross the stage boundary.
                table[fingerprint_text_ids(text_ids)] = [
                    (k[:, :max_real_len].contiguous().cpu(), v[:, :max_real_len].contiguous().cpu())
                    for k, v in cached_kv_full
                ]
        finally:
            if is_fsdp:
                transformer.reshard()

        payload_mib = (
            sum(k.numel() * k.element_size() + v.numel() * v.element_size() for kv in table.values() for k, v in kv)
            / 2**20
        )
        # Every branch has the same layout -- same tower, same config -- so one
        # tensor describes all of them.
        sample_branch = next(iter(table.values()))
        sample_k = sample_branch[0][0]
        logger.info(
            "Cosmos3 reasoner: %d branch(es) (cfg=%s), K/V payload=%.1f MiB, target=%dx%d",
            len(table),
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
                max(k.shape[1] for kv in table.values() for k, _v in kv),
                max_sequence_length,
            )
        return {
            KV_KEY: table,
            META_KEY: {
                "height": height,
                "width": width,
                "max_sequence_length": max_sequence_length,
                "use_system_prompt": use_system_prompt,
                "num_branches": len(table),
                "payload_mib": round(payload_mib, 1),
                # K/V layout, so the generator can report a stage-configuration
                # mismatch in terms of the two stages' settings instead of a bare
                # shape error from inside attention. Read off the tensors that were
                # actually produced rather than recomputed from the config, so it
                # cannot describe a payload this stage did not emit. UND K/V is
                # TP-sharded, so ``tp_size`` belongs here too -- it is the one part
                # of the layout no shape reveals.
                "num_layers": len(sample_branch),
                "num_kv_heads_local": int(sample_k.shape[-2]),
                "head_dim": int(sample_k.shape[-1]),
                "tp_size": _tp_world_size(),
            },
        }


class Cosmos3GeneratorPipeline(_Cosmos3TowerPipeline):
    """Stage 1 -- the GEN / diffusion tower only.

    Replaces the UND tower with a replay stub fed by the reasoner payload, then
    runs the stock denoise loop and VAE decode unmodified.
    """

    #: Skip the engine's warmup run. ``DiffusionEngine._dummy_run`` returns
    #: before it even builds a request when this is <= 0, which is what we want:
    #: the synthetic warmup request carries no reasoner K/V, so the replay stub
    #: would have nothing to look up. There is no meaningful loss -- the real
    #: first request warms the same GEN kernels.
    dummy_run_num_frames: ClassVar[int] = 0

    def _drop_unused_tower(self) -> None:
        transformer = self.transformer
        language_model = transformer.language_model
        _drop_blocks(language_model.layers, "UND (reasoner)")
        # Swap the (now block-less) tower for the replay stub, carrying the real
        # rotary embedding across -- see _ReplayLanguageModel's docstring. Both
        # towers are built from ``transformer.num_hidden_layers``, so that is also
        # the per-layer K/V count the reasoner will ship.
        #
        # The expected K/V shape is read from the module that will actually receive
        # the replayed tensors, rather than recomputed from the config and the TP
        # world size. Cosmos3CrossAttention already resolved
        # ``num_kv_heads // tp_size`` for itself at construction time, so taking it
        # from there cannot disagree with the consumer.
        consumer = self._kv_consumer()
        self.transformer.language_model = _ReplayLanguageModel(
            transformer.num_hidden_layers,
            language_model.rotary_emb,
            num_kv_heads_local=consumer.num_kv_heads_local,
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
                "replay reasoner K/V into. _drop_unused_tower dropped the wrong tower."
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
        self.install_und_kv(payload)
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

    def install_und_kv(self, payload: dict[str, Any]) -> None:
        """Load the reasoner's K/V into the replay stub for this request."""
        table = payload.get(KV_KEY) or {}
        if not table:
            raise ValueError("Cosmos3 generator stage received an empty reasoner K/V payload.")
        stub = self.transformer.language_model
        if not isinstance(stub, _ReplayLanguageModel):
            raise RuntimeError(
                "Cosmos3 generator stage is not running the replay UND stub; "
                "the pipeline was not built by Cosmos3GeneratorPipeline."
            )
        meta = payload.get(META_KEY) or {}
        self._check_meta_layout(meta, stub)
        stub.install(table, dtype=self.transformer.proj_in.weight.dtype)
        logger.info(
            "Cosmos3 generator: installed reasoner K/V for %d branch(es) (%s MiB); UND tower skipped",
            len(table),
            meta.get("payload_mib", "?"),
        )

    def _check_meta_layout(self, meta: dict[str, Any], stub: _ReplayLanguageModel) -> None:
        """Compare the reasoner's declared K/V layout with this stage's.

        ``_ReplayLanguageModel.install`` already validates the tensors themselves,
        so this is not what makes replay safe -- it is what makes a
        stage-configuration mistake *diagnosable*. The reasoner reports the
        ``tensor_parallel_size`` it sharded at, which no tensor shape reveals: a
        payload built at TP 2 with 16 KV heads and one built at TP 1 with 8 are
        indistinguishable by shape alone, so without this the operator sees a
        shape mismatch with no hint that the two stages' TP sizes disagree.

        Silent when the reasoner reported no layout at all (an older stage, or a
        hand-built payload in a test); ``install`` still checks those.
        """
        expected = {
            "num_layers": stub.num_hidden_layers,
            "num_kv_heads_local": stub.num_kv_heads_local,
            "head_dim": stub.head_dim,
        }
        mismatched = {
            field: (meta[field], want) for field, want in expected.items() if field in meta and meta[field] != want
        }
        if not mismatched:
            return
        detail = ", ".join(
            f"{field}={got} from reasoner, {want} here" for field, (got, want) in sorted(mismatched.items())
        )
        raise RuntimeError(
            f"Cosmos3 reasoner and generator stages disagree on the UND K/V layout: {detail}. "
            f"Reasoner ran at tensor_parallel_size={meta.get('tp_size', '?')}, this stage at "
            f"{_tp_world_size()}. UND K/V is TP-sharded, so both stages must "
            "load the same checkpoint and run with the same tensor_parallel_size."
        )


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
