# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-0 talker for MiniMax Music 3.

A Qwen3 backbone predicts one audio frame per decode step. Each frame is
eight RVQ codebooks deep: the backbone's own ``lm_head`` emits ``c0`` and the
four-layer depth decoder walks ``c1..c7``. The conditioning signal handed to
the acoustic stage is the backbone hidden state concatenated with the seven
depth hidden states, so 8 x 4096 = 32768 dimensions per frame.

Two things make this model unusual inside vLLM:

* **Every request decodes as two rows.** Classifier-free guidance is not
  optional here. The conditioned row sees the real prompt and the
  unconditioned row sees the same prompt with the caption-and-lyrics span
  replaced by ``<|audio_cfg|>``. Codes are drawn from the guided blend and
  then fed back to *both* rows, so the pair stays token-identical. Rows are
  paired by ``cfg_pair_id`` rather than by adjacency, because batch position
  is not stable across engine steps.
* **The decode input is not a token embedding.** After prefill, each step's
  input is the composed embedding of the previous frame's eight codes. The
  sampled ``c0`` token id still flows through vLLM's normal machinery so that
  stop handling and accounting work, but the embedding it would produce is
  replaced before the backbone runs.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen3 import Qwen3Model
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.v1.outputs import SamplerOutput

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .chunking import chunk_windows
from .constants import (
    AR_CHUNK_HOP_FRAMES,
    AUDIO_VOCAB_SIZE,
    BACKBONE_HIDDEN_SIZE,
    C0_VOCAB_SIZE,
    DEFAULT_MAX_AUDIO_FRAMES,
    NUM_CODEBOOKS,
)
from .depth_graph import DepthGraphCache
from .frame_state import RequestARState
from .prompt import AUDIO_CODE_OFFSET, SPECIAL_TOKEN_IDS
from .rvq_decoder import RVQDepthDecoder
from .sampling import (
    apply_c0_cfg,
    apply_depth_cfg,
    draw_from_gumbel,
    gumbel_noise,
    sample_topk_seeded,
)
from .staging import PinnedIndexStage
from .weights import load_depth_decoder_weights

logger = init_logger(__name__)

__all__ = ["MiniMaxMusic3TalkerForConditionalGeneration"]

_COND = "cond"
_UNCOND = "uncond"
_STOP_TOKEN = SPECIAL_TOKEN_IDS["<|audio_end|>"]
# Emitted on the step a song ends, to keep the request alive for one more step
# while its terminal span is delivered. It is never fed back as audio: the
# state is already marked finished, so no frame is composed from it.
_HOLD_TOKEN = AUDIO_CODE_OFFSET

# The unconditioned twin's request id is the conditioned one plus this suffix,
# which is what lets a pair be recognized without a serving-layer injection.
_CFG_UNCOND_SUFFIX = "__cfg_uncond"


@dataclass(frozen=True)
class _RowConstants:
    """Per-request values that never change once the request is admitted."""

    request_id: str
    role: str
    pair_id: str
    seed: int
    max_frames: int


def _request_key(value: Any) -> str:
    """Normalize a request id to the form used to key per-request state.

    The engine injects ``global_request_id`` as a single-element list, while
    ``on_requests_finished`` receives the bare string. Keying state on the
    unwrapped value keeps allocation and release on the same key; getting this
    wrong leaks a song's worth of frames per request.
    """
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    return "" if value is None else str(value)


def _pair_id_from_request(request_id: str) -> str:
    """Derive the CFG pair id when the request does not carry one."""
    if request_id.endswith(_CFG_UNCOND_SUFFIX):
        return request_id[: -len(_CFG_UNCOND_SUFFIX)]
    return request_id


def _role_from_request(request_id: str) -> str:
    """Derive the CFG role from the companion suffix."""
    return _UNCOND if request_id.endswith(_CFG_UNCOND_SUFFIX) else _COND


def _first_int(value: Any, default: int) -> int:
    """Read an int that the serving layer wraps in a single-element list."""
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


class MiniMaxMusic3TalkerForConditionalGeneration(nn.Module):
    """Qwen3 backbone plus the audio embedding table and RVQ depth decoder."""

    # Sampling, feedback and window emission are all model-owned.
    prefer_model_sampler: bool = True
    have_multimodal_outputs: bool = True
    has_postprocess: bool = True
    skips_model_sampler_output_token_history: bool = True
    postprocess_uses_hidden_states: bool = False
    postprocess_uses_multimodal_outputs: bool = False
    postprocess_uses_req_infos: bool = True
    supports_omni_query_start_loc: bool = True
    # A row still completing a chunked prefill reaches the sampler but its
    # token is discarded; drawing a frame for it would corrupt the song.
    requires_request_sample_eligibility: bool = True
    # The acoustic stage consumes the composed 32768-d frame, never the raw
    # backbone hidden state, so suppress the runner's automatic "hidden" key.
    omni_pooler_payload_include_hidden: bool = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        config = vllm_config.model_config.hf_config
        self.config = config
        hidden_size = int(getattr(config, "hidden_size", BACKBONE_HIDDEN_SIZE))

        self.model = Qwen3Model(
            vllm_config=vllm_config,
            prefix=f"{prefix}.model" if prefix else "model",
        )
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                int(config.vocab_size),
                hidden_size,
                prefix=f"{prefix}.lm_head" if prefix else "lm_head",
            )
        self.logits_processor = LogitsProcessor(int(config.vocab_size))

        # c1..c7 share one table; codebook i occupies rows [i*1024, (i+1)*1024).
        self.audio_embeddings = nn.Embedding(AUDIO_VOCAB_SIZE * (NUM_CODEBOOKS - 1), hidden_size)
        self.rvq_decoder = RVQDepthDecoder(hidden_size=hidden_size)

        self.register_buffer(
            "audio_embedding_offsets",
            (torch.arange(NUM_CODEBOOKS - 1) * AUDIO_VOCAB_SIZE).unsqueeze(0),
            persistent=False,
        )
        # The only token ids c0 sampling can return, in vocabulary order:
        # <|audio_end|> first, then the c0 code range. Column 0 therefore means
        # stop and column code+1 means code.
        self.register_buffer(
            "c0_logit_ids",
            torch.cat(
                (
                    torch.tensor([_STOP_TOKEN], dtype=torch.long),
                    torch.arange(
                        AUDIO_CODE_OFFSET,
                        AUDIO_CODE_OFFSET + C0_VOCAB_SIZE,
                        dtype=torch.long,
                    ),
                )
            ),
            persistent=False,
        )
        self.frame_embedding_scale = float(NUM_CODEBOOKS) ** -0.5

        scheduler_config = getattr(vllm_config, "scheduler_config", None)
        # Guidance doubles the rows, so the batch is twice the admitted
        # request count.
        self._max_rows = max(64, 2 * int(getattr(scheduler_config, "max_num_seqs", 16)))

        # Per-step scratch, rebuilt every step from request-keyed state.
        # Nothing here may survive a step: vLLM compacts and reorders the
        # persistent batch, so a row index is only meaningful within the step
        # that produced it.
        self._last_logits_hidden: torch.Tensor | None = None
        self._step_engine_ids: list[str] = []
        self._row_request_ids: list[str] = []
        self._row_roles: list[str] = []
        self._row_pair_ids: list[str] = []
        self._row_seeds: list[int] = []
        self._row_max_frames: list[int] = []
        self._row_prompt_tokens: list[int] = []
        self._row_sample_eligible: list[bool] = []
        self._feedback_positions: torch.Tensor | None = None
        self._feedback_codes: torch.Tensor | None = None

        # Per-step index vectors are staged through pinned memory rather than
        # built with ``torch.tensor(..., device="cuda")``; see
        # ``staging.PinnedIndexStage`` for why that matters here. Allocated on
        # first use because the device is not known at construction time.
        # Rows are (cond_row, uncond_row, seed, position).
        self._sample_stage: PinnedIndexStage | None = None
        # One row, the flat token offsets of this step's decoding rows.
        self._feedback_stage: PinnedIndexStage | None = None

        # Depth-loop draw offsets, ``[1..NUM_CODEBOOKS)``. Held as a buffer so
        # the per-step position arithmetic is one broadcast add rather than one
        # scalar add per codebook.
        self.register_buffer(
            "depth_draw_offsets",
            torch.arange(1, NUM_CODEBOOKS, dtype=torch.int64),
            persistent=False,
        )

        # Frame feedback slab. A captured decode graph replays fixed addresses
        # and fixed shapes, so the rows to overwrite cannot be expressed as a
        # variable-length ``index_copy`` against a freshly built index tensor.
        # Instead every row of the reserved slab carries a codes entry and a
        # flag, both written in place before the graph runs, and the forward
        # selects with ``torch.where`` over the whole row set. Sized to the
        # widest decode batch; wider prefill batches take the eager path in
        # ``_apply_frame_feedback``, and prefill is never graph-replayed.
        self.register_buffer(
            "frame_feedback_codes",
            torch.zeros((self._max_rows, NUM_CODEBOOKS), dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "frame_feedback_active",
            torch.zeros(self._max_rows, dtype=torch.bool),
            persistent=False,
        )

        # Cross-step state. Keyed by CFG pair id, because a guided request is
        # two engine rows sharing one song. The engine renames requests to
        # internal UUIDs on admission, so the pair id is derived from the
        # user-facing id and the engine ids are mapped onto it separately.
        self._states: dict[str, RequestARState] = {}
        self._pair_of_engine_id: dict[str, str] = {}
        # Per-request constants, keyed by engine id. ``forward`` is the only
        # hook the engine hands the sampling extra args to, but its Python body
        # does not run on a step replayed from a CUDA graph, so what it learns
        # is cached here and re-bound to rows in ``prepare_runner_inputs``.
        self._row_constants: dict[str, _RowConstants] = {}
        self._row_binding_complete = False
        self._incomplete_steps = 0

        # The depth loop is recorded per pair count and replayed after that.
        self._depth_graphs = DepthGraphCache(self._depth_decode_eager)

    # ---------------------------------------------------------------- forward

    def prepare_runner_inputs(
        self,
        *,
        req_ids: list[str],
        num_computed_tokens: Any,
        num_scheduled_tokens: Any,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bind this step's batch rows to requests, before the forward runs.

        This is the only hook that sees the authoritative row-to-request
        mapping for the step about to execute. vLLM compacts and reorders the
        persistent batch whenever a request finishes or is preempted, so any
        state indexed by row number from a previous step is stale by the time
        it is read. Everything the model carries across steps is therefore
        keyed by request id and gathered into row order here.

        It also resolves which rows are decoding. A row is decoding when its
        prompt is already computed and it is advancing by a single token; a
        row still working through a chunked prefill is not, and must keep its
        real prompt embeddings.

        Python runs here on every step, including steps whose forward is
        replayed from a CUDA graph rather than executed, so this is also where
        the row-to-request binding that ``sample`` reads has to be rebuilt.
        """
        self._step_engine_ids = [str(rid) for rid in req_ids]
        computed = [int(value) for value in num_computed_tokens]
        scheduled = [int(value) for value in num_scheduled_tokens]
        # Prompt length for a row that is completing its prefill on this step.
        # Only meaningful when the row is about to produce its first logit,
        # which is the only moment state is created.
        self._row_prompt_tokens = [computed[row] + scheduled[row] for row in range(len(req_ids))]

        # Flat token offset of each row's first scheduled token.
        offsets: list[int] = []
        running = 0
        for count in scheduled:
            offsets.append(running)
            running += count

        feedback_rows: list[int] = []
        feedback_codes: list[torch.Tensor] = []
        for row, engine_id in enumerate(self._step_engine_ids):
            # Both members of a guided pair are fed the same frame, so state
            # is keyed by pair rather than by request.
            pair_id = self._pair_of_engine_id.get(engine_id)
            state = self._states.get(pair_id) if pair_id else None
            if state is None or state.last_codes is None:
                continue
            # A decode row advances by exactly one token past a complete
            # prompt. A row still working through a chunked prefill keeps its
            # real prompt embeddings.
            if scheduled[row] != 1 or computed[row] < state.prompt_tokens:
                continue
            feedback_rows.append(offsets[row])
            feedback_codes.append(state.last_codes)

        self._write_frame_feedback(feedback_rows, feedback_codes, input_ids.device)
        self._bind_rows(computed, scheduled)
        return input_ids, positions

    def _write_frame_feedback(
        self,
        feedback_rows: list[int],
        feedback_codes: list[torch.Tensor],
        device: torch.device,
    ) -> None:
        """Publish this step's frame feedback for the forward to consume.

        Two forms are written, because two forwards can consume it. The slab
        is what a captured decode graph reads: fixed address, fixed shape,
        selected row by row with a flag. The loose index and code tensors are
        the eager fallback for a batch wider than the slab, which is only ever
        a prefill batch and therefore never a replayed one.
        """
        if self._feedback_stage is None:
            self._feedback_stage = PinnedIndexStage(1, self._max_rows, device)

        active = self.frame_feedback_active
        if not feedback_rows:
            self._feedback_positions = None
            self._feedback_codes = None
            active.zero_()
            return

        codes = torch.cat(feedback_codes, dim=0).to(device)
        if max(feedback_rows) < self._max_rows:
            positions = self._feedback_stage.push([feedback_rows], len(feedback_rows))[0]
            self.frame_feedback_codes.index_copy_(0, positions, codes)
            active.zero_()
            active.index_fill_(0, positions, True)
        else:
            # Too wide for the slab. Nothing reads it on this step, but a stale
            # flag would corrupt the next graph replay if the batch shrank.
            positions = torch.tensor(feedback_rows, dtype=torch.long, device=device)
            active.zero_()
        self._feedback_positions = positions
        self._feedback_codes = codes

    def _bind_rows(self, computed: list[int], scheduled: list[int]) -> None:
        """Rebuild the row-order request metadata from the per-request cache.

        ``forward`` learns these values from the engine's own kwargs, but its
        Python body is skipped whenever the step is replayed from a CUDA graph,
        so the binding is re-established here. A row whose request has not been
        seen by ``forward`` yet leaves the binding incomplete; that only
        happens on the step a request is admitted, which is a prefill step, and
        prefill is never replayed, so ``forward`` fills it in before the
        sampler runs.
        """
        constants = [self._row_constants.get(engine_id) for engine_id in self._step_engine_ids]
        if not constants or any(entry is None for entry in constants):
            self._row_binding_complete = False
            return
        self._row_request_ids = [entry.request_id for entry in constants]
        self._row_roles = [entry.role for entry in constants]
        self._row_pair_ids = [entry.pair_id for entry in constants]
        self._row_seeds = [entry.seed for entry in constants]
        self._row_max_frames = [entry.max_frames for entry in constants]

        # A row is eligible to draw a frame once it has computed its whole
        # prompt. Mirrors the engine's own rule; the prompt length is taken
        # from the pair's state, which exists for every row whose request has
        # already produced a logit.
        eligible: list[bool] = []
        for row, entry in enumerate(constants):
            state = self._states.get(entry.pair_id)
            if state is None:
                self._row_binding_complete = False
                return
            eligible.append(computed[row] + scheduled[row] >= state.prompt_tokens)
        self._row_sample_eligible = eligible
        self._row_binding_complete = True

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del intermediate_tensors
        self._capture_row_metadata(kwargs)

        if inputs_embeds is None:
            hidden_states = self.model.embed_tokens(input_ids)
            hidden_states = self._apply_frame_feedback(hidden_states)
        else:
            hidden_states = inputs_embeds

        residual: torch.Tensor | None = None
        for layer in self.model.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        norm_out = self.model.norm(hidden_states, residual)
        if isinstance(norm_out, tuple):
            norm_out = norm_out[0]
        return norm_out

    def _capture_row_metadata(self, kwargs: dict[str, Any]) -> None:
        """Read this step's per-row CFG role, pair id and seed.

        Row order here matches ``prepare_runner_inputs``: both are built from
        ``input_batch.req_ids``. The request ids resolved there are preferred,
        and the info dicts are only a fallback for paths that do not route
        through that hook.
        """
        if self._row_binding_complete:
            # Every row was resolved from the per-request cache before the
            # forward, so there is nothing here the sampler still needs. Bail
            # out rather than rebuild it: this runs on every eager decode step.
            return
        infos = kwargs.get("model_intermediate_buffer")
        if infos is None:
            infos = kwargs.get("runtime_additional_information")
        extra_args = kwargs.get("sampling_extra_args")
        eligible = kwargs.get("request_sample_eligible")
        self._row_sample_eligible = [bool(flag) for flag in eligible] if eligible else []
        if infos is None and extra_args is None:
            # Warm-up dummy run: no omni kwargs are supplied.
            self._row_pair_ids = []
            return

        infos = infos or []
        extra_args = extra_args or []
        rows = max(len(infos), len(extra_args), len(self._step_engine_ids))
        request_ids: list[str] = []
        roles: list[str] = []
        pair_ids: list[str] = []
        seeds: list[int] = []
        max_frames: list[int] = []
        for index in range(rows):
            info = infos[index] if index < len(infos) else {}
            info = info if isinstance(info, dict) else {}
            extra = extra_args[index] if index < len(extra_args) else {}
            extra = extra if isinstance(extra, dict) else {}
            # The user-facing id. The engine renames a request to an internal
            # UUID on admission, so this is the only id that both the info
            # dicts and postprocess agree on, and the only one that carries
            # the companion suffix a pair is recognized by.
            request_id = _request_key(info.get("global_request_id")) or f"row-{index}"
            pair_id = str(extra.get("cfg_pair_id") or _pair_id_from_request(request_id))
            request_ids.append(request_id)
            roles.append(str(extra.get("cfg_role") or _role_from_request(request_id)))
            pair_ids.append(pair_id)
            seeds.append(int(extra.get("tts_local_seed") or 0))
            max_frames.append(_first_int(info.get("max_audio_frames"), DEFAULT_MAX_AUDIO_FRAMES))
            # Remember which pair each engine-side row belongs to, so the next
            # step's feedback can be gathered before the info dicts are seen,
            # and cache the whole row constant so a graph-replayed step can be
            # bound without the info dicts at all.
            if index < len(self._step_engine_ids):
                engine_id = self._step_engine_ids[index]
                self._pair_of_engine_id[engine_id] = pair_id
                self._row_constants[engine_id] = _RowConstants(
                    request_id=request_id,
                    role=roles[index],
                    pair_id=pair_id,
                    seed=seeds[index],
                    max_frames=max_frames[index],
                )
        self._row_request_ids = request_ids
        self._row_roles = roles
        self._row_pair_ids = pair_ids
        self._row_seeds = seeds
        self._row_max_frames = max_frames

    def _apply_frame_feedback(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Replace each decoding row's token embedding with its frame embedding.

        The rows to replace and the codes to write were resolved in
        ``prepare_runner_inputs`` against this step's own batch, so a mixed
        prefill and decode step touches only the decoding rows and a prompt
        token is never overwritten.

        Which of the two forms is read depends only on the row count, which is
        a constant inside a captured graph: a decode batch fits the reserved
        slab and is selected with ``torch.where`` over fixed buffers, while a
        prefill batch wider than the slab falls back to a variable-length
        ``index_copy``. Prefill is never replayed, so the fallback never has to
        be graph-safe.
        """
        rows = hidden_states.shape[0]
        if rows <= self._max_rows:
            frame_embeds = self.embed_frames(self.frame_feedback_codes[:rows]).to(dtype=hidden_states.dtype)
            return torch.where(
                self.frame_feedback_active[:rows].unsqueeze(1),
                frame_embeds,
                hidden_states,
            )
        positions = self._feedback_positions
        codes = self._feedback_codes
        if positions is None or codes is None:
            return hidden_states
        frame_embeds = self.embed_frames(codes).to(dtype=hidden_states.dtype)
        return hidden_states.index_copy(0, positions, frame_embeds)

    def embed_frames(self, codes: torch.Tensor) -> torch.Tensor:
        """Compose one conditioning embedding per row from all eight codebooks.

        Args:
            codes: ``[rows, 8]`` codebook indices, ``c0`` zero-based.

        Returns:
            ``[rows, hidden]`` in the embedding table's dtype.
        """
        c0 = self.model.embed_tokens(codes[:, 0] + AUDIO_CODE_OFFSET)
        extra = self.audio_embeddings(codes[:, 1:] + self.audio_embedding_offsets)
        return (c0 + extra.sum(dim=1).to(c0.dtype)) * self.frame_embedding_scale

    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata: Any = None) -> torch.Tensor:
        # sample() has no other route to the backbone hidden state.
        self._last_logits_hidden = hidden_states
        return self.logits_processor(self.lm_head, hidden_states)

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    # ----------------------------------------------------------------- sample

    def sample(self, logits: torch.Tensor, sampling_metadata: Any) -> SamplerOutput | None:
        """Draw one frame per conditioned request and feed it back to both rows."""
        del sampling_metadata
        hidden = self._last_logits_hidden
        self._last_logits_hidden = None
        if hidden is None or not self._row_pair_ids:
            return None

        rows = int(logits.shape[0])
        pairs = self._resolve_pairs(rows)
        if not pairs:
            # Nothing to draw this step. Emit a benign audio code so every row
            # stays alive until its partner is admitted; returning None would
            # hand the rows to the stock sampler, whose token would be read as
            # a real code.
            held = torch.full((rows, 1), _HOLD_TOKEN, dtype=torch.long, device=logits.device)
            return SamplerOutput(sampled_token_ids=held, logprobs_tensors=None)

        states = [self._state_for(self._row_pair_ids[pair[0]], pair[0]) for pair in pairs]
        cond_rows, uncond_rows, seeds, positions = self._stage_step_vectors(
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
            [state.seed for state in states],
            [state.advance_position() for state in states],
            logits.device,
        )

        narrowed = logits.index_select(1, self.c0_logit_ids).float()
        c0_logits = apply_c0_cfg(narrowed.index_select(0, cond_rows), narrowed.index_select(0, uncond_rows))
        drawn = sample_topk_seeded(c0_logits, seeds, positions)
        stopped = drawn == 0
        c0 = (drawn - 1).clamp_min(0)

        codes, depth_hidden = self._depth_decode(
            hidden.index_select(0, cond_rows),
            hidden.index_select(0, uncond_rows),
            c0,
            seeds,
            positions,
        )
        frame_hidden = torch.cat((hidden.index_select(0, cond_rows), depth_hidden), dim=-1)

        stopped_flags = stopped.tolist()
        drain_rows: list[int] = []
        for index, (cond_row, uncond_row) in enumerate(pairs):
            state = states[index]
            if state.finished:
                # The drain step. The terminal span written last step is on
                # its way out in this step's payload, so the pair can stop now.
                state.pending_stop = False
                drain_rows.extend((index, cond_row, uncond_row))
                continue
            if stopped_flags[index]:
                state.finished = True
                state.pending_stop = True
                state.finish_reason = "stop"
                state.last_codes = None
                continue
            # Both rows of the pair are fed the same frame next step; the
            # codes live on the request, not on a row, because the row this
            # request occupies may change before then.
            state.last_codes = codes[index : index + 1]
            state.frames.append(frame_hidden[index : index + 1])
            state.codes.append(codes[index : index + 1])
            state.generated_frames += 1
            if state.generated_frames >= state.max_frames:
                state.finished = True
                state.pending_stop = True
                state.finish_reason = "length"
            del cond_row, uncond_row

        # Rows that resolved into no pair keep this value. A hold token rather
        # than id 0, which is an arbitrary text token: a row waiting for its
        # partner should wait, not emit something the vocabulary reads as
        # prose. It is never fed back as audio, because feedback is gated on
        # the pair's own state.
        token_ids = torch.full((rows, 1), _HOLD_TOKEN, dtype=torch.long, device=logits.device)
        sampled = self.c0_logit_ids[drawn]
        token_ids[cond_rows, 0] = sampled
        token_ids[uncond_rows, 0] = sampled

        # The stop token is deferred by exactly one step. A pair that ends this
        # step must stay alive so that the terminal span ``postprocess`` is
        # about to write can ride out on the next step's payload; a pair that
        # already published its tail stops now.
        #
        # This covers the two ways a song ends on its own: the audio-end token,
        # and the request's frame budget, which reaches the model as
        # ``max_audio_frames``. It does not cover the engine ending a request
        # underneath the model, because the payload for a step is fixed before
        # the model is told. Hitting ``max_model_len`` mid-song therefore drops
        # the final window. In practice the frame budget is reached first; a
        # request can only hit the context limit with a prompt long enough that
        # the two limits collide, which the deploy config documents.
        for index, (cond_row, uncond_row) in enumerate(pairs):
            state = states[index]
            if state.pending_stop:
                token_ids[cond_row, 0] = _HOLD_TOKEN
                token_ids[uncond_row, 0] = _HOLD_TOKEN
        for _, cond_row, uncond_row in zip(drain_rows[0::3], drain_rows[1::3], drain_rows[2::3], strict=True):
            token_ids[cond_row, 0] = _STOP_TOKEN
            token_ids[uncond_row, 0] = _STOP_TOKEN
        return SamplerOutput(sampled_token_ids=token_ids, logprobs_tensors=None)

    def _stage_step_vectors(
        self,
        cond_rows: list[int],
        uncond_rows: list[int],
        seeds: list[int],
        positions: list[int],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Move this step's four small int64 vectors to the device in one copy.

        The copy is asynchronous, so the four returned views are only valid
        once the stream reaches them. Every consumer is a kernel on that same
        stream, which is what makes this safe.
        """
        if self._sample_stage is None:
            self._sample_stage = PinnedIndexStage(4, self._max_rows, device)
        block = self._sample_stage.push([cond_rows, uncond_rows, seeds, positions], len(cond_rows))
        return block[0], block[1], block[2], block[3]

    def _resolve_pairs(self, rows: int) -> list[tuple[int, int]]:
        """Map ``cfg_pair_id`` to its ``(cond_row, uncond_row)`` batch indices.

        Guidance is mandatory for this checkpoint, so an incomplete pair is a
        hard error rather than something to route around. Skipping the step
        would leave the conditioned request producing no frames and the client
        waiting forever, which is the failure shape this module has been bitten
        by before; raising surfaces it to the caller instead.

        Raises:
            RuntimeError: If any pair is missing a member in this step.
        """
        by_pair: dict[str, dict[str, int]] = {}
        limit = min(rows, len(self._row_pair_ids))
        for row in range(limit):
            # A row still finishing a chunked prefill is handed to the sampler
            # but its token is discarded, so drawing a frame for it would add
            # a bogus frame and desync the seeded RNG.
            if row < len(self._row_sample_eligible) and not self._row_sample_eligible[row]:
                continue
            by_pair.setdefault(self._row_pair_ids[row], {})[self._row_roles[row]] = row

        pairs: list[tuple[int, int]] = []
        incomplete: list[str] = []
        for pair_id, roles in by_pair.items():
            if _COND in roles and _UNCOND in roles:
                pairs.append((roles[_COND], roles[_UNCOND]))
            else:
                incomplete.append(f"{pair_id}({'+'.join(sorted(roles))})")
        if incomplete:
            # A member can be absent for a step while its partner is still
            # being admitted. Holding the present row costs one wasted decode
            # step; raising would abort the whole engine step for every other
            # request in the batch, so the lone row simply does not sample this
            # step and waits. A pair that never converges is bounded by the
            # request's own token budget rather than hanging forever.
            self._incomplete_steps += 1
            if self._incomplete_steps in (1, 64, 512):
                logger.warning(
                    "MiniMax Music 3 waiting on incomplete CFG pair(s) %s "
                    "(step %d). If this does not resolve, the companion was "
                    "never enqueued: check that the stage declares "
                    "extra_args.cfg_role in its deploy config.",
                    incomplete[:4],
                    self._incomplete_steps,
                )
        return pairs

    def _depth_decode(
        self,
        cond_hidden: torch.Tensor,
        uncond_hidden: torch.Tensor,
        c0: torch.Tensor,
        seeds: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Walk ``c1..c7``, replaying the loop from a CUDA graph where possible.

        Seven four-layer forwards over at most eight positions is far too
        little work per kernel to cover its own launch, so the loop is recorded
        once per pair count and replayed after that. The cache falls back to
        the eager loop for any shape it has not recorded or could not record.
        """
        return self._depth_graphs.run(cond_hidden, uncond_hidden, c0, seeds, positions)

    def _depth_decode_eager(
        self,
        cond_hidden: torch.Tensor,
        uncond_hidden: torch.Tensor,
        c0: torch.Tensor,
        seeds: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Walk ``c1..c7``, returning the codes and the stacked depth hiddens.

        Both branches run so the depth heads can be guided the same way the
        ``c0`` head is; the drawn code is shared, so the two branches stay
        aligned.
        """
        decoder = self.rvq_decoder
        pair_hidden = torch.cat((cond_hidden, uncond_hidden), dim=0)
        c0_paired = c0.repeat(2)
        c0_embed = self.model.embed_tokens(c0_paired + AUDIO_CODE_OFFSET)
        seq = [
            decoder.projection(pair_hidden).unsqueeze(1),
            decoder.projection(c0_embed).unsqueeze(1),
        ]
        n_pairs = int(c0.shape[0])
        codes = [c0]
        depth_parts: list[torch.Tensor] = []
        # Every residual draw's noise comes from the same hash of
        # ``(seed, position + codebook, candidate)``, so all seven are produced
        # in one batch here instead of re-running the twenty-odd integer ops
        # per codebook inside the loop. The arithmetic is elementwise, so the
        # values are the same ones the per-codebook calls produced.
        depth_gumbel = gumbel_noise(
            seeds,
            positions.unsqueeze(1) + self.depth_draw_offsets,
            decoder.audio_vocab_size,
        )
        for index in range(1, NUM_CODEBOOKS):
            out = decoder(torch.cat(seq, dim=1))[:, -1]
            depth_parts.append(out[:n_pairs].detach())
            head_logits = decoder.audio_heads[index - 1](out).float()
            guided = apply_depth_cfg(head_logits[:n_pairs], head_logits[n_pairs:])
            code = draw_from_gumbel(guided, depth_gumbel[:, index - 1])
            codes.append(code)
            if index < NUM_CODEBOOKS - 1:
                embedding = self.audio_embeddings(code.repeat(2) + (index - 1) * AUDIO_VOCAB_SIZE)
                seq.append(decoder.projection(embedding).unsqueeze(1))
        return torch.stack(codes, dim=1), torch.cat(depth_parts, dim=-1)

    def _state_for(self, pair_id: str, row: int) -> RequestARState:
        """Fetch or create the state for a guided pair.

        Created on the step the pair first produces a logit, which is the step
        its prefill completes, so the recorded prompt length is the real one.
        """
        state = self._states.get(pair_id)
        if state is None:
            seed = self._row_seeds[row] if row < len(self._row_seeds) else 0
            state = RequestARState(
                seed=seed,
                max_frames=self._row_max_frames[row] if row < len(self._row_max_frames) else DEFAULT_MAX_AUDIO_FRAMES,
                prompt_tokens=self._row_prompt_tokens[row] if row < len(self._row_prompt_tokens) else 0,
            )
            self._states[pair_id] = state
            logger.info(
                "MiniMax Music 3 AR start pair=%s seed=%d max_frames=%d prompt_tokens=%d",
                pair_id,
                state.seed,
                state.max_frames,
                state.prompt_tokens,
            )
        return state

    # ------------------------------------------------------------ emit / free

    def postprocess(
        self,
        hidden_states_slice: torch.Tensor,
        multimodal_outputs: Any = None,
        **req_infos: Any,
    ) -> dict[str, Any]:
        """Publish this request's next span of conditioning frames.

        Mid-stream this is one window, released a frame late so that a window
        handed downstream is provably not the last. On the final step it is
        instead a single contiguous span covering every window not yet
        emitted, because one step produces exactly one payload and the tail
        cannot be split across two.
        """
        del hidden_states_slice, multimodal_outputs
        request_id = _request_key(req_infos.get("global_request_id"))
        # The pair's two rows share one song, so only the conditioned row
        # publishes it. The companion would otherwise emit a duplicate span
        # and the acoustic stage would decode the song twice.
        if _role_from_request(request_id) == _UNCOND:
            return {}
        state = self._states.get(_pair_id_from_request(request_id))
        if state is None:
            return {}
        if state.finished:
            return self._terminal_payload(state)
        window = state.ready_window()
        if window is None:
            return {}
        chunk = state.take_window(window, is_final=False)
        return self._span_payload(state, chunk, start_frame=window.start, is_final=False)

    def _terminal_payload(self, state: RequestARState) -> dict[str, Any]:
        """Emit every remaining frame as one span, once."""
        start = state.emitted_windows * AR_CHUNK_HOP_FRAMES
        end = state.frames.end_frame
        if end <= start:
            return {}
        chunk = state.frames.slice(start, end)
        state.emitted_windows = len(chunk_windows(state.generated_frames))
        state.frames.clear()
        return self._span_payload(state, chunk, start_frame=start, is_final=True)

    def _span_payload(
        self,
        state: RequestARState,
        chunk: torch.Tensor,
        *,
        start_frame: int,
        is_final: bool,
    ) -> dict[str, Any]:
        """Build the cross-stage payload for one contiguous frame span.

        ``num_processed_tokens`` is the span's first global AR frame index, so
        the acoustic stage can re-derive window boundaries and therefore which
        window is first and which is last.
        """
        frames = int(chunk.shape[0])
        codes = state.take_codes(start_frame, start_frame + frames)
        payload: dict[str, Any] = {
            "latent": chunk.contiguous(),
            "meta": {
                "num_processed_tokens": start_frame,
                "codec_chunk_frames": frames,
                "last_chunk": is_final,
                "audio_seed": state.seed,
            },
        }
        if codes is not None:
            payload["codes"] = {"audio": codes}
        return payload

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        infos = kwargs.get("model_intermediate_buffer")
        if infos is None:
            infos = kwargs.get("runtime_additional_information")
        if not infos:
            return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs=None)

        latents: list[torch.Tensor] = []
        codes: list[torch.Tensor] = []
        metas: list[dict[str, Any]] = []
        any_payload = False
        for info in infos:
            info = info if isinstance(info, dict) else {}
            latent = info.get("latent")
            if isinstance(latent, torch.Tensor) and latent.numel():
                latents.append(latent)
                sub = info.get("codes")
                codes.append(sub.get("audio") if isinstance(sub, dict) else torch.empty(0))
                meta = info.get("meta")
                metas.append(dict(meta) if isinstance(meta, dict) else {})
                any_payload = True
                # The runner merges postprocess output into this buffer and
                # never clears it, so a window left behind would be re-emitted
                # on every later step and the acoustic stage would decode the
                # same conditioning again and again. Consume it here.
                info.pop("latent", None)
                info.pop("codes", None)
                info.pop("meta", None)
            else:
                latents.append(torch.empty(0))
                codes.append(torch.empty(0))
                metas.append({})
        if not any_payload:
            return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs=None)
        payload: dict[str, Any] = {"latent": latents}
        if any(isinstance(entry, torch.Tensor) and entry.numel() for entry in codes):
            payload["codes"] = {"audio": codes}
        # The span's start frame and terminal flag ride along with it: without
        # them the acoustic stage cannot tell which window it is looking at,
        # and therefore cannot crop it correctly.
        #
        # Every field is a list with one entry per batch row, in the same order
        # as ``latent``. The payload is partitioned per request by indexing
        # list values, so a scalar here would be broadcast to the whole batch
        # and every request would decode with the last request's seed, window
        # offset and terminal flag.
        meta_keys = {key for meta in metas for key in meta}
        if meta_keys:
            payload["meta"] = {key: [meta.get(key) for meta in metas] for key in sorted(meta_keys)}
        return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs=payload)

    def on_requests_finished(self, finished_req_ids: Iterable[str]) -> None:
        """Release per-request state once the engine has retired the request.

        The key must be normalized the same way it was when the state was
        created, or every finished song leaks its frame buffer.
        """
        for engine_id in finished_req_ids:
            # These are engine-side ids, so translate through the mapping the
            # forward pass built. A pair's state is released when its second
            # member retires.
            self._row_constants.pop(str(engine_id), None)
            pair_id = self._pair_of_engine_id.pop(str(engine_id), None)
            if pair_id is None or any(pair_id == value for value in self._pair_of_engine_id.values()):
                continue
            request_id = pair_id
            state = self._states.pop(pair_id, None)
            if state is not None:
                logger.info(
                    "MiniMax Music 3 AR done request=%s frames=%d reason=%s peak_buffer=%d",
                    request_id,
                    state.generated_frames,
                    state.finish_reason,
                    state.frames.peak_frames,
                )
                state.release()

    # --------------------------------------------------------------- weights

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the Qwen3 backbone, then the audio modules from their component.

        The backbone comes from the stage's own checkpoint folder and loads
        through vLLM. The audio embedding table and the depth decoder live in
        a separate component of the repo and are loaded explicitly.
        """
        loader = AutoWeightsLoader(self, skip_prefixes=["audio_embeddings.", "rvq_decoder."])
        loaded = loader.load_weights(weights)
        loaded |= load_depth_decoder_weights(
            self.vllm_config,
            audio_embeddings=self.audio_embeddings,
            rvq_decoder=self.rvq_decoder,
        )
        return loaded
