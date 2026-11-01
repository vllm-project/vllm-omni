# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from nvidia/Nemotron-Labs-Audex-2B (Apache-2.0):
#   inference_scripts_vllm/audiogen_scripts/tta_rvq_logits_processor.py
#   inference_scripts_vllm/audiogen_scripts/run_audio_gen_vllm_rvq_logit_mask.py
# Divergences from the official files are commented at the divergence site.
"""Audex text-to-audio (TTA) token space: RVQ phase masking and helpers.

TTA output is an interleaved 4-codebook RVQ stream over ``<audiocodec_N>``
tokens: generated position ``p`` (counted from ``<audiogen_start>``) must
come from codebook ``p % 4``, whose codec-id range is
``[phase * 1024, (phase + 1) * 1024)``. The tokenizer defines 8192
``<audiocodec_*>`` symbols; only codec ids 0..4095 (the 4 quantizers XCodec1
decodes) are generated — the upper half stays disallowed.

Unguided/unmasked sampling emits phase-invalid sequences that XCodec1 cannot
decode, so :class:`TTARVQPhaseMaskLogitsProcessor` is required for validity,
driven per request by ``SamplingParams.extra_args["tta_rvq"]``:

    {
        "phase_token_ids": [[...1024 ids...] x 4],
        "start_tid": <audiogen_start id>,
        "end_tid": <audiogen_end id>,
        "codec_cap": <max codec tokens or None>,
        "start_in_prompt": <bool>,
    }
"""

from __future__ import annotations

import os
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.sample.logits_processor import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

logger = init_logger(__name__)

NEG_INF = float("-inf")
XCODEC1_NUM_CODEBOOKS = 4
XCODEC1_CODEBOOK_SIZE = 1024

# Pinned by a unit test against the real checkpoint tokenizer.
AUDEX_AUDIOGEN_START_TOKEN_ID = 131073
AUDEX_AUDIOGEN_END_TOKEN_ID = 131074
AUDEX_AUDIOCODEC_TOKEN_OFFSET = 196613
AUDEX_AUDIOCODEC_VOCAB_SIZE = 8192


def build_tta_phase_token_ids(tokenizer: Any) -> tuple[list[list[int]], int, int]:
    """Group tokenizer ids of ``<audiocodec_N>`` into the 4 RVQ phases.

    Phase ``p`` collects the tokenizer ids for codec ids
    ``[p*1024, (p+1)*1024)``. Fails loudly on missing markers or an
    incomplete codec vocab.

    Returns ``(phase_token_ids, audiogen_start_tid, audiogen_end_tid)``.
    """
    vocab = tokenizer.get_vocab()
    start_tid = vocab.get("<audiogen_start>")
    end_tid = vocab.get("<audiogen_end>")
    if start_tid is None or end_tid is None:
        raise ValueError(f"Missing marker token ids: audiogen_start={start_tid}, audiogen_end={end_tid}")

    phase_token_ids: list[list[int]] = []
    for phase in range(XCODEC1_NUM_CODEBOOKS):
        ids: list[int] = []
        missing: list[int] = []
        for codec_id in range(phase * XCODEC1_CODEBOOK_SIZE, (phase + 1) * XCODEC1_CODEBOOK_SIZE):
            tok = vocab.get(f"<audiocodec_{codec_id}>")
            if tok is None:
                missing.append(codec_id)
            else:
                ids.append(tok)
        if missing:
            raise ValueError(
                f"Incomplete codec vocab for RVQ phase {phase}: missing {len(missing)} codec ids (e.g. {missing[:5]})"
            )
        phase_token_ids.append(ids)

    return phase_token_ids, int(start_tid), int(end_tid)


def validate_rvq_phase(
    codec_ids: list[int],
    codebook_size: int = XCODEC1_CODEBOOK_SIZE,
    num_codebooks: int = XCODEC1_NUM_CODEBOOKS,
) -> dict[str, Any]:
    """Check that ``codec_ids[i]`` falls in RVQ phase ``i % 4``'s codec range."""
    mismatch_count = 0
    first_mismatch = None
    for i, codec_id in enumerate(codec_ids):
        expected = i % num_codebooks
        lo = expected * codebook_size
        if not (lo <= codec_id < lo + codebook_size):
            mismatch_count += 1
            if first_mismatch is None:
                first_mismatch = [i, int(codec_id), int(codec_id // codebook_size), int(expected)]
    return {
        "phase_valid": mismatch_count == 0,
        "mismatch_count": mismatch_count,
        "first_mismatch": first_mismatch,
        "codec_count": len(codec_ids),
    }


class TTARVQPhaseMaskLogitsProcessor(LogitsProcessor):
    """Mask logits to the RVQ-phase-valid token set for Audex TTA requests.

    Per batch row:
      ``_req[idx]``           -> {"start_tid", "end_tid", "codec_cap", "start_in_prompt"}
      ``_output_tokens[idx]`` -> live reference to the request's output ids
      ``_start_pos[idx]``     -> cached index of <audiogen_start> once seen;
                                 -1 means the prompt already ended with it.

    The disallow masks are vocab-sized boolean tensors built once on the
    first ``apply`` (using the real ``logits.shape[-1]`` so vocab padding is
    also forbidden).
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool) -> None:
        self.device = device
        self._req: dict[int, dict[str, Any]] = {}
        self._output_tokens: dict[int, list[int]] = {}
        self._start_pos: dict[int, int] = {}

        self._phase_token_ids: list[list[int]] | None = None
        self._mask_end_tid: int | None = None

        self._disallow_phase: torch.Tensor | None = None
        self._disallow_phase0_end: torch.Tensor | None = None
        self._disallow_except_end: torch.Tensor | None = None

    def is_argmax_invariant(self) -> bool:
        return False

    def _reset(self) -> None:
        self._req.clear()
        self._output_tokens.clear()
        self._start_pos.clear()

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if batch_update is None:
            return

        for idx in batch_update.removed:
            self._req.pop(idx, None)
            self._output_tokens.pop(idx, None)
            self._start_pos.pop(idx, None)

        if not self._req and batch_update.added:
            self._reset()

        for idx, params, _, output_token_ids in batch_update.added:
            extra_args = params.extra_args if params else None
            tta = extra_args.get("tta_rvq") if extra_args else None
            if tta:
                self._req[idx] = {
                    "start_tid": tta["start_tid"],
                    "end_tid": tta["end_tid"],
                    "codec_cap": tta.get("codec_cap"),
                    "start_in_prompt": bool(tta.get("start_in_prompt", False)),
                }
                self._output_tokens[idx] = output_token_ids
                if self._req[idx]["start_in_prompt"]:
                    self._start_pos[idx] = -1
                else:
                    self._start_pos.pop(idx, None)
                if self._phase_token_ids is None:
                    self._phase_token_ids = tta["phase_token_ids"]
                    self._mask_end_tid = tta["end_tid"]
            else:
                self._req.pop(idx, None)
                self._output_tokens.pop(idx, None)
                self._start_pos.pop(idx, None)

        if self._req:
            for src, dst, direction in batch_update.moved:
                src_req = self._req.pop(src, None)
                dst_req = self._req.pop(dst, None)
                src_tokens = self._output_tokens.pop(src, None)
                dst_tokens = self._output_tokens.pop(dst, None)
                src_pos = self._start_pos.pop(src, None)
                dst_pos = self._start_pos.pop(dst, None)
                if src_req is not None:
                    self._req[dst] = src_req
                if src_tokens is not None:
                    self._output_tokens[dst] = src_tokens
                if src_pos is not None:
                    self._start_pos[dst] = src_pos
                if direction == MoveDirectionality.SWAP:
                    if dst_req is not None:
                        self._req[src] = dst_req
                    if dst_tokens is not None:
                        self._output_tokens[src] = dst_tokens
                    if dst_pos is not None:
                        self._start_pos[src] = dst_pos

    def _build_masks(self, vocab_size: int) -> None:
        if self._phase_token_ids is None or self._mask_end_tid is None:
            raise RuntimeError(
                "TTARVQPhaseMaskLogitsProcessor: phase token ids missing; "
                "extra_args['tta_rvq'] was not provided for a TTA request."
            )
        if len(self._phase_token_ids) != XCODEC1_NUM_CODEBOOKS:
            raise RuntimeError(
                f"TTARVQPhaseMaskLogitsProcessor: expected {XCODEC1_NUM_CODEBOOKS} phase id "
                f"lists, got {len(self._phase_token_ids)}."
            )

        allow = torch.zeros((XCODEC1_NUM_CODEBOOKS, vocab_size), dtype=torch.bool, device=self.device)
        for phase, ids in enumerate(self._phase_token_ids):
            if not ids:
                raise RuntimeError(f"TTARVQPhaseMaskLogitsProcessor: empty token id list for RVQ phase {phase}.")
            idx = torch.as_tensor(ids, dtype=torch.long, device=self.device)
            if int(idx.max()) >= vocab_size or int(idx.min()) < 0:
                raise RuntimeError(
                    f"TTARVQPhaseMaskLogitsProcessor: phase {phase} token id out of range for vocab_size={vocab_size}."
                )
            allow[phase].index_fill_(0, idx, True)

        end_onehot = torch.zeros(vocab_size, dtype=torch.bool, device=self.device)
        if self._mask_end_tid < 0 or self._mask_end_tid >= vocab_size:
            raise RuntimeError(
                f"TTARVQPhaseMaskLogitsProcessor: end token id out of range "
                f"for vocab_size={vocab_size}: {self._mask_end_tid}."
            )
        end_onehot[self._mask_end_tid] = True

        self._disallow_phase = ~allow
        self._disallow_phase0_end = ~(allow[0] | end_onehot)
        self._disallow_except_end = ~end_onehot
        logger.info(
            "TTARVQPhaseMaskLogitsProcessor: built masks vocab=%d end_tid=%d (pid=%d)",
            vocab_size,
            self._mask_end_tid,
            os.getpid(),
        )

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self._req:
            return logits

        if self._disallow_phase is None:
            self._build_masks(logits.shape[-1])

        num_rows = logits.shape[0]
        for idx, state in self._req.items():
            # Rows beyond this step's logits (pure-prefill / partially
            # scheduled steps) have nothing to mask; the persistent-batch
            # index re-enters range on the decode steps that sample.
            if idx >= num_rows:
                continue
            tokens = self._output_tokens.get(idx)
            if tokens is None:
                continue
            n = len(tokens)

            start_pos = self._start_pos.get(idx)
            if start_pos is None:
                start_tid = state["start_tid"]
                for i in range(n):
                    if tokens[i] == start_tid:
                        start_pos = i
                        self._start_pos[idx] = i
                        break
                if start_pos is None:
                    continue

            position = n - start_pos - 1
            if position < 0:
                continue

            cap = state["codec_cap"]
            if cap is not None and position >= cap:
                disallow = self._disallow_except_end
            else:
                phase = position & (XCODEC1_NUM_CODEBOOKS - 1)
                if phase == 0:
                    disallow = self._disallow_phase0_end if position > 0 else self._disallow_phase[0]
                else:
                    disallow = self._disallow_phase[phase]

            logits[idx].masked_fill_(disallow, NEG_INF)

        return logits
