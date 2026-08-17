# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Moshi-free PersonaPlex lockstep engine (native temporal + depformer + Mimi).

This is the real-time duplex backend built entirely from vllm-omni-native
components: :class:`PersonaPlexTemporalStreaming` (ring-KV Helium),
:class:`PersonaPlexInputEmbeddings`, :class:`PersonaPlexDepformer` and
:class:`PersonaPlexMimiCodec`. It reimplements the reference lockstep semantics
(staging ring, per-codebook delays, warmup gates, hybrid system-prompt prefill,
per-slot elastic recycle) as first-class policy code, so the external ``moshi``
package is no longer needed at runtime.

Lockstep contract per 80 ms tick (17 token rows: text, 8 agent codebooks, 8
user codebooks; delays ``[0, 1x8-ish from checkpoint]``):

1. Mimi-encode the user frame and write it (delayed) into the staging ring.
2. Read the model input at ``(o - 1) % CT``, embed, step the temporal.
3. Greedy text token; depformer expands the 16 audio rows (teacher-forced where
   the staging ring already holds provided values).
4. Write sampled rows at ``o % CT``; gather the de-delayed output frame; decode
   agent rows 1..8 through Mimi (rows still in warmup are sanitized to silence
   tokens: the LM's initial audio token id equals the codec cardinality and is
   out of range for the codebook tables).

Elastic recycle mirrors the verified design from the moshi-hosted path: the
temporal ring KV masks the recycled row (plus the one LM "sacrifice tick"
write), the staging row is wiped to initial tokens, and the system prompt
(voice embeddings, silence spacer, persona tokens, silence) is replayed per-row
while other slots keep conversing.
"""

from __future__ import annotations

import logging
import os
import tarfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from vllm_omni.experimental.fullduplex.personaplex.config import PersonaPlexConfig
from vllm_omni.experimental.fullduplex.personaplex.policy import (
    AUDIO_INITIAL_TOKEN,
    AUDIO_SILENCE_FRAME_CNT,
    NON_TEXT_TOKEN_IDS,
    SILENCE_TOKENS,
    SINE_TOKENS,
    TEXT_INITIAL_TOKEN,
    TEXT_VOCAB_SIZE,
    ZERO_TEXT_TOKEN,
    PrefillStep,
    wrap_with_system_tags,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class PersonaPlexEngine:
    """Frame-lockstep PersonaPlex on native components only (no moshi).

    One class serves both surfaces: the single-session ``FrameStepper`` protocol
    (``open_session`` / ``step``) and the batched slot API consumed by
    ``serving/batched.py`` (``open_batch`` / ``prefill_steps`` / ``recycle_slot``
    / ``voice_end`` / ``end_prefill`` / ``step_batch``).
    """

    def __init__(self, config: PersonaPlexConfig | None = None) -> None:
        self.config = config or PersonaPlexConfig()
        self.sample_rate = self.config.sample_rate
        self.frame_size = self.config.frame_size
        # Serializes the single-session FrameStepper surface (open_session/step)
        # across worker threads: a cancelled asyncio task cannot stop a running
        # engine call, so a racing open() must wait for the in-flight step.
        # The batched surface is serialized by BatchedSessionManager instead.
        self._fs_lock = threading.Lock()
        self._loaded = False
        self._opened = False
        self._tokenizer = None
        self._codec = None
        self._temporal = None
        self._emb = None
        self._dep = None
        self._voice_embeddings = None  # [T, 1, 1, H]
        self._voice_cache = None  # [1, K, CT]
        self._loaded_voice = None  # name/path of the currently loaded voice
        # lockstep state (built at open time)
        self._cache = None  # [B, K, CT] long
        self._provided = None  # [B, K, CT] bool
        self._initial = None  # [K] long
        self._delays = None  # [K] long (python list)
        self._phys = 0
        self._local = None  # [B] cpu long

    # -- loading --------------------------------------------------------------

    def load(self) -> PersonaPlexEngine:
        if self._loaded:
            return self
        import sentencepiece
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        from vllm_omni.model_executor.models.personaplex.configuration_personaplex import (
            PersonaPlexConfig as _PPModelConfig,
        )
        from vllm_omni.model_executor.models.personaplex.personaplex_depformer import (
            PersonaPlexDepformer,
        )
        from vllm_omni.model_executor.models.personaplex.personaplex_embeddings import (
            PersonaPlexInputEmbeddings,
        )
        from vllm_omni.model_executor.models.personaplex.personaplex_mimi import (
            PersonaPlexMimiCodec,
        )
        from vllm_omni.model_executor.models.personaplex.personaplex_temporal import (
            PersonaPlexTemporalStreaming,
        )

        cfg = self.config
        B = cfg.batch_size
        logger.info("PersonaPlex native: loading Mimi codec")
        self._codec = PersonaPlexMimiCodec(device=cfg.device)
        self._codec.streaming_init(B)

        tok_path = hf_hub_download(cfg.hf_repo, "tokenizer_spm_32k_3.model")
        self._tokenizer = sentencepiece.SentencePieceProcessor(tok_path)

        logger.info("PersonaPlex native: loading temporal + depformer + embeddings")
        sd = load_file(hf_hub_download(cfg.hf_repo, "model.safetensors"), device=cfg.device)
        dtype = torch.bfloat16
        self._temporal = PersonaPlexTemporalStreaming().to(cfg.device, dtype)
        self._temporal.load_weights(sd)
        self._temporal.eval()
        self._temporal.streaming_init(B)

        ppcfg = _PPModelConfig()
        self._emb = PersonaPlexInputEmbeddings(ppcfg).to(cfg.device, dtype)
        self._emb.load_weights(sd)
        self._emb.eval()
        self._dep = PersonaPlexDepformer(ppcfg.depformer_config, temporal_hidden_size=4096, text_card=32000).to(
            cfg.device, dtype
        )
        self._dep.load_weights(sd)
        self._dep.eval()
        del sd
        torch.accelerator.empty_cache()

        K = 17
        self._delays = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]  # text, agent cb0..7, user cb0..7
        self._initial = torch.tensor(
            [TEXT_INITIAL_TOKEN] + [AUDIO_INITIAL_TOKEN] * 16, dtype=torch.long, device=cfg.device
        )
        ct = max(self._delays) + 3
        self._cache = torch.full((B, K, ct), AUDIO_INITIAL_TOKEN, dtype=torch.long, device=cfg.device)
        self._provided = torch.zeros((B, K, ct), dtype=torch.bool, device=cfg.device)
        self._local = torch.zeros(B, dtype=torch.long)
        self._sine = torch.tensor(SINE_TOKENS, dtype=torch.long, device=cfg.device)
        self._silence = torch.tensor(SILENCE_TOKENS, dtype=torch.long, device=cfg.device)

        self._load_voice(cfg.voice_prompt)
        self._loaded = True
        return self

    def _load_voice(self, voice: str) -> None:
        import torch

        if voice == self._loaded_voice:
            return
        path = self._resolve_voice_prompt(voice)
        if not path.endswith(".pt"):
            raise ValueError(f"native stack needs a .pt voice-embedding bundle; got {path!r}")
        state = torch.load(path, map_location=self.config.device, weights_only=True)
        self._voice_embeddings = state["embeddings"].to(self.config.device)
        self._voice_cache = state["cache"].to(self.config.device)
        self._loaded_voice = voice

    def _resolve_voice_prompt(self, voice: str) -> str:
        from huggingface_hub import hf_hub_download

        if os.path.exists(voice):
            return voice
        tgz = Path(hf_hub_download(self.config.hf_repo, "voices.tgz"))
        vdir = tgz.parent / "voices"
        if not vdir.exists():
            with tarfile.open(tgz, "r:gz") as tar:
                tar.extractall(path=tgz.parent)  # noqa: S202 - trusted first-party bundle
        cand = vdir / voice
        if not cand.exists():
            raise FileNotFoundError(f"voice prompt {voice!r} not found in voices.tgz")
        return str(cand)

    # -- lockstep core ----------------------------------------------------------

    def _reset_all_streaming(self) -> None:
        self._codec.reset_streaming()
        self._temporal.reset_streaming()
        self._cache.fill_(AUDIO_INITIAL_TOKEN)
        self._cache[:, 0, :] = TEXT_INITIAL_TOKEN
        self._provided.fill_(False)
        self._phys = 0
        self._local.zero_()

    def _tick(
        self,
        user_codes,  # [B, 8] long or None (prefill rows overridden separately)
        forced_moshi=None,  # [B, 8] long (values used on force_mask rows)
        forced_text=None,  # [B] long
        force_mask=None,  # [B] bool (device)
        embed_override=None,  # ([B, 1, H], [B] bool rows)
    ):
        """One lockstep step. Returns (ran, out[B,17] | None)."""
        import torch

        B, K, CT = self._cache.shape
        o = self._phys
        lo = self._local

        if user_codes is not None:
            for q in range(8):
                k = 9 + q
                wp = (o + self._delays[k]) % CT
                self._cache[:, k, wp] = user_codes[:, q]
                self._provided[:, k, wp] = True
        if forced_moshi is not None and force_mask is not None:
            for q in range(8):
                k = 1 + q
                wp = (o + self._delays[k]) % CT
                self._cache[force_mask, k, wp] = forced_moshi[force_mask, q]
                self._provided[force_mask, k, wp] = True
        if forced_text is not None and force_mask is not None:
            wp = (o + self._delays[0]) % CT
            self._cache[force_mask, 0, wp] = forced_text[force_mask]
            self._provided[force_mask, 0, wp] = True

        for k, delay in enumerate(self._delays):
            warm = lo <= delay  # [B] cpu bool
            if bool(warm.any()):
                wd = warm.to(self._cache.device)
                pos = o % CT
                self._cache[:, k, pos] = torch.where(wd, self._initial[k].expand(B), self._cache[:, k, pos])
                self._provided[:, k, pos] = self._provided[:, k, pos] | wd

        if o == 0:
            # Genuine cold start: no prior physical frame to read as model input.
            self._cache[:, :, 0] = self._initial.view(1, K).expand(B, K)
            self._phys += 1
            self._local += 1
            return False, None

        mip = (o - 1) % CT
        tp = o % CT
        input_ = self._cache[:, :, mip]  # [B, K]
        target_ = self._cache[:, :, tp]
        provided_ = self._provided[:, :, tp]

        emb = self._emb(input_.unsqueeze(-1))  # [B, 1, H]
        if embed_override is not None:
            ov, rows = embed_override
            emb = torch.where(rows.view(B, 1, 1), ov.to(emb.dtype), emb)
        hidden, text_logits = self._temporal.step(emb)

        sampled_text = text_logits.float().view(B, -1).argmax(-1)  # greedy
        next_text = torch.where(provided_[:, 0], target_[:, 0], sampled_text)
        audio = self._dep(next_text, hidden, audio_tokens=target_[:, 1:], audio_provided=provided_[:, 1:])  # [B, 16]

        self._provided[:, :, mip] = False
        self._cache[:, 0, tp] = torch.where(~provided_[:, 0], sampled_text, self._cache[:, 0, tp])
        self._cache[:, 1:, tp] = torch.where(~provided_[:, 1:], audio, self._cache[:, 1:, tp])

        max_delay = max(self._delays)
        valid = lo > max_delay
        if not bool(valid.any()):
            self._phys += 1
            self._local += 1
            return True, None

        idx = torch.tensor(
            [(self._phys - max_delay + d) % CT for d in self._delays],
            device=self._cache.device,
            dtype=torch.long,
        )
        out = self._cache.gather(2, idx.view(1, K, 1).expand(B, K, 1)).squeeze(-1)  # [B, K]
        self._last_valid = valid.clone()
        self._phys += 1
        self._local += 1
        return True, out

    def _boot_prefill(self, persona: str | None) -> None:
        """Batch-wide hybrid system prompt: voice -> cache restore -> silence ->
        persona -> silence (all rows identical)."""
        import torch

        B = self.config.batch_size
        dev = self.config.device
        dummy_user = self._initial[9:].view(1, 8).expand(B, 8)
        dummy_moshi = self._initial[1:9].view(1, 8).expand(B, 8)
        zero_text = torch.full((B,), ZERO_TEXT_TOKEN, dtype=torch.long, device=dev)
        all_rows = torch.ones(B, dtype=torch.bool, device=dev)

        for e in self._voice_embeddings:
            ov = e.view(1, 1, -1).expand(B, 1, -1)
            while True:
                ran, _ = self._tick(dummy_user, dummy_moshi, zero_text, all_rows, embed_override=(ov, all_rows))
                if ran:
                    break
        self._cache.copy_(self._voice_cache.expand_as(self._cache))

        sine = self._sine.view(1, 8).expand(B, 8)
        silence = self._silence.view(1, 8).expand(B, 8)
        text = persona if persona is not None else self.config.persona
        toks = self._tokenizer.encode(wrap_with_system_tags(text)) if text else []
        for phase_toks in (
            [ZERO_TEXT_TOKEN] * AUDIO_SILENCE_FRAME_CNT,
            toks,
            [ZERO_TEXT_TOKEN] * AUDIO_SILENCE_FRAME_CNT,
        ):
            for t in phase_toks:
                tt = torch.full((B,), int(t), dtype=torch.long, device=dev)
                self._tick(sine, silence, tt, all_rows)
        self._codec.reset_streaming()

    # -- batched surface (mirrors the previous engine's API) --------------------

    def open_batch(self) -> None:
        if not self._loaded:
            self.load()
        self._reset_all_streaming()
        self._boot_prefill(None)
        self._opened = True
        logger.info("PersonaPlex native: %d slots prefilled", self.config.batch_size)

    def prefill_steps(self, persona: str | None = None) -> list[PrefillStep]:
        zero = self._silence
        silence = [
            PrefillStep("tokens", moshi_tokens=zero, text_token=ZERO_TEXT_TOKEN, user_sine=True)
            for _ in range(AUDIO_SILENCE_FRAME_CNT)
        ]
        steps: list[PrefillStep] = [PrefillStep("sacrifice")]
        for e in self._voice_embeddings:
            steps.append(PrefillStep("voice", embedding=e.view(1, 1, -1)))
        steps.append(PrefillStep("voice_end"))
        steps.extend(silence)
        text = persona if persona is not None else self.config.persona
        if text:
            for tok in self._tokenizer.encode(wrap_with_system_tags(text)):
                steps.append(PrefillStep("tokens", moshi_tokens=zero, text_token=int(tok), user_sine=True))
        steps.extend(silence)
        return steps

    def recycle_slot(self, b: int) -> None:
        self._temporal.reset_slot(b)
        self._temporal.bump_slot_start(b)  # LM sacrifice-tick semantics
        self._local[b] = 0
        self._cache[b] = self._initial.view(-1, 1).expand(-1, self._cache.shape[2])
        self._provided[b] = False

    def voice_end(self, b: int) -> None:
        import torch

        ct = self._cache.shape[2]
        shift = int((self._phys - int(self._local[b])) % ct)
        self._cache[b] = torch.roll(self._voice_cache[0], shifts=shift, dims=-1)

    def end_prefill(self, b: int) -> None:
        self._codec.reset_slot(b)

    def step_batch(self, user_pcm: NDArray[np.float32], prefill: dict[int, PrefillStep] | None = None):
        import torch

        if not self._opened:
            raise RuntimeError("open_batch() must be called before step_batch()")
        cfg = self.config
        B = cfg.batch_size
        pcm = np.ascontiguousarray(user_pcm, dtype=np.float32).reshape(B, self.frame_size)
        codes = self._codec.encode_frame(torch.from_numpy(pcm).to(cfg.device)).long()
        return self._step_from_codes(codes, prefill)

    def step_codes(self, user_codes, prefill: dict[int, PrefillStep] | None = None):
        """Drive the lockstep loop from precomputed user codes (golden replays).

        Kept public deliberately: bit-exact golden-replay verification must
        bypass the Mimi encoder, whose streaming conv state makes encoded
        codes run-order dependent (valid but not reproducible token-for-token).
        """
        return self._step_from_codes(user_codes.to(self.config.device).long(), prefill)

    def _step_from_codes(self, codes, prefill):
        import torch

        from vllm_omni.experimental.fullduplex.personaplex.engine import FrameOutput

        cfg = self.config
        B = cfg.batch_size
        dev = cfg.device
        forced_moshi = forced_text = force_mask = embed_override = None
        if prefill:
            force_mask = torch.zeros(B, dtype=torch.bool, device=dev)
            embed_rows = torch.zeros(B, dtype=torch.bool, device=dev)
            emb_full = None
            forced_moshi = self._initial[1:9].view(1, 8).expand(B, 8).clone()
            forced_text = torch.full((B,), ZERO_TEXT_TOKEN, dtype=torch.long, device=dev)
            for b, stp in prefill.items():
                if stp.kind == "voice_end":
                    raise ValueError("voice_end is not a tick; call engine.voice_end(b)")
                codes[b] = self._sine if stp.user_sine else self._initial[9:]
                if stp.kind == "voice":
                    embed_rows[b] = True
                    if emb_full is None:
                        emb_full = torch.zeros(B, 1, stp.embedding.shape[-1], dtype=stp.embedding.dtype, device=dev)
                    emb_full[b] = stp.embedding[0]
                    force_mask[b] = True  # dummy staging like step_embeddings
                else:
                    force_mask[b] = True
                    if stp.kind == "tokens":
                        forced_moshi[b] = stp.moshi_tokens
                        forced_text[b] = int(stp.text_token)
            if bool(embed_rows.any()):
                embed_override = (emb_full, embed_rows)

        ran, out = self._tick(codes, forced_moshi, forced_text, force_mask, embed_override)
        self.last_out_tokens = None if out is None else out.detach().cpu()
        if not ran or out is None:
            return [None] * B

        valid = self._last_valid  # [B] cpu bool
        emit = [bool(valid[b]) and (not prefill or b not in prefill) for b in range(B)]
        audio_tokens = out[:, 1:9].clone()
        if not all(emit):
            hide = torch.tensor([not e for e in emit], device=dev).view(B, 1)
            audio_tokens = torch.where(hide, self._silence.view(1, 8).expand(B, 8), audio_tokens)
        pcm_out = self._codec.decode_frame(audio_tokens)  # [B, frame]
        outs = []
        for b in range(B):
            if not emit[b]:
                outs.append(None)
                continue
            audio = pcm_out[b].float().detach().cpu().numpy().astype(np.float32)
            outs.append(FrameOutput(audio=audio, text=self._decode_text(int(out[b, 0].item()))))
        return outs

    # -- single-session FrameStepper surface -------------------------------------

    def open_session(self, voice_prompt: str | None = None, persona: str | None = None) -> None:
        with self._fs_lock:
            if not self._loaded:
                self.load()
            # Always resolve a voice: falling back to the configured default here
            # keeps a previous session's custom voice from leaking into this one.
            self._load_voice(voice_prompt or self.config.voice_prompt)
            self._reset_all_streaming()
            self._boot_prefill(persona)
            self._opened = True

    def step(self, user_pcm: NDArray[np.float32]):
        with self._fs_lock:
            pcm = np.ascontiguousarray(user_pcm, dtype=np.float32).reshape(-1)
            if pcm.shape[0] != self.frame_size:
                raise ValueError(f"expected {self.frame_size} samples per frame, got {pcm.shape[0]}")
            outs = self.step_batch(pcm.reshape(1, -1))
        from vllm_omni.experimental.fullduplex.personaplex.engine import FrameOutput

        return outs[0] if outs[0] is not None else FrameOutput(audio=None, text=None)

    def _decode_text(self, token_id: int) -> str | None:
        if token_id in NON_TEXT_TOKEN_IDS or token_id >= TEXT_VOCAB_SIZE:
            return None
        return self._tokenizer.id_to_piece(token_id).replace("▁", " ")
