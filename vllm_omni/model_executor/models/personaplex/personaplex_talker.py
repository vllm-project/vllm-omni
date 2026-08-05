# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PersonaPlex talker: the temporal transformer as a vLLM-native omni AR stage.

This is the stage-0 (``LLM_AR``) model the OmniGPUModelRunner drives. It composes
three pieces, each verified in isolation against Moshi:

* the Helium temporal transformer (:class:`HeliumModel`) on vLLM paged attention,
  consuming per-frame ``inputs_embeds`` and producing the per-frame hidden state;
* the input embeddings (:class:`PersonaPlexInputEmbeddings`, ``embed_codes``) that
  build those ``inputs_embeds`` from the delayed 17-row token stack;
* the depformer (:class:`PersonaPlexDepformer`) that, conditioned on the temporal
  hidden state and the sampled text token, predicts the per-frame audio codes.

Per-frame protocol (OmniGPUModelRunner, gpu_model_runner.py):

1. ``compute_logits`` produces the text logits; the engine samples the text token.
2. ``preprocess`` (per-request, with that request's ``additional_information``)
   carries Moshi's acoustic-delay cache and the precomputed user-audio code stream,
   builds the base ``inputs_embeds`` for the current frame, and exposes the
   previous frame's temporal hidden + text-step embedding via ``mtp_inputs``.
3. ``talker_mtp`` (batched, stateless) runs the depformer to predict the agent
   codes and finishes the next frame's ``inputs_embeds``; the codes are stored
   under ``talker_mtp_output_key=("codes","audio")`` for the Mimi code2wav stage.

Phase 1 is turn-based: the user-audio rows (9..16) come from a precomputed Mimi
encode of the input WAV (built in ``preprocess``); live duplex is Phase 2.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.personaplex.configuration_personaplex import (
    PersonaPlexConfig,
)
from vllm_omni.model_executor.models.personaplex.modeling_helium import HeliumModel
from vllm_omni.model_executor.models.personaplex.personaplex_depformer import (
    PersonaPlexDepformer,
)
from vllm_omni.model_executor.models.personaplex.personaplex_embeddings import (
    PersonaPlexInputEmbeddings,
)

__all__ = ["PersonaPlexTalkerForConditionalGeneration"]


class PersonaPlexTalkerForConditionalGeneration(nn.Module):
    """vLLM-native PersonaPlex talker (temporal transformer + depformer).

    Plain ``nn.Module`` (not ``SupportsPP``): inheriting a vLLM Protocol puts
    ``Protocol`` in the MRO and breaks vLLM's runtime ``isinstance`` check for
    ``VllmModelForTextGeneration``, which would misclassify the talker as a
    non-generate model. Qwen3-TTS's talker is plain ``nn.Module`` for the same
    reason; PersonaPlex does not need pipeline parallelism.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        config: PersonaPlexConfig = vllm_config.model_config.hf_config  # type: ignore[assignment]
        self.config = config
        self.temporal_config = config.temporal_config
        hidden = self.temporal_config.hidden_size
        self._dtype = getattr(vllm_config.model_config, "dtype", torch.bfloat16)

        # Temporal backbone on vLLM paged attention (consumes inputs_embeds).
        self.model = HeliumModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            config=self.temporal_config,
        )
        # Text head (Moshi text_linear -> lm_head), vocab = text_vocab_size (32000).
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.text_vocab_size,
                hidden,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.text_vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        # Verified custom components: embed_codes + depformer.
        self.input_embeddings = PersonaPlexInputEmbeddings(config)
        self.depformer = PersonaPlexDepformer(
            config.depformer_config,
            temporal_hidden_size=hidden,
            text_card=config.text_vocab_size,
        )

        # Omni AR runner contract.
        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = True  # capture each frame's hidden for the next depformer step
        self.requires_full_prefix_cached_hidden_states = False
        # Keep the per-frame "last" hidden on GPU (avoids a CPU round-trip each step).
        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {("hidden_states", "last")}
        self.mtp_hidden_size = hidden
        self.talker_mtp_output_key = ("codes", "audio")
        # dep_q audio codebooks per frame; only cb 0..num_active are vocoded.
        self.dep_q = config.depformer_config.dep_q
        self.num_active_codebooks = config.depformer_config.num_active_codebooks

    # ------------------------------------------------------------------
    # Core forward / logits
    # ------------------------------------------------------------------
    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        """Placeholder embedding for vLLM's ``VllmModel`` protocol.

        The talker is driven by precomputed ``inputs_embeds`` (built in
        ``preprocess`` via ``embed_codes``); ``input_ids`` are only in-vocab
        bookkeeping placeholders. Return a zero embedding of the temporal hidden
        size — the runner replaces it with the real per-frame ``inputs_embeds``.
        """
        return torch.zeros(
            (input_ids.shape[0], self.mtp_hidden_size),
            device=input_ids.device,
            dtype=self._dtype,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        if hidden_states.dim() == 3:
            b, s, h = hidden_states.shape
            logits = self.logits_processor(self.lm_head, hidden_states.reshape(b * s, h))
            return None if logits is None else logits.reshape(b, s, -1)
        return self.logits_processor(self.lm_head, hidden_states)

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        hidden = model_outputs
        info_dicts = kwargs.get("model_intermediate_buffer") or kwargs.get("runtime_additional_information") or []
        audio_codes_list: list[torch.Tensor] = []
        for info in info_dicts:
            if not isinstance(info, dict):
                continue
            ac = info.get("codes", {}).get("audio")
            if isinstance(ac, torch.Tensor):
                audio_codes_list.append(ac)
        if not audio_codes_list:
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})
        audio_codes = torch.cat(audio_codes_list, dim=0)
        hidden = hidden[: int(audio_codes.shape[0])]
        return OmniOutput(text_hidden_states=hidden, multimodal_outputs={"codes": {"audio": audio_codes}})

    # ------------------------------------------------------------------
    # Per-frame omni AR protocol: preprocess (stateful) + talker_mtp (batched)
    # ------------------------------------------------------------------
    def _initial_frame_embed(self, device: torch.device) -> torch.Tensor:
        """``embed_codes`` of Moshi's initial token frame (text+audio SOS)."""
        text_init = self.config.text_vocab_size  # text_initial_token_id
        audio_init = self.config.audio_vocab_size  # card == initial audio token
        n_q = self.config.num_audio_codebooks
        stack = torch.empty((1, 1 + n_q, 1), dtype=torch.long, device=device)
        stack[:, 0] = text_init
        stack[:, 1:] = audio_init
        return self.input_embeddings(stack).reshape(1, -1)  # [1, hidden]

    def _build_frame_embed(
        self,
        text_token: torch.Tensor,
        last_agent: torch.Tensor | None,
        prev_agent: torch.Tensor | None,
        device: torch.device,
        user_d0: torch.Tensor | None = None,
        user_d1: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Full ``inputs_embeds`` for a decode frame (Moshi cache read at offset-1).

        Acoustic-delay decomposition (delays ``[0, 0,1x7, 0,1x7]``): after a
        depformer step, Moshi writes every unforced agent codebook into the
        current cache target. The next temporal input therefore reads the whole
        previous effective agent frame. User writes are delayed by codebook, so
        user cb0 and cb1..7 still come from ``user_d0`` and ``user_d1``.
        Unfilled positions default to the initial token (card).
        """
        n_q = self.config.num_audio_codebooks  # 16 (8 agent + 8 user)
        n_user = n_q // 2  # 8
        audio_init = self.config.audio_vocab_size  # card == initial audio token (Moshi cache default)
        # No-history / not-yet-generated delayed positions = the initial token (card),
        # NOT 0 (verified against Moshi's per-frame input stack).
        stack = torch.full((1, 1 + n_q, 1), audio_init, dtype=torch.long, device=device)
        stack[:, 0] = text_token.reshape(1)
        # The full effective agent frame was stored at the previous cache target
        # after applying any teacher forcing. The next cache read sees all eight
        # codebooks from that frame.
        if last_agent is not None and last_agent.numel() >= n_user:
            stack[0, 1 : 1 + n_user, 0] = last_agent.reshape(-1)[:n_user].to(device)
        # user cb0 (row 9) delay-0; user cb1..7 (rows 10..16) delay-1.
        user_base = 1 + n_user  # row index 9
        if user_d0 is not None and user_d0.numel() >= 1:
            stack[0, user_base, 0] = user_d0.reshape(-1)[0].to(device)
        if user_d1 is not None and user_d1.numel() >= n_user:
            stack[0, user_base + 1 : user_base + n_user, 0] = user_d1.reshape(-1)[1:n_user].to(device)
        return self.input_embeddings(stack).reshape(1, -1)  # [1, hidden]

    def _build_prefill_embed(
        self,
        prefill_text: torch.Tensor,
        offset: int,
        span: int,
        device: torch.device,
        silence: torch.Tensor | None = None,
        user_sine: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-frame embeds for the persona system-prompt prefill.

        Each frame forces the persona/silence text token (row 0). The agent rows
        use Mimi-encoded silence, while the user rows optionally use the encoded
        sine frame from PersonaPlex's native text-prompt prefill. Without an
        explicit user frame, both sides fall back to silence.
        """
        n_q = self.config.num_audio_codebooks
        n_user = n_q // 2
        zero_text = 3  # Moshi LMGen.zero_text_code (silence/pad text)
        sil = (
            silence.reshape(-1).to(device) if isinstance(silence, torch.Tensor) and silence.numel() >= n_user else None
        )
        user = (
            user_sine.reshape(-1).to(device)
            if isinstance(user_sine, torch.Tensor) and user_sine.numel() >= n_user
            else sil
        )
        total = int(prefill_text.numel())
        rows = []
        for i in range(span):
            pos = offset + i
            text_tok = int(prefill_text[pos].item()) if pos < total else zero_text
            stack = torch.zeros((1, 1 + n_q, 1), dtype=torch.long, device=device)
            stack[:, 0] = text_tok
            if sil is not None:
                stack[0, 1 : 1 + n_user, 0] = sil[:n_user]  # agent rows = encoded silence
            if user is not None:
                stack[0, 1 + n_user : 1 + 2 * n_user, 0] = user[:n_user]
            rows.append(self.input_embeddings(stack).reshape(1, -1))
        return torch.cat(rows, dim=0)  # [span, hidden]

    @staticmethod
    def _user_frame(info: dict[str, Any], frame_idx: int) -> torch.Tensor | None:
        """Fetch the Mimi-encoded user codes for ``frame_idx`` (or None / out of range)."""
        uc = info.get("pplex_user_codes")
        if uc is None:
            return None
        if not isinstance(uc, torch.Tensor):
            uc = torch.as_tensor(uc, dtype=torch.long)
        if uc.ndim != 2 or frame_idx < 0 or frame_idx >= uc.shape[0]:
            return None
        return uc[frame_idx]

    @staticmethod
    def _last_agent_codes(info: dict[str, Any]) -> torch.Tensor | None:
        codes = info.get("codes")
        if isinstance(codes, dict):
            ac = codes.get("audio")
            if isinstance(ac, torch.Tensor) and ac.numel() > 0:
                return (ac[-1] if ac.ndim == 2 else ac).reshape(-1)
        return None

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Build the current frame's ``inputs_embeds`` and the mtp inputs.

        Prefill emits the initial-token frame embedding; each decode step builds the
        delayed base frame (everything but agent cb0, which ``talker_mtp`` adds).
        """
        additional = info_dict.get("additional_information")
        if isinstance(additional, dict):
            merged = {k: v for k, v in info_dict.items() if k != "additional_information"}
            for k, v in additional.items():
                merged.setdefault(k, v)
            info_dict = merged
        meta = info_dict.get("meta", {}) if isinstance(info_dict.get("meta"), dict) else {}

        device = input_ids.device
        span = int(input_ids.shape[0])
        is_prefill_raw = info_dict.get("_omni_is_prefill")
        if isinstance(is_prefill_raw, bool):
            is_prefill = is_prefill_raw
        else:
            is_prefill = span > 1

        duplex = info_dict.get("duplex")
        if isinstance(duplex, dict) and duplex.get("data_plane") is True and is_prefill:
            prompt_len_raw = info_dict.get("duplex_prompt_len", span)
            try:
                prompt_len = int(prompt_len_raw)
            except (TypeError, ValueError):
                prompt_len = span
            prepared = self._duplex_stage0_runtime().prepare_append(
                duplex,
                prompt_len=prompt_len,
                request_id=(str(info_dict["request_id"]) if isinstance(info_dict.get("request_id"), str) else None),
            )
            offset_raw = info_dict.get("duplex_token_offset", 0)
            try:
                offset = max(0, int(offset_raw))
            except (TypeError, ValueError):
                offset = 0
            local_offset = offset - prepared.prompt_offset
            if local_offset < 0:
                raise ValueError(
                    "PersonaPlex scheduled span precedes the current append: "
                    f"offset={offset}, append_offset={prepared.prompt_offset}, "
                    f"span={span}, prompt={prompt_len}"
                )
            req_embeds = prepared.inputs_embeds[local_offset : local_offset + span].to(
                device=device,
                dtype=self._dtype,
            )
            req_input_ids = prepared.input_ids[local_offset : local_offset + span].to(device=device)
            if req_embeds.shape[0] != span or req_input_ids.shape[0] != span:
                raise ValueError(
                    "PersonaPlex duplex prompt slice is shorter than the scheduled span: "
                    f"offset={offset}, span={span}, prompt={prompt_len}"
                )
            return req_input_ids, req_embeds, prepared.info_update

        zero_hidden = torch.zeros((1, self.mtp_hidden_size), device=device, dtype=self._dtype)

        prefill_text = info_dict.get("pplex_prefill_text")
        prefill_len = int(prefill_text.numel()) if isinstance(prefill_text, torch.Tensor) else 0

        if is_prefill:
            offset = max(0, int(info_dict.get("_omni_num_computed_tokens", 0) or 0))
            if isinstance(prefill_text, torch.Tensor) and prefill_text.numel() > 0:
                # Persona system-prompt prefill (step_system_prompts analog).
                silence = info_dict.get("pplex_silence_codes")
                emb = self._build_prefill_embed(prefill_text, offset, span, device, silence)
            else:
                emb = self._initial_frame_embed(device)
                if span > 1:
                    emb = emb.expand(span, -1).contiguous()
            ids_out = input_ids.clone()
            info_update = {
                "meta": {
                    "pplex_frame": int(meta.get("pplex_frame", 0)) + span,
                    "pplex_prefill_len": prefill_len,
                },
                "mtp_inputs": (zero_hidden, zero_hidden),
            }
            return ids_out, emb, info_update

        # Decode: input_ids == the text token sampled last step (delay-0 text).
        text_token = input_ids.reshape(-1)[:1]
        # Agent acoustic delay: cb0 (delay 0) = gen[t-1] = the last stored codes;
        # cb1..7 (delay 1) = gen[t-2] = the codes carried from the previous frame.
        last_agent = self._last_agent_codes(info_dict)
        prev_agent = info_dict.get("pplex_prev_agent")
        if prev_agent is not None and not isinstance(prev_agent, torch.Tensor):
            prev_agent = torch.as_tensor(prev_agent, dtype=torch.long)
        # Real user stream (Phase-1 turn-based): user cb0 delay-0, cb1..7 delay-1.
        # Index into the user stream relative to the first decode frame (after the
        # persona prefill), so the user audio aligns with the agent's response.
        prefill_len = int(meta.get("pplex_prefill_len", prefill_len) or 0)
        # decode_frame = pplex_frame - prefill_len (the prefill frame already advanced
        # pplex_frame, so this is d+1 at decode step d). Verified vs Moshi's per-frame
        # stack: user cb0 (delay 0) = enc[decode_frame-1] = enc[d]; cb1..7 (delay 1) =
        # enc[decode_frame-2]. (The agent stream lags one more: agent cb0 = gen[d-1].)
        decode_frame = max(0, int(meta.get("pplex_frame", 0)) - prefill_len)
        user_d0 = self._user_frame(info_dict, decode_frame - 1)
        user_d1 = self._user_frame(info_dict, decode_frame - 2)
        base = self._build_frame_embed(text_token, last_agent, prev_agent, device, user_d0, user_d1)
        text_step = self.input_embeddings.text_emb(text_token.reshape(1, 1)).reshape(1, -1)
        # The previous frame's temporal hidden (written by postprocess into
        # hidden_states["last"]) conditions this frame's depformer. Falls back to
        # zeros only on the very first decode step (no prior hidden yet).
        hs = info_dict.get("hidden_states", {}) if isinstance(info_dict.get("hidden_states"), dict) else {}
        last_hidden = hs.get("last")
        if isinstance(last_hidden, torch.Tensor) and last_hidden.numel() > 0:
            last_hidden = last_hidden.reshape(1, -1).to(device=device, dtype=self._dtype)
        else:
            last_hidden = zero_hidden
        info_update = {
            "meta": {"pplex_frame": int(meta.get("pplex_frame", 0)) + 1},
            "mtp_inputs": (last_hidden, text_step),
        }
        # Carry this frame's gen[t-1] forward; next frame reads it as gen[t-2] (cb1..7).
        if last_agent is not None:
            info_update["pplex_prev_agent"] = last_agent.detach().to(torch.long).cpu()
        return input_ids, base, info_update

    def _duplex_stage0_runtime(self):
        runtime = getattr(self, "_personaplex_duplex_stage0_runtime", None)
        if runtime is not None:
            return runtime
        from vllm_omni.experimental.fullduplex.personaplex.stage0 import (
            PersonaPlexStage0DuplexRuntime,
        )

        model_path = str(getattr(self.vllm_config.model_config, "model", ""))
        device = str(next(self.parameters()).device)
        runtime = PersonaPlexStage0DuplexRuntime(
            self,
            model_path=model_path,
            device=device,
            max_sessions=int(getattr(self.vllm_config.model_config, "duplex_max_sessions", 1)),
        )
        self._personaplex_duplex_stage0_runtime = runtime
        return runtime

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        runtime = getattr(self, "_personaplex_duplex_stage0_runtime", None)
        if runtime is None:
            return
        for request_id in finished_req_ids:
            runtime.close_request(request_id)

    def postprocess(self, hidden_states: torch.Tensor, **_: Any) -> dict[str, Any]:
        """Capture this frame's last hidden for the next step's depformer (mtp)."""
        if hidden_states is None or hidden_states.numel() == 0:
            return {}
        return {"hidden_states": {"last": hidden_states[-1, :].detach()}}

    def talker_mtp(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        last_talker_hidden: torch.Tensor,
        text_step: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the depformer for the frame and finish its ``inputs_embeds``.

        Returns ``(inputs_embeds, audio_codes[B, dep_q])``. ``audio_codes`` are
        stored under ``("codes","audio")``; ``inputs_embeds`` (base from preprocess
        + the fresh agent cb0 embedding, the delay-0 acoustic token) feeds the next
        temporal forward.
        """
        bsz = int(input_ids.shape[0])
        dtype = self._dtype
        text_token = input_ids.reshape(bsz).to(torch.long)
        hidden = last_talker_hidden.reshape(bsz, 1, -1).to(dtype)

        codes = self.depformer(text_token, hidden)  # [B, dep_q] == gen[t]

        # The frame's inputs_embeds is built fully in preprocess (Moshi cache read at
        # offset-1, with agent cb0 = gen[t-1]); gen[t] feeds the NEXT frame, so pass
        # the embed through unchanged and just emit this frame's codes.
        inputs_embeds = input_embeds.reshape(bsz, -1).to(dtype)
        return inputs_embeds, codes.to(torch.long)

    def post_sample_talker_mtp(
        self,
        *,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        req_ids: list[str],
        req_infos: list[dict[str, Any]],
    ) -> torch.Tensor:
        """Generate depformer codes for a one-token resumable duplex segment.

        The normal runner invokes ``talker_mtp`` at the start of the next decode
        step. PersonaPlex's unified duplex request instead appends one audio
        frame and stops after one sampled text token, so there is no next decode
        step. Run the same depformer dependency immediately from the current
        sampled text token and temporal hidden state.
        """
        bsz = int(input_ids.shape[0])
        if len(req_infos) != bsz:
            raise ValueError(
                f"PersonaPlex depformer request information does not match batch: {len(req_infos)} != {bsz}"
            )
        text_token = input_ids.reshape(bsz).to(torch.long)
        hidden = hidden_states.reshape(bsz, 1, -1).to(self._dtype)
        audio_tokens: list[torch.Tensor] = []
        audio_provided: list[torch.Tensor] = []
        for info in req_infos:
            tokens = info.get("pplex_depformer_audio_tokens")
            provided = info.get("pplex_depformer_audio_provided")
            if not isinstance(tokens, torch.Tensor) or not isinstance(provided, torch.Tensor):
                raise ValueError("PersonaPlex duplex depformer teacher-forcing state is missing")
            tokens = tokens.reshape(-1)
            provided = provided.reshape(-1)
            if tokens.shape != provided.shape:
                raise ValueError(
                    "PersonaPlex depformer teacher-forcing token/mask shapes differ: "
                    f"{tuple(tokens.shape)} != {tuple(provided.shape)}"
                )
            audio_tokens.append(tokens)
            audio_provided.append(provided)
        codes = self.depformer(
            text_token,
            hidden,
            audio_tokens=torch.stack(audio_tokens).to(
                device=hidden.device,
                dtype=torch.long,
            ),
            audio_provided=torch.stack(audio_provided).to(
                device=hidden.device,
                dtype=torch.bool,
            ),
        ).to(torch.long)
        runtime = self._duplex_stage0_runtime()
        for row, request_id in enumerate(req_ids):
            runtime.record_sample(
                request_id=request_id,
                text_token=text_token[row],
                agent_codes=codes[row],
            )
        return codes

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Route the Moshi checkpoint into the three components.

        * ``transformer.*`` / ``out_norm.alpha`` -> the Helium temporal backbone
          (same q/k-split + gate/up-split + alpha-squeeze map as HeliumForCausalLM).
        * ``text_linear.weight`` -> ``lm_head``.
        * ``emb.*`` / ``text_emb.weight`` -> input embeddings.
        * ``depformer*`` / ``linears.*`` -> depformer.
        """
        weights = list(weights)
        params = dict(self.named_parameters(remove_duplicate=False))
        loaded: set[str] = set()

        emb_w: list[tuple[str, torch.Tensor]] = []
        dep_w: list[tuple[str, torch.Tensor]] = []
        for name, w in weights:
            if name.startswith(("emb.", "text_emb.")):
                emb_w.append((name, w))
            elif name.startswith(("depformer.", "depformer_in.", "depformer_emb.", "depformer_text_emb", "linears.")):
                dep_w.append((name, w))
            elif name == "text_linear.weight":
                self._load_direct("lm_head.weight", w, params, loaded)
            else:
                loaded |= self._load_temporal(name, w, params)

        # Delegate to the verified component loaders, prefixed into this module.
        for sub, sub_w in (("input_embeddings", emb_w), ("depformer", dep_w)):
            module = getattr(self, sub)
            for tgt in module.load_weights(sub_w):
                loaded.add(f"{sub}.{tgt}")
        return loaded

    def _load_temporal(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        params: dict[str, nn.Parameter],
    ) -> set[str]:
        out: set[str] = set()
        if name == "out_norm.alpha":
            self._load_direct("model.norm.weight", loaded_weight.squeeze(), params, out)
            return out
        prefix = "transformer.layers."
        if not name.startswith(prefix):
            return out
        rest = name.removeprefix(prefix)
        layer_index, _, suffix = rest.partition(".")
        if not layer_index.isdigit() or not suffix:
            return out
        base = f"model.layers.{layer_index}"
        if suffix == "self_attn.in_proj_weight":
            q, k, v = loaded_weight.chunk(3, dim=0)
            pname = f"{base}.self_attn.qkv_proj.weight"
            self._load_shard(pname, q, "q", params)
            self._load_shard(pname, k, "k", params)
            self._load_shard(pname, v, "v", params)
            out.add(pname)
        elif suffix == "self_attn.out_proj.weight":
            self._load_direct(f"{base}.self_attn.o_proj.weight", loaded_weight, params, out)
        elif suffix == "gating.linear_in.weight":
            gate, up = loaded_weight.chunk(2, dim=0)
            pname = f"{base}.mlp.gate_up_proj.weight"
            self._load_shard(pname, gate, 0, params)
            self._load_shard(pname, up, 1, params)
            out.add(pname)
        elif suffix == "gating.linear_out.weight":
            self._load_direct(f"{base}.mlp.down_proj.weight", loaded_weight, params, out)
        elif suffix == "norm1.alpha":
            self._load_direct(f"{base}.input_layernorm.weight", loaded_weight.squeeze(), params, out)
        elif suffix == "norm2.alpha":
            self._load_direct(f"{base}.post_attention_layernorm.weight", loaded_weight.squeeze(), params, out)
        return out

    @staticmethod
    def _load_direct(
        name: str,
        loaded_weight: torch.Tensor,
        params: dict[str, nn.Parameter],
        loaded: set[str],
    ) -> None:
        if name not in params:
            return
        param = params[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)
        loaded.add(name)

    @staticmethod
    def _load_shard(
        name: str,
        loaded_weight: torch.Tensor,
        shard_id: str | int,
        params: dict[str, nn.Parameter],
    ) -> None:
        if name not in params:
            return
        param = params[name]
        param.weight_loader(param, loaded_weight, shard_id)
