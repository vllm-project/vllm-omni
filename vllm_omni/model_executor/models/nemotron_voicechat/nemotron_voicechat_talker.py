# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage-1 talker for NemotronVoiceChat: frame-locked text -> 31-quantizer codes.

``LLM_AR`` stage driven per frame by the vLLM engine, with the vendored NeMo
EAR-TTS model (28-layer Gemma3-style backbone + MoG head) executed inside the
per-request ``preprocess`` hook. Classifier-free guidance (internal batch
doubling) and MoG sampling happen inside the vendored forward exactly as in
NeMo — vLLM's sampler never touches the audio codes.

Execution contract (mirrors the NeMo ``NemotronVoiceChat.offline_inference``
TTS half):

* The request's ``additional_information`` carries the full frame-aligned text
  timeline (built by ``thinker2talker_token_only``), INCLUDING the PAD prompt
  region — the talker's KV state at the first acoustic frame depends on having
  processed it (NeMo trims prompt-region codes only after the loop).
* vLLM prompt = one placeholder token (timeline position 0; NeMo's loop starts
  at t=1). Each decode step t runs ``infer_codes_one_step`` with
  ``current_subword_id=timeline[t]``, ``prev_subword_id=timeline[t-1]`` (the
  first step uses the warmup's last init subword, as in NeMo), and the previous
  step's code stack; the resulting ``[1, 31]`` codes accumulate under
  ``("codes","audio")`` in the request state for the full-payload producer.
* Warmup: ``set_init_inputs(speaker_name=...)`` (pre-baked "Aria" latent) +
  one ``tts_model(**init_inputs)`` forward seeds ``past_key_values`` and the
  initial code, exactly as in the reference.
* Completion: after consuming timeline position ``T-1`` the stage emits a stop
  placeholder token (the deploy yaml sets ``stop_token_ids: [1]``); codes for
  positions ``1..T-1`` are shipped and the producer trims the prompt region.

Heavyweight per-request state (HF ``past_key_values`` etc.) lives in a
model-side session dict keyed by request id (PersonaPlex pattern), cleaned up
via ``on_requests_finished``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.nemotron_voicechat.runtime_info import (
    merge_runtime_info,
    require_request_id,
    scalar_bool,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)

_CONTINUE_TOKEN = 0
_STOP_TOKEN = 1


def sanitize_tts_model_cfg(tts_model_cfg: dict[str, Any]) -> dict[str, Any]:
    """Make the vendored DuplexEARTTS construction config-only.

    Every parameter comes from the unified checkpoint streamed in
    ``load_weights``; the vendored constructors must never download or load
    external pretrained weights. Returns a deep copy so the shared stage
    config is never mutated:

    * ``tts_config.pretrained_text_name`` would make ``RVQEARTTSModel``
      download a full LLM backbone — dropped (the backbone builds from
      ``backbone_type``/``backbone_config``).
    * ``pretrained_codec_model`` would make ``setup_audio_codec`` load an
      external codec checkpoint at construction — dropped (the codec weights
      arrive with the ``tts_model.audio_codec.*`` subtree, and the RVQ
      embedding binding is overwritten by the same ``load_state_dict``).
    * ``tts_config.cas_config.pretrained_tokenizer_name`` is dropped, matching
      NeMo's own inference override (the CAS vocab is baked into the weights).
    * ``pretrained_lm_name`` survives ONLY as the tokenizer reference (the
      context-LM load path is disabled by ``context_hidden_size: null``); the
      ``NEMOTRON_VOICECHAT_LLM_PATH`` override keeps it air-gap friendly.
    """
    import copy
    import os

    cfg = copy.deepcopy(tts_model_cfg)
    cfg.pop("pretrained_codec_model", None)
    tts_config = cfg.get("tts_config")
    if isinstance(tts_config, dict):
        tts_config.pop("pretrained_text_name", None)
        cas_cfg = tts_config.get("cas_config")
        if isinstance(cas_cfg, dict):
            cas_cfg.pop("pretrained_tokenizer_name", None)
    llm_override = os.environ.get("NEMOTRON_VOICECHAT_LLM_PATH")
    if llm_override:
        cfg["pretrained_lm_name"] = llm_override
    return cfg


class NemotronVoiceChatTalkerForConditionalGeneration(nn.Module):
    """vLLM AR stage wrapping the vendored NeMo EAR-TTS per-frame step."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        talker_cfg = getattr(self.config, "talker_config", None)
        self._hidden = int(getattr(talker_cfg, "hidden_size", 1152))
        self._vocab = max(int(getattr(talker_cfg, "vocab_size", 1024)), 2)
        self._dtype = getattr(vllm_config.model_config, "dtype", torch.bfloat16)
        self._speaker_name = getattr(self.config, "inference_speaker_name", "Aria") or "Aria"

        # Omni AR runner contract flags.
        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = False
        self.requires_full_prefix_cached_hidden_states = False

        # The vendored (eager / captured-step) paths share single-session
        # buffers (StaticCache, capture-time addresses) and stay batch=1; the
        # NATIVE path keeps all per-request state in _sessions (paged KV,
        # per-request MoG on each request's own hidden slice) and supports
        # multi-session batching (validated at max_num_seqs<=4). The native
        # flag is read below, so peek at it here for the guard.
        max_num_seqs = int(getattr(vllm_config.scheduler_config, "max_num_seqs", 1))
        if max_num_seqs != 1 and not bool(getattr(self.config, "use_native_talker", False)):
            raise NotImplementedError(
                f"NemotronVoiceChat talker supports max_num_seqs=1 only (got {max_num_seqs}) on the "
                "vendored paths; the EAR-TTS session buffers are not batch-safe. Use "
                "hf_overrides.use_native_talker for multi-session batching."
            )

        # CUDA-graph fast path (hf_overrides.use_talker_cuda_graphs). Off by
        # default: the eager path is the NeMo bit-parity acceptance path, while
        # the captured step replays the same math ~3x faster with a
        # graph-managed RNG (audio equivalent but not bit-identical).
        self._use_cuda_graphs = bool(getattr(self.config, "use_talker_cuda_graphs", False))
        self._graph_max_cache_len = int(getattr(self.config, "talker_graph_max_cache_len", 4096))
        self._step_graph: Any | None = None

        # Native-vLLM path (hf_overrides.use_native_talker, experimental): the
        # Gemma3 backbone runs as a real vLLM model on PagedAttention (KV and
        # per-step attention cost scale with the ACTUAL session length; vLLM's
        # own decode CUDA graphs apply), mirroring NVIDIA's deployment. The
        # per-frame MoG sampling runs on the backbone's hidden state in
        # make_omni_output. Classifier-free guidance is NOT applied yet on
        # this path (single conditional stream; the paired-request CFG design
        # is a documented follow-up), so audio quality trails the vendored
        # CFG paths — use for development only.
        self._use_native_backbone = bool(getattr(self.config, "use_native_talker", False))
        # Classifier-free guidance on the native path
        # (hf_overrides.native_talker_guidance): the conditional stream stays
        # on vLLM paged KV; the unconditional stream mirrors it on the vendored
        # HF backbone (see talker_native.UncondStream) and the pair is blended
        # per NeMo's generate_step math. Off by default (measured ASR-equivalent
        # at this checkpoint's guidance_scale=0.2, but kept available for
        # listening-test parity with the vendored paths).
        self._native_guidance = self._use_native_backbone and bool(
            getattr(self.config, "native_talker_guidance", False)
        )
        # Captured MoG sampling step (built lazily on the first decode frame;
        # falls back to eager sampling if capture fails).
        self._mog_graph: Any | None = None
        self._uncond_graph: Any | None = None
        self.backbone: nn.Module | None = None
        if self._use_native_backbone:
            if self._use_cuda_graphs:
                raise ValueError(
                    "NemotronVoiceChat talker: use_native_talker and use_talker_cuda_graphs are mutually exclusive."
                )
            from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix

            from vllm_omni.model_executor.models.nemotron_voicechat.talker_native import (
                synthesize_backbone_config,
            )

            tts_config = dict(getattr(self.config, "tts_cfg", {}) or {}).get("tts_config") or {}
            backbone_cfg = synthesize_backbone_config(dict(tts_config))
            self.backbone = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=backbone_cfg,
                prefix=maybe_prefix(prefix, "backbone"),
                architectures=["Gemma3ForCausalLM"],
            )

        # Vendored DuplexEARTTS; constructed in load_weights.
        self.tts: nn.Module | None = None
        # request_id -> mutable session state (past_key_values, prev code, ...).
        self._sessions: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # vLLM model protocol (placeholder token loop).
    # ------------------------------------------------------------------
    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return torch.zeros((input_ids.shape[0], self._hidden), device=input_ids.device, dtype=self._dtype)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        if self._use_native_backbone:
            # Real Gemma3 forward on PagedAttention; preprocess injected the
            # fused embeddings, make_omni_output turns the hidden state into
            # codes (MoG) and the CONTINUE/STOP flag.
            return self.backbone(
                input_ids=None,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
            )
        # The per-frame compute happens in preprocess; the "hidden state" only
        # transports the stop flag (column 0) to compute_logits.
        if inputs_embeds is not None:
            return inputs_embeds
        assert input_ids is not None
        return self.embed_input_ids(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> torch.Tensor:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        hs = hidden_states
        flag = hs[..., 0].to(torch.float32)  # 1.0 => finished
        logits = torch.full((hs.shape[0], self._vocab), -1.0e4, device=hs.device, dtype=torch.float32)
        logits[:, _CONTINUE_TOKEN] = (1.0 - flag) * 10.0
        logits[:, _STOP_TOKEN] = flag * 10.0
        return logits

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        if self._use_native_backbone:
            return self._native_make_omni_output(model_outputs, **kwargs)
        hidden = model_outputs
        info_dicts = kwargs.get("model_intermediate_buffer") or kwargs.get("runtime_additional_information") or []
        codes_list: list[torch.Tensor] = []
        prompt_len: int | None = None
        codec_streaming = False
        for info in info_dicts:
            if not isinstance(info, dict):
                continue
            codes = info.get("codes", {})
            audio = codes.get("audio") if isinstance(codes, dict) else None
            if isinstance(audio, torch.Tensor):
                codes_list.append(audio)
            meta = info.get("meta")
            reported = meta.get("nvc_logical_prompt_len") if isinstance(meta, dict) else None
            if reported is None:
                reported = info.get("meta.nvc_logical_prompt_len")
            if reported is not None:
                prompt_len = int(reported)
            streaming = meta.get("codec_streaming") if isinstance(meta, dict) else None
            if streaming is None:
                streaming = info.get("meta.codec_streaming")
            codec_streaming = codec_streaming or scalar_bool(streaming)
        if not codes_list:
            return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})
        codes = torch.cat(codes_list, dim=0)
        logger.debug(
            "Nemotron VoiceChat talker emitted codes: shapes=%s combined=%s prompt_len=%s",
            [tuple(item.shape) for item in codes_list],
            tuple(codes.shape),
            prompt_len,
        )
        outputs: dict[str, Any] = {"codes": {"audio": codes}}
        # The prompt length must ride the WIRE payload (multimodal_outputs):
        # the async-chunk dispatcher hands exactly this dict to the
        # talker2code2wav producer, while request.additional_information is
        # overwritten concurrently by arriving thinker chunks.
        output_meta: dict[str, Any] = {}
        if prompt_len is not None:
            # 1-D on purpose: the wire-payload builder indexes element.shape[0].
            output_meta["nvc_logical_prompt_len"] = torch.tensor([prompt_len], dtype=torch.long)
        if codec_streaming:
            output_meta["codec_streaming"] = torch.tensor([True], dtype=torch.bool)
        if output_meta:
            outputs["meta"] = output_meta
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs=outputs,
        )

    # ------------------------------------------------------------------
    # Per-frame vendored TTS step.
    # ------------------------------------------------------------------
    def _try_timeline(self, info: dict[str, Any]) -> torch.Tensor | None:
        """Extract the frame-locked text timeline from either payload shape.

        Sync mode ships ``nvc_text_timeline`` (+ ``nvc_logical_prompt_len``) in
        additional_information; async-chunk mode replaces the request info with
        the connector payload each chunk, carrying the CUMULATIVE timeline under
        ``ids.all`` (and the logical prompt length under
        ``meta.num_processed_tokens``). Returns None when neither is present.
        """
        timeline = info.get("nvc_text_timeline")
        if timeline is not None and info.get("nvc_logical_prompt_len") is None:
            raise ValueError(
                "NemotronVoiceChat talker got 'nvc_text_timeline' without "
                "'nvc_logical_prompt_len'; the code2wav producer needs the prompt "
                "length to trim prompt-region codes."
            )
        if timeline is None:
            ids = info.get("ids")
            timeline = ids.get("all") if isinstance(ids, dict) else None
            if timeline is None:
                timeline = info.get("ids.all")
        if timeline is None:
            return None
        if isinstance(timeline, torch.Tensor):
            return timeline.reshape(-1).to(torch.long)
        return torch.as_tensor(list(timeline), dtype=torch.long)

    @staticmethod
    def _logical_prompt_len(info: dict[str, Any]) -> int | None:
        """Logical prompt length from either payload shape (sync/async)."""
        value = info.get("nvc_logical_prompt_len")
        if value is None:
            meta = info.get("meta")
            value = meta.get("num_processed_tokens") if isinstance(meta, dict) else None
            if value is None:
                value = info.get("meta.num_processed_tokens")
        return int(value) if value is not None else None

    @staticmethod
    def _upstream_finished(info: dict[str, Any]) -> bool:
        """True when the async producer marked its final chunk (meta.finished)."""
        meta = info.get("meta")
        flag = meta.get("finished") if isinstance(meta, dict) else None
        if flag is None:
            flag = info.get("meta.finished")
        if isinstance(flag, torch.Tensor):
            return bool(flag.reshape(-1)[0].item()) if flag.numel() else False
        return bool(flag)

    def _timeline(self, info: dict[str, Any]) -> torch.Tensor:
        timeline = self._try_timeline(info)
        if timeline is None:
            # Explicit failure — no prompt_token_ids fallback: an implicit
            # timeline would silently skip the prompt-region trim downstream.
            raise ValueError(
                "NemotronVoiceChat talker request is missing its timeline metadata "
                "('nvc_text_timeline' in additional_information, or 'ids.all' on the "
                "async-chunk payload); the stage cannot synthesize speech without "
                "the thinker's frame-locked tokens."
            )
        return timeline

    @staticmethod
    def _expected_total_tokens(info: dict[str, Any]) -> int | None:
        """Producer-announced FINAL timeline length (meta.expected_total_tokens)."""
        meta = info.get("meta")
        value = meta.get("expected_total_tokens") if isinstance(meta, dict) else None
        if value is None:
            value = info.get("meta.expected_total_tokens")
        return int(value) if value is not None else None

    def _graph_session_usable(self, timeline: torch.Tensor, expected_len: int | None) -> bool:
        """Whether the CUDA-graph fast path can serve this session.

        ``expected_len`` is the known FINAL timeline length (sync mode, or a
        frame-locked async producer's meta.expected_total_tokens); the graph
        picks the smallest fitting cache bucket from it, since per-step
        attention cost scales with the bucket size. ``None`` = unbounded
        (duplex) -> largest bucket.
        """
        if self._step_graph is None:
            return False
        needed = max(int(timeline.numel()), expected_len or 0)
        if not self._step_graph.fits(needed):
            logger.warning(
                "NemotronVoiceChat talker: timeline of %d positions exceeds the CUDA-graph "
                "capacity; using the eager per-frame step for this request.",
                needed,
            )
            return False
        return self._step_graph.ensure_captured(expected_len)

    def _init_session(self, request_id: str, info: dict[str, Any], device: torch.device) -> dict[str, Any]:
        assert self.tts is not None
        timeline = self._timeline(info).to(device)
        if timeline.numel() < 2:
            raise ValueError(
                f"NemotronVoiceChat talker timeline for request {request_id} has "
                f"{timeline.numel()} positions; need at least 2 (prompt + one frame)."
            )
        prompt_len = self._logical_prompt_len(info)
        if prompt_len is None:
            raise ValueError(
                "NemotronVoiceChat talker request is missing its logical prompt length "
                "('nvc_logical_prompt_len' in additional_information, or async "
                "meta.num_processed_tokens); the code2wav producer cannot trim the "
                "prompt-region codes without it."
            )
        meta = info.get("meta")
        codec_streaming = scalar_bool(meta.get("codec_streaming")) if isinstance(meta, dict) else False
        codec_streaming = codec_streaming or scalar_bool(info.get("meta.codec_streaming"))
        codec_request_id = meta.get("request_id") if isinstance(meta, dict) else None
        # Warmup: pre-baked speaker latent (NeMo offline_inference lines around
        # get_init_inputs); the init prompt forward runs in the eager branch
        # below, or inside the CUDA-graph session's own StaticCache prefill.
        self.tts.set_init_inputs(speaker_name=self._speaker_name)
        guidance_enabled = self._guidance_enabled()
        session = {
            "timeline": timeline,
            "prompt_len": prompt_len,
            "step": 1,  # NeMo's loop starts at t=1
            "current_subword_mask": torch.ones(1, 1, device=device, dtype=torch.bool),
            "guidance_enabled": guidance_enabled,
            "codes_rows": [],
            "codec_streaming": codec_streaming,
            "codec_request_id": codec_request_id,
            # Sync mode ships the whole timeline up front via
            # nvc_text_timeline; async-chunk mode grows it across chunks and
            # the last chunk sets meta.finished.
            "sync_mode": info.get("nvc_text_timeline") is not None,
            "upstream_finished": self._upstream_finished(info) or info.get("nvc_text_timeline") is not None,
        }
        if session["sync_mode"]:
            expected_len: int | None = int(timeline.numel())
        else:
            expected_len = self._expected_total_tokens(info)
        if self._graph_session_usable(timeline, expected_len):
            self._step_graph.start_session(timeline, expected_len=expected_len)
            session["use_graph"] = True
        else:
            init_inputs = self.tts.get_init_inputs(B=1)
            generation_config = self.tts._get_generation_config(guidance_enabled)
            init_inputs.update({"use_cache": True, "past_key_values": None, "guidance_enabled": guidance_enabled})
            with torch.inference_mode():
                outputs = self.tts.tts_model(**init_inputs)
            session.update(
                {
                    "use_graph": False,
                    "past_key_values": outputs.past_key_values,
                    "code": init_inputs["code"][:, -1:],
                    "first_context_subword_id": init_inputs["subword_ids"][:, -1].unsqueeze(-1),
                    "tts_initialized": False,
                    "generation_config": generation_config,
                }
            )
        self._sessions[request_id] = session
        return session

    # ------------------------------------------------------------------
    # Native-vLLM path (experimental; see __init__).
    # ------------------------------------------------------------------
    def _encode_cas_rows(self, ids: torch.Tensor) -> torch.Tensor:
        """Batched CharAwareSubwordEncoder rows for timeline tokens ([n, H])."""
        model = self.tts.tts_model
        mask = torch.ones(1, ids.numel(), dtype=torch.bool, device=ids.device)
        with torch.inference_mode():
            return model.embed_subword(ids.unsqueeze(0), mask)[0]

    def _init_native_session(self, request_id: str, info: dict[str, Any], device: torch.device) -> dict[str, Any]:
        from vllm_omni.model_executor.models.nemotron_voicechat.talker_native import (
            build_prefill_embeds,
        )

        timeline = self._timeline(info).to(device)
        prompt_len = self._logical_prompt_len(info)
        if prompt_len is None:
            raise ValueError(
                "NemotronVoiceChat talker request is missing its logical prompt length "
                "('nvc_logical_prompt_len' or async meta.num_processed_tokens); the code2wav "
                "producer cannot trim the prompt-region codes without it."
            )
        meta = info.get("meta")
        codec_streaming = scalar_bool(meta.get("codec_streaming")) if isinstance(meta, dict) else False
        codec_streaming = codec_streaming or scalar_bool(info.get("meta.codec_streaming"))
        codec_request_id = meta.get("request_id") if isinstance(meta, dict) else None
        sync_mode = info.get("nvc_text_timeline") is not None
        if not sync_mode and not codec_streaming:
            raise NotImplementedError(
                "NemotronVoiceChat native talker supports the sync pipeline and duplex "
                "(resumable) serving; the offline async-chunk pipeline would free-run past "
                "the received timeline (non-resumable requests cannot park per segment) — "
                "use the captured-step fast_streaming yaml for offline streaming."
            )
        self.tts.set_init_inputs(speaker_name=self._speaker_name)
        init_inputs = self.tts.get_init_inputs(B=1)
        init_len = int(init_inputs["code"].shape[1])
        model = self.tts.tts_model
        with torch.inference_mode():
            prefill = build_prefill_embeds(
                model,
                code=init_inputs["code"],
                subword_ids=init_inputs["subword_ids"],
                subword_mask=init_inputs["subword_mask"],
                audio_mask=init_inputs["audio_mask"],
                audio_prompt_latent=init_inputs.get("audio_prompt_latent"),
                uncond=False,
            )[0]  # [init_len, H]
        uncond_stream = None
        if self._native_guidance:
            from vllm_omni.model_executor.models.nemotron_voicechat.talker_native import (
                UncondStepGraph,
                UncondStream,
            )

            with torch.inference_mode():
                uncond_prefill = build_prefill_embeds(
                    model,
                    code=init_inputs["code"],
                    subword_ids=init_inputs["subword_ids"],
                    subword_mask=init_inputs["subword_mask"],
                    audio_mask=init_inputs["audio_mask"],
                    audio_prompt_latent=init_inputs.get("audio_prompt_latent"),
                    uncond=True,
                ).to(self._dtype)
            # Captured StaticCache step when the graph is free (one session at
            # a time; sync sessions pick the smallest fitting bucket, duplex
            # takes the largest). Concurrent sessions step eagerly.
            if self._uncond_graph is None:
                self._uncond_graph = UncondStepGraph(
                    model,
                    hidden_size=int(uncond_prefill.shape[-1]),
                    dtype=self._dtype,
                    device=timeline.device,
                    max_cache_len=self._graph_max_cache_len,
                )
            total_positions = (init_len + int(timeline.numel()) + 8) if sync_mode else None
            if self._uncond_graph.start_session(request_id, uncond_prefill, total_positions):
                uncond_stream = self._uncond_graph
            else:
                uncond_stream = UncondStream(model)
                uncond_stream.prefill(uncond_prefill)
        session = {
            "use_native": True,
            "timeline": timeline,
            "cas_table": self._encode_cas_rows(timeline),
            "prompt_len": int(prompt_len),
            "init_len": init_len,
            "prefill_embeds": prefill,
            "step": 1,  # NeMo's loop starts at t=1
            "code": init_inputs["code"][:, -1:],
            # Kept for KV-recompute replay: step t=1's feedback code, before
            # session["code"] starts tracking the latest sampled frame.
            "initial_code": init_inputs["code"][:, -1:],
            # CFG: native_talker_guidance mirrors the stream on the vendored
            # HF backbone (UncondStream); otherwise a single conditional
            # stream (measured ASR-equivalent at guidance_scale=0.2; see yaml).
            "generation_config": self.tts._get_generation_config(self._native_guidance),
            "uncond_stream": uncond_stream,
            "uncond_hidden": None,
            # NeMo's inference_force_speech_silence_on_eos (default on): EOS
            # text frames feed the silence codec frame back instead of the
            # sampled codes.
            "silence_codes": (
                self.tts.codec_silence_tokens.reshape(1, 1, -1).to(device=timeline.device, dtype=torch.long)
                if bool(self.tts.cfg.get("inference_force_speech_silence_on_eos", True))
                else None
            ),
            "text_eos_id": int(self.tts.text_eos_id),
            "codes_rows": [],
            "pending_step": None,
            "codec_streaming": codec_streaming,
            "codec_request_id": codec_request_id,
            "sync_mode": info.get("nvc_text_timeline") is not None,
            "upstream_finished": self._upstream_finished(info) or info.get("nvc_text_timeline") is not None,
        }
        self._sessions[request_id] = session
        return session

    def _refresh_native_timeline(self, session: dict[str, Any], info: dict[str, Any], device: torch.device) -> None:
        """Adopt a longer timeline from the latest chunk; CAS-encode new rows."""
        latest = self._try_timeline(info)
        old_len = int(session["timeline"].numel())
        if latest is not None and latest.numel() > old_len:
            latest = latest.to(device)
            new_rows = self._encode_cas_rows(latest[old_len:].to(torch.long))
            session["timeline"] = latest
            session["cas_table"] = torch.cat([session["cas_table"], new_rows.to(session["cas_table"].dtype)], dim=0)
        if self._upstream_finished(info):
            session["upstream_finished"] = True

    def _native_preprocess(
        self,
        input_ids: torch.Tensor,
        info: dict[str, Any],
        device: torch.device,
        span: int,
        request_id: str,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        from vllm_omni.model_executor.models.nemotron_voicechat.talker_native import (
            build_decode_embeds,
        )

        # NOTE: resumable (duplex) segments report _omni_is_prefill=True on
        # EVERY wake (each appended placeholder extends the prompt), so the
        # branch is decided by the engine POSITION, never by the prefill flag:
        # positions [0, init_len) carry the speaker-prompt prefill embeddings,
        # every later position consumes exactly one timeline step.
        session = self._sessions.get(request_id)
        if session is None:
            session = self._init_native_session(request_id, info, device)
        else:
            self._refresh_native_timeline(session, info, device)
        offset = max(0, int(info.get("_omni_num_computed_tokens", 0) or 0))
        init_len = int(session["init_len"])
        step = int(session["step"])
        total = int(session["timeline"].numel())
        prefill_head: torch.Tensor | None = None
        if offset < init_len:
            # Speaker-prompt prefill region (possibly chunked across steps).
            if offset + span <= init_len:
                prefill: torch.Tensor = session["prefill_embeds"]
                return input_ids, prefill[offset : offset + span].to(self._dtype), {}
            # A span crossing the boundary is only legal while a KV-cache
            # recomputation replays already-generated positions in bulk; a
            # LIVE first timeline step sharing a chunk with the prefill means
            # a timeline chunk outpaced the speaker prefill.
            if step <= 1:
                raise ValueError(
                    f"NemotronVoiceChat native talker got a span [{offset}, {offset + span}) crossing "
                    f"the speaker-prompt boundary ({init_len}); the stage prompt must be exactly "
                    f"{init_len} placeholder tokens (set hf_overrides.talker_init_len / "
                    f"async_chunk_prewarm_prompt_len to {init_len} in the deploy yaml). A timeline "
                    "chunk extended this request before its speaker prefill finished — timeline "
                    "positions are strictly sequential and cannot share a step with the prefill."
                )
            prefill_head = session["prefill_embeds"][offset:init_len]
            span -= init_len - offset
            offset = init_len

        t_first = offset - init_len + 1
        t_last = t_first + span - 1
        if t_last > step:
            raise RuntimeError(
                f"NemotronVoiceChat native talker was scheduled timeline steps {t_first}..{t_last} but "
                f"the session is at step {step}; the per-frame code feedback is strictly sequential. "
                "The duplex producer drips one new timeline token per chunk — a position past the "
                "session step means chunks outpaced engine steps (raise the frame budget or "
                "investigate the engine stall)."
            )
        if t_last >= total:
            raise RuntimeError(f"NemotronVoiceChat native talker stepped past the timeline: step {t_last} of {total}.")
        replay_last = min(t_last, step - 1)
        if replay_last >= t_first and len(session["codes_rows"]) < replay_last - 1:
            raise RuntimeError(
                f"NemotronVoiceChat native talker cannot replay steps {t_first}..{replay_last}: only "
                f"{len(session['codes_rows'])} stored code rows."
            )
        replay_rows: list[torch.Tensor] = []
        if replay_last >= t_first:
            # KV-cache recomputation replays already-generated engine
            # positions.  Rebuild their decode embeds deterministically from
            # the stored code history: no resampling (pending_step is only set
            # for a live step) and the manual unconditional stream is not
            # advanced (it was never rewound).
            if not session.get("replaying"):
                session["replaying"] = True
                logger.warning(
                    "NemotronVoiceChat native talker replaying timeline steps %d..%d after KV recompute.",
                    t_first,
                    replay_last,
                )
            replay_rows = [self._native_replay_embeds(session, rt) for rt in range(t_first, replay_last + 1)]
        if t_last < step:
            # Pure replay chunk: no live position this step.
            embeds = torch.cat(replay_rows, dim=0)
            if prefill_head is not None:
                embeds = torch.cat([prefill_head.to(embeds.dtype), embeds], dim=0)
            return input_ids, embeds.to(self._dtype), {}
        session.pop("replaying", None)
        t = step
        model = self.tts.tts_model
        with torch.inference_mode():
            prev_codes = session["code"]
            if session["silence_codes"] is not None:
                # NeMo's inference_force_speech_silence_on_eos: on an EOS text
                # frame the code feedback becomes the silence frame, so the
                # turn ends in clean silence instead of a trailing artifact.
                # Tensor-level select — no host sync on the timeline token.
                is_eos = session["timeline"][t : t + 1].reshape(1, 1, 1) == session["text_eos_id"]
                prev_codes = torch.where(is_eos, session["silence_codes"].expand_as(prev_codes), prev_codes)
            embeds = build_decode_embeds(
                model,
                prev_codes=prev_codes,
                subword_embed=session["cas_table"][t : t + 1].unsqueeze(0),
                uncond=False,
            )[0]  # [1, H]
            if session["uncond_stream"] is not None:
                # Advance the unconditional stream in lockstep: same prev codes,
                # null conditioning. Its hidden row pairs with this step's
                # conditional row in make_omni_output.
                uncond_embed = build_decode_embeds(
                    model,
                    prev_codes=prev_codes,
                    subword_embed=session["cas_table"][t : t + 1].unsqueeze(0),
                    uncond=True,
                ).to(self._dtype)
                session["uncond_hidden"] = session["uncond_stream"].step(uncond_embed)
        session["pending_step"] = t
        embeds = embeds.to(self._dtype)
        if replay_rows:
            embeds = torch.cat([*replay_rows, embeds], dim=0)
        if prefill_head is not None:
            embeds = torch.cat([prefill_head.to(embeds.dtype), embeds], dim=0)
        return input_ids, embeds, {}

    def _native_replay_embeds(self, session: dict[str, Any], t: int) -> torch.Tensor:
        """Decode embeds for an already-generated step ``t`` (KV recompute).

        Reconstructs the exact feedback the original step saw: the codes
        sampled at step ``t-1`` (or the speaker-prompt code for ``t == 1``)
        with the same silence-on-EOS substitution.
        """
        from vllm_omni.model_executor.models.nemotron_voicechat.talker_native import (
            build_decode_embeds,
        )

        model = self.tts.tts_model
        with torch.inference_mode():
            template: torch.Tensor = session["code"]
            if t == 1:
                prev_codes = session["initial_code"]
            else:
                prev_codes = session["codes_rows"][t - 2].reshape(template.shape).to(dtype=template.dtype)
            if session["silence_codes"] is not None:
                is_eos = session["timeline"][t : t + 1].reshape(1, 1, 1) == session["text_eos_id"]
                prev_codes = torch.where(is_eos, session["silence_codes"].expand_as(prev_codes), prev_codes)
            embeds = build_decode_embeds(
                model,
                prev_codes=prev_codes,
                subword_embed=session["cas_table"][t : t + 1].unsqueeze(0),
                uncond=False,
            )[0]
        return embeds.to(self._dtype)

    def _native_make_omni_output(self, hidden: torch.Tensor, **kwargs: Any) -> OmniOutput:
        infos = kwargs.get("model_intermediate_buffer") or kwargs.get("runtime_additional_information") or []
        spans = kwargs.get("request_token_spans")
        flag = torch.zeros(hidden.shape[0], 2, dtype=torch.float32, device=hidden.device)
        # Batch-aligned per-request lists: the generic mm splitter
        # (to_payload_element) hands list element ``idx`` to request ``idx``
        # but clones a shared tensor to EVERY request, so a single shared
        # payload would leak the last request's codes into the others'
        # streams.  Requests with nothing to ship this step keep placeholder
        # entries.
        empty_codes = torch.empty(0, dtype=torch.long)
        audio_list: list[torch.Tensor] = [empty_codes for _ in infos]
        prompt_len_list: list[torch.Tensor] = [torch.zeros(1, dtype=torch.long) for _ in infos]
        streaming_list: list[torch.Tensor] = [torch.zeros(1, dtype=torch.bool) for _ in infos]
        emitted = False

        def _emit(index: int, session_state: dict[str, Any]) -> None:
            nonlocal emitted
            audio_list[index] = torch.cat(session_state["codes_rows"], dim=0)
            # 1-D tensors on purpose: the wire-payload builder indexes
            # element.shape[0] (see the eager make_omni_output).
            prompt_len_list[index] = torch.tensor([session_state["prompt_len"]], dtype=torch.long)
            streaming_list[index] = torch.tensor([bool(session_state["codec_streaming"])], dtype=torch.bool)
            emitted = True

        for i, info in enumerate(infos):
            if not isinstance(info, dict):
                continue
            request_id = info.get("request_id")
            session = self._sessions.get(str(request_id)) if request_id is not None else None
            if not session or not session.get("use_native"):
                continue
            pending = session.get("pending_step")
            if pending is None:
                # Prefill forward: no new codes this step, but keep the
                # request's cumulative history in its own slot so the
                # splitter never falls back to a shared tensor.
                if session.get("codes_rows"):
                    _emit(i, session)
                continue
            session["pending_step"] = None
            if spans is None:
                raise RuntimeError("NemotronVoiceChat native talker needs request_token_spans from the runner.")
            start, end = spans[i]
            model = self.tts.tts_model
            if session["uncond_stream"] is not None:
                # CFG: pair the conditional row with this frame's uncond row
                # ([cond; uncond] order — generate_step chunks in that order)
                # and let generate_step apply NeMo's hidden-space blends
                # (lm_head EOS blend + the post-MLP blend inside MoGHead.infer).
                uncond_hidden = session["uncond_hidden"]
                if uncond_hidden is None:
                    raise RuntimeError(
                        "NemotronVoiceChat native talker CFG desynchronized: no unconditional "
                        "hidden row for this frame (preprocess/make_omni_output mismatch)."
                    )
                session["uncond_hidden"] = None
                pair = torch.cat(
                    [hidden[end - 1 : end].reshape(1, 1, -1), uncond_hidden.reshape(1, 1, -1)],
                    dim=0,
                ).to(self._dtype)
                if self._mog_graph is None:
                    from vllm_omni.model_executor.models.nemotron_voicechat.talker_native import (
                        MoGStepGraph,
                    )

                    self._mog_graph = MoGStepGraph(model, session["generation_config"], batch=2)
                codes = self._mog_graph.run(pair)
            else:
                if self._mog_graph is None:
                    from vllm_omni.model_executor.models.nemotron_voicechat.talker_native import (
                        MoGStepGraph,
                    )

                    self._mog_graph = MoGStepGraph(model, session["generation_config"], batch=1)
                codes = self._mog_graph.run(hidden[end - 1 : end].reshape(1, 1, -1))
            session["code"] = codes
            session["codes_rows"].append(codes.reshape(1, -1).to(torch.long))
            session["step"] = int(pending) + 1
            # Duplex (codec_streaming) sessions stop per segment when the
            # received timeline is exhausted and resume with KV intact when
            # the next thinker frame arrives; sync sessions stop at the end.
            finished = int(session["step"]) >= int(session["timeline"].numel()) and self._segment_stops_at_exhaustion(
                session
            )
            if finished:
                flag[start:end, 0] = 1.0
            _emit(i, session)
            # Belt and braces: also mirror into the live per-request buffer so
            # the flush-time full-payload producer can read it from
            # additional_information as well as from the wire payload.
            info["codes"] = {"audio": audio_list[i]}
            info["meta"] = {
                "nvc_tts_step": int(session["step"]),
                "nvc_logical_prompt_len": int(session["prompt_len"]),
                "codec_streaming": bool(session["codec_streaming"]),
                "request_id": session["codec_request_id"],
            }
        outputs: dict[str, Any] = {}
        if emitted:
            outputs = {
                "codes": {"audio": audio_list},
                "meta": {
                    "nvc_logical_prompt_len": prompt_len_list,
                    "codec_streaming": streaming_list,
                },
            }
        return OmniOutput(text_hidden_states=flag, multimodal_outputs=outputs)

    def _refresh_session_timeline(self, session: dict[str, Any], info: dict[str, Any], device: torch.device) -> None:
        """Adopt a longer timeline from the latest async-chunk payload."""
        latest = self._try_timeline(info)
        if latest is not None and latest.numel() > int(session["timeline"].numel()):
            session["timeline"] = latest.to(device)
            if session.get("use_graph"):
                self._step_graph.extend_timeline(session["timeline"])
        if self._upstream_finished(info):
            session["upstream_finished"] = True

    def _guidance_enabled(self) -> bool:
        return bool(getattr(self.config, "tts_cfg", {}).get("inference_guidance_enabled", True))

    def _step_session(self, session: dict[str, Any]) -> tuple[torch.Tensor, bool]:
        """One NeMo TTS step; returns (codes_row [1, Q], finished)."""
        assert self.tts is not None
        timeline: torch.Tensor = session["timeline"]
        t = int(session["step"])
        total = int(timeline.numel())
        if t >= total:
            # The drain loop in preprocess never enters here; a direct call
            # past the received timeline is a programming error. Hard-fail
            # rather than fabricating a frame — a repeated/guessed subword
            # would silently desync the audio from the text channel.
            raise RuntimeError(
                f"NemotronVoiceChat talker stepped past the received timeline: step {t} "
                f"needs timeline position {t} but only {total} positions have arrived "
                f"(upstream_finished={session['upstream_finished']})."
            )
        current_subword_id = timeline[t].reshape(1, 1)
        if not session["tts_initialized"]:
            prev_subword_id = session["first_context_subword_id"]
            session["tts_initialized"] = True
        else:
            prev_subword_id = timeline[t - 1].reshape(1, 1)
        with torch.inference_mode():
            code, past_key_values = self.tts.infer_codes_one_step(
                current_subword_id=current_subword_id,
                prev_subword_id=prev_subword_id,
                current_subword_mask=session["current_subword_mask"],
                prev_audio_tokens=session["code"],
                past_key_values=session["past_key_values"],
                guidance_enabled=session["guidance_enabled"],
                generation_config=session["generation_config"],
                ignore_eos_flag_stop=True,
            )
        session["code"] = code
        session["past_key_values"] = past_key_values
        session["step"] = t + 1
        finished = (t + 1) >= int(session["timeline"].numel()) and self._segment_stops_at_exhaustion(session)
        return code.reshape(1, -1).to(torch.long), finished

    @staticmethod
    def _segment_stops_at_exhaustion(session: dict[str, Any]) -> bool:
        """Whether exhausting the received timeline should emit the stop token.

        Duplex serving submits RESUMABLE talker requests and marks them with
        ``codec_streaming``: exhausting the timeline ends the current scheduler
        segment, and the parked request resumes (TTS state intact) when the
        next thinker frame arrives. Offline requests are NOT resumable — a
        stop token would end them permanently — so mid-stream exhaustion emits
        CONTINUE instead, and the stop waits for the upstream's final chunk
        (``meta.finished``; sync mode ships the whole timeline up front, so it
        counts as finished from the start)."""
        return bool(session["codec_streaming"]) or bool(session["upstream_finished"])

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        info = merge_runtime_info(info_dict)
        device = input_ids.device
        span = int(input_ids.shape[0])
        request_id = require_request_id(info, "talker")

        if self._use_native_backbone:
            return self._native_preprocess(input_ids, info, device, span, request_id)

        embeds = torch.zeros((span, self._hidden), device=device, dtype=self._dtype)
        if request_id not in self._sessions:
            # The request id, not vLLM's per-segment prefill flag, is the
            # session boundary. Resumable updates are prefill again in v0.28
            # and must preserve the existing EAR-TTS cache and step.
            self._init_session(request_id, info, device)
            return input_ids, embeds, {}

        session = self._sessions[request_id]
        self._refresh_session_timeline(session, info, device)
        # DRAIN every received timeline position on each wake, in both modes.
        # The engine tokens are placeholders (CONTINUE/STOP) and the codes ride
        # the cumulative payload, so multiple TTS steps per engine step are
        # safe — and necessary:
        #   * sync mode ships the WHOLE timeline up front, so the first decode
        #     wake runs every TTS step back-to-back instead of paying a full
        #     engine round trip (scheduler + sampler + IPC) per 80 ms frame;
        #     the codes are computed in the same order as the stepwise loop,
        #     so the output is bit-identical,
        #   * the save_async background thread can coalesce thinker steps into
        #     one chunk (fewer wakes than new positions), and
        #   * the prompt-region PAD steps (logical_prompt_len - 1 of them) all
        #     become available with chunk 0 and must not cost one upstream
        #     chunk each.
        # Rows stay on the GPU (tiny [1, Q] longs): a per-frame .cpu() would
        # stall the dispatch pipeline once per frame; the stage producers move
        # the final payload to CPU when shipping.
        finished = False
        steps_run = 0
        total = int(session["timeline"].numel())
        if session.get("use_graph"):
            steps_run = total - int(session["step"])
            if steps_run > 0:
                self._step_graph.run_steps(steps_run)
                session["step"] = total
                finished = self._segment_stops_at_exhaustion(session)
        else:
            while int(session["step"]) < total:
                codes_row, finished = self._step_session(session)
                session["codes_rows"].append(codes_row)
                steps_run += 1
        if steps_run == 0:
            # Zero-progress wake (duplicate/terminal-marker chunk): no TTS step
            # was run, so never fabricate a frame. Stop only when this session
            # may end at exhaustion (see _segment_stops_at_exhaustion);
            # otherwise emit CONTINUE and wait for more upstream input.
            finished = int(session["step"]) >= int(session["timeline"].numel()) and self._segment_stops_at_exhaustion(
                session
            )
            if finished:
                embeds[:, 0] = 1.0
            return input_ids, embeds, {}
        # The code2wav producers own the suffix slicing and therefore receive
        # the cumulative history on every wake. Rows stay on the GPU (tiny
        # [1, Q] longs; a per-step .cpu() would stall dispatch once per frame);
        # the producers move the shipped payload to CPU.
        if session.get("use_graph"):
            cumulative = self._step_graph.codes_rows(1, int(session["step"]))
        else:
            cumulative = torch.cat(session["codes_rows"], dim=0)
        if finished:
            embeds[:, 0] = 1.0  # stop flag -> compute_logits emits the stop token
        info_update: dict[str, Any] = {
            "codes": {"audio": cumulative},
            # nvc_logical_prompt_len rides EVERY step's payload: the request's
            # additional_information is overwritten both by arriving thinker
            # chunks and by these model updates, so the async code2wav producer
            # reads the prompt length from the per-step multimodal_output
            # instead of racing the request-level dict.
            "meta": {
                "nvc_tts_step": int(session["step"]),
                "nvc_logical_prompt_len": int(session["prompt_len"]),
                "codec_streaming": bool(session["codec_streaming"]),
                "request_id": session["codec_request_id"],
            },
        }
        return input_ids, embeds, info_update

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        for request_id in finished_req_ids:
            self._sessions.pop(str(request_id), None)
            if self._uncond_graph is not None:
                self._uncond_graph.release(str(request_id))

    # ------------------------------------------------------------------
    # Weights.
    # ------------------------------------------------------------------
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Build the vendored DuplexEARTTS and load the ``tts_model.*`` subtree.

        The backbone/MoG run in the engine dtype (bf16 by default); the codec
        submodule inside DuplexEARTTS stays fp32 (it is not used by this stage —
        decode happens in code2wav — but keeping its dtype canonical avoids
        surprises if warmup paths touch it).
        """
        from vllm_omni.model_executor.models.nemotron_voicechat.nemo_vendored.duplex_ear_tts import (
            DuplexEARTTS,
        )

        tts_model_cfg = dict(getattr(self.config, "tts_cfg", {}) or {})
        if not tts_model_cfg:
            raise ValueError(
                "NemotronVoiceChat checkpoint config lacks 'model.speech_generation.model'; cannot build the talker."
            )
        tts_model_cfg = sanitize_tts_model_cfg(tts_model_cfg)
        tts_data = dict(getattr(self.config, "tts_data", {}) or {})
        tts_data.setdefault("source_sample_rate", 22050)
        tts_data.setdefault("target_sample_rate", int(getattr(self.config, "target_sample_rate", 22050)))
        tts_data.setdefault("frame_length", float(getattr(self.config, "frame_length", 0.08)))
        # DuplexEARTTS consumes the whole speech_generation section layout
        # ({"data": ..., "model": ...}), matching NeMo's constructor call.
        tts = DuplexEARTTS({"data": tts_data, "model": tts_model_cfg})

        prefix = "tts_model."
        backbone_prefix = "tts_model.tts_model.backbone."
        state: dict[str, torch.Tensor] = {}
        backbone_weights: list[tuple[str, torch.Tensor]] = []
        for name, tensor in weights:
            if not name.startswith(prefix):
                continue
            if self._use_native_backbone and name.startswith(backbone_prefix):
                # The backbone runs as a native vLLM Gemma3 model instead.
                backbone_weights.append((name, tensor))
                if not self._native_guidance:
                    continue
                # CFG keeps a second copy in the vendored module: the
                # unconditional stream decodes on the HF backbone.
            target = name.removeprefix(prefix)
            if target.startswith("audio_codec."):
                state[target] = tensor.to(dtype=torch.float32)
            else:
                state[target] = tensor.to(dtype=self._dtype)
        if self._use_native_backbone:
            from vllm_omni.model_executor.models.nemotron_voicechat.talker_native import (
                iter_backbone_weights,
            )

            loaded = self.backbone.load_weights(
                iter_backbone_weights(backbone_weights, hidden_size=self._hidden, dtype=self._dtype)
            )
            logger.info("NemotronVoiceChat native talker backbone loaded %d vLLM tensors", len(loaded))
        if not state:
            raise ValueError(
                "No 'tts_model.*' tensors found in the checkpoint; the NemotronVoiceChat "
                "unified model.safetensors is required for the talker stage."
            )
        if self._use_native_backbone and not self._native_guidance:
            # The HF backbone inside the vendored model is unused on the native
            # path (only the embedding/CAS/MoG modules are consulted); drop it
            # before loading so its ~2.4 GiB of fp32 parameters are freed and
            # its (rerouted) weights are not reported missing. With CFG the
            # backbone stays: the unconditional stream decodes on it.
            tts.tts_model.backbone = nn.Identity()
        missing, unexpected = tts.load_state_dict(state, strict=False)
        real_missing = [m for m in missing if not m.startswith("audio_codec.")]
        if self._use_native_backbone and not self._native_guidance:
            real_missing = [m for m in real_missing if not m.startswith("tts_model.backbone.")]
        if real_missing:
            raise ValueError(f"NemotronVoiceChat talker is missing TTS weights: {sorted(real_missing)[:10]} ...")
        if unexpected:
            logger.warning(
                "NemotronVoiceChat talker ignoring %d unexpected TTS tensors (e.g. %s)",
                len(unexpected),
                sorted(unexpected)[:5],
            )
        device = self.vllm_config.device_config.device
        tts = tts.to(device=device)
        tts.tts_model.to(dtype=self._dtype)
        if hasattr(tts, "audio_codec") and tts.audio_codec is not None:
            tts.audio_codec.to(dtype=torch.float32)
        self.tts = tts.eval()
        logger.info(
            "NemotronVoiceChat talker loaded EAR-TTS: %d tensors, speaker=%s, guidance=%s",
            len(state),
            self._speaker_name,
            self._guidance_enabled(),
        )
        if self._use_cuda_graphs:
            from vllm_omni.model_executor.models.nemotron_voicechat.talker_graph import (
                TalkerStepGraph,
            )

            step_graph = TalkerStepGraph(
                self.tts,
                dtype=self._dtype,
                device=torch.device(device) if not isinstance(device, torch.device) else device,
                max_cache_len=self._graph_max_cache_len,
                guidance_enabled=self._guidance_enabled(),
            )
            supported, reason = step_graph.config_supported()
            if supported:
                self._step_graph = step_graph
            else:
                logger.warning(
                    "NemotronVoiceChat talker CUDA graphs requested but unsupported (%s); "
                    "using the eager per-frame step.",
                    reason,
                )
        if self._use_native_backbone:
            # Capture the per-frame MoG sampling graph at load time; a lazy
            # first-frame capture would otherwise cost the first session a
            # few hundred ms of first-packet latency.
            from vllm_omni.model_executor.models.nemotron_voicechat.talker_native import (
                MoGStepGraph,
            )

            mog_batch = 2 if self._native_guidance else 1
            self._mog_graph = MoGStepGraph(
                self.tts.tts_model,
                self.tts._get_generation_config(self._native_guidance),
                batch=mog_batch,
            )
            self._mog_graph.ensure_captured(torch.zeros(mog_batch, 1, self._hidden, dtype=self._dtype, device=device))
        return {name for name, _ in self.named_parameters()}
