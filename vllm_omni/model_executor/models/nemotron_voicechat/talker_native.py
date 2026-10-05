# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native-vLLM building blocks for the NemotronVoiceChat talker.

The eager/CUDA-graph talker executes the vendored HF Gemma3 backbone inside the
per-request ``preprocess`` hook, next to (not inside) the vLLM engine. The
native path (``hf_overrides.use_native_talker``) instead runs the backbone as a
real vLLM model — PagedAttention KV cache (capacity and per-step attention cost
scale with the ACTUAL session length, no StaticCache buckets), vLLM's own decode
CUDA graphs, and, later, multi-session batching. This mirrors NVIDIA's own
deployment, whose EAR-TTS is served by a vLLM engine.

This module hosts the pieces that are independent of the talker class:

* ``synthesize_backbone_config``: the checkpoint stores the backbone as a NeMo
  ``tts_config.backbone_config`` dict; vLLM needs a real ``Gemma3TextConfig``.
  The embedding table is never used (the fused embeddings are built from
  codes + text conditioning), so a tiny dummy vocab keeps it negligible.
* ``iter_backbone_weights``: remaps the checkpoint's
  ``tts_model.tts_model.backbone.*`` HF-Gemma3 tensors onto the names vLLM's
  ``Gemma3ForCausalLM`` loader expects, and fabricates the dummy
  ``model.embed_tokens.weight``.
* ``build_prefill_embeds`` / ``build_decode_embeds``: the per-token fused input
  embeddings, numerically verified against the vendored
  ``RVQEARTTSModel.forward`` embedding section for both CFG roles (the
  speaker-prompt prefill must use the checkpoint's pre-baked
  ``audio_prompt_latent``, not the frozen projection).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F

_DUMMY_VOCAB_SIZE = 8


def synthesize_backbone_config(tts_config: dict[str, Any]) -> Any:
    """Build a ``Gemma3TextConfig`` from the checkpoint's NeMo backbone dict."""
    from transformers import AutoConfig

    backbone_type = tts_config.get("backbone_type")
    if backbone_type != "gemma3_text":
        raise NotImplementedError(
            f"NemotronVoiceChat native talker supports a gemma3_text backbone only (got {backbone_type!r})."
        )
    backbone_cfg = dict(tts_config.get("backbone_config") or {})
    backbone_cfg.pop("use_cache", None)
    config = AutoConfig.for_model(
        backbone_type,
        **backbone_cfg,
        vocab_size=_DUMMY_VOCAB_SIZE,
        tie_word_embeddings=True,
    )
    return config


def iter_backbone_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    hidden_size: int,
    dtype: torch.dtype,
) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield vLLM-Gemma3 weights from the unified checkpoint stream.

    Checkpoint names: ``tts_model.tts_model.backbone.<hf gemma3 names>``.
    vLLM ``Gemma3ForCausalLM`` expects ``model.<hf gemma3 names>`` plus an
    embedding table (tied lm_head); the embedding is never consulted (fused
    embeddings are injected via ``inputs_embeds``), so a dummy is fabricated.
    """
    prefix = "tts_model.tts_model.backbone."
    for name, tensor in weights:
        if name.startswith(prefix):
            yield "model." + name.removeprefix(prefix), tensor.to(dtype=dtype)
    yield (
        "model.embed_tokens.weight",
        torch.zeros(_DUMMY_VOCAB_SIZE, hidden_size, dtype=dtype),
    )


def build_prefill_embeds(
    model: Any,
    *,
    code: torch.Tensor,
    subword_ids: torch.Tensor,
    subword_mask: torch.Tensor,
    audio_mask: torch.Tensor,
    audio_prompt_latent: torch.Tensor | None,
    uncond: bool,
) -> torch.Tensor:
    """Speaker-prompt prefill embeddings for one CFG role.

    Mirrors the ``audio_mask`` inference branch of the vendored
    ``RVQEARTTSModel.forward`` (verified to ~4e-6 max abs diff against the
    embeddings the vendored forward feeds its backbone). ``model`` is the
    vendored ``RVQEARTTSModel``. Returns ``[1, T, H]``.
    """
    shifted_code = F.pad(code[:, :-1], [0, 0, 1, 0])
    code_embed = model.depthsum_embedding(shifted_code)
    bos_mask = (audio_mask & (~F.pad(audio_mask[:, :-1], [1, 0]))).unsqueeze(-1)
    pre_bos_mask = bos_mask.cumsum(dim=1) == 0
    code_embed = model.embed_code(code_embed)
    if model.config.get("use_audio_prompt_frozen_projection", False):
        if audio_prompt_latent is None:
            weight = model.audio_prompt_projection_W.to(code_embed.device, code_embed.dtype)
            audio_prompt_latent = torch.nn.functional.linear(code_embed, weight.T)
        code_embed = torch.where(pre_bos_mask, audio_prompt_latent, code_embed)
    code_embeds = code_embed + bos_mask * model.bos_emb
    flag = torch.full((code.size(0), 1, 1), bool(uncond), dtype=torch.bool, device=code.device)
    cond = model._prepare_conditioning(None, subword_ids, subword_mask, flag)
    if model.config.use_gated_fusion_for_text_audio:
        return model.gated_fusion_audio_text(code_embeds, cond)
    return code_embeds + cond


class MoGStepGraph:
    """One captured CUDA graph for the per-frame MoG sampling step.

    On the native path the backbone runs under vLLM's own decode CUDA graphs,
    but the MoG iterative unmasking (``RVQEARTTSModel.generate_step``: 5
    effective MoG-head passes + a 31-level depthsum quantization loop) would
    otherwise run eagerly in ``make_omni_output`` — ~1000 kernel launches for
    ~2 ms of GPU work per 80 ms frame. All shapes are static and the unmask
    schedule is Python ints, so the whole step captures into one graph whose
    replay is a single launch. MoG noise draws from the CUDA-graph-safe Philox
    stream (fresh randomness per replay, not bit-identical to eager).

    ``batch`` is the leading hidden dim (1 for the single conditional stream;
    2 once paired-CFG lands). Capture failures fall back to eager permanently.
    """

    def __init__(self, model: Any, generation_config: dict[str, Any], *, batch: int = 1) -> None:
        self.model = model
        self.generation_config = dict(generation_config)
        self.batch = int(batch)
        self._graph: torch.cuda.CUDAGraph | None = None
        self._broken = False
        self._hidden_buf: torch.Tensor | None = None
        self._code_buf: torch.Tensor | None = None

    def _step(self) -> None:
        codes, _, _ = self.model.generate_step(self._hidden_buf, ignore_eos_flag_stop=True, **self.generation_config)
        self._code_buf.copy_(codes)

    def ensure_captured(self, hidden: torch.Tensor) -> bool:
        if self._graph is not None:
            return True
        if self._broken:
            return False
        try:
            with torch.inference_mode():
                self._hidden_buf = hidden.clone()
                # Under CFG (guidance_scale set) generate_step chunks the
                # [cond; uncond] pair and returns codes for the effective
                # batch only.
                out_batch = self.batch
                if self.generation_config.get("guidance_scale") is not None:
                    out_batch = max(self.batch // 2, 1)
                self._code_buf = torch.zeros(
                    out_batch, 1, int(self.model.config.num_quantizers), dtype=torch.long, device=hidden.device
                )
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):
                        self._step()
                torch.cuda.current_stream().wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self._step()
                self._graph = graph
            return True
        except Exception:
            self._broken = True
            self._graph = None
            import logging

            logging.getLogger(__name__).exception(
                "NemotronVoiceChat native talker: MoG step graph capture failed; sampling eagerly."
            )
            return False

    def run(self, hidden: torch.Tensor) -> torch.Tensor:
        """MoG-sample one frame from ``hidden`` ([batch, 1, H]) -> codes [batch, 1, Q]."""
        if self.ensure_captured(hidden):
            with torch.inference_mode():
                self._hidden_buf.copy_(hidden)
                self._graph.replay()
                return self._code_buf.clone()
        with torch.inference_mode():
            codes, _, _ = self.model.generate_step(hidden, ignore_eos_flag_stop=True, **self.generation_config)
            return codes


def build_decode_embeds(
    model: Any,
    *,
    prev_codes: torch.Tensor,
    subword_embed: torch.Tensor,
    uncond: bool,
) -> torch.Tensor:
    """One decode-step fused embedding for one CFG role (``[1, 1, H]``).

    ``subword_embed`` is the precomputed CharAwareSubwordEncoder row for the
    current timeline token (host-sync free; see the CAS table precompute).
    """
    code_embeds = model.embed_code(model.depthsum_embedding(prev_codes))
    flag = torch.full((prev_codes.size(0), 1, 1), bool(uncond), dtype=torch.bool, device=prev_codes.device)
    cond = model._prepare_conditioning(None, None, None, flag, subword_embeds=subword_embed)
    if model.config.use_gated_fusion_for_text_audio:
        return model.gated_fusion_audio_text(code_embeds, cond)
    return code_embeds + cond


class UncondStream:
    """The unconditional CFG stream, run on the vendored HF Gemma3 backbone.

    NeMo applies classifier-free guidance by duplicating every backbone input
    into a [cond; uncond] batch (``RVQEARTTSModel.forward``) whose second half
    replaces the whole conditioning sum with ``null_emb``, then blending the
    two hidden rows inside ``generate_step`` (lm_head EOS blend and the
    post-MLP blend inside ``MoGHead.infer``).

    On the native path the conditional stream's KV lives inside vLLM's paged
    cache, where a second same-length sequence cannot ride along without
    scheduler-level pair locking. This class instead mirrors the stream
    position-for-position on the vendored HF backbone with its own
    ``DynamicCache`` — one extra [1, 1, H] decode step per frame. Both streams
    are fed the SAME sampled code row each frame (prev-codes conditioning is
    shared in NeMo's layout; only the text conditioning differs).
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self.cache: Any = None
        self.pos = 0

    @torch.inference_mode()
    def prefill(self, embeds: torch.Tensor) -> None:
        """Consume the speaker-prompt prefill embeddings (``[1, T, H]``)."""
        from transformers import DynamicCache

        self.cache = DynamicCache()
        length = int(embeds.shape[1])
        position_ids = torch.arange(length, device=embeds.device).unsqueeze(0)
        outputs = self.model.backbone(
            inputs_embeds=embeds,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=self.cache,
            use_cache=True,
        )
        self.cache = outputs.past_key_values
        self.pos = length

    @torch.inference_mode()
    def step(self, embed: torch.Tensor) -> torch.Tensor:
        """One decode step (``[1, 1, H]`` in) -> uncond hidden (``[1, 1, H]``)."""
        if self.cache is None:
            raise RuntimeError("UncondStream.step called before prefill")
        position_ids = torch.tensor([[self.pos]], device=embed.device)
        outputs = self.model.backbone(
            inputs_embeds=embed,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=self.cache,
            use_cache=True,
        )
        self.cache = outputs.past_key_values
        self.pos += 1
        return outputs.last_hidden_state


class UncondStepGraph:
    """Captured per-frame decode step for the unconditional CFG stream.

    Backbone-only StaticCache stepping with the same capture hygiene as
    ``talker_graph.TalkerStepGraph``: a prepared 4D boolean attention mask
    (triggers the HF early-exit so no python-int cache length is baked into
    the graph), explicit ``position_ids``/``cache_position`` tensors, and the
    in-graph slot-open ``index_fill_``. One session owns the graph at a time
    (StaticCache addresses are capture-bound); concurrent sessions fall back
    to the eager ``UncondStream``.
    """

    _BUCKETS = (512, 1024, 2048, 4096)

    def __init__(
        self,
        model: Any,
        *,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
        max_cache_len: int = 4096,
    ) -> None:
        self.model = model
        self.dtype = dtype
        self.device = device
        self.max_cache_len = int(max_cache_len)
        self._buckets: dict[int, dict[str, Any]] = {}
        self._broken = False
        self.owner: str | None = None
        self._embed_buf = torch.zeros(1, 1, hidden_size, dtype=dtype, device=device)
        self._hidden_buf = torch.zeros(1, 1, hidden_size, dtype=dtype, device=device)
        self._cache_pos = torch.zeros(1, dtype=torch.long, device=device)
        self._attn_mask = torch.zeros(1, 1, 1, self.max_cache_len, dtype=torch.bool, device=device)
        self._cache: Any = None
        self._active_bucket = 0

    def bucket_for(self, total_positions: int | None) -> int | None:
        """Smallest bucket holding ``total_positions``; None if it cannot fit."""
        for bucket in self._BUCKETS:
            if bucket > self.max_cache_len:
                break
            if total_positions is not None and total_positions <= bucket:
                return bucket
        largest = max(b for b in self._BUCKETS if b <= self.max_cache_len)
        return largest if total_positions is None or total_positions <= largest else None

    def _get_bucket(self, bucket_len: int) -> dict[str, Any]:
        state = self._buckets.get(bucket_len)
        if state is None:
            from transformers import StaticCache

            state = {
                "cache": StaticCache(
                    config=self.model.backbone.config,
                    max_batch_size=1,
                    max_cache_len=bucket_len,
                    device=self.device,
                    dtype=self.dtype,
                ),
                "graph": None,
            }
            self._buckets[bucket_len] = state
        return state

    def _one_step(self) -> None:
        self._attn_mask.index_fill_(3, self._cache_pos, True)
        outputs = self.model.backbone(
            inputs_embeds=self._embed_buf,
            attention_mask=self._attn_mask[..., : self._active_bucket],
            position_ids=self._cache_pos.reshape(1, 1),
            past_key_values=self._cache,
            use_cache=True,
            cache_position=self._cache_pos,
        )
        self._hidden_buf.copy_(outputs.last_hidden_state)
        self._cache_pos.add_(1)

    def start_session(self, request_id: str, prefill_embeds: torch.Tensor, total_positions: int | None) -> bool:
        """Claim the graph for one session and prefill its cache. False -> use eager."""
        if self._broken or self.owner is not None:
            return False
        bucket_len = self.bucket_for(total_positions)
        if bucket_len is None:
            return False
        try:
            with torch.inference_mode():
                state = self._get_bucket(bucket_len)
                self._cache = state["cache"]
                self._active_bucket = bucket_len
                self._cache.reset()
                init_len = int(prefill_embeds.shape[1])
                self.model.backbone(
                    inputs_embeds=prefill_embeds.to(self.dtype),
                    position_ids=torch.arange(init_len, device=self.device).unsqueeze(0),
                    past_key_values=self._cache,
                    use_cache=True,
                    cache_position=torch.arange(init_len, device=self.device),
                )
                self._attn_mask.zero_()
                self._attn_mask[..., :init_len] = True
                self._cache_pos.fill_(init_len)
                if state["graph"] is None:
                    side = torch.cuda.Stream()
                    side.wait_stream(torch.cuda.current_stream())
                    saved_pos = int(self._cache_pos.item())
                    with torch.cuda.stream(side):
                        for _ in range(3):
                            self._one_step()
                    torch.cuda.current_stream().wait_stream(side)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        self._one_step()
                    state["graph"] = graph
                    # Rewind the warmup/capture side effects: re-prefill so the
                    # cache holds only the speaker prompt again.
                    self._cache.reset()
                    self.model.backbone(
                        inputs_embeds=prefill_embeds.to(self.dtype),
                        position_ids=torch.arange(init_len, device=self.device).unsqueeze(0),
                        past_key_values=self._cache,
                        use_cache=True,
                        cache_position=torch.arange(init_len, device=self.device),
                    )
                    self._attn_mask.zero_()
                    self._attn_mask[..., :init_len] = True
                    self._cache_pos.fill_(saved_pos)
            self.owner = request_id
            return True
        except Exception:
            self._broken = True
            import logging

            logging.getLogger(__name__).exception(
                "NemotronVoiceChat native talker: uncond step graph capture failed; stepping eagerly."
            )
            return False

    def step(self, embed: torch.Tensor) -> torch.Tensor:
        """One captured decode step: embed [1,1,H] -> uncond hidden [1,1,H]."""
        if int(self._cache_pos.item()) >= self._active_bucket:
            raise RuntimeError(
                f"NemotronVoiceChat uncond CFG stream exceeded its StaticCache bucket "
                f"({self._active_bucket} positions)."
            )
        with torch.inference_mode():
            self._embed_buf.copy_(embed.to(self.dtype))
            self._buckets[self._active_bucket]["graph"].replay()
            return self._hidden_buf.clone()

    def release(self, request_id: str) -> None:
        if self.owner == request_id:
            self.owner = None
