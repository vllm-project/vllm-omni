# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA-graph fast path for the NemotronVoiceChat talker per-frame step.

The eager talker step is launch-bound: one 80 ms frame costs ~3200 kernel
launches (28-layer Gemma3 backbone + 5 MoG unmasking passes + a 31-level
depthsum loop) for ~9 ms of actual GPU work, so the eager wall time is
~25-34 ms/frame. This module captures the WHOLE step into one CUDA graph so
the drain loop degenerates to ``graph.replay()`` per frame (~12 ms/frame,
GPU-bound). This mirrors NVIDIA's own deployment of this model, which serves
EAR-TTS through vLLM decode CUDA graphs with CFG baked into the capture.

What makes the step capturable:

* ``StaticCache`` (preallocated KV, write index via ``cache_position``)
  instead of a growing HF ``DynamicCache``;
* the CharAwareSubwordEncoder — dynamic char-length shapes plus per-call
  ``.cpu().tolist()`` / ``subword_mask.any()`` host syncs — is hoisted OUT of
  the step: the frame-locked text timeline is known ahead of the audio, so its
  subword embeddings are precomputed in one batched call per session and the
  in-graph step just gathers a row (``subword_embeds`` fast path in the
  vendored model);
* the code feedback, output row write, step counter, and cache position all
  advance INSIDE the graph, so a replay is self-contained.

The captured state machine (all device tensors, no host round trips):

    cur_id  = timeline_ids[step_ctr]        # for EOS-silence gating
    cur_emb = cas_table[step_ctr]           # precomputed CAS row
    code    = one EAR-TTS step (backbone + MoG, CFG batch=2, StaticCache)
    code_buf, codes_out[step_ctr] = code    # feedback + output
    step_ctr += 1; cache_pos += 1

MoG sampling noise draws from the CUDA graph-safe Philox generator, which
advances per replay — audio is NOT bit-identical to the eager path (nor is the
eager path bit-stable across process RNG states); the frame-locked text channel
is unaffected. Sessions longer than ``max_cache_len`` fall back to the eager
path (the talker decides per session).
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


class TalkerStepGraph:
    """Owns the StaticCache, the per-session buffers, and the captured graph.

    Sessions are strictly serial (the talker enforces ``max_num_seqs=1``); the
    same buffers and capture are reused across requests via ``start_session``.
    """

    def __init__(
        self,
        tts: Any,
        *,
        dtype: torch.dtype,
        device: torch.device,
        max_cache_len: int,
        guidance_enabled: bool,
    ) -> None:
        self.tts = tts
        self.model = tts.tts_model
        self.device = device
        self.dtype = dtype
        self.max_cache_len = int(max_cache_len)
        self.guidance_enabled = bool(guidance_enabled)
        self.generation_config = tts._get_generation_config(self.guidance_enabled)

        # Attention in StaticCache mode runs over the FULL padded cache length,
        # so per-step cost scales with the bucket size (~11 ms/step at 512
        # positions vs ~20 ms/step at 4096 in fp32 on H100). Sessions with a
        # known timeline (sync mode) pick the smallest fitting bucket; growing
        # (async) sessions use the largest, since codes already shipped cannot
        # be regenerated under a different cache length mid-request.
        self._bucket_lens = [b for b in (512, 1024, 2048) if b < self.max_cache_len] + [self.max_cache_len]
        self._buckets: dict[int, dict[str, Any]] = {}
        self._active_bucket: int | None = None
        self._broken = False
        self._init_len: int | None = None
        self._timeline_len = 0

        hidden = int(self.model.hidden_size)
        quantizers = int(self.model.config.num_quantizers)
        with torch.inference_mode():
            self._timeline_ids = torch.zeros(self.max_cache_len, dtype=torch.long, device=device)
            self._cas_table = torch.zeros(self.max_cache_len, hidden, dtype=dtype, device=device)
            self._codes_out = torch.zeros(self.max_cache_len, quantizers, dtype=torch.long, device=device)
            self._code_buf = torch.zeros(1, 1, quantizers, dtype=torch.long, device=device)
            self._step_ctr = torch.ones(1, dtype=torch.long, device=device)
            self._cache_pos = torch.zeros(1, dtype=torch.long, device=device)
            self._ones_mask = torch.ones(1, 1, dtype=torch.bool, device=device)
            # Prepared 4D attention mask over the cache slots (True = attend).
            # Passing a ready 4D mask makes transformers' mask pipeline return
            # it as-is; the HF-derived mask (and anything else that reads the
            # cache's python-int lengths) would otherwise bake the capture-time
            # position into the graph. Per-bucket graphs slice a leading view.
            self._attn_mask = torch.zeros(1, 1, 1, self.max_cache_len, dtype=torch.bool, device=device)
        self._cache: Any = None  # the ACTIVE bucket's StaticCache

    # ------------------------------------------------------------------
    # Support / capacity checks.
    # ------------------------------------------------------------------
    def config_supported(self) -> tuple[bool, str]:
        """Static-config gate for the graph path (independent of the request)."""
        if self.device.type != "cuda":
            return False, f"device {self.device} is not CUDA"
        if not self.guidance_enabled:
            return False, "classifier-free guidance disabled (capture assumes the CFG batch layout)"
        if self.tts.cfg.tts_config.context_hidden_size is not None:
            return False, "context-LM conditioning enabled (prev-subword context is not captured)"
        if self.model.embed_subword is None:
            return False, "no CharAwareSubwordEncoder to precompute"
        sliding_window = getattr(self.model.backbone.config, "sliding_window", None)
        if sliding_window is not None and self.max_cache_len >= int(sliding_window):
            # The single prepared mask stands in for both the full and the
            # sliding mask, which is only equivalent while no position can
            # fall outside the window.
            return False, (
                f"max_cache_len {self.max_cache_len} >= backbone sliding_window {sliding_window}; "
                "the prepared causal mask would diverge from the sliding mask"
            )
        return True, ""

    def fits(self, timeline_len: int) -> bool:
        init_len = self._init_len if self._init_len is not None else 64
        return timeline_len <= self.max_cache_len - init_len - 1

    def _bucket_len_for(self, timeline_len: int | None) -> int:
        """Smallest bucket that fits ``init_len + timeline_len + 1`` positions.

        ``None`` (growing async timeline) selects the largest bucket: codes
        already shipped cannot be regenerated under a different cache length.
        """
        if timeline_len is None:
            return self._bucket_lens[-1]
        init_len = self._init_len if self._init_len is not None else 64
        needed = init_len + timeline_len + 1
        for bucket in self._bucket_lens:
            if needed <= bucket:
                return bucket
        return self._bucket_lens[-1]

    # ------------------------------------------------------------------
    # Capture.
    # ------------------------------------------------------------------
    def _get_bucket(self, bucket_len: int) -> dict[str, Any]:
        bucket = self._buckets.get(bucket_len)
        if bucket is None:
            from transformers import StaticCache

            bucket = {
                "cache": StaticCache(
                    config=self.model.backbone.config,
                    max_batch_size=2 if self.guidance_enabled else 1,
                    max_cache_len=bucket_len,
                    device=self.device,
                    dtype=self.dtype,
                ),
                "graph": None,
            }
            self._buckets[bucket_len] = bucket
        return bucket

    def _init_forward(self) -> None:
        """Eager speaker-prompt prefill into the StaticCache (per session)."""
        init_inputs = self.tts.get_init_inputs(B=1)
        init_len = int(init_inputs["code"].shape[1])
        if self._init_len is None:
            self._init_len = init_len
        elif init_len != self._init_len:
            raise RuntimeError(
                f"NemotronVoiceChat talker graph: init prompt length changed ({self._init_len} -> {init_len}); "
                "the captured graph assumes a fixed speaker prompt."
            )
        init_inputs.update(
            {
                "use_cache": True,
                "past_key_values": self._cache,
                "guidance_enabled": self.guidance_enabled,
                "cache_position": torch.arange(init_len, device=self.device),
            }
        )
        self.model(**init_inputs)
        self._cache_pos.fill_(init_len)
        self._code_buf.copy_(init_inputs["code"][:, -1:])
        # Seed the prepared mask: the speaker-prompt slots are attendable, the
        # per-step slot bits are set inside the captured step.
        self._attn_mask.zero_()
        self._attn_mask[..., :init_len] = True

    def _one_step(self) -> None:
        """The captured body: one EAR-TTS frame step + in-graph state advance."""
        cur_id = self._timeline_ids.index_select(0, self._step_ctr).reshape(1, 1)
        cur_emb = self._cas_table.index_select(0, self._step_ctr).reshape(1, 1, -1)
        # Open this step's own cache slot before attending (a token attends to
        # itself), then run the step with fully tensor-driven mask/positions.
        self._attn_mask.index_fill_(3, self._cache_pos, True)
        code, _ = self.tts.infer_codes_one_step(
            current_subword_id=cur_id,
            # prev_subword_id only feeds the context-LM channel, which
            # config_supported() guarantees is disabled.
            prev_subword_id=cur_id,
            current_subword_mask=self._ones_mask,
            prev_audio_tokens=self._code_buf,
            past_key_values=self._cache,
            guidance_enabled=self.guidance_enabled,
            generation_config=self.generation_config,
            ignore_eos_flag_stop=True,
            current_subword_embed=cur_emb,
            cache_position=self._cache_pos,
            attention_mask=self._attn_mask[..., : self._active_bucket],
            position_ids=self._cache_pos.reshape(1, 1),
        )
        self._code_buf.copy_(code)
        self._codes_out.index_copy_(0, self._step_ctr, code.reshape(1, -1))
        self._step_ctr.add_(1)
        self._cache_pos.add_(1)

    def ensure_captured(self, timeline_len: int | None = None) -> bool:
        """Warm up and capture the bucket for this timeline (throwaway session)."""
        if self._broken:
            return False
        bucket_len = self._bucket_len_for(timeline_len)
        bucket = self._get_bucket(bucket_len)
        if bucket["graph"] is not None:
            return True
        try:
            with torch.inference_mode():
                # Throwaway session: buffer VALUES are irrelevant for capture,
                # only shapes/addresses matter. PAD-filled timeline rows and a
                # zero CAS table are fine.
                self._cache = bucket["cache"]
                self._active_bucket = bucket_len
                self._cache.reset()
                self._step_ctr.fill_(1)
                self._cache_pos.zero_()
                self._init_forward()
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):
                        self._one_step()
                torch.cuda.current_stream().wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self._one_step()
                bucket["graph"] = graph
            logger.info(
                "NemotronVoiceChat talker CUDA graph captured (bucket=%d, init_len=%d)",
                bucket_len,
                self._init_len,
            )
            return True
        except Exception:
            self._broken = True
            logger.exception(
                "NemotronVoiceChat talker CUDA graph capture failed; falling back to the eager per-frame step."
            )
            return False

    # ------------------------------------------------------------------
    # Session lifecycle.
    # ------------------------------------------------------------------
    def start_session(self, timeline: torch.Tensor, *, expected_len: int | None) -> None:
        """Reset all captured state and prefill for a new request.

        ``expected_len`` is the FINAL timeline length when known up front
        (sync mode ships the whole timeline; frame-locked async producers ship
        meta.expected_total_tokens) and selects the smallest fitting bucket;
        ``None`` (unbounded growth, e.g. duplex) selects the largest bucket.
        """
        bucket_len = self._bucket_len_for(expected_len)
        bucket = self._buckets[bucket_len]
        assert bucket["graph"] is not None, "ensure_captured() must succeed before start_session()"
        with torch.inference_mode():
            self._cache = bucket["cache"]
            self._active_bucket = bucket_len
            self._cache.reset()
            self._step_ctr.fill_(1)
            self._cache_pos.zero_()
            self._codes_out.zero_()
            self._timeline_len = 0
            self._init_forward()
            self.extend_timeline(timeline)

    def extend_timeline(self, timeline: torch.Tensor) -> None:
        """Adopt a (possibly longer) timeline; CAS-encode only the new rows."""
        new_len = int(timeline.numel())
        if new_len <= self._timeline_len:
            return
        if not self.fits(new_len):
            raise RuntimeError(
                f"NemotronVoiceChat talker graph: timeline of {new_len} positions exceeds the "
                f"captured capacity ({self.max_cache_len - (self._init_len or 0) - 1} frames). Raise "
                "hf_overrides.talker_graph_max_cache_len or disable hf_overrides.use_talker_cuda_graphs."
            )
        active = self._active_bucket
        if active is not None and (self._init_len or 0) + new_len + 1 > active:
            raise RuntimeError(
                f"NemotronVoiceChat talker graph: timeline of {new_len} positions exceeds the active "
                f"cache bucket ({active} positions). Raise hf_overrides.talker_graph_max_cache_len or "
                "disable hf_overrides.use_talker_cuda_graphs."
            )
        start = self._timeline_len
        with torch.inference_mode():
            new_ids = timeline[start:new_len].to(device=self.device, dtype=torch.long).reshape(-1)
            self._timeline_ids[start:new_len] = new_ids
            mask = torch.ones(1, new_ids.numel(), dtype=torch.bool, device=self.device)
            # One batched CAS call per chunk of new tokens (the encoder's host
            # syncs and dynamic char shapes stay OUTSIDE the captured step).
            embeds = self.model.embed_subword(new_ids.unsqueeze(0), mask)[0]
            self._cas_table[start:new_len] = embeds.to(self.dtype)
        self._timeline_len = new_len

    def run_steps(self, num_steps: int) -> None:
        graph = self._buckets[self._active_bucket]["graph"]
        for _ in range(num_steps):
            graph.replay()

    def codes_rows(self, first_step: int, last_step: int) -> torch.Tensor:
        """Codes for NeMo steps ``first_step..last_step-1`` (row t = step t)."""
        with torch.inference_mode():
            return self._codes_out[first_step:last_step].clone()
