# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""dots.tts talker — vLLM-native AR base LM + audio side path.

Mirrors upstream rednote-hilab/dots.tts (pinned @ a393d2e):

  * Qwen2.5-1.5B base LM runs under vLLM PagedAttention (stock
    ``Qwen2Model``); weights come from the checkpoint's ``llm.model.*``
    namespace.  ``DotsTTSConfig`` hoists the LM fields to top-level
    attributes.
  * Everything else is side computation, invisible to the vLLM
    scheduler and driven once per step by ``_finish_decode``:
    DiT flow-matching head (N-step Euler, fp32 integration state,
    deterministic per-request noise) → io_helper denormalize →
    patch_encoder AR loopback (produces the next step's LLM input
    embedding) + streaming AudioVAE decode (per-request
    ``BigVGANStreamState`` sliding window, so every latent patch is
    decoded with real left conv context) → 48 kHz wav chunks pushed to
    ``_audio_queue`` → eos_proj stop signal via ``_results_queue``
    (threshold 0.8, upstream default; the stop step also drains the
    vocoder's 2-frame lookahead tail via ``stream_flush``).
  * Cross-step state is isolated per request in ``_RequestState``
    (keyed by request_id); eviction is deferred to the end of
    ``forward()`` so a finishing request's audio still drains.
  * The CAM++ speaker encoder is constructed and its weights load, but
    voice cloning is not exposed yet — generation is zero-shot and the
    DiT falls back to its null conditioning (``fm_null_g_cond``).

Debugging: set ``DOTS_TTS_BETA_TRACE=1`` for an env-gated per-patch
rms/max trace across the full side path (zero overhead when unset).

Reference implementations: ``vllm_omni/model_executor/models/voxcpm2/``
(architectural template: vLLM-native base LM + side path + per-request
state) and ``ming_flash_omni`` (AudioVAE weight-loading pattern).
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.dots_tts.dots_tts_dit import DiT
from vllm_omni.model_executor.models.dots_tts.dots_tts_patch_encoder import (
    VAESemanticEncoder,
)
from vllm_omni.model_executor.models.dots_tts.dots_tts_speaker_encoder import (
    SpeakerXVectorFeatures,
)
from vllm_omni.model_executor.models.dots_tts.dots_tts_vocoder import (
    AudioVAE,
    AudioVAEConfig,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


def _require_config_block(hf_config: Any, name: str) -> dict[str, Any]:
    """Fetch a sub-config dict (``vocoder`` / ``DiT`` / ``PatchEncoder``)
    from the checkpoint's config.json, failing loudly when absent —
    silently falling back to another checkpoint's values would produce
    garbage audio, not an error."""
    block = getattr(hf_config, name, None)
    if not isinstance(block, dict) or not block:
        raise ValueError(
            f"dots.tts checkpoint config.json has no '{name}' block — cannot "
            "build the audio side path.  Expected the upstream dots.tts config "
            "layout (e.g. rednote-hilab/dots.tts-soar)."
        )
    return block


def _build_audio_vae_config(hf_config: Any) -> AudioVAEConfig:
    # AudioVAEConfig declares exactly the checkpoint's 17 ``vocoder`` keys,
    # so an unknown / missing key raises TypeError instead of guessing.
    return AudioVAEConfig(**_require_config_block(hf_config, "vocoder"))


# Architecture constants used by the per-request FM workspace math.  All
# released dots.tts checkpoints (soar / base / mf) share these values;
# _validate_architecture_constants refuses a checkpoint that disagrees
# instead of silently generating garbage.
_FM_HIDDEN = 1024  # DiT.hidden_size  (also LLM↔DiT projection target)
_LATENT_DIM = 128  # AudioVAE.latent_dim
_LATENT_PATCH_SIZE = 4  # config.patch_size — DiT samples 4 latent frames / audio patch
_HIDDEN_PATCH_SIZE = 1  # upstream core.py hardcodes self.hidden_patch_size = 1
_MAX_AUDIO_PATCHES = 1024  # ~164 s @ 4×1920/48k; bounds per-request FM static buffer
_DIT_NUM_STEPS = int(
    os.environ.get("DOTS_TTS_DIT_NUM_STEPS", "10")
)  # env-gated; upstream default 10 for fixed-step Euler
_DIT_GUIDANCE_SCALE = 1.2  # upstream default guidance_scale for soar
_DIT_NOISE_SEED = 20260601  # base seed for per-request FM noise (voxcpm2 parity)
_PATCH_ENCODER_OUT_DS_RATE = 2  # patch_size / in_ds_rate = 4 / 2 (VAESemanticEncoder hardcodes in_ds_rate=2)


class _IOHelper:
    """Latent stats wrapper.  Holds (mean, var) over the 128-dim VAE latent
    space and normalizes / denormalizes between DiT-internal space (which
    operates on normalized latents) and AudioVAE / patch_encoder input space
    (which expects raw latents).

    Mirrors upstream IOHelper (rednote-hilab/dots.tts @ a393d2e
    models/dots_tts/core.py:684).  Plain class (not nn.Module) so it stays
    out of state_dict() and load_weights() doesn't have to mute its buffers.
    Stats are loaded from latent_stats.pt — independent of safetensors.
    """

    def __init__(self, latent_stats_path: str | None = None) -> None:
        if latent_stats_path is None:
            self.global_mean: torch.Tensor | None = None
            self.global_var: torch.Tensor | None = None
            return
        stats = torch.load(latent_stats_path, weights_only=False)
        self.global_mean = torch.as_tensor(stats["mean"])
        self.global_var = torch.as_tensor(stats["var"])

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.global_mean is None or self.global_var is None:
            return x
        mean = self.global_mean.to(device=x.device, dtype=x.dtype)
        var = self.global_var.to(device=x.device, dtype=x.dtype)
        return (x - mean) / torch.sqrt(var)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.global_mean is None or self.global_var is None:
            return x
        mean = self.global_mean.to(device=x.device, dtype=x.dtype)
        var = self.global_var.to(device=x.device, dtype=x.dtype)
        return x * torch.sqrt(var) + mean


# Per-request state container for AR-loop continuity.  All cross-step
# side-path state lives here, keyed by request_id, so concurrent requests
# stay fully isolated (the voxcpm2 _RequestState pattern).
@dataclass
class _RequestState:
    request_id: str
    # AR loopback: patch_encoder's previous decode output, fed to the LM
    # as next-step inputs_embeds.  Shape: [1, llm_hidden] = [1, 1536].
    # (patch_encoder collapses its internal _PATCH_ENCODER_OUT_DS_RATE
    # positions into one LLM token via out_proj; see
    # VAESemanticEncoder._project_embeddings.)  _finish_decode writes a
    # fresh tensor every call.
    curr_embed_for_next: torch.Tensor | None = None
    # Stop-signal cache: _finish_decode populates precomputed_stop_logits;
    # compute_logits drains it.
    precomputed_stop_logits: torch.Tensor | None = None
    is_stopping: bool = False
    prefill_completed: bool = False
    # Per-request FM noise counter: draw #n of this request hashes to a
    # deterministic Generator seed (see _run_dit_n_step_euler), so outputs
    # are reproducible run-to-run and concurrent requests cannot perturb
    # each other's noise streams (voxcpm2 _fill_deterministic_cfm_noise).
    noise_step: int = 0
    # Per-request FM static workspace (lazy-allocated by
    # _initialize_request_fm_state on first _finish_decode call).  Sized for
    # _MAX_AUDIO_PATCHES × (_HIDDEN_PATCH_SIZE + _LATENT_PATCH_SIZE) = 1024 × 5
    # = 5120 positions.
    fm_sequence: torch.Tensor | None = None  # [1, fm_capacity, _FM_HIDDEN]
    fm_cfg_sequence: torch.Tensor | None = None  # [1, fm_capacity, _FM_HIDDEN]
    fm_null_g_cond: torch.Tensor | None = None  # [1, _FM_HIDDEN]
    fm_seq_len: int = 0
    fm_capacity: int = 0
    # Speaker x-vector after _xvec_proj.  Zero-shot generation leaves this
    # None → DiT falls back to fm_null_g_cond; voice-clone wiring is a
    # follow-up.
    g_cond: torch.Tensor | None = None
    # AR-loop bookkeeping.
    # patch_encoder_state holds conv_tail + per-layer KV caches for the
    # patch_encoder's streaming decode (mirrors upstream state.patch_encoder_state).
    # Lazy-allocated by _run_patch_encoder_loopback on first audio step.
    patch_encoder_state: Any | None = None
    # Review M2: streaming vocoder state (BigVGANStreamState — LSTM hidden
    # + decoder sliding latent window).  Gives each patch real left conv
    # context; output lags input by decoder.stream_lookahead (2 frames =
    # 3840 samples).  Lazy-allocated by _run_vocoder_stream_step.
    vocoder_stream_state: Any | None = None


# DiT transformer hyperparameters from rednote-hilab/dots.tts-soar config.json
# (the ``DiT`` block).  Mirrors the shape of an upstream pydantic config: DiT's
# constructor calls ``transformer_config.to_dict()`` and reads ``.hidden_size``
# / ``.num_layers``, so this @dataclass plus the ``to_dict`` helper satisfies
# that minimal interface without dragging in upstream's config system.
#
# Note: ``attn_dropout`` in the JSON spells differently from MultiHeadAttention's
# ``attn_drop`` kwarg — the value gets absorbed by ``**_kwargs`` and the MHA
# default 0.0 applies.  Same numeric outcome for the soar checkpoint (both 0.0),
# so this is benign; flagging it because it would mask non-default values if
# upstream ever raised this field.
@dataclass
class _DiTConfig:
    num_layers: int = 18
    num_heads: int = 16
    hidden_size: int = 1024
    ffn_hidden_size: int = 4096
    modulation: bool = True
    qkv_bias: bool = False
    qk_norm: bool = True
    attn_dropout: float = 0.0
    dropout: float = 0.0
    norm_layer: str = "RMSNorm"
    alibi_bias: bool = False
    rotary_bias: bool = True
    rotary_theta: float = 10000.0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build_dit_config(hf_config: Any) -> _DiTConfig:
    # _DiTConfig declares exactly the checkpoint's 13 ``DiT`` keys, so an
    # unknown key raises TypeError instead of being silently dropped.
    return _DiTConfig(**_require_config_block(hf_config, "DiT"))


# Patch encoder hyperparameters from rednote-hilab/dots.tts-soar config.json
# (the top-level ``patch_size`` and the ``PatchEncoder`` sub-block).
#
# Two layers of config:
#   * Outer (``_PatchEncoderConfig``): VAESemanticEncoder reads ``.patch_size``
#     and forwards ``.PatchEncoder`` to its inner SuperviseEncoder.
#   * Inner (``_PatchEncoderInner``): the constructor uses BOTH attribute
#     access (``config.PatchEncoder.hidden_size``) and dict-style
#     ``config.get("hidden_size", 1024)``.  Upstream's pydantic ConfigBase
#     supports both natively; we add a tiny ``.get()`` helper to a dataclass.
#
# Upstream's ``_EncoderConfig`` also defines qk_norm / rotary_bias /
# rotary_theta / input_dim, but ``SuperviseEncoder.__init__`` only reads six
# keys (num_layers / num_heads / hidden_size / ffn_hidden_size / norm_layer /
# causal).  The rest are swallowed and MultiHeadAttention falls back to its
# own defaults — matching upstream behaviour means *not* plumbing them through.
@dataclass
class _PatchEncoderInner:
    num_layers: int = 24
    num_heads: int = 16
    hidden_size: int = 1024
    ffn_hidden_size: int = 4096
    norm_layer: str = "RMSNorm"
    causal: bool = True

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class _PatchEncoderConfig:
    patch_size: int = 4
    PatchEncoder: _PatchEncoderInner = dataclasses.field(default_factory=_PatchEncoderInner)


def _build_patch_encoder_config(hf_config: Any) -> _PatchEncoderConfig:
    block = _require_config_block(hf_config, "PatchEncoder")
    # SuperviseEncoder reads only the six declared fields; upstream's
    # pydantic config swallows the other checkpoint keys, so filtering
    # here matches upstream behaviour exactly.
    inner_fields = {f.name for f in dataclasses.fields(_PatchEncoderInner)}
    inner = _PatchEncoderInner(**{k: v for k, v in block.items() if k in inner_fields})
    return _PatchEncoderConfig(
        patch_size=int(hf_config.patch_size),
        PatchEncoder=inner,
    )


def _validate_architecture_constants(hf_config: Any, dit_config: _DiTConfig) -> None:
    """The FM workspace helpers size buffers off module-level constants;
    a checkpoint that disagrees would corrupt the DiT conditioning, so
    refuse it up front (making these fully config-driven is deferred
    until a released checkpoint actually differs)."""
    checks = [
        ("latent_dim", getattr(hf_config, "latent_dim", None), _LATENT_DIM),
        ("patch_size", getattr(hf_config, "patch_size", None), _LATENT_PATCH_SIZE),
        ("DiT.hidden_size", dit_config.hidden_size, _FM_HIDDEN),
    ]
    mismatches = [f"{name}={got!r} (expected {want})" for name, got, want in checks if got != want]
    if mismatches:
        raise ValueError(
            "dots.tts checkpoint config disagrees with this integration's "
            "architecture constants: " + ", ".join(mismatches)
        )


class DotsTTSForConditionalGeneration(nn.Module):
    """dots.tts AR talker: Qwen2.5 base LM + DiT / AudioVAE / CAM++ side path."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config

        # vllm-omni capability flags — gpu_model_runner checks these to decide
        # whether to call preprocess() / postprocess() / accept multimodal
        # outputs.  Without these, the runner bypasses our preprocess hook
        # and _pending_requests stays empty (forward never dispatches the
        # side-path).  Voxcpm2 sets the same three.
        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = True

        self.model = Qwen2Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        # Side-path module hyperparameters come off the checkpoint's
        # config.json blocks (fail loudly if a block is absent); the FM
        # workspace constants are cross-checked against the same config.
        dit_config = _build_dit_config(self.config)
        _validate_architecture_constants(self.config, dit_config)
        llm_hidden = self.config.hidden_size
        fm_hidden = dit_config.hidden_size
        latent_dim = int(self.config.latent_dim)
        xvec_dim = int(getattr(self.config, "campplus_embedding_size", 512))

        self._audio_vae = AudioVAE(_build_audio_vae_config(self.config))
        # Upstream serializes vocoder.safetensors with weight_norm folded into
        # plain `weight` on the decoder (encoder kept weight_norm — it's not in
        # the synthesis hot path).  Match that layout before load_weights runs,
        # so checkpoint keys align with our state_dict 1:1.
        self._audio_vae.remove_weight_norm()
        # AudioVAE / speaker encoder stay fp32 — upstream only casts
        # `model.core` to bf16 (runtime.py:81), leaving vocoder + xvector
        # extractor in fp32 because their kernels (conv1d / fbank) need
        # higher precision.  Without this _audio_vae would inherit talker's
        # bf16 dtype via vLLM's auto-cast and dtype-mismatch under
        # stream_step (input .float() vs bf16 weights).

        # DiT flow-matching head.  Dimensions per upstream core.py:101-104:
        #   in_dim  = config.DiT.hidden_size  (DiT internal space)
        #   out_dim = config.latent_dim       (AudioVAE input space)
        # mode is "flow_matching" for soar/base; "meanflow" only for the mf
        # checkpoint (handled by a separate factory in a later step).
        self._head = DiT(
            in_dim=fm_hidden,
            out_dim=latent_dim,
            transformer_config=dit_config,
            mode="flow_matching",
        )

        # Patch encoder: closes the AR loop by mapping each DiT-emitted audio
        # latent patch back into the LLM hidden space, so Qwen2 "hears" what it
        # has already generated.  Upstream core.py:75 calls this with
        # in_dim=latent_dim and out_dim=llm.config.hidden_size.
        self._patch_encoder = VAESemanticEncoder(
            in_dim=latent_dim,
            out_dim=llm_hidden,
            config=_build_patch_encoder_config(self.config),
        )

        # Five thin projectors (upstream core.py:82-113).  Soar dimensions:
        # llm_hidden=1536, fm_hidden=1024, latent_dim=128, xvec_dim=512.
        # These are kept as direct attributes (no vendor file) — they are flat
        # nn.Linear / nn.Sequential, and live at the top of the checkpoint
        # namespace under their own short prefixes (see load_weights).
        self._hidden_proj = nn.Linear(llm_hidden, fm_hidden)
        self._latent_proj = nn.Linear(latent_dim, fm_hidden)
        self._coordinate_proj = nn.Linear(latent_dim, fm_hidden)
        self._xvec_proj = nn.Sequential(
            nn.Linear(xvec_dim, fm_hidden),
            nn.LayerNorm(fm_hidden),
        )
        # eos_proj is the stop predictor — feeds LLM hidden through a 2-layer
        # MLP and produces a 2-way logit (continue / stop).
        self._eos_proj = nn.Sequential(
            nn.Linear(llm_hidden, llm_hidden),
            nn.SiLU(),
            nn.Linear(llm_hidden, 2),
        )

        # CAM++ speaker encoder (3D-Speaker), produces the x-vector from
        # reference audio for voice cloning.  sample_rate matches AudioVAE's
        # input space (48 kHz); the wrapper resamples to 16k internally for
        # CAM++.  Weights are frozen (upstream does `requires_grad = False`).
        self._speaker_encoder = SpeakerXVectorFeatures(
            sample_rate=self._audio_vae.sample_rate,
            campplus_embedding_size=xvec_dim,
            max_audio_seconds=float(getattr(self.config, "xvec_max_audio_seconds", 10.0)),
        )

        # Pin AudioVAE + speaker encoder to fp32 (mirror upstream
        # runtime.py:81 where only `model.core` is cast to bf16).  vLLM's
        # auto-cast would otherwise lower these to bf16, breaking
        # stream_step (input .float() vs bf16 weights) and the fbank
        # pipeline.
        self._audio_vae.float()
        self._speaker_encoder.float()

        # Latent stats — independent of safetensors (torch.load on
        # latent_stats.pt sitting next to the checkpoint files).  Used by
        # _finish_decode to bridge between DiT's normalized latent space and
        # the raw latent space expected by patch_encoder + AudioVAE.
        latent_stats_path = self._resolve_latent_stats_path(vllm_config.model_config.model)
        if latent_stats_path is not None:
            self._io_helper = _IOHelper(latent_stats_path)
        else:
            logger.warning(
                "latent_stats.pt not resolvable for %s; running with "
                "identity normalize/denormalize.  Wav will be ~10x quiet "
                "and distorted (no sqrt(var) denormalize).",
                vllm_config.model_config.model,
            )
            self._io_helper = _IOHelper()

        # (β) diagnostic: env-gated per-patch rms/max trace across the full
        # side path (LLM hidden → DiT → denormalize → patch_encoder → wav →
        # eos).  Zero overhead when DOTS_TTS_BETA_TRACE is unset (single
        # bool check per step); kept as the model's debugging instrument
        # (cf. voxcpm2's enable_profiling runtime knob).
        self._beta_trace = bool(os.environ.get("DOTS_TTS_BETA_TRACE"))

        # Per-request state plumbing.
        # _active_states: per-request state keyed by request_id (created in
        #   preprocess prefill, evicted by _flush_deferred_cleanup).
        # _pending_requests: per-step list of (req_id, is_prefill, embeds,
        #   span_len) populated by preprocess(), consumed by forward().
        # _deferred_cleanup_ids: requests vLLM has marked finished but whose
        #   side-path audio still needs draining in the current forward step;
        #   evicted from _active_states at the end of forward().
        self._active_states: dict[str, _RequestState] = {}
        self._pending_requests: list[tuple[str, bool, torch.Tensor, int]] = []
        self._deferred_cleanup_ids: set[str] = set()

        # Audio side-channel:
        # _audio_queue: _finish_decode pushes (req_id, audio_tensor) tuples
        #   here; make_omni_output drains them into multimodal_outputs each
        #   step.
        # _sample_rate: cached from AudioVAE config (= 48000 for soar).
        self._audio_queue: list[tuple[str, torch.Tensor]] = []
        self._sample_rate = self._audio_vae.sample_rate

        # Stop-signal side-channel:
        # _results_queue: _finish_decode pushes (req_id, stop_logits) tuples
        #   here; compute_logits drains them into the [bsz, vocab_size]
        #   logits tensor each step.
        self._results_queue: list[tuple[str, torch.Tensor | None]] = []

        logger.info(
            "DotsTTS talker built (base_lm=Qwen2[28L,12H,1536d], "
            "audio_vae=AudioVAE, dit=DiT[18L,16H,1024d], "
            "patch_encoder=VAESemanticEncoder[24L,16H,1024d], "
            "5 projectors, speaker_encoder=CAM++[7.2M], "
            "io_helper=%s, model=%s, dit_steps=%d); "
            "_finish_decode: DiT → denormalize → patch_encoder AR + "
            "streaming AudioVAE decode → wav + eos_proj → stop signal.",
            "loaded" if self._io_helper.global_mean is not None else "identity",
            vllm_config.model_config.model,
            _DIT_NUM_STEPS,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        """Drive the base LM, then the per-request audio side path.

        One vLLM step: base LM forward under PagedAttention, then slice
        the scaffold hidden states per request (in _pending_requests
        order) and run _finish_decode on each — DiT sampling, streaming
        AudioVAE decode, AR loopback, and stop signal all happen there.
        vLLM only ever sees the returned hidden states; audio leaves
        through the _audio_queue side channel.
        """
        output = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        if isinstance(output, IntermediateTensors):
            return output
        if isinstance(output, tuple):
            output = output[0]

        # Per-request side-path: slice scaffold_hidden by token span_len
        # in the order preprocess() appended to _pending_requests.
        token_offset = 0
        for req_id, is_prefill, _embeds, span_len in self._pending_requests:
            req_hidden = output[token_offset : token_offset + span_len]
            token_offset += span_len
            self._finish_decode(req_id, req_hidden, is_prefill)

        # End-of-step cleanup
        self._pending_requests.clear()
        self._flush_deferred_cleanup()

        return output

    # ── vllm-omni protocol methods ──

    @staticmethod
    def _resolve_latent_stats_path(model_arg: str) -> str | None:
        """Resolve latent_stats.pt to a local filesystem path.

        ``vllm_config.model_config.model`` is typically an HF repo ID like
        ``"rednote-hilab/dots.tts-soar"`` — joining it with ``"latent_
        stats.pt"`` gives a non-existent path.  Try the local-dir form
        first (covers ``--model /local/path``), then fall back to HF
        cache lookup with ``local_files_only=True`` so we never trigger
        a download from this side (the file was fetched alongside the
        safetensors at load time).
        """
        local = os.path.join(model_arg, "latent_stats.pt")
        if os.path.exists(local):
            return local
        try:
            from huggingface_hub import hf_hub_download

            return hf_hub_download(
                repo_id=model_arg,
                filename="latent_stats.pt",
                local_files_only=True,
            )
        except Exception:
            return None

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        """Map token ids to embeddings via the base LM's embed_tokens.

        Required by vLLM's ``is_text_generation_model`` Protocol check
        (interfaces_base._check_vllm_model_embed_input_ids).  Without this
        the model fails ``--runner generate`` validation at engine startup.
        """
        return self.model.embed_tokens(input_ids)

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Per-step embedding + per-request state registration.

        Prefill: embed the wrapped prompt
        ``[文本]<text>[文本对应语音]<AUDIO_GEN_START>`` via the LM's
        embed_tokens.  Caller (serving layer or test harness) is
        responsible for the wrap — talker just embeds whatever it gets.

        Decode (span_len == 1): use ``state.curr_embed_for_next`` from
        the previous step's AR loopback (DiT → patch_encoder output);
        zero fallback only if the loopback hasn't produced one yet.

        Voice clone is intentionally not wired here — zero-shot only.

        Returns ``(input_ids unchanged, embeds [span_len, hidden_size],
        {})``.  ``forward()`` consumes ``self._pending_requests`` to
        slice scaffold_hidden per request.
        """
        # voxcpm2 protocol compatibility: flatten additional_information.
        additional = info_dict.get("additional_information")
        if isinstance(additional, dict):
            merged = {k: v for k, v in info_dict.items() if k != "additional_information"}
            for k, v in additional.items():
                merged.setdefault(k, v)
            info_dict = merged

        span_len = int(input_ids.shape[0])
        dev = input_ids.device
        req_id = info_dict.get("request_id", "default")
        state = self._active_states.get(req_id)
        # Prefill is detected by state lifecycle (not span_len), so decode
        # can use span_len=1 with a multi-position curr_embed_for_next.
        is_prefill = state is None or not state.prefill_completed

        if is_prefill:
            if state is None:
                state = _RequestState(request_id=req_id)
                self._active_states[req_id] = state
            # Reset per-step state on every prefill — previous run is stale.
            state.curr_embed_for_next = None
            state.precomputed_stop_logits = None
            state.is_stopping = False
            state.prefill_completed = False
            state.noise_step = 0
            # Reset AR-loop state on every prefill.
            state.patch_encoder_state = None
            state.vocoder_stream_state = None
            embeds = self.model.embed_tokens(input_ids)
        else:
            curr = state.curr_embed_for_next
            if curr is not None:
                embeds = curr.to(dev)  # [1, llm_hidden]
            else:
                # First decode step before AR loop closed — zero fallback.
                embed_dtype = self.model.embed_tokens.weight.dtype
                embeds = torch.zeros(
                    1,
                    self.config.hidden_size,
                    device=dev,
                    dtype=embed_dtype,
                )

        self._pending_requests.append((req_id, is_prefill, embeds, span_len))
        return input_ids, embeds, {}

    def postprocess(self, *args: Any, **kwargs: Any) -> dict:
        return {}

    def on_requests_finished(self, finished_req_ids: Iterable[str]) -> None:
        """Mark finished requests for deferred eviction.

        vLLM scheduler calls this BEFORE forward() to notify that certain
        requests have completed.  We can't drop their _RequestState yet —
        the current forward step still needs to drain side-path audio
        chunks.  Real eviction happens at the end of forward() via
        _flush_deferred_cleanup.
        """
        for req_id in finished_req_ids:
            if req_id in self._active_states:
                self._deferred_cleanup_ids.add(req_id)

    def _flush_deferred_cleanup(self) -> None:
        """Evict states marked by on_requests_finished.  Called at the
        tail of forward()."""
        for req_id in self._deferred_cleanup_ids:
            self._active_states.pop(req_id, None)
        self._deferred_cleanup_ids.clear()

    def _finish_decode(
        self,
        req_id: str,
        req_hidden: torch.Tensor,
        is_prefill: bool,
    ) -> None:
        """One request's per-step side path: DiT → AR loopback → wav → stop.

        One vLLM step = one audio patch.  patch_encoder.decode_patch
        collapses its internal _PATCH_ENCODER_OUT_DS_RATE positions into
        a single LLM input embedding (see VAESemanticEncoder._project_
        embeddings), so vLLM's span_len=1 decode matches naturally — no
        slot bookkeeping needed.

        Per-call flow (prefill or decode):
          1. _hidden_proj(last_hidden) → fm_sequence (+1 position)
          2. DiT N-step Euler → audio_patch [1, 4, 128] (normalized)
          3. _latent_proj(audio_patch) → fm_sequence (+4 positions, normalized)
          4. io_helper.denormalize(audio_patch) → audio_patch_raw
          5. patch_encoder(audio_patch_raw) → next-step embed [1, 1536]
          6. AudioVAE stream_step(audio_patch_raw) → wav chunk pushed to
             _audio_queue (prefill's patch #0 included; output lags 2
             frames behind input)
          7. (decode only) eos_proj(last_hidden) → stop_logits pushed to
             _results_queue; threshold > 0.8 flips state.is_stopping and
             stream_flush drains the decoder's lookahead tail.
        """
        state = self._active_states.get(req_id)
        if state is None:
            return

        if state.fm_sequence is None:
            self._initialize_request_fm_state(
                state,
                device=req_hidden.device,
                dtype=req_hidden.dtype,
            )

        # This step's append would overflow the per-request FM buffer.
        # Stop gracefully (same path as a model-decided stop below) instead
        # of raising past _append_hidden_chunk/_append_history_chunk —
        # those raises are engine-fatal, not request-fatal, and would take
        # every other in-flight and future request down with this one.
        if state.fm_seq_len + _HIDDEN_PATCH_SIZE + _LATENT_PATCH_SIZE > state.fm_capacity:
            state.is_stopping = True
            stop_logits = torch.tensor([[0.0, 1.0]], device=req_hidden.device, dtype=req_hidden.dtype)
            state.precomputed_stop_logits = stop_logits
            tail = self._audio_vae.stream_flush(state.vocoder_stream_state)
            if tail.size(-1) > 0:
                self._audio_queue.append((req_id, tail.reshape(-1)))
            self._results_queue.append((req_id, stop_logits))
            return

        # 1. Append last LLM hidden to fm_sequence (+1 position).
        last_hidden = req_hidden[-_HIDDEN_PATCH_SIZE:].unsqueeze(0)

        self._append_hidden_chunk(state, last_hidden)

        # 2. DiT N-step Euler → audio latent patch [1, 4, 128] (normalized).
        audio_patch = self._run_dit_n_step_euler(state)

        # 3. Append latent to fm_sequence (+4 positions).  Stays in
        #    normalized space — DiT's KV cache lives in that space.
        self._append_history_chunk(state, audio_patch)

        # 4. Denormalize for downstream consumers (patch_encoder, AudioVAE).
        audio_patch_raw = self._io_helper.denormalize(audio_patch)

        # 5. patch_encoder AR loopback → next-step inputs_embeds [1, 1536].
        next_embeds = self._run_patch_encoder_loopback(state, audio_patch_raw)
        state.curr_embed_for_next = next_embeds.squeeze(0).detach()

        # 6. Streaming AudioVAE decode → wav chunk.  Runs on prefill too:
        #    upstream zero-shot emits every DiT patch including the first
        #    (_generate_latents_stream drops one patch only under prompt
        #    prefill / voice clone), and our prefill patch #0 corresponds
        #    to upstream's first decode-loop patch.
        wav = self._run_vocoder_stream_step(state, audio_patch_raw)
        if wav.size(-1) > 0:
            self._audio_queue.append((req_id, wav.reshape(-1)))

        if is_prefill:
            if self._beta_trace:
                self._beta_trace_log(
                    state,
                    is_prefill=True,
                    last_hidden=last_hidden,
                    audio_patch=audio_patch,
                    audio_patch_raw=audio_patch_raw,
                    next_embeds=next_embeds,
                    wav=wav if wav.size(-1) > 0 else None,
                    prob_stop=None,
                )
            state.prefill_completed = True
            # Keep len(_results_queue) == bsz: compute_logits pairs queue
            # entries with batch rows positionally, so every request in the
            # batch must push exactly one entry per step (voxcpm2_talker.py
            # prefill placeholder pattern).  None → forced continue via the
            # stop_logits-is-None branch in compute_logits.
            self._results_queue.append((req_id, None))
            return

        # 7. eos_proj → stop probability.  Mirrors upstream
        #    model.py:1681 — softmax([continue, stop]) on detached LLM
        #    hidden, threshold the stop slot at 0.8 (upstream default
        #    eos_threshold).  When stop fires for this step, compute_logits
        #    drains the high-stop logit on the next dispatch and vLLM
        #    terminates the request.
        eos_logits = self._eos_proj(last_hidden.detach()).softmax(dim=-1)
        stop_logits = eos_logits.squeeze(0)
        state.precomputed_stop_logits = stop_logits
        if eos_logits[0, -1, 1].item() > 0.8:
            state.is_stopping = True
            # Upstream flushes after its decode loop (model.py:1898); our
            # request ends when vLLM sees this step's stop token, so drain
            # the decoder's 2-frame lookahead tail in the same step.
            # Requests cut by max_tokens/abort never reach this path and
            # lose the 40 ms tail — no engine hook fires late enough to
            # flush after the final step.
            tail = self._audio_vae.stream_flush(state.vocoder_stream_state)
            if tail.size(-1) > 0:
                self._audio_queue.append((req_id, tail.reshape(-1)))
        self._results_queue.append((req_id, stop_logits))

        if self._beta_trace:
            self._beta_trace_log(
                state,
                is_prefill=False,
                last_hidden=last_hidden,
                audio_patch=audio_patch,
                audio_patch_raw=audio_patch_raw,
                next_embeds=next_embeds,
                wav=wav if wav.size(-1) > 0 else None,
                prob_stop=eos_logits[0, -1, 1].item(),
            )

    def _beta_trace_log(
        self,
        state: _RequestState,
        *,
        is_prefill: bool,
        last_hidden: torch.Tensor,
        audio_patch: torch.Tensor,
        audio_patch_raw: torch.Tensor,
        next_embeds: torch.Tensor,
        wav: torch.Tensor | None,
        prob_stop: float | None,
    ) -> None:
        """(β) diagnostic — patch label matches output.wav indexing in
        decode rows; prefill row labels itself ``pre``."""

        def rmsmax(t: torch.Tensor) -> tuple[float, float]:
            f = t.detach().float()
            return f.pow(2).mean().sqrt().item(), f.abs().max().item()

        patch_label = "pre" if is_prefill else f"{state.fm_seq_len // 5 - 2:2d}"
        req_tail = state.request_id[-8:]
        hid_rms, hid_max = rmsmax(last_hidden)
        dit_rms, dit_max = rmsmax(audio_patch)
        raw_rms, raw_max = rmsmax(audio_patch_raw)
        nxt_rms, nxt_max = rmsmax(next_embeds)
        if wav is not None:
            w_rms, w_max = rmsmax(wav)
            wav_part = f"wav:rms={w_rms:.3f} max={w_max:.3f}"
        else:
            wav_part = "wav:rms=  -   max=  -  "
        stop_part = f"stop={prob_stop:.3f}" if prob_stop is not None else "stop=  -  "
        logger.info(
            f"[β req={req_tail} patch={patch_label}] "
            f"hid:rms={hid_rms:.3f} max={hid_max:.3f} | "
            f"dit:rms={dit_rms:.3f} max={dit_max:.3f} | "
            f"raw:rms={raw_rms:.3f} max={raw_max:.3f} | "
            f"next:rms={nxt_rms:.3f} max={nxt_max:.3f} | "
            f"{wav_part} | {stop_part}"
        )

    # ── FM helpers (per-request workspace + DiT N-step Euler) ──

    def _initialize_request_fm_state(
        self,
        state: _RequestState,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Lazy-allocate per-request FM workspace.

        Mirrors upstream _allocate_generate_state (model.py:413) but per-
        request (no workspace sharing) so concurrent requests stay
        isolated.
        """
        fm_capacity = _MAX_AUDIO_PATCHES * (_HIDDEN_PATCH_SIZE + _LATENT_PATCH_SIZE)
        state.fm_capacity = fm_capacity
        state.fm_sequence = torch.zeros(
            (1, fm_capacity, _FM_HIDDEN),
            device=device,
            dtype=dtype,
        )
        state.fm_cfg_sequence = torch.zeros(
            (1, fm_capacity, _FM_HIDDEN),
            device=device,
            dtype=dtype,
        )
        state.fm_null_g_cond = torch.zeros(
            (1, _FM_HIDDEN),
            device=device,
            dtype=dtype,
        )
        state.fm_seq_len = 0

    def _append_hidden_chunk(
        self,
        state: _RequestState,
        hidden_chunk: torch.Tensor,
    ) -> None:
        """Project last _HIDDEN_PATCH_SIZE LLM hiddens via _hidden_proj into fm_sequence.

        Mirrors upstream _append_hidden_chunk (model.py:1308): cfg buffer
        gets the null-projected counterpart (``_hidden_proj`` of zero) so
        CFG dropout sees no past hidden, only past audio history.
        """
        last_hidden = hidden_chunk[:, -_HIDDEN_PATCH_SIZE:, :]
        projected = self._hidden_proj(last_hidden)
        null_projected = self._hidden_proj(torch.zeros_like(last_hidden))
        end = state.fm_seq_len + projected.size(1)
        if end > state.fm_capacity:
            raise RuntimeError(f"FM buffer overflow on hidden append: end={end} cap={state.fm_capacity}")
        state.fm_sequence[:, state.fm_seq_len : end].copy_(projected.to(state.fm_sequence.dtype))
        state.fm_cfg_sequence[:, state.fm_seq_len : end].copy_(null_projected.to(state.fm_cfg_sequence.dtype))
        state.fm_seq_len = end

    def _append_history_chunk(
        self,
        state: _RequestState,
        latent_chunk: torch.Tensor,
    ) -> None:
        """Project audio latent via _latent_proj into fm_sequence (history append).

        Mirrors upstream _append_history_chunk (model.py:1325): cfg
        buffer gets the same value (latent history is shared between
        conditional and unconditional CFG paths — only the hidden chunk
        gets nulled).
        """
        history_latent = self._latent_proj(latent_chunk)
        end = state.fm_seq_len + history_latent.size(1)
        if end > state.fm_capacity:
            raise RuntimeError(f"FM buffer overflow on latent append: end={end} cap={state.fm_capacity}")
        state.fm_sequence[:, state.fm_seq_len : end].copy_(history_latent.to(state.fm_sequence.dtype))
        state.fm_cfg_sequence[:, state.fm_seq_len : end].copy_(history_latent.to(state.fm_cfg_sequence.dtype))
        state.fm_seq_len = end

    def _build_fm_attn_mask(
        self,
        state: _RequestState,
        total_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build DiT attention mask for one decode step.

        Mirrors upstream _build_fm_attn_mask (model.py:1203).  Layout:
          * [0, fm_seq_len) — committed history.  Causal over the past
            up to a non-causal tail block of size _HIDDEN_PATCH_SIZE that
            sees itself and the new noise patch (the just-appended hidden).
          * [latent_start, total_len) — new noise patch, bidirectional
            with the committed history and itself.
        """
        attn_mask = torch.zeros((1, total_len, total_len), device=device, dtype=torch.bool)
        latent_start = total_len - _LATENT_PATCH_SIZE
        block_start = state.fm_seq_len - _HIDDEN_PATCH_SIZE
        if block_start > 0:
            causal_mask = torch.ones((block_start, block_start), device=device, dtype=torch.bool).triu(1).logical_not()
            attn_mask[:, :block_start, :block_start] = causal_mask
        attn_mask[:, block_start : state.fm_seq_len, : state.fm_seq_len] = True
        attn_mask[:, block_start : state.fm_seq_len, latent_start:] = True
        attn_mask[:, latent_start:, : state.fm_seq_len] = True
        attn_mask[:, latent_start:, latent_start:] = True
        if latent_start > state.fm_seq_len:
            padding_indices = torch.arange(state.fm_seq_len, latent_start, device=device)
            attn_mask[:, padding_indices, padding_indices] = True
        return attn_mask

    def _build_fm_pos_ids(
        self,
        state: _RequestState,
        total_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build DiT position ids for one decode step.

        Mirrors upstream _build_fm_pos_ids (model.py:1236).  Position ids
        stay fp32 — DiT's RotaryEmbedding does its own dtype handling.
        """
        pos_ids = torch.zeros((1, total_len), device=device, dtype=torch.float32)
        latent_start = total_len - _LATENT_PATCH_SIZE
        if state.fm_seq_len > 0:
            pos_ids[:, : state.fm_seq_len] = torch.arange(
                state.fm_seq_len,
                device=device,
                dtype=pos_ids.dtype,
            )
        pos_ids[:, latent_start:] = torch.arange(
            state.fm_seq_len,
            state.fm_seq_len + _LATENT_PATCH_SIZE,
            device=device,
            dtype=pos_ids.dtype,
        )
        return pos_ids

    def _run_dit_n_step_euler(
        self,
        state: _RequestState,
        *,
        num_steps: int = _DIT_NUM_STEPS,
        guidance_scale: float = _DIT_GUIDANCE_SCALE,
    ) -> torch.Tensor:
        """DiT N-step Euler integration → audio latent patch [1, 4, 128].

        Manual Euler loop (upstream uses torchdyn.odeint; we manualize to
        avoid the dep and keep control flow explicit).  Each step:
          1. z_proj = _coordinate_proj(z), placed into the last
             _LATENT_PATCH_SIZE positions of input_sequence / cfg_sequence.
          2. Batch [conditional, unconditional] → DiT velocity field.
          3. CFG blend: v_c + guidance_scale * (v_c - v_u).
          4. Euler: z += v * dt.

        Mirrors upstream fm_solver_step (core.py:295) + _flow_matching_step_fm
        (core.py:463), reduced to a fixed-step Euler integrator.
        """
        assert state.fm_sequence is not None
        assert state.fm_cfg_sequence is not None
        assert state.fm_null_g_cond is not None
        device = state.fm_sequence.device
        dtype = state.fm_sequence.dtype
        total_len = state.fm_seq_len + _LATENT_PATCH_SIZE

        # Workspace: committed history + 4 noise slots (overwritten per step).
        input_sequence = torch.zeros((1, total_len, _FM_HIDDEN), device=device, dtype=dtype)
        input_sequence[:, : state.fm_seq_len] = state.fm_sequence[:, : state.fm_seq_len]
        cfg_sequence = torch.zeros((1, total_len, _FM_HIDDEN), device=device, dtype=dtype)
        cfg_sequence[:, : state.fm_seq_len] = state.fm_cfg_sequence[:, : state.fm_seq_len]

        attn_mask = self._build_fm_attn_mask(state, total_len, device)
        pos_ids = self._build_fm_pos_ids(state, total_len, device)

        g_cond = state.g_cond if state.g_cond is not None else state.fm_null_g_cond
        g_cond = g_cond.to(device=device, dtype=dtype)
        g_cond_batched = torch.cat([g_cond, torch.zeros_like(g_cond)], dim=0)

        latent_start = total_len - _LATENT_PATCH_SIZE
        # fp32 ODE integration (same fix as Ming's CFM sampler, PR #4341):
        # bf16's 7-bit mantissa loses the small per-step increments in the
        # z += v*dt accumulation (hidden-state cos drift vs upstream past
        # step ~7) and quantizes the linspace timesteps.  Integration
        # state stays fp32; DiT matmuls still run bf16 under the autocast
        # block below, and the result is cast back to the sequence dtype
        # on return.
        # Deterministic per-request noise (voxcpm2 _fill_deterministic_cfm_
        # noise pattern): hash seed:request_key:draw# into a private
        # Generator instead of the global CUDA RNG, so outputs reproduce
        # run-to-run and concurrent requests cannot perturb each other's
        # noise streams.  request_key strips the engine's per-run "<idx>_"
        # uuid suffix so replay across runs keys on the stable batch index.
        request_key = state.request_id.split("_", 1)[0]
        if not request_key.isdigit():
            request_key = state.request_id
        noise_key = f"{_DIT_NOISE_SEED}:{request_key}:{state.noise_step}".encode()
        digest = hashlib.blake2b(noise_key, digest_size=8).digest()
        gen = torch.Generator(device=device)
        gen.manual_seed(int.from_bytes(digest, "little") & 0x7FFF_FFFF_FFFF_FFFF)
        z = torch.empty((1, _LATENT_PATCH_SIZE, _LATENT_DIM), device=device, dtype=torch.float32)
        z.normal_(generator=gen)
        state.noise_step += 1
        dt = 1.0 / num_steps
        times = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=torch.float32)

        # Match upstream's autocast wrapper (model.py:280).  DiT's
        # TimestepEmbedder internally forces fp32 (`.float()` on freqs);
        # without autocast the fp32 output hits a bf16 Linear → dtype
        # mismatch.  autocast bridges this for matmuls inside DiT.
        use_amp = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
        with torch.autocast(
            device_type=device.type if device.type == "cuda" else "cuda",
            dtype=dtype if use_amp else torch.float32,
            enabled=use_amp,
        ):
            for step in range(num_steps):
                t = times[step].reshape(1)
                z_proj = self._coordinate_proj(z)
                z_c = input_sequence.clone()
                z_c[:, latent_start:] = z_proj
                z_u = cfg_sequence.clone()
                z_u[:, latent_start:] = z_proj
                z_batched = torch.cat([z_c, z_u], dim=0)
                t_batched = t.repeat(2)
                vt = self._head(
                    x=z_batched,
                    timesteps=t_batched,
                    attn_mask=attn_mask,
                    pos_ids=pos_ids,
                    g_cond=g_cond_batched,
                )
                # Upcast DiT output before CFG blend + Euler update so the
                # accumulation arithmetic stays fp32 (autocast only affects
                # matmuls; these pointwise ops keep their input dtype).
                vt = vt[:, latent_start:].float()
                vt_c, vt_u = vt[0:1], vt[1:2]
                velocity = vt_c + guidance_scale * (vt_c - vt_u)
                z = z + velocity * dt
        return z.to(dtype)

    def _run_patch_encoder_loopback(
        self,
        state: _RequestState,
        audio_patch_raw: torch.Tensor,
    ) -> torch.Tensor:
        """Decode one audio latent patch → 1 LLM input embed.

        Mirrors upstream _consume_audio_patch (model.py:1557) — patch_encoder
        side only.  Lazy-allocates state.patch_encoder_state on first call.
        Returns ``llm_embedding`` of shape ``[1, 1, llm_hidden]`` — the
        _PATCH_ENCODER_OUT_DS_RATE encoder positions fold into the feature
        axis before ``out_proj``, not into separate output positions.

        ``audio_patch_raw`` must already be denormalized (raw VAE latent
        space).  Caller (_finish_decode) runs io_helper.denormalize on
        DiT's normalized output before passing in.
        """
        if state.patch_encoder_state is None:
            state.patch_encoder_state = self._patch_encoder.init_decode_state(
                max_audio_patch_count=_MAX_AUDIO_PATCHES,
                batch_size=1,
                device=audio_patch_raw.device,
                dtype=audio_patch_raw.dtype,
            )
        pe_state = state.patch_encoder_state
        positions = (
            torch.arange(
                _PATCH_ENCODER_OUT_DS_RATE,
                device=audio_patch_raw.device,
                dtype=torch.long,
            )
            + pe_state.seq_len
        )
        llm_embedding, conv_tail = self._patch_encoder.decode_patch(
            audio_patch_raw,
            pe_state.conv_tail,
            pe_state.layer_caches,
            positions,
        )
        pe_state.conv_tail.copy_(conv_tail)
        pe_state.seq_len += _PATCH_ENCODER_OUT_DS_RATE
        return llm_embedding

    def _run_vocoder_stream_step(
        self,
        state: _RequestState,
        audio_patch_raw: torch.Tensor,
    ) -> torch.Tensor:
        """Streaming AudioVAE decode of one latent patch (review M2).

        Mirrors upstream generate_audio_stream (model.py:1881-1900): a
        per-request BigVGANStreamState carries the decoder's LSTM hidden
        and sliding latent window across steps, so each patch is decoded
        with real left context instead of inference_from_latents'
        zero-padded isolated boundary.  Output lags input by
        decoder.stream_lookahead (2 frames): the first call emits 2 frames
        (3840 samples), steady state 4 (7680); the stop path in
        _finish_decode drains the tail via stream_flush.

        audio_patch_raw: [1, T=4, D=128] denormalized latents.
        Returns wav chunk [1, 1, n_samples] (n_samples may be 0).
        """
        if state.vocoder_stream_state is None:
            state.vocoder_stream_state = self._audio_vae.init_stream_state(
                batch_size=1,
                chunk_size=_LATENT_PATCH_SIZE,
            )
        # stream_step expects [B, latent_dim, T]; the vocoder is pinned
        # fp32 (__init__), so cast the (possibly bf16) latents up.
        return self._audio_vae.stream_step(
            audio_patch_raw.transpose(1, 2).float(),
            state.vocoder_stream_state,
        )

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | OmniOutput,
        **kwargs: Any,
    ) -> OmniOutput:
        """Drain self._audio_queue into the OmniOutput's multimodal_outputs.

        Output contract is **delta** (SKILL I1): each call emits only
        audio produced since the previous step; the engine's
        _consolidate_multimodal_tensors concatenates across steps for
        offline consumers.

        Same-request multi-chunk: chunks for the same req_id within one
        step are cat'd together.  Multi-request: one entry per req_id
        in multimodal_outputs["model_outputs"].

        Note: ``sr`` is not a declared OmniPayload key but the engine /
        serving consumer reads it (voxcpm2 convention).
        """
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        mm: dict[str, Any] = {}
        if self._audio_queue:
            audio_by_req: dict[str, torch.Tensor] = {}
            for req_id, audio in self._audio_queue:
                if audio is None:
                    continue
                if req_id in audio_by_req:
                    audio_by_req[req_id] = torch.cat(
                        [audio_by_req[req_id].reshape(-1), audio.reshape(-1)],
                        dim=0,
                    )
                else:
                    audio_by_req[req_id] = audio
            if audio_by_req:
                sr = torch.tensor(self._sample_rate, dtype=torch.int32)
                ready_req_ids = list(audio_by_req)
                chunks = [audio_by_req[req_id].reshape(-1) for req_id in ready_req_ids]
                mm["model_outputs"] = chunks
                mm["sr"] = [sr for _ in ready_req_ids]
                # sparse_audio: ["1"] flags GPUARModelRunner to skip the default
                # payload["hidden"] = scaffold_hidden injection (gpu_ar_model_
                # runner.py:1114).  Otherwise prefill scaffold (24 hidden × 1536
                # = 29184 numbers observed) leaks into mm["audio"] via the
                # output_processor "hidden" → target_key="audio" rename
                # (output_processor.py:84), prepending ~0.61 s of noise.
                mm["meta"] = {"req_id": ready_req_ids, "sparse_audio": ["1"]}
            self._audio_queue.clear()
        else:
            # Empty-audio step (prefill) still needs the sparse marker so the
            # engine doesn't bleed scaffold_hidden into mm["audio"].
            mm["model_outputs"] = []
            mm["sr"] = []
            mm["meta"] = {"req_id": [], "sparse_audio": ["1"]}

        if self._beta_trace:
            payload = mm.get("model_outputs")
            if payload:
                for i, t in enumerate(payload):
                    logger.info(
                        f"[β-emit] mm[model_outputs][{i}]: shape={tuple(t.shape)} "
                        f"dtype={t.dtype} samples={t.numel()} "
                        f"rms={t.float().pow(2).mean().sqrt().item():.4f} "
                        f"max={t.abs().max().item():.4f}"
                    )
            else:
                logger.info("[β-emit] mm empty / no model_outputs")
            mo_kind = type(model_outputs).__name__
            mo_info = f"shape={tuple(model_outputs.shape)}" if hasattr(model_outputs, "shape") else "no_shape"
            logger.info(f"[β-emit] text_hidden_states arg: {mo_kind} {mo_info}")

        return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs=mm)

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        """Per-step stop signal logits (vLLM sampling protocol).

        Encodes continue/stop into ``logits[i, 0]`` / ``logits[i, 1]``
        (voxcpm2 convention — rest of vocab stays at -inf so the sampler
        picks between the two slots).  Source of stop logits is
        ``self._results_queue``, populated by forward()'s _finish_decode.
        When the queue is empty (preprocess-only steps), defaults to
        all-continue.
        """
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None

        bsz = hidden_states.shape[0]
        logits = torch.full(
            (bsz, self.config.vocab_size),
            float("-inf"),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        if self._results_queue:
            for i, (req_id, stop_logits) in enumerate(self._results_queue):
                if i >= bsz:
                    break
                state = self._active_states.get(req_id)
                if stop_logits is not None:
                    if state is not None and state.is_stopping:
                        logits[i, 0] = 0.0
                        logits[i, 1] = 1.0
                    else:
                        # is_stopping=False means prob_stop <= 0.8 — force
                        # continue.  Feeding the raw softmax (continue, stop)
                        # pair as logits would let the greedy sampler compare
                        # them directly, silently lowering the effective stop
                        # threshold to 0.5 (early stop, swallowed endings).
                        logits[i, 0] = 1.0
                    if state is not None:
                        state.precomputed_stop_logits = None
                else:
                    # No stop signal pushed this step (prefill placeholder).
                    # Default to continue — stop is driven by eos_proj.
                    logits[i, 0] = 1.0  # continue
            self._results_queue.clear()
        else:
            logits[:, 0] = 1.0  # all continue

        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load all dots.tts checkpoint weights — step 6b.

        Routing branches in a single pass over the input iterator (it can only
        be consumed once):

        * **AudioVAE**: keys from ``vocoder.safetensors`` (no parent prefix).
          Matched via set membership against ``self._audio_vae.state_dict()``;
          acts as the final catch-all so it must run last.
        * **DiT**: keys from ``model.safetensors`` under ``velocity_field_predictor.``.
        * **patch_encoder**: keys from ``model.safetensors`` under ``patch_encoder.``.
        * **Five projectors** (``hidden_proj`` / ``latent_proj`` /
          ``coordinate_proj`` / ``xvec_proj`` / ``eos_proj``): each lives at
          its own top-level prefix; dispatched through a shared table.
        * **Qwen2 base LM**: keys under ``llm.model.``, handed off to vLLM's
          Qwen2Model fuser (q/k/v_proj -> qkv_proj, gate/up_proj ->
          gate_up_proj).  ``llm.lm_head.*`` is silently dropped (soar uses
          ``tie_word_embeddings=True`` and our ``self.model`` is a Qwen2Model
          with no lm_head module).
        * **CAM++ speaker encoder**: keys from ``speaker_encoder.safetensors``
          (no parent prefix, same convention as AudioVAE).  Routed by set
          membership against ``self._speaker_encoder.state_dict()`` (keys are
          ``model.*`` / ``resample.kernel``, no distinguishing top-level
          prefix).  **Placed before the AudioVAE fallback** so its keys don't
          fall through to the catch-all.
        """
        vae_state_keys = set(self._audio_vae.state_dict().keys())
        dit_state_keys = set(self._head.state_dict().keys())
        patch_state_keys = set(self._patch_encoder.state_dict().keys())
        speaker_state_keys = set(self._speaker_encoder.state_dict().keys())

        # (top-level prefix in checkpoint, local attr name, local nn.Module)
        projector_specs: list[tuple[str, str, nn.Module]] = [
            ("hidden_proj.", "_hidden_proj", self._hidden_proj),
            ("latent_proj.", "_latent_proj", self._latent_proj),
            ("coordinate_proj.", "_coordinate_proj", self._coordinate_proj),
            ("xvec_proj.", "_xvec_proj", self._xvec_proj),
            ("eos_proj.", "_eos_proj", self._eos_proj),
        ]
        projector_state_keys = {prefix: set(mod.state_dict().keys()) for prefix, _, mod in projector_specs}
        projector_matched: dict[str, list[tuple[str, torch.Tensor]]] = {prefix: [] for prefix, _, _ in projector_specs}

        matched_vae: list[tuple[str, torch.Tensor]] = []
        matched_dit: list[tuple[str, torch.Tensor]] = []
        matched_patch: list[tuple[str, torch.Tensor]] = []
        matched_llm: list[tuple[str, torch.Tensor]] = []
        matched_speaker: list[tuple[str, torch.Tensor]] = []
        skipped_lm_head = 0

        DIT_PREFIX = "velocity_field_predictor."
        PATCH_PREFIX = "patch_encoder."
        LLM_MODEL_PREFIX = "llm.model."
        LLM_LM_HEAD_PREFIX = "llm.lm_head."
        for name, tensor in weights:
            if name.startswith(DIT_PREFIX):
                candidate = name[len(DIT_PREFIX) :]
                if candidate in dit_state_keys:
                    matched_dit.append((candidate, tensor))
                continue
            if name.startswith(PATCH_PREFIX):
                candidate = name[len(PATCH_PREFIX) :]
                if candidate in patch_state_keys:
                    matched_patch.append((candidate, tensor))
                continue
            if name.startswith(LLM_LM_HEAD_PREFIX):
                # Qwen2.5-1.5B-Base has tie_word_embeddings=True, and our
                # ``self.model`` is a Qwen2Model (no lm_head submodule).  Drop.
                skipped_lm_head += 1
                continue
            if name.startswith(LLM_MODEL_PREFIX):
                candidate = name[len(LLM_MODEL_PREFIX) :]
                # Hand off untouched — Qwen2Model.load_weights does its own
                # fusion (q/k/v_proj -> qkv_proj, gate/up_proj -> gate_up_proj)
                # and tolerates unknown keys, so no local key set to gate.
                matched_llm.append((candidate, tensor))
                continue
            # Projectors: 5 candidate top-level prefixes, first hit wins.
            hit_projector = False
            for prefix, _attr, _mod in projector_specs:
                if name.startswith(prefix):
                    candidate = name[len(prefix) :]
                    if candidate in projector_state_keys[prefix]:
                        projector_matched[prefix].append((candidate, tensor))
                    hit_projector = True
                    break
            if hit_projector:
                continue
            # Speaker encoder: set-membership match (no distinguishing prefix
            # in checkpoint; placed before the AudioVAE fallback so its keys
            # don't get silently dropped by the catch-all).
            if name in speaker_state_keys:
                matched_speaker.append((name, tensor))
                continue
            candidate = name[len("vocoder.") :] if name.startswith("vocoder.") else name
            if candidate in vae_state_keys:
                matched_vae.append((candidate, tensor))

        loaded: set[str] = set()

        if matched_vae:
            vae_loader = AutoWeightsLoader(self._audio_vae)
            loaded_vae = vae_loader.load_weights(iter(matched_vae))
            loaded.update(f"_audio_vae.{name}" for name in loaded_vae)
            logger.info(
                "DotsTTS load_weights: loaded %d/%d AudioVAE tensors.",
                len(loaded_vae),
                len(vae_state_keys),
            )

        if matched_dit:
            dit_loader = AutoWeightsLoader(self._head)
            loaded_dit = dit_loader.load_weights(iter(matched_dit))
            loaded.update(f"_head.{name}" for name in loaded_dit)
            logger.info(
                "DotsTTS load_weights: loaded %d/%d DiT tensors.",
                len(loaded_dit),
                len(dit_state_keys),
            )

        if matched_patch:
            patch_loader = AutoWeightsLoader(self._patch_encoder)
            loaded_patch = patch_loader.load_weights(iter(matched_patch))
            loaded.update(f"_patch_encoder.{name}" for name in loaded_patch)
            logger.info(
                "DotsTTS load_weights: loaded %d/%d patch_encoder tensors.",
                len(loaded_patch),
                len(patch_state_keys),
            )

        any_projector_matched = False
        for prefix, attr_name, mod in projector_specs:
            matched = projector_matched[prefix]
            if not matched:
                continue
            any_projector_matched = True
            proj_loader = AutoWeightsLoader(mod)
            loaded_proj = proj_loader.load_weights(iter(matched))
            loaded.update(f"{attr_name}.{name}" for name in loaded_proj)
            logger.info(
                "DotsTTS load_weights: loaded %d/%d %s tensors.",
                len(loaded_proj),
                len(projector_state_keys[prefix]),
                prefix[:-1],
            )

        if matched_llm:
            # vLLM's Qwen2Model.load_weights does its own fusion + tolerates
            # unknown keys.  It returns the set of param names it actually
            # accepted; re-prefix with ``model.`` for our return value (our
            # local attr is ``self.model``).
            loaded_llm = self.model.load_weights(iter(matched_llm))
            loaded.update(f"model.{name}" for name in loaded_llm)
            logger.info(
                "DotsTTS load_weights: loaded %d Qwen2 tensors (fed %d; %d llm.lm_head.* skipped — tied embeddings).",
                len(loaded_llm),
                len(matched_llm),
                skipped_lm_head,
            )

        if matched_speaker:
            speaker_loader = AutoWeightsLoader(self._speaker_encoder)
            loaded_speaker = speaker_loader.load_weights(iter(matched_speaker))
            loaded.update(f"_speaker_encoder.{name}" for name in loaded_speaker)
            logger.info(
                "DotsTTS load_weights: loaded %d/%d CAM++ speaker_encoder tensors.",
                len(loaded_speaker),
                len(speaker_state_keys),
            )

        if (
            not matched_vae
            and not matched_dit
            and not matched_patch
            and not any_projector_matched
            and not matched_llm
            and not matched_speaker
        ):
            logger.warning(
                "DotsTTS load_weights: no AudioVAE / DiT / "
                "patch_encoder / projector / Qwen2 / speaker_encoder keys matched."
            )

        return loaded
