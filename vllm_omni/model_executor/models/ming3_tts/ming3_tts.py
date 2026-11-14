# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""vLLM-Omni wrapper for Ming3-TTS: two stages, or one with the AudioVAE decode in Stage 0."""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.bailing_moe_v3 import BailingMoeV3ForCausalLM
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid, SupportsPP
from vllm.model_executor.models.utils import init_vllm_registered_model
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.model_loader.weight_utils import resolve_model_to_local_path
from vllm_omni.model_executor.models.common.ming.audio_vae import AudioVAE
from vllm_omni.model_executor.models.ming_tts import config_ming_tts as state_keys
from vllm_omni.model_executor.models.ming_tts.audio_prep import (
    _find_audio_placeholder_positions,
    _find_speaker_placeholder_positions,
    _initial_history,
    _normalize_prompt_waveform,
    _take_scalar,
    pad_prompt_waveform,
)
from vllm_omni.model_executor.models.ming_tts.ming_tts import (
    _ModelSampleAdapter,
)
from vllm_omni.model_executor.models.ming_tts.prompt_encoder import _load_prompt_encoder, _resolve_prompt_latents
from vllm_omni.utils.device_copy import to_device_nonblocking

from .batched_campplus import BatchedCampplus
from .config_ming3_tts import Ming3TTSRuntimeConfig, get_ming3_backbone_config
from .constants import (
    KEY_CFG,
    KEY_MAX_DECODE_STEPS,
    KEY_MIN_DECODE_STEPS,
    KEY_PROMPT_AUDIO_START,
    KEY_REQUEST_ID,
    KEY_SIGMA,
    KEY_SPEAKER_EMBEDDING,
    KEY_SPEAKER_POSITIONS,
    KEY_SPEAKER_WAVEFORM,
    KEY_STOP_REASON,
    KEY_TEMPERATURE,
)
from .packed_prompt_encoder import PackedPromptEncoder
from .prompt_encoder_graph import DEFAULT_BUCKET_GROUPS, PROJECTION_BUCKET_ROWS, PaddedBucketGraphs
from .speaker_extractor import (
    extract_speaker_embeddings,
    finish_speaker_embeddings,
    plan_speaker_embeddings,
    resolve_speaker_embeddings,
)
from .step_vae_decoder import StepVAEDecoder
from .weight_accounting import validate_ming3_checkpoint_keys

logger = init_logger(__name__)


@dataclass
class _PrefillConditioning:
    """Request-owned conditioning reused across prefill slices and recomputation."""

    speaker_positions: tuple[int, ...]
    speaker_embeds: torch.Tensor
    prompt_start: int
    prompt_embeds: torch.Tensor | None
    history: torch.Tensor


class Ming3TTSForConditionalGeneration(
    nn.Module,
    HasInnerState,
    IsHybrid,
    SupportsPP,
    CustomProcessMixin,
):
    packed_modules_mapping = BailingMoeV3ForCausalLM.packed_modules_mapping
    requires_request_sample_eligibility = True
    # Stage 1 consumes only the latent patches; skip the per-step hidden-state D2H.
    omni_pooler_payload_include_hidden = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        del prefix
        self.vllm_config = vllm_config
        self.ming_config = Ming3TTSRuntimeConfig.from_hf_config(vllm_config.model_config.hf_config)
        self.ming_config.validate()
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.requires_raw_input_tokens = False
        self.model_stage = vllm_config.model_config.model_stage
        self._prompt_encoder = None
        self._prompt_encoder_graphs: PaddedBucketGraphs | None = None
        self._prompt_projection_graphs: PaddedBucketGraphs | None = None
        self._packed_prompt_encoder: PackedPromptEncoder | None = None
        # Prompt work batched across the prefill rows of one step (MRv2): encoder
        # outputs by request id, and the projections the rows left to the batch.
        self._prefetched_prompt_outputs: dict[str, tuple[int, torch.Tensor]] = {}
        self._prompt_request_id: str | None = None
        # Content key of the prefilling request's reference audio; its encoder
        # output is cached under it (posterior sampling stays per request).
        self._prompt_cache_key: Any = None
        # V1 completion carries runner IDs; model state may use API IDs.
        self._request_state_ids: dict[str, str] = {}
        self._speaker_extractor = None
        self._prefill_conditioning: dict[str, _PrefillConditioning] = {}
        # CAM++ on the GPU (see batched_campplus), and the embeddings it made
        # for this step's new requests, by request id.
        self._speaker_encoder: BatchedCampplus | None = None
        self._prefetched_speakers: dict[str, list[torch.Tensor]] = {}

        # Single-stage serving: Stage 0 also loads the AudioVAE decoder and
        # returns PCM (see model_state.Ming3ModelState).
        self.single_stage = self.model_stage == "llm" and vllm_config.model_config.engine_output_type == "audio"
        self.step_vae: StepVAEDecoder | None = None
        if self.single_stage:
            # The per-step PCM is allocated by each step; the runner need not copy it.
            self.mm_outputs_fresh_per_step = True
        if self.model_stage == "llm":
            from vllm_omni.utils.speaker_cache import get_speaker_cache

            self._speaker_cache = get_speaker_cache()
            self.model = init_vllm_registered_model(
                vllm_config=vllm_config,
                architectures=["Ming3TTSLLMModel"],
            )
            self.has_preprocess = True
            self.has_postprocess = True
            self.set_custom_preprocess(self.preprocess)
            self.set_custom_postprocess(self.postprocess)
        elif self.model_stage == "audio_vae":
            self.model = init_vllm_registered_model(
                vllm_config=vllm_config,
                architectures=["Ming3TTSAudioVAEModel"],
            )
            self.requires_raw_input_tokens = True
        else:
            raise ValueError(f"Invalid Ming3 model_stage={self.model_stage!r}")
        self.make_empty_intermediate_tensors = getattr(self.model, "make_empty_intermediate_tensors", lambda: None)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        return BailingMoeV3ForCausalLM.get_mamba_state_shape_from_config(get_ming3_backbone_config(vllm_config))

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        return BailingMoeV3ForCausalLM.get_mamba_state_dtype_from_config(get_ming3_backbone_config(vllm_config))

    @classmethod
    def get_mamba_state_copy_func(cls):
        return BailingMoeV3ForCausalLM.get_mamba_state_copy_func()

    @cached_property
    def sampler(self):
        if hasattr(self.model, "sample"):
            return _ModelSampleAdapter(self.model)
        if hasattr(self.model, "sampler"):
            return self.model.sampler
        return Sampler()

    def embed_input_ids(self, input_ids: torch.Tensor, **kwargs: Any):
        return self.model.embed_input_ids(input_ids=input_ids, **kwargs)

    def forward(self, *args: Any, **kwargs: Any):
        return self.model(*args, **kwargs)

    def make_omni_output(self, model_outputs: Any, **kwargs: Any) -> Any:
        # Stage 0 returns backbone hidden states from (possibly graph-replayed)
        # forward; the flow head turns them into latent outputs here.
        make_output = getattr(self.model, "make_omni_output", None)
        return make_output(model_outputs, **kwargs) if callable(make_output) else model_outputs

    def compute_logits(self, hidden_states, sampling_metadata=None):
        return self.model.compute_logits(hidden_states, sampling_metadata=sampling_metadata)

    def sample(self, logits, sampling_metadata):
        return self.model.sample(logits, sampling_metadata)

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        state_ids = set(finished_req_ids)
        state_ids.update(self._request_state_ids.pop(request_id, request_id) for request_id in finished_req_ids)
        for request_id in state_ids:
            self._prefill_conditioning.pop(request_id, None)
            self._prefetched_prompt_outputs.pop(request_id, None)
            self._prefetched_speakers.pop(request_id, None)
        callback = getattr(self.model, "on_requests_finished", None)
        if callback is not None:
            callback(state_ids)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = list(weights)
        validate_ming3_checkpoint_keys(name for name, _ in weights)
        allowed = (
            ("model.", "linear_proj_audio.", "flowloss.", "stop_head.", "spk_head.")
            if self.model_stage == "llm"
            else ("audio.decoder.",)
        )
        selected = [(name, value) for name, value in weights if name.startswith(allowed)]
        if not selected:
            raise RuntimeError(f"Ming3 {self.model_stage} stage received no weights with prefixes {allowed}")
        loaded = self.model.load_weights(selected)
        loaded_names = {f"model.{name}" for name in loaded}
        if self.single_stage:
            loaded_names |= self._load_step_vae(
                [(name, value) for name, value in weights if name.startswith("audio.decoder.")]
            )
        if self.model_stage == "llm":
            self._capture_prompt_encoder_graphs()
            self._load_speaker_encoder()
            if self._prompt_encoder is not None:
                # The prompt encoder is loaded from the checkpoint's
                # audio.encoder.* tensors by _load_prompt_encoder.
                loaded_names |= {f"_prompt_encoder.{name}" for name, _ in self._prompt_encoder.named_parameters()}
                loaded_names |= {f"_prompt_encoder.{name}" for name, _ in self._prompt_encoder.named_buffers()}
        return loaded_names

    def _load_step_vae(self, weights: list[tuple[str, torch.Tensor]]) -> set[str]:
        """Build the AudioVAE decoder for single-stage serving; return the loaded parameter names."""
        param = next(self.parameters())
        vae = AudioVAE(self.ming_config.audio_tokenizer_config, build_encoder=False).eval()
        state = vae.state_dict()
        with torch.no_grad():
            for name, value in weights:
                key = name[len("audio.") :]
                if key not in state:
                    raise RuntimeError(f"Ming3 single-stage AudioVAE received an unknown tensor: {name}")
                state[key].copy_(value)
        missing = {f"audio.{key}" for key in state} - {name for name, _ in weights}
        if missing:
            raise RuntimeError(f"Ming3 single-stage AudioVAE left tensors unloaded: {sorted(missing)[:8]}")
        vae.to(device=param.device, dtype=self.vllm_config.model_config.dtype)
        vae.decoder.upsampling.frame_holdback = True
        # Registered as a submodule so the runner counts its memory before profiling.
        self.audio_vae = vae
        self.step_vae = StepVAEDecoder(vae.decoder, max_slots=int(self.vllm_config.scheduler_config.max_num_seqs))
        if not self.vllm_config.model_config.enforce_eager:
            self.step_vae.capture_graphs()
        return {f"audio_vae.{name}" for name in vae.state_dict()}

    def _load_speaker_encoder(self) -> None:
        # Reference clips get their CAM++ embeddings here, batched over a
        # step's new requests on the GPU, instead of one by one on the CPU.
        device = next(self.parameters()).device
        if device.type != "cuda":
            return
        try:
            model_dir = resolve_model_to_local_path(
                self.vllm_config.model_config.model, allow_download=False, allow_patterns=["campplus.onnx"]
            )
            onnx_path = os.path.join(model_dir, "campplus.onnx")
            if not os.path.exists(onnx_path):
                return
            start = time.perf_counter()
            encoder = BatchedCampplus.from_onnx(onnx_path, device)
            if not self.vllm_config.model_config.enforce_eager:
                encoder.capture_graphs()
        except Exception:
            logger.warning("Ming3 GPU speaker encoder unavailable; CAM++ runs on the CPU.", exc_info=True)
            return
        self._speaker_encoder = encoder
        logger.info("Ming3 GPU speaker encoder ready in %.1f ms", (time.perf_counter() - start) * 1000.0)

    def _capture_prompt_encoder_graphs(self) -> None:
        # Load the prompt-audio encoder with the model (instead of on the first
        # reference request) and graph it together with the latent projection,
        # so a prefill does not launch them kernel by kernel.
        if self.vllm_config.model_config.enforce_eager or self.ming_config.audio_tokenizer_config is None:
            return
        device = next(self.parameters()).device
        if device.type != "cuda":
            return
        try:
            encoder = _load_prompt_encoder(self).encoder
            encoder_dtype = next(encoder.parameters()).dtype
            group_samples = int((float(self.ming_config.sample_rate) / 12.5) * int(self.ming_config.patch_size))
            packed = PackedPromptEncoder(encoder, max(DEFAULT_BUCKET_GROUPS) * group_samples // encoder.hop_size)
            encoder_graphs = PaddedBucketGraphs(
                packed,
                torch.zeros((1, group_samples), device=device, dtype=encoder_dtype),
                in_dim=1,
                in_unit=group_samples,
                out_dim=1,
            )
            projection_graphs = PaddedBucketGraphs(
                self.model.linear_proj_audio,
                torch.zeros(
                    (1, self.ming_config.patch_size, self.ming_config.latent_dim),
                    device=device,
                    dtype=self.model.fm_dtype,
                ),
                in_dim=0,
                in_unit=1,
                out_dim=0,
                # Project the complete reference once, independently of prefill chunking.
                bucket_groups=PROJECTION_BUCKET_ROWS,
            )
            start = time.perf_counter()
            pool = torch.cuda.graph_pool_handle()
            encoder_graphs.capture(pool)
            projection_graphs.capture(pool)
            # Batched and single-clip graphs may replay in either order, so
            # their intermediate buffers must not share a graph pool.
            packed.capture_graphs(torch.cuda.graph_pool_handle())
        except Exception:
            logger.warning("Ming3 prompt encoder CUDA graph capture failed; using eager.", exc_info=True)
            return
        # Kept on the (non-Module) graph runner so it is not registered as a submodule.
        encoder_graphs.target = encoder
        self._prompt_encoder_graphs = encoder_graphs
        self._prompt_projection_graphs = projection_graphs
        self._packed_prompt_encoder = packed
        logger.info(
            "Ming3 prompt encoder and projection CUDA graphs captured for %s latent groups in %.1f ms",
            encoder_graphs.bucket_groups,
            (time.perf_counter() - start) * 1000.0,
        )

    def _prompt_encoder_cache_key(self, info: dict[str, Any]) -> Any:
        """Cache key of the prompt encoder output: the reference clips' content keys from the API."""
        audio_keys = info.get("speaker_audio_keys")
        if info.get("prompt_waveform") is None or not audio_keys or not all(audio_keys):
            return None
        model = self.vllm_config.model_config.model
        return self._speaker_cache.make_cache_key("|".join(audio_keys), model_type=f"ming3_tts.prompt:{model}")

    def _prompt_encoder_forward(self, encoder: Any, waveform: torch.Tensor) -> torch.Tensor | None:
        key = self._prompt_cache_key
        cached = self._speaker_cache.get(key) if key is not None else None
        if cached is not None and cached["samples"] == waveform.shape[-1]:
            return cached["encoder_output"]
        prefetched = self._prefetched_prompt_outputs.pop(self._prompt_request_id, None)
        if prefetched is not None and prefetched[0] == waveform.shape[-1]:
            output = prefetched[1]
        else:
            graphs = self._prompt_encoder_graphs
            if graphs is None or getattr(graphs, "target", None) is not encoder.encoder:
                return None
            param = next(encoder.encoder.parameters())
            output = graphs(to_device_nonblocking(waveform, param.device).to(param.dtype))
        if key is not None and output is not None:
            # A graph output is its static buffer; the cache keeps a copy.
            self._speaker_cache.put(key, {"encoder_output": output.clone(), "samples": waveform.shape[-1]})
        return output

    def _project_prompt_patches(self, patches: torch.Tensor) -> torch.Tensor:
        graphs = self._prompt_projection_graphs
        projected = graphs(patches) if graphs is not None else None
        return self.model.linear_proj_audio(patches) if projected is None else projected

    def prefetch_prompt_encodings(self, infos: list[dict[str, Any]]) -> None:
        """Encode the reference clips of several new requests in one packed pass.

        The per-request prefill then takes its encoder output by request id
        instead of replaying the encoder graph once per clip; posterior
        sampling stays per request. A single clip keeps the graph path.
        """
        infos = [info for info in infos if _request_id(info) not in self._prefill_conditioning]
        self._prefetch_speaker_embeddings(infos)
        packed = self._packed_prompt_encoder
        if packed is None or len(infos) < 2:
            return
        encoder_param = next(packed.encoder.parameters())
        max_samples = packed.max_frames * packed.encoder.hop_size
        request_ids, clips, keys = [], [], set()
        for info in infos:
            raw = info.get("prompt_waveform")
            if raw is None or info.get(state_keys.KEY_PROMPT_LATENTS, info.get("prompt_latents")) is not None:
                continue
            if _take_scalar(info.get("mute_speaker"), 0) or _take_scalar(info.get("no_speaker_slot"), 0):
                continue
            # A cached reference needs no pass; a reference shared within the step
            # is encoded once, by the first request, and cached for the others.
            key = self._prompt_encoder_cache_key(info)
            if key is not None and (key in keys or self._speaker_cache.get(key) is not None):
                continue
            if key is not None:
                keys.add(key)
            # The same normalization and padding as _encode_prompt_waveform_to_latents.
            waveform = pad_prompt_waveform(
                _normalize_prompt_waveform(raw, target_sr=self.ming_config.sample_rate),
                patch_size=self.ming_config.patch_size,
                sample_rate=self.ming_config.sample_rate,
            )
            if waveform.shape[-1] > max_samples:
                continue
            request_ids.append(_request_id(info))
            # The packed pass uploads all clips in one copy.
            clips.append(waveform.to(encoder_param.dtype))
        if len(clips) < 2:
            return
        with torch.inference_mode():
            outputs = packed.encode_clips(clips)
        for request_id, clip, output in zip(request_ids, clips, outputs, strict=True):
            self._prefetched_prompt_outputs[request_id] = (clip.shape[-1], output)

    def _prefetch_speaker_embeddings(self, infos: list[dict[str, Any]]) -> None:
        """Embed the reference clips of several new requests in one GPU pass."""
        if self._speaker_encoder is None:
            return
        plans, clips, index_of = [], [], {}
        for info in infos:
            if info.get(KEY_SPEAKER_WAVEFORM) is None or info.get(KEY_SPEAKER_EMBEDDING) is not None:
                continue
            if _take_scalar(info.get("mute_speaker"), 0) or _take_scalar(info.get("no_speaker_slot"), 0):
                continue
            try:
                plan = plan_speaker_embeddings(self, info)
            except (RuntimeError, ValueError):
                continue  # The request's prefill reports it.
            rows = []
            for _, waveform, rate, key in plan.clips:
                # Requests of one step may share a reference clip.
                if key is None or key not in index_of:
                    clips.append((waveform, rate))
                    if key is not None:
                        index_of[key] = len(clips) - 1
                rows.append(index_of[key] if key is not None else len(clips) - 1)
            if isinstance(request_id := _request_id(info), str):
                plans.append((request_id, plan, rows))
        embeddings = extract_speaker_embeddings(self, clips) if clips else []
        for request_id, plan, rows in plans:
            self._prefetched_speakers[request_id] = finish_speaker_embeddings(
                self, plan, [embeddings[row] for row in rows]
            )

    def end_prompt_batch(self) -> None:
        """Drop encoder outputs no prefill consumed."""
        self._prefetched_prompt_outputs.clear()
        self._prefetched_speakers.clear()

    def create_omni_model_state(self, vllm_config: VllmConfig, encoder_cache: Any, device: torch.device):
        from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

        if self.model_stage != "llm":
            return OmniModelState(vllm_config, self, encoder_cache, device)
        from .model_state import Ming3ModelState

        return Ming3ModelState(vllm_config, self, encoder_cache, device)

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info: Any,
    ):
        if self.model_stage != "llm":
            return input_ids, input_embeds, {}
        runner_id, state_id = info.get("request_id"), _request_id(info)
        if isinstance(runner_id, str) and isinstance(state_id, str) and runner_id != state_id:
            # Keep sampling IDs unchanged: prompt and flow RNGs may differ.
            self._request_state_ids[runner_id] = state_id
        input_embeds = self.model.embed_input_ids(input_ids).clone()
        return (
            self._prefill_preprocess(input_ids, input_embeds, info)
            if bool(info.get("_omni_is_prefill", input_ids.shape[0] > 1))
            else self._decode_preprocess(input_ids, input_embeds, info)
        )

    def _prefill_preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        info: dict[str, Any],
    ):
        """Inject cached full-prompt projections into the scheduled token span."""
        request_id = _request_id(info)
        offset = int(info.get("_omni_num_computed_tokens", 0))
        conditioning = self._prefill_conditioning.get(request_id)
        update: dict[str, Any] = {}
        if conditioning is None:
            conditioning = self._prepare_prefill_conditioning(input_ids, input_embeds, info)
            if isinstance(request_id, str):
                self._prefill_conditioning[request_id] = conditioning
                self.model.set_initial_history(request_id, conditioning.history)
            else:
                update[state_keys.KEY_LATENT_HISTORY] = conditioning.history.detach().cpu().contiguous()
            update[state_keys.KEY_DECODE_STEP] = 0

        end = offset + input_ids.shape[0]
        for index, position in enumerate(conditioning.speaker_positions):
            if offset <= position < end:
                input_embeds[position - offset] = conditioning.speaker_embeds[index]
        if conditioning.prompt_embeds is not None:
            start = max(offset, conditioning.prompt_start)
            stop = min(end, conditioning.prompt_start + conditioning.prompt_embeds.shape[0])
            if start < stop:
                input_embeds[start - offset : stop - offset] = conditioning.prompt_embeds[
                    start - conditioning.prompt_start : stop - conditioning.prompt_start
                ]
        self._copy_request_state(update, info)
        return input_ids, input_embeds, update

    def _prepare_prefill_conditioning(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        info: dict[str, Any],
    ) -> _PrefillConditioning:
        """Encode reference conditioning once, using positions from the full prompt.

        Direct callers with a complete prompt may omit the position metadata.
        Scheduled partial prompts require the metadata emitted by the builder.
        """
        offset = int(info.get("_omni_num_computed_tokens", 0))
        prompt_length = int(info.get("_omni_prompt_len", input_ids.shape[0]))
        if KEY_SPEAKER_POSITIONS not in info and (offset or input_ids.shape[0] != prompt_length):
            raise ValueError("Chunked Ming3 prefill requires conditioning positions from build_ming3_prompt")
        slots = tuple(
            info[KEY_SPEAKER_POSITIONS]
            if KEY_SPEAKER_POSITIONS in info
            else _find_speaker_placeholder_positions(input_ids, self.ming_config)
        )
        if any(position < 0 or position >= prompt_length for position in slots) or len(set(slots)) != len(slots):
            raise ValueError("Invalid Ming3 speaker positions")
        mute_speaker = bool(_take_scalar(info.get("mute_speaker"), 0))
        no_speaker_slot = bool(_take_scalar(info.get("no_speaker_slot"), 0))
        if mute_speaker and no_speaker_slot:
            raise ValueError("Speaker slot cannot be both absent and muted")
        if no_speaker_slot and any(
            info.get(key) is not None for key in (KEY_SPEAKER_EMBEDDING, KEY_SPEAKER_WAVEFORM, "prompt_waveform")
        ):
            raise ValueError("No-speaker-slot conditioning cannot include speaker or prompt audio")
        prompt_generator = None
        if info.get("prompt_waveform") is not None:
            prompt_generator = self.model._get_request_generator(info, input_embeds.device)
        request_id = _request_id(info)
        self._prompt_request_id = request_id
        self._prompt_cache_key = self._prompt_encoder_cache_key(info)
        try:
            prompt_latents = (
                None
                if mute_speaker or no_speaker_slot
                else _resolve_prompt_latents(self, info, generator=prompt_generator)
            )
        finally:
            self._prompt_request_id = None
            self._prompt_cache_key = None
        history = _initial_history(
            prompt_latents["frames"] if prompt_latents is not None else None,
            history_size=self.ming_config.history_patch_size,
            latent_dim=self.ming_config.latent_dim,
            device=input_embeds.device,
            dtype=torch.float32,
        )
        speaker_embeds = input_embeds.new_empty((0, input_embeds.shape[-1]))
        if no_speaker_slot:
            if slots:
                raise RuntimeError("No-speaker-slot conditioning must not contain speaker slots")
        elif mute_speaker:
            if len(slots) != 1:
                raise RuntimeError("Masked speaker conditioning requires exactly one speaker slot")
            speaker_embeds = input_embeds.new_zeros((1, input_embeds.shape[-1]))
        else:
            speakers = self._prefetched_speakers.get(request_id) or resolve_speaker_embeddings(self, info)
            if not slots or len(slots) != len(speakers):
                raise RuntimeError(f"Ming3 speaker slot/embedding count mismatch: {len(slots)}/{len(speakers)}")
            speaker_embeds = self.model.project_speaker_embedding(
                to_device_nonblocking(torch.stack(speakers), input_embeds.device).to(input_embeds.dtype)
            ).detach()

        prompt_embeds = None
        prompt_start = 0
        if prompt_latents is not None:
            patches = prompt_latents["patches"].to(device=input_embeds.device, dtype=self.model.fm_dtype)
            # Retain an owned result: graph replay reuses its output buffer.
            prompt_embeds = self._project_prompt_patches(patches).squeeze(1).to(input_embeds.dtype).clone()
            if KEY_PROMPT_AUDIO_START in info:
                prompt_start = int(info[KEY_PROMPT_AUDIO_START])
            else:
                positions = _find_audio_placeholder_positions(input_ids, self.ming_config)
                if positions.numel() != prompt_embeds.shape[0]:
                    raise RuntimeError("Ming3 prompt latent/token count mismatch")
                prompt_start = int(positions[0])
            if prompt_start < 0 or prompt_start + prompt_embeds.shape[0] != prompt_length:
                raise RuntimeError("Ming3 prompt latent/token count mismatch")

        return _PrefillConditioning(slots, speaker_embeds, prompt_start, prompt_embeds, history)

    def _decode_preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        info: dict[str, Any],
    ):
        update: dict[str, Any] = {state_keys.KEY_DECODE_STEP: int(info.get(state_keys.KEY_DECODE_STEP, 0))}
        next_embeds = self._next_embeds(info)
        if next_embeds is not None:
            input_embeds[0] = next_embeds.reshape(-1, self.ming_config.llm_hidden_size)[0].to(
                input_embeds.device, input_embeds.dtype
            )
        self._copy_request_state(update, info)
        return input_ids, input_embeds, update

    def preprocess_decode_batch(
        self,
        input_ids: torch.Tensor,
        req_infos: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        """Build all one-token decode rows at once from device-resident feedback."""
        hidden_size = self.ming_config.llm_hidden_size
        rows = [self._next_embeds(info) for info in req_infos]
        if all(row is not None for row in rows):
            input_embeds = torch.cat([row.reshape(1, hidden_size) for row in rows]).to(dtype=self.model.fm_dtype)
        else:
            input_embeds = self.model.embed_input_ids(input_ids.reshape(-1)).clone()
            for index, row in enumerate(rows):
                if row is not None:
                    input_embeds[index] = row.reshape(hidden_size).to(input_embeds.dtype)
        updates = []
        for info in req_infos:
            update: dict[str, Any] = {state_keys.KEY_DECODE_STEP: int(info.get(state_keys.KEY_DECODE_STEP, 0))}
            self._copy_request_state(update, info)
            updates.append(update)
        return input_ids, input_embeds, updates

    def _next_embeds(self, info: dict[str, Any]) -> torch.Tensor | None:
        request_id = _request_id(info)
        next_embeds = self.model.next_embeds_for(request_id) if isinstance(request_id, str) else None
        if next_embeds is None:
            next_embeds = info.get(state_keys.KEY_NEXT_EMBEDS)
        return next_embeds if isinstance(next_embeds, torch.Tensor) else None

    def _copy_request_state(self, update: dict[str, Any], info: dict[str, Any]) -> None:
        request_id = info.get(state_keys.KEY_REQUEST_ID, info.get(KEY_REQUEST_ID, info.get("request_id")))
        if request_id is not None:
            update[state_keys.KEY_REQUEST_ID] = request_id
        if "tts_local_seed" in info:
            update["tts_local_seed"] = info["tts_local_seed"]
        mappings = (
            (KEY_CFG, state_keys.KEY_CFG),
            (KEY_SIGMA, state_keys.KEY_SIGMA),
            (KEY_TEMPERATURE, state_keys.KEY_TEMPERATURE),
            (KEY_MAX_DECODE_STEPS, state_keys.KEY_MAX_DECODE_STEPS),
            (KEY_MIN_DECODE_STEPS, state_keys.KEY_MIN_DECODE_STEPS),
        )
        for public_key, state_key in mappings:
            if state_key in info:
                update[state_key] = info[state_key]
            elif public_key in info:
                update[state_key] = info[public_key]

    def postprocess(self, hidden_states: torch.Tensor, **info: Any) -> dict[str, Any]:
        if self.model_stage != "llm" or hidden_states.numel() == 0:
            return {}
        request_id = info.get(state_keys.KEY_REQUEST_ID, info.get("request_id"))
        pending = self.model.pop_postprocess_update(request_id)
        if not pending:
            return {}
        # The latent history and next embedding stay on the device in the
        # Stage-0 model; only the scalar decode state goes through the buffer.
        update: dict[str, Any] = {state_keys.KEY_DECODE_STEP: int(info.get(state_keys.KEY_DECODE_STEP, 0)) + 1}
        if "tts_local_seed" in info:
            update["tts_local_seed"] = info["tts_local_seed"]
        stop_reason = pending.get(KEY_STOP_REASON)
        if isinstance(stop_reason, str):
            update[KEY_STOP_REASON] = stop_reason
        if request_id is not None:
            update[state_keys.KEY_REQUEST_ID] = request_id
        return update


def _request_id(info: dict[str, Any]) -> Any:
    return info.get(state_keys.KEY_REQUEST_ID, info.get(KEY_REQUEST_ID, info.get("request_id")))


__all__ = ["Ming3TTSForConditionalGeneration"]
