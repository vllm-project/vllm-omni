# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections.abc import Iterable, Mapping
from dataclasses import replace
from functools import cached_property
from pathlib import Path

import regex as re
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.voxtral_tts.cuda_graph_acoustic_transformer_wrapper import (
    CUDAGraphAcousticTransformerWrapper,
)
from vllm_omni.model_executor.models.voxtral_tts.voxtral_tts_audio_generation import (
    VoxtralTTSDummyInputsBuilder,
    VoxtralTTSMultiModalProcessor,
    VoxtralTTSProcessingInfo,
)
from vllm_omni.transformers_utils.repo_utils import hf_api

logger = init_logger(__name__)


def parse_batched_audio_input(input_ids: torch.Tensor, num_codebooks: int) -> tuple[list[torch.Tensor], list[int]]:
    """Parse batched input_ids with [ctx_frames, context_length, ...tokens] format.

    Each request in the batch is laid out as:
        [ctx_frames, context_length, <flat audio tokens>]
    where the flat audio tokens have length (ctx_frames + context_length) * num_codebooks.

    Returns:
        all_audio_tokens: list of (num_frames, num_codebooks) tensors per request.
        all_ctx_frames: list of ctx_frames values per request.
    """
    all_audio_tokens: list[torch.Tensor] = []
    all_ctx_frames: list[int] = []
    # One D2H copy up-front avoids 2 syncs per request inside the loop.
    header_view = input_ids.cpu() if input_ids.is_cuda else input_ids
    offset = 0
    while offset < input_ids.numel():
        ctx_frames = int(header_view[offset])
        context_length = int(header_view[offset + 1])
        offset += 2

        req_input_ids_length = (ctx_frames + context_length) * num_codebooks
        req_input_ids = input_ids[offset : offset + req_input_ids_length]
        offset += req_input_ids_length
        assert req_input_ids.numel() % num_codebooks == 0, (
            f"Number of elements must be divisible by {num_codebooks} but get {req_input_ids.numel()=}"
        )
        num_of_audios = req_input_ids.numel() // num_codebooks
        audio_tokens = req_input_ids.view(num_of_audios, num_codebooks)
        all_audio_tokens.append(audio_tokens)
        all_ctx_frames.append(ctx_frames)
    return all_audio_tokens, all_ctx_frames


def apply_ctx_frames_cutting(
    batch_audio_arrays: list[torch.Tensor],
    all_ctx_frames: list[int],
    downsample_factor: int,
) -> list[torch.Tensor]:
    """Cut leading context samples from decoded audio arrays.

    Args:
        batch_audio_arrays: raw decoded audio arrays, one per request.
        all_ctx_frames: number of context frames per request.
        downsample_factor: samples per frame in the audio tokenizer.

    Returns:
        List of audio arrays with context samples removed.
    """
    result: list[torch.Tensor] = []
    for audio_array, ctx_frames in zip(batch_audio_arrays, all_ctx_frames):
        if ctx_frames > 0:
            new_sample_cut = downsample_factor * ctx_frames
            audio_array = audio_array[new_sample_cut:]
        result.append(audio_array)
    return result


@MULTIMODAL_REGISTRY.register_processor(
    VoxtralTTSMultiModalProcessor,
    info=VoxtralTTSProcessingInfo,
    dummy_inputs=VoxtralTTSDummyInputsBuilder,
)
class VoxtralTTSForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    CustomProcessMixin,
):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.config = config
        self.repo_id = vllm_config.model_config.model
        self.is_hf_model = not Path(self.repo_id).is_dir()
        self.model_stage = vllm_config.model_config.model_stage
        if self.model_stage == "audio_generation":
            self.has_preprocess = True
            self.requires_raw_input_tokens = True
            self.set_custom_preprocess(self.tts_preprocess)
            self.audio_generation = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config,
                prefix=maybe_prefix(prefix, "audio_generation"),
                architectures=["VoxtralTTSAudioGeneration"],
            )
            self.model = self.audio_generation
            self.audio_tokenizer = None
            self._cudagraph_acoustic_transformer = None
            self._vllm_config = vllm_config
            tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
            self._audio_token_id = tokenizer.instruct.audio_encoder.special_ids.audio
            speaker_id = config.audio_config.get("speaker_id", None)
            if speaker_id:
                self.voice_to_embedding = {}
                for sid in speaker_id:
                    if self.is_hf_model:
                        path = hf_api().hf_hub_download(repo_id=self.repo_id, filename=f"voice_embedding/{sid}.pt")
                    else:
                        path = Path(self.repo_id) / "voice_embedding" / f"{sid}.pt"
                    if Path(path).exists():
                        self.voice_to_embedding[sid] = torch.load(path, map_location="cpu")
                    else:
                        logger.warning("Voice embedding not found: %s", path)
                logger.info("Available voice embeddings: %s", list(self.voice_to_embedding.keys()))
            else:
                self.voice_to_embedding = {}
                logger.warning("No speaker_id configured in audio_config. No voice embeddings will be available.")
        elif self.model_stage == "audio_tokenizer":
            self.requires_raw_input_tokens = True
            self.audio_generation = None
            if vllm_config.quant_config is not None:
                vllm_config = replace(vllm_config, quant_config=None)
                logger.info("Voxtral TTS FP8 routing: stage-1 audio_tokenizer=bf16")
            self.audio_tokenizer = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config,
                prefix=maybe_prefix(prefix, "audio_tokenizer"),
                architectures=["VoxtralTTSAudioTokenizer"],
            )
            self.model = self.audio_tokenizer
            # forward() returns a runtime-length per-request list; FULL CUDAGraphWrapper
            # would freeze that length at capture.
            self.supports_cudagraph_full = False
        else:
            raise ValueError("Invalid model stage")

    def create_omni_model_state(self, vllm_config, encoder_cache, device):
        from .model_state import VoxtralTTSModelState

        return VoxtralTTSModelState(vllm_config, self, encoder_cache, device)

    # -------------------- CUDA Graph for acoustic transformer --------------------

    def get_language_model(self) -> "nn.Module":
        """Return the language model for upstream MoE detection."""
        if hasattr(self.model, "get_language_model"):
            return self.model.get_language_model()
        return self.model

    def encoder_loaded(self):
        if self.audio_tokenizer is not None:
            return self.audio_tokenizer.encoder_loaded
        return False

    def _enable_acoustic_transformer_cudagraph(self):
        """Initialize and capture CUDA graphs for compute_mm_logits."""
        if self.model_stage != "audio_generation" or not hasattr(self, "_vllm_config"):
            return

        model_cfg = getattr(self._vllm_config, "model_config", None)
        if model_cfg and getattr(model_cfg, "enforce_eager", False):
            logger.info("CUDA Graph for acoustic transformer not enabled: --enforce-eager is set")
            return

        try:
            acoustic_transformer = self.model.acoustic_transformer
            device = next(acoustic_transformer.parameters()).device
            dtype = next(acoustic_transformer.parameters()).dtype
            hidden_dim = self.model.config.text_config.hidden_size

            wrapper = CUDAGraphAcousticTransformerWrapper(self.model)
            wrapper._warmup_and_capture(device, dtype, hidden_dim)
            self._cudagraph_acoustic_transformer = wrapper
            logger.info("CUDA Graph for acoustic transformer enabled")
        except Exception:
            logger.warning("Failed to enable CUDA Graph for acoustic transformer", exc_info=True)
            self._cudagraph_acoustic_transformer = None

    # -------------------- Device utilities --------------------
    @cached_property
    def sampler(self):
        if hasattr(self.model, "sampler"):
            return self.model.sampler
        return Sampler()

    @staticmethod
    def _is_prefill_step(info_dict: Mapping[str, object], input_ids: torch.Tensor) -> bool:
        is_prefill_raw = info_dict.get("_omni_is_prefill")
        if isinstance(is_prefill_raw, bool):
            return is_prefill_raw
        try:
            return int(info_dict["_omni_num_computed_tokens"]) < int(info_dict["_omni_prompt_len"])
        except (KeyError, TypeError, ValueError):
            return input_ids.shape[0] > 1

    @staticmethod
    def _select_feedback_frames(history: object, frame_indices: list[int], device: torch.device) -> torch.Tensor:
        if isinstance(history, torch.Tensor):
            frames = history
            if frames.ndim == 1:
                frames = frames.unsqueeze(0)
            elif frames.ndim > 2:
                frames = frames.reshape(frames.shape[0], -1)
            if frame_indices and max(frame_indices) >= frames.shape[0]:
                raise RuntimeError(
                    f"Voxtral acoustic history has {frames.shape[0]} frames; "
                    f"replay requested frame {max(frame_indices)}"
                )
            indices = torch.tensor(frame_indices, dtype=torch.long, device=frames.device)
            return frames.index_select(0, indices).to(device=device)

        if isinstance(history, (list, tuple)):
            if frame_indices and max(frame_indices) >= len(history):
                raise RuntimeError(
                    f"Voxtral acoustic history has {len(history)} frames; replay requested frame {max(frame_indices)}"
                )
            selected = [history[index].reshape(-1) for index in frame_indices]
            if not selected:
                return torch.empty((0, 0), device=device)
            output = torch.empty(
                (len(selected), selected[0].numel()),
                dtype=selected[0].dtype,
                device=device,
            )
            groups: dict[torch.device, list[tuple[int, torch.Tensor]]] = {}
            for row_index, frame in enumerate(selected):
                groups.setdefault(frame.device, []).append((row_index, frame))
            for source_device, group in groups.items():
                row_indices = [row for row, _ in group]
                batch = torch.stack([frame for _, frame in group])
                if source_device != device:
                    batch = batch.to(device=device)
                output.index_copy_(
                    0,
                    torch.tensor(row_indices, dtype=torch.long, device=device),
                    batch,
                )
            return output

        raise TypeError(f"Unsupported Voxtral acoustic history: {type(history).__name__}")

    def tts_preprocess(self, input_ids: torch.Tensor, input_embeds: torch.Tensor, **info_dict: dict | None):
        codes = info_dict.get("codes")
        audio_tokens = codes.get("audio") if isinstance(codes, Mapping) else None
        audio_history = codes.get("audio_history") if isinstance(codes, Mapping) else None
        # Seed persistent runner state before make_omni_output mutates the request
        # info in place. V1 can pass a temporary empty dict when no state exists.
        feedback_state_update = {"codes": {"audio_history": []}} if audio_history is None else {}
        prompt_len = info_dict.get("_omni_prompt_len")
        num_computed_tokens = info_dict.get("_omni_num_computed_tokens")
        replay_mask = None
        frame_embeddings = None
        has_replay_positions = prompt_len is not None and num_computed_tokens is not None
        history_is_empty = (
            audio_history is None
            or (isinstance(audio_history, (list, tuple)) and len(audio_history) == 0)
            or (isinstance(audio_history, torch.Tensor) and audio_history.numel() == 0)
        )

        # Normal autoregressive decode schedules one sampled token at the
        # absolute position matching the newest acoustic frame. Voxtral's
        # logits admit only an audio token or EOS; EOS requests are not
        # scheduled again, so this one-token continuation is known to be an
        # audio token. Use host metadata to recognize it before touching
        # input_ids, avoiding CUDA bool()/tolist() synchronization per step.
        # Async scheduling can leave the latest audio payload out of alignment
        # with this token; only take the shortcut when history confirms the
        # position, otherwise let the absolute-position replay path handle it.
        if (
            isinstance(audio_tokens, torch.Tensor)
            and audio_history is not None
            and prompt_len is not None
            and num_computed_tokens is not None
            and input_ids.numel() == 1
        ):
            prompt_len = int(prompt_len)
            num_computed_tokens = int(num_computed_tokens)
            history_length = (
                len(audio_history)
                if isinstance(audio_history, (list, tuple))
                else int(audio_history.shape[0])
                if isinstance(audio_history, torch.Tensor) and audio_history.ndim > 0
                else 0
            )
            if (
                history_length > 0
                and num_computed_tokens >= prompt_len
                and num_computed_tokens - prompt_len == history_length - 1
            ):
                multimodal_embeddings = self.model.embed_multimodal(audio_tokens=audio_tokens.to(input_ids.device))
                return input_ids, multimodal_embeddings[0], feedback_state_update

        # A preempted request is replayed from its prompt. Generated Voxtral
        # audio tokens need the acoustic frame emitted for that token, by
        # absolute token position, rather than the one frame that was newest
        # when the request lost its runner slot.
        if audio_history is not None and prompt_len is not None and num_computed_tokens is not None:
            prompt_len = int(prompt_len)
            start = int(num_computed_tokens)
            positions = torch.arange(start, start + input_ids.numel(), device=input_ids.device)
            replay_mask = (positions >= prompt_len) & (input_ids == self._audio_token_id)
            if bool(replay_mask.any()):
                frame_indices = (positions[replay_mask] - prompt_len).tolist()
                history_length = (
                    len(audio_history)
                    if isinstance(audio_history, (list, tuple))
                    else int(audio_history.shape[0])
                    if isinstance(audio_history, torch.Tensor) and audio_history.ndim > 0
                    else 0
                )
                if (
                    len(frame_indices) == 1
                    and frame_indices[0] == history_length - 1
                    and isinstance(audio_tokens, torch.Tensor)
                ):
                    # Live decode normally consumes only the newest frame. Keep it on
                    # device; older frames are needed only when replaying preempted work.
                    selected_frames = audio_tokens.to(input_ids.device).reshape(1, -1)
                else:
                    selected_frames = self._select_feedback_frames(audio_history, frame_indices, input_ids.device)
                multimodal_embeddings = self.model.embed_multimodal(audio_tokens=selected_frames.unsqueeze(0))
                frame_embeddings = multimodal_embeddings[0]
                replay_count = len(frame_indices)
                if frame_embeddings.shape[0] != replay_count:
                    raise RuntimeError(
                        "Voxtral acoustic embedding count does not match replayed audio tokens: "
                        f"{frame_embeddings.shape[0]} embeddings for {replay_count} tokens"
                    )
                input_embeds = input_embeds.clone()
                input_embeds[replay_mask] = frame_embeddings

        # Preserve the existing path for source audio and for decode requests
        # without saved replay history.
        if audio_tokens is None and history_is_empty:
            # Backward compatible with request state written before #4527 moved
            # Voxtral TTS feedback frames under ``codes.audio``.
            audio_tokens = info_dict.get("audio")
        if audio_tokens is not None and (audio_history is None or not has_replay_positions):
            kwargs = {"audio_tokens": audio_tokens.to(input_ids.device)}
            multimodal_embeddings = self.model.embed_multimodal(**kwargs)
            if input_ids[0] == self._audio_token_id:
                input_embeds = multimodal_embeddings[0]
            return input_ids, input_embeds, feedback_state_update

        voice = info_dict.get("voice")
        if voice is not None and self._is_prefill_step(info_dict, input_ids):
            voice = info_dict.get("voice")
            if isinstance(voice, list):
                voice = voice[0]
            multimodal_embeddings = self.voice_to_embedding[voice].to(input_ids.device).clone().detach()
            is_multimodal = input_ids == self._audio_token_id
            if replay_mask is not None and prompt_len is not None and num_computed_tokens is not None:
                positions = torch.arange(
                    int(num_computed_tokens),
                    int(num_computed_tokens) + input_ids.numel(),
                    device=input_ids.device,
                )
                is_multimodal &= positions < int(prompt_len)
            input_embeds = self.embed_input_ids(
                input_ids=input_ids,
                multimodal_embeddings=multimodal_embeddings,
                is_multimodal=is_multimodal,
            )
            # The voice embedding covers original prompt audio markers; replay
            # embeddings cover generated markers in the same scheduled chunk.
            if replay_mask is not None and bool(replay_mask.any()):
                input_embeds = input_embeds.clone()
                input_embeds[replay_mask] = frame_embeddings
            return input_ids, input_embeds, feedback_state_update
        return input_ids, input_embeds, feedback_state_update

    @staticmethod
    def _filter_ineligible_feedback_outputs(
        multimodal_outputs: object,
        request_sample_eligible: list[bool] | tuple[bool, ...] | torch.Tensor | None,
    ) -> None:
        if request_sample_eligible is None:
            return
        if isinstance(request_sample_eligible, torch.Tensor):
            request_sample_eligible = request_sample_eligible.tolist()
        if not isinstance(request_sample_eligible, (list, tuple)):
            raise TypeError("Voxtral sample eligibility must be a per-request sequence")
        codes = multimodal_outputs.get("codes") if isinstance(multimodal_outputs, dict) else None
        if not isinstance(codes, dict):
            return
        audio = codes.get("audio")
        if isinstance(audio, torch.Tensor):
            frames = [audio] if audio.ndim == 1 else list(audio.unbind(0))
        elif isinstance(audio, (list, tuple)):
            frames = list(audio)
        else:
            return
        if len(frames) != len(request_sample_eligible):
            raise RuntimeError(
                "Voxtral acoustic output count does not match sample eligibility: "
                f"frames={len(frames)} eligible={len(request_sample_eligible)}"
            )
        if any(not bool(eligible) for eligible in request_sample_eligible):
            codes["audio"] = [
                frame if bool(eligible) else None
                for frame, eligible in zip(frames, request_sample_eligible, strict=True)
            ]

    @staticmethod
    def _store_feedback_frames(
        multimodal_outputs: object,
        request_infos: object,
        request_sample_eligible: list[bool] | tuple[bool, ...] | torch.Tensor | None = None,
        *,
        keep_history_on_device: bool = False,
    ) -> None:
        """Save each emitted acoustic frame in its request-local model state.

        The next decode step consumes this frame in tts_preprocess. Keeping it
        in request-local state avoids a model-global cursor, which can associate
        frames with the wrong request under batching. MRV2 may keep active
        history on device and offload it when preempted; V1 keeps CPU history.
        MRV2 device memory grows with generated frames across active requests.
        request_sample_eligible excludes intermediate prompt chunks; the final
        prompt chunk remains eligible for the first generated token.
        """
        if not isinstance(multimodal_outputs, Mapping) or not isinstance(request_infos, (list, tuple)):
            return
        codes = multimodal_outputs.get("codes")
        audio = codes.get("audio") if isinstance(codes, Mapping) else None
        if isinstance(audio, torch.Tensor):
            frames = [audio] if audio.ndim == 1 else list(audio.unbind(0))
        elif isinstance(audio, (list, tuple)):
            frames = list(audio)
        else:
            return
        if len(frames) != len(request_infos):
            logger.warning(
                "Skipping Voxtral feedback update: got %d audio frames for %d request states",
                len(frames),
                len(request_infos),
            )
            return
        if request_sample_eligible is not None:
            if isinstance(request_sample_eligible, torch.Tensor):
                request_sample_eligible = request_sample_eligible.tolist()
            if not isinstance(request_sample_eligible, (list, tuple)) or len(request_sample_eligible) != len(
                request_infos
            ):
                logger.warning("Skipping Voxtral feedback update: sample eligibility count does not match requests")
                return
            eligible_pairs = [
                (request_info, frame)
                for request_info, frame, eligible in zip(request_infos, frames, request_sample_eligible, strict=True)
                if bool(eligible)
            ]
            request_infos = [request_info for request_info, _ in eligible_pairs]
            frames = [frame for _, frame in eligible_pairs]
        for request_info, frame in zip(request_infos, frames, strict=True):
            if not isinstance(request_info, dict) or not isinstance(frame, torch.Tensor):
                continue
            # compute_mm_logits emits (1, 1, codebooks) frames. Normalize
            # tensor outputs to the same shape expected by multimodal embeds.
            if frame.ndim == 1:
                frame = frame.reshape(1, 1, -1)
            elif frame.ndim == 2:
                frame = frame.unsqueeze(0)
            request_codes = request_info.setdefault("codes", {})
            if isinstance(request_codes, dict):
                # Own the frame so request state survives reuse of model-output buffers.
                latest_frame = frame.detach().clone()
                # MRV2 owns request state directly, so keep active replay
                # frames with the request on device and offload them together
                # only if scheduler preemption suspends that request. V1's
                # runner serializes intermediate updates to CPU and keeps the
                # historical behavior below.
                history_frame = (
                    latest_frame if keep_history_on_device else frame.detach().to(device="cpu", copy=True).contiguous()
                )
                request_codes["audio"] = latest_frame
                history = request_codes.setdefault("audio_history", [])
                if isinstance(history, list):
                    history.append(history_frame)
                elif isinstance(history, torch.Tensor):
                    # Resumed requests normally restore a list; tolerate an
                    # older serialized runner payload that restored a tensor.
                    stored_history = history.detach().contiguous()
                    if not keep_history_on_device:
                        stored_history = stored_history.to(device="cpu", copy=True).contiguous()
                    request_codes["audio_history"] = [*stored_history.unbind(0), history_frame]
                else:
                    request_codes["audio_history"] = [history_frame]

    def _cfg_alpha_from_request_infos(
        self,
        request_infos: list[dict],
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if not request_infos:
            return self._extract_cfg_alpha(hidden_states)
        values = []
        for info in request_infos:
            sampling_params = info.get("sampling_params")
            extra_args = getattr(sampling_params, "extra_args", None) or {}
            values.append(extra_args.get("cfg_alpha", self._DEFAULT_CFG_ALPHA))
        return torch.tensor(values, device=hidden_states.device, dtype=hidden_states.dtype)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        if self.model_stage == "audio_tokenizer":
            return torch.zeros_like(input_ids).reshape(-1, 1).repeat(1, self.config.text_config.hidden_size)
        return self.model.embed_input_ids(
            input_ids=input_ids, multimodal_embeddings=multimodal_embeddings, is_multimodal=is_multimodal
        )

    def embed_multimodal(self, **kwargs):
        # Delegate to generation model for multimodal processing
        return self.model.embed_multimodal(**kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors | OmniOutput:
        if self.model_stage == "audio_generation":
            if inputs_embeds is not None:
                input_ids = None
            hidden_states = self.model(
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds=inputs_embeds,
            )
            return hidden_states

        if self.model_stage == "audio_tokenizer":
            if (input_ids == 0).all():
                logger.info("audio_tokenizer: sample run with dummy input")
                # TODO(chenyo): Move this to dummy_inputs creation
                num_codebooks = self.audio_tokenizer.num_codebooks
                audio_tokens = torch.randint(low=2, high=100, size=(116, num_codebooks), dtype=torch.int32)
                audio_tokens[-1, :] = 0
                audio_tokens[-1, 0] = 1
                batch_audio_arrays = self.audio_tokenizer.decode_helper_batch_async([audio_tokens])
                return OmniOutput(
                    text_hidden_states=None,
                    multimodal_outputs={"audio": batch_audio_arrays},
                )
            all_audio_tokens, all_ctx_frames = parse_batched_audio_input(
                input_ids, num_codebooks=self.audio_tokenizer.num_codebooks
            )

            # Batch decode all requests at once
            batch_audio_arrays_raw = self.audio_tokenizer.decode_helper_batch_async(all_audio_tokens)
            batch_audio_arrays = apply_ctx_frames_cutting(
                batch_audio_arrays_raw, all_ctx_frames, self.audio_tokenizer.downsample_factor
            )

            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"audio": batch_audio_arrays},
            )

    _DEFAULT_CFG_ALPHA = 1.2

    @property
    def requires_request_sample_eligibility(self) -> bool:
        return self.model_stage == "audio_generation"

    def _extract_cfg_alpha(self, input_hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Extract per-request cfg_alpha from sampling_extra_args.

        Returns a 1-D tensor of shape (B,) with per-request cfg_alpha values.
        Falls back to default if sampling_extra_args is missing or incomplete.
        """
        B = input_hidden_states.shape[0]
        sampling_extra_args = kwargs.get("sampling_extra_args")
        if sampling_extra_args is None:
            return torch.full(
                (B,),
                self._DEFAULT_CFG_ALPHA,
                device=input_hidden_states.device,
                dtype=input_hidden_states.dtype,
            )
        cfg_alpha_values = [ea.get("cfg_alpha", self._DEFAULT_CFG_ALPHA) for ea in sampling_extra_args]
        return torch.tensor(
            cfg_alpha_values,
            device=input_hidden_states.device,
            dtype=input_hidden_states.dtype,
        )

    def make_omni_output(
        self, model_outputs: torch.Tensor | OmniOutput | tuple, logits_index: int | None = None, **kwargs
    ) -> OmniOutput:
        if isinstance(model_outputs, torch.Tensor):
            if self.model_stage == "audio_generation":
                hidden_states = model_outputs
                assert logits_index is not None
                input_hidden_states = hidden_states[logits_index]
                cfg_alpha = self._extract_cfg_alpha(input_hidden_states, **kwargs)
                if self._cudagraph_acoustic_transformer is not None:
                    fake_eos, multimodal_outputs = self._cudagraph_acoustic_transformer(
                        input_hidden_states, cfg_alpha=cfg_alpha
                    )
                else:
                    fake_eos, multimodal_outputs = self.model.compute_mm_logits(
                        input_hidden_states, cfg_alpha=cfg_alpha
                    )
                hidden_states[logits_index, 0] = fake_eos
                request_infos = kwargs.get("runtime_additional_information")
                if request_infos is None:
                    request_infos = kwargs.get("model_intermediate_buffer")
                sample_eligible = kwargs.get("request_sample_eligible")
                self._filter_ineligible_feedback_outputs(multimodal_outputs, sample_eligible)
                self._store_feedback_frames(multimodal_outputs, request_infos, sample_eligible)
                return OmniOutput(
                    text_hidden_states=hidden_states,
                    multimodal_outputs=multimodal_outputs,
                )

            raise ValueError(f"Unsupported {self.model_stage} for model_outputs type: {type(model_outputs)}")
        if isinstance(model_outputs, OmniOutput):
            if self.model_stage == "audio_tokenizer":
                return model_outputs

            raise ValueError(f"Unsupported {self.model_stage} for model_outputs type: {type(model_outputs)}")

        raise ValueError(f"Unsupported model_outputs type: {type(model_outputs)}")

    def make_omni_output_mrv2(
        self,
        model_outputs: torch.Tensor | OmniOutput,
        *,
        input_batch,
        req_states,
        model_intermediate_buffer: list[dict],
    ) -> OmniOutput:
        """Build Voxtral's per-request acoustic output for Model Runner V2."""
        del req_states
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        if self.model_stage != "audio_generation":
            raise ValueError(f"Unsupported MRV2 model output for stage {self.model_stage!r}")

        num_reqs = int(input_batch.num_reqs)
        logits_indices = input_batch.logits_indices[:num_reqs].to(device=model_outputs.device, dtype=torch.long)
        input_hidden_states = model_outputs.index_select(0, logits_indices)
        cfg_alpha = self._cfg_alpha_from_request_infos(model_intermediate_buffer, input_hidden_states)
        if self._cudagraph_acoustic_transformer is not None:
            fake_eos, multimodal_outputs = self._cudagraph_acoustic_transformer(
                input_hidden_states, cfg_alpha=cfg_alpha
            )
        else:
            fake_eos, multimodal_outputs = self.model.compute_mm_logits(input_hidden_states, cfg_alpha=cfg_alpha)
        model_outputs[logits_indices, 0] = fake_eos.reshape(-1)
        num_computed_prefill = input_batch.num_computed_prefill_tokens_np[:num_reqs]
        num_scheduled_tokens = input_batch.num_scheduled_tokens[:num_reqs]
        prefill_lens = input_batch.prefill_len_np[:num_reqs]
        request_sample_eligible = [
            int(computed) + int(scheduled) >= int(prompt_len)
            for computed, scheduled, prompt_len in zip(
                num_computed_prefill, num_scheduled_tokens, prefill_lens, strict=True
            )
        ]
        self._filter_ineligible_feedback_outputs(multimodal_outputs, request_sample_eligible)
        self._store_feedback_frames(
            multimodal_outputs,
            model_intermediate_buffer,
            request_sample_eligible,
            keep_history_on_device=True,
        )
        return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs=multimodal_outputs)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput) -> torch.Tensor | None:
        # Handle OmniOutput type
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if self.model_stage == "audio_generation":
            text_logits = self.model.fake_logits_for_audio_tokens(fake_eos=hidden_states)
            return text_logits
        if self.model_stage == "audio_tokenizer":
            raise ValueError("Invalid model stage")

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        return self.model.sample(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_weights = set()
        remapping_rules = [
            (r"^audio_tokenizer\.(.*)$", r"\1"),  # noqa: E501
            (
                r"^mm_audio_embeddings\.audio_codebook_embeddings\.embeddings\.(weight|bias)",
                r"audio_token_embedding.embeddings.\1",
            ),  # noqa: E501
            (r"^mm_audio_embeddings\.tok_embeddings\.weight", r"tok_embeddings.weight"),  # noqa: E501
        ]

        def llm_weights_generator():
            nonlocal loaded_weights
            for name, w in weights:
                is_audio_tokenizer = name.startswith(
                    "mm_audio_embeddings.audio_codebook_embeddings"
                ) or name.startswith("audio_tokenizer.")

                if is_audio_tokenizer and self.audio_tokenizer is not None:
                    # Remap name only when loading audio_tokenizer (Stage-1).
                    # Yield the original name in Stage-0 so audio_generation.load_weights()
                    # can still filter by prefix using its own remapping rules.
                    remapped = name
                    for pattern, repl in remapping_rules:
                        if re.fullmatch(pattern, remapped):
                            remapped = re.sub(pattern, repl, remapped)
                    name = self.audio_tokenizer.load_weight((remapped, w))
                    loaded_weights.add(f"audio_tokenizer.{name}")
                    continue

                yield (name, w)

        audio_generation_weights = list(llm_weights_generator())
        if self.audio_generation is not None:
            for name in self.audio_generation.load_weights(audio_generation_weights):
                loaded_weights.add(f"audio_generation.{name}")

        # If encoder weights were not in the checkpoint, mark them as
        # "loaded" so the weight-validation does not fail.
        # encode_waveforms() will raise at runtime if called without encoder weights.
        if self.audio_tokenizer is not None and not self.audio_tokenizer._encoder_loaded:
            for name, _ in self.audio_tokenizer.named_parameters():
                if name.startswith(self.audio_tokenizer._encoder_weight_prefixes):
                    loaded_weights.add(f"audio_tokenizer.{name}")

        # Capture CUDA graphs for compute_mm_logits after weights are loaded
        self._enable_acoustic_transformer_cudagraph()

        return loaded_weights
