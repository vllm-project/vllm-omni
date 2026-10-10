# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import hashlib
import os
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from functools import partial
from math import gcd
from threading import Lock
from types import MethodType
from typing import Any

import numpy as np
import onnxruntime
import torch
import torch.nn as nn
from scipy.signal import resample_poly
from transformers import Qwen2Config
from transformers.feature_extraction_utils import BatchFeature
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.inputs import MultiModalDataDict
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseProcessingInfo,
    ProcessorInputs,
    PromptIndexTargets,
    PromptInsertion,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import random_sample
from vllm.v1.sample.sampler import Sampler

from vllm_omni.data_entry_keys import EmbeddingsStruct, OmniPayloadStruct, to_dict, to_struct
from vllm_omni.inputs.mm_processor import OmniDummyInputsBuilder, OmniMultiModalProcessor
from vllm_omni.model_executor.models.cosyvoice3.ras_sampler import MAX_FUSED_TOP_K, fused_ras_sample
from vllm_omni.model_executor.models.cosyvoice3.runtime import (
    cosyvoice3_batch_flow_debug,
    cosyvoice3_batch_flow_enabled,
    cosyvoice3_packed_inference_enabled,
    cosyvoice3_standard_sampling,
)
from vllm_omni.model_executor.models.cosyvoice3.tokenizer import get_qwen_tokenizer
from vllm_omni.model_executor.models.cosyvoice3.utils import (
    concat_text_with_prompt_ids,
    extract_speech_feat,
    extract_spk_embedding,
    extract_spk_embedding_trt,
    extract_text_token,
    mel_spectrogram,
    unpad_prompt_conditioning,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.platforms import current_omni_platform
from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config
from vllm_omni.transformers_utils.repo_utils import hf_api
from vllm_omni.utils.speaker_cache import get_speaker_cache

logger = init_logger(__name__)

# Process-wide cache of per-model mm-processor runtime components (tokenizer,
# feat_extractor, campplus session/engine). The mm processor is re-created per
# request (mm_processor_cache_gb: 0), so this avoids rebuilding them every time.
_RUNTIME_COMPONENTS_CACHE: dict[str, dict] = {}


def _normalize_request_conditioning(payload: dict) -> dict:
    """Unwrap singleton conditioning containers at the per-request boundary.

    A processor/prefix-cache passthrough can retain a one-item tensor list.
    The code2wav schema consumes one request, so that wrapper carries no
    batch dimension. A multi-item list here is an unsplit batch and must
    fail instead of silently choosing another request's voice.
    """
    embed = payload.get("embed")
    if not isinstance(embed, Mapping):
        return payload
    normalized = dict(embed)
    for key in ("speech_token", "speech_feat", "embedding", "speech_token_len"):
        value = normalized.get(key)
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"CosyVoice3 per-request {key} contains an unsplit batch of {len(value)} items")
            normalized[key] = value[0]
    return {**payload, "embed": normalized}


def _cosyvoice3_trt_enabled() -> bool:
    """COSYVOICE3_TRT env toggle (default on) for the optional TensorRT paths.

    Gates both the talker speaker-embedding (campplus) engine and the code2wav
    flow-decoder estimator engine. Env-var toggle (cf. mimo_audio's
    ``MIMO_AUDIO_TOKENIZER_CUDA_GRAPH``) because these run outside the stage
    worker, so deploy-yaml ``hf_overrides`` / per-stage ``env`` do not reach
    them — export ``COSYVOICE3_TRT=0`` in the launching shell to disable.
    """
    return os.environ.get("COSYVOICE3_TRT", "1") not in ("0", "false", "False", "")


def _campplus_onnx_providers() -> list[str]:
    """ONNX-Runtime providers for the campplus speaker-embedding session.

    Prefer ``MUSAExecutionProvider`` (from the onnxruntime-musa build) with a
    CPU fallback when it is available; otherwise CPU only. The MUSA kernels
    differ from the CPU reference by a few percent on the embedding, which
    conditions voice cloning -- verify voice similarity if that matters.
    """
    if "MUSAExecutionProvider" in onnxruntime.get_available_providers():
        return ["MUSAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _audio_conditioning(proc, audio, model_dir: str, config, mm_kwargs: Mapping[str, object]) -> dict:
    """Reference-audio conditioning (speech tokens, mel, speaker embedding).

    Reuses bounded, process-local artifacts keyed by voice name or by the
    waveform content, and returns independent tensors so downstream mutation
    cannot poison a cache hit.
    """
    device = "cpu"
    voice_name = mm_kwargs.get("voice_name")
    cache_key = None
    if voice_name and isinstance(voice_name, str):
        cache_key = proc._speaker_cache.make_cache_key(
            voice_name,
            model_type="cosyvoice3",
            created_at=int(mm_kwargs.get("voice_created_at") or 0),
        )
    else:
        # Cache audio artifacts only: reference/target text is tokenized
        # separately. Keep dtype and sample rate in the identity to avoid
        # aliases between numerically distinct preprocessing inputs.
        waveform, sample_rate = audio
        waveform = np.ascontiguousarray(waveform)
        digest = hashlib.sha256()
        digest.update(str((int(sample_rate), waveform.shape, waveform.dtype.str)).encode())
        digest.update(memoryview(waveform).cast("B"))
        backend = "trt" if proc.campplus_trt is not None else "onnx"
        cache_key = proc._speaker_cache.make_cache_key(
            digest.hexdigest(), model_type=f"cosyvoice3-reference:{model_dir}:{backend}"
        )
    cached = proc._speaker_cache.get(cache_key)
    if os.environ.get("COSYVOICE3_REFERENCE_CACHE_DEBUG") == "1":
        _REFERENCE_CACHE_DEBUG[cached is not None] += 1
        if sum(_REFERENCE_CACHE_DEBUG.values()) % 128 == 0:
            logger.info(
                "CosyVoice3 reference cache lookups: hits=%d misses=%d model_dir=%s",
                _REFERENCE_CACHE_DEBUG[True],
                _REFERENCE_CACHE_DEBUG[False],
                model_dir,
            )
    if cached is None:
        # Speech-token extraction via the S3Tokenizer PyTorch model on GPU
        # (~30x faster than the bundled ``speech_tokenizer_v3.onnx`` CPU path).
        speech_token, speech_token_len = proc._extract_speech_token_via_s3(audio, device)
        speech_feat, speech_feat_len = extract_speech_feat(audio, proc.feat_extractor, device)
        if config.sample_rate == 24000:
            token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
            speech_feat, speech_feat_len[:] = speech_feat[:, : 2 * token_len], 2 * token_len
            speech_token, speech_token_len[:] = speech_token[:, :token_len], token_len
        if proc.campplus_trt is not None:
            embedding = extract_spk_embedding_trt(audio, proc.campplus_trt, device)
        else:
            embedding = extract_spk_embedding(audio, proc.campplus_session, device)
        cached = {
            "speech_feat": speech_feat.detach().cpu().clone(),
            "speech_token": speech_token.detach().cpu().clone(),
            "speech_token_len": speech_token_len.detach().cpu().clone(),
            "embedding": embedding.detach().cpu().clone(),
        }
        proc._speaker_cache.put(cache_key, cached)
    return {key: value.clone() for key, value in cached.items()}


_PROMPT_SCRATCH_TOKENS = 1 << 20
_MRV2_CONDITIONING_ENTRIES = 4096


def _generated_only_penalty_writes(state) -> None:
    """``PenaltiesState.apply_staged_writes`` that leaves the prompt unpenalized."""
    if state._new_penalties_reqs:
        from vllm.utils.torch_utils import async_tensor_h2d
        from vllm.v1.worker.gpu.sample.penalties import bincount

        scratch = getattr(state, "_prompt_scratch_mask", None)
        if scratch is None:
            columns = max(state.prompt_bin_mask.shape[1], _PROMPT_SCRATCH_TOKENS // 32)
            scratch = state._prompt_scratch_mask = torch.zeros(
                state.prompt_bin_mask.shape[0], columns, dtype=torch.int32, device=state.device
            )
        rows = async_tensor_h2d(state._new_penalties_reqs, dtype=torch.int32, device=state.device)
        prefill_lens = state.req_states.prefill_len.np[state._new_penalties_reqs]
        # Prompt bits land in the scratch mask; resumed output tokens (between
        # prompt_len and prefill_len) still seed the output counts.
        bincount(
            rows,
            state.req_states.all_token_ids.gpu,
            state.req_states.prompt_len.gpu,
            state.req_states.prefill_len.gpu,
            scratch,
            state.output_bin_counts,
            int(prefill_lens.max()),
        )
        state.prompt_bin_mask.index_fill_(0, rows.long(), 0)
        state._new_penalties_reqs.clear()
    state.repetition_penalty.copy_to_uva()
    state.frequency_penalty.copy_to_uva()
    state.presence_penalty.copy_to_uva()


def _ras_mrv2_sample(
    sampler,
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    idx_mapping: torch.Tensor,
    idx_mapping_np: np.ndarray,
    pos: torch.Tensor,
    input_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    return_logprobs: bool = False,
    *,
    default_top_p: float,
    default_top_k: int,
    win_size: int,
    tau_r: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``Sampler.sample`` for MRv2: CosyVoice3 repetition-aware sampling on the GPU.

    Same distribution as ``CosyVoice3Model._ras_sample_batch`` (V1): top-p over
    the full distribution capped at top-k; a draw repeating within the last
    ``win_size`` generated tokens is replaced by a draw from the full remaining
    distribution. History is read from the runner's token table and rejection
    is resolved per row on the device, so no step waits on the host. Penalties
    are not applied, as in V1 RAS. Draws use the request seed and position; the
    replacement draw uses a disjoint noise stream.
    """
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

    logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)
    sampler.logit_bias_state.apply_logit_bias(logits, expanded_idx_mapping, idx_mapping_np, pos)
    states = sampler.sampling_states
    temperature = states.temperature.gpu
    req_states = sampler.req_states
    top_k_np = states.top_k.np[idx_mapping_np]
    use_top_k = bool((top_k_np != states.vocab_size).any())
    if logits.is_cuda and not return_logprobs and (top_k_np.max() if use_top_k else default_top_k) <= MAX_FUSED_TOP_K:
        # One kernel per step instead of ~70 (see ras_sampler.py).
        use_top_p = bool((states.top_p.np[idx_mapping_np] != 1.0).any())
        sampled = fused_ras_sample(
            logits,
            expanded_idx_mapping,
            temperature,
            states.top_k.gpu if use_top_k else None,
            states.top_p.gpu if use_top_p else None,
            states.seeds.gpu,
            pos,
            req_states.all_token_ids.gpu,
            req_states.total_len.gpu,
            req_states.prompt_len.gpu,
            default_top_k=default_top_k,
            default_top_p=default_top_p,
            win_size=win_size,
            tau_r=tau_r,
            eps=eps,
        )
        return sampled, logits
    rows = expanded_idx_mapping.long()
    row_temperature = temperature[rows]
    scores = torch.log_softmax(logits / row_temperature.clamp_min(eps).unsqueeze(1), dim=1)
    num_rows, vocab = scores.shape
    top_k, top_p = states.get_top_k_top_p(expanded_idx_mapping, idx_mapping_np)
    if top_p is None:
        top_p = scores.new_full((num_rows,), default_top_p)
    if top_k is None:
        top_k = torch.full((num_rows,), default_top_k, dtype=torch.int32, device=scores.device)
    sorted_scores, sorted_ids = scores.sort(dim=1, descending=True, stable=True)
    sorted_probs = sorted_scores.exp()
    # Top-p is decided on the full distribution, before the top-k cap.
    keep = (sorted_probs.cumsum(dim=1) - sorted_probs) < top_p.unsqueeze(1)
    ranks = torch.arange(vocab, device=scores.device)
    keep &= (top_k.unsqueeze(1) <= 0) | (ranks.unsqueeze(0) < top_k.unsqueeze(1))
    nucleus = sorted_scores.masked_fill(~keep, float("-inf"))
    seeds = states.seeds.gpu
    draws = gumbel_sample(nucleus, expanded_idx_mapping, temperature, seeds, pos, False, False)
    sampled = sorted_ids.gather(1, draws.unsqueeze(1))
    if win_size <= 0:
        return sampled.squeeze(1), scores

    total_len = req_states.total_len.gpu[rows].long()
    prompt_len = req_states.prompt_len.gpu[rows].long()
    offsets = total_len.unsqueeze(1) - 1 - torch.arange(win_size, device=scores.device).unsqueeze(0)
    in_output = offsets >= prompt_len.unsqueeze(1)
    history = req_states.all_token_ids.gpu[rows.unsqueeze(1), offsets.clamp_min(0)]
    repeats = ((history == sampled) & in_output).sum(dim=1)
    rejected = (repeats >= win_size * tau_r) & in_output.any(dim=1) & (row_temperature > 0)
    remaining = scores.scatter(1, sampled, float("-inf"))
    remaining = torch.where(torch.isfinite(remaining).any(dim=1, keepdim=True), remaining, scores)
    # Without top-k/top-p, from a noise stream independent of the first draw.
    replacement = gumbel_sample(remaining, expanded_idx_mapping, temperature, seeds, pos, False, True)
    return torch.where(rejected, replacement, sampled.squeeze(1)), scores


class _SpeechTokenBatcher:
    """Run concurrent reference speech-token extractions as one S3 batch.

    Cold references are conditioned by several worker threads. One S3 call per
    reference runs batch-1 FP32 GEMMs, which cost ~3x more GPU time per
    reference than a batch of 8 and compete with the talker and codec stages.
    A single thread drains every request already queued into one padded
    ``quantize`` call; it never waits for more, so an idle server adds no
    latency. Padding changes ~0.1% of prompt tokens versus batch-1 extraction.
    """

    def __init__(self, model, s3, device, max_batch: int = 16) -> None:
        import queue
        import threading

        self._model, self._s3, self._device, self._max_batch = model, s3, device, max_batch
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._empty = queue.Empty
        threading.Thread(target=self._run, name="cosyvoice3-s3-batch", daemon=True).start()

    def tokens(self, mel: torch.Tensor) -> torch.Tensor:
        from concurrent.futures import Future

        future: Future = Future()
        self._queue.put((mel, future))
        return future.result()

    def _run(self) -> None:
        while True:
            items = [self._queue.get()]
            while len(items) < self._max_batch:
                try:
                    items.append(self._queue.get_nowait())
                except self._empty:
                    break
            try:
                mels, lens = self._s3.padding([mel for mel, _ in items])
                with torch.inference_mode():
                    codes, codes_lens = self._model.quantize(mels.to(self._device), lens.to(self._device))
                codes = codes.cpu()
                for (_, future), row, n in zip(items, codes, codes_lens.tolist()):
                    future.set_result(row[:n].clone())
            except BaseException as exc:  # noqa: BLE001 - surfaced to every waiting caller
                for _, future in items:
                    if not future.done():
                        future.set_exception(exc)


_REFERENCE_CONDITIONERS: dict[str, "CosyVoice3MultiModalProcessor"] = {}
_REFERENCE_CACHE_DEBUG = {True: 0, False: 0}
_REFERENCE_CONDITIONERS_LOCK = Lock()


def prefetch_reference_conditioning(model_dir: str, config, audio) -> None:
    """Warm the reference cache for ``audio`` (already at ``target_sr``).

    Runs in a worker thread before the request enters the synchronous input
    processor, so concurrent cold references are prepared in parallel (the
    ONNX/TensorRT/torch work releases the GIL) and the processor hits the
    cache instead of blocking the serving event loop.
    """
    with _REFERENCE_CONDITIONERS_LOCK:
        conditioner = _REFERENCE_CONDITIONERS.get(model_dir)
        if conditioner is None:
            conditioner = object.__new__(CosyVoice3MultiModalProcessor)
            conditioner._ensure_cached_runtime_components(model_dir, config)
            _REFERENCE_CONDITIONERS[model_dir] = conditioner
    _audio_conditioning(conditioner, audio, model_dir, config, {})


class CosyVoice3MultiModalProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        """If the config is not already present pass it
        as a class and it will try to find it in your
        model directory just copy the config class there also.
        """
        return self.ctx.get_hf_config(CosyVoice3Config)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        """How many audio can you pass. I think I should keep it as 1
        For now I have kept it None.
        """
        return {"audio": None}

    def get_data_parser(self):
        return MultiModalDataParser(
            target_sr=self.ctx.get_hf_config().target_sr,
            expected_hidden_size=self._get_expected_hidden_size(),
        )


class CosyVoice3MultiModalProcessor(OmniMultiModalProcessor[CosyVoice3MultiModalProcessingInfo]):
    def apply(self, inputs: ProcessorInputs, timing_ctx):
        tokenizer = self.info.get_tokenizer()
        prompt_text = tokenizer.decode(inputs.prompt, skip_special_tokens=False)
        config = self.info.ctx.get_hf_config()
        model_dir = self.info.ctx.model_config.model
        self._ensure_cached_runtime_components(model_dir, config)

        text_token, text_token_len = extract_text_token(
            prompt_text,
            self.tokenizer,
            config.allowed_special,
        )
        if inputs.mm_data_items.get_all_counts().get("audio", 0):
            reference_text = inputs.hf_processor_mm_kwargs.get("prompt_text")
            if not isinstance(reference_text, str):
                raise ValueError(f"prompt text is None : {reference_text}")
            prompt_text_token, prompt_text_token_len = extract_text_token(
                reference_text,
                self.tokenizer,
                config.allowed_special,
            )
            text_token, _ = concat_text_with_prompt_ids(
                text_token,
                text_token_len,
                prompt_text_token,
                prompt_text_token_len,
            )

        inputs = replace(
            inputs,
            prompt=text_token.reshape(-1).tolist(),
            hf_processor_mm_kwargs={
                **inputs.hf_processor_mm_kwargs,
                self._OMNI_PROMPT_TEXT_KEY: prompt_text,
            },
        )
        return super().apply(inputs, timing_ctx)

    def _ensure_cached_runtime_components(self, model_dir: str, config: CosyVoice3Config) -> None:
        cached_model_dir = getattr(self, "_cached_model_dir", None)
        if cached_model_dir == model_dir:
            return

        # The mm processor is re-created per request (deploy config sets
        # ``mm_processor_cache_gb: 0``), so the instance-level guard above never
        # hits across requests. Without a process-wide cache, every request
        # would re-run ``snapshot_download``, rebuild the Qwen tokenizer and
        # create a fresh ONNX campplus session — ~hundreds of ms of pure
        # overhead on the TTFP critical path. Build the heavy components once
        # per process, keyed by model_dir, and reuse them.
        comps = _RUNTIME_COMPONENTS_CACHE.get(model_dir)
        if comps is None:
            comps = self._build_runtime_components(model_dir, config)
            _RUNTIME_COMPONENTS_CACHE[model_dir] = comps

        self.tokenizer = comps["tokenizer"]
        self.feat_extractor = comps["feat_extractor"]
        self.campplus_session = comps["campplus_session"]
        self.campplus_trt = comps["campplus_trt"]
        self._cached_model_dir = model_dir
        self._speaker_cache = get_speaker_cache()

    def _build_runtime_components(self, model_dir: str, config: CosyVoice3Config) -> dict:
        """Build the per-model runtime components once (cached process-wide)."""
        # If model_dir is an HF repo ID (not a local path), resolve to cache.
        if not os.path.isdir(model_dir):
            model_dir = hf_api().snapshot_download(model_dir)

        tokenizer = get_qwen_tokenizer(
            token_path=os.path.join(model_dir, config.qwen_pretrain_path),
            skip_special_tokens=config.skip_special_tokens,
            version=config.version,
        )
        feat_extractor = partial(mel_spectrogram, **getattr(config, "feat_extractor", {}))

        campplus_onnx_path = os.path.join(model_dir, config.campplus_onxx_path)
        # TensorRT speaker-embedding path (default on); a prebuilt TRT engine
        # runs campplus on GPU. The engine itself is cached process-wide by
        # ``get_campplus_trt``.
        campplus_trt = None
        if self._speaker_embedding_trt_enabled() and torch.cuda.is_available():
            try:
                from vllm_omni.model_executor.models.cosyvoice3.speaker_embedding_trt import (
                    get_campplus_trt,
                )

                campplus_trt = get_campplus_trt(campplus_onnx_path, device="cuda")
                logger.info("CosyVoice3: using TensorRT campplus speaker embedding")
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.warning(
                    "CosyVoice3: TensorRT campplus build failed (%s); falling back to ONNX-Runtime speaker embedding",
                    exc,
                )
                campplus_trt = None

        # Only build the CPU ONNX campplus session when TRT is unavailable —
        # otherwise it is never used and creating it (ORT_ENABLE_ALL graph
        # optimization) costs ~hundreds of ms.
        campplus_session = None
        if campplus_trt is None:
            option = onnxruntime.SessionOptions()
            option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            option.intra_op_num_threads = 1
            campplus_session = onnxruntime.InferenceSession(
                campplus_onnx_path,
                sess_options=option,
                providers=_campplus_onnx_providers(),
            )

        return {
            "tokenizer": tokenizer,
            "feat_extractor": feat_extractor,
            "campplus_session": campplus_session,
            "campplus_trt": campplus_trt,
        }

    @staticmethod
    def _speaker_embedding_trt_enabled() -> bool:
        """Whether to use the TensorRT campplus speaker-embedding path (default on)."""
        return _cosyvoice3_trt_enabled()

    # Class-level cached s3tokenizer model — loaded once per process on first
    # call to ``_extract_speech_token_via_s3`` and shared across all
    # processor instances.
    _s3_model = None

    @classmethod
    def _ensure_s3_model(cls):
        if cls._s3_model is not None:
            return cls._s3_model
        with _REFERENCE_CONDITIONERS_LOCK:
            if cls._s3_model is None:
                cls._s3_model = cls._load_s3_model()
        return cls._s3_model

    @classmethod
    def _load_s3_model(cls):
        # s3tokenizer is imported lazily (kept off the module top level) so
        # callers that don't use the CosyVoice3 talker need not install it.
        try:
            import s3tokenizer as _s3
        except ImportError as e:
            raise ImportError(
                "CosyVoice3 speech-token extraction requires the 's3tokenizer' "
                "package; install it with `pip install s3tokenizer`."
            ) from e

        model = _s3.load_model("speech_tokenizer_v3_25hz")
        device = current_omni_platform.get_torch_device()
        model = model.to(device).eval()
        return (model, _s3, device)

    _s3_batcher: "_SpeechTokenBatcher | None" = None

    @classmethod
    def _ensure_s3_batcher(cls) -> "_SpeechTokenBatcher":
        if cls._s3_batcher is None:
            model, s3, device = cls._ensure_s3_model()
            with _REFERENCE_CONDITIONERS_LOCK:
                if cls._s3_batcher is None:
                    cls._s3_batcher = _SpeechTokenBatcher(model, s3, device)
        return cls._s3_batcher

    def _extract_speech_token_via_s3(self, audio, return_device):
        """Drop-in replacement for ``extract_speech_token`` that uses the
        S3Tokenizer PyTorch model on GPU. Returns the same
        ``(speech_token[1, T], speech_token_len[1])`` int32 tensors as the
        ONNX path so the rest of ``_call_hf_processor`` is unchanged.
        """
        model, _s3, dev = self._ensure_s3_model()

        # audio is a (waveform_ndarray, sr) tuple; resample to 16 kHz mono float32.
        wav, sr = audio
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if int(sr) != 16000:
            g = gcd(int(sr), 16000)
            wav = resample_poly(wav, 16000 // g, int(sr) // g).astype(np.float32)
        audio_t = torch.from_numpy(wav)

        mel = _s3.log_mel_spectrogram(audio_t)
        if torch.device(dev).type == "cuda":
            codes = self._ensure_s3_batcher().tokens(mel)
        else:
            mels_p, mels_lens = _s3.padding([mel])
            with torch.inference_mode():
                codes, codes_lens = model.quantize(mels_p.to(dev), mels_lens.to(dev))
            codes = codes[0, : int(codes_lens[0].item())]
        speech_token = codes.reshape(1, -1).to(dtype=torch.int32, device=return_device)
        speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32, device=return_device)
        return speech_token, speech_token_len

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """
        apply-> cached_apply_hf_processor -> apply_hf_processor_mm ->
        _call_hf_processor.
        _call_hf_processor takes input prompt and mm_data and returns
        token ids and tensors
        """
        config = self.info.ctx.get_hf_config()
        model_dir = self.info.ctx.model_config.model
        self._ensure_cached_runtime_components(model_dir, config)

        audio = mm_data.get("audio", None)

        if audio is None:
            audio = mm_data.get("audios")
            if audio is not None:
                audio = audio[0], config.target_sr

        text_token, text_token_len = extract_text_token(prompt, self.tokenizer, config.allowed_special)
        if audio is None:
            # Text-only path for profiling/cache
            return BatchFeature({"input_ids": text_token, "input_len": [text_token_len]})

        prompt_text = mm_kwargs.get("prompt_text")

        if not isinstance(prompt_text, str):
            raise ValueError(f"prompt text is None : {prompt_text}")

        prompt_text_token, prompt_text_token_len = extract_text_token(
            prompt_text, self.tokenizer, config.allowed_special
        )

        input_ids, input_len = concat_text_with_prompt_ids(
            text_token,
            text_token_len,
            prompt_text_token,
            prompt_text_token_len,
        )
        logger.debug(
            "cosyvoice _call_hf_processor: prompt_text_token=%s text_token=%s input_ids=%s "
            "prompt_text_len=%s text_len=%s input_len=%s",
            prompt_text_token.tolist(),
            text_token.tolist(),
            input_ids.tolist(),
            int(prompt_text_token_len),
            int(text_token_len),
            int(input_len),
        )
        conditioning = _audio_conditioning(self, audio, model_dir, config, mm_kwargs)
        return BatchFeature(
            {
                "input_ids": input_ids,
                "speech_feat": conditioning["speech_feat"],
                "speech_token": conditioning["speech_token"],
                "speech_token_len": [conditioning["speech_token_len"]],
                "embedding": conditioning["embedding"],
            }
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: "BatchFeature",
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
            "speech_feat": MultiModalFieldConfig.batched("audio"),
            "speech_token": MultiModalFieldConfig.batched("audio"),
            "speech_token_len": MultiModalFieldConfig.batched("audio"),
            "embedding": MultiModalFieldConfig.batched("audio"),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        def insertion_end(item_idx):
            # TODO: Think if this can be done better
            # sos + task + audio token ... ideally this needs to be split into
            # two start and end but somehow I couldn't pass two of these
            # wutg target .start() and .end()
            token_len = out_mm_kwargs["audio"][0]["speech_token_len"].data[0].item()
            return [1] * (1 + 1 + token_len)

        return [
            PromptInsertion(
                modality="audio",
                target=PromptIndexTargets.start(),
                insertion=insertion_end,
            ),
        ]


class CosyVoice3DummyInputsBuilder(OmniDummyInputsBuilder[CosyVoice3MultiModalProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "Hello, this is a test of the CosyVoice3 system capability."

    def get_dummy_mm_data(
        self, seq_len: int, mm_counts: Mapping[str, int], mm_options: Mapping[str, BaseDummyOptions] | None = None
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio")
        max_prompt_seconds = 30
        prompt_sample_rate = 24000
        target_audio_length = max_prompt_seconds * prompt_sample_rate

        audio_overrides = mm_options.get("audio") if mm_options else None
        mm_data = {
            "audio": (
                self._get_dummy_audios(
                    length=target_audio_length,
                    num_audios=num_audios,
                    overrides=audio_overrides,
                )[0],
                24000,
            ),
        }
        return mm_data

    def get_dummy_processor_inputs(
        self, seq_len: int, mm_counts: Mapping[str, int], mm_options: Mapping[str, BaseDummyOptions] | None = None
    ) -> ProcessorInputs:
        inputs = super().get_dummy_processor_inputs(seq_len, mm_counts, mm_options)
        inputs.hf_processor_mm_kwargs = {"prompt_text": "Testing my voices. Why should I not?"}
        return inputs


@MULTIMODAL_REGISTRY.register_processor(
    CosyVoice3MultiModalProcessor,
    info=CosyVoice3MultiModalProcessingInfo,
    dummy_inputs=CosyVoice3DummyInputsBuilder,
)
class CosyVoice3Model(
    nn.Module,
    SupportsMultiModal,
):
    supports_multimodal_raw_input_only = True
    supports_multimodal = True
    requires_raw_input_tokens = True
    supports_embed_input_ids_query_start_loc = True
    prefer_model_sampler = True
    _sampling_eps = 1e-5

    @property
    def owns_generation_output_storage(self) -> bool:
        """Expose the codec's ownership contract on the model seen by the runner."""
        return self.model_stage == "cosyvoice3_code2wav" and (
            getattr(self.code2wav, "owns_generation_output_storage", False) is True
        )

    @property
    def logits_vocab_size(self) -> int | None:
        """Width of the logits ``compute_logits`` returns (the speech head).

        The runners drop stop ids beyond it from min-tokens masking: vLLM folds
        the text tokenizer's EOS into every request's stop set, which this
        head can never emit (and MRv2's unchecked mask kernel would write past
        the logits row).
        """
        if self.model_stage != "cosyvoice3_talker":
            return None
        return int(self.config.llm["speech_token_size"]) + 200

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        standard_sampling = cosyvoice3_standard_sampling(self.config)
        logger.info("CosyVoice3 sampling policy: %s", "standard" if standard_sampling else "ras")
        self.have_multimodal_outputs = True
        # Full-response codec consumes tokens and conditioning, not LLM hidden states.
        self.omni_pooler_payload_include_hidden = not cosyvoice3_packed_inference_enabled()
        self.model_stage = vllm_config.model_config.model_stage
        model_dir = vllm_config.model_config.model
        if not os.path.isdir(model_dir):
            model_dir = hf_api().snapshot_download(model_dir)
        self.model_dir = model_dir
        self.model = None
        if self.model_stage == "cosyvoice3_talker":
            # Code2Wav consumes sampled tokens and prompt conditioning, not
            # hidden states. The processor opts into token-only chunk updates.
            if getattr(vllm_config.model_config, "async_chunk", False):
                self.omni_pooler_payload_include_hidden = False
            # Initialize talker stage (text to speech tokens)
            from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_talker import CosyVoice3LM, VLLMQwen2Encoder

            llm_vllm_config = self._create_llm_vllm_config(vllm_config)
            llm = VLLMQwen2Encoder(vllm_config=llm_vllm_config, prefix="model")
            self.talker = CosyVoice3LM(
                llm_input_size=self.config.llm["llm_input_size"],
                llm_output_size=self.config.llm["llm_output_size"],
                speech_token_size=self.config.llm["speech_token_size"],
                llm=llm,
                length_normalized_loss=self.config.llm["length_normalized_loss"],
                lsm_weight=self.config.llm["lsm_weight"],
                mix_ratio=self.config.llm["mix_ratio"],
            )
            # KV cache is now managed externally by vLLM's PagedAttention
            # No need for self.llm_cache
            self.model = self.talker
            # The talker consumes only llm.pt; without this override the
            # default loader streams every *.pt in the model dir (flow.pt,
            # hift.pt) into load_weights, whose keys belong to other stages.
            self.allow_patterns_overrides = ["llm.pt"]
        elif self.model_stage == "cosyvoice3_code2wav":
            # Initialize code2wav stage (flow matching + vocoder)
            from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import CosyVoice3Code2Wav

            self.code2wav = CosyVoice3Code2Wav(self.config)
            self.model = self.code2wav.flow_model
            self.hift = self.code2wav.hift
            # Keep additional information synchronized for async_chunk updates.
            self.enable_update_additional_information = True

            self._stream_audio_cache_lock = Lock()
            self._stream_vocoder_cache_by_req: dict[str, dict[str, torch.Tensor]] = {}
        else:
            raise ValueError(f"Model stage not supported {self.model_stage}")

    def get_language_model(self) -> "nn.Module":
        """Return the language model for upstream MoE detection."""
        if hasattr(self.model, "get_language_model"):
            return self.model.get_language_model()
        return self.model

    def _create_llm_vllm_config(self, parent_config: VllmConfig) -> VllmConfig:
        """Create VllmConfig for the inner Qwen2 LLM.

        This creates a modified VllmConfig with the Qwen2 HF config loaded from
        the pretrained model directory. The cache config is inherited from the parent
        to enable PagedAttention with the same memory configuration.
        """
        qwen_config_path = os.path.join(self.model_dir, self.config.llm["llm"]["pretrain_path"])
        qwen_hf_config = Qwen2Config.from_pretrained(qwen_config_path)

        # Use parent's cache config - critical for PagedAttention to work correctly
        return parent_config.with_hf_config(qwen_hf_config, architectures=["Qwen2Model"])

    def on_requests_finished(self, request_ids: set[str]) -> None:
        """Release per-request vocoder state on completion, cancellation or error."""
        if hasattr(self, "_stream_vocoder_cache_by_req"):
            with self._stream_audio_cache_lock:
                for request_id in request_ids:
                    self._stream_vocoder_cache_by_req.pop(request_id, None)

    def _stitch_stream_audio(self, req_id: str | None, audio: torch.Tensor, stream_finished: bool) -> torch.Tensor:
        """Pass-through stitching for async_chunk.

        Chunk overlap is already removed in mel domain via token_offset_tokens.
        Applying an additional waveform-domain fade/cache step introduces either
        duplicated overlap (if no tail trim) or duration shrink (if tail trim).
        """
        if req_id is not None and stream_finished and hasattr(self, "_stream_vocoder_cache_by_req"):
            with self._stream_audio_cache_lock:
                self._stream_vocoder_cache_by_req.pop(req_id, None)
        return audio

    @staticmethod
    def _split_request_ids(ids: torch.Tensor, seq_token_counts: list[int] | None = None) -> list[torch.Tensor]:
        """Split concatenated input_ids into per-request segments."""
        if seq_token_counts is not None:
            boundaries = [0]
            for count in seq_token_counts:
                boundaries.append(boundaries[-1] + int(count))
            total = ids.numel()
            return [ids[boundaries[i] : min(boundaries[i + 1], total)] for i in range(len(seq_token_counts))]

        if is_forward_context_available():
            slices = get_forward_context().ubatch_slices
            if slices is not None and len(slices) > 1 and not any(hasattr(s, "token_slice") for s in slices):
                boundaries = [0]
                for s in slices:
                    boundaries.append(boundaries[-1] + int(s))
                return [ids[boundaries[i] : boundaries[i + 1]] for i in range(len(boundaries) - 1)]

        return [ids]

    def _sanitize_codec_tokens(self, req_ids: torch.Tensor) -> torch.Tensor:
        """Filter non-code tokens before feeding flow token embedding."""
        vocab_size = int(self.code2wav.input_embedding.num_embeddings)
        valid_mask = (req_ids >= 0) & (req_ids < vocab_size)
        return req_ids[valid_mask]

    @staticmethod
    def _req_scalar(param: torch.Tensor | None, req_idx: int, default: float | int) -> float | int:
        if param is None or param.numel() == 0:
            return default
        index = min(req_idx, int(param.numel()) - 1)
        value = param.reshape(-1)[index].item()
        if isinstance(default, int):
            return int(value)
        return float(value)

    @staticmethod
    def _random_sample_one(probs: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        return random_sample(probs.unsqueeze(0), {} if generator is None else {0: generator}).reshape(())

    @classmethod
    def _nucleus_sample_one(
        cls,
        weighted_scores: torch.Tensor,
        *,
        top_p: float,
        top_k: int,
        generator: torch.Generator | None,
    ) -> int:
        """Vectorized nucleus + top-k sampling.

        Distribution-equivalent to the reference iterative implementation: the
        keep-set is identical (token i is kept iff
        ``cumsum(sorted_probs)[i] - sorted_probs[i] < top_p`` AND ``i < top_k``)
        and the renormalized sampling distribution matches, but the exact token
        drawn for a given seed is NOT guaranteed to match. The reference draws
        via ``multinomial`` over the stacked kept subset while this draws over
        the full sorted vector (zeroed outside the keep-set), so the generator
        advances over different-sized inputs and may yield a different sample.
        The win: no per-token ``.item()`` D2H syncs from the Python loop —
        those dominated the sampler CPU time in profiling.
        """
        probs = weighted_scores.softmax(dim=0)
        sorted_prob, sorted_idx = probs.sort(descending=True, stable=True)
        cum_before = sorted_prob.cumsum(dim=0) - sorted_prob
        mask = cum_before < top_p
        if top_k > 0:
            n = sorted_prob.shape[0]
            mask = mask & (torch.arange(n, device=mask.device) < min(int(top_k), n))
        weights = sorted_prob * mask.to(sorted_prob.dtype)
        # First token always passes (cum_before[0] = 0 < top_p for any top_p > 0),
        # so ``weights`` is guaranteed to have at least one nonzero entry. The
        # final ``.item()`` is the ONLY D2H sync per call.
        sample_idx = cls._random_sample_one(weights, generator=generator)
        return int(sorted_idx[sample_idx].item())

    @classmethod
    def _ras_sample_one(
        cls,
        weighted_scores: torch.Tensor,
        decoded_tokens: Sequence[int],
        *,
        top_p: float,
        top_k: int,
        win_size: int,
        tau_r: float,
        generator: torch.Generator | None,
    ) -> int:
        top_id = cls._nucleus_sample_one(
            weighted_scores,
            top_p=top_p,
            top_k=top_k,
            generator=generator,
        )
        if win_size > 0 and decoded_tokens:
            recent = torch.as_tensor(
                list(decoded_tokens[-win_size:]),
                device=weighted_scores.device,
                dtype=torch.long,
            )
            rep_num = int((recent == top_id).sum().item())
            if rep_num >= win_size * tau_r:
                weighted_scores = weighted_scores.clone()
                original_score = weighted_scores[top_id].clone()
                weighted_scores[top_id] = float("-inf")
                weighted_scores[top_id] = torch.where(
                    torch.isfinite(weighted_scores).any(),
                    weighted_scores[top_id],
                    original_score,
                )
                top_id = int(cls._random_sample_one(weighted_scores.softmax(dim=0), generator=generator).item())
        return top_id

    def _cosyvoice3_ras_enabled(self, sampling_metadata: SamplingMetadata) -> bool:
        if self.model_stage != "cosyvoice3_talker":
            return False
        if sampling_metadata.max_num_logprobs is not None:
            return False
        if sampling_metadata.temperature is None:
            return False
        if bool(sampling_metadata.bad_words_token_ids):
            return False
        if torch.any(sampling_metadata.frequency_penalties != 0):
            return False
        if torch.any(sampling_metadata.presence_penalties != 0):
            return False
        return True

    def _ras_sample_batch(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        *,
        default_top_p: float,
        default_top_k: int,
        win_size: int,
        tau_r: float,
    ) -> torch.Tensor:
        """Batch random RAS without per-row scalar parameter/token transfers.

        Rejection flags cross to the host once per batch so only requests that
        actually reject a token consume a second RNG draw. Per-request seeded
        generators retain their ownership when rejection compacts the batch.
        """
        batch_size = logits.shape[0]

        def parameter(value: torch.Tensor | None, default: float | int) -> torch.Tensor:
            if value is None or value.numel() == 0:
                return torch.full((batch_size,), default, device=logits.device)
            value = value.reshape(-1).to(device=logits.device)
            if value.numel() < batch_size:
                value = torch.cat((value, value[-1:].expand(batch_size - value.numel())))
            return value[:batch_size]

        temperature = parameter(sampling_metadata.temperature, 1.0)
        top_p = parameter(sampling_metadata.top_p, default_top_p)
        top_k = parameter(sampling_metadata.top_k, default_top_k)
        weighted_scores = torch.log_softmax(logits / temperature.clamp_min(self._sampling_eps).unsqueeze(1), dim=1)
        sorted_probs, sorted_ids = weighted_scores.softmax(dim=1).sort(dim=1, descending=True, stable=True)
        # Keep top-p based on the full distribution, before top-k masking.
        keep = (sorted_probs.cumsum(dim=1) - sorted_probs) < top_p.unsqueeze(1)
        ranks = torch.arange(logits.shape[1], device=logits.device)
        keep &= (top_k.unsqueeze(1) <= 0) | (ranks.unsqueeze(0) < top_k.unsqueeze(1))
        weights = sorted_probs * keep.to(sorted_probs.dtype)
        generators = {i: g for i, g in sampling_metadata.generators.items() if i < batch_size}
        draws = random_sample(weights, generators).reshape(-1, 1).long()
        sampled = sorted_ids.gather(1, draws).squeeze(1)

        histories = sampling_metadata.output_token_ids
        if win_size > 0 and any(histories[:batch_size]):
            recent = []
            for i in range(batch_size):
                row = list(histories[i][-win_size:]) if i < len(histories) else []
                recent.append([-1] * (win_size - len(row)) + row)
            history = torch.tensor(recent, dtype=torch.long, device=logits.device)
            repeated = ((history == sampled.unsqueeze(1)).sum(dim=1) >= win_size * tau_r) & (history >= 0).any(dim=1)
            rejected_rows = [i for i, reject in enumerate(repeated.tolist()) if reject]
            if rejected_rows:
                rows = torch.tensor(rejected_rows, device=logits.device, dtype=torch.long)
                scores = weighted_scores.index_select(0, rows)
                rejected_ids = sampled.index_select(0, rows).unsqueeze(1)
                original = scores.gather(1, rejected_ids)
                scores.scatter_(1, rejected_ids, float("-inf"))
                replacement = torch.where(
                    torch.isfinite(scores).any(dim=1, keepdim=True),
                    torch.full_like(original, float("-inf")),
                    original,
                )
                scores.scatter_(1, rejected_ids, replacement)
                compact_generators = {i: generators[row] for i, row in enumerate(rejected_rows) if row in generators}
                # RAS rejection samples the complete remaining distribution,
                # without applying top-k/top-p a second time.
                replacement_ids = random_sample(scores.softmax(dim=1), compact_generators).reshape(-1).long()
                sampled.index_copy_(0, rows, replacement_ids)
        return sampled.to(torch.int32)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        if logits is None or logits.numel() == 0:
            return None
        if self.model_stage != "cosyvoice3_talker":
            return None

        if cosyvoice3_standard_sampling(self.config):
            sampler = getattr(self, "_talker_sampler", None)
            if sampler is None:
                sampler = Sampler()
                self._talker_sampler = sampler
            # Penalize generated tokens only, never text or reference
            # speech in the multimodal prompt. Padding is ignored by vLLM.
            if sampling_metadata.prompt_token_ids is not None:
                sampling_metadata = replace(
                    sampling_metadata,
                    prompt_token_ids=torch.full_like(sampling_metadata.prompt_token_ids, logits.shape[-1]),
                )
            return sampler(logits=logits, sampling_metadata=sampling_metadata)

        if not self._cosyvoice3_ras_enabled(sampling_metadata):
            sampler = getattr(self, "_talker_sampler", None)
            if sampler is None:
                sampler = Sampler()
                self._talker_sampler = sampler
            return sampler(logits=self._full_vocab_logits(logits), sampling_metadata=sampling_metadata)

        logits = logits.to(torch.float32)
        mask = sampling_metadata.allowed_token_ids_mask
        if mask is not None and mask.shape[-1] != logits.shape[-1]:
            sampling_metadata = replace(sampling_metadata, allowed_token_ids_mask=mask[..., : logits.shape[-1]])
        # Apply logits processors directly — RAS handles its own repetition
        # logic.  We avoid instantiating Sampler() here because its import
        # chain pulls in flashinfer / GPU deps that fail in CPU-only tests.
        if sampling_metadata.allowed_token_ids_mask is not None:
            logits.masked_fill_(sampling_metadata.allowed_token_ids_mask, float("-inf"))
        for processor in sampling_metadata.logitsprocs.non_argmax_invariant:
            logits = processor.apply(logits)
        # The text tokenizer vocabulary is much larger than the speech head.
        # compute_logits pads its output for the generic sampler/processors;
        # RAS only needs speech codes and the merged stop token. Keep the
        # original token indices, and trim only after full-vocabulary masks
        # and processors have run. The generic sampler fallback above retains
        # its full-vocabulary contract (including logprobs and bad words).
        speech_head_size = int(self.config.llm["speech_token_size"]) + 200
        logits = logits[..., :speech_head_size]
        finite_logits = torch.isfinite(logits)
        if not finite_logits.any(dim=-1).all().item():
            raise ValueError("CosyVoice3 sampling received a row with no finite logits")
        logits.masked_fill_(~finite_logits, float("-inf"))

        sampling_cfg = dict(self.config.llm.get("sampling", {}))
        default_top_p = float(sampling_cfg.get("top_p", 0.8))
        default_top_k = int(sampling_cfg.get("top_k", 25))
        win_size = int(sampling_cfg.get("win_size", 10))
        tau_r = float(sampling_cfg.get("tau_r", 0.1))

        if sampling_metadata.all_random:
            sampled = self._ras_sample_batch(
                logits,
                sampling_metadata,
                default_top_p=default_top_p,
                default_top_k=default_top_k,
                win_size=win_size,
                tau_r=tau_r,
            )
            return SamplerOutput(sampled_token_ids=sampled.unsqueeze(-1), logprobs_tensors=None)

        sampled_ids: list[int] = []
        for req_idx in range(int(logits.shape[0])):
            row_logits = logits[req_idx]

            temperature = float(self._req_scalar(sampling_metadata.temperature, req_idx, 1.0))
            if temperature < self._sampling_eps:
                sampled_ids.append(int(torch.argmax(row_logits).item()))
                continue

            top_p = float(self._req_scalar(sampling_metadata.top_p, req_idx, default_top_p))
            top_k = int(self._req_scalar(sampling_metadata.top_k, req_idx, default_top_k))
            generator = sampling_metadata.generators.get(req_idx)
            weighted_scores = torch.log_softmax(row_logits / max(temperature, self._sampling_eps), dim=0)
            decoded_tokens = (
                sampling_metadata.output_token_ids[req_idx] if req_idx < len(sampling_metadata.output_token_ids) else []
            )
            sampled_ids.append(
                self._ras_sample_one(
                    weighted_scores,
                    decoded_tokens,
                    top_p=top_p,
                    top_k=top_k,
                    win_size=win_size,
                    tau_r=tau_r,
                    generator=generator,
                )
            )

        sampled = torch.tensor(sampled_ids, device=logits.device, dtype=torch.int32)
        return SamplerOutput(sampled_token_ids=sampled.unsqueeze(-1), logprobs_tensors=None)

    def mrv2_custom_sampler(self, sampler: Any) -> None:
        """Model Runner V2: the V1 sampling policies on the MRv2 sampler.

        RAS replaces the sampling step (``_ras_mrv2_sample``). Standard
        sampling penalizes generated speech tokens only: the V1 path hides the
        prompt from penalties (SGLang parity: text and reference speech are
        never penalized), while MRv2 keeps penalty statistics on the GPU and
        bins every prompt token of a new request, whose text ids also lie
        outside the narrow speech head. Bin the prompt into a scratch mask
        instead, keeping resumed output tokens counted.
        """
        if self.model_stage != "cosyvoice3_talker":
            return None
        if not cosyvoice3_standard_sampling(self.config):
            sampling_cfg = dict(self.config.llm.get("sampling", {}))
            ras = partial(
                _ras_mrv2_sample,
                default_top_p=float(sampling_cfg.get("top_p", 0.8)),
                default_top_k=int(sampling_cfg.get("top_k", 25)),
                win_size=int(sampling_cfg.get("win_size", 10)),
                tau_r=float(sampling_cfg.get("tau_r", 0.1)),
                eps=self._sampling_eps,
            )
            sampler.sample = MethodType(ras, sampler)
        state = getattr(sampler, "penalties_state", None)
        if state is not None:
            state.apply_staged_writes = MethodType(_generated_only_penalty_writes, state)
        # MRv2 captures tensor outputs and constructs OmniOutput afterwards.
        # This contract also applies when only the codec stage enables packed Flow.
        self._mrv2_tensor_output = True
        # MRv2 hands make_omni_output no multimodal kwargs; carry each prompt's
        # conditioning from the encoder call to its batch row instead.
        self._mrv2_encoded_conditioning = OrderedDict()
        return None

    def _full_vocab_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Pad the speech head to the text vocabulary for the generic sampler contract."""
        pad_size = int(self.config.vocab_size) - logits.size(-1)
        if pad_size <= 0:
            return logits
        return torch.cat([logits, logits.new_full(logits.shape[:-1] + (pad_size,), float("-inf"))], dim=-1)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if self.model_stage == "cosyvoice3_talker":
            logits = self.model.llm_decoder(hidden_states)
            # The decoder outputs speech_token_size + 200 logits.  The official
            # CosyVoice3 treats ALL tokens >= speech_token_size (the last 200)
            # as stop signals.  Merge their probabilities into a single EOS
            # token (6562) via logsumexp so that vLLM's stop_token_ids=[6562]
            # fires with the correct aggregate stop probability.
            speech_token_size = self.config.llm["speech_token_size"]
            eos_idx = self.config.llm["eos_token_id"]
            if not cosyvoice3_standard_sampling(self.config):
                stop_logits = logits[..., speech_token_size:]  # last 200
                merged_stop = torch.logsumexp(stop_logits, dim=-1, keepdim=True)
                logits[..., speech_token_size:] = float("-inf")  # mask all
                logits[..., eos_idx] = merged_stop.squeeze(-1)  # restore merged
            # The model owns sampling and the runner accepts a narrow codec
            # head, so keep only the speech head: padding every decode step to
            # the ~152k text vocabulary multiplied sampler and penalty work.
            # The generic-sampler fallback pads on demand (_full_vocab_logits).
            return logits
        else:
            raise RuntimeError(f"compute_logits is only valid for {self.model_stage}.")

    def embed_multimodal(self, **kwargs: object):
        if self.model_stage == "cosyvoice3_talker":
            speech_token = kwargs["speech_token"]
            # vLLM's _execute_mm_encoder batches ALL multimodal items scheduled
            # in one engine step into a single call, expecting one embedding
            # tensor per item back. When >=2 requests prefill in the same step
            # (likely once mm preprocessing is fast, e.g. TensorRT speaker
            # embedding), speech_token arrives as a list of per-item tensors
            # (variable length), so embed it per item and return a list of
            # [T_i, emb] tensors. The single-item path keeps returning a
            # [1, T, emb] tensor (the caller's extend() iterates dim 0).
            if isinstance(speech_token, (list, tuple)):
                emb_dim = self.model.speech_embedding.weight.shape[1]
                embeddings = [self.model.speech_embedding(t).reshape(-1, emb_dim) for t in speech_token]
            else:
                embeddings = self.model.speech_embedding(speech_token)
            encoded = getattr(self, "_mrv2_encoded_conditioning", None)
            if encoded is not None:
                conditioning = self._split_prompt_conditioning(
                    speech_token, kwargs.get("speech_feat"), kwargs.get("embedding"), kwargs.get("speech_token_len")
                )
                for item, embedding in enumerate(embeddings):
                    # The encoder cache hands this item's tensor back (whole) to
                    # embed_input_ids, where its storage identifies the prompt.
                    # Identical references share one cached tensor, so entries
                    # stay until a newer encoding reuses the storage.
                    key = embedding.data_ptr()
                    encoded[key] = tuple(None if values is None else values[item] for values in conditioning)
                    encoded.move_to_end(key)
                while len(encoded) > _MRV2_CONDITIONING_ENTRIES:
                    encoded.popitem(last=False)
            return embeddings
        else:
            raise RuntimeError(f"embed_multimodal is only valid for {self.model_stage}.")

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
        query_start_loc: Sequence[int] | None = None,
    ) -> torch.Tensor:
        if self.model_stage == "cosyvoice3_talker":
            # Decode-only steps carry no speech embeddings; checking the
            # (device) placeholder mask would synchronize every step.
            if not multimodal_embeddings or is_multimodal is None or not torch.any(is_multimodal):
                return self.model.speech_embedding.weight[input_ids]

            # Requests can interleave new prefills and ongoing decodes after
            # scheduler slot reuse. A placeholder group identifies the start
            # of a prompt, but cannot identify its end: use runner boundaries.
            total_tokens = input_ids.numel()
            boundaries = list(query_start_loc) if query_start_loc is not None else [0, total_tokens]
            if not boundaries or boundaries[0] != 0 or boundaries[-1] != total_tokens:
                raise ValueError("CosyVoice3 input embedding boundaries must cover all scheduled tokens")
            if any(end <= start for start, end in zip(boundaries, boundaries[1:])):
                raise ValueError("CosyVoice3 input embedding boundaries must be strictly increasing")
            mm_mask = is_multimodal.reshape(-1).tolist()
            # Conditioning contains prefill rows only, whereas the downstream
            # payload splitter indexes the full (prefill + decode) request batch.
            self._conditioning_request_rows = [
                i for i, (start, end) in enumerate(zip(boundaries, boundaries[1:])) if any(mm_mask[start:end])
            ]
            self._conditioning_request_count = len(boundaries) - 1
            self._mrv2_step_conditioning = []

            text_embeds = self.model.llm.model.embed_tokens(input_ids)
            sos = self.model.speech_embedding.weight[self.model.sos].reshape(1, -1)
            task_id = self.model.speech_embedding.weight[self.model.task_id].reshape(1, -1)
            segments = []
            mm_index = 0
            for start, end in zip(boundaries, boundaries[1:]):
                request_mask = mm_mask[start:end]
                if not any(request_mask):
                    segments.append(self.model.speech_embedding.weight[input_ids[start:end]])
                    continue
                if multimodal_embeddings is None or mm_index >= len(multimodal_embeddings):
                    raise ValueError("CosyVoice3 prefill is missing its speech embedding")
                speech = multimodal_embeddings[mm_index]
                mm_index += 1
                encoded = getattr(self, "_mrv2_encoded_conditioning", None)
                if encoded is not None:
                    key = speech.data_ptr()
                    conditioning = encoded.get(key)
                    if conditioning is None:
                        raise ValueError("CosyVoice3 MRv2 prefill lost its prompt conditioning")
                    encoded.move_to_end(key)
                    self._mrv2_step_conditioning.append(conditioning)
                prefix_len = 2 + speech.shape[0]
                # The processor marks SOS, TASK_ID and all speech placeholders.
                # Rearrangement needs the complete prompt block in this call.
                if request_mask != [True] * prefix_len + [False] * (end - start - prefix_len):
                    raise ValueError("CosyVoice3 requires a complete prefill block within each request boundary")
                segments.extend((sos, text_embeds[start + prefix_len : end], task_id, speech))
            if mm_index != len(multimodal_embeddings):
                raise ValueError("CosyVoice3 speech embeddings must match prefill requests one to one")
            return torch.cat(segments, dim=0)
        elif self.model_stage == "cosyvoice3_code2wav":
            assert input_ids.dim() == 1
            return torch.zeros((input_ids.shape[0], int(self.config.hidden_size)), device=input_ids.device)
        else:
            raise RuntimeError(f"embed_input_ids is not valid for {self.model_stage}.")

    def _align_prompt_conditioning(self, values):
        rows = getattr(self, "_conditioning_request_rows", None)
        if values is None or rows is None:
            return values
        if len(rows) != len(values):
            raise ValueError("CosyVoice3 conditioning must match the prefill request rows")
        aligned = [None] * self._conditioning_request_count
        for row, value in zip(rows, values):
            aligned[row] = value
        return aligned

    @staticmethod
    def _split_prompt_conditioning(speech_token, speech_feat, embedding, speech_token_len):
        """Split collated prompt conditioning into per-request lists.

        Returns ``(speech_token_list, speech_feat_list, embedding_list,
        speech_token_len_list)``, one rank-correct tensor per request:
        ``speech_token`` ``[1, T]`` (possibly right-padded), ``speech_feat``
        ``[1, 2T, F]``, ``embedding`` ``[1, D]``, ``speech_token_len`` ``[1, 1]``.

        This runs inside the talker ``forward`` which may be CUDA-graph
        captured, so it must NOT trigger any host<->device sync (no ``.item()``
        / ``.tolist()`` / Python-int slicing by tensor value). Splitting is pure
        on-device indexing; right-padding is dropped later in the (eager)
        code2wav stage using the per-request ``speech_token_len``. Robust to
        inputs arriving as padded ``[B, ...]`` batch tensors or per-request
        lists.
        """

        def _rows(x, want_dim):
            if isinstance(x, (list, tuple)):
                items = list(x)
            elif isinstance(x, torch.Tensor):
                # ``x.shape[0]`` is a static Python int — no device sync.
                items = [x[i] for i in range(x.shape[0])]
            else:
                return None
            out = []
            for t in items:
                if isinstance(t, torch.Tensor):
                    while t.dim() < want_dim:
                        t = t.unsqueeze(0)
                    if t.shape[0] != 1:
                        t = t[:1]
                    t = t.contiguous()
                out.append(t)
            return out

        st_out = _rows(speech_token, 2)
        sf_out = _rows(speech_feat, 3)
        emb_out = _rows(embedding, 2)
        stl_out = _rows(speech_token_len, 2)
        return st_out, sf_out, emb_out, stl_out

    def _resolve_flow_estimator_onnx(self) -> str | None:
        """Locate the flow-decoder estimator ONNX for the TensorRT engine.

        Prefers the fp16 (strongly-typed) ONNX → fp16 engine. Order: env
        ``COSYVOICE3_ESTIMATOR_ONNX`` → ``<model_dir>/<fp16 name>`` → fetched
        from ``flow_estimator_onnx_repo`` → bundled fp32 ONNX (fp32+TF32). Each
        is checked for existence; returns ``None`` if nothing is found.
        """
        env_path = os.environ.get("COSYVOICE3_ESTIMATOR_ONNX")
        if env_path and os.path.exists(env_path):
            return env_path

        fp16_name = getattr(self.config, "flow_estimator_onnx_path", "flow.decoder.estimator.autocast_fp16.onnx")
        local_fp16 = os.path.join(self.model_dir, fp16_name)
        if os.path.exists(local_fp16):
            return local_fp16

        repo = getattr(self.config, "flow_estimator_onnx_repo", None)
        if repo:
            try:
                fetched_dir = hf_api().snapshot_download(repo, allow_patterns=[fp16_name])
                fetched = os.path.join(fetched_dir, fp16_name)
                if os.path.exists(fetched):
                    return fetched
            except Exception as exc:  # pragma: no cover - network/repo issues
                logger.warning("CosyVoice3 code2wav: could not fetch fp16 estimator ONNX from %s (%s)", repo, exc)

        fp32_name = getattr(self.config, "flow_estimator_onnx_path_fp32", "flow.decoder.estimator.fp32.onnx")
        local_fp32 = os.path.join(self.model_dir, fp32_name)
        if os.path.exists(local_fp32):
            logger.info("CosyVoice3 code2wav: fp16 estimator ONNX unavailable, falling back to fp32 (%s)", local_fp32)
            return local_fp32
        return None

    def _maybe_enable_code2wav_trt(self) -> None:
        """Swap the flow-decoder estimator to a TensorRT engine once (lazy).

        Runs on the first code2wav step — after weights are loaded — so the
        torch estimator is fully built first and then dropped. No-op unless
        ``COSYVOICE3_TRT`` is on (default), CUDA is available, and the estimator
        ONNX ships with the model. Falls back to the torch estimator on any
        failure. The upstream ``CausalConditionalCFM.forward_estimator`` detects
        the non-``nn.Module`` estimator and drives the TRT engine.
        """
        if getattr(self, "_code2wav_trt_done", False):
            return
        self._code2wav_trt_done = True

        # Packed Flow owns the torch estimator; speaker TRT remains independent.
        if cosyvoice3_packed_inference_enabled():
            return
        if not (_cosyvoice3_trt_enabled() and torch.cuda.is_available()):
            return
        onnx_path = self._resolve_flow_estimator_onnx()
        if onnx_path is None:
            logger.warning("CosyVoice3 code2wav: no flow-estimator ONNX available; keeping torch estimator")
            return
        try:
            from vllm_omni.model_executor.models.cosyvoice3.flow_estimator_trt import (
                build_flow_estimator_trt,
            )

            wrapper = build_flow_estimator_trt(onnx_path, device="cuda")
            # ``estimator`` is a registered nn.Module submodule; delete it first
            # (frees the torch estimator weights) so the TRT wrapper can be set
            # as a plain attribute — nn.Module.__setattr__ rejects non-Modules.
            decoder = self.code2wav.flow_model.decoder
            del decoder.estimator
            decoder.estimator = wrapper
            logger.info("CosyVoice3: using TensorRT flow-decoder estimator (code2wav)")
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(
                "CosyVoice3 code2wav: TensorRT estimator build failed (%s); keeping torch estimator",
                exc,
            )

    def make_omni_output(self, hidden_states: torch.Tensor, **kwargs) -> OmniOutput:
        """Attach live prefill conditioning outside the full-response CUDA graph.

        Graph replay reuses capture-time output containers. Returning conditioning
        from the captured forward would replay dummy prompt tensors on every
        decode step. The runner calls this adapter with the current step's kwargs.
        """
        multimodal_outputs = {}

        step_conditioning = getattr(self, "_mrv2_step_conditioning", None)
        if "speech_token" not in kwargs and step_conditioning:
            self._mrv2_step_conditioning = None
            speech_token, speech_feat, embedding, speech_token_len = (
                list(values) for values in zip(*step_conditioning)
            )
            multimodal_outputs = to_dict(
                OmniPayloadStruct(
                    embed=EmbeddingsStruct(
                        speech_token=self._align_prompt_conditioning(speech_token),
                        speech_feat=self._align_prompt_conditioning(speech_feat),
                        speech_token_len=self._align_prompt_conditioning(speech_token_len),
                        embedding=self._align_prompt_conditioning(embedding),
                    ),
                )
            )
        elif "speech_token" in kwargs:
            # Prompt conditioning tensors for code2wav: live under
            # ``embed.*`` per OmniPayloadStruct schema.
            #
            # vLLM hands these mm fields to forward() as collated, padded
            # batch tensors (speech_token [B, maxT], speech_feat
            # [B, 2*maxT, F], embedding [B, D]). Emitting them raw makes the
            # downstream per-request payload split fragile at batch>1: it
            # intermittently de-batches speech_token to 1-D and leaks the
            # whole [B, D] embedding to every request, which corrupts voice
            # conditioning and crashes code2wav (`prompt_token.shape[1]`).
            # Instead, split into an explicit per-request list of unpadded,
            # correctly-ranked tensors so ``to_payload_element`` splits them
            # deterministically by request index (list[idx]).
            speech_token_list, speech_feat_list, embedding_list, speech_token_len_list = (
                self._split_prompt_conditioning(
                    kwargs.get("speech_token"),
                    kwargs.get("speech_feat"),
                    kwargs.get("embedding"),
                    kwargs.get("speech_token_len"),
                )
            )
            speech_token_list = self._align_prompt_conditioning(speech_token_list)
            speech_feat_list = self._align_prompt_conditioning(speech_feat_list)
            embedding_list = self._align_prompt_conditioning(embedding_list)
            speech_token_len_list = self._align_prompt_conditioning(speech_token_len_list)
            multimodal_outputs = to_dict(
                OmniPayloadStruct(
                    embed=EmbeddingsStruct(
                        speech_token=speech_token_list,
                        speech_feat=speech_feat_list,
                        speech_token_len=speech_token_len_list,
                        embedding=embedding_list,
                    ),
                )
            )

        return OmniOutput(text_hidden_states=hidden_states, multimodal_outputs=multimodal_outputs)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        additional_information: dict[str, object] | None = None,
        **kwargs: object,
    ) -> torch.Tensor | OmniOutput:
        if self.model_stage == "cosyvoice3_talker":
            if inputs_embeds is None:
                inputs_embeds = self.embed_input_ids(input_ids)

            # [total_tokens, hidden]
            hidden_states = self.model.llm(inputs_embeds, positions)

            # Both runners attach the live step's conditioning after replay.
            # Capturing an OmniOutput here would retain the dummy prefill lists
            # and re-emit them during decode, replacing the real reference.
            return hidden_states
        elif self.model_stage == "cosyvoice3_code2wav":
            # Lazily swap the flow-decoder estimator to a TensorRT engine on the
            # first code2wav step (after weights are loaded), gated by the same
            # COSYVOICE3_TRT env toggle as the talker speaker embedding.
            self._maybe_enable_code2wav_trt()

            runtime_info = kwargs.get("model_intermediate_buffer")
            if runtime_info is None:
                runtime_info = kwargs.get("runtime_additional_information", [])
            if "runtime_additional_information" in kwargs and "model_intermediate_buffer" not in kwargs:
                logger.warning_once("runtime_additional_information is deprecated, use model_intermediate_buffer")

            seq_token_counts = kwargs.get("seq_token_counts")
            flat_ids = input_ids.reshape(-1).to(dtype=torch.long)
            request_ids_list = self._split_request_ids(flat_ids, seq_token_counts)

            num_reqs = max(1, len(request_ids_list))
            debug_batch_flow = cosyvoice3_batch_flow_debug()
            if debug_batch_flow:
                logger.info(
                    "CosyVoice3 code2wav debug: forward num_reqs=%d seq_token_counts=%s flat_ids=%d",
                    num_reqs,
                    seq_token_counts,
                    int(flat_ids.numel()),
                )
            sample_rate = torch.tensor(int(self.config.sample_rate), dtype=torch.int32)
            empty_audio = torch.zeros((0,), dtype=torch.float32, device=input_ids.device)
            audios: list[torch.Tensor] = [empty_audio] * num_reqs
            srs: list[torch.Tensor] = [sample_rate] * num_reqs
            if not isinstance(runtime_info, list):
                runtime_info = []
            streaming_flow_items: list[dict[str, object]] = []

            for idx, req_ids in enumerate(request_ids_list):
                raw = runtime_info[idx] if idx < len(runtime_info) and isinstance(runtime_info[idx], dict) else {}
                payload = to_struct(_normalize_request_conditioning(raw))
                meta = payload.meta
                embed = payload.embed

                req_id = meta.req_id[0] if (meta and meta.req_id) else None
                stream_finished = (
                    bool(meta.stream_finished.item()) if (meta and meta.stream_finished is not None) else False
                )
                speech_token = embed.speech_token if embed else None
                speech_feat = embed.speech_feat if embed else None
                embedding = embed.embedding if embed else None
                # Drop any right-padding carried from batched talker emission.
                if speech_token is not None and speech_feat is not None:
                    speech_token, speech_feat = unpad_prompt_conditioning(
                        speech_token, speech_feat, embed.speech_token_len if embed else None
                    )
                if speech_token is None or speech_feat is None or embedding is None:
                    if stream_finished and req_id is not None and hasattr(self, "_stream_vocoder_cache_by_req"):
                        with self._stream_audio_cache_lock:
                            self._stream_vocoder_cache_by_req.pop(req_id, None)
                    audios[idx] = self._stitch_stream_audio(req_id, empty_audio, stream_finished)
                    if req_ids.numel() > 0 and (
                        (meta and meta.left_context_size is not None) or payload.generated_len is not None
                    ):
                        info_keys = ",".join(
                            sorted(f for f in payload.__struct_fields__ if getattr(payload, f) is not None)
                        )
                        logger.warning_once(
                            "CosyVoice3 code2wav missing prompt conditioning for non-empty codec tokens: "
                            "raw_len=%d info_keys=%s",
                            int(req_ids.numel()),
                            info_keys,
                        )
                    continue

                token = self._sanitize_codec_tokens(req_ids)
                if token.numel() == 0:
                    audios[idx] = self._stitch_stream_audio(req_id, empty_audio, stream_finished)
                    if req_ids.numel() > 0:
                        logger.warning_once(
                            "CosyVoice3 code2wav received no valid codec tokens after filtering: "
                            "raw_len=%d raw_range=[%d,%d] vocab_size=%d",
                            req_ids.numel(),
                            int(req_ids.min().item()),
                            int(req_ids.max().item()),
                            int(self.code2wav.input_embedding.num_embeddings),
                        )
                    continue

                # `generated_len` is injected for many models by the generic
                # runner, so only explicit chunk-routing fields should switch
                # code2wav into the streaming path.
                uses_streaming_decode = meta and (
                    meta.stream_finished is not None or meta.left_context_size is not None
                )
                if uses_streaming_decode:
                    token_offset = max(0, meta.left_context_size or 0)

                    cache_state = None
                    if req_id is not None and hasattr(self, "_stream_vocoder_cache_by_req"):
                        with self._stream_audio_cache_lock:
                            cache_state = self._stream_vocoder_cache_by_req.get(req_id)

                    streaming_flow_items.append(
                        {
                            "index": idx,
                            "req_id": req_id,
                            "stream_finished": stream_finished,
                            "token": token.unsqueeze(0),
                            "prompt_token": speech_token[:1],
                            "prompt_feat": speech_feat[:1],
                            "embedding": embedding[:1],
                            "cache_state": cache_state,
                            "token_offset_tokens": token_offset,
                            "finalize": stream_finished,
                        }
                    )
                    continue
                else:
                    token_offset = max(0, meta.talker_prefill_offset or 0) if meta else 0
                    tts_speech = self.code2wav.forward(
                        token=token.unsqueeze(0),
                        prompt_token=speech_token[:1],
                        prompt_feat=speech_feat[:1],
                        embedding=embedding[:1],
                        n_timesteps=10,
                        token_offset_tokens=token_offset,
                    )

                audio = tts_speech.reshape(-1).to(dtype=torch.float32)

                audios[idx] = self._stitch_stream_audio(req_id, audio, stream_finished)

            if streaming_flow_items:
                if debug_batch_flow:
                    item_shapes = [
                        (
                            tuple(item["token"].shape),  # type: ignore[union-attr]
                            tuple(item["prompt_token"].shape),  # type: ignore[union-attr]
                            tuple(item["prompt_feat"].shape),  # type: ignore[union-attr]
                            bool(item.get("finalize", False)),
                        )
                        for item in streaming_flow_items
                    ]
                    logger.info(
                        "CosyVoice3 code2wav debug: streaming_items=%d item_shapes=%s",
                        len(streaming_flow_items),
                        item_shapes,
                    )
                if (
                    (len(streaming_flow_items) > 1 or cosyvoice3_packed_inference_enabled())
                    and cosyvoice3_batch_flow_enabled()
                    and hasattr(self.code2wav, "forward_streaming_batch")
                ):
                    streaming_results = self.code2wav.forward_streaming_batch(
                        streaming_flow_items,
                        n_timesteps=10,
                    )
                else:
                    streaming_results = [
                        self.code2wav.forward_streaming(
                            token=item["token"],  # type: ignore[arg-type]
                            prompt_token=item["prompt_token"],  # type: ignore[arg-type]
                            prompt_feat=item["prompt_feat"],  # type: ignore[arg-type]
                            embedding=item["embedding"],  # type: ignore[arg-type]
                            cache_state=item.get("cache_state"),  # type: ignore[arg-type]
                            n_timesteps=10,
                            token_offset_tokens=int(item.get("token_offset_tokens", 0)),
                            finalize=bool(item.get("finalize", False)),
                        )
                        for item in streaming_flow_items
                    ]

                for item, (tts_speech, new_cache_state) in zip(streaming_flow_items, streaming_results):
                    idx = int(item["index"])
                    req_id = item.get("req_id")
                    stream_finished = bool(item.get("stream_finished", False))
                    if req_id is not None and hasattr(self, "_stream_vocoder_cache_by_req"):
                        with self._stream_audio_cache_lock:
                            if new_cache_state is None or stream_finished:
                                self._stream_vocoder_cache_by_req.pop(req_id, None)
                            else:
                                self._stream_vocoder_cache_by_req[req_id] = new_cache_state

                    audio = tts_speech.reshape(-1).to(dtype=torch.float32)
                    audios[idx] = self._stitch_stream_audio(req_id, audio, stream_finished)

            return OmniOutput(text_hidden_states=None, multimodal_outputs={"audio": audios, "sr": srs})
        else:
            raise ValueError(f"Unsupported model_stage: {self.model_stage}")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str] | None:
        if self.model_stage == "cosyvoice3_talker":
            # Consume the provided iterator when it yields tensors; fall back
            # to reading llm.pt from the model dir when it is empty (e.g. the
            # dummy load format). Previously the iterator was silently
            # discarded and the file was re-read on every call, so runtime
            # weight updates (RL training loops, in-place reloads) were
            # reverted to the on-disk checkpoint without any error.
            weights = list(weights)
            # The first load must cover every talker-specific parameter, as
            # the strict load_state_dict path did; later calls may carry any
            # subset. The flag is set only once a load has succeeded.
            first_load = not getattr(self, "_talker_weights_loaded", False)
            if weights:
                self._load_talker_weights(weights, require_complete=first_load)
            elif first_load:
                llm_weight_path = os.path.join(self.model_dir, "llm.pt")
                device = next(self.parameters()).device
                checkpoint = torch.load(llm_weight_path, map_location=device)
                self._load_talker_weights(checkpoint.items(), require_complete=True)
                self.model.to(device)
            self._talker_weights_loaded = True
            self.model.eval()
        elif self.model_stage == "cosyvoice3_code2wav":
            # Load weights for code2wav stage (flow + hift)
            device = next(self.parameters()).device
            self.code2wav.load_weights(self.model_dir, device)
        else:
            raise ValueError(f"{self.model_stage} not supported yet!")
        # None keeps the loader's strict loaded-weights tracking off: the
        # tracker compares module-path parameter names, while this model
        # consumes checkpoint-schema names (and deliberately leaves the
        # unused text lm_head uninitialized on the talker stage).
        return None

    def _load_talker_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        require_complete: bool = False,
    ) -> None:
        """Load checkpoint-schema tensors into the live talker modules.

        Names follow the native ``llm.pt`` layout: ``llm.model.model.*`` for
        the transformer (loaded via vLLM's Qwen2 loader, which handles the
        qkv_proj/gate_up_proj stacked params), ``speech_embedding.*`` and
        ``llm_decoder.*`` for the talker-specific modules, and
        ``llm.model.lm_head.*`` (the text head), which the talker never uses
        and is skipped. Accepts the full checkpoint or any subset, e.g.
        buckets streamed by a weight-update loop, and loads in place so
        CUDA graphs stay valid.

        With ``require_complete`` every ``speech_embedding.*`` and
        ``llm_decoder.*`` parameter must be present. Names and shapes are
        validated before anything is copied, so a rejected call leaves the
        talker-specific parameters untouched.
        """
        talker_params = {
            f"{module_name}.{attr}": param
            for module_name in ("speech_embedding", "llm_decoder")
            for attr, param in getattr(self.model, module_name).named_parameters()
        }
        qwen_weights: list[tuple[str, torch.Tensor]] = []
        talker_updates: dict[str, torch.Tensor] = {}
        for name, tensor in weights:
            if name.startswith("llm.model.model."):
                qwen_weights.append((name[len("llm.model.model.") :], tensor))
            elif name in talker_params:
                talker_updates[name] = tensor
            elif name.startswith("llm.model.lm_head."):
                continue
            else:
                raise ValueError(f"unexpected CosyVoice3 talker checkpoint tensor {name!r}")

        if require_complete:
            missing = sorted(talker_params.keys() - talker_updates.keys())
            if missing:
                raise ValueError(f"CosyVoice3 talker checkpoint is missing required tensors: {missing}")
        # copy_ broadcasts, so compare shapes explicitly before copying.
        mismatched = [
            f"{name}: checkpoint {tuple(tensor.shape)} vs parameter {tuple(talker_params[name].shape)}"
            for name, tensor in talker_updates.items()
            if tensor.shape != talker_params[name].shape
        ]
        if mismatched:
            raise ValueError(f"CosyVoice3 talker checkpoint tensor shape mismatch: {'; '.join(mismatched)}")

        for name, tensor in talker_updates.items():
            param = talker_params[name]
            param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))
        if qwen_weights:
            self.model.llm.model.load_weights(iter(qwen_weights))
