# Stage 1: CAT codec decoder for MOSS-TTS-Local (convert a sequence of 32-layer RVQ codes to a waveform)
# 
# The CAT codec (Causal Audio Tokenizer) is MOSS's 1.6B Transformer-based
# audio tokenizer from https://github.com/OpenMOSS/MOSS-Audio-Tokenizer.

import logging
import os
import time
from collections.abc import Iterable
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
from torch.profiler import record_function
from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models import SupportsPP
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.models.moss_tts._local_stage1_timing import get_timer
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = logging.getLogger(__name__)
_TIMER = get_timer()


def _decoder_debug_enabled() -> bool:
    return os.environ.get("MOSS_TTS_DECODER_DEBUG", "0") == "1"

# =======================================================================================
#  CAT Codec Worker
# =======================================================================================

class CATCodecWorker:
    _DEFAULT_CHUNK_DURATION = 8.0

    def __init__(self, device_str: str, codec_path: str):
        self.device = torch.device(device_str)
        if os.path.exists(codec_path):
            codec_path = os.path.realpath(codec_path)
        logger.info("[MossTTS Decoder] Loading CAT codec from %s on %s", codec_path, device_str)

        self.codec = self._load_codec_from_local_repo(codec_path)
        self.codec = self.codec.to(self.device).eval().float()

        self.sample_rate: int = getattr(self.codec.config, "sampling_rate", 24_000)
        self.n_vq: int = getattr(self.codec.config, "num_quantizers", 32)
        self.downsample_rate: int = getattr(self.codec.config, "downsample_rate", 1_920)
        self.chunk_frame_length = self._get_chunk_frame_length()

        logger.info(
            "[MossTTS Decoder] CAT codec loaded: sample_rate=%d, n_vq=%d, "
            "chunk_frame_length=%d",
            self.sample_rate,
            self.n_vq,
            self.chunk_frame_length,
        )

    def _load_codec_from_local_repo(self, codec_path: str) -> nn.Module:
        if not os.path.isdir(codec_path):
            raise ValueError(
                "MOSS_AUDIO_TOKENIZER_PATH must point to a local "
                "MOSS-Audio-Tokenizer snapshot for the vLLM-Omni native "
                f"loader. Got: {codec_path!r}"
            )

        from safetensors.torch import load_file

        from vllm_omni.model_executor.models.moss_tts.configuration_moss_audio_tokenizer import (
            MossAudioTokenizerConfig,
        )
        from vllm_omni.model_executor.models.moss_tts.modeling_moss_audio_tokenizer import (
            MossAudioTokenizerModel,
        )

        root = Path(codec_path)
        config = MossAudioTokenizerConfig.from_pretrained(str(root))
        codec = MossAudioTokenizerModel(config)

        shard_paths = sorted(root.glob("model-*.safetensors"))
        if not shard_paths:
            single_path = root / "model.safetensors"
            if single_path.exists():
                shard_paths = [single_path]
        if not shard_paths:
            raise FileNotFoundError(
                f"No safetensors checkpoint shards found under {root}."
            )

        state_dict: dict[str, torch.Tensor] = {}
        for shard_path in shard_paths:
            state_dict.update(load_file(str(shard_path), device="cpu"))
        missing, unexpected = codec.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(
                "[MossTTS Decoder] Native codec loader missing %d keys; first keys: %s",
                len(missing),
                list(missing)[:10],
            )
        if unexpected:
            logger.warning(
                "[MossTTS Decoder] Native codec loader found %d unexpected keys; first keys: %s",
                len(unexpected),
                list(unexpected)[:10],
            )
        return codec

    def _get_chunk_frame_length(self) -> int:
        chunk_duration = min(
            self._DEFAULT_CHUNK_DURATION,
            float(getattr(self.codec.config, "causal_transformer_context_duration", 10.0)),
        )
        chunk_length = int(round(chunk_duration * self.sample_rate))
        if chunk_length <= 0:
            return 0
        return max(1, chunk_length // self.downsample_rate)

    def _streaming_context(self, batch_size: int) -> ExitStack:
        stack = ExitStack()
        for decoder_module in self.codec.decoder:
            if hasattr(decoder_module, "streaming") and callable(decoder_module.streaming):
                stack.enter_context(decoder_module.streaming(batch_size=batch_size))
        return stack

    def _decode_frame_to_cpu(
        self,
        codes: torch.Tensor,
        code_lengths: torch.Tensor,
    ) -> list[torch.Tensor]:
        result = self.codec._decode_frame(codes, code_lengths)
        if result.audio is None or result.audio_lengths is None:
            raise RuntimeError("Internal error: `_decode_frame` returned empty audio.")

        wav_batch = result.audio[:, 0].float().cpu()
        audio_lengths = result.audio_lengths.cpu()
        outputs: list[torch.Tensor] = []
        for idx in range(wav_batch.shape[0]):
            wav_len = (
                int(audio_lengths[idx].item())
                if audio_lengths.numel() > idx
                else int(wav_batch.shape[-1])
            )
            outputs.append(wav_batch[idx, :wav_len])
        return outputs

    @torch.inference_mode()
    def decode_batch(
        self,
        padded_codes: torch.Tensor,
        code_lengths: torch.Tensor,
    ) -> list[torch.Tensor]:
        padded_codes = padded_codes.to(self.device, dtype=torch.long, non_blocking=True)
        code_lengths = code_lengths.to(self.device, dtype=torch.long, non_blocking=True)
        return self._decode_frame_to_cpu(padded_codes, code_lengths)

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        codes = codes.to(self.device, dtype=torch.long, non_blocking=True)
        if codes.dim() == 2:
            codes = codes.unsqueeze(1)
        if codes.dim() != 3:
            raise ValueError(
                "Expected CAT codec codes with shape [nq, T] or [nq, B, T], "
                f"got {tuple(codes.shape)}."
            )
        if codes.shape[1] != 1:
            raise ValueError(
                "CATCodecWorker.decode() handles one request. Use decode_batch() "
                f"for batch_size={codes.shape[1]}."
            )

        code_length = int(codes.shape[-1])
        if code_length == 0:
            return torch.zeros(0, dtype=torch.float32)

        chunk_frame_length = self.chunk_frame_length
        if chunk_frame_length <= 0 or code_length <= chunk_frame_length:
            lengths = torch.tensor([code_length], device=self.device, dtype=torch.long)
            return self._decode_frame_to_cpu(codes, lengths)[0]

        wav_chunks: list[torch.Tensor] = []
        with self._streaming_context(batch_size=1):
            for start_idx in range(0, code_length, chunk_frame_length):
                code_length_i = min(chunk_frame_length, code_length - start_idx)
                if code_length_i <= 0:
                    break
                lengths_i = torch.tensor([code_length_i], device=self.device, dtype=torch.long)
                codes_i = codes[:, :, start_idx: start_idx + code_length_i]
                wav_chunks.append(self._decode_frame_to_cpu(codes_i, lengths_i)[0])
        if not wav_chunks:
            return torch.zeros(0, dtype=torch.float32)
        return torch.cat(wav_chunks, dim=0)


# Module-level cache: (device_type, codec_path) → CATCodecWorker
_CODEC_WORKER_CACHE: dict[tuple[str, str], CATCodecWorker] = {}

def _get_codec_worker(device: torch.device, codec_path: str) -> CATCodecWorker:
    key = (device.type, os.path.realpath(codec_path))
    if key not in _CODEC_WORKER_CACHE:
        _CODEC_WORKER_CACHE[key] = CATCodecWorker(device.type, codec_path)
    return _CODEC_WORKER_CACHE[key]


# =======================================================================================
#  Flat-code parsing helpers (Reshape a 1D flat code tensor into [n_vq, T])
# =======================================================================================

def _parse_flat_codes(
    flat_codes: torch.Tensor,
    n_vq: int,
) -> Optional[torch.Tensor]:
    flat_codes = flat_codes.reshape(-1).to(torch.long)
    total = flat_codes.numel()
    if total == 0 or total % n_vq != 0:
        return None
    T = total // n_vq
    return flat_codes.reshape(T, n_vq).transpose(0, 1).contiguous() 

# split a flat code tensor into per-request slices
def _split_per_request(
    ids: torch.Tensor,
    runtime_info: Optional[list[dict[str, Any]]],
    seq_token_counts: Optional[list[int]],
) -> list[torch.Tensor]:
    n = ids.numel()
    if n == 0:
        return [ids]

    if runtime_info and all(
        isinstance(info.get("code_flat_numel"), int) and info["code_flat_numel"] > 0
        for info in runtime_info
    ):
        sizes = [int(info["code_flat_numel"]) for info in runtime_info]
        if sum(sizes) == n:
            parts, offset = [], 0
            for sz in sizes:
                parts.append(ids[offset: offset + sz])
                offset += sz
            return parts

    if seq_token_counts and len(seq_token_counts) > 1:
        boundaries = [0]
        for c in seq_token_counts:
            boundaries.append(boundaries[-1] + c)
        return [ids[boundaries[i]: min(boundaries[i + 1], n)] for i in range(len(seq_token_counts))]
    return [ids]


# =======================================================================================
#  Stage 1 - Decoder Model
# =======================================================================================
class MossTTSDecoderModel(nn.Module, SupportsPP):
    have_multimodal_outputs    = True
    enable_update_additional_information = True

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        cfg = vllm_config.model_config.hf_config
        self.config = cfg

        self.n_vq: int = cfg.n_vq         
        self.sample_rate: int = getattr(cfg, "sampling_rate", 24_000)

        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

        codec_path = (
            getattr(vllm_config.model_config, "audio_tokenizer_path", None)
            or os.environ.get("MOSS_AUDIO_TOKENIZER_PATH")
            or "OpenMOSS-Team/MOSS-Audio-Tokenizer" 
        )

        self._codec: CATCodecWorker = _get_codec_worker(self.device, codec_path)

        # dummy logits processor and sampler required by vllm's model protocol
        self.logits_processor = LogitsProcessor(cfg.language_config.vocab_size)
        self.sampler = Sampler()

        # Per-request streaming state: request_id → ExitStack holding codec KV-cache context
        # Entries are created on the first chunk for a request and closed when is_finished=True
        self._streaming_states: dict[str, ExitStack] = {}
        # Batched streaming state for multi-request async decode
        self._batched_streaming_stack: ExitStack | None = None
        self._batched_streaming_request_ids: list[str] | None = None
        # Async correctness path: accumulate per-request codec codes and
        # statelessly decode the full prefix each chunk, then emit audio delta.
        self._request_code_buffers: dict[str, list[torch.Tensor]] = {}
        self._request_audio_offsets: dict[str, int] = {}

        # workaround for handling request boundaries when the connector doesn't propagate request_id / finished flags
        self._active_key: Optional[str] = None
        self._last_gen_len: Optional[int] = None
        self._run_counter: int = 0

        # a fallback to compute TTFA metric: "first.first_chunk.ts" will be written for the first request that produces audio
        self._first_chunk_dir: Optional[str] = os.environ.get("MOSS_FIRST_CHUNK_DIR")
        self._first_chunk_seen: set[str] = set()
        if self._first_chunk_dir:
            try:
                os.makedirs(self._first_chunk_dir, exist_ok=True)
            except OSError:
                self._first_chunk_dir = None

    def _record_first_chunk(self, request_id: Optional[str]) -> None:
        now = time.time()
        logger.info(
            "[MossTTS Decoder][TIMING] non-empty wav produced at wall=%.3f "
            "request_id=%s",
            now, request_id,
        )
        if not self._first_chunk_dir:
            return
        
        keys = [k for k in (request_id, "first") if k]
        for key in keys:
            if key in self._first_chunk_seen:
                continue
            self._first_chunk_seen.add(key)
            path = os.path.join(self._first_chunk_dir, f"{key}.first_chunk.ts")
            try:
                with open(path, "w") as f:
                    f.write(f"{now:.6f}\n")
            except OSError as exc:
                logger.debug("[MossTTS Decoder] Could not write %s: %s", path, exc)

    def _enter_streaming(self, request_id: str) -> None:
        if request_id in self._streaming_states:
            return
        stack = ExitStack()
        codec = self._codec.codec
        for decoder_module in codec.decoder:
            if hasattr(decoder_module, "streaming") and callable(decoder_module.streaming):
                stack.enter_context(decoder_module.streaming(batch_size=1))
        self._streaming_states[request_id] = stack

    def _exit_streaming(self, request_id: str) -> None:
        stack = self._streaming_states.pop(request_id, None)
        if stack is not None:
            stack.close()

    # close all active streaming contexts when switching modes
    def _reset_streaming_topology(self) -> None:
        for request_id in list(self._streaming_states.keys()):
            self._exit_streaming(request_id)
        self._exit_batched_streaming()

    # acquire a shared batched streaming context for a set of requests that are active at the same time
    def _enter_batched_streaming(self, request_ids: list[str]) -> None:
        if self._batched_streaming_stack is not None:
            if self._batched_streaming_request_ids != request_ids:
                if _decoder_debug_enabled():
                    logger.warning(
                        "[MossTTS Decoder] Batched streaming request set changed "
                        "from %s to %s; resetting codec streaming state.",
                        self._batched_streaming_request_ids,
                        request_ids,
                    )
                self._exit_batched_streaming()
            else:
                return

        if self._streaming_states:
            if _decoder_debug_enabled():
                logger.warning(
                    "[MossTTS Decoder] Switching from per-request streaming to "
                    "batched streaming; resetting %d active single-request states.",
                    len(self._streaming_states),
                )
            self._reset_streaming_topology()

        stack = ExitStack()
        codec = self._codec.codec
        batch_size = len(request_ids)
        for decoder_module in codec.decoder:
            if hasattr(decoder_module, "streaming") and callable(decoder_module.streaming):
                stack.enter_context(decoder_module.streaming(batch_size=batch_size))
        self._batched_streaming_stack = stack
        self._batched_streaming_request_ids = list(request_ids)

    def _exit_batched_streaming(self) -> None:
        if self._batched_streaming_stack is not None:
            self._batched_streaming_stack.close()
        self._batched_streaming_stack = None
        self._batched_streaming_request_ids = None

    def _decode_one_request(
        self,
        flat_codes: torch.Tensor,
        request_id: Optional[str] = None,
        is_finished: bool = False,
    ) -> torch.Tensor:
        empty = torch.zeros(0, dtype=torch.float32)

        with record_function("stage1/decode_one_request"), _TIMER.gpu("stage1/decode_one_request"):
            if flat_codes is None or flat_codes.numel() == 0:
                if request_id and is_finished:
                    self._exit_streaming(request_id)
                    self._request_code_buffers.pop(request_id, None)
                    self._request_audio_offsets.pop(request_id, None)
                return empty

            codes = _parse_flat_codes(flat_codes, self.n_vq) 
            if codes is None:
                if request_id and is_finished:
                    self._exit_streaming(request_id)
                    self._request_code_buffers.pop(request_id, None)
                    self._request_audio_offsets.pop(request_id, None)
                return empty

            try:
                if request_id is not None:
                    # The CAT codec's streaming context is module-global and
                    # cannot hold independent interleaved request states. For
                    # async chunks, decode the full per-request code prefix
                    # statelessly and emit only the new audio suffix.
                    code_buffers = self._request_code_buffers.setdefault(request_id, [])
                    code_buffers.append(flat_codes.reshape(-1).detach().cpu().to(torch.long))
                    all_codes = torch.cat(code_buffers, dim=0)
                    full_codes = _parse_flat_codes(all_codes, self.n_vq)
                    if full_codes is None:
                        wav = empty
                    else:
                        full_wav = self._codec.decode(full_codes)
                        offset = self._request_audio_offsets.get(request_id, 0)
                        if full_wav.numel() > offset:
                            wav = full_wav[offset:]
                            self._request_audio_offsets[request_id] = int(full_wav.numel())
                        else:
                            wav = empty
                    if is_finished:
                        self._request_code_buffers.pop(request_id, None)
                        self._request_audio_offsets.pop(request_id, None)
                else:
                    # fallback to stateless single-call decoder
                    wav = self._codec.decode(codes)

                if wav is not None and wav.numel() > 0:
                    self._record_first_chunk(request_id)
                return wav
            except Exception as exc:
                logger.error("[MossTTS Decoder] Codec decode failed: %s", exc, exc_info=True)
                if request_id and is_finished:
                    self._exit_streaming(request_id)
                return empty

    @torch.inference_mode()
    def _batch_decode(
        self,
        request_codes_list: list[torch.Tensor],
        request_ids: Optional[list[Optional[str]]] = None,
        finished_flags: Optional[list[bool]] = None,
    ) -> list[torch.Tensor]:
        empty = torch.zeros(0, dtype=torch.float32)
        if _decoder_debug_enabled():
            logger.info(
                "[MossTTS Decoder][DEBUG] _batch_decode num_req=%d request_ids=%s "
                "finished=%s batched_streaming_active=%s single_streams=%d",
                len(request_codes_list),
                request_ids,
                finished_flags,
                self._batched_streaming_stack is not None,
                len(self._streaming_states),
            )

        if (
            request_ids
            and len(request_ids) > 1
            and all(isinstance(rid, str) and rid for rid in request_ids)
        ):
            if _decoder_debug_enabled():
                logger.info(
                    "[MossTTS Decoder][DEBUG] taking batched streaming path for "
                    "request_ids=%s",
                    request_ids,
                )
            return self._batch_decode_streaming(
                request_codes_list,
                [rid for rid in request_ids if isinstance(rid, str)],
                finished_flags or [False] * len(request_codes_list),
            )

        if self._batched_streaming_stack is not None:
            if _decoder_debug_enabled():
                logger.warning(
                    "[MossTTS Decoder] Falling back to non-batched decode for %d "
                    "request(s); resetting active batched streaming state.",
                    len(request_codes_list),
                )
            self._reset_streaming_topology()

        results: list[torch.Tensor] = []
        for i, req_codes in enumerate(request_codes_list):
            req_id = request_ids[i] if request_ids else None
            finished = finished_flags[i] if finished_flags else False
            if _decoder_debug_enabled():
                logger.info(
                    "[MossTTS Decoder][DEBUG] taking per-request path idx=%d "
                    "request_id=%s finished=%s numel=%d",
                    i,
                    req_id,
                    finished,
                    int(req_codes.numel()) if req_codes is not None else -1,
                )
            wav = self._decode_one_request(req_codes, request_id=req_id, is_finished=finished)
            results.append(wav if wav.numel() > 0 else empty)
        return results

    # decode multiple active requests through a single batched codec stream
    def _batch_decode_streaming(
        self,
        request_codes_list: list[torch.Tensor],
        request_ids: list[str],
        finished_flags: list[bool],
    ) -> list[torch.Tensor]:
        empty = torch.zeros(0, dtype=torch.float32)

        if len(request_codes_list) != len(request_ids):
            raise ValueError(
                "request_codes_list and request_ids must have the same length "
                f"(got {len(request_codes_list)} vs {len(request_ids)})."
            )

        parsed_codes: list[torch.Tensor] = []
        lengths: list[int] = []
        for flat_codes in request_codes_list:
            if flat_codes is None or flat_codes.numel() == 0:
                parsed = torch.zeros(self.n_vq, 0, dtype=torch.long)
            else:
                parsed = _parse_flat_codes(flat_codes, self.n_vq)
                if parsed is None:
                    parsed = torch.zeros(self.n_vq, 0, dtype=torch.long)
            parsed_codes.append(parsed)
            lengths.append(int(parsed.shape[-1]))

        if _decoder_debug_enabled():
            logger.info(
                "[MossTTS Decoder][DEBUG] batched stream request_ids=%s lengths=%s "
                "finished=%s",
                request_ids,
                lengths,
                finished_flags,
            )

        if max(lengths, default=0) == 0:
            if all(finished_flags):
                self._exit_batched_streaming()
            return [empty for _ in request_codes_list]

        self._enter_batched_streaming(request_ids)

        max_len = max(lengths)
        batch_size = len(request_ids)
        padded = torch.zeros(
            self.n_vq,
            batch_size,
            max_len,
            dtype=torch.long,
            device=self.device,
        )
        for idx, (codes, code_len) in enumerate(zip(parsed_codes, lengths)):
            if code_len > 0:
                padded[:, idx, :code_len] = codes.to(self.device)

        lengths_t = torch.tensor(lengths, device=self.device, dtype=torch.long)

        try:
            results: list[torch.Tensor] = []
            wav_batch = self._codec.decode_batch(padded, lengths_t)
            for wav, req_id in zip(wav_batch, request_ids):
                if wav.numel() > 0:
                    self._record_first_chunk(req_id)
                results.append(wav if wav.numel() > 0 else empty)
        except Exception:
            if all(finished_flags):
                self._exit_batched_streaming()
            raise

        if all(finished_flags):
            self._exit_batched_streaming()

        return results

    def on_requests_finished(self, request_ids) -> None:
        finished = {str(request_id) for request_id in request_ids}
        for request_id in finished:
            self._exit_streaming(request_id)
            self._request_code_buffers.pop(request_id, None)
            self._request_audio_offsets.pop(request_id, None)
        if (
            self._batched_streaming_request_ids is not None
            and any(request_id in finished for request_id in self._batched_streaming_request_ids)
        ):
            self._exit_batched_streaming()
        if self._active_key in finished:
            self._active_key = None
            self._last_gen_len = None

    def forward(
        self,
        input_ids: Optional[torch.Tensor]              = None,
        codes: Optional[torch.Tensor]                  = None,
        runtime_additional_information: Optional[list[dict[str, Any]]] = None,
        **kwargs,
    ) -> OmniOutput:
        with record_function("stage1/forward"), _TIMER.cpu("stage1/forward_total"):
            if runtime_additional_information is None:
                runtime_additional_information = (
                    kwargs.get("model_intermediate_buffer")
                    or kwargs.get("runtime_additional_information")
                )

            code_tensor = codes if codes is not None else input_ids
            empty = torch.zeros(0, dtype=torch.float32)

            with record_function("stage1/extract_codes"), _TIMER.cpu("stage1/extract_codes"):
                # resolve missing code tensor from runtime_additional_information if not present in inputs
                if (code_tensor is None or code_tensor.numel() == 0) and runtime_additional_information:
                    code_parts: list[torch.Tensor] = []
                    for info in runtime_additional_information:
                        if not isinstance(info, dict):
                            continue
                        cp = info.get("code_predictor_codes")
                        if cp is None:
                            continue
                        if not isinstance(cp, torch.Tensor):
                            cp = torch.tensor(cp, dtype=torch.long)
                        code_parts.append(cp.reshape(-1).to(torch.long))
                    if code_parts:
                        code_tensor = torch.cat(code_parts, dim=0)

            if code_tensor is None or code_tensor.numel() == 0:
                return OmniOutput(
                    text_hidden_states=None,
                    multimodal_outputs={"model_outputs": [empty]},
                )

            # Skip decode during CUDA graph capture
            if torch.cuda.is_current_stream_capturing():
                n = len(runtime_additional_information) if runtime_additional_information else 1
                return OmniOutput(
                    text_hidden_states=None,
                    multimodal_outputs={"model_outputs": [empty] * n},
                )

            with record_function("stage1/split_requests"), _TIMER.cpu("stage1/split_requests"):
                ids = code_tensor.reshape(-1).to(torch.long)

                # split flat tensor into per-request slices
                request_codes_list = _split_per_request(
                    ids,
                    runtime_additional_information,
                    kwargs.get("seq_token_counts"),
                )

            # extract per-request streaming metadata from runtime_additional_information
            request_ids: Optional[list[Optional[str]]] = None
            finished_flags: Optional[list[bool]] = None
            with record_function("stage1/metadata_prep"), _TIMER.cpu("stage1/metadata_prep"):
                if runtime_additional_information:
                    request_ids = []
                    finished_flags = []
                    for info in runtime_additional_information:
                        if not isinstance(info, dict):
                            request_ids.append(None)
                            finished_flags.append(False)
                            continue

                        rid = info.get("request_id")
                        if rid is None:
                            gen_len = info.get("generated_len")
                            if (
                                self._active_key is None
                                or (
                                    isinstance(gen_len, int)
                                    and isinstance(self._last_gen_len, int)
                                    and gen_len < self._last_gen_len
                                )
                            ):
                                if self._active_key is not None:
                                    self._exit_streaming(self._active_key)
                                self._run_counter += 1
                                self._active_key = f"run_{self._run_counter}"
                            rid = self._active_key
                            if isinstance(gen_len, int):
                                self._last_gen_len = gen_len

                        request_ids.append(rid)

                        fin = info.get("finished")
                        if isinstance(fin, torch.Tensor):
                            fin = bool(fin.item())
                        finished_flags.append(bool(fin) if fin is not None else False)
                    if _decoder_debug_enabled() and not getattr(self, "_dumped_info_keys", False):
                        self._dumped_info_keys = True
                        for idx, info in enumerate(runtime_additional_information):
                            if isinstance(info, dict):
                                logger.info(
                                    "[MossTTS Decoder][DEBUG] info[%d] keys=%s types=%s",
                                    idx, list(info.keys()),
                                    {k: type(v).__name__ for k, v in info.items()},
                                )
                            else:
                                logger.info(
                                    "[MossTTS Decoder][DEBUG] info[%d] type=%s (not a dict)",
                                    idx, type(info).__name__,
                                )

            with record_function("stage1/decode"), _TIMER.gpu("stage1/decode_total"):
                audios = self._batch_decode(request_codes_list, request_ids, finished_flags)

            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": audios},
            )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal: bool = False,
    ) -> torch.Tensor:
        # since the decoder has no meaningful text embeddings, a zero tensor is returned to satisfy the pipeline interface
        hidden_size = self.vllm_config.model_config.get_hidden_size()
        return torch.zeros(
            input_ids.shape[0], hidden_size,
            dtype=torch.bfloat16,
            device=self.device,
        )

    def compute_logits(
        self, hidden_states: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if hidden_states is None or hidden_states.numel() == 0:
            return None
        vocab_size = self.config.language_config.vocab_size
        return torch.zeros(
            hidden_states.shape[0], vocab_size,
            dtype=hidden_states.dtype,
            device=self.device,
        )

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        if logits is None or logits.numel() == 0:
            return None
        return self.sampler(logits, sampling_metadata)

    def make_omni_output(self, model_output: Any, **kwargs) -> OmniOutput:
        if isinstance(model_output, OmniOutput):
            return model_output
        empty = torch.zeros(0, dtype=torch.float32)
        if model_output is None:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty]},
            )
        if isinstance(model_output, torch.Tensor):
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": [model_output.float().reshape(-1)]
                },
            )
        raise TypeError(f"Unexpected model output type: {type(model_output)}")

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        **kwargs,
    ) -> set[str]:
        # Added to support the vLLM model protocol. The CAT codec is loaded
        # separately by the native vLLM-Omni codec loader in __init__.
        logger.info(
            "[MossTTS Decoder] load_weights() called — CAT codec already "
            "loaded in __init__; nothing to do here."
        )
        return set()

    def _clear_warmup_state(self) -> None:
        _TIMER.reset()
