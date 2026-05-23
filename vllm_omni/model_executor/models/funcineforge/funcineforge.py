# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FunCineForge model for vLLM-Omni.

Three-stage dubbing model: LM (Qwen2 AR with face/timespk embeddings)
→ Flow Matching (DiT CFM) → Causal HiFiGAN vocoder.

Within vLLM-Omni the model is split into two stages:
  - Stage 0 "funcineforge_talker":  LM autoregressive generation.
  - Stage 1 "funcineforge_code2wav": FM DiT + Causal HiFiGAN.

Weight layout (DeepSpeed):
  funcineforge_zh_en/
    llm/mp_rank_00_model_states.pt      (keys prefixed with ``module.``)
    flow/mp_rank_00_model_states.pt
    vocoder/mp_rank_00_model_states.pt
    camplus.onnx
"""

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from threading import Lock

import torch
import torch.nn as nn
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
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    ProcessorInputs,
    PromptIndexTargets,
    PromptInsertion,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_omni.data_entry_keys import EmbeddingsStruct, OmniPayloadStruct, to_dict, to_struct
from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import CosyVoice3Model as _CosyVoice3Sampling
from vllm_omni.model_executor.models.funcineforge.config import FunCineForgeConfig
from vllm_omni.model_executor.models.funcineforge.utils import load_deepspeed_checkpoint
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)

_FACE_TOKEN = 6565


# ---------------------------------------------------------------------------
# Multimodal processing
# ---------------------------------------------------------------------------


class FunCineForgeMultiModalProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(FunCineForgeConfig)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # code2wav receives codec tokens from stage connector, not multimodal input.
        model_stage = getattr(self.ctx.model_config, "model_stage", "")
        if model_stage == "funcineforge_code2wav":
            return {}
        return {"audio": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        model_stage = getattr(self.ctx.model_config, "model_stage", "")
        if model_stage == "funcineforge_code2wav":
            return {}
        return {"audio": 750}

    def get_data_parser(self):
        return MultiModalDataParser(
            target_sr=self.ctx.get_hf_config().sample_rate,
            expected_hidden_size=self._get_expected_hidden_size(),
        )


class FunCineForgeMultiModalProcessor(BaseMultiModalProcessor[FunCineForgeMultiModalProcessingInfo]):
    def _ensure_cached_runtime_components(self, model_dir: str, config: FunCineForgeConfig) -> None:
        cached_model_dir = getattr(self, "_cached_model_dir", None)
        if cached_model_dir == model_dir:
            return

        if not os.path.isdir(model_dir):
            from huggingface_hub import snapshot_download

            model_dir = snapshot_download(model_dir)

        import onnxruntime

        from vllm_omni.model_executor.models.funcineforge.tokenizer import get_funcineforge_tokenizer
        from vllm_omni.model_executor.models.funcineforge.utils import mel_spectrogram

        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1

        self.tokenizer = get_funcineforge_tokenizer(
            token_path=os.path.join(model_dir, config.llm["llm"]["pretrain_path"]),
        )
        self.speech_tokenizer = onnxruntime.InferenceSession(
            os.path.join(model_dir, config.speech_tokenizer_path),
            sess_options=option,
            providers=["CPUExecutionProvider"],
        )
        self.feat_extractor = mel_spectrogram
        self.campplus_session = onnxruntime.InferenceSession(
            os.path.join(model_dir, config.campplus_onnx),
            sess_options=option,
            providers=["CPUExecutionProvider"],
        )
        self._cached_model_dir = model_dir

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        config = self.info.ctx.get_hf_config()
        model_dir = self.info.ctx.model_config.model
        self._ensure_cached_runtime_components(model_dir, config)

        audio = mm_data.get("audio", None)
        if audio is None:
            audio = mm_data.get("audios")
            if audio is not None:
                audio = audio[0], config.sample_rate

        # Tokenize text prompt
        from vllm_omni.model_executor.models.funcineforge.utils import extract_text_token

        text_token, text_token_len = extract_text_token(prompt, self.tokenizer)

        if audio is None:
            return BatchFeature({"input_ids": text_token, "input_len": [text_token_len]})

        prompt_text = mm_kwargs.get("prompt_text")
        if not isinstance(prompt_text, str):
            raise ValueError(f"prompt_text is required for FunCineForge: got {prompt_text}")

        prompt_text_token, prompt_text_token_len = extract_text_token(prompt_text, self.tokenizer)

        from vllm_omni.model_executor.models.funcineforge.utils import (
            concat_text_with_prompt_ids,
            dialogue_to_timespk_ids,
            extract_speech_feat,
            extract_speech_token,
            extract_spk_embedding,
            speech_type_to_id,
        )

        speech_type = mm_kwargs.get("speech_type")
        dialogue = mm_kwargs.get("dialogue")
        timespk_ids = dialogue_to_timespk_ids(dialogue if isinstance(dialogue, list) else None, config)

        input_ids, input_len = concat_text_with_prompt_ids(
            text_token,
            text_token_len,
            prompt_text_token,
            prompt_text_token_len,
            sos=config.sos,
            turn_of_speech=config.turn_of_speech,
            type_id=speech_type_to_id(speech_type if isinstance(speech_type, str) else None, config),
            timespk_ids=timespk_ids,
            startofclue_token=config.startofclue_token,
            endofclue_token=config.endofclue_token,
            lm_use_prompt=True,
        )

        device = "cpu"
        speech_token, speech_token_len = extract_speech_token(audio, self.speech_tokenizer, device)
        speech_feat, _ = extract_speech_feat(audio, self.feat_extractor, device)

        # Align lengths for 24kHz (token_rate=25, feat_rate=50)
        # speech_feat shape: (1, n_mels, T) — time is last dim.
        if config.sample_rate == 24000:
            feat_time = speech_feat.shape[-1]
            tok_len_val = speech_token.shape[1]
            token_len = min(feat_time // 2, tok_len_val)
            speech_feat = speech_feat[..., : 2 * token_len]
            speech_token = speech_token[:, :token_len]
            speech_token_len = torch.tensor([token_len], dtype=torch.int32)

        embedding = extract_spk_embedding(audio, self.campplus_session, device)

        # Face embeddings: from mm_kwargs or default zeros
        face_emb = mm_kwargs.get("face_embedding", None)
        if face_emb is not None and isinstance(face_emb, torch.Tensor):
            pass
        else:
            speech_len = mm_kwargs.get("speech_len")
            max_face_len = int(speech_len) if speech_len is not None else speech_token.shape[1]
            face_emb = torch.zeros(1, max_face_len, config.face_size, dtype=torch.float32)

        # Face placeholder tokens are NOT inserted here — they are added via
        # PromptInsertion in _get_prompt_updates (matching CosyVoice3 pattern).
        # This lets vLLM's multimodal framework track placeholder positions.

        return BatchFeature(
            {
                "input_ids": input_ids,
                "input_len": [input_ids.shape[1]],
                "speech_feat": speech_feat,
                "speech_token": speech_token,
                "speech_token_len": speech_token_len.long()
                if isinstance(speech_token_len, torch.Tensor)
                else torch.tensor([speech_token_len], dtype=torch.long),
                "embedding": embedding,
                "face_embedding": face_emb,
            }
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
            "speech_feat": MultiModalFieldConfig.batched("audio"),
            "speech_token": MultiModalFieldConfig.batched("audio"),
            "speech_token_len": MultiModalFieldConfig.batched("audio"),
            "embedding": MultiModalFieldConfig.batched("audio"),
            "face_embedding": MultiModalFieldConfig.batched("audio"),
        }

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        return False

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        # code2wav stage has no SOS token in its input — skip prompt insertion.
        model_stage = getattr(self.info.ctx.model_config, "model_stage", "")
        if model_stage == "funcineforge_code2wav":
            return []

        # Insert face placeholder tokens after SOS via PromptInsertion.
        # This matches CosyVoice3's pattern and lets vLLM's multimodal
        # framework correctly track placeholder positions + budget.
        def insertion_content(item_idx: int) -> list[int]:
            audio_kwargs = out_mm_kwargs["audio"][item_idx]
            face_embedding = audio_kwargs.get("face_embedding")
            if face_embedding is not None:
                face_data = getattr(face_embedding, "data", face_embedding)
                if isinstance(face_data, torch.Tensor) and face_data.numel() > 0:
                    face_len = face_data.shape[-2] if face_data.dim() >= 2 else face_data.shape[0]
                    return [_FACE_TOKEN] * int(face_len)

            stl = audio_kwargs["speech_token_len"].data
            speech_token_len = int(stl.item() if stl.dim() == 0 else stl[0].item())
            return [_FACE_TOKEN] * speech_token_len

        return [
            PromptInsertion(
                modality="audio",
                target=PromptIndexTargets.prefix([self.info.get_hf_config().sos]),
                insertion=insertion_content,
            ),
        ]


class FunCineForgeDummyInputsBuilder(BaseDummyInputsBuilder[FunCineForgeMultiModalProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "Hello, this is a test of the FunCineForge dubbing system."

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
        inputs.hf_processor_mm_kwargs = {"prompt_text": "Testing my voices."}
        return inputs


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------


@MULTIMODAL_REGISTRY.register_processor(
    FunCineForgeMultiModalProcessor,
    info=FunCineForgeMultiModalProcessingInfo,
    dummy_inputs=FunCineForgeDummyInputsBuilder,
)
class FunCineForgeModel(
    nn.Module,
    SupportsMultiModal,
):
    supports_multimodal_raw_input_only = True
    supports_multimodal = True
    requires_raw_input_tokens = True
    prefer_model_sampler = True
    _sampling_eps = 1e-5

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.have_multimodal_outputs = True
        self.model_stage = vllm_config.model_config.model_stage
        model_dir = vllm_config.model_config.model
        if not os.path.isdir(model_dir):
            from huggingface_hub import snapshot_download

            model_dir = snapshot_download(model_dir)
        self.model_dir = model_dir
        self.model = None

        if self.model_stage == "funcineforge_talker":
            from vllm_omni.model_executor.models.funcineforge.funcineforge_talker import (
                FunCineForgeTalker,
                VLLMQwen2Encoder,
            )

            llm_vllm_config = self._create_llm_vllm_config(vllm_config)
            llm = VLLMQwen2Encoder(vllm_config=llm_vllm_config, prefix="model")
            self.talker = FunCineForgeTalker(
                codec_unit=self.config.codec_unit,
                timespk_unit=self.config.timespk_unit,
                face_size=self.config.face_size,
                llm=llm,
            )
            self.model = self.talker
        elif self.model_stage == "funcineforge_code2wav":
            multimodal_config = vllm_config.model_config.multimodal_config
            if multimodal_config is not None:
                multimodal_config.skip_mm_profiling = True
            from vllm_omni.model_executor.models.funcineforge.funcineforge_code2wav import (
                FunCineForgeCode2Wav,
            )

            logger.info("code2wav: creating FunCineForgeCode2Wav ...")
            self.code2wav = FunCineForgeCode2Wav(self.config)
            logger.info("code2wav: init OK — DiT depth=%d, vocoder ready", self.code2wav.flow_model.depth)
            self.model = self.code2wav.flow_model
            self.enable_update_additional_information = True

            self._stream_audio_cache_lock = Lock()
            self._stream_vocoder_cache_by_req: dict[str, dict[str, torch.Tensor]] = {}
            self._stream_prompt_embed_by_req: dict[str, EmbeddingsStruct] = {}
        else:
            raise ValueError(f"Model stage not supported: {self.model_stage}")

        self._debug_tokens = bool(os.environ.get("FUNCINEFORGE_DEBUG_TOKENS"))
        self._eos_diag_step = 0

    def _create_llm_vllm_config(self, parent_config: VllmConfig) -> VllmConfig:
        """Create VllmConfig for the inner Qwen2 LLM."""
        from transformers import Qwen2Config

        qwen_config_path = os.path.join(self.model_dir, self.config.llm["llm"]["pretrain_path"])
        qwen_hf_config = Qwen2Config.from_pretrained(qwen_config_path)
        return parent_config.with_hf_config(qwen_hf_config, architectures=["Qwen2Model"])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cleanup_stream_cache(self, req_id: str | None, audio: torch.Tensor, stream_finished: bool) -> torch.Tensor:
        if req_id is not None and stream_finished:
            with self._stream_audio_cache_lock:
                if hasattr(self, "_stream_vocoder_cache_by_req"):
                    self._stream_vocoder_cache_by_req.pop(req_id, None)
                if hasattr(self, "_stream_prompt_embed_by_req"):
                    self._stream_prompt_embed_by_req.pop(req_id, None)
        return audio

    def _get_prompt_embed_for_code2wav(
        self,
        req_id: str | None,
        embed: EmbeddingsStruct | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        speech_token = embed.speech_token if embed else None
        speech_feat = embed.speech_feat if embed else None
        embedding = embed.embedding if embed else None
        has_prompt = speech_token is not None and speech_feat is not None and embedding is not None
        if req_id is None or not hasattr(self, "_stream_prompt_embed_by_req"):
            return speech_token, speech_feat, embedding

        with self._stream_audio_cache_lock:
            if has_prompt:
                self._stream_prompt_embed_by_req[req_id] = EmbeddingsStruct(
                    speech_token=speech_token,
                    speech_feat=speech_feat,
                    embedding=embedding,
                )
            else:
                cached = self._stream_prompt_embed_by_req.get(req_id)
                if cached is not None:
                    speech_token = cached.speech_token
                    speech_feat = cached.speech_feat
                    embedding = cached.embedding

        return speech_token, speech_feat, embedding

    @staticmethod
    def _debug_codec_tokens(
        req_id: str | None, token: torch.Tensor, raw_ids: torch.Tensor, stream_finished: bool, *, enabled: bool = False
    ) -> None:
        if not enabled:
            return
        token_cpu = token.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        raw_cpu = raw_ids.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        logger.info(
            "funcineforge code2wav tokens req=%s finished=%s len=%d first=%s last=%s"
            " raw_len=%d raw_first=%s raw_last=%s",
            req_id,
            stream_finished,
            int(token_cpu.numel()),
            token_cpu[:20].tolist(),
            token_cpu[-20:].tolist(),
            int(raw_cpu.numel()),
            raw_cpu[:20].tolist(),
            raw_cpu[-20:].tolist(),
        )

    @staticmethod
    def _split_request_ids(ids: torch.Tensor, seq_token_counts: list[int] | None = None) -> list[torch.Tensor]:
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
        vocab_size = int(self.code2wav.input_embedding.num_embeddings)
        valid_mask = (req_ids >= 0) & (req_ids < vocab_size)
        return req_ids[valid_mask]

    def _funcineforge_ras_enabled(self, sampling_metadata: SamplingMetadata) -> bool:
        if self.model_stage != "funcineforge_talker":
            return False
        if sampling_metadata.max_num_logprobs is not None:
            if self._debug_tokens:
                logger.info("funcineforge ras disabled: logprobs requested")
            return False
        if bool(sampling_metadata.bad_words_token_ids):
            if self._debug_tokens:
                logger.info("funcineforge ras disabled: bad_words_token_ids")
            return False
        if torch.any(sampling_metadata.frequency_penalties != 0):
            if self._debug_tokens:
                logger.info("funcineforge ras disabled: frequency_penalties")
            return False
        if torch.any(sampling_metadata.presence_penalties != 0):
            if self._debug_tokens:
                logger.info("funcineforge ras disabled: presence_penalties")
            return False
        return True

    # ------------------------------------------------------------------
    # Sampling (RAS)
    # ------------------------------------------------------------------

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        if logits is None or logits.numel() == 0:
            return None
        if self.model_stage != "funcineforge_talker":
            return None

        sampler = getattr(self, "_talker_sampler", None)
        if sampler is None:
            sampler = Sampler()
            self._talker_sampler = sampler

        if not self._funcineforge_ras_enabled(sampling_metadata):
            return sampler(logits=logits, sampling_metadata=sampling_metadata)

        logits = logits.to(torch.float32)
        sampling_for_processors = replace(sampling_metadata, no_penalties=True)
        logits = sampler.apply_logits_processors(logits, sampling_for_processors, predict_bonus_token=False)

        # RAS parameters from config
        sampling_cfg = dict(self.config.llm.get("sampling", {}))
        default_top_p = float(sampling_cfg.get("top_p", 0.8))
        default_top_k = int(sampling_cfg.get("top_k", 25))
        win_size = int(sampling_cfg.get("win_size", 10))
        tau_r = float(sampling_cfg.get("tau_r", 0.1))

        sampled_ids: list[int] = []
        for req_idx in range(int(logits.shape[0])):
            row_logits = logits[req_idx]
            temperature = float(_CosyVoice3Sampling._req_scalar(sampling_metadata.temperature, req_idx, 1.0))
            if temperature < self._sampling_eps:
                sampled_ids.append(int(torch.argmax(row_logits).item()))
                continue

            top_p = float(_CosyVoice3Sampling._req_scalar(sampling_metadata.top_p, req_idx, default_top_p))
            top_k = int(_CosyVoice3Sampling._req_scalar(sampling_metadata.top_k, req_idx, default_top_k))
            generator = sampling_metadata.generators.get(req_idx)
            weighted_scores = torch.log_softmax(row_logits / max(temperature, self._sampling_eps), dim=0)
            decoded_tokens = (
                sampling_metadata.output_token_ids[req_idx] if req_idx < len(sampling_metadata.output_token_ids) else []
            )
            sampled_id = _CosyVoice3Sampling._ras_sample_one(
                weighted_scores,
                decoded_tokens,
                top_p=top_p,
                top_k=top_k,
                win_size=win_size,
                tau_r=tau_r,
                generator=generator,
            )
            if self._debug_tokens and len(decoded_tokens) < 35:
                probs = weighted_scores.softmax(dim=0)
                top_probs, top_ids = probs.topk(5)
                logger.info(
                    "funcineforge ras trace step=%d argmax=%d chosen=%d top5=%s probs=%s top_p=%.3f top_k=%d temp=%.3f",
                    len(decoded_tokens),
                    int(torch.argmax(weighted_scores).item()),
                    int(sampled_id),
                    top_ids.detach().cpu().tolist(),
                    [round(float(x), 6) for x in top_probs.detach().cpu().tolist()],
                    top_p,
                    top_k,
                    temperature,
                )
            sampled_ids.append(sampled_id)

        sampled = torch.tensor(sampled_ids, device=logits.device, dtype=torch.int32)
        return SamplerOutput(sampled_token_ids=sampled.unsqueeze(-1), logprobs_tensors=None)

    # ------------------------------------------------------------------
    # Logits / embeddings
    # ------------------------------------------------------------------

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if self.model_stage == "funcineforge_talker":
            logits = self.model.codec_head(hidden_states)
            # No logit masking — match original FunCineForge (llm_decoding.py):
            # the model operates on the full codec_unit (6761) logit space.
            # stop_token_ids=[6562] in pipeline.py handles EOS detection.
            # Pad to vocab_size for vLLM sampler.
            vocab_size = self.config.vocab_size  # 151936
            pad_size = vocab_size - logits.size(-1)
            if pad_size > 0:
                pad_shape = logits.shape[:-1] + (pad_size,)
                pad = logits.new_full(pad_shape, float("-inf"))
                logits = torch.cat([logits, pad], dim=-1)
            if self._debug_tokens:
                self._eos_diag_step += 1
                if self._eos_diag_step <= 5 or self._eos_diag_step % 200 == 0:
                    eos_idx = self.config.eos
                    codebook_size = int(self.config.flow.get("codebook_size", 6561))
                    eos_v = float(logits[0, eos_idx].item()) if logits.dim() >= 2 else float(logits[eos_idx].item())
                    a_max = (
                        float(logits[0, :codebook_size].max().item())
                        if logits.dim() >= 2
                        else float(logits[:codebook_size].max().item())
                    )
                    rank = int((logits[0, :codebook_size] > eos_v).sum().item()) + 1 if logits.dim() >= 2 else 0
                    logger.info(
                        "EOS diag step=%d: eos=%.2f audio_max=%.2f rank=%d/%d",
                        self._eos_diag_step,
                        eos_v,
                        a_max,
                        rank,
                        codebook_size,
                    )
            return logits
        else:
            raise RuntimeError(f"compute_logits is only valid for funcineforge_talker, got {self.model_stage}.")

    def embed_multimodal(self, **kwargs: object) -> torch.Tensor:
        if self.model_stage == "funcineforge_talker":
            face_embedding = kwargs.get("face_embedding")
            if isinstance(face_embedding, list):
                face_embedding = face_embedding[0] if face_embedding else None
            if not isinstance(face_embedding, torch.Tensor):
                raise ValueError("face_embedding is required for FunCineForge multimodal embedding")
            if face_embedding.dim() == 2:
                face_embedding = face_embedding.unsqueeze(0)
            return self.model.face_linear(
                face_embedding.to(
                    device=self.model.face_linear.weight.device, dtype=self.model.face_linear.weight.dtype
                )
            )
        else:
            raise RuntimeError("embed_multimodal is only valid for funcineforge_talker.")

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
        is_multimodal=None,
    ) -> torch.Tensor:
        if self.model_stage == "funcineforge_talker":
            # Decode path: generated codec IDs can arrive as one token or as
            # a short replay block around KV-cache block boundaries.  Route
            # all pure codec-codebook chunks through codec_embed, matching the
            # official AR decoder's token_embedder path.
            # Prefill can be scheduled in chunks; chunks after the face span no
            # longer contain _FACE_TOKEN, but still must use the official
            # text/timespk/codec embedding streams rather than codec_embed.
            codebook_size = int(self.config.flow.get("codebook_size", 6561))
            flat_ids = input_ids.reshape(-1)
            if flat_ids.numel() > 0 and torch.all(flat_ids >= 0) and torch.all(flat_ids < codebook_size):
                return self._embed_by_token_range(input_ids)
            return self._embed_talker_multimodal(input_ids, multimodal_embeddings)
        elif self.model_stage == "funcineforge_code2wav":
            assert input_ids.dim() == 1
            hidden = int(self.config.hidden_size)
            return torch.zeros(
                (input_ids.shape[0], hidden),
                device=input_ids.device,
            )
        else:
            raise RuntimeError(f"embed_input_ids is not valid for {self.model_stage}.")

    def _embed_by_token_range(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed tokens via codec_embed for decode / profiling path.

        During AR decode, the talker only generates codec IDs (0 .. codec_unit-1).
        During profiling/warmup, all tokens are zero which maps to codec_embed's
        padding_idx — harmless.  This mirrors CosyVoice3's approach of always
        routing decode tokens through the speech embedding table.
        """
        return self.model.codec_embed(input_ids.clamp(max=self.config.codec_unit - 1))

    def _embed_talker_multimodal(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings=None,
    ) -> torch.Tensor:
        """Build combined embeddings using the three-embedding-table pattern.

        During prefill the input sequence is::

            [SOS, face_token×M, startofclue, clue_text..., endofclue,
             target_text..., type_id, timespk_ids..., turn_of_speech]

        No speech tokens in LM input — reference audio flows to code2wav
        via forward() kwargs only.

        Flag assignment (by token value):

          * codec_flag  → SOS/EOS/turn_of_speech/face_token
          * timespk_flag → IDs in [pangbai, timespk_unit)
          * text_flag   → everything else

        face_token positions → face_linear(face_embedding) from the processor.
        """
        cfg = self.config
        ids = input_ids  # (1, T) or (T,)

        # --- Build flag tensors ---
        # Official FunCineForge marks the entire [type_id, timespk_ids...]
        # span by position, not by token value.  The time indices inside
        # timespk_ids can be small numbers (e.g. 1, 144), so value-based
        # checks would accidentally send them through Qwen text embeddings.
        codec_flag = torch.zeros_like(ids, dtype=torch.bool)
        text_flag = torch.ones_like(ids, dtype=torch.bool)
        timespk_flag = torch.zeros_like(ids, dtype=torch.bool)

        flat_ids = ids[0] if ids.dim() == 2 else ids
        face_mask = flat_ids == _FACE_TOKEN
        codec_mask = (flat_ids == cfg.sos) | (flat_ids == cfg.eos) | (flat_ids == cfg.turn_of_speech) | face_mask
        type_positions = (
            (flat_ids == cfg.pangbai) | (flat_ids == cfg.dubai) | (flat_ids == cfg.duihua) | (flat_ids == cfg.duoren)
        ).nonzero(as_tuple=True)[0]
        tos_positions = (flat_ids == cfg.turn_of_speech).nonzero(as_tuple=True)[0]
        if len(type_positions) > 0 and len(tos_positions) > 0:
            type_pos = int(type_positions[-1].item())
            tos_pos = int(tos_positions[-1].item())
            if type_pos < tos_pos:
                if ids.dim() == 2:
                    timespk_flag[0, type_pos:tos_pos] = True
                else:
                    timespk_flag[type_pos:tos_pos] = True

        if ids.dim() == 2:
            codec_flag[0] = codec_mask
        else:
            codec_flag = codec_mask
        text_flag = ~(timespk_flag | codec_flag)

        text_flag_f = text_flag.float()
        timespk_flag_f = timespk_flag.float()
        codec_flag_f = codec_flag.float()

        # --- Three embedding streams ---
        text_embeds = self.model.llm.model.embed_tokens(ids) * text_flag_f.unsqueeze(-1)

        timespk_ids = (ids * timespk_flag.long()).clamp(max=cfg.timespk_unit - 1)
        timespk_embeds = self.model.timespk_embed(timespk_ids) * timespk_flag_f.unsqueeze(-1)

        codec_ids = (ids * codec_flag.long()).clamp(max=cfg.codec_unit - 1)
        codec_embeds = self.model.codec_embed(codec_ids) * codec_flag_f.unsqueeze(-1)

        inputs_embeds = text_embeds + timespk_embeds + codec_embeds

        # Replace face placeholder positions with the projected face embeddings.
        face_mask = ids == _FACE_TOKEN
        if ids.dim() == 2:
            face_pos = face_mask[0].nonzero(as_tuple=True)[0]
        else:
            face_pos = face_mask.nonzero(as_tuple=True)[0]
        if len(face_pos) > 0:
            n_face = len(face_pos)
            face_proj = None
            if multimodal_embeddings is not None:
                face_proj = (
                    multimodal_embeddings[0]
                    if isinstance(multimodal_embeddings, (list, tuple))
                    else multimodal_embeddings
                )
                if isinstance(face_proj, torch.Tensor):
                    if face_proj.dim() == 3:
                        face_proj = face_proj.reshape(-1, face_proj.shape[-1])
                    face_proj = face_proj[:n_face].to(
                        device=inputs_embeds.device,
                        dtype=inputs_embeds.dtype,
                    )
            if not isinstance(face_proj, torch.Tensor) or face_proj.shape[0] < n_face:
                face_zeros = torch.zeros(
                    n_face,
                    cfg.face_size,
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype,
                )
                face_proj = self.model.face_linear(face_zeros)
            if inputs_embeds.dim() == 3:
                inputs_embeds[0, face_pos] = face_proj
            else:
                inputs_embeds[face_pos] = face_proj

        return inputs_embeds

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        additional_information: dict[str, object] | None = None,
        **kwargs: object,
    ) -> OmniOutput:
        if self.model_stage == "funcineforge_talker":
            if inputs_embeds is None:
                inputs_embeds = self.embed_input_ids(input_ids)

            hidden_states = self.model.llm(inputs_embeds, positions)

            multimodal_outputs = {}
            if "speech_token" in kwargs:
                multimodal_outputs = to_dict(
                    OmniPayloadStruct(
                        embed=EmbeddingsStruct(
                            speech_token=kwargs.get("speech_token"),
                            speech_feat=kwargs.get("speech_feat"),
                            embedding=kwargs.get("embedding"),
                        ),
                    )
                )

            return OmniOutput(text_hidden_states=hidden_states, multimodal_outputs=multimodal_outputs)

        elif self.model_stage == "funcineforge_code2wav":
            runtime_info = kwargs.get("model_intermediate_buffer")
            if runtime_info is None:
                runtime_info = kwargs.get("runtime_additional_information", [])
            if "runtime_additional_information" in kwargs and "model_intermediate_buffer" not in kwargs:
                logger.warning_once("runtime_additional_information is deprecated, use model_intermediate_buffer")

            seq_token_counts = kwargs.get("seq_token_counts")
            flat_ids = input_ids.reshape(-1).to(dtype=torch.long)
            request_ids_list = self._split_request_ids(flat_ids, seq_token_counts)

            num_reqs = max(1, len(request_ids_list))
            sample_rate = torch.tensor(int(self.config.sample_rate), dtype=torch.int32)
            empty_audio = torch.zeros((0,), dtype=torch.float32, device=input_ids.device)
            audios: list[torch.Tensor] = [empty_audio] * num_reqs
            srs: list[torch.Tensor] = [sample_rate] * num_reqs
            if not isinstance(runtime_info, list):
                runtime_info = []

            for idx, req_ids in enumerate(request_ids_list):
                raw = runtime_info[idx] if idx < len(runtime_info) and isinstance(runtime_info[idx], dict) else {}
                payload = to_struct(raw)
                meta = payload.meta
                embed = payload.embed

                req_id = meta.req_id[0] if (meta and meta.req_id) else None
                stream_finished = (
                    bool(meta.stream_finished.item()) if (meta and meta.stream_finished is not None) else False
                )
                speech_token, speech_feat, embedding = self._get_prompt_embed_for_code2wav(req_id, embed)
                if speech_token is None or speech_feat is None or embedding is None:
                    if stream_finished and req_id is not None and hasattr(self, "_stream_vocoder_cache_by_req"):
                        with self._stream_audio_cache_lock:
                            self._stream_vocoder_cache_by_req.pop(req_id, None)
                            if hasattr(self, "_stream_prompt_embed_by_req"):
                                self._stream_prompt_embed_by_req.pop(req_id, None)
                    audios[idx] = self._cleanup_stream_cache(req_id, empty_audio, stream_finished)
                    continue

                token = self._sanitize_codec_tokens(req_ids)
                self._debug_codec_tokens(req_id, token, req_ids, stream_finished, enabled=self._debug_tokens)
                if token.numel() == 0:
                    audios[idx] = self._cleanup_stream_cache(req_id, empty_audio, stream_finished)
                    continue

                uses_streaming_decode = meta and (
                    meta.stream_finished is not None or meta.left_context_size is not None
                )
                if uses_streaming_decode:
                    token_offset = max(0, meta.left_context_size or 0)

                    cache_state = None
                    if req_id is not None and hasattr(self, "_stream_vocoder_cache_by_req"):
                        with self._stream_audio_cache_lock:
                            cache_state = self._stream_vocoder_cache_by_req.get(req_id)

                    n_timesteps = int(self.config.flow.get("n_timesteps", 10))
                    tts_speech, new_cache_state = self.code2wav.forward_streaming(
                        token=token.unsqueeze(0),
                        prompt_token=speech_token[:1],
                        prompt_feat=speech_feat[:1],
                        embedding=embedding[:1],
                        cache_state=cache_state,
                        n_timesteps=n_timesteps,
                        token_offset_tokens=token_offset,
                        finalize=stream_finished,
                    )

                    if req_id is not None and hasattr(self, "_stream_vocoder_cache_by_req"):
                        with self._stream_audio_cache_lock:
                            if new_cache_state is None or stream_finished:
                                self._stream_vocoder_cache_by_req.pop(req_id, None)
                            else:
                                self._stream_vocoder_cache_by_req[req_id] = new_cache_state
                else:
                    n_timesteps = int(self.config.flow.get("n_timesteps", 10))
                    tts_speech = self.code2wav.forward(
                        token=token.unsqueeze(0),
                        prompt_token=speech_token[:1],
                        prompt_feat=speech_feat[:1],
                        embedding=embedding[:1],
                        n_timesteps=n_timesteps,
                    )

                audio = tts_speech.reshape(-1).to(dtype=torch.float32)
                audios[idx] = self._cleanup_stream_cache(req_id, audio, stream_finished)

            return OmniOutput(text_hidden_states=None, multimodal_outputs={"audio": audios, "sr": srs})
        else:
            raise ValueError(f"Unsupported model_stage: {self.model_stage}")

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        if self.model_stage == "funcineforge_talker":
            device = next(self.parameters()).device
            ckpt_path = os.path.join(self.model_dir, self.config.llm_ckpt)
            checkpoint = load_deepspeed_checkpoint(ckpt_path, device)

            # 1. Load Qwen2 model weights into vLLM's Qwen2Model
            # DeepSpeed keys: "llm.model.X" → vLLM Qwen2Model expects "X"
            qwen_weights = []
            for name, weight in checkpoint.items():
                if name.startswith("llm.model."):
                    vllm_name = name.replace("llm.model.", "", 1)
                    qwen_weights.append((vllm_name, weight))
            logger.info("talker: loading %d Qwen2 weights", len(qwen_weights))
            self.model.llm.model.load_weights(iter(qwen_weights))

            # 2. Load FunCineForge-specific weights
            codec_embed_state = {
                k.replace("codec_embed.", ""): v for k, v in checkpoint.items() if k.startswith("codec_embed.")
            }
            if codec_embed_state:
                self.model.codec_embed.load_state_dict(codec_embed_state)

            timespk_embed_state = {
                k.replace("timespk_embed.", ""): v for k, v in checkpoint.items() if k.startswith("timespk_embed.")
            }
            if timespk_embed_state:
                self.model.timespk_embed.load_state_dict(timespk_embed_state)

            codec_head_state = {
                k.replace("codec_head.", ""): v for k, v in checkpoint.items() if k.startswith("codec_head.")
            }
            if codec_head_state:
                self.model.codec_head.load_state_dict(codec_head_state)

            face_linear_state = {
                k.replace("face_linear.", ""): v for k, v in checkpoint.items() if k.startswith("face_linear.")
            }
            if face_linear_state:
                self.model.face_linear.load_state_dict(face_linear_state)

            self.model.to(device).eval()

        elif self.model_stage == "funcineforge_code2wav":
            device = next(self.parameters()).device
            logger.info("code2wav: load_weights device=%s model_dir=%s", device, self.model_dir)
            self.code2wav.load_weights(self.model_dir, device)
            logger.info("code2wav: load_weights OK")
        else:
            raise ValueError(f"{self.model_stage} not supported yet!")

        # Return None to skip diffusers_loader strict weight check —
        # weights are loaded manually from DeepSpeed checkpoints above.
        # (Returning empty set() would flag ALL parameters as missing.)
