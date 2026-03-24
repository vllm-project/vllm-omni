from collections.abc import Iterable
from functools import cached_property
from pathlib import Path

import regex as re
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
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
    offset = 0
    while offset < input_ids.numel():
        ctx_frames = int(input_ids[offset].item())
        context_length = int(input_ids[offset + 1].item())
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
            self.has_postprocess = True
            self.requires_raw_input_tokens = True
            self.set_custom_preprocess(self.tts_preprocess)
            self.set_custom_postprocess(self.tts_postprocess)
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
            if speaker_id and self.is_hf_model:
                self.voice_to_embedding = {
                    sid: torch.load(hf_hub_download(repo_id=self.repo_id, filename=f"voice_embedding/{sid}.pt"))
                    for sid in speaker_id
                }
                logger.info("Available voice embeddings: %s", list(self.voice_to_embedding.keys()))
            else:
                self.voice_to_embedding = {}
                logger.warning("No speaker_id configured in audio_config.No voice embeddings will be available.")
        elif self.model_stage == "audio_tokenizer":
            self.requires_raw_input_tokens = True
            self.audio_generation = None
            self.audio_tokenizer = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config,
                prefix=maybe_prefix(prefix, "audio_tokenizer"),
                architectures=["VoxtralTTSAudioTokenizer"],
            )
            self.model = self.audio_tokenizer
        else:
            raise ValueError("Invalid model stage")

    # -------------------- CUDA Graph for acoustic transformer --------------------
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

    def tts_preprocess(self, input_ids: torch.Tensor, input_embeds: torch.Tensor, **info_dict: dict | None):
        self.post_process_idx = 0
        audio_tokens = info_dict.pop("audio", None)
        if audio_tokens is not None:
            kwargs = {"audio_tokens": audio_tokens.to(input_ids.device)}
            multimodal_embeddings = self.model.embed_multimodal(**kwargs)
            if input_ids[0] == self._audio_token_id:
                input_embeds = multimodal_embeddings[0]
            return input_ids, input_embeds, info_dict
        voice = info_dict.pop("voice", None)
        if voice is not None:
            if isinstance(voice, list):
                voice = voice[0]
            multimodal_embeddings = self.voice_to_embedding[voice].to(input_ids.device).clone().detach()
            is_multimodal = input_ids == self._audio_token_id
            input_embeds = self.embed_input_ids(
                input_ids=input_ids, multimodal_embeddings=multimodal_embeddings, is_multimodal=is_multimodal
            )
            return input_ids, input_embeds, info_dict
        return input_ids, input_embeds, info_dict

    def tts_postprocess(self, hidden_states: torch.Tensor, multimodal_outputs: object, **info_dict: object | None):
        update_dict = {}
        if isinstance(multimodal_outputs, dict) and "audio" in multimodal_outputs:
            assert self.post_process_idx < len(multimodal_outputs["audio"]), (
                f"Expect {self.post_process_idx=} < {len(multimodal_outputs['audio'])=}"
            )
            update_dict["audio"] = multimodal_outputs["audio"][self.post_process_idx]
            self.post_process_idx += 1
        return update_dict

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

    def last_index_of(self, list, value):
        return len(list) - 1 - list[::-1].index(value)

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

    def make_omni_output(
        self, model_outputs: torch.Tensor | OmniOutput | tuple, logits_index: int | None = None, **kwargs
    ) -> OmniOutput:
        if isinstance(model_outputs, torch.Tensor):
            if self.model_stage == "audio_generation":
                hidden_states = model_outputs
                assert logits_index is not None
                input_hidden_states = hidden_states[logits_index]
                if self._cudagraph_acoustic_transformer is not None:
                    fake_eos, multimodal_outputs = self._cudagraph_acoustic_transformer(input_hidden_states)
                else:
                    fake_eos, multimodal_outputs = self.model.compute_mm_logits(input_hidden_states)
                hidden_states[logits_index, 0] = fake_eos
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
