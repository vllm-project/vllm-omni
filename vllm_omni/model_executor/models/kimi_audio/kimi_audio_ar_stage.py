# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Kimi-Audio dual-stream AR forward, weight loading, and input encoding.

Worker registration and downstream audio output are still pending.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .audio_processing import CHUNK_SAMPLES, SAMPLE_RATE, SAMPLES_PER_TOKEN, KimiAudioWhisperInputs
from .prompt import KimiAudioEncodedAudio, KimiAudioSpecialTokens
from .sampling import KimiAudioSamplingParams, sample_kimi_audio_step

if TYPE_CHECKING:
    from transformers import WhisperFeatureExtractor
    from vllm.config import VllmConfig
    from vllm.model_executor.models.kimi_audio import KimiAudioWhisperEncoder
    from vllm.sequence import IntermediateTensors
    from vllm.v1.outputs import SamplerOutput
    from vllm.v1.sample.metadata import SamplingMetadata

    from vllm_omni.model_executor.models.common.whisper_vq import WhisperVQEncoder
    from vllm_omni.model_executor.models.output_templates import OmniOutput


class KimiAudioInputEncoder(torch.nn.Module):
    """Own the GLM tokenizer and continuous audio encoder inside the AR worker.

    Construction does not import vLLM, read checkpoints, or allocate model
    weights. When audio input is enabled, the enclosing AR model must call
    ``load_glm_weights`` and ``load_whisper_weights`` from its weight-loading
    hook, before memory profiling and KV-cache allocation. Deployments without
    audio input can omit both; audio-text history alone does not need Whisper.
    The enclosing vLLM loader also performs normal weight postprocessing.
    """

    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.prefix = prefix
        self.audio_tokenizer: WhisperVQEncoder | None = None
        self.glm_feature_extractor: WhisperFeatureExtractor | None = None
        self._glm_tokenizer_path: Path | None = None
        self._glm_loaded_weights: set[str] = set()
        self.whisper_encoder: KimiAudioWhisperEncoder | None = None
        self._whisper_loaded_weights: set[str] = set()

    def load_glm_weights(self, tokenizer_path: str) -> set[str]:
        """Load the internal WhisperVQ encoder from a local GLM-4-Voice snapshot.

        The enclosing model resolves the GLM checkpoint separately from the
        Kimi LLM checkpoint. Keep the checkpoint's FP32 tensors, independently
        of the LLM dtype. Returned weight names are relative to this input
        module, as with ``load_whisper_weights``.
        """
        path = Path(tokenizer_path).expanduser().resolve()
        if self.audio_tokenizer is not None:
            if path != self._glm_tokenizer_path:
                raise ValueError("This input encoder already holds a different GLM checkpoint")
            return set(self._glm_loaded_weights)
        if not path.is_dir():
            raise ValueError("GLM tokenizer_path must be a resolved local checkpoint directory")

        from safetensors.torch import load_file
        from transformers import WhisperConfig, WhisperFeatureExtractor

        from vllm_omni.model_executor.models.common.whisper_vq import WhisperVQEncoder

        config = WhisperConfig.from_pretrained(path, local_files_only=True)
        feature_extractor = WhisperFeatureExtractor.from_pretrained(path, local_files_only=True)
        config.quantize_encoder_only = True
        # Construct on meta to avoid allocating/initializing the unused
        # post-quantization layers. Assignment preserves checkpoint dtypes.
        with torch.device("meta"):
            tokenizer = WhisperVQEncoder(
                config,
                causal_block_size=config.quantize_causal_block_size,
                preserve_padding=True,
            )
        stride = tokenizer.conv1.stride[0] * tokenizer.conv2.stride[0] * config.pooling_kernel_size
        if feature_extractor.sampling_rate != SAMPLE_RATE or stride * feature_extractor.hop_length != SAMPLES_PER_TOKEN:
            raise ValueError("Expected a GLM-4-Voice tokenizer with 16 kHz input and 12.5 Hz output")
        weights = load_file(str(path / "model.safetensors"))
        # EMA statistics belong to codebook training, not inference. Every
        # inference tensor must still match; do not silently skip model keys.
        weights.pop("ema_count", None)
        weights.pop("ema_weight", None)
        tokenizer.load_state_dict(weights, strict=True, assign=True)
        device = self.vllm_config.load_config.device or self.vllm_config.device_config.device
        tokenizer = tokenizer.to(device=device).eval()
        loaded = {f"audio_tokenizer.{name}" for name, _ in tokenizer.named_parameters()}
        self.audio_tokenizer = tokenizer
        self.glm_feature_extractor = feature_extractor
        self._glm_tokenizer_path = path
        self._glm_loaded_weights = loaded
        return set(loaded)

    def load_whisper_weights(self) -> set[str]:
        """Load once during worker initialization, with distributed state ready.

        Returned names are relative to this input module, for the enclosing
        model's loaded-weight bookkeeping. A failed load leaves no cached
        encoder, so it cannot accidentally be used with incomplete weights.
        """
        if self.whisper_encoder is not None:
            return set(self._whisper_loaded_weights)

        from vllm.model_executor.model_loader import DefaultModelLoader
        from vllm.model_executor.models.kimi_audio import KimiAudioWhisperEncoder
        from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix
        from vllm.utils.torch_utils import set_default_torch_dtype

        config = self.vllm_config
        load_device = config.load_config.device or config.device_config.device
        with set_default_torch_dtype(config.model_config.dtype), torch.device(load_device):
            encoder = KimiAudioWhisperEncoder(vllm_config=config, prefix=maybe_prefix(self.prefix, "whisper_encoder"))
        source = DefaultModelLoader.Source(
            model_or_path=config.model_config.model,
            revision=config.model_config.revision,
            subfolder="whisper-large-v3",
        )
        weights = DefaultModelLoader(config.load_config)._get_weights_iterator(source)
        # Match vLLM's Kimi parent mapping; the reused encoder handles fused
        # Q/K/V loading itself. Whisper's text decoder is not an input encoder.
        mapper = WeightsMapper(
            orig_to_new_prefix={"model.encoder.": "", "model.decoder.": None, "proj_out.": None},
            orig_to_new_substr={".fc1.": ".mlp.fc1.", ".fc2.": ".mlp.fc2."},
        )
        loaded = encoder.load_weights(mapper.apply(weights))
        missing = set(dict(encoder.named_parameters())) - loaded
        if missing:
            raise ValueError(f"Missing Kimi-Audio Whisper weights: {sorted(missing)}")

        self.whisper_encoder = encoder.eval()
        self._whisper_loaded_weights = {f"whisper_encoder.{name}" for name in loaded}
        return set(self._whisper_loaded_weights)

    @torch.inference_mode()
    def encode_audio(
        self,
        waveform: np.ndarray,
        *,
        sampling_rate: int,
        whisper_inputs: KimiAudioWhisperInputs | None = None,
    ) -> KimiAudioEncodedAudio:
        """Encode one resolved recording for the prompt builder.

        ``audio`` messages provide CPU-prepared Whisper inputs; ``audio-text``
        history omits them and only needs GLM codes. This request path never
        loads models. Both encoders were prepared during worker initialization
        and are reused across recordings. Continuous features stay on Whisper's
        device.
        """
        if sampling_rate != SAMPLE_RATE:
            raise ValueError("Kimi-Audio expects audio resampled to 16000 Hz by the input layer")
        if waveform.ndim != 1 or waveform.size == 0 or not np.issubdtype(waveform.dtype, np.floating):
            raise ValueError("Expected a nonempty mono floating point waveform")
        if not np.isfinite(waveform).all():
            raise ValueError("Audio waveform must contain finite samples")
        token_lengths = tuple(
            (min(CHUNK_SAMPLES, waveform.size - start) - 1) // SAMPLES_PER_TOKEN + 1
            for start in range(0, waveform.size, CHUNK_SAMPLES)
        )
        whisper_encoder = self.whisper_encoder
        if whisper_inputs is not None:
            if whisper_encoder is None:
                raise RuntimeError("Load Whisper weights during worker initialization before encoding audio")
            if (
                whisper_inputs.token_lengths != token_lengths
                or whisper_inputs.input_features.ndim != 3
                or whisper_inputs.input_features.shape[0] != len(token_lengths)
            ):
                raise ValueError("Whisper inputs do not match the recording's chunk lengths")

        audio_tokenizer = self.audio_tokenizer
        feature_extractor = self.glm_feature_extractor
        if audio_tokenizer is None or feature_extractor is None:
            raise RuntimeError("Load GLM weights during worker initialization before encoding audio")

        # Preserve the official GLM flow: batch up to 128 consecutive 30 s
        # chunks, pad the waveform to the codec stride, and filter tokens only
        # after encoding. Whisper's full 30 s mel padding is a separate path.
        chunks = [
            waveform[start : start + CHUNK_SAMPLES].astype(np.float32, copy=False)
            for start in range(0, waveform.size, CHUNK_SAMPLES)
        ]
        weight = audio_tokenizer.conv1.weight
        conv_stride = audio_tokenizer.conv1.stride[0] * audio_tokenizer.conv2.stride[0]
        pooling = audio_tokenizer.config.pooling_kernel_size
        codes = []
        for start in range(0, len(chunks), 128):
            glm_features = feature_extractor(
                chunks[start : start + 128],
                sampling_rate=SAMPLE_RATE,
                return_attention_mask=True,
                return_tensors="pt",
                device=weight.device,
                padding="longest",
                pad_to_multiple_of=conv_stride * pooling * feature_extractor.hop_length,
            )
            outputs = audio_tokenizer(
                input_features=glm_features.input_features.to(device=weight.device, dtype=weight.dtype),
                attention_mask=glm_features.attention_mask.to(device=weight.device),
            )
            raw_codes = outputs.quantized_token_ids
            mask = glm_features.attention_mask[:, ::conv_stride][:, ::pooling].to(
                device=weight.device, dtype=torch.bool
            )
            if raw_codes is None or raw_codes.shape != mask.shape:
                raise ValueError("GLM codes do not match the padded feature mask")
            if raw_codes.dtype not in (torch.int32, torch.int64):
                raise ValueError("GLM must return integer codebook IDs")
            codes.extend(raw_codes[mask].tolist())
        if len(codes) != sum(token_lengths):
            raise ValueError("GLM code count does not match the recording's 12.5 Hz length")

        continuous_features = None
        if whisper_inputs is not None and whisper_encoder is not None:
            # Encode the complete padded mel, then crop and group four frames.
            weight = whisper_encoder.conv1.weight
            features = []
            for mel, length in zip(whisper_inputs.input_features, token_lengths):
                hidden = whisper_encoder(mel.unsqueeze(0).to(device=weight.device, dtype=weight.dtype))
                frame_count = length * 4
                if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] < frame_count:
                    raise ValueError("Whisper output has fewer frames than the recording requires")
                features.append(hidden[0, :frame_count].reshape(length, 4 * hidden.shape[-1]))
            continuous_features = torch.cat(features, dim=0)

        return KimiAudioEncodedAudio(codes=codes, continuous_features=continuous_features)


class KimiAudioARStage(torch.nn.Module):
    """Own both output branches and load all three input/model checkpoints.

    Decoder layers use vLLM's Qwen2 implementation and follow Kimi's shared
    trunk and text/audio branches. This class is not yet registered as a
    runnable model. Pipeline parallel partitioning is pending.

    ``additional_config["kimi_audio"]["glm_tokenizer_path"]`` can point to an
    existing local GLM snapshot; otherwise loading resolves the pinned source.
    Construction itself does not fetch or load any checkpoint.
    """

    has_preprocess = True
    have_multimodal_outputs = True
    prefer_model_sampler = True
    requires_request_sample_eligibility = True
    skips_model_sampler_output_token_history = True
    omni_pooler_payload_include_hidden = False
    # Keep real IDs even when the request finishes on stage 0. Non-async
    # stage bridging also consumes RequestOutput, so retain audio there too.
    omni_client_multimodal_output_keys = ("ids.output", "codes.audio", "meta.finished")
    # The packed branch states are only needed for current-step logits. They
    # are not downstream conditioning and have width 2H, unlike the H-wide
    # model activations stored by Omni's full-prefix hidden-state cache.
    requires_full_prefix_cached_hidden_states = False

    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = "") -> None:
        super().__init__()
        from transformers import Qwen2Config
        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
        from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer
        from vllm.model_executor.models.utils import maybe_prefix

        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("Kimi-Audio dual-stream pipeline parallel partitioning is not implemented")
        if getattr(vllm_config, "speculative_config", None) is not None:
            raise ValueError("Kimi-Audio dual-stream speculative decoding is not implemented")
        if getattr(vllm_config.cache_config, "enable_prefix_caching", False):
            # Generated audio IDs are absent from the scheduler's token hashes.
            # The prompt's cache_salt cannot distinguish divergent audio tails.
            raise ValueError(
                "Kimi-Audio dual-stream prefix hashes are not implemented; set enable_prefix_caching=False"
            )
        self.vllm_config = vllm_config
        self.config = config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.branch_layer = config.kimia_mimo_transformer_from_layer_index
        if not 0 <= self.branch_layer < config.num_hidden_layers or config.kimia_mimo_layers <= 0:
            raise ValueError("Kimi-Audio requires a valid shared branch point and audio decoder layers")
        if config.tie_word_embeddings:
            raise ValueError("Kimi-Audio expects separate embedding, text head, and audio head weights")

        # Normalize the original HF config to Qwen2's current RoPE fields without
        # changing the Kimi config held by the worker.
        decoder_config = Qwen2Config(**config.to_dict())
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.layers = torch.nn.ModuleList(
            Qwen2DecoderLayer(
                config=decoder_config,
                cache_config=vllm_config.cache_config,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, f"layers.{index}"),
            )
            for index in range(config.num_hidden_layers)
        )
        self.mimo_layers = torch.nn.ModuleList(
            Qwen2DecoderLayer(
                config=decoder_config,
                cache_config=vllm_config.cache_config,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, f"mimo_layers.{index}"),
            )
            for index in range(config.kimia_mimo_layers)
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mimo_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Both checkpoint heads span the FULL vocabulary. Official inference
        # also samples their full logits; do not slice by *_output_vocab.
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.mimo_output = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "mimo_output"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        # Ephemeral handoff for ONE execute_model -> sample_tokens call. The
        # histories themselves belong to the runner's per-request buffers.
        self._sampling_context = None
        self._audio_logits = None
        if config.use_whisper_feature:
            # Preserve official VQAdaptor indices, SiLU, and LayerNorm epsilon.
            self.vq_adaptor = torch.nn.Sequential(
                torch.nn.Linear(config.kimia_adaptor_input_dim, config.hidden_size),
                torch.nn.SiLU(),
                torch.nn.Dropout(0.0),
                torch.nn.Linear(config.hidden_size, config.hidden_size),
                torch.nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps),
            )
        self.input_encoder = KimiAudioInputEncoder(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "input_encoder")
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: "IntermediateTensors | None" = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Run the shared trunk and both branches on already-fused embeddings.

        Return [num_tokens, 2 * hidden_size], ordered [text | audio], so the
        runner's row selection keeps both branches aligned for compute_logits.
        Each vLLM attention layer owns its framework-managed KV-cache binding;
        this method neither creates a cache nor runs a generation loop.
        """
        if intermediate_tensors is not None:
            raise ValueError("Kimi-Audio dual-stream pipeline parallel partitioning is not implemented")
        if inputs_embeds is None:
            raise ValueError("Kimi-Audio forward requires fused text/audio inputs_embeds from preprocess")
        if (
            inputs_embeds.ndim != 2
            or inputs_embeds.shape[1] != self.config.hidden_size
            or positions.ndim != 1
            or positions.shape[0] != inputs_embeds.shape[0]
        ):
            raise ValueError("Kimi-Audio expects [num_tokens, hidden_size] embeddings and matching 1D positions")

        hidden_states = inputs_embeds
        residual = None
        mimo_hidden_states = mimo_residual = None
        for index, layer in enumerate(self.layers):
            hidden_states, residual = layer(positions, hidden_states, residual)
            if index == self.branch_layer:
                # The official model clones the completed layer output here.
                # vLLM carries it as TWO tensors until the next fused add/norm.
                # Clone both: later text layers may overwrite either in place.
                mimo_hidden_states = hidden_states.clone()
                mimo_residual = residual.clone()

        hidden_states, _ = self.norm(hidden_states, residual)
        assert mimo_hidden_states is not None and mimo_residual is not None
        for layer in self.mimo_layers:
            mimo_hidden_states, mimo_residual = layer(positions, mimo_hidden_states, mimo_residual)
        mimo_hidden_states, _ = self.mimo_norm(mimo_hidden_states, mimo_residual)
        return torch.cat((hidden_states, mimo_hidden_states), dim=-1)

    def make_omni_output(self, model_outputs: torch.Tensor, **kwargs: Any) -> "OmniOutput":
        """Bind request context outside the compiled/CUDA-graph forward."""
        from vllm_omni.model_executor.models.output_templates import OmniOutput

        infos = kwargs.get("model_intermediate_buffer")
        if not infos:
            return OmniOutput(text_hidden_states=model_outputs)
        if self._sampling_context is not None:
            raise RuntimeError("Kimi-Audio previous sampling context has not been consumed")
        extras = kwargs.get("sampling_extra_args", ())
        eligible = kwargs.get("request_sample_eligible", ())
        if len(infos) != len(extras) or len(infos) != len(eligible):
            raise ValueError("Kimi-Audio needs aligned request buffers, sampling_extra_args and eligibility")

        context = []
        for info, extra, can_sample in zip(infos, extras, eligible, strict=True):
            if not info.get("kimi_audio_request_validated"):
                raise ValueError("Kimi-Audio requires prepare_kimi_audio_request before engine admission")
            state = info.get("kimi_audio_generation")
            if state is None:
                raise ValueError("Missing Kimi-Audio generation state from preprocess")
            special = KimiAudioSpecialTokens(**state["special_tokens"])
            checkpoint_eos = (
                getattr(self.config, "eos_token_ids", None) or getattr(self.config, "eos_token_id", None) or []
            )
            checkpoint_eos = [checkpoint_eos] if isinstance(checkpoint_eos, int) else checkpoint_eos
            if special.kimia_text_blank in checkpoint_eos:
                raise ValueError("Kimi-Audio scheduler blank must not be a checkpoint EOS")
            if state["max_tokens"] is None:
                raise ValueError("Kimi-Audio needs the runner's _omni_max_tokens for final output delivery")
            context.append((state, extra, special, bool(can_sample)))
        # The runner retains this mapping through compute_logits/sample, then
        # snapshots it for output. sample fills ONLY this step's deltas before
        # that snapshot; a later batch gets a new mapping. Per-request lists
        # prevent token-span slicing from confusing batch rows with prefill.
        outputs = {
            "ids": {"output": [torch.empty(0, dtype=torch.long) for _ in context]},
            "codes": {"audio": [torch.empty(0, dtype=torch.long) for _ in context]},
            "meta": {"finished": [torch.tensor(False) for _ in context]},
        }
        self._sampling_context = (context, outputs)
        return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs=outputs)

    def compute_logits(
        self, hidden_states: torch.Tensor, sampling_metadata: "SamplingMetadata | None" = None
    ) -> torch.Tensor | None:
        """Expose normal [requests, vocab] text logits to the runner.

        The aligned audio logits are retained only until sample() consumes the
        same batch. Both heads use native vocabulary projection/TP gathering.
        """
        if hidden_states.ndim != 2 or hidden_states.shape[1] != 2 * self.config.hidden_size:
            raise ValueError("Kimi-Audio logits require packed [text | audio] hidden states")
        if self._audio_logits is not None:
            raise RuntimeError("Kimi-Audio previous audio logits have not been consumed")
        text_hidden, audio_hidden = hidden_states.chunk(2, dim=-1)
        text_logits = self.logits_processor(self.lm_head, text_hidden)
        audio_logits = self.logits_processor(self.mimo_output, audio_hidden)
        if text_logits is None or audio_logits is None:
            self._sampling_context = None
            return None
        self._audio_logits = audio_logits
        return text_logits

    def sample(self, logits: torch.Tensor, sampling_metadata: "SamplingMetadata") -> "SamplerOutput":
        """Sample once per eligible request and retain its actual dual-stream IDs.

        Scheduler IDs are blanks followed by msg_end, declared as a stop token
        at admission. Real text/audio deltas and the end marker go into this
        step's OmniOutput, including the final pair on stop or length limit.
        """
        from vllm.v1.outputs import SamplerOutput

        handoff, audio_logits = self._sampling_context, self._audio_logits
        self._sampling_context = self._audio_logits = None
        if handoff is None or audio_logits is None:
            raise ValueError("Kimi-Audio sample requires the matching request context and both logits")
        context, outputs = handoff
        if logits.shape != audio_logits.shape or len(context) != len(logits):
            raise ValueError("Kimi-Audio sample requires the matching request context and both logits")

        resolved = [
            KimiAudioSamplingParams.from_sampling_metadata(sampling_metadata, row, extra)
            for row, (_, extra, _, _) in enumerate(context)
        ]
        if any(max(params.text_top_k, params.audio_top_k) > self.config.vocab_size for params in resolved):
            raise ValueError("Kimi-Audio top_k exceeds the checkpoint vocabulary")
        pending, scheduler_tokens = [], []
        for row, ((state, _, special, eligible), params) in enumerate(zip(context, resolved, strict=True)):
            if not eligible:
                # Partial prefill/recomputation has no accepted output token.
                # In particular, it must not advance either history or RNG.
                scheduler_tokens.append(special.kimia_text_blank)
                continue
            if state["finished"]:
                raise ValueError("Kimi-Audio cannot sample an already finished request")
            generator = sampling_metadata.generators.get(row)
            own_generator = generator is None and state["seed"] is not None
            if own_generator:
                # Native text-greedy requests have no seeded sampler generator,
                # even when their audio override enables random sampling.
                generator = torch.Generator(device=logits.device).manual_seed(state["seed"])
                if state.get("rng_state") is not None:
                    generator.set_state(state["rng_state"])
            result = sample_kimi_audio_step(
                logits[row],
                audio_logits[row],
                text_history=state["text_history"],
                audio_history=state["audio_history"],
                text_finished=state["text_finished"],
                output_type=state["output_type"],
                special_tokens=special,
                audio_delay=self.config.kimia_mimo_audiodelaytokens,
                params=params,
                generator=generator,
            )
            token = special.msg_end if result.finished else special.kimia_text_blank
            scheduler_tokens.append(token)
            pending.append((row, state, result, token, generator.get_state() if own_generator else None))

        # Commit only after the whole batch sampled successfully.
        for row, state, result, token, rng_state in pending:
            state["text_history"].append(result.text_token)
            state["audio_history"].append(result.audio_token)
            state["scheduler_history"].append(token)
            generated = len(state["scheduler_history"])
            # Mirror the native length boundary for the payload's final flag.
            # Leave the scheduler token unchanged: native check_stop remains
            # responsible for reporting stopped vs length-capped completion.
            length_capped = (
                generated >= state["max_tokens"]
                or state["prompt_len"] + generated >= self.vllm_config.model_config.max_model_len
            )
            state["text_finished"] = result.text_finished
            state["finished"] = result.finished or length_capped
            # Official text trimming excludes EOS and the forced post-EOS
            # blanks. Audio stays in the LLM vocabulary here; the downstream
            # adapter will filter control tokens and remove kimia_token_offset.
            if not result.text_finished and result.text_token < self.config.kimia_token_offset:
                outputs["ids"]["output"][row] = torch.tensor([result.text_token], dtype=torch.long)
            if state["output_type"] == "both" and generated > self.config.kimia_mimo_audiodelaytokens:
                outputs["codes"]["audio"][row] = torch.tensor([result.audio_token], dtype=torch.long)
            outputs["meta"]["finished"][row] = torch.tensor(state["finished"])
            if rng_state is not None:
                state["rng_state"] = rng_state
        return SamplerOutput(
            sampled_token_ids=torch.tensor(scheduler_tokens, device=logits.device, dtype=torch.int32).reshape(-1, 1),
            logprobs_tensors=None,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from vllm.model_executor.models.qwen2 import Qwen2Model
        from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper

        mapper = Qwen2Model.hf_to_vllm_mapper | WeightsMapper(
            orig_to_new_prefix={"model.vq_adaptor.layers.": "vq_adaptor.", "model.": ""},
            # vLLM derives RoPE frequencies from config. Both MIMO and text
            # branch parameters must load; do not inherit the ASR-only skips.
            orig_to_new_suffix={".rotary_emb.inv_freq": None},
        )
        loaded = AutoWeightsLoader(self).load_weights(weights, mapper=mapper)

        settings = self.vllm_config.additional_config.get("kimi_audio", {})
        tokenizer_path = settings.get("glm_tokenizer_path")
        if tokenizer_path is None:
            from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

            tokenizer_path = download_weights_from_hf_specific(
                model_name_or_path="THUDM/glm-4-voice-tokenizer",
                revision="a5f2404e63c84e92f5238908e1706316324ebafa",
                cache_dir=self.vllm_config.load_config.download_dir,
                allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"],
                require_all=True,
            )
        loaded.update(f"input_encoder.{name}" for name in self.input_encoder.load_glm_weights(tokenizer_path))
        if self.config.use_whisper_feature:
            loaded.update(f"input_encoder.{name}" for name in self.input_encoder.load_whisper_weights())
        # The enclosing framework loader checks these names against ALL owned
        # parameters, including the newly materialized input encoders.
        return loaded

    @torch.inference_mode()
    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Prepare a scheduled span from prompt embeddings or accepted dual IDs.

        Full embeddings are cached in the runner's per-request buffer, never
        on the model instance. Absolute scheduler offsets also handle replay
        and one-token prefill tails. Replay can cross from the prompt into
        already generated tokens without advancing either stream's history.
        """
        import msgspec

        from vllm_omni.data_entry_keys import deserialize_payload
        from vllm_omni.engine import AdditionalInformationPayload

        generation = info_dict.get("kimi_audio_generation")
        if not info_dict.get("_omni_is_prefill", False) and generation is None:
            raise ValueError("Missing Kimi-Audio generation state for decode")
        offset = int(info_dict["_omni_num_computed_tokens"])
        prompt_len = int(info_dict["_omni_prompt_len"])
        end = offset + input_ids.numel()
        generated_len = len(generation["scheduler_history"]) if generation is not None else 0
        if input_ids.ndim != 1 or not 0 <= offset < end <= prompt_len + generated_len:
            raise ValueError("Invalid Kimi-Audio scheduled input span")

        cached_embeds = info_dict.get("embed", {}).get("prefill")
        cached_ids = info_dict.get("ids", {}).get("prompt")
        update = {}
        if cached_embeds is None and cached_ids is None:
            wire = info_dict.get("kimi_audio_input")
            if wire is None:
                raise ValueError("Missing Kimi-Audio prepared input; use prepare_kimi_audio_inputs")
            payload = deserialize_payload(msgspec.convert(wire, AdditionalInformationPayload))
            audio_ids = list(payload["audio_token_ids"])
            text_ids = payload["text_token_ids"]
            continuous = payload["is_continuous_mask"]
            if not len(audio_ids) == len(text_ids) == len(continuous) == prompt_len:
                raise ValueError("Kimi-Audio prepared input length differs from the scheduled prompt")
            if input_ids.tolist() != audio_ids[offset:end]:
                raise ValueError("Kimi-Audio scheduled IDs do not match the prepared prompt")

            features = []
            last_end = 0
            for index, start, stop in payload["audio_spans"]:
                if not last_end <= start < stop <= prompt_len:
                    raise ValueError("Invalid or overlapping Kimi-Audio audio spans")
                last_end = stop
                whisper_inputs = None
                if f"whisper_features_{index}" in payload:
                    whisper_inputs = KimiAudioWhisperInputs(
                        payload[f"whisper_features_{index}"], tuple(payload[f"whisper_lengths_{index}"])
                    )
                if continuous[start:stop] != [whisper_inputs is not None] * (stop - start):
                    raise ValueError("Kimi-Audio continuous mask does not match the audio span")
                encoded = self.input_encoder.encode_audio(
                    payload[f"waveform_{index}"].numpy(), sampling_rate=SAMPLE_RATE, whisper_inputs=whisper_inputs
                )
                if len(encoded.codes) != stop - start:
                    raise ValueError("Kimi-Audio encoder output does not fill its reserved token span")
                if any(
                    not 0 <= code < self.config.vocab_size - self.config.kimia_token_offset for code in encoded.codes
                ):
                    raise ValueError("Expected raw GLM codebook IDs before applying the Kimi offset")
                audio_ids[start:stop] = [code + self.config.kimia_token_offset for code in encoded.codes]
                if whisper_inputs is not None:
                    if encoded.continuous_features is None:
                        raise ValueError("Missing Kimi-Audio continuous features")
                    features.append(encoded.continuous_features)

            device = input_ids.device
            audio_embeds = self.embed_tokens(torch.tensor(audio_ids, device=device, dtype=torch.long))
            if features:
                mask = torch.tensor(continuous, device=device, dtype=torch.bool)
                whisper = torch.cat(features).to(device=device, dtype=audio_embeds.dtype)
                if whisper.shape != (sum(continuous), self.config.kimia_adaptor_input_dim):
                    raise ValueError("Kimi-Audio continuous features do not match the prompt mask")
                # Official fusion: only continuous audio positions receive the
                # projected Whisper feature and sqrt(2); then add text embeds.
                audio_embeds[mask] = (audio_embeds[mask] + self.vq_adaptor(whisper)) * audio_embeds.new_tensor(
                    2.0
                ).sqrt()
            elif any(continuous):
                raise ValueError("Kimi-Audio continuous positions have no features")
            full_embeds = audio_embeds + self.embed_tokens(torch.tensor(text_ids, device=device, dtype=torch.long))
            cached_embeds = full_embeds.detach().to("cpu").contiguous()
            cached_ids = audio_ids
            update = {"embed": {"prefill": cached_embeds}, "ids": {"prompt": cached_ids}}
            if generation is None:
                KimiAudioSpecialTokens(**payload["special_tokens"])
                if any(not 0 <= value < self.config.vocab_size for value in payload["special_tokens"].values()):
                    raise ValueError("Kimi-Audio special tokens exceed the checkpoint vocabulary")
                if payload["output_type"] not in ("text", "both"):
                    raise ValueError("Kimi-Audio requires output_type text/both")
                generation = {
                    "special_tokens": payload["special_tokens"],
                    "seed": info_dict.get("_omni_seed"),
                    "max_tokens": info_dict.get("_omni_max_tokens"),
                    "prompt_len": prompt_len,
                    "output_type": payload["output_type"],
                    "text_history": [],
                    "audio_history": [],
                    "scheduler_history": [],
                    "text_finished": False,
                    "finished": False,
                }
                update["kimi_audio_generation"] = generation

        if (
            not isinstance(cached_embeds, torch.Tensor)
            or cached_embeds.shape != (prompt_len, self.config.hidden_size)
            or not isinstance(cached_ids, list)
            or len(cached_ids) != prompt_len
        ):
            raise ValueError("Incomplete Kimi-Audio prefill cache")
        if generation is None or not (
            len(generation["text_history"]) == len(generation["audio_history"]) == len(generation["scheduler_history"])
        ):
            raise ValueError("Incomplete Kimi-Audio dual-stream history")

        ids, embeds = [], []
        if offset < prompt_len:
            stop = min(end, prompt_len)
            ids.extend(cached_ids[offset:stop])
            embeds.append(cached_embeds[offset:stop].to(device=input_ids.device))
        if end > prompt_len:
            start, stop = max(0, offset - prompt_len), end - prompt_len
            scheduled = input_ids[max(0, prompt_len - offset) :].tolist()
            if scheduled != generation["scheduler_history"][start:stop]:
                raise ValueError("Kimi-Audio scheduled decode IDs do not match accepted history")
            audio = generation["audio_history"][start:stop]
            text = generation["text_history"][start:stop]
            ids.extend(audio)
            embeds.append(
                self.embed_tokens(torch.tensor(audio, device=input_ids.device, dtype=torch.long))
                + self.embed_tokens(torch.tensor(text, device=input_ids.device, dtype=torch.long))
            )
        return (
            torch.tensor(ids, device=input_ids.device, dtype=input_ids.dtype),
            torch.cat(embeds),
            update,
        )
