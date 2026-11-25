# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LFM3-Omni temporal model and MaskGIT depth head in one AR stage.

Depth frames feed the next temporal step, so they cannot be moved to an
independent downstream talker. Only waveform decoding is a separate stage.
"""

import importlib
from collections.abc import Iterable
from types import ModuleType

import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.generation.logits_process import TemperatureLogitsWarper, TopKLogitsWarper
from vllm.config import VllmConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.utils import _merge_multimodal_embeddings, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .backbone import Lfm3TemporalModel, RequestState
from .processing import Lfm3OmniDummyInputsBuilder, Lfm3OmniMultiModalProcessor, Lfm3OmniProcessingInfo
from .reference import reference_module


@MULTIMODAL_REGISTRY.register_processor(
    Lfm3OmniMultiModalProcessor, info=Lfm3OmniProcessingInfo, dummy_inputs=Lfm3OmniDummyInputsBuilder
)
class Lfm3OmniForConditionalGeneration(nn.Module, SupportsMultiModal):
    have_multimodal_outputs = True
    has_preprocess = True
    has_postprocess = False
    omni_pooler_payload_include_hidden = False
    # Checkpoint module and class that define the encoders and depth head.
    _reference = ("modeling_lfm3_omni", "Lfm3OmniModel")

    def _build_reference_modules(self, ref: ModuleType, c: PretrainedConfig) -> None:
        self.model.audio_encoder = ref.Lfm3OmniEncoder(c.audio_encoder_config)
        self.model.audio_projector = ref.Lfm3OmniProjector(c)
        self.model.vision_tower = ref.Lfm3VlVisionModel(c.vision_config)
        self.model.multi_modal_projector = ref.Lfm3OmniMultiModalProjector(c)
        self.model.depth_decoder = ref.Lfm3OmniDepthDecoder(c)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        mc = vllm_config.model_config
        if vllm_config.parallel_config.tensor_parallel_size != 1:
            raise ValueError("LFM3-Omni currently requires tensor_parallel_size=1")
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("LFM3-Omni currently requires pipeline_parallel_size=1")
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("LFM3-Omni convolution and audio feedback require enable_prefix_caching=False")
        if not mc.enforce_eager:
            raise ValueError("LFM3-Omni request-owned convolution currently requires enforce_eager=True")
        if vllm_config.scheduler_config.enable_chunked_prefill:
            raise ValueError("LFM3-Omni audio feedback currently requires enable_chunked_prefill=False")
        self.config = c = mc.hf_config
        self._supports_mm_inputs = mc.supports_multimodal_inputs
        ref = reference_module(mc, *self._reference)
        backbone_ref = importlib.import_module(ref.Lfm3Model.__module__)
        self.model = nn.Module()
        self.model.language_model = Lfm3TemporalModel(
            vllm_config, backbone_ref, maybe_prefix(prefix, "model.language_model")
        )
        device = self.model.language_model.embed_tokens.weight.device
        # Match Transformers' construction of non-persistent rotary buffers
        # in every reference component, not only the temporal backbone.
        with torch.device("cpu"):
            self._build_reference_modules(ref, c)
        for name, component in self.model.named_children():
            if name != "language_model":
                component.to(device)
        self.model.codebook_embedding = nn.Embedding(c.num_codebooks * c.codebook_size, c.text_config.hidden_size)
        # All encoder/depth attention is bidirectional or explicitly masked by
        # the reference. Do not replace it with causal PagedAttention.
        for subconfig in (c.audio_encoder_config.text_config, c.depth_config, getattr(c, "vision_config", None)):
            if subconfig is not None:
                subconfig._attn_implementation = "eager"
        # Media placeholders are never sampled as text.
        self._placeholder_token_ids = [
            getattr(c, name)
            for name in ("image_token_id", "video_token_id", "audio_token_id", "audio_out_token_id")
            if hasattr(c, name)
        ]
        # Like the reference's generate(), speech settings a request omits come
        # from the checkpoint's generation config (they differ per checkpoint).
        generation = mc.try_get_generation_config()
        self._speech_defaults = {
            name: generation[name]
            for name in ("audio_temperature", "audio_top_k", "audio_repetition_penalty", "audio_repetition_window")
            if generation.get(name) is not None
        }
        self.lm_head = nn.Linear(c.text_config.hidden_size, c.text_config.vocab_size, bias=False)
        self.lm_head.weight = self.model.language_model.embed_tokens.weight
        self._depth_graph_enabled = bool(getattr(c, "lfm3_depth_graph", False))
        self._vectorized_greedy = bool(getattr(c, "lfm3_vectorized_greedy", False))
        self._depth_graph = None
        self._states: dict[str, RequestState] = {}
        self._forced_tokens: list[int | None] = []
        self._batch_states: list[RequestState] = []

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str:
        if modality.startswith("image"):
            return "<|image|>"
        if modality.startswith("video"):
            return "<|video|>"
        if modality.startswith("audio"):
            return "<|audio|>"
        raise ValueError(f"Unsupported modality: {modality}")

    def embed_input_ids(self, input_ids, multimodal_embeddings=None, is_multimodal=None):
        embeddings = self.model.language_model.embed_tokens(input_ids)
        if multimodal_embeddings is not None:
            embeddings = _merge_multimodal_embeddings(embeddings, multimodal_embeddings, is_multimodal=is_multimodal)
        return embeddings

    def get_language_model(self):
        return self

    def embed_multimodal(self, **kwargs):
        embeddings = []
        # Preserve the modality ordering used by the engine's encoder batch.
        for key in kwargs:
            if key in ("pixel_values", "pixel_values_videos"):
                grid_key = "image_grid_thw" if key == "pixel_values" else "video_grid_thw"
                grid = kwargs[grid_key]
                values = kwargs[key]
                if isinstance(values, list):
                    values = torch.cat(values)
                vision = self.model.vision_tower(
                    values.to(next(self.model.vision_tower.parameters()).dtype),
                    grid_thw=grid.to(values.device),
                    return_dict=True,
                ).last_hidden_state
                projected = self.model.multi_modal_projector(vision)
                embeddings.extend(projected.split((grid.prod(-1) // self.config.downsample_factor**2).tolist()))
            elif key == "input_features":
                features, mask = kwargs[key], kwargs["feature_attention_mask"]
                if isinstance(features, list):
                    # Variable-duration clips are independent encoder requests.
                    pairs = zip(features, mask)
                else:
                    pairs = zip(features.unbind(), mask.unbind())
                for feature, valid in pairs:
                    encoded, lengths = self.model.audio_encoder(
                        feature.unsqueeze(0).to(self.model.audio_encoder.downsample.conv_in.weight.dtype),
                        valid.sum().reshape(1),
                    )
                    embeddings.append(self.model.audio_projector(encoded[0, : int(lengths[0])]))
        return embeddings or None

    def _advance_interleaved(self, state: RequestState) -> None:
        if state.mode == "interleaved" and not state.text_done:
            state.slots_left -= 1
            if state.slots_left == 0:
                state.speaking = not state.speaking
                state.slots_left = self.config.interleaved_n_audio if state.speaking else self.config.interleaved_n_text

    def preprocess(self, input_ids, input_embeds, **info):
        req_id = info["request_id"]
        is_prefill = info["_omni_is_prefill"]
        state = self._states.get(req_id)
        if state is None:
            state = RequestState(slots_left=self.config.interleaved_n_text, **self._speech_defaults)
            state.generator = torch.Generator(device=input_ids.device)
            seed = info.get("_omni_seed")
            state.generator.manual_seed(0 if seed is None else seed)
            self._states[req_id] = state
        token = int(input_ids[-1])
        if is_prefill:
            if token in (self.config.audio_start_token_id, self.config.audio_out_token_id):
                state.mode = "audio"
            elif token == self.config.mixed_start_token_id:
                state.mode = "interleaved"
        elif not state.last_was_audio:
            if state.mode == "interleaved":
                if token == self.config.text_end_token_id:
                    state.text_done = state.speaking = True
                self._advance_interleaved(state)
            elif token == self.config.text_end_token_id:
                state.mode = "finished"
            elif token == self.config.audio_start_token_id:
                state.mode = "audio"
            elif token == self.config.mixed_start_token_id:
                state.mode = "interleaved"
                state.slots_left = self.config.interleaved_n_text
                state.speaking = state.text_done = False
        # With all modality limits set to zero the runner passes an unfilled
        # scratch buffer to preprocess. Only the multimodal route embeds it.
        if input_embeds is None or not self._supports_mm_inputs:
            input_embeds = self.embed_input_ids(input_ids)
        if not is_prefill and state.next_audio_embed is not None:
            input_embeds = state.next_audio_embed.reshape(1, -1)
            state.next_audio_embed = None
        return input_ids, input_embeds, {"lfm3_span": len(input_ids), "request_id": req_id}

    def _depth_logits(self, hidden, codes, masked):
        # Sampling stays outside the graph: replay must not change request RNG
        # progression or the MaskGIT reveal order.
        if not self._depth_graph_enabled or hidden.device.type != "npu":
            return self.model.depth_decoder(hidden, codes, masked)
        if self._depth_graph is None:
            from vllm_omni.platforms.npu.graph_tools import NPUExactGraphRunner

            self._depth_graph = NPUExactGraphRunner(
                max_graphs=1,
                component_name="LFM3-Omni depth decoder",
                disable_config_hint="set hf_overrides.lfm3_depth_graph=false",
            )
        return self._depth_graph.run(
            "depth",
            (hidden, codes, masked),
            (),
            lambda h, c, m: (self.model.depth_decoder(h, c, m),),
        )[0]

    def _audio_frame(self, hidden: torch.Tensor, state: RequestState) -> torch.Tensor:
        c = self.config
        codes = torch.zeros((1, c.num_codebooks), dtype=torch.long, device=hidden.device)
        revealed = torch.zeros_like(codes, dtype=torch.bool)
        order = torch.randperm(c.num_codebooks, generator=state.generator, device=hidden.device)
        history = torch.stack(state.audio_history) if state.audio_history else None
        for pick in order.split(c.maskgit_schedule):
            logits = self._depth_logits(hidden[None], codes, ~revealed)[0].float()
            temperature, top_k = state.audio_temperature, state.audio_top_k
            greedy = temperature == 0 or (temperature is None and not top_k)
            if self._vectorized_greedy and greedy:
                scores = logits[pick]
                if history is not None:
                    ids = history[:, pick].transpose(0, 1)
                    previous = scores.gather(1, ids)
                    penalty = state.audio_repetition_penalty
                    scores.scatter_(1, ids, torch.where(previous < 0, previous * penalty, previous / penalty))
                codes[0, pick] = scores.argmax(-1)
                revealed[:, pick] = True
                continue
            for index in pick.tolist():
                scores = logits[index]
                if history is not None:
                    ids = history[:, index]
                    previous = scores[ids]
                    penalty = state.audio_repetition_penalty
                    scores[ids] = torch.where(previous < 0, previous * penalty, previous / penalty)
                if greedy:
                    sampled = scores.argmax(-1)
                else:
                    scores = scores[None]
                    if temperature:
                        scores = TemperatureLogitsWarper(float(temperature))(None, scores)
                    if top_k:
                        scores = TopKLogitsWarper(int(top_k))(None, scores)
                    sampled = torch.multinomial(scores.softmax(-1), 1, generator=state.generator)[0, 0]
                codes[0, index] = sampled
            revealed[:, pick] = True
        return codes[0]

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        infos = kwargs.get("runtime_additional_information") or []
        requests, self._batch_states = [], []
        start = 0
        extras = kwargs.get("sampling_extra_args") or [{} for _ in infos]
        for info, extra in zip(infos, extras, strict=True):
            state = self._states[info["request_id"]]
            for name in ("audio_temperature", "audio_top_k", "audio_repetition_penalty", "audio_repetition_window"):
                if name in extra:
                    setattr(state, name, extra[name])
            if state.audio_repetition_penalty <= 0 or state.audio_repetition_window < 1:
                raise ValueError("Audio repetition penalty and window must be positive")
            state.text_only = extra.get("generation_mode") == "text"
            end = start + info["lfm3_span"]
            requests.append((start, end, state))
            self._batch_states.append(state)
            start = end
        x = self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds
        hidden = self.model.language_model(x, positions, requests)
        self._forced_tokens = []
        frames = []
        for start, end, state in requests:
            is_audio = state.mode == "audio" or (state.mode == "interleaved" and state.speaking)
            state.last_was_audio = is_audio
            if is_audio:
                frame = self._audio_frame(hidden[end - 1], state)
                eos = bool((frame == self.config.audio_eos_code).any())
                if eos:
                    frame = torch.full_like(frame, self.config.audio_eos_code)
                    state.mode = "text"
                else:
                    state.audio_history.append(frame)
                    # Match the reference's default repetition window.
                    state.audio_history = state.audio_history[-state.audio_repetition_window :]
                offsets = torch.arange(self.config.num_codebooks, device=frame.device) * self.config.codebook_size
                state.next_audio_embed = self.model.codebook_embedding(frame + offsets).sum(0)
                frames.append(frame[None] if not eos else frame.new_empty((0, self.config.num_codebooks)))
                self._forced_tokens.append(self.config.audio_out_token_id)
                self._advance_interleaved(state)
            else:
                frames.append(torch.empty((0, self.config.num_codebooks), device=hidden.device, dtype=torch.long))
                self._forced_tokens.append(self.config.text_config.eos_token_id if state.mode == "finished" else None)
        if not infos:
            return hidden
        return OmniOutput(text_hidden_states=hidden, multimodal_outputs={"codes": {"audio": frames}})

    def compute_logits(self, hidden_states):
        logits = self.lm_head(hidden_states).float()
        c = self.config
        logits[:, self._placeholder_token_ids] = -torch.inf
        for row, forced in enumerate(self._forced_tokens):
            if self._batch_states[row].text_only:
                logits[row, [c.audio_start_token_id, c.mixed_start_token_id]] = -torch.inf
            if forced is not None:
                logits[row].fill_(-torch.inf)
                logits[row, forced] = 0
        return logits

    def on_requests_finished(self, finished_req_ids):
        for req_id in finished_req_ids:
            state = self._states.pop(req_id, None)
            if state is not None:
                # The last batch can still hold the state object while idle.
                state.conv.clear()
                state.kv.clear()
                state.audio_history.clear()
                state.next_audio_embed = None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded = set()
        for name, tensor in weights:
            if name == "lm_head.weight":
                name = "model.language_model.embed_tokens.weight"
            if name not in params:
                raise ValueError(f"Unexpected LFM3-Omni checkpoint parameter: {name}")
            default_weight_loader(params[name], tensor)
            loaded.add(name)
        missing = set(params) - loaded
        if missing:
            raise ValueError(f"Missing LFM3-Omni checkpoint parameters: {sorted(missing)}")
        return loaded
