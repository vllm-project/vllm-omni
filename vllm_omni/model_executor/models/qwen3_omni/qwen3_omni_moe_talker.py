from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
)
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import _ACTIVATION_REGISTRY
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniThinkerDummyInputsBuilder,
)
from vllm.model_executor.models.qwen3_moe import Qwen3MoeMLP
from vllm.model_executor.models.qwen3_omni_moe_thinker import Qwen3Omni_VisionTransformer
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
    sequence_parallel_chunk,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_code_predictor_mtp import (
    Qwen3OmniMoeTalkerCodePredictor,
)
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3MoeLLMForCausalLM,
    Qwen3OmniMoeConditionalGenerationMixin,
    Qwen3OmniMoeThinkerMultiModalProcessor,
    Qwen3OmniMoeThinkerProcessingInfo,
)

try:
    import flash_attn
except (ImportError, ModuleNotFoundError):
    flash_attn = None

from vllm_omni.model_executor.models.utils import safe_tensor_reshape

logger = init_logger(__name__)

Qwen3OmniMoeThinkerDummyInputsBuilder = Qwen2_5OmniThinkerDummyInputsBuilder


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3OmniMoeThinkerMultiModalProcessor,
    info=Qwen3OmniMoeThinkerProcessingInfo,
    dummy_inputs=Qwen3OmniMoeThinkerDummyInputsBuilder,
)
class Qwen3OmniMoeTalkerForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
    Qwen3OmniMoeConditionalGenerationMixin,
):
    """
    Qwen3 Omni MoE Talker - Converts text to audio codec codes.

    The talker is the second stage of Qwen3 Omni MoE's TTS pipeline:
    1. Thinker: Generates text response + hidden states
    2. Talker: Converts those to 8-layer audio codec codes
    3. Code2Wav: Converts codes to waveform

    ## Key Components:
    - text_projection: Projects thinker text embeddings → talker dimension
    - hidden_projection: Projects thinker hidden states → talker dimension
    - language_model: Main MoE transformer (generates layer 0)
    - codec_head: Projects to codec vocabulary (layer 0 logits)
    - code_predictor: Small transformer for layers 1-num_layers-1
    """

    logger = init_logger(__name__)

    # Weight mapping from HuggingFace to vLLM naming convention
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # Main MoE transformer model
            "talker.model.": "language_model.model.",
            # Codec head remains separate (outputs audio codes, not text)
            "talker.codec_head.": "codec_head.",
            # Code predictor: Now matches HF structure exactly (has .model sub-module)
            # e.g., "talker.code_predictor.model.codec_embedding.0" → "code_predictor.model.codec_embedding.0"
            "talker.code_predictor.": "code_predictor.",
            # Projection layers
            "talker.text_projection.": "text_projection.",
            "talker.hidden_projection.": "hidden_projection.",
            # Fallback: strip talker prefix
            "talker.": "",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        talker_config: Qwen3OmniMoeTalkerConfig = vllm_config.model_config.hf_config
        talker_config.text_config.rope_parameters = talker_config.text_config.rope_scaling
        talker_config.text_config.rope_parameters["rope_theta"] = talker_config.text_config.rope_theta
        self.quant_config = vllm_config.quant_config
        self.prefix = prefix
        self.vllm_config = vllm_config
        self.config = talker_config
        self.vocab_size = talker_config.text_config.vocab_size
        self.router_aux_loss_coef = talker_config.text_config.router_aux_loss_coef
        self.num_experts = talker_config.text_config.num_experts
        self.num_experts_per_tok = talker_config.text_config.num_experts_per_tok
        # thinker projection components for talker
        self.text_projection = Qwen3OmniMoeTalkerResizeMLP(self.config)
        self.hidden_projection = Qwen3OmniMoeTalkerResizeMLP(self.config)
        self.codec_head = nn.Linear(self.config.text_config.hidden_size, self.config.text_config.vocab_size, bias=False)

        self.rope_deltas = None
        self.spatial_merge_size = self.config.spatial_merge_size

        self.vocab_size = self.config.code_predictor_config.vocab_size
        self.num_code_groups = self.config.code_predictor_config.num_code_groups

        self.language_model = Qwen3OmniMoeModel(
            vllm_config=vllm_config,
            talker_config=self.config,
            prefix=maybe_prefix(prefix, "language_model"),
        )

        self.code_predictor = Qwen3OmniMoeTalkerCodePredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "code_predictor")
        )

    def code_predictor_forward(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        *,
        temperature: float = 1.0,
        top_k: int = 50,  # Match transformers default
        top_p: float = 0.8,  # Match transformers default
        generation_steps: int | None = None,
        last_talker_hidden: torch.Tensor | None = None,
        **_: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate full RVQ codec codes for the provided sequence.

        The code predictor consumes the layer-0 codec codes produced by the talker
        alongside the talker's hidden states, and autoregressively predicts the remaining
        residual layers (to num_codec_groups).

        Returns:
            tuple containing:
                - residual_codes: A tensor of shape [batch, num_code_groups, seq_len] containing
                  the complete set of codec codes
                - summed_embeddings: A tensor of shape [batch, seq_len, hidden_size]
                  Sum of all layer embeddings at each position (like Transformers)
        """
        if input_ids is None:
            raise ValueError("`input_ids` containing layer-0 codec codes must be provided.")
        if inputs_embeds is None:
            raise ValueError("`inputs_embeds` containing talker hidden states must be provided.")

        if inputs_embeds.ndim == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        # Ensure the tensors are contiguous for the autoregressive sampling loop
        inputs_embeds = inputs_embeds.contiguous()
        input_ids = input_ids.contiguous()

        # Generate full codec codes using MTP
        # This will be the parallel prediction implementation
        batch_size, seq_len = input_ids.shape

        # For now, use sequential generation (TODO: implement parallel)
        # Result will be [batch, num_code_groups, seq_len]
        # - all_codes_per_position will collect [batch, num_code_groups, 1] for each position
        all_codes_per_position = []
        middle_hidden_states = []  # Collect hidden states for each position

        # Generate residual layers for each position
        for pos in range(seq_len):
            layer0_code = input_ids[:, pos : pos + 1]  # [batch, 1]

            # Predict all residual layers (layers 1 to num_code_groups-1) autoregressively
            pos_codes = [layer0_code]  # Start with layer 0: [batch, 1]

            # Initial input: [last_talker_hidden, layer0_embed]
            layer0_embed = self.embed_input_ids(layer0_code)
            prev_embed = layer0_embed  # Track previous layer embedding
            try:
                current_input = torch.cat([last_talker_hidden, prev_embed], dim=1)  # [batch, 2, hidden_size]
            except Exception as e:
                print(f"Error in current_input: {e}")
                print(f"last_talker_hidden shape: {last_talker_hidden.shape}")
                print(f"prev_embed shape: {prev_embed.shape}")
                raise e

            for layer_idx in range(self.num_code_groups - 1):
                # Input for this layer: [last_talker_hidden, prev_layer_embed]

                # Forward through code_predictor model
                outputs = self.code_predictor.model(
                    inputs_embeds=current_input,
                    attention_mask=None,
                    position_ids=None,
                    past_key_values=None,
                    use_cache=False,
                    cache_position=None,
                )

                hidden_state = outputs.last_hidden_state  # [batch, 2, hidden_size]

                # Use the corresponding lm_head for this layer
                logits = self.code_predictor.lm_head[layer_idx](hidden_state[:, -1:, :])  # [batch, 1, vocab_size]
                from transformers.generation.logits_process import (
                    LogitsProcessorList,
                    TopKLogitsWarper,
                    TopPLogitsWarper,
                )

                logits_processors = LogitsProcessorList(
                    [
                        TopKLogitsWarper(top_k=top_k),
                        TopPLogitsWarper(top_p=top_p),
                    ]
                )
                input_ids_for_logits_processors = torch.tensor([pos_codes[1:]]).to(logits.device, dtype=torch.long)
                logits = logits_processors(input_ids_for_logits_processors, logits.squeeze(0)).unsqueeze(0)

                # Sample from the filtered distribution
                probs = F.softmax(logits, dim=-1)
                code = torch.multinomial(probs.squeeze(1), num_samples=1)  # [batch, 1]
                pos_codes.append(code)

                # Update prev_embed for next layer (if not last layer)
                # layer_idx=0 predicts layer 1, embed with codec_embedding[1]
                new_embed = self.code_predictor.model.codec_embedding[layer_idx](code)  # [batch, 1, hidden_size]
                current_input = torch.cat([current_input, new_embed], dim=1)  # [batch, 3~n, hidden_size]

            # Stack all layers for this position: [batch, num_code_groups, 1]
            pos_all_layers = torch.stack(pos_codes, dim=1)  # [batch, num_code_groups, 1]
            all_codes_per_position.append(pos_all_layers)
            middle_hidden_states.append(current_input[:, 2:-1, :])

        # Concatenate across positions: [batch, num_code_groups, seq_len]
        result_codes = torch.cat(all_codes_per_position, dim=2)

        # Build summed embeddings for each position (like Transformers)
        # This combines layer-0 embed, mid layers hidden states, and last layer embed
        all_summed_embeddings = []

        for pos in range(seq_len):
            # Layer 0 embedding
            layer0_code = result_codes[:, 0, pos : pos + 1]  # [batch, 1]
            layer0_embed = self.embed_input_ids(layer0_code)  # [batch, 1, hidden_size]

            # mid layers hidden states (from CodePredictor)
            mid_residual_hiddens = middle_hidden_states[pos]  # [batch, num_code_groups-2, hidden_size]
            mid_list = list(mid_residual_hiddens.split(1, dim=1))

            # last layer embedding
            last_layer_code = result_codes[:, -1, pos : pos + 1]  # [batch, 1]
            last_residual_hidden = self.code_predictor.model.codec_embedding[-1](last_layer_code)

            # Concatenate all layers: [batch, num_code_groups, hidden_size]
            pos_codec_hiddens = torch.cat(
                [layer0_embed] + mid_list + [last_residual_hidden],
                dim=1,
            )

            # Sum across layers: [batch, 1, hidden_size] (like Transformers)
            pos_summed = pos_codec_hiddens.sum(dim=1, keepdim=True)
            all_summed_embeddings.append(pos_summed)

        # Concatenate across positions: [batch, seq_len, hidden_size]
        summed_embeddings = torch.cat(all_summed_embeddings, dim=1)

        return result_codes, summed_embeddings

    def init_multi_modal(self, thinker_config: Any) -> None:
        """
        Initialize multimodal components from the thinker.

        Unlike Qwen2.5 Omni which creates audio_tower and visual encoders here,
        Qwen3 Omni MoE has a cleaner separation: the thinker is the ONLY module
        that processes raw multimodal inputs. The talker only handles text-to-audio
        conversion using pre-processed embeddings from the thinker.

        This method exists for API compatibility and stores the thinker config
        for reference. The actual multimodal processing components (audio_tower,
        visual) are ONLY in the thinker, not duplicated in the talker.

        Args:
            thinker_config: Configuration from the thinker model (for reference only)
        """
        self.audio_tower = Qwen3OmniMoeAudioEncoder(thinker_config.audio_config)
        self.visual = Qwen3Omni_VisionTransformer(
            vision_config=thinker_config.vision_config,
            norm_eps=getattr(thinker_config.text_config, "rms_norm_eps", 1e-6),
            quant_config=self.quant_config,
            prefix=maybe_prefix(self.prefix, "visual"),
            # attn_backend_override=attn_backend_override,
        )

    def project_thinker_outputs(
        self,
        thinker_embeds: torch.Tensor | None = None,
        thinker_hidden_states: torch.Tensor | None = None,
        is_multimodal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Project thinker outputs to talker's hidden dimension.

        The talker has a different hidden size than the thinker, so we need
        to project the inputs appropriately:
        - Text embeddings (from thinker's embedding layer) → text_projection
        - Hidden states (from thinker's last layer, for multimodal) → hidden_projection

        Args:
            thinker_embeds: Text embeddings from thinker [batch, seq, thinker_hidden]
            thinker_hidden_states: Hidden states from thinker's last layer [batch, seq, thinker_hidden]
            is_multimodal_mask: Boolean mask indicating multimodal positions [batch, seq]

        Returns:
            projected_embeds: [batch, seq, talker_hidden]
        """
        if thinker_embeds is None and thinker_hidden_states is None:
            raise ValueError("Either thinker_embeds or thinker_hidden_states must be provided")

        # If only embeddings provided, project all as text
        if thinker_hidden_states is None or is_multimodal_mask is None:
            return self.text_projection(thinker_embeds)

        # If only hidden states provided, project all as hidden
        if thinker_embeds is None:
            return self.hidden_projection(thinker_hidden_states)

        # Mixed case: use mask to decide which projection
        batch_size, seq_len, _ = thinker_embeds.shape
        output = torch.empty(
            (batch_size, seq_len, self.config.text_config.hidden_size),
            device=thinker_embeds.device,
            dtype=thinker_embeds.dtype,
        )

        # Project multimodal regions using hidden states
        if is_multimodal_mask.any():
            mm_hidden = thinker_hidden_states[is_multimodal_mask]
            projected_mm = self.hidden_projection(mm_hidden)
            output[is_multimodal_mask] = projected_mm

        # Project text regions using embeddings
        if (~is_multimodal_mask).any():
            text_embeds = thinker_embeds[~is_multimodal_mask]
            projected_text = self.text_projection(text_embeds)
            output[~is_multimodal_mask] = projected_text

        return output

    def forward(
        self,
        batched_talker_inputs: dict,
        intermediate_tensors: IntermediateTensors | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Forward pass through the talker model.

        For inference, the talker receives inputs_embeds that should already be
        projected to talker's hidden dimension. If receiving raw thinker outputs,
        use project_thinker_outputs() first.
        """
        # If intermediate_tensors is provided (pipeline parallel),
        # inputs_embeds should be None
        batched_input_ids = []
        batched_positions = []
        batched_input_embeds = []
        batched_code_predictor_codes = []
        if intermediate_tensors is not None:
            inputs_embeds = None
        for rid in batched_talker_inputs:
            input_ids = safe_tensor_reshape(batched_talker_inputs[rid]["input_ids"], (1, -1))
            positions = batched_talker_inputs[rid]["positions"]
            inputs_embeds = safe_tensor_reshape(
                batched_talker_inputs[rid]["inputs_embeds"], (-1, self.config.text_config.hidden_size)
            )
            text_step = safe_tensor_reshape(batched_talker_inputs[rid]["text_step"], (1, -1))
            last_talker_hidden = safe_tensor_reshape(
                batched_talker_inputs[rid]["last_talker_hidden"], (1, 1, self.config.text_config.hidden_size)
            )
            # for profiling
            if inputs_embeds.shape[-1] == 2048:
                inputs_embeds = self.text_projection(inputs_embeds)
            if inputs_embeds.shape[0] == 1:
                code_predictor_codes, summed_embeddings = self.code_predictor_forward(
                    input_ids, inputs_embeds.clone(), last_talker_hidden=last_talker_hidden
                )
                inputs_embeds = summed_embeddings.clone()
            else:
                code_predictor_codes = torch.zeros((0, self.num_code_groups), dtype=torch.long)
            batched_input_ids.append(input_ids.reshape(-1))
            batched_positions.append(positions)
            batched_input_embeds.append((inputs_embeds + text_step).reshape(-1, self.config.text_config.hidden_size))
            batched_code_predictor_codes.append(code_predictor_codes.squeeze(-1).detach().to("cpu").contiguous())
        try:
            talker_input_ids = torch.cat(batched_input_ids, dim=0)
            talker_positions = torch.cat(batched_positions, dim=0)
            talker_input_embeds = torch.cat(batched_input_embeds, dim=0)
        except Exception as e:
            print(f"Error in talker_input_embeds: {e}")
            print(f"talker_input_embeds shape: {[tensor.shape for tensor in batched_input_embeds]}")
            print(f"talker_input_ids shape: {[tensor.shape for tensor in batched_input_ids]}")
            print(f"talker_positions shape: {[tensor.shape for tensor in batched_positions]}")
            raise e
        talker_hidden_states, _ = self.language_model.model(
            talker_input_ids,
            talker_positions,
            intermediate_tensors,
            inputs_embeds=talker_input_embeds,
            **kwargs,
        )

        # Pass talker hidden states to code predictor
        # Returns: (residual_codes, summed_embeddings=sum of all layer embeddings at each position)

        # Return both talker hidden states and code predictor results
        # code_predictor_codes: [batch, num_code_groups, seq_len]
        # summed_embeddings: [batch, seq_len, hidden_size]
        #   - Sum of all layer embeddings at each position (like Transformers)

        return talker_hidden_states, batched_code_predictor_codes

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute logits for audio codec codes (layer 0 of RVQ).

        This projects the hidden states to the codec vocabulary space.
        For full audio generation, layers except 0 would be predicted by
        the code_predictor after sampling.
        """
        logits = self.codec_head(hidden_states)
        return logits

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        """Create empty intermediate tensors for pipeline parallelism."""
        return self.language_model.make_empty_intermediate_tensors(batch_size, dtype, device)

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if input_key in ("pixel_values", "image_embeds") and "image" not in mm_input_by_modality:
                mm_input_by_modality["image"] = self._parse_and_validate_image_input(**kwargs)
            if input_key in ("pixel_values_videos", "video_embeds") and "video" not in mm_input_by_modality:
                mm_input_by_modality["video"] = self._parse_and_validate_video_input(**kwargs)
            if input_key in ("input_audio_features") and "audio" not in mm_input_by_modality:
                mm_input_by_modality["audio"] = self._parse_and_validate_audio_input(**kwargs)
        return mm_input_by_modality

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return []

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        # TODO: do projection for all multimodel
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                image_embeddings = self._process_image_input(multimodal_input)
                image_embeddings = self.hidden_projection(image_embeddings)
                multimodal_embeddings += tuple(image_embeddings)
            if modality == "video":
                video_embeddings = self._process_video_input(multimodal_input)
                video_video_embeddings_project = ()
                for video_embed in video_embeddings:
                    proj = nn.Linear(8192, 2048).to(device=video_embed.device, dtype=torch.bfloat16)
                    video_embed = proj(video_embed)
                    video_embed_project = self.hidden_projection(video_embed)
                    video_video_embeddings_project += (video_embed_project,)
                multimodal_embeddings += tuple(video_video_embeddings_project)
            if modality == "audio":
                audio_embeddings = self._process_audio_input(multimodal_input)
                audio_embeddings = self.hidden_projection(audio_embeddings)
                multimodal_embeddings += tuple(audio_embeddings)
        return multimodal_embeddings

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        is_multimodal: bool = False,
    ):
        """Get the input embedding layer (for codec tokens)."""
        return self.language_model.embed_input_ids(input_ids)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights for the talker model.

        The weight mapping translates from HuggingFace naming convention
        to vLLM's internal structure. Code predictor weights are routed
        to its custom loader for vocab extension support.
        """
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["thinker.", "code2wav."],
            # "code_predictor."],
        )
        # Don't apply mapper again since we already did it
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

        # Log load summary
        try:
            total_bytes = 0
            for name, param in self.named_parameters():
                if param is not None and param.data is not None:
                    total_bytes += param.data.numel() * param.data.element_size()
            device = next(self.parameters()).device
            logger.info(
                "[Model Loaded] name=%s, success=%s, size=%.2f MB, device=%s",
                self.__class__.__name__,
                True,
                total_bytes / (1024**2),
                str(device),
            )
        except Exception:
            pass

        multi_model_weights = set()
        for name, param in self.visual.named_parameters():
            multi_model_weights.add("visual." + name)
        for name, param in self.audio_tower.named_parameters():
            multi_model_weights.add("audio_tower." + name)
        loaded.update(multi_model_weights)

        return loaded


class Qwen3OmniMoeTalkerResizeMLP(nn.Module):
    """
    MLP for projecting between thinker and talker hidden dimensions.

    The thinker and talker have different hidden sizes:
    - Thinker: config.thinker_hidden_size (e.g., 3584)
    - Talker: config.text_config.hidden_size (e.g., 2048)

    This MLP projects from thinker → talker dimension.
    Two instances are used:
    - text_projection: For text embeddings from thinker's embedding layer
    - hidden_projection: For hidden states from thinker's last transformer layer
    """

    def __init__(self, config: Qwen3OmniMoeTalkerConfig):
        super().__init__()
        self.linear_fc1 = nn.Linear(config.thinker_hidden_size, config.text_config.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(config.text_config.intermediate_size, config.text_config.hidden_size, bias=True)
        self.act_fn = _ACTIVATION_REGISTRY[config.text_config.hidden_act]  # silu

    def forward(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class Qwen3OmniMoeModel(Qwen3MoeLLMForCausalLM):
    def __init__(self, vllm_config, talker_config, prefix):
        talker_vllm_config = vllm_config.with_hf_config(
            talker_config.text_config, architectures=["Qwen3MoeForCausalLM"]
        )
        talker_vllm_config.model_config.hf_text_config = talker_vllm_config.model_config.hf_config
        super().__init__(
            vllm_config=talker_vllm_config,
            prefix=prefix,
        )

        self.config = talker_config

        # Remove the inherited LM head so the talker only exposes codec outputs.
        if hasattr(self, "lm_head"):
            del self.lm_head

        # Replace the base embed tokens with codec embedding (defined below).
        if hasattr(self.model, "embed_tokens"):
            del self.model.embed_tokens

        # Codec embedding for RVQ code generation
        self.model.codec_embedding = nn.Embedding(
            talker_config.text_config.vocab_size, talker_config.text_config.hidden_size
        )

        # Add shared expert to each MoE layer and patch the forward method
        layer_idx = 0
        for layer in self.model.layers:
            # add shared expert to Qwen3OmniMoeSparseMoeBlock layers
            if hasattr(layer.mlp, "experts"):  # Check if it's a SparseMoeBlock
                # Shared expert is a regular gated MLP (SwiGLU)
                layer.mlp.shared_expert = Qwen3MoeMLP(
                    hidden_size=self.config.text_config.hidden_size,
                    intermediate_size=self.config.text_config.shared_expert_intermediate_size,
                    hidden_act=self.config.text_config.hidden_act,
                    quant_config=talker_vllm_config.quant_config,
                    reduce_results=False,  # Don't reduce since we'll add it manually
                    prefix=f"{prefix}.layers.{layer_idx}.mlp.shared_expert",
                )

                # Shared expert gate outputs a single scalar per token
                layer.mlp.shared_expert_gate = ReplicatedLinear(
                    self.config.text_config.hidden_size,
                    1,  # Output single scalar per token
                    bias=False,
                    quant_config=None,
                    prefix=f"{prefix}.layers.{layer_idx}.mlp.shared_expert_gate",
                )

                # Store MoE config values for router computation
                layer.mlp.top_k = self.config.text_config.num_experts_per_tok
                layer.mlp.norm_topk_prob = self.config.text_config.norm_topk_prob
                layer.mlp.num_experts = self.config.text_config.num_experts

                # Monkey-patch the forward method to use shared expert
                layer.mlp.forward = self._create_moe_forward_with_shared_expert(layer.mlp)

            layer_idx += 1

    def _create_moe_forward_with_shared_expert(self, moe_layer):
        """Create a forward method that includes shared expert computation.

        This matches the Transformers implementation where:
        1. Compute shared expert output (regular MLP)
        2. Gate it with sigmoid(shared_expert_gate(x))
        3. Apply softmax BEFORE top-k selection (matches Transformers router)
        4. Add to routed expert outputs
        """

        def forward_with_shared_expert(hidden_states: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
            # Save original shape
            orig_shape = hidden_states.shape
            hidden_dim = hidden_states.shape[-1]
            hidden_states = hidden_states.view(-1, hidden_dim)

            # handle sequence parallel if needed
            if hasattr(moe_layer, "is_sequence_parallel") and moe_layer.is_sequence_parallel:
                hidden_states = sequence_parallel_chunk(hidden_states)

            # Compute shared expert output
            # The shared expert is a regular MLP, not a routed MoE
            shared_output = None
            if hasattr(moe_layer, "shared_expert") and moe_layer.shared_expert is not None:
                # Forward through shared expert MLP
                shared_output = moe_layer.shared_expert(hidden_states)

                # Apply gating with sigmoid: sigmoid(gate(x)) * shared_expert(x)
                if hasattr(moe_layer, "shared_expert_gate") and moe_layer.shared_expert_gate is not None:
                    gate_logits, _ = moe_layer.shared_expert_gate(hidden_states)
                    gate_values = F.sigmoid(gate_logits)  # [batch, 1]
                    shared_output = gate_values * shared_output  # Broadcasting: [batch, 1] * [batch, hidden]

            # Compute experts results
            # router_logits: (num_tokens, n_experts)
            router_logits, _ = moe_layer.gate(hidden_states)
            experts_output = moe_layer.experts(hidden_states=hidden_states, router_logits=router_logits)

            # combine experts and shared expert results
            if shared_output is not None:
                final_hidden_states = experts_output + shared_output

            # Handle sequence parallel if needed
            if hasattr(moe_layer, "is_sequence_parallel") and moe_layer.is_sequence_parallel:
                from vllm.distributed import tensor_model_parallel_all_gather

                num_tokens = orig_shape[0] if len(orig_shape) > 1 else 1
                final_hidden_states = tensor_model_parallel_all_gather(final_hidden_states, 0)
                final_hidden_states = final_hidden_states[:num_tokens]
            try:
                final_hidden_states.view(orig_shape)
            except Exception as e:
                print(f"Error viewing final hidden states: {e}")
                print(f"final_hidden_states.shape: {final_hidden_states.shape}")
                print(f"orig_shape: {orig_shape}")
                raise e
            # Return with original shape
            return final_hidden_states.view(orig_shape)

        return forward_with_shared_expert

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        return self.model.codec_embedding(input_ids)
