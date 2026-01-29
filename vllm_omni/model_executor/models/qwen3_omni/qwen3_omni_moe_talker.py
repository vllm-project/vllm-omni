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
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import _ACTIVATION_REGISTRY
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsPP,
)
from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniThinkerDummyInputsBuilder,
)
from vllm.model_executor.models.qwen3_moe import Qwen3MoeMLP, Qwen3MoeSparseMoeBlock
from vllm.model_executor.models.qwen3_omni_moe_thinker import Qwen3Omni_VisionTransformer
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
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


logger = init_logger(__name__)

Qwen3OmniMoeThinkerDummyInputsBuilder = Qwen2_5OmniThinkerDummyInputsBuilder


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3OmniMoeThinkerMultiModalProcessor,
    info=Qwen3OmniMoeThinkerProcessingInfo,
    dummy_inputs=Qwen3OmniMoeThinkerDummyInputsBuilder,
)
class Qwen3OmniMoeTalkerForConditionalGeneration(
    nn.Module,
    # SupportsMultiModal,
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
        self.layer0_embed_buffer = torch.zeros(
            (vllm_config.scheduler_config.max_num_seqs, 1, self.config.text_config.hidden_size),
            dtype=vllm_config.model_config.dtype,
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

            # Initial input: [last_talker_hidden, layer0_embed]
            layer0_embed = self.embed_input_ids(layer0_code)
            self.layer0_embed_buffer[:batch_size].copy_(layer0_embed)
            pos_all_layers, current_input = self.code_predictor(
                layer0_code, self.layer0_embed_buffer[:batch_size], last_talker_hidden
            )

            # Stack all layers for this position: [batch, num_code_groups, 1]
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
        summed_embeddings = torch.cat(all_summed_embeddings, dim=1).squeeze(1)

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
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Forward pass through the talker model."""
        talker_hidden_states, _ = self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        return talker_hidden_states

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

        logger.warning(
            "\n\n\n"
            "THIS FUNCTION RETURNS DUMMY MULTIMODAL EMBEDDINGS FOR PROFILE RUN, "
            "SHOULD NOT BE CALLED IN INFERENCE."
            "\n\n\n"
        )

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        dummy_multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        # TODO: do projection for all multimodel
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                image_embeddings = self._process_image_input(multimodal_input)
                dummy_image_embeddings = ()
                for image_embed in image_embeddings:
                    dummy_image_embeddings += (
                        torch.zeros(
                            image_embed.shape[0],
                            self.config.text_config.hidden_size,
                            device=image_embed.device,
                            dtype=torch.bfloat16,
                        ),
                    )
                dummy_multimodal_embeddings += tuple(image_embeddings)
            if modality == "video":
                video_embeddings = self._process_video_input(multimodal_input)
                dummy_video_video_embeddings = ()
                for video_embed in video_embeddings:
                    dummy_video_video_embeddings += (
                        torch.zeros(
                            video_embed.shape[0],
                            self.config.text_config.hidden_size,
                            device=video_embed.device,
                            dtype=torch.bfloat16,
                        ),
                    )
                dummy_multimodal_embeddings += tuple(dummy_video_video_embeddings)
            if modality == "audio":
                audio_embeddings = self._process_audio_input(multimodal_input)
                dummy_audio_embeddings = ()
                for audio_embed in audio_embeddings:
                    dummy_audio_embeddings += (
                        torch.zeros(
                            audio_embed.shape[0],
                            self.config.text_config.hidden_size,
                            device=audio_embed.device,
                            dtype=torch.bfloat16,
                        ),
                    )
                dummy_multimodal_embeddings += tuple(dummy_audio_embeddings)
        return dummy_multimodal_embeddings

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
            logger.error("Error logging model load summary")

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


class Qwen3OmniMoeTalkerSharedExpertWrapper(nn.Module):
    """
    Wrapper that combines shared_expert MLP with its sigmoid gate.

    This matches the HuggingFace weight structure where:
    - mlp.shared_expert.{gate_proj, up_proj, down_proj}.weight
    - mlp.shared_expert_gate.weight  (sibling, not child)

    The wrapper applies: sigmoid(shared_expert_gate(x)) * shared_expert(x).

    It also exposes the underlying shared_expert interface to keep
    compatibility with backends that split shared-expert computation.
    """

    def __init__(
        self,
        shared_expert: Qwen3MoeMLP,
        shared_expert_gate: nn.Linear,
    ):
        super().__init__()
        self._shared_expert = shared_expert
        self._shared_expert_gate = shared_expert_gate

    @property
    def gate_up_proj(self):
        return self._shared_expert.gate_up_proj

    @property
    def down_proj(self):
        return self._shared_expert.down_proj

    @property
    def act_fn(self):
        return self._shared_expert.act_fn

    def expert_gate(self, x: torch.Tensor):
        gate_out = self._shared_expert_gate(x)
        if isinstance(gate_out, tuple):
            return gate_out
        return gate_out, None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._shared_expert(x)
        gate_out = self._shared_expert_gate(x)
        if isinstance(gate_out, tuple):
            gate_out = gate_out[0]
        gate_values = F.sigmoid(gate_out)  # [batch, 1]
        return gate_values * out  # Broadcasting: [batch, 1] * [batch, hidden]


class Qwen3OmniMoeTalkerSparseMoeBlock(nn.Module):
    """
    Sparse MoE block for Qwen3 Omni MoE Talker with shared expert support.

    This block uses SharedFusedMoE to efficiently compute both routed experts
    and the shared expert, potentially overlapping computation with communication.

    Weight structure matches HuggingFace:
    - mlp.gate.weight (router)
    - mlp.shared_expert.{gate_proj, up_proj, down_proj}.weight
    - mlp.shared_expert_gate.weight
    - mlp.experts.{0..n}.{gate_proj, up_proj, down_proj}.weight
    """

    def __init__(
        self,
        config: Qwen3OmniMoeTalkerConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        text_config = config.text_config
        self.tp_size = get_tensor_model_parallel_world_size()

        if self.tp_size > text_config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than the number of experts {text_config.num_experts}."
            )

        # Router gate for selecting top-k experts
        self.gate = ReplicatedLinear(
            text_config.hidden_size,
            text_config.num_experts,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate",
        )

        # Shared expert MLP (matches HF: mlp.shared_expert.*)
        if text_config.shared_expert_intermediate_size > 0:
            self.shared_expert = Qwen3MoeMLP(
                hidden_size=text_config.hidden_size,
                intermediate_size=text_config.shared_expert_intermediate_size,
                hidden_act=text_config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,  # Don't reduce, we'll handle it
                prefix=f"{prefix}.shared_expert",
            )
            # Shared expert gate (matches HF: mlp.shared_expert_gate.weight)
            # This is a sibling of shared_expert, not a child
            self.shared_expert_gate = torch.nn.Linear(text_config.hidden_size, 1, bias=False)
            # Create wrapper for SharedFusedMoE
            self._shared_expert_wrapper = Qwen3OmniMoeTalkerSharedExpertWrapper(
                self.shared_expert, self.shared_expert_gate
            )
        else:
            self.shared_expert = None
            self.shared_expert_gate = None
            self._shared_expert_wrapper = None

        # Fused MoE with shared expert support
        self.experts = SharedFusedMoE(
            shared_experts=self._shared_expert_wrapper,
            num_experts=text_config.num_experts,
            top_k=text_config.num_experts_per_tok,
            hidden_size=text_config.hidden_size,
            intermediate_size=text_config.moe_intermediate_size,
            reduce_results=False,  # We'll reduce manually after combining
            renormalize=text_config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Compute router logits
        router_logits, _ = self.gate(hidden_states)

        # Forward through SharedFusedMoE
        # Returns (shared_out, fused_out) when shared_expert is present
        final_hidden_states = self.experts(hidden_states=hidden_states, router_logits=router_logits)

        # Combine shared and routed expert outputs
        if self._shared_expert_wrapper is not None:
            # SharedFusedMoE returns tuple: (shared_out, fused_out)
            final_hidden_states = final_hidden_states[0] + final_hidden_states[1]

        # Apply tensor parallel reduction if needed
        if self.tp_size > 1:
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(final_hidden_states)

        return final_hidden_states.view(orig_shape)


class Qwen3OmniMoeModel(Qwen3MoeLLMForCausalLM):
    """
    Qwen3 Omni MoE Talker language model.

    This model extends Qwen3MoeLLMForCausalLM with:
    - Shared expert support via SharedFusedMoE
    - Codec embedding instead of text embedding
    - No LM head (codec head is separate in the parent class)
    """

    def __init__(self, vllm_config: VllmConfig, talker_config: Qwen3OmniMoeTalkerConfig, prefix: str):
        # Create a vllm_config for the talker's text model
        talker_vllm_config = vllm_config.with_hf_config(
            talker_config.text_config, architectures=["Qwen3MoeForCausalLM"]
        )
        talker_vllm_config.model_config.hf_text_config = talker_vllm_config.model_config.hf_config

        super().__init__(
            vllm_config=talker_vllm_config,
            prefix=prefix,
        )

        self.config = talker_config
        self.talker_vllm_config = talker_vllm_config

        # Remove the inherited LM head so the talker only exposes codec outputs.
        if hasattr(self, "lm_head"):
            del self.lm_head

        # Replace the base embed tokens with codec embedding.
        if hasattr(self.model, "embed_tokens"):
            del self.model.embed_tokens

        # Codec embedding for RVQ code generation
        self.model.codec_embedding = nn.Embedding(
            talker_config.text_config.vocab_size,
            talker_config.text_config.hidden_size,
        )

        # Replace MoE blocks with shared expert versions
        self._replace_moe_blocks_with_shared_expert(prefix)

    def _replace_moe_blocks_with_shared_expert(self, prefix: str) -> None:
        """
        Replace Qwen3MoeSparseMoeBlock layers with Qwen3OmniMoeTalkerSparseMoeBlock
        that includes shared expert support via SharedFusedMoE.
        """
        # Get compilation config to clean up registered layer names
        compilation_config = self.talker_vllm_config.compilation_config

        for layer_idx, layer in enumerate(self.model.layers):
            # Check if this layer has a MoE block (has experts attribute)
            if isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
                # Remove old layer registration from static_forward_context
                old_experts_prefix = f"{prefix}.model.layers.{layer_idx}.mlp.experts"
                if old_experts_prefix in compilation_config.static_forward_context:
                    del compilation_config.static_forward_context[old_experts_prefix]

                # Create new MoE block with shared expert support
                layer.mlp = Qwen3OmniMoeTalkerSparseMoeBlock(
                    config=self.config,
                    quant_config=self.talker_vllm_config.quant_config,
                    prefix=f"{prefix}.model.layers.{layer_idx}.mlp",
                )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Embed codec input IDs."""
        return self.model.codec_embedding(input_ids)
