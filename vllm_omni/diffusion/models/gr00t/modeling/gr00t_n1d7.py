# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import os
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature

from vllm_omni.diffusion.models.gr00t.configs.gr00t_n1d7 import Gr00tN1d7Config
from vllm_omni.diffusion.models.gr00t.modeling.modules.dit import AlternateVLDiT, DiT, SelfAttentionTransformer
from vllm_omni.diffusion.models.gr00t.modeling.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)
from vllm_omni.diffusion.models.gr00t.modeling.processing_gr00t_n1d7 import Gr00tN1d7DataCollator

logger = logging.getLogger(__name__)


class Gr00tN1d7ActionHead(nn.Module):
    """Action head component for flow matching diffusion policy."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d7Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
            logger.info("Using AlternateVLDiT for diffusion model")
        else:
            self.model = DiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
            )
            logger.info("Using DiT for diffusion model")
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim * config.state_history_length,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.vlln = nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()

        vl_self_attention_cfg = config.vl_self_attention_cfg
        if vl_self_attention_cfg and vl_self_attention_cfg.get("num_layers", 0) > 0:
            self.vl_self_attention = SelfAttentionTransformer(**vl_self_attention_cfg)
        else:
            self.vl_self_attention = nn.Identity()

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.num_timestep_buckets = config.num_timestep_buckets

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def _encode_features(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Encode features for the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_history_length, max_state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - state_features: [B, 1, input_embedding_dim]
        """
        backbone_output = self.process_backbone_output(backbone_output)

        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        state = action_input.state
        current_T = state.shape[1]
        assert current_T == self.config.state_history_length, "current_T != state_history_length"
        # [B, state_history_length, max_state_dim] -> [B, 1, state_history_length * max_state_dim]
        state = state.view(state.shape[0], 1, -1)

        state_features = self.state_encoder(state, embodiment_id)

        return BatchFeature(data={"backbone_features": vl_embeds, "state_features": state_features})

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model
        """
        vl_embeds = backbone_features

        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        _seed_env = os.environ.get("GR00T_NOISE_SEED")
        if _seed_env is not None:
            _gen = torch.Generator(device=device).manual_seed(int(_seed_env))
            actions = torch.randn(
                size=(batch_size, self.config.action_horizon, self.action_dim),
                dtype=vl_embeds.dtype,
                device=device,
                generator=_gen,
            )
        else:
            actions = torch.randn(
                size=(batch_size, self.config.action_horizon, self.action_dim),
                dtype=vl_embeds.dtype,
                device=device,
            )

        dt = 1.0 / self.num_inference_timesteps
        vel_strength = torch.ones_like(actions)

        if "action" in action_input:
            # If action in input when doing get action, it means we want to use RTC.
            # action_horizon is the action horizon of the input action.
            # rtc_overlap_steps is the number of steps to overlap with the previous action chunks.
            # rtc_frozen_steps is the policy inference latency expressed as frozen action steps.
            # rtc_ramp_rate is the rate of the ramp of denoising the actions.
            assert options is not None, "options is not None"
            assert "action_horizon" in options, "action_horizon is not in options"
            assert "rtc_overlap_steps" in options, "rtc_overlap_steps is not in options"
            assert "rtc_frozen_steps" in options, "rtc_frozen_steps is not in options"
            assert "rtc_ramp_rate" in options, "rtc_ramp_rate is not in options"

            action_horizon_before_padding = options["action_horizon"]

            # Use previous action instead of pure noise to do inpainting
            actions[:, : options["rtc_overlap_steps"], :] = action_input["action"][
                :,
                action_horizon_before_padding - options["rtc_overlap_steps"] : action_horizon_before_padding,
                :,
            ]
            vel_strength[:, : options["rtc_frozen_steps"], :] = 0.0
            # NOTE: use an exponential ramp strength to set the remaining unfrozen rtc_steps
            intermediate_steps = options["rtc_overlap_steps"] - options["rtc_frozen_steps"]
            # Create exponential ramp from 0 to 1 over intermediate steps
            t = torch.linspace(0.0, 1.0, intermediate_steps + 2, device=device)
            ramp = 1 - torch.exp(-options["rtc_ramp_rate"] * t)
            ramp = ramp / ramp[-1].clamp_min(1e-8)  # normalize to [0,1]
            ramp = ramp[1:-1]  # we will only take the middle part of the ramp, ignore the 0.0 and 1.0
            # Apply ramp to the intermediate steps [batch, intermediate_steps, action_dim]
            vel_strength[
                :,
                options["rtc_frozen_steps"] : options["rtc_overlap_steps"],
                :,
            ] = ramp[None, :, None].to(device)

        # Run denoising steps.
        for t in range(self.num_inference_timesteps):
            t_discretized = t * self.num_timestep_buckets // self.num_inference_timesteps

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            sa_embs = torch.cat((state_features, action_features), dim=1)

            # Run model forward.
            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity * vel_strength

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - action_pred: [B, action_horizon, action_dim] predicted actions
        """
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
            action_input=action_input,
            options=options,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input batch for the action head."""
        return BatchFeature(data=batch)


def _load_qwen3_vl_cls() -> type:
    """Import Qwen3-VL lazily so the missing-dep failure is clear and deferred."""
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise ImportError(
            "GR00T-N1.7 requires transformers>=4.57.1 for transformers.models.qwen3_vl "
            "(the repo's declared floor is 4.56.0). Please upgrade transformers."
        ) from exc
    return Qwen3VLForConditionalGeneration


class _Qwen3VLBackbone(nn.Module):
    """GR00T adapter around the shared Qwen3-VL implementation."""

    def __init__(
        self,
        model_name: str,
        select_layer: int,
        backbone_embedding_dim: int,
        load_bf16: bool,
        transformers_loading_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()
        backbone_config = AutoConfig.from_pretrained(
            model_name, **(transformers_loading_kwargs or {"trust_remote_code": True})
        )
        backbone_config.text_config._attn_implementation = "sdpa"
        backbone_config.vision_config._attn_implementation = "sdpa"
        self.model = _load_qwen3_vl_cls()(backbone_config).eval()
        if load_bf16:
            self.model.to(dtype=torch.bfloat16)

        target_layers = select_layer if select_layer >= 0 else len(self.model.model.language_model.layers)
        while len(self.model.model.language_model.layers) > target_layers:
            self.model.model.language_model.layers.pop(-1)

        # Drop the unused lm_head: feature-extractor only, and .to(dtype) above unties it (~0.58 GiB).
        self.model.lm_head = None

    def set_frozen_modules_to_eval_mode(self) -> None:
        self.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        keys_to_use = [
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
            "mm_token_type_ids",
        ]
        vl_input = {key: vl_input[key] for key in keys_to_use if key in vl_input}
        # GR00T was trained against the **pre-final-RMSNorm** output of the last
        # decoder layer (this was `outer.hidden_states[-1]` under HF 4.57.x semantics,
        # where the @check_model_inputs tie-logic skipped overwriting because
        # Qwen3VLCausalLMOutputWithPast has no `last_hidden_state` field).
        # HF >=5.x changed the output_hidden_states tuple plumbing so that
        # `hidden_states[-1]` is post-norm, which is a different tensor and breaks
        # the trained action head. Capture the pre-norm tensor via a hook on the
        # final RMSNorm regardless of transformers version.
        _captured: list[torch.Tensor] = []

        def _pre_norm_hook(_module, args, _out):
            _captured.append(args[0])

        norm = self.model.model.language_model.norm
        handle = norm.register_forward_hook(_pre_norm_hook)
        try:
            self.model.model(**vl_input, return_dict=True)
        finally:
            handle.remove()
        if not _captured:
            raise RuntimeError("Failed to capture pre-norm hidden states from Qwen3-VL backbone")
        backbone_features = _captured[-1]
        image_mask = vl_input["input_ids"] == self.model.config.image_token_id
        attention_mask = vl_input["attention_mask"] == 1
        return BatchFeature(
            data={
                "backbone_features": backbone_features,
                "backbone_attention_mask": attention_mask,
                "image_mask": image_mask,
            }
        )


def get_backbone_cls(config: Gr00tN1d7Config):
    if "nvidia/Cosmos-Reason2" in config.model_name:
        return _Qwen3VLBackbone
    raise ValueError(f"Unsupported model name: {config.model_name}")


class Gr00tN1d7(PreTrainedModel):
    """Gr00tN1d7: VLA model with Cosmos-Reason2-2B (Qwen3-VL) backbone."""

    config_class = Gr00tN1d7Config
    supports_gradient_checkpointing = True
    _tp_plan = {}

    @property
    def all_tied_weights_keys(self) -> dict[str, Any]:
        return {}

    def __init__(
        self,
        config: Gr00tN1d7Config,
        transformers_loading_kwargs: dict | None = None,
    ):
        super().__init__(config)
        if transformers_loading_kwargs is None:
            transformers_loading_kwargs = {"trust_remote_code": True}
        self.config = config

        backbone_cls = get_backbone_cls(config)
        self.backbone = backbone_cls(
            model_name=config.model_name,
            select_layer=config.select_layer,
            backbone_embedding_dim=config.backbone_embedding_dim,
            load_bf16=config.load_bf16,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        # Initialize action head
        self.action_head = Gr00tN1d7ActionHead(config)

        self.collator = Gr00tN1d7DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

    def prepare_input(self, inputs: dict) -> tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action head."""

        # NOTE -- currently the eval code doesn't use collator, so we need to add it here
        if "vlm_content" in inputs:
            # Fix for n_envs > 1: Process all environments' VLM content, not just the first
            vlm_content_list = inputs["vlm_content"]
            # Ensure vlm_content_list is always a list for consistent processing
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]

            # Process all VLM contents through the collator
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        backbone_inputs = backbone_inputs.to(device=self.device, dtype=self.dtype)
        action_inputs = action_inputs.to(device=self.device, dtype=self.dtype)

        return backbone_inputs, action_inputs

    def forward(self, inputs: dict) -> BatchFeature:
        """
        Forward pass through the complete model.

        Args:
            inputs: Dictionary containing:
                - Action inputs (state, action, embodiment_id, etc.)

        Returns:
            BatchFeature containing loss and other outputs
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head(backbone_outputs, action_inputs)

        return action_outputs

    def get_action(self, inputs: dict, options: dict[str, Any] | None = None) -> BatchFeature:
        """
        Generate actions using the complete model.
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Forward through backbone
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head.get_action(backbone_outputs, action_inputs, options)

        return action_outputs

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# Register the model with HuggingFace
AutoConfig.register("Gr00tN1d7", Gr00tN1d7Config)
AutoModel.register(Gr00tN1d7Config, Gr00tN1d7)
