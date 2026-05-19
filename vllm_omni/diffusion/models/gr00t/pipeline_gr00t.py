# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""GR00T-N1.7 diffusion pipeline for vllm-omni.

``Gr00tN1d7Pipeline.forward`` is a thin orchestrator:

  1. :func:`transform.encode` builds model-ready tensors from the raw
     OpenPI observation (normalize state, tokenize prompt+images,
     resolve embodiment_id, stash metadata for decode).
  2. The GR00T model code runs: ``self.backbone`` (Qwen3-VL truncated to
     ``select_layer``) → ``self.action_head`` (4-step Euler flow-matching).
  3. :func:`transform.decode` denormalizes the model's output and
     reconstructs absolute actions (SE(3) compose for ``eef_9d``,
     element-wise add for joint keys).

Clients send **raw** observations and receive **absolute** per-action-key
ndarrays — no client-side stats, no SE(3) math, no checkpoint deps.  All
data-pipeline logic lives in :mod:`vllm_omni.diffusion.models.gr00t.transform`.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.gr00t import transform as gr00t_transform
from vllm_omni.diffusion.models.gr00t.adapter_qwen3_vl import Qwen3VLBackbone
from vllm_omni.diffusion.models.gr00t.modeling.action_head import Gr00tN1d7ActionHead
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.transformers_utils.configs.gr00t import Gr00tN1d7Config

logger = logging.getLogger(__name__)


class Gr00tN1d7Pipeline(nn.Module):
    """GR00T-N1.7 vision-language-action pipeline.

    Submodules — both pure model code:
      ``self.backbone``    : :class:`Qwen3VLBackbone` (Cosmos-Reason2-2B
        truncated to ``select_layer=16``).
      ``self.action_head`` : :class:`Gr00tN1d7ActionHead` (flow-matching DiT
        with 4 Euler steps).

    Server-only state (not part of the model graph):
      ``self._norm_stats`` : per-embodiment q01/q99 quantiles loaded once
        from ``experiment_cfg/dataset_statistics.json``.  Consumed by
        :func:`transform.encode` (state normalize) and :func:`transform.decode`
        (action denormalize).
      ``self._processor``  : lazily-loaded Qwen3VLProcessor.
    """

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.od_config = od_config
        self.prefix = prefix

        model_path = od_config.model
        local_files_only = os.path.exists(model_path)

        self.gr00t_config = self._load_gr00t_config(model_path, local_files_only)
        self._norm_stats = self._load_norm_stats(model_path, local_files_only)

        # `model_config.backbone_model_name` is an optional deploy-yaml
        # override for the Qwen3-VL backbone HF repo id baked into the
        # GR00T checkpoint's config.json (``nvidia/Cosmos-Reason2-2B`` is
        # gated).  Users with a local clone can point this at a path so the
        # backbone + processor load without an HF token.
        _model_config_dict = getattr(od_config, "model_config", {}) or {}
        backbone_override = _model_config_dict.get("backbone_model_name")
        if backbone_override:
            logger.info(
                "Gr00tN1d7Pipeline: overriding backbone model_name "
                "%r → %r per deploy-yaml model_config.backbone_model_name",
                self.gr00t_config.model_name, backbone_override,
            )
            self.gr00t_config.model_name = backbone_override
            # Wipe stale overlay (from the original gated repo id) and
            # re-resolve text_config / vision_config from the overridden
            # backbone source.
            for attr in ("text_config", "vision_config"):
                if hasattr(self.gr00t_config, attr):
                    delattr(self.gr00t_config, attr)
            self.gr00t_config._overlay_backbone(backbone_override, {})

        self.backbone = Qwen3VLBackbone(self.gr00t_config)
        self.action_head = Gr00tN1d7ActionHead(self.gr00t_config)
        # The GR00T DiT carries dropout layers (attn_dropout=0.2 in the
        # checkpoint config) that would otherwise be active during
        # inference and make outputs stochastic — even with the seed pin
        # below.  Disable training mode on the whole pipeline.
        self.eval()

        offload = getattr(od_config, "enable_cpu_offload", False) or getattr(
            od_config, "enable_layerwise_offload", False
        )
        if not offload:
            self.to(device=get_local_device(), dtype=od_config.dtype)

        # Embodiment name → projector index override map (deploy yaml).
        # Falls back to the verbatim EMBODIMENT_TAG_TO_PROJECTOR_INDEX
        # inside transform.encode for tags not listed here.
        model_config = getattr(od_config, "model_config", {}) or {}
        self.embodiment_name_to_id: dict[str, int] = model_config.get(
            "embodiment_name_to_id", {}
        )

        self._weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model_path,
                subfolder=None,
                revision=None,
                prefix="",
                fall_back_to_pt=False,
                allow_patterns_overrides=[
                    "model-*.safetensors",
                    "model.safetensors",
                ],
            ),
        ]

    # ------------------------------------------------------------------
    # Config / stats loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_repo_json(
        model_path: str, relative_path: str, local_files_only: bool
    ) -> dict | None:
        if local_files_only and os.path.isdir(model_path):
            json_path = os.path.join(model_path, relative_path)
            if not os.path.exists(json_path):
                return None
            with open(json_path) as f:
                return json.load(f)
        try:
            json_path = hf_hub_download(model_path, relative_path)
            with open(json_path) as f:
                return json.load(f)
        except Exception:
            logger.warning("Failed to load %s from %s", relative_path, model_path)
            return None

    @classmethod
    def _load_gr00t_config(
        cls, model_path: str, local_files_only: bool
    ) -> Gr00tN1d7Config:
        raw = cls._load_repo_json(model_path, "config.json", local_files_only)
        if raw is None:
            raise ValueError(
                f"GR00T-N1.7 requires root config.json in {model_path}."
            )
        raw.pop("architectures", None)
        return Gr00tN1d7Config(**raw)

    @classmethod
    def _load_norm_stats(
        cls, model_path: str, local_files_only: bool
    ) -> dict:
        """Load ``experiment_cfg/dataset_statistics.json`` from the GR00T
        checkpoint and cache it on the pipeline.

        Returns ``{embodiment_tag: {"state": {...}, "action": {...},
        "relative_action": {...}}}``.  Returns an empty dict when the file
        is missing — the server will then fall back to no-op normalization
        and emit a warning at startup.
        """
        raw = cls._load_repo_json(
            model_path, "experiment_cfg/dataset_statistics.json", local_files_only
        )
        if raw is None:
            logger.warning(
                "Gr00tN1d7Pipeline: experiment_cfg/dataset_statistics.json not "
                "found under %s; state/action norm + relative→absolute decode "
                "will be no-op.  This is expected only for synthetic-data smoke "
                "tests.",
                model_path,
            )
            return {}
        return raw

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    @property
    def weights_sources(self):
        return self._weights_sources

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Route checkpoint tensors by prefix.

        ``backbone.<rest>`` lands on :attr:`backbone`; ``action_head.<rest>``
        lands on :attr:`action_head`.  Anything else is logged and dropped.
        """
        backbone_state: dict[str, torch.Tensor] = {}
        action_head_state: dict[str, torch.Tensor] = {}
        ignored: list[str] = []
        for name, tensor in weights:
            if name.startswith("backbone."):
                backbone_state[name[len("backbone."):]] = tensor
            elif name.startswith("action_head."):
                action_head_state[name[len("action_head."):]] = tensor
            else:
                ignored.append(name)

        loaded: set[str] = set()
        if backbone_state:
            res = self.backbone.load_state_dict(backbone_state, strict=False)
            if res.missing_keys:
                logger.warning(
                    "Gr00tN1d7Pipeline.backbone missing %d keys: %s",
                    len(res.missing_keys),
                    res.missing_keys[:5],
                )
            if res.unexpected_keys:
                logger.warning(
                    "Gr00tN1d7Pipeline.backbone unexpected %d keys: %s",
                    len(res.unexpected_keys),
                    res.unexpected_keys[:5],
                )
            for key in backbone_state:
                if key not in res.unexpected_keys:
                    loaded.add(f"backbone.{key}")
        if action_head_state:
            res = self.action_head.load_state_dict(action_head_state, strict=False)
            if res.missing_keys:
                logger.warning(
                    "Gr00tN1d7Pipeline.action_head missing %d keys: %s",
                    len(res.missing_keys),
                    res.missing_keys[:5],
                )
            if res.unexpected_keys:
                logger.warning(
                    "Gr00tN1d7Pipeline.action_head unexpected %d keys: %s",
                    len(res.unexpected_keys),
                    res.unexpected_keys[:5],
                )
            for key in action_head_state:
                if key not in res.unexpected_keys:
                    loaded.add(f"action_head.{key}")
        if ignored:
            logger.warning(
                "Gr00tN1d7Pipeline ignored %d unrecognized weight keys "
                "(first few: %s)",
                len(ignored),
                ignored[:5],
            )
        logger.info(
            "Gr00tN1d7Pipeline.load_weights: loaded %d tensors", len(loaded)
        )
        return loaded

    # ------------------------------------------------------------------
    # Forward — thin orchestrator (encode → model → decode)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, req: OmniDiffusionRequest, **kwargs) -> DiffusionOutput:
        extra_args = req.sampling_params.extra_args or {}
        robot_obs = extra_args.get("robot_obs")
        if robot_obs is None:
            return self._maybe_dummy_warmup(req)

        embodiment_tag = robot_obs.get("embodiment", "")
        stats_for_emb = self._norm_stats.get(embodiment_tag, {})
        state_stats = stats_for_emb.get("state", {})
        action_stats = stats_for_emb.get("action", {})
        rel_action_stats = stats_for_emb.get("relative_action", {})

        # 1) PRE-TRANSFORM: raw obs → model-ready CPU tensors
        encoded = gr00t_transform.encode(
            robot_obs,
            gr00t_config=self.gr00t_config,
            processor=self._get_processor(),
            state_norm_stats=state_stats,
            embodiment_name_to_id=self.embodiment_name_to_id,
        )

        device = next(self.parameters()).device
        dtype = self.od_config.dtype
        input_ids = encoded["input_ids"].to(device=device)
        attention_mask = encoded["attention_mask"].to(device=device)
        state = encoded["state"].to(device=device, dtype=dtype)
        embodiment_id = encoded["embodiment_id"].to(device=device)
        pixel_values = encoded.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(device=device, dtype=dtype)
        image_grid_thw = encoded.get("image_grid_thw")
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(device=device)

        # 2) MODEL FORWARD — pure GR00T model code
        # Optional `robot_obs["seed"]` pins the flow-matching initial-noise
        # sample for reproducible parity checks; production requests usually
        # omit it and get stochastic exploration.
        seed = robot_obs.get("seed")
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        bb_out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        actions = self.action_head.get_action(
            vl_embeds=bb_out["backbone_features"],
            vl_attn_mask=bb_out["backbone_attention_mask"],
            image_mask=bb_out["image_mask"],
            state=state,
            embodiment_id=embodiment_id,
        )

        # 3) POST-TRANSFORM: normalized [horizon, max_action_dim] → absolute
        # per-action-key dict (issue #3553 wire shape).
        actions_np = actions.squeeze(0).float().cpu().numpy()
        actions_dict = gr00t_transform.decode(
            actions_np,
            raw_state=encoded["raw_state"],
            modality=encoded["modality"],
            action_norm_stats=action_stats,
            relative_action_norm_stats=rel_action_stats,
        )
        return DiffusionOutput(output={"actions": actions_dict})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _maybe_dummy_warmup(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        """vllm-omni's diffusion engine warms each worker with
        ``prompt="dummy run"`` / ``num_inference_steps=1`` requests that
        have no ``robot_obs``.  Return a zero-shaped placeholder so the
        warmup succeeds (mirrors DreamZeroPipeline)."""
        first_prompt = req.prompts[0] if req.prompts else ""
        prompt_text = (
            first_prompt
            if isinstance(first_prompt, str)
            else (first_prompt.get("prompt") or "")
        )
        is_dummy_warmup = (
            prompt_text == "dummy run"
            and getattr(req.sampling_params, "num_inference_steps", None) == 1
        )
        if not is_dummy_warmup:
            raise KeyError("robot_obs (set via sampling_params.extra_args)")
        logger.info("Skipping GR00T dummy warmup request without robot_obs.")
        return DiffusionOutput(
            output={
                "actions": np.zeros(
                    (self.gr00t_config.action_horizon, self.gr00t_config.max_action_dim),
                    dtype=np.float32,
                )
            }
        )

    def _get_processor(self):
        """Lazy-load + cache the Qwen3VLProcessor.

        ``model_name`` (from the GR00T config) is either an HF repo id
        (e.g. ``nvidia/Cosmos-Reason2-2B``, which is gated — set
        ``HF_TOKEN`` / ``huggingface-cli login`` to access it) or a local
        checkout path.
        """
        if getattr(self, "_processor", None) is not None:
            return self._processor
        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self.gr00t_config.model_name, trust_remote_code=True
        )
        return self._processor
