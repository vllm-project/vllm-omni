# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Single-stage, stateless LingBot action inference over the OpenPI protocol."""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from .config import LingBotVLAV2Config
from .modeling_lingbot_vla_v2 import LingBotVLAV2ForActionPrediction
from .processor_lingbot_vla_v2 import LingBotVLAV2Processor

CHECKPOINT_REVISIONS = {
    "lerobot/lingbot_vla_v2_base": "5dd4b5299ed85c782a987d58ff3f24f95a95474d",
    "lerobot/lingbot_vla_v2_robotwin": "54f79a4b6e9166bda91b5010088097ff3191bca2",
}


def _action_postprocess(output: DiffusionOutput) -> DiffusionOutput:
    """Identity post-process; module level so it pickles across the orchestrator boundary."""
    return output


def get_lingbot_vla_v2_post_process_func(od_config: OmniDiffusionConfig):
    """The pipeline already returns calibrated robot actions."""
    return _action_postprocess


class LingBotVLAV2Pipeline(nn.Module):
    """Raw observation -> full action chunk. No state is shared across observations."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.prefix = prefix
        if str(od_config.dtype).removeprefix("torch.") != "float32":
            raise ValueError("LingBot-VLA 2.0 currently requires dtype=float32")
        parallel = od_config.parallel_config
        if parallel.world_size != 1 or parallel.use_hsdp:
            raise ValueError("LingBot-VLA 2.0 currently supports single-GPU inference only")
        if od_config.quantization_config is not None:
            raise ValueError("LingBot-VLA 2.0 does not support quantized checkpoints")
        self.model_dir = self._resolve_model_dir(od_config.model, od_config.revision)
        self.config = LingBotVLAV2Config.from_pretrained(self.model_dir, od_config.model_config)
        from vllm_omni.diffusion.distributed.utils import get_local_device

        self.device = get_local_device()
        # The processor checks calibration/assets before allocating the model.
        self.processor = LingBotVLAV2Processor(self.config, self.model_dir, self.device)
        # Engine loading may itself run under inference_mode; parameters need version counters.
        with torch.inference_mode(False):
            self.model = LingBotVLAV2ForActionPrediction(self.config).float()
            weights_path = self.model_dir / "model.safetensors"
            if not weights_path.is_file():
                raise FileNotFoundError(f"Expected LeRobot-format LingBot weights at {weights_path}")
            with safe_open(weights_path, framework="pt", device="cpu") as checkpoint:
                loaded = self.model.load_weights((key, checkpoint.get_tensor(key)) for key in checkpoint.keys())
            self.loaded_weight_names: set[str] = {f"model.{name}" for name in loaded}
            self.model.to(self.device).eval()
        self.requires_grad_(False)
        self._validate_handshake()

    @staticmethod
    def _resolve_model_dir(model: str, revision: str | None) -> Path:
        if Path(model).is_dir():
            return Path(model)
        from vllm_omni.transformers_utils.repo_utils import hf_api

        resolved_revision = revision or CHECKPOINT_REVISIONS.get(model)
        if resolved_revision is None:
            raise ValueError("Provide an explicit revision for an additional LingBot checkpoint")
        return Path(
            hf_api().snapshot_download(
                repo_id=model,
                revision=resolved_revision,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "tokenizer/*",
                    "chat_template_tokenizer/*",
                    "image_processor/*",
                ],
            )
        )

    def _validate_handshake(self):
        metadata = self.config.policy_server_config
        expected = {
            "action_dim": self.config.action_dim,
            "action_horizon": self.config.chunk_size,
            "max_action_dim": self.config.max_action_dim,
            "needs_session_id": False,
        }
        for key, value in expected.items():
            if key in metadata and metadata[key] != value:
                raise ValueError(f"OpenPI {key}={metadata[key]!r} disagrees with checkpoint value {value!r}")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]] = ()) -> set[str]:
        """Report the checkpoint loaded in __init__ to the loader's strict check; no weights_sources are exposed."""
        streamed = next(iter(weights), None)
        if streamed is not None:
            raise ValueError(f"LingBot loads model.safetensors itself; got framework weight {streamed[0]}")
        return self.loaded_weight_names

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch, **kwargs) -> DiffusionOutput:
        if req.num_reqs != 1:
            raise ValueError("LingBot currently accepts one observation per request batch")
        extra = req.sampling_params.extra_args or {}
        obs = extra.get("robot_obs")
        if obs is None:
            if req.is_dummy_run():
                return DiffusionOutput(
                    output={"actions": np.zeros((self.config.chunk_size, self.config.action_dim), dtype=np.float32)}
                )
            raise ValueError("LingBot requires sampling_params.extra_args['robot_obs']")
        inputs = self.processor.build_model_inputs(obs)
        actions = self.model.sample_actions(
            **inputs,
            num_steps=req.sampling_params.num_inference_steps,
            generator=req.sampling_params.generator,
        )
        return DiffusionOutput(output={"actions": self.processor.build_model_outputs(actions)})
