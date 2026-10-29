# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""FastWAM-family robot policy pipeline (CAE-FastWAM by default).

vLLM-Omni owns the serving integration: OpenPI observations arrive in
``sampling_params.extra_args["robot_obs"]`` (see
:mod:`~vllm_omni.diffusion.models.fastwam.processing_fastwam` for the keys), the
pipeline predicts one action chunk per request, and returns it as a float32
``[action_horizon, action_dim]`` array in ``DiffusionOutput.output["actions"]``.

FastWAM keeps no state between requests, so ``reset`` and ``session_id`` are
accepted and ignored. The request ``seed`` seeds a CPU generator for the initial
action noise, as the reference evaluator does, so the same seed and observation
give the same chunk on any device. Without a seed (the engine then assigns a
random one) the noise differs per request.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.fastwam.config import FastWAMConfig, resolve_config
from vllm_omni.diffusion.models.fastwam.policy import (
    _BASE_MODEL_PATTERNS,
    DEFAULT_BASE_MODEL,
    FastWAMPolicy,
    checkpoint_patterns,
    resolve_local_dir,
)
from vllm_omni.diffusion.models.fastwam.processing_fastwam import (
    ACTION_POSTPROCESS_MODES,
    FastWAMObservation,
    parse_observation,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

logger = init_logger(__name__)

# bfloat16 is the reference (and validated) dtype; float32 is accepted for
# debugging and CPU parity work.
SUPPORTED_DTYPES = (torch.bfloat16, torch.float32)
WARMUP_TASK = "warm up the policy"


def _resolve_dtype(dtype: Any) -> torch.dtype:
    resolved = dtype if isinstance(dtype, torch.dtype) else getattr(torch, str(dtype).split(".")[-1], None)
    if resolved not in SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported FastWAM dtype {dtype!r}; use bfloat16 (validated) or float32.")
    return resolved


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int(value)


def _noise_generator(sampling_params: OmniDiffusionSamplingParams) -> torch.Generator | None:
    """CPU generator for the initial action noise.

    The reference draws the noise in float32 on a CPU generator seeded with the
    evaluator seed, then moves it to the device. Seeding a CPU generator from
    ``sampling_params.seed`` here (rather than using the runner's device-side
    generator) reproduces the reference chunk for the same seed on any device.
    """
    if sampling_params.seed is not None:
        return torch.Generator(device="cpu").manual_seed(int(sampling_params.seed))
    generator = sampling_params.generator
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]
    if isinstance(generator, torch.Generator) and generator.device.type == "cpu":
        return generator
    return None


class FastWAMPipeline(nn.Module):
    """FastWAM world-action model served as an OpenPI robot policy.

    Registered as ``"FastWAMPipeline"``. The variant is chosen by
    ``model_config.fastwam_variant`` (default ``cae_fastwam_libero``).
    """

    # ``enforce_eager: false`` compiles every transformer layer of both experts
    # (``FastWAMModel._repeated_blocks``) with vLLM-Omni's regional torch.compile.
    _dit_modules = ("model",)

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        model_config = dict(od_config.model_config or {})
        cfg = resolve_config(model_config)
        dtype = _resolve_dtype(od_config.dtype)
        if dtype != torch.bfloat16:
            logger.warning("FastWAM was validated in bfloat16; running in %s.", dtype)
        if not cfg.validated:
            logger.warning(
                "FastWAM variant %s loads through the same code as CAE-FastWAM but has not been validated "
                "(no parity or success-rate check yet).",
                cfg.variant,
            )
        self.num_inference_steps = _positive_int(model_config.get("num_inference_steps", 10), "num_inference_steps")
        action_postprocess = str(model_config.get("action_postprocess", "libero"))
        if action_postprocess not in ACTION_POSTPROCESS_MODES:
            raise ValueError(
                f"action_postprocess must be one of {ACTION_POSTPROCESS_MODES}, got {action_postprocess!r}."
            )
        self.warmup = bool(model_config.get("warmup", True))

        if not od_config.model:
            raise ValueError("FastWAMPipeline needs od_config.model (a FastWAM checkpoint directory or Hub ID).")
        model_dir = resolve_local_dir(od_config.model, checkpoint_patterns(cfg), od_config.revision)
        base_model_dir = resolve_local_dir(
            str(model_config.get("base_model") or DEFAULT_BASE_MODEL), _BASE_MODEL_PATTERNS
        )
        self.policy = FastWAMPolicy(
            cfg=cfg,
            model_dir=model_dir,
            base_model_dir=base_model_dir,
            device=get_local_device(),
            dtype=dtype,
            action_postprocess=action_postprocess,
            default_suite=model_config.get("task_suite"),
        )
        # Registered as a child so the runner sees the parameters and can compile it.
        self.model = self.policy.model
        self._validate_policy_server_config(model_config.get("policy_server_config"))

    @property
    def cfg(self) -> FastWAMConfig:
        return self.policy.cfg

    def _validate_policy_server_config(self, psc: Mapping[str, Any] | None) -> None:
        """Fail fast when the OpenPI handshake disagrees with the loaded checkpoint."""
        if not isinstance(psc, Mapping):
            return
        cfg = self.cfg
        expected: dict[str, Any] = {
            "action_horizon": cfg.action_horizon,
            "action_dim": cfg.action_dim,
            "image_resolution": list(cfg.camera_size),
            "needs_wrist_camera": cfg.num_cameras > 1,
            "n_external_cameras": 1,
            "needs_session_id": False,
        }
        for key, value in expected.items():
            if key in psc:
                got = list(psc[key]) if isinstance(psc[key], (list, tuple)) else psc[key]
                if got != value:
                    raise ValueError(f"policy_server_config.{key}={psc[key]!r} != loaded FastWAM model's {value!r}.")

    @property
    def weights_sources(self) -> tuple[Any, ...]:
        return ()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Weights are loaded strictly by :class:`FastWAMPolicy` at construction."""
        consumed = list(weights)
        if consumed:
            raise RuntimeError(
                f"FastWAMPipeline.load_weights received {len(consumed)} tensors; weights_sources=() should "
                "prevent this. FastWAM loads its checkpoints itself."
            )
        return {name for name, _ in self.named_parameters()}

    def reset(self) -> None:
        """FastWAM is stateless across requests."""

    def _warmup_observation(self) -> FastWAMObservation:
        height, width = self.cfg.camera_size
        image = np.zeros((height, width, 3), dtype=np.uint8)
        return FastWAMObservation(
            images=(image,) * self.cfg.num_cameras,
            state=np.zeros((self.cfg.proprio_dim,), dtype=np.float32),
            task=WARMUP_TASK,
            suite=self.policy.default_suite,
        )

    def _dummy_output(self) -> DiffusionOutput:
        if self.warmup:
            # A real forward, so a regional-compile deployment compiles at startup
            # instead of on the first robot request.
            actions = self.policy.infer(
                self._warmup_observation(),
                num_steps=self.num_inference_steps,
                generator=torch.Generator(device="cpu").manual_seed(0),
            )
        else:
            actions = np.zeros((self.cfg.action_horizon, self.cfg.action_dim), dtype=np.float32)
        return DiffusionOutput(output={"actions": actions})

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch, **kwargs) -> DiffusionOutput:
        del kwargs
        extra_args = req.sampling_params.extra_args or {}
        robot_obs = extra_args.get("robot_obs")
        if robot_obs is None:
            if OmniDiffusionRequest.is_dummy_run_request_id(req.request_id):
                return self._dummy_output()
            return DiffusionOutput(error="FastWAMPipeline.forward expects sampling_params.extra_args['robot_obs'].")
        if not isinstance(robot_obs, Mapping):
            return DiffusionOutput(error=f"robot_obs must be a dict, got {type(robot_obs).__name__}.")
        try:
            obs = parse_observation(
                robot_obs,
                proprio_dim=self.cfg.proprio_dim,
                num_cameras=self.cfg.num_cameras,
                default_suite=self.policy.default_suite,
            )
            num_steps = req.sampling_params.num_inference_steps
            num_steps = (
                self.num_inference_steps if num_steps is None else _positive_int(num_steps, "num_inference_steps")
            )
        except ValueError as exc:
            return DiffusionOutput(error=f"Invalid FastWAM request: {exc}")
        if obs.suite not in self.cfg.suite_ids and self.cfg.suite_conditioned_stats:
            return DiffusionOutput(
                error=f"Unknown task_suite {obs.suite!r}; expected one of {sorted(self.cfg.suite_ids)}."
            )
        actions = self.policy.infer(obs, num_steps=num_steps, generator=_noise_generator(req.sampling_params))
        return DiffusionOutput(output={"actions": actions})
