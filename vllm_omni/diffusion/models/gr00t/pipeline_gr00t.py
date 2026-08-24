# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.gr00t.policy import Gr00tPolicy
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

logger = init_logger(__name__)


def _to_float32_action_dict(actions: Mapping[str, Any]) -> dict[str, np.ndarray]:
    converted = {str(key): np.asarray(value, dtype=np.float32) for key, value in actions.items()}
    if not converted:
        raise RuntimeError("GR00T policy returned an empty action dict.")
    return converted


class Gr00tN1d7Pipeline(nn.Module):
    """GR00T N1.7 policy pipeline backed by vLLM-Omni's local GR00T port.

    vLLM-Omni owns the serving integration: OpenPI observations arrive through
    `sampling_params.extra_args["robot_obs"]`, this pipeline runs GR00T policy
    inference, and actions are returned through `DiffusionOutput.output["actions"]`.
    """

    # forward() consumes the whole DiffusionRequestBatch: per-request
    # observations are concatenated along the batch axis and served by one
    # Gr00tPolicy.get_action() call. The scheduler only builds waves larger
    # than one request when the stage's max_num_seqs allows it, so the
    # bundled deploy config (max_num_seqs: 1) keeps one-request waves.
    supports_request_batch = True

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        model_config = od_config.model_config
        self.model_path = od_config.model
        self.embodiment_tag = str(model_config.get("embodiment_tag") or "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT")
        self.strict = bool(model_config.get("strict", True))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading GR00T N1.7 policy from %s with embodiment_tag=%s", self.model_path, self.embodiment_tag)
        self.policy = Gr00tPolicy(
            model_path=self.model_path,
            embodiment_tag=self.embodiment_tag,
            device=self.device,
            strict=self.strict,
        )
        self._validate_policy_server_config(model_config.get("policy_server_config"))

    def _validate_policy_server_config(self, psc: Mapping[str, Any] | None) -> None:
        """Fail fast if the deploy handshake drifts from the loaded checkpoint.

        ``policy_server_config`` is sent verbatim to the OpenPI client, so its
        model/embodiment-specific values must match what the loaded policy actually
        produces; otherwise the client is handed the wrong action contract.
        """
        if not isinstance(psc, Mapping):
            return
        action_config = self.policy.modality_configs["action"]
        expected_horizon = len(action_config.delta_indices)
        expected_keys = set(action_config.modality_keys)

        if "action_horizon" in psc and psc["action_horizon"] != expected_horizon:
            raise ValueError(
                f"policy_server_config.action_horizon={psc['action_horizon']} != loaded model's "
                f"action horizon {expected_horizon}."
            )
        if "action_keys" in psc and set(psc["action_keys"]) != expected_keys:
            raise ValueError(
                f"policy_server_config.action_keys={list(psc['action_keys'])} != loaded model's "
                f"action keys {sorted(expected_keys)}."
            )
        psc_embodiment = psc.get("embodiment_tag")
        if psc_embodiment is not None and psc_embodiment != self.embodiment_tag:
            raise ValueError(
                f"policy_server_config.embodiment_tag={psc_embodiment!r} != "
                f"model_config.embodiment_tag={self.embodiment_tag!r}."
            )

    def reset(self) -> dict[str, Any]:
        return self.policy.reset() or {}

    @property
    def weights_sources(self) -> tuple[Any, ...]:
        return ()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        consumed = list(weights)
        if consumed:
            raise RuntimeError(
                f"Gr00tN1d7Pipeline.load_weights received {len(consumed)} weight tensors; "
                "weights_sources=() should prevent this. GR00T weights are loaded directly by Gr00tPolicy."
            )
        return set()

    def _dummy_actions(self) -> dict[str, np.ndarray]:
        embodiment_value = self.policy.embodiment_tag.value
        action_config = self.policy.modality_configs["action"]
        horizon = len(action_config.delta_indices)
        norm_params = self.policy.processor.state_action_processor.norm_params[embodiment_value]["action"]
        actions = {}
        for key in action_config.modality_keys:
            dim = norm_params[key]["dim"]
            dim = int(dim.item() if hasattr(dim, "item") else dim)
            actions[key] = np.zeros((1, horizon, dim), dtype=np.float32)
        return actions

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch, **kwargs) -> list[DiffusionOutput]:
        """Run one policy inference for every request in the wave.

        Returns one DiffusionOutput per request, in ``req.requests`` order.
        Malformed requests fail individually; they never take the rest of the
        wave down with them. ``reset`` is honoured before inference — the
        policy is process-wide, so a reset requested by any session applies to
        all of them (Gr00tPolicy.reset() is stateless today, see reset()).
        """
        del kwargs
        outputs: list[DiffusionOutput | None] = [None] * req.num_reqs
        pending: list[tuple[int, dict[str, Any]]] = []

        for idx, request in enumerate(req.requests):
            extra_args = request.sampling_params.extra_args or {}
            robot_obs = extra_args.get("robot_obs")
            if robot_obs is None:
                if request.is_dummy_run_request_id(request.request_id):
                    outputs[idx] = DiffusionOutput(output={"actions": self._dummy_actions()})
                else:
                    outputs[idx] = DiffusionOutput(
                        error="Gr00tN1d7Pipeline.forward expects sampling_params.extra_args['robot_obs']."
                    )
                continue
            if not isinstance(robot_obs, Mapping):
                outputs[idx] = DiffusionOutput(error=f"robot_obs must be a dict, got {type(robot_obs).__name__}.")
                continue
            if extra_args.get("reset"):
                self.reset()
            pending.append((idx, _normalize_observation(robot_obs, language_key=self.policy.language_key)))

        if pending:
            policy_outputs = self._policy_outputs([obs for _, obs in pending])
            for (idx, _), output in zip(pending, policy_outputs):
                outputs[idx] = output

        assert all(output is not None for output in outputs)
        return outputs  # type: ignore[return-value]

    def _policy_outputs(self, obs_list: list[dict[str, Any]]) -> list[DiffusionOutput]:
        """Serve normalized observations, batched into one policy call when possible.

        A single observation takes the direct path, byte-for-byte the previous
        single-request behavior (exceptions propagate to the runner). Multiple
        observations are concatenated along the batch axis for one
        ``get_action`` call; if they are not structurally compatible, or the
        batched call fails, each observation falls back to its own call so a
        poisoned request only fails itself.
        """
        if len(obs_list) == 1:
            return [self._forward_one(obs_list[0])]

        merged = _merge_observations(obs_list)
        if merged is not None:
            batched_obs, sizes = merged
            logger.debug("GR00T request batch: %d observations in one policy call.", len(obs_list))
            try:
                result = self.policy.get_action(batched_obs)
                actions = result[0] if isinstance(result, tuple) else result
                if isinstance(actions, Mapping):
                    per_request = _split_actions(_to_float32_action_dict(actions), sizes)
                    return [DiffusionOutput(output={"actions": actions_i}) for actions_i in per_request]
                logger.warning(
                    "GR00T batched get_action returned %s; falling back to per-request inference.",
                    type(actions).__name__,
                )
            except Exception:
                logger.warning(
                    "GR00T batched get_action failed for %d observations; falling back to per-request inference.",
                    len(obs_list),
                    exc_info=True,
                )

        outputs = []
        for obs in obs_list:
            try:
                outputs.append(self._forward_one(obs))
            except Exception as exc:
                outputs.append(DiffusionOutput(error=f"GR00T policy inference failed: {exc}"))
        return outputs

    def _forward_one(self, policy_obs: dict[str, Any]) -> DiffusionOutput:
        result = self.policy.get_action(policy_obs)
        actions = result[0] if isinstance(result, tuple) else result
        if not isinstance(actions, Mapping):
            return DiffusionOutput(error=f"GR00T policy returned {type(actions).__name__}; expected dict actions.")
        # Return actions via output.output (like the DreamZero OpenPI policy) so the engine's
        # empty-output guard passes.
        return DiffusionOutput(output={"actions": _to_float32_action_dict(actions)})


def _merge_observations(
    obs_list: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[int]] | None:
    """Concatenate normalized observations along the batch axis.

    Mirrors ``Gr00tPolicy._unbatch_observation``: every array modality carries
    a leading batch dimension, language is a per-sample list. Returns the
    merged observation and each observation's batch size, or ``None`` when the
    observations are not structurally compatible (different modality or
    per-modality keys, missing video, mismatched trailing shapes) — the caller
    then serves each observation individually.
    """
    try:
        first = obs_list[0]
        if any(set(obs) != set(first) for obs in obs_list[1:]):
            return None
        for modality in first:
            if not all(isinstance(obs[modality], Mapping) for obs in obs_list):
                return None
            if any(set(obs[modality]) != set(first[modality]) for obs in obs_list[1:]):
                return None

        sizes: list[int] = []
        for obs in obs_list:
            video = obs.get("video")
            if not isinstance(video, Mapping) or not video:
                return None
            sizes.append(len(next(iter(video.values()))))

        merged: dict[str, Any] = {}
        for modality, first_value in first.items():
            if modality == "language":
                merged[modality] = {
                    key: [sample for obs in obs_list for sample in obs[modality][key]] for key in first_value
                }
                continue
            merged[modality] = {
                key: np.concatenate([np.asarray(obs[modality][key]) for obs in obs_list], axis=0) for key in first_value
            }
    except (TypeError, ValueError):
        # Malformed or mismatched observations must not take the wave down;
        # the per-observation fallback surfaces the error on the culprit only.
        return None
    return merged, sizes


def _split_actions(actions: dict[str, np.ndarray], sizes: list[int]) -> list[dict[str, np.ndarray]]:
    """Slice a batched action dict back into per-request action dicts."""
    total = sum(sizes)
    for key, value in actions.items():
        if value.shape[0] != total:
            raise RuntimeError(
                f"GR00T policy returned action '{key}' with batch size {value.shape[0]} "
                f"for a request batch of total size {total}."
            )
    per_request: list[dict[str, np.ndarray]] = []
    offset = 0
    for size in sizes:
        per_request.append({key: value[offset : offset + size] for key, value in actions.items()})
        offset += size
    return per_request


def _normalize_observation(robot_obs: Mapping[str, Any], *, language_key: str) -> dict[str, Any]:
    obs: dict[str, Any] = {}
    if "video" in robot_obs:
        obs["video"] = robot_obs["video"]
    elif "images" in robot_obs:
        obs["video"] = robot_obs["images"]
    if "state" in robot_obs:
        obs["state"] = robot_obs["state"]
    if "language" in robot_obs:
        obs["language"] = robot_obs["language"]
    else:
        prompt = robot_obs.get("prompt")
        if prompt is not None:
            obs["language"] = {language_key: [[str(prompt)]]}
    return obs
