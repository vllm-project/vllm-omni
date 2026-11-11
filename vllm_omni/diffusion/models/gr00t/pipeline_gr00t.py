# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections.abc import Iterable, Mapping
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.gr00t.policy import Gr00tPolicy
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

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

    The request's `seed` (materialised by the runner into `sampling_params.generator`)
    draws the flow-matching noise, so repeating a request with the same seed reproduces
    its action chunk.
    """

    # forward() serves the whole scheduler wave: single-observation requests are
    # concatenated along the batch axis and served by one Gr00tPolicy.get_action()
    # call. The bundled deploy config keeps max_num_seqs: 1, so a wave holds one
    # request unless a deployment raises it.
    supports_request_batch = True

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        model_config = od_config.model_config
        self.model_path = od_config.model
        self.embodiment_tag = str(model_config.get("embodiment_tag") or "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT")
        self.strict = bool(model_config.get("strict", True))
        self.device = get_local_device()

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
        """Run policy inference for every request in the wave.

        Returns one DiffusionOutput per request, in ``req.requests`` order. A malformed
        request fails on its own and does not affect the rest of the wave. ``reset`` is
        applied before inference; the policy is shared by all sessions, so a reset from
        any request applies to the whole wave.
        """
        del kwargs
        outputs: list[DiffusionOutput | None] = [None] * req.num_reqs
        pending: list[tuple[int, dict[str, Any], torch.Generator | None]] = []
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
            policy_obs = _normalize_observation(robot_obs, language_key=self.policy.language_key)
            pending.append((idx, policy_obs, _noise_generator(request.sampling_params, self.device)))

        if len(pending) == 1:
            # A single request keeps the unbatched path; its exceptions reach the runner as before.
            idx, policy_obs, generator = pending[0]
            outputs[idx] = self._run_policy(policy_obs, generator)
        elif pending:
            for (idx, _, _), output in zip(pending, self._run_policy_batch(pending)):
                outputs[idx] = output

        assert all(output is not None for output in outputs), "every request must get an output"
        return cast(list[DiffusionOutput], outputs)

    def _run_policy(self, policy_obs: dict[str, Any], generator: torch.Generator | None) -> DiffusionOutput:
        result = self.policy.get_action(policy_obs, {"generator": generator} if generator is not None else None)
        actions = result[0] if isinstance(result, tuple) else result
        if not isinstance(actions, Mapping):
            return DiffusionOutput(error=f"GR00T policy returned {type(actions).__name__}; expected dict actions.")
        # Return actions via output.output (like the DreamZero OpenPI policy) so the engine's
        # empty-output guard passes.
        return DiffusionOutput(output={"actions": _to_float32_action_dict(actions)})

    def _run_policy_batch(
        self, pending: list[tuple[int, dict[str, Any], torch.Generator | None]]
    ) -> list[DiffusionOutput]:
        """Serve several requests with one ``get_action`` call when their observations allow it.

        Each request's generator draws its own sample's noise, so a request gets the same
        initial noise as when it is served alone. If the observations cannot be merged, or
        the batched call fails, every request is served on its own and only the failing
        ones report an error.
        """
        observations = [policy_obs for _, policy_obs, _ in pending]
        generators = [generator for _, _, generator in pending]
        merged = _merge_observations(observations)
        if merged is not None:
            options = {"generator": generators} if any(g is not None for g in generators) else None
            # The batched call draws each request's noise from its generator before it can
            # fail, so keep the states to give a retried request the noise it draws alone.
            generator_states = [g.get_state() if g is not None else None for g in generators]
            try:
                result = self.policy.get_action(merged, options)
                actions = result[0] if isinstance(result, tuple) else result
                if not isinstance(actions, Mapping):
                    raise TypeError(f"GR00T policy returned {type(actions).__name__}; expected dict actions.")
                per_request = _split_actions(_to_float32_action_dict(actions), len(pending))
            # The runner does not isolate exceptions per request, so a failure here would fail
            # every robot in the wave. Retry one request at a time so only the culprit errors.
            except Exception:
                logger.warning(
                    "GR00T: batched inference of %d requests failed; serving them one at a time.",
                    len(pending),
                    exc_info=True,
                )
                for generator, state in zip(generators, generator_states):
                    if generator is not None and state is not None:
                        generator.set_state(state)
            else:
                return [DiffusionOutput(output={"actions": actions_i}) for actions_i in per_request]
        else:
            logger.debug("GR00T: %d observations cannot be merged; serving them one at a time.", len(pending))

        outputs = []
        for policy_obs, generator in zip(observations, generators):
            try:
                outputs.append(self._run_policy(policy_obs, generator))
            # Reported on this request only; the other requests in the wave still get actions.
            except Exception as exc:
                outputs.append(DiffusionOutput(error=f"GR00T policy inference failed: {exc}"))
        return outputs


def _noise_generator(
    sampling_params: OmniDiffusionSamplingParams, device: torch.device | str
) -> torch.Generator | None:
    """Pick the generator that draws the action head's initial noise.

    The runner materialises ``sampling_params.seed`` into ``sampling_params.generator`` on
    the model device before every forward, and ``OmniDiffusionRequest`` assigns a random
    seed when the caller omits one, so in serving the generator is always present. The
    seed fallback serves callers that bypass the runner (direct use, unit tests). ``None``
    leaves the action head on the global RNG, as upstream Isaac-GR00T runs. GR00T draws one
    noise tensor per request, so a generator in any other form (for example a list of several
    generators) is ignored with a warning and the seed fallback applies.
    """
    generator = sampling_params.generator
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]
    if isinstance(generator, torch.Generator):
        return generator
    if generator is not None:
        received = f"a list of {len(generator)} generators" if isinstance(generator, list) else type(generator).__name__
        logger.warning(
            "GR00T expects a single torch.Generator per request but got %s; ignoring it. The initial noise "
            "falls back to sampling_params.seed=%s, or to the global RNG when that is None.",
            received,
            sampling_params.seed,
        )
    if sampling_params.seed is not None:
        return torch.Generator(device=device).manual_seed(int(sampling_params.seed))
    return None


def _merge_observations(observations: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Concatenate single-sample observations along the batch axis.

    Mirrors ``Gr00tPolicy._unbatch_observation``: video and state arrays carry a leading
    batch dimension and language is a per-sample list. Returns ``None`` when the
    observations cannot be merged: different modalities or keys, a sample with a batch
    size other than one, or mismatched trailing shapes.
    """
    first = observations[0]
    if not isinstance(first.get("video"), Mapping) or not first["video"]:
        return None
    try:
        for obs in observations:
            if set(obs) != set(first):
                return None
            for modality in first:
                if not isinstance(obs[modality], Mapping) or set(obs[modality]) != set(first[modality]):
                    return None
                # A value without a length (None, a scalar) raises TypeError here, so the
                # wave falls back to one call per request and only that request errors.
                if any(len(value) != 1 for value in obs[modality].values()):
                    return None
        merged: dict[str, Any] = {}
        for modality, keys in first.items():
            if modality == "language":
                merged[modality] = {
                    key: [sample for obs in observations for sample in obs[modality][key]] for key in keys
                }
            else:
                merged[modality] = {
                    key: np.concatenate([np.asarray(obs[modality][key]) for obs in observations], axis=0)
                    for key in keys
                }
    except (TypeError, ValueError):
        return None
    return merged


def _split_actions(actions: dict[str, np.ndarray], num_requests: int) -> list[dict[str, np.ndarray]]:
    """Slice a batched action dict into one single-sample action dict per request."""
    for key, value in actions.items():
        if value.shape[0] != num_requests:
            raise RuntimeError(
                f"GR00T policy returned action {key!r} with batch size {value.shape[0]} for {num_requests} requests."
            )
    return [{key: value[i : i + 1] for key, value in actions.items()} for i in range(num_requests)]


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
