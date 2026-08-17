# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenPI robot-serving adapter for OpenVLA.

Supplies the three model-specific pieces the generic OpenPI serving layer
cannot know for a token-based action policy: what to advertise in the
handshake, how an observation becomes a prompt, and how the generated token ids
become an action array.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.transformers_utils.processors.openvla import to_rgb_image

from vllm_omni.entrypoints.openpi.adapters import RobotARRequest
from vllm_omni.model_executor.models.openvla.action_decode import (
    OpenVLAActionDecoder,
    build_prompt_token_ids,
)

# OpenPI clients are not consistent about where the policy camera lives; this
# is a preference order, most specific first.
_IMAGE_KEYS = (
    "image",
    "observation/image",
    "observation.image",
    "observation/exterior_image_1_left",
    "base_image",
    "primary_image",
)
_PROMPT_KEYS = ("prompt", "instruction", "task")


def _first_present(obs: dict, keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = obs.get(key)
        if value is not None:
            return value
    return None


def _as_text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


class OpenVLARobotAdapter:
    """Bridges one OpenPI observation to one autoregressive OpenVLA request."""

    def __init__(self, decoder: OpenVLAActionDecoder) -> None:
        self.decoder = decoder

    @classmethod
    def from_engine_client(cls, engine_client: Any) -> OpenVLARobotAdapter:
        model_config = getattr(engine_client, "model_config", None)
        hf_config = getattr(model_config, "hf_config", None)
        # openvla-7b ships statistics for 25 embodiments, so a deployment has to
        # pick one; `hf_overrides: {unnorm_key: ...}` in the deploy config is
        # how, and a request may still override it per observation.
        unnorm_key = getattr(hf_config, "unnorm_key", None)
        return cls(OpenVLAActionDecoder.from_hf_config(hf_config, unnorm_key))

    def policy_server_values(self) -> dict[str, Any]:
        return self.decoder.policy_server_values()

    def build_request(self, obs: dict, *, tokenizer: Any, request_id: str) -> RobotARRequest:
        raw_image = _first_present(obs, _IMAGE_KEYS)
        if raw_image is None:
            raise ValueError(
                f"OpenVLA observation needs an image under one of {list(_IMAGE_KEYS)}; got keys {sorted(obs)}"
            )
        instruction = _as_text(_first_present(obs, _PROMPT_KEYS) or "")
        if not instruction.strip():
            raise ValueError(f"OpenVLA observation needs a language instruction under one of {list(_PROMPT_KEYS)}")

        action_dim = self.decoder.action_dim(obs.get("unnorm_key"))
        return RobotARRequest(
            prompt={
                # Token ids rather than text: the checkpoint expects a trailing
                # empty token that no chat template produces.
                "prompt_token_ids": build_prompt_token_ids(tokenizer, instruction),
                # to_rgb_image normalises PIL / ndarray / tensor to a PIL image.
                # Handing vLLM a bare 3-D torch tensor would be parsed as image
                # *embeddings* instead of an image.
                "multi_modal_data": {"image": to_rgb_image(raw_image)},
            },
            sampling_params=SamplingParams(
                temperature=0.0,
                max_tokens=action_dim,
                min_tokens=action_dim,
                ignore_eos=True,
                detokenize=False,
                output_kind=RequestOutputKind.FINAL_ONLY,
            ),
            request_id=request_id,
        )

    def decode_actions(self, result: Any, unnorm_key: str | None = None) -> np.ndarray:
        outputs = getattr(result, "outputs", None) or []
        if not outputs:
            raise RuntimeError("OpenVLA request produced no completion output.")
        completion = outputs[0]
        token_ids = list(getattr(completion, "token_ids", None) or ())
        if not token_ids:
            # Populated on every completion output; `token_ids` is a delta under
            # a streaming output kind.
            token_ids = list(getattr(completion, "cumulative_token_ids", None) or ())
        if not token_ids:
            raise RuntimeError("OpenVLA request produced no action token ids.")
        # (action_horizon, action_dim). OpenVLA predicts one step at a time, so
        # the horizon is 1, but the wire shape matches the other robot policies.
        return self.decoder.decode(token_ids, unnorm_key).reshape(1, -1)
