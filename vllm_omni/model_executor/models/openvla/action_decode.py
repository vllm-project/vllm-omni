# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Turn OpenVLA's generated token ids back into a robot action.

OpenVLA does not have an action head. It discretises each action dimension into
256 uniform bins and maps bin *i* onto the language-model token
``vocab_size - 1 - i``, i.e. actions live in the last 256 ids of the Llama-2
vocabulary. Generation therefore produces seven ordinary tokens and the policy
output is whatever those ids decode to.

Upstream vLLM runs the model but stops at the token ids: it reads
``n_action_bins`` off the config and never uses it, and it carries no
``norm_stats``. Everything below is the missing half, and it is pure CPU numpy
so it is unit-testable without a GPU.

The arithmetic is transcribed from ``OpenVLAForActionPrediction.predict_action``
in the reference checkpoint (``openvla/openvla-7b``,
``modeling_prismatic.py``); the two details that are easy to get wrong are that
``vocab_size`` here is the *unpadded* 32000 rather than the checkpoint's 32064,
and that ``mask`` is applied per dimension — in every one of openvla-7b's 25
embodiments its last element is ``False``, so the gripper dimension is returned
as a raw bin centre and is never un-normalised.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

# The SentencePiece "▁" piece. Training wrapped the assistant turn as
# ``"In: {msg}\nOut: "`` and the prompt builder right-strips it, so inference
# has to put the space back explicitly or the model sees a prompt it never saw
# during training.
OPENVLA_EMPTY_TOKEN_ID = 29871

PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"


def build_prompt(instruction: str) -> str:
    """The exact inference-time prompt the checkpoint was trained for."""
    return PROMPT_TEMPLATE.format(instruction=instruction.lower())


def build_prompt_token_ids(tokenizer: Any, instruction: str) -> list[int]:
    """Tokenise the action prompt and append the trailing empty token.

    ``encode`` rather than ``__call__``: the tokenizer protocol guarantees
    ``encode`` returns a flat ``list[int]``, while ``__call__`` returns a
    backend-dependent ``BatchEncoding``.
    """
    token_ids = list(tokenizer.encode(build_prompt(instruction), add_special_tokens=True))
    if not token_ids or token_ids[-1] != OPENVLA_EMPTY_TOKEN_ID:
        token_ids.append(OPENVLA_EMPTY_TOKEN_ID)
    return token_ids


@dataclass(frozen=True)
class OpenVLAActionDecoder:
    """Decodes action token ids for one checkpoint's embodiment statistics."""

    norm_stats: Mapping[str, Any]
    bin_centers: np.ndarray
    # Not the config's ``vocab_size``: the checkpoint pads the embedding table
    # up to a multiple of 64, and the bin mapping was fitted before that padding.
    vocab_size: int
    # ``None`` when the checkpoint carries more than one embodiment and the
    # deployment did not pick one; requests then have to name it themselves.
    default_unnorm_key: str | None

    @classmethod
    def from_hf_config(
        cls,
        hf_config: Any,
        default_unnorm_key: str | None = None,
    ) -> OpenVLAActionDecoder:
        norm_stats = getattr(hf_config, "norm_stats", None)
        if not norm_stats:
            raise ValueError(
                "OpenVLA action decoding needs `norm_stats` on the HF config; "
                "the checkpoint appears not to be an OpenVLA action policy."
            )
        n_bins = int(getattr(hf_config, "n_action_bins", 256))
        bins = np.linspace(-1.0, 1.0, n_bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0

        vocab_size = cls._resolve_vocab_size(hf_config)

        if default_unnorm_key is None and len(norm_stats) == 1:
            default_unnorm_key = next(iter(norm_stats))
        elif default_unnorm_key is not None and default_unnorm_key not in norm_stats:
            raise ValueError(f"Unknown unnorm_key {default_unnorm_key!r}; available: {sorted(norm_stats)}")

        return cls(
            norm_stats=norm_stats,
            bin_centers=bin_centers,
            vocab_size=vocab_size,
            default_unnorm_key=default_unnorm_key,
        )

    @staticmethod
    def _resolve_vocab_size(hf_config: Any) -> int:
        """The *unpadded* vocab size, which is where the bin mapping starts.

        The checkpoint pads the embedding table to a multiple of 64, but the
        action-token mapping was fitted before that padding: for openvla-7b it
        is 32000, not the config's 32064. Getting this wrong is not a crash —
        using the padded value shifts every bin index by exactly the padding
        (+64 here) and saturates the top of the range, so the policy returns a
        systematically wrong action and reports nothing. That is why both
        sources are required to agree rather than one being defaulted away.
        """
        text_config = getattr(hf_config, "text_config", None)
        padded = int(getattr(text_config, "vocab_size", 0) or 0)
        pad = getattr(hf_config, "pad_to_multiple_of", None)
        if not padded or pad is None:
            raise ValueError(
                "Cannot derive OpenVLA's unpadded vocab size: the HF config needs "
                "both text_config.vocab_size and pad_to_multiple_of."
            )
        vocab_size = padded - int(pad)
        if vocab_size <= 0:
            raise ValueError(f"Nonsensical OpenVLA vocab size: {padded} - {pad} = {vocab_size}.")
        return vocab_size

    def _resolve_unnorm_key(self, unnorm_key: str | None) -> str:
        key = unnorm_key or self.default_unnorm_key
        if key is None:
            raise ValueError(
                "This checkpoint carries action statistics for "
                f"{len(self.norm_stats)} embodiments, so the request must name one "
                f"(`unnorm_key`). Available: {sorted(self.norm_stats)}"
            )
        if key not in self.norm_stats:
            raise ValueError(f"Unknown unnorm_key {key!r}; available: {sorted(self.norm_stats)}")
        return key

    @property
    def embodiments(self) -> tuple[str, ...]:
        return tuple(sorted(self.norm_stats))

    def action_dim(self, unnorm_key: str | None = None) -> int:
        if unnorm_key is None and self.default_unnorm_key is None:
            dims = {len(v["action"]["q01"]) for v in self.norm_stats.values()}
            if len(dims) == 1:
                return dims.pop()
        return len(self.norm_stats[self._resolve_unnorm_key(unnorm_key)]["action"]["q01"])

    def decode(
        self,
        token_ids: Sequence[int],
        unnorm_key: str | None = None,
    ) -> np.ndarray:
        """Seven token ids in, one un-normalised action vector out."""
        key = self._resolve_unnorm_key(unnorm_key)
        stats = self.norm_stats[key]["action"]
        q01 = np.asarray(stats["q01"], dtype=np.float64)
        q99 = np.asarray(stats["q99"], dtype=np.float64)
        mask = np.asarray(stats.get("mask", np.ones_like(q01, dtype=bool)), dtype=bool)

        expected = len(q01)
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.shape != (expected,):
            raise ValueError(
                f"Expected {expected} action tokens for embodiment {key!r}, got {ids.shape[0] if ids.ndim else 0}."
            )

        # Bin index counts down from the top of the vocabulary. Clipping matches
        # the reference: a token outside the action range is folded onto the
        # nearest bin rather than raising.
        discretized = np.clip(self.vocab_size - ids - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized = self.bin_centers[discretized]
        actions = np.where(mask, 0.5 * (normalized + 1.0) * (q99 - q01) + q01, normalized)
        return actions.astype(np.float32)

    def policy_server_values(self, image_resolution: Sequence[int] = (224, 224)) -> dict[str, Any]:
        """Handshake payload for the OpenPI client.

        Everything here is read off the checkpoint rather than configured, so a
        fine-tuned OpenVLA with a different embodiment set advertises itself
        correctly without touching the deploy config.
        """
        return {
            "image_resolution": list(image_resolution),
            "needs_session_id": False,
            "action_horizon": 1,
            "action_dim": self.action_dim(),
            "action_space": "delta_eef_gripper",
            "unnorm_key": self.default_unnorm_key,
            "supported_embodiments": list(self.embodiments),
        }
