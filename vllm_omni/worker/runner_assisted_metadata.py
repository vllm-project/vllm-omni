# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple, Protocol


class RunnerAssistedFullAttentionMetadataRequest(NamedTuple):
    num_reqs_padded: int
    for_cudagraph_capture: bool


class RunnerAssistedAttentionMetadataProvider(Protocol):
    """Optional model hook for runner-built full attention metadata."""

    def get_runner_assisted_full_attention_metadata_request(
        self,
        *,
        req_ids: Sequence[str],
        num_reqs: int,
        num_scheduled_tokens: Sequence[int],
        num_computed_tokens: Sequence[int],
        max_num_scheduled_tokens: int,
    ) -> RunnerAssistedFullAttentionMetadataRequest | None: ...

    def set_runner_assisted_full_attention_metadata_context(
        self,
        *,
        enabled: bool,
        num_reqs: int = 0,
    ) -> None: ...
