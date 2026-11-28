# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A condition reservation contains no model tensors and consumes no GPU slot."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RealizationConditionTicket:
    session_id: str
    fence: int
    block: int
    tenant: str = "local"

    def __post_init__(self):
        if (
            not isinstance(self.session_id, str)
            or not self.session_id
            or not isinstance(self.tenant, str)
            or not self.tenant
            or type(self.fence) is not int
            or self.fence < 0
            or type(self.block) is not int
            or self.block < 0
        ):
            raise ValueError("Invalid Realization condition ticket")
