# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Author: wanren
# Date: 2025/10/21
from .re_base import REBase


class REChar(REBase):
    name = "char"

    def __init__(self, txt):
        super().__init__(txt)

    def to_string(self):
        return self.txt
