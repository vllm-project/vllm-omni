# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Author: wanren
# Date: 2025/10/21


class REBase:
    name = "base"

    def __init__(self, txt):
        self.txt = txt

    def __repr__(self):
        return f"{self.name}: {self.txt}"

    def append_txt(self, txt):
        self.txt += txt

    @classmethod
    def merge_punc(cls, stack, idx):
        return stack, idx + 1
