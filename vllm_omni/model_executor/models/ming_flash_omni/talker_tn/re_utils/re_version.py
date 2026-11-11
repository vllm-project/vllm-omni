# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Author: wanren
# Date: 2025/10/21


class REVersion:
    name = "version"

    def __init__(self, ver_s):
        self.ver_s = ver_s

    def to_string(self):
        return "点".join(ii.to_string() for ii in self.ver_s)

    @classmethod
    def score_digits(cls, stack, idx):
        digits = stack[idx]
        if len(digits) == 5 and digits[1].txt == "." and digits[3].txt == ".":
            if idx - 1 >= 0:
                if stack[idx - 1].txt[-1] in {"v", "V"}:
                    return 90
                if "版本" in stack[idx - 1].txt:
                    return 90
            return 50
        return -1

    @classmethod
    def deal_digits(cls, stack, idx):
        digits = stack[idx]
        stack[idx] = REVersion([digits[0], digits[2], digits[4]])
        return stack, idx + 1

    @classmethod
    def merge_punc(cls, stack, idx):
        return stack, idx + 1
