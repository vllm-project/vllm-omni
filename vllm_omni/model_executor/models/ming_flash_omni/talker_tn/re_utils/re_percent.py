# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Author: wanren
# Date: 2025/10/21
from .re_utils import merge_punc_measure


class REPercent:
    name = "percent"

    def __init__(self, digit, range_digit=None):
        self.digit = digit
        self.range_digit = range_digit

    def to_string(self):
        if self.range_digit:
            return f"百分之{self.range_digit.to_string()}至{self.digit.to_string()}"
        return f"百分之{self.digit.to_string()}"

    @classmethod
    def try_digit(cls, stack, idx):
        if idx + 1 < len(stack) and stack[idx + 1].txt.startswith("%"):
            stack[idx + 1].txt = stack[idx + 1].txt[1:]
            stack[idx] = REPercent(stack[idx])
            return True
        return False

    @classmethod
    def merge_punc(cls, stack, idx):
        return merge_punc_measure(stack, idx)
