# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Author: wanren
# Date: 2025/10/21
from .re_utils import merge_punc_measure


class REDecimal:
    name = "decimal"

    def __init__(self, int_part, frac_part, range_digit=None):
        self.txt = f"{int_part.txt}.{frac_part.txt}"
        self.int_part, self.frac_part = int_part, frac_part
        self.range_digit = range_digit

    def to_string(self):
        if self.range_digit:
            return f"{self.range_digit.to_string()}至{self.int_part.to_string()}点{self.frac_part.to_string_direct()}"
        return f"{self.int_part.to_string()}点{self.frac_part.to_string_direct()}"

    @classmethod
    def score_digits(cls, stack, idx):
        digits = stack[idx]
        if len(digits) == 3 and digits[1].txt == ".":
            return 50
        return -1

    @classmethod
    def deal_digits(cls, stack, idx):
        digits = stack[idx]
        stack[idx] = REDecimal(digits[0], digits[2])
        return stack, idx + 1

    @classmethod
    def merge_punc(cls, stack, idx):
        return merge_punc_measure(stack, idx)
