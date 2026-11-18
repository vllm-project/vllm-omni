# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Author: wanren
# Date: 2025/10/21
from .re_utils import merge_punc_measure


class REFrac:
    name = "frac"

    def __init__(self, num, den, range_digit=None):
        self.num, self.den = num, den
        self.range_digit = range_digit

    def to_string(self):
        if self.range_digit:
            return f"{self.range_digit.to_string()}至{self.den.to_string()}分之{self.num.to_string()}"
        return f"{self.den.to_string()}分之{self.num.to_string()}"

    @classmethod
    def score_digits(cls, stack, idx):
        digits = stack[idx]
        if len(digits) == 3 and digits[1].txt == "/":
            return 50
        elif len(digits) == 5 and digits[1].txt in {"-", "~", "～"} and digits[3].txt == "/":
            return 50
        return -1

    @classmethod
    def deal_digits(cls, stack, idx):
        digits = stack[idx]
        if len(digits) == 3:
            stack[idx] = REFrac(digits[0], digits[2])
        else:
            assert len(digits) == 5
            stack[idx] = REFrac(digits[2], digits[4], digits[0])
        return stack, idx + 1

    @classmethod
    def merge_punc(cls, stack, idx):
        return merge_punc_measure(stack, idx)
