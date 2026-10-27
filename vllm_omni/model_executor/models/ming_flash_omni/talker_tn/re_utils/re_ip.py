# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Author: wanren
# Date: 2025/10/21
from .re_utils import merge_punc_range


class REIp:
    name = "ip"

    def __init__(self, ip_s):
        self.ip_s = ip_s

    def to_string(self):
        return "点".join(ii.to_string_yao() for ii in self.ip_s)

    @classmethod
    def score_digits(cls, stack, idx):
        digits = stack[idx]
        if len(digits) == 7 and digits[1].txt == digits[3].txt == digits[5].txt == ".":
            return 50
        return -1

    @classmethod
    def deal_digits(cls, stack, idx):
        digits = stack[idx]
        stack[idx] = REIp([digits[0], digits[2], digits[4], digits[6]])
        return stack, idx + 1

    @classmethod
    def merge_punc(cls, stack, idx):
        return merge_punc_range(stack, idx)
