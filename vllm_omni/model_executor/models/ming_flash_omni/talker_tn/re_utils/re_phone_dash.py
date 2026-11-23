# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Author: wanren
# Date: 2025/10/21


class REPhoneDash:
    name = "phone_dash"

    def __init__(self, num_s):
        self.num_s = num_s

    def to_string(self):
        return "".join(ii.to_string_yao() for ii in self.num_s)

    @classmethod
    def score_digits(cls, stack, idx):
        digits = stack[idx]
        if len(digits) == 3 and len(digits[0].txt) == 3 and digits[1].txt == "-" and len(digits[2].txt) == 8:
            return 80
        if len(digits) == 3 and len(digits[0].txt) == 4 and digits[1].txt == "-" and len(digits[2].txt) in {7, 8}:
            return 80
        if (
            len(digits) == 5
            and (digits[0].txt == "400" or (len(digits[0].txt) in {3, 4} and digits[0].txt[0] == "0"))
            and digits[1].txt == digits[3].txt == "-"
            and len(digits[2].txt) in {3, 4}
            and len(digits[4].txt) == 4
        ):
            return 90
        return -1

    @classmethod
    def deal_digits(cls, stack, idx):
        digits = stack[idx]
        if len(digits) == 3:
            stack[idx] = REPhoneDash([digits[0], digits[2]])
        else:
            assert len(digits) == 5
            stack[idx] = REPhoneDash([digits[0], digits[2], digits[4]])
        return stack, idx + 1

    @classmethod
    def merge_punc(cls, stack, idx):
        return stack, idx + 1
