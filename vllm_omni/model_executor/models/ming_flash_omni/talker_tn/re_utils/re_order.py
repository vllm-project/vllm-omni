# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Author: wanren
# Date: 2025/10/22


class REOrder:
    name = "order"

    def __init__(self, digit):
        self.digit = digit

    def to_string(self):
        return f"{self.digit.to_string()}"

    @classmethod
    def try_digit(cls, stack, idx):
        if (
            len(stack[idx].txt) < 2
            and idx + 2 < len(stack)
            and stack[idx + 1].txt in {"."}
            and stack[idx + 2].txt.startswith(" ")
        ):
            stack[idx] = REOrder(stack[idx])
            return True
        return False

    @classmethod
    def merge_punc(cls, stack, idx):
        return stack, idx + 1
