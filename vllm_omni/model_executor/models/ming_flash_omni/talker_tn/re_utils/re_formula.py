# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Author: wanren
# Date: 2025/10/21


class REFormula:
    name = "formula"
    operator_d = {
        "+": "加",
        "-": "减",
        "x": "乘",
        "*": "乘",
        "/": "除",
        "÷": "除",
        "=": "等于",
        "≥": "大于等于",
        "≤": "小于等于",
        ">=": "大于等于",
        "<=": "小于等于",
        ">": "大于",
        "<": "小于",
    }

    def __init__(self, formula_s):
        self.formula_s = formula_s

    def to_string(self):
        return "".join(
            self.operator_d[self.formula_s[ii].txt] if ii % 2 else self.formula_s[ii].to_string()
            for ii in range(len(self.formula_s))
        )

    @classmethod
    def score_digits(cls, stack, idx):
        digits = stack[idx]
        if len(digits) < 3:
            return -1
        op_set = {digits[ii].txt for ii in range(1, len(digits), 2)}
        if op_set == {"-"}:
            return -1
        if all(ii in cls.operator_d for ii in op_set):
            if "=" in op_set:
                return 85
            else:
                return 10
        return -1

    @classmethod
    def deal_digits(cls, stack, idx):
        digits = stack[idx]
        stack[idx] = REFormula(digits)
        return stack, idx + 1

    @classmethod
    def merge_punc(cls, stack, idx):
        return stack, idx + 1
