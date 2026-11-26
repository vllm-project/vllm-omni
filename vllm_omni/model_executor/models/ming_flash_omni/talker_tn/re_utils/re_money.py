# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Author: wanren
# Date: 2025/10/21
from .re_utils import merge_punc_measure, try_digit_after, try_digit_before


class REMoney:
    name = "money"
    money_d, money_len_s = {}, set()

    @classmethod
    def update(cls, money_d):
        cls.money_d = money_d
        money_len_set = set()
        for key in cls.money_d.keys():
            money_len_set.add(len(key))
        cls.money_len_s = list(money_len_set)
        cls.money_len_s.sort(reverse=True)

    def __init__(self, re_digit, tag_money, range_digit=None):
        self.re_digit, self.tag_money = re_digit, tag_money
        self.range_digit = range_digit

    def to_string(self):
        if self.range_digit:
            return f"{self.range_digit.to_string()}至{self.re_digit.to_string()}{self.tag_money}"
        return f"{self.re_digit.to_string()}{self.tag_money}"

    @classmethod
    def merge_punc(cls, stack, idx):
        return merge_punc_measure(stack, idx)


class REMoneyBefore(REMoney):
    @classmethod
    def try_digit(cls, stack, idx):
        return try_digit_before(stack, idx, cls.money_d, cls.money_len_s, cls)


class REMoneyAfter(REMoney):
    @classmethod
    def try_digit(cls, stack, idx):
        return try_digit_after(stack, idx, cls.money_d, cls.money_len_s, cls)
