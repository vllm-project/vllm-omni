# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Author: wanren
# Date: 2025/10/21
from .re_utils import merge_punc_measure, try_digit_after, try_digit_before


class REMeasureBefore:
    name = "measure_before"
    measure_d, measure_len_s = {}, set()

    @classmethod
    def update(cls, measure_d):
        cls.measure_d = measure_d
        measure_len_set = set()
        for key in cls.measure_d.keys():
            measure_len_set.add(len(key))
        cls.measure_len_s = list(measure_len_set)
        cls.measure_len_s.sort(reverse=True)

    def __init__(self, re_digit, tag_measure, range_digit=None):
        self.re_digit, self.tag_measure = re_digit, tag_measure
        self.range_digit = range_digit

    def to_string(self):
        if self.range_digit:
            return f"{self.tag_measure}{self.range_digit.to_string()}至{self.re_digit.to_string()}"
        return f"{self.tag_measure}{self.re_digit.to_string()}"

    @classmethod
    def try_digit(cls, stack, idx):
        if try_digit_before(stack, idx, cls.measure_d, cls.measure_len_s, cls):
            return True
        if idx > 0 and stack[idx - 1].name == "measure_after" and stack[idx - 1].tag_measure == "等于":
            stack[idx] = REMeasureBefore(stack[idx], "")
            return True
        return False

    @classmethod
    def merge_punc(cls, stack, idx):
        return merge_punc_measure(stack, idx)


class REMeasureAfter:
    name = "measure_after"
    measure_d, measure_len_s = {}, set()

    @classmethod
    def update(cls, measure_d):
        cls.measure_d = measure_d
        measure_len_set = set()
        for key in cls.measure_d.keys():
            measure_len_set.add(len(key))
        cls.measure_len_s = list(measure_len_set)
        cls.measure_len_s.sort(reverse=True)

    def __init__(self, re_digit, tag_measure, range_digit=None):
        self.re_digit, self.tag_measure = re_digit, tag_measure
        self.range_digit = range_digit

    def to_string(self):
        if self.range_digit:
            return f"{self.range_digit.to_string()}至{self.re_digit.to_string()}{self.tag_measure}"
        return f"{self.re_digit.to_string()}{self.tag_measure}"

    @classmethod
    def try_digit(cls, stack, idx):
        return try_digit_after(stack, idx, cls.measure_d, cls.measure_len_s, cls)

    @classmethod
    def merge_punc(cls, stack, idx):
        return merge_punc_measure(stack, idx)
