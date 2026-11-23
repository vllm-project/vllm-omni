# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Author: wanren
# Date: 2025/10/21
from .re_utils import try_digit_after, try_digit_before


class RESerialBefore:
    name = "serial_before"
    serial_d, serial_len_s = {}, set()

    @classmethod
    def update(cls, serial_d):
        cls.serial_d = serial_d
        serial_len_set = set()
        for key in cls.serial_d.keys():
            serial_len_set.add(len(key))
        cls.serial_len_s = list(serial_len_set)
        cls.serial_len_s.sort(reverse=True)

    def __init__(self, re_digit, tag_serial):
        self.re_digit, self.tag_serial = re_digit, tag_serial

    def to_string(self):
        return f"{self.tag_serial}{self.re_digit.to_string_yao()}"

    @classmethod
    def try_digit(cls, stack, idx):
        return try_digit_before(stack, idx, cls.serial_d, cls.serial_len_s, cls)

    @classmethod
    def merge_punc(cls, stack, idx):
        return stack, idx + 1


class RESerialAfter:
    name = "serial_after"
    serial_d, serial_len_s = {}, set()

    @classmethod
    def update(cls, serial_d):
        cls.serial_d = serial_d
        serial_len_set = set()
        for key in cls.serial_d.keys():
            serial_len_set.add(len(key))
        cls.serial_len_s = list(serial_len_set)
        cls.serial_len_s.sort(reverse=True)

    def __init__(self, re_digit, tag_serial):
        self.re_digit, self.tag_serial = re_digit, tag_serial

    def to_string(self):
        return f"{self.re_digit.to_string_yao()}{self.tag_serial}"

    @classmethod
    def try_digit(cls, stack, idx):
        return try_digit_after(stack, idx, cls.serial_d, cls.serial_len_s, cls)

    @classmethod
    def merge_punc(cls, stack, idx):
        return stack, idx + 1
