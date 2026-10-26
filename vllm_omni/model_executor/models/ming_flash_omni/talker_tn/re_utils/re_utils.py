# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Author: wanren
# Date: 2025/10/22
import string

from .re_char import REChar


def try_digit_before(stack, idx, money_d, money_len_s, cur_cls):
    if idx - 1 < 0:
        return False
    if stack[idx - 1].name not in {"char", "punc"}:
        return False
    re_bf_char, tag_money = stack[idx - 1], None
    for cur_l in money_len_s:
        if cur_l > len(re_bf_char.txt):
            continue
        match_res = re_bf_char.txt[-cur_l:]
        if match_res in money_d:
            if (
                match_res[0] in string.ascii_letters
                and cur_l < len(re_bf_char.txt)
                and re_bf_char.txt[-cur_l - 1] in string.ascii_letters
            ):
                continue
            tag_money = money_d[re_bf_char.txt[-cur_l:]]
            re_bf_char.txt = re_bf_char.txt[:-cur_l]
            stack[idx] = cur_cls(stack[idx], tag_money)
            return True
    return False


def try_digit_after(stack, idx, measure_d, measure_len_s, cur_cls):
    if idx + 1 >= len(stack):
        return False
    if stack[idx + 1].name not in {"char", "punc"}:
        return False
    re_af_char, tag_measure = stack[idx + 1], None
    for cur_l in measure_len_s:
        if cur_l > len(re_af_char.txt):
            continue
        match_res = re_af_char.txt[:cur_l]
        if match_res in measure_d:
            if (
                match_res[-1] in string.ascii_letters
                and cur_l < len(re_af_char.txt)
                and re_af_char.txt[cur_l] in string.ascii_letters
            ):
                continue
            tag_measure = measure_d[match_res]
            re_af_char.txt = re_af_char.txt[cur_l:]
            stack[idx] = cur_cls(stack[idx], tag_measure)
            return True
    return False


def merge_punc_measure(stack, idx):
    if idx - 1 >= 0 and stack[idx - 1].name == "punc" and stack[idx - 1].txt in {"-", "~", "～"}:
        if idx - 2 >= 0 and (
            stack[idx - 2].name == stack[idx].name
            or (stack[idx - 2].name in {"measure_before", "money"} and stack[idx].name in {"measure_after", "money"})
        ):
            stack[idx - 1] = REChar("至")
        elif idx - 2 >= 0 and stack[idx - 2].name == "digit":
            stack[idx].range_digit = stack[idx - 2]
            return stack[: idx - 2] + stack[idx:], idx - 1
        elif idx - 2 < 0 or stack[idx - 2].name == "char":
            stack[idx - 1] = REChar("负") if stack[idx - 1].txt == "-" else REChar("大约")
    return stack, idx + 1


def merge_punc_range(stack, idx):
    if (
        idx - 2 >= 0
        and stack[idx - 1].name == "punc"
        and stack[idx - 1].txt in {"-", "~", "～"}
        and stack[idx - 2].name == stack[idx].name
    ):
        stack[idx - 1] = REChar("至")
    return stack, idx + 1
