# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Author: wanren
# Date: 2025/10/21
from .re_utils import merge_punc_range


class RETime:
    name = "time"
    noon_d = {
        "a m": "上午",
        "a.m.": "上午",
        "am": "上午",
        "A M": "上午",
        "AM": "上午",
        "p m": "下午",
        "p.m.": "下午",
        "pm": "下午",
        "P M": "下午",
        "PM": "下午",
    }

    def __init__(self, hour, minute, second, noon):
        self.hour, self.minute, self.second, self.noon = hour, minute, second, noon

    def to_string(self):
        if self.noon:
            res = self.noon
        else:
            res = ""
        if self.hour.txt in {"02", "2"}:
            res = f"{res}两点"
        else:
            res = f"{res}{self.hour.to_string()}点"
        if self.minute.txt in {"0", "00"} and (self.second is None or self.second.txt in {"0", "00"}):
            return f"{res}整"
        if self.minute.txt in {"0", "00"}:
            res = f"{res}零分"
        elif len(self.minute.txt) == 1 or self.minute.txt[0] == "0":
            res = f"{res}零{self.minute.to_string()}分"
        else:
            res = f"{res}{self.minute.to_string()}分"
        if self.second:
            if self.second.txt in {"0", "00"}:
                res = f"{res}零秒"
            elif len(self.second.txt) == 1 or self.second.txt[0] == "0":
                res = f"{res}零{self.second.to_string()}秒"
            else:
                res = f"{res}{self.second.to_string()}秒"
        return res

    @classmethod
    def score_digits(cls, stack, idx):
        digits = stack[idx]
        pre_txt = idx - 1 > 0 and stack[idx - 1].txt[-2:] in {
            "早上",
            "上午",
            "中午",
            "下午",
            "晚上",
            "傍晚",
            "凌晨",
            "半夜",
        }
        noon = None
        if idx + 1 < len(stack):
            for cur_l in range(2, 5):
                if stack[idx + 1].txt[:cur_l] in cls.noon_d:
                    noon = cls.noon_d[stack[idx + 1].txt[:cur_l]]
                    # stack[idx + 1].txt = stack[idx+1].txt[cur_l:]
                    break
        if len(digits) == 3 and digits[1].txt in {":", "："} and len(digits[0].txt) < 3 and len(digits[2].txt) < 3:
            if pre_txt or noon:
                return 90
            else:
                return 60
        if (
            len(digits) == 5
            and digits[1].txt in {":", "："}
            and digits[3].txt in {":", "："}
            and len(digits[0].txt) < 3
            and len(digits[2].txt) < 3
            and len(digits[4].txt) < 3
        ):
            if pre_txt or noon:
                return 90
            else:
                return 80
        # if len(digits) == 7 and digits[1].txt in {':', '：'} and \
        #         digits[3].txt in {'-', '~'} and digits[5].txt in {':', '：'}:
        #     if pre_txt or noon:
        #         return 90
        #     return 80
        # if len(digits) == 11 and digits[5].txt in {':', '：'} and all(
        #         digits[ii].txt in {':', '：'} for ii in [1, 3, 7, 9]):
        #     if pre_txt or noon:
        #         return 90
        #     return 80
        return -1

    @classmethod
    def deal_digits(cls, stack, idx):
        noon = None
        if idx + 1 < len(stack):
            for cur_l in range(2, 5):
                if stack[idx + 1].txt[:cur_l] in cls.noon_d:
                    noon = cls.noon_d[stack[idx + 1].txt[:cur_l]]
                    stack[idx + 1].txt = stack[idx + 1].txt[cur_l:]
                    break
        digits = stack[idx]
        if len(digits) == 3:
            stack[idx] = RETime(digits[0], digits[2], None, noon)
        elif len(digits) == 5:
            stack[idx] = RETime(digits[0], digits[2], digits[4], noon)
        # elif len(digits) == 7:
        #     pass
        # else:  # 11
        #     pass
        return stack, idx + 1

    @classmethod
    def merge_punc(cls, stack, idx):
        return merge_punc_range(stack, idx)
