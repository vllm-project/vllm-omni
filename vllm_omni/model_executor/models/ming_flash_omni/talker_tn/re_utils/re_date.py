# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Author: wanren
# Date: 2025/10/21
from .re_utils import merge_punc_range


class REDate:
    name = "date"
    ava_mm = {
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
        "11",
        "12",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    }
    ava_dd = {
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "20",
        "21",
        "22",
        "23",
        "24",
        "25",
        "26",
        "27",
        "28",
        "29",
        "30",
        "31",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    }

    def __init__(self, yyyy, mm, dd, day):
        self.yyyy, self.mm, self.dd = yyyy, mm, dd
        self.day = day

    def to_string(self):
        res = ""
        if self.yyyy:
            res += f"{self.yyyy.to_string_rm0()}年"
        if self.mm:
            res += f"{self.mm.to_string()}月"
        if self.dd:
            if self.day:
                res += f"{self.dd.to_string()}{self.day}"
            else:
                res += f"{self.dd.to_string()}号"
        return res

    @classmethod
    def try_digit(cls, stack, idx):
        # yyyy年mm月
        # mm月dd日
        if (
            idx + 5 < len(stack)
            and stack[idx].name == "digit" == stack[idx + 2].name == stack[idx + 4].name
            and stack[idx + 1].txt in {"年"}
            and stack[idx + 3].txt in {"月"}
            and stack[idx + 5].txt[:1] in {"日", "号"}
            and stack[idx + 2].txt in cls.ava_mm
            and stack[idx + 4].txt in cls.ava_dd
        ):
            # yyyy年mm月dd日
            yyyy = stack.pop(idx)
            stack.pop(idx)
            mm = stack.pop(idx)
            stack.pop(idx)
            dd = stack.pop(idx)
            day = stack[idx].txt[0]
            stack[idx].txt = stack[idx].txt[1:]
            stack.insert(idx, REDate(yyyy, mm, dd, day))
            return True
        if (
            idx + 3 < len(stack)
            and stack[idx].name == "digit" == stack[idx + 2].name
            and stack[idx + 1].txt in {"年"}
            and stack[idx + 3].txt[:1] in {"月"}
            and stack[idx + 2].txt in cls.ava_mm
        ):
            yyyy = stack.pop(idx)
            stack.pop(idx)
            mm = stack.pop(idx)
            stack[idx].txt = stack[idx].txt[1:]
            stack.insert(idx, REDate(yyyy, mm, None, None))
            return True
        if (
            idx + 3 < len(stack)
            and stack[idx].name == "digit" == stack[idx + 2].name
            and stack[idx + 1].txt in {"月"}
            and stack[idx + 3].txt[:1] in {"日", "号"}
            and stack[idx].txt in cls.ava_mm
            and stack[idx + 2].txt in cls.ava_dd
        ):
            mm = stack.pop(idx)
            stack.pop(idx)
            dd = stack.pop(idx)
            day = stack[idx].txt[0]
            stack[idx].txt = stack[idx].txt[1:]
            stack.insert(idx, REDate(None, mm, dd, day))
            return True
        if idx + 1 < len(stack) and stack[idx + 1].txt[:1] == "年" and len(stack[idx].txt) == 4:
            stack[idx] = REDate(stack[idx], None, None, None)
            stack[idx + 1].txt = stack[idx + 1].txt[1:]
            return True
        return False

    @classmethod
    def score_digits(cls, stack, idx):
        digits = stack[idx]
        if (
            len(digits) == 5
            and digits[1].txt == digits[3].txt
            and digits[1].txt in {".", "-", "/"}
            and len(digits[0].txt) == 4
            and digits[2].txt in cls.ava_mm
            and digits[4].txt in cls.ava_dd
        ):
            return 80
        return -1

    @classmethod
    def deal_digits(cls, stack, idx):
        digits = stack[idx]
        stack[idx] = REDate(digits[0], digits[2], digits[4], None)
        return stack, idx + 1

    @classmethod
    def merge_punc(cls, stack, idx):
        return merge_punc_range(stack, idx)
