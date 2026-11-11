# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Author: wanren
# Date: 2025/10/21
from .re_base import REBase


class REDigit(REBase):
    name = "digit"
    d2c = {"0": "零", "1": "一", "2": "二", "3": "三", "4": "四", "5": "五", "6": "六", "7": "七", "8": "八", "9": "九"}
    d2c_liang = {
        "0": "零",
        "1": "一",
        "2": "两",
        "3": "三",
        "4": "四",
        "5": "五",
        "6": "六",
        "7": "七",
        "8": "八",
        "9": "九",
    }
    d2c_yao = {
        "0": "零",
        "1": "幺",
        "2": "二",
        "3": "三",
        "4": "四",
        "5": "五",
        "6": "六",
        "7": "七",
        "8": "八",
        "9": "九",
    }

    def __init__(self, txt):
        super().__init__(txt)

    @classmethod
    def d4_to_chi(cls, num_str):
        res = ""
        while num_str and num_str[0] == "0":
            num_str = num_str[1:]
        if len(num_str) == 2 and num_str[0] == "1":
            if num_str[1] == "0":
                return "十"
            return f"十{cls.d2c[num_str[1]]}"
        while num_str:
            if num_str[0] == "0":
                while num_str and num_str[0] == "0":
                    num_str = num_str[1:]
                if num_str:
                    res += "零"
                    continue
                else:
                    break
            elif len(num_str) == 4:
                res += f"{cls.d2c_liang[num_str[0]]}千"
            elif len(num_str) == 3:
                res += f"{cls.d2c_liang[num_str[0]]}百"
            elif len(num_str) == 2:
                res += f"{cls.d2c[num_str[0]]}十"
            else:
                res += cls.d2c[num_str[0]]
            num_str = num_str[1:]
        return res

    @classmethod
    def d_to_direct_yao(cls, num_str):
        return "".join(cls.d2c_yao[ii] for ii in num_str)

    num_unit = ["", "万", "亿", "兆", "京", "垓", "秭", "穰", "沟", "涧", "正", "载", "极"]

    def to_string(self):  # 常规数字读法money, measure
        if self.txt == "0":
            return "零"
        num_str = self.txt
        res = ""
        for ii in range(len(self.num_unit)):
            if not num_str:
                break
            cur_d4 = self.d4_to_chi(num_str[-4:])
            if cur_d4:
                res = f"{cur_d4}{self.num_unit[ii]}{res}"
            num_str = num_str[:-4]
        if num_str:
            res = f"{self.d_to_direct_yao(num_str)}{res}"
        if not res:
            return "零"
        return res

    def to_string_rm0(self):  # date
        num_str = self.txt
        while num_str and num_str[0] == "0":
            num_str = num_str[1:]
        return "".join(self.d2c[ii] for ii in num_str)

    def to_string_direct(self):  # decimal-frac
        return "".join(self.d2c[ii] for ii in self.txt)

    def to_string_yao(self):  # yao
        return "".join(self.d2c_yao[ii] for ii in self.txt)
