# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Author: wanren
# Date: 2025/10/23


class RESN:
    name = "serial_number"
    sn_d = {ii: ii for ii in range(ord("A"), ord("Z"), 1)}
    sn_d.update(
        {
            ord("0"): "零",
            ord("1"): "幺",
            ord("2"): "二",
            ord("3"): "三",
            ord("4"): "四",
            ord("5"): "五",
            ord("6"): "六",
            ord("7"): "七",
            ord("8"): "八",
            ord("9"): "九",
            ord("-"): "杠",
        }
    )

    def __init__(self, txt):
        self.txt = txt
        self.is_det = False

    def to_string(self):
        if self.is_det:
            return self.txt.translate(self.sn_d)
        return self.txt

    @classmethod
    def det(cls, stack, idx):
        cur_sn = stack[idx]
        if idx > 0:
            for kwd in ["SN", "sn", "序列号", "序列"]:
                for suffix in ["", ":", "："]:
                    if f"{kwd}{suffix}" in stack[idx - 1].txt:
                        cur_sn.is_det = True
                        return True
        interleave = []
        for ii in cur_sn.txt:
            if ii.isdecimal():
                cur_type = "0"
            elif ii.isalpha():
                cur_type = "A"
            else:
                cur_type = "-"
            if not interleave:
                interleave.append(cur_type)
            elif interleave[-1] != cur_type:
                interleave.append(cur_type)
        interleave = "".join(interleave)
        if len(interleave) > 5:
            cur_sn.is_det = True
            return True
        return False
