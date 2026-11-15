# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Author: wanren
# Date: 2025/10/21
import json

import regex as re

from .re_utils.re_char import REChar
from .re_utils.re_date import REDate
from .re_utils.re_decimal import REDecimal
from .re_utils.re_digit import REDigit
from .re_utils.re_formula import REFormula
from .re_utils.re_frac import REFrac
from .re_utils.re_ip import REIp
from .re_utils.re_measure import REMeasureAfter, REMeasureBefore
from .re_utils.re_money import REMoney, REMoneyAfter, REMoneyBefore
from .re_utils.re_order import REOrder
from .re_utils.re_percent import REPercent
from .re_utils.re_phone_dash import REPhoneDash
from .re_utils.re_punc import REPunc
from .re_utils.re_serial import RESerialAfter, RESerialBefore
from .re_utils.re_sn import RESN
from .re_utils.re_time import RETime
from .re_utils.re_version import REVersion


class TalkerRE:
    digit_punc = {"-", "~", "～", ":", "：", ".", "+", "-", "x", "*", "/", "÷", "=", "≥", "≤", ">", "<"}
    char_map0 = {}
    proper_noun = {}
    re_proper_noun = None
    re_sn = re.compile(r"(?=[A-Z0-9-]*[A-Z])(?=[A-Z0-9-]*[0-9])[A-Z0-9-]+")

    def __init__(self):
        pass

    @classmethod
    def update(cls, json_p):
        with open(json_p, encoding="utf-8") as fr:
            cfg = json.load(fr)
        REMoney.update(cfg["money"])
        REMeasureBefore.update(cfg["measure_before"])
        REMeasureAfter.update(cfg["measure_after"])
        RESerialBefore.update(cfg["serial_before"])
        RESerialAfter.update(cfg["serial_after"])
        cls.char_map0 = {ord(k): v for k, v in cfg["char_map0"].items()}
        cls.proper_noun = cfg["proper_noun"]
        sorted_keys = sorted(cls.proper_noun.keys(), key=len, reverse=True)
        cls.re_proper_noun = re.compile(r"(?<!\d)(" + r"|".join(map(re.escape, sorted_keys)) + r")(?!\d)")

    def __call__(self, text):
        text = text.translate(self.char_map0)
        text = self.re_proper_noun.sub(lambda re_match: self.proper_noun[re_match.group(0)], text)

        sn_match_s = [ii.span() for ii in self.re_sn.finditer(text)]
        if sn_match_s:
            base_s, last_pos = [], 0
            for sn_beg, sn_end in sn_match_s:
                base_s.append(REChar(text[last_pos:sn_beg]))
                base_s.append(RESN(text[sn_beg:sn_end]))
                last_pos = sn_end
            base_s.append(REChar(text[last_pos:]))
            for idx, cur_re in enumerate(base_s):
                if cur_re.name == "serial_number":
                    cur_re.det(base_s, idx)
            text = "".join(ii.to_string() for ii in base_s)

        base_s = []
        for cur_char in text:
            if cur_char.isdecimal():
                cur_cls = REDigit
            elif cur_char in self.digit_punc:
                cur_cls = REPunc
            else:
                cur_cls = REChar
            if base_s and isinstance(base_s[-1], cur_cls):
                base_s[-1].append_txt(cur_char)
            else:
                base_s.append(cur_cls(cur_char))
        base_s.append(REChar(""))

        stack, last_char, idx = [], -1, 0
        while idx < len(base_s):
            if base_s[idx].name == "char":
                if base_s[last_char + 1].name == "punc":
                    stack.append(base_s[last_char + 1])
                    last_char += 1
                if idx - last_char > 3:  # dpd
                    if idx > 0 and base_s[idx - 1].name == "punc":
                        stack.append(base_s[last_char + 1 : idx - 1])
                        stack.append(base_s[idx - 1])
                    else:
                        stack.append(base_s[last_char + 1 : idx])
                else:
                    stack.extend(base_s[last_char + 1 : idx])
                stack.append(base_s[idx])
                last_char = idx
            idx += 1
        idx = 0
        while idx < len(stack):
            if isinstance(stack[idx], list):
                score_s, sco_cls_s = [], [REDate, REDecimal, RETime, REVersion, REIp, REFrac, REPhoneDash, REFormula]
                for cur_cls in sco_cls_s:
                    score_s.append([cur_cls.score_digits(stack, idx), cur_cls])
                score_s.sort(key=lambda lam_tmp: lam_tmp[0])
                if score_s[-1][0] < 0:
                    digits = stack[idx]
                    rang_idx_s = [ii for ii, digit in enumerate(digits) if digit.txt in {"-", "~", "～"}]
                    rang_idx_s.sort(key=lambda lam_tmp: abs(lam_tmp - len(digits) / 2))
                    if rang_idx_s:
                        rang_idx = rang_idx_s[0]
                        stack2 = (
                            stack[:idx]
                            + [stack[idx][:rang_idx]]
                            + [REChar("至")]
                            + [stack[idx][rang_idx + 1 :]]
                            + stack[idx + 1 :]
                        )
                        sco_s0, sco_s1 = [], []
                        for cur_cls in sco_cls_s:
                            sco_s0.append([cur_cls.score_digits(stack2, idx), cur_cls])
                            sco_s1.append([cur_cls.score_digits(stack2, idx + 2), cur_cls])
                        sco_s0.sort(key=lambda lam_tmp: lam_tmp[0])
                        sco_s1.sort(key=lambda lam_tmp: lam_tmp[0])
                        if sco_s0[-1][0] >= 0 and sco_s1[-1][0] >= 0 and sco_s0[-1][1] == sco_s1[-1][1]:
                            stack, idx = sco_s0[-1][1].deal_digits(stack2, idx)
                            stack, idx = sco_s1[-1][1].deal_digits(stack, idx + 1)
                            continue
                    stack, idx = stack[:idx] + stack[idx] + stack[idx + 1 :], idx + len(stack[idx])
                else:
                    stack, idx = score_s[-1][1].deal_digits(stack, idx)
            else:
                idx += 1

        idx = 0
        while idx < len(stack):
            if stack[idx].name == "char" and len(stack[idx].txt) == 0:
                stack.pop(idx)
                continue
            if stack[idx].name not in {"digit", "decimal"}:
                idx += 1
                continue

            for cur_cls in [
                REMoneyBefore,
                REDate,
                REMoneyAfter,
                REMeasureBefore,
                REMeasureAfter,
                REPercent,
                RESerialBefore,
                RESerialAfter,
                REOrder,
            ]:
                if cur_cls.try_digit(stack, idx):
                    break
            else:
                idx += 1

        idx = 0
        while idx < len(stack):
            stack, idx = stack[idx].merge_punc(stack, idx)

        # return ''.join(ii.to_string() if ii.name != 'digit' else ii.to_string_yao() for ii in stack)
        return "".join(ii.to_string() if ii.name != "digit" else ii.txt for ii in stack)
