# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Author: wanren
# Date: 2025/10/16
import regex as re


class TalkerPreProcessor:
    def __init__(self):
        self.default_re = [
            [re.compile(r"\$\$.*?\$\$|\$.*?\$", re.DOTALL), ""],
            [re.compile(r"https?://[a-zA-Z0-9./?=&_:-]+|www\.[a-zA-Z0-9./?=&_:-]+"), ""],
            [re.compile(r"`.*?`", re.DOTALL), ""],
            [re.compile(r"```.*?```", re.DOTALL), ""],
            [re.compile(r"!\[.*?\]\(.*?\)"), ""],
            [re.compile(r"(?<=\d),(?=\d{3})"), ""],
            [re.compile(r"(?<=[^a-zA-Z0-9.])[ ]+(?=[^a-zA-Z0-9.])"), ""],
            [re.compile(r"(?<=[a-zA-Z])[ ]+(?=[a-zA-Z])"), " "],
            [re.compile(r"\*\*\*|\*\*|###|---"), ""],
        ]
        self.sil_char = {ii: None for ii in range(ord("\U0001f600"), ord("\U0001f64f") + 1, 1)}
        # "[\U0001F600-\U0001F64F"  # emoticons
        # "\U0001F300-\U0001F5FF"  # symbols & pictographs
        # "\U0001F680-\U0001F6FF"  # transport & map symbols
        # "\U0001F700-\U0001F77F"  # alchemical symbols
        # "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        # "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        # "\U0001F900-\U0001F9FF"  # Supplemental Emoji
        # "\U0001FA00-\U0001FA6F"  # Chess symbols
        # "\U00002702-\U000027B0"  # Dingbats
        # "\U000024C2-\U0001F251"  # Enclosed characters
        self.sil_char.update({ord(ii): None for ii in "▼●★☆■□▲△◆◇○◎√×✔︎✖︎"})
        self.sil_char.update({ord(ii): None for ii in "\t\n\r\f"})

    def __call__(self, text):
        for cur_re, replace in self.default_re:
            text = cur_re.sub(replace, text)
        text = text.translate(self.sil_char)
        # print(text)
        return text
