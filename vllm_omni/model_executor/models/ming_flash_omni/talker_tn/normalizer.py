# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Author: wanren
# Date: 2025/10/16

from pynini import Fst, escape, shortestpath

from .token_parser import TokenParser


class Normalizer:
    def __init__(self, prefix_p, ordertype="tn"):
        self.tagger = Fst.read(f"{prefix_p}_tagger.fst").optimize()
        self.verbalizer = Fst.read(f"{prefix_p}_verbalizer.fst").optimize()
        self.ordertype = ordertype

    def tag(self, input):
        if len(input) == 0:
            return ""
        input = escape(input)
        lattice = input @ self.tagger
        return shortestpath(lattice, nshortest=1, unique=True).string()

    def verbalize(self, input):
        # Only words from the blacklist are contained.
        if len(input) == 0:
            return ""
        output = TokenParser(self.ordertype).reorder(input)
        # We need escape for pynini to build the fst from string.
        lattice = escape(output) @ self.verbalizer
        return shortestpath(lattice, nshortest=1, unique=True).string()

    def normalize(self, input):
        tmp = self.tag(input)
        # print(tmp)
        return self.verbalize(tmp)
