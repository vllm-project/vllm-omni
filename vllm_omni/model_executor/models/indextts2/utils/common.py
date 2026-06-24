# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re

import torch


def tokenize_by_CJK_char(line: str, do_upper_case=True) -> str:
    """
    Tokenize a line of text with CJK char.

    Note: All return characters will be upper case.

    Example:
      input = "你好世界是 hello world 的中文"
      output = "你 好 世 界 是 HELLO WORLD 的 中 文"

    Args:
      line:
        The input text.

    Return:
      A new string tokenize by CJK char.
    """
    # CJK ranges from nltk/tokenize/util.py (alvations/nltk@79eed6d)
    CJK_RANGE_PATTERN = (
        r"([\u1100-\u11ff\u2e80-\ua4cf\ua840-\uD7AF\uF900-\uFAFF\uFE30-\uFE4F\uFF65-\uFFDC\U00020000-\U0002FFFF])"
    )
    chars = re.split(CJK_RANGE_PATTERN, line.strip())
    return " ".join([w.strip().upper() if do_upper_case else w.strip() for w in chars if w.strip()])


def de_tokenized_by_CJK_char(line: str, do_lower_case=False) -> str:
    """
    Example:
      input = "你 好 世 界 是 HELLO WORLD 的 中 文"
      output = "你好世界是 hello world 的中文"

    do_lower_case:
      input = "SEE YOU!"
      output = "see you!"
    """
    # replace english words in the line with placeholders
    english_word_pattern = re.compile(r"([A-Z]+(?:[\s'-][A-Z-]+)*)", re.IGNORECASE)
    english_sents = english_word_pattern.findall(line)
    for i, sent in enumerate(english_sents):
        line = line.replace(sent, f"<sent_{i}>")

    words = line.split()
    # restore english sentences
    sent_placeholder_pattern = re.compile(r"(<sent_(\d+)>)")
    for i in range(len(words)):
        all_matches = sent_placeholder_pattern.findall(words[i])
        if all_matches:
            # restore the english word
            for h, j in all_matches:
                placeholder_index = int(j)
                words[i] = words[i].replace(h, english_sents[placeholder_index])
                if do_lower_case:
                    words[i] = words[i].lower()
    return "".join(words)


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask
