import re

_EN_WORD_RE = re.compile(r"^[A-Za-z]+(?:'[A-Za-z]+)*$")
_ZH_WORD_RE = re.compile(r"[\u4e00-\u9fff]")

EN_FLAG = "en_"
YUE_FLAG = "yue_"
ZH_FLAG = "zh_"

_g2p_zh = None
_g2p_en = None


def _ensure_g2p_backends() -> None:
    global _g2p_zh, _g2p_en
    if _g2p_zh is None:
        from g2pM import G2pM

        _g2p_zh = G2pM()
    if _g2p_en is None:
        from g2p_en import G2p as G2pE

        _g2p_en = G2pE()


def is_chinese_char(word: str) -> bool:
    if len(word) != 1:
        return False
    return bool(_ZH_WORD_RE.fullmatch(word))


def is_english_word(word: str) -> bool:
    if not word:
        return False
    return bool(_EN_WORD_RE.fullmatch(word))


def g2p_cantonese(sent):
    import ToJyutping

    return ToJyutping.get_jyutping_list(sent)


def g2p_mandarin(sent):
    _ensure_g2p_backends()
    return _g2p_zh(sent, tone=True, char_split=False)


def g2p_english(word):
    _ensure_g2p_backends()
    return _g2p_en(word)


def g2p_transform(words, lang):
    zh_words = []
    transformed_words = [0] * len(words)

    for idx, w in enumerate(words):
        if w == "<SP>":
            transformed_words[idx] = w
            continue

        w = w.replace("?", "").replace(".", "").replace("!", "").replace(",", "")

        if is_chinese_char(w):
            zh_words.append([idx, w])
        else:
            if is_english_word(w):
                w = EN_FLAG + "-".join(g2p_english(w.lower()))
            else:
                w = "<SP>"
        transformed_words[idx] = w

    sent = "".join([k[1] for k in zh_words])

    if len(sent) > 0:
        if lang == "Cantonese":
            g2pm_rst = g2p_cantonese(sent)
            g2pm_rst = [YUE_FLAG + k[1] for k in g2pm_rst]
        else:
            g2pm_rst = g2p_mandarin(sent)
            g2pm_rst = [ZH_FLAG + k for k in g2pm_rst]
        for p, w in zip([k[0] for k in zh_words], g2pm_rst):
            transformed_words[p] = w

    return transformed_words
