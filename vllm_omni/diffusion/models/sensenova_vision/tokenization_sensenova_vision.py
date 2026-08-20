# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SenseNova-Vision Qwen2 tokenizer that preserves the checkpoint's token ids.

Background
----------
``sensenova/SenseNova-Vision-7B-MoT`` ships a base Qwen BPE vocab of 151643
tokens (``vocab.json``/``merges.txt``) whose ``tokenizer_config.json`` declares
2033 *added* tokens (``added_tokens_decoder``) spanning ids **149632-151664**,
including the four special tokens that BAGEL relies on:

    <|im_start|>     -> 151644
    <|im_end|>       -> 151645
    <|vision_start|> -> 151652
    <|vision_end|>   -> 151653

The LLM checkpoint (``llm_config.json``) has ``vocab_size: 152064`` and the
embedding/head weights (``ema.safetensors``) have exactly 152064 rows, so every
token id the model ever embeds must be ``< 152064``.

Why a custom tokenizer is required
----------------------------------
When HuggingFace loads this checkpoint with its stock tokenizer machinery
(``AutoTokenizer.from_pretrained`` or even ``PreTrainedTokenizerFast``), the
added tokens are re-registered against the *Rust* backend, which assigns fresh
contiguous ids starting after the base vocab.  The 2033 added tokens therefore
get relabelled 151643-153675, pushing the four controls to 153655-153664.
``BagelPipeline.__init__`` then computes ``vocab_size = max(152064,
len(tokenizer), max_control_id + 1)`` = 153676, builds the language model with
153676 rows, and the 152064-row checkpoint embedding trips
``VocabParallelEmbedding``'s ``org_vocab_size`` assertion during weight load.

(``from_pretrained`` itself preserves ids verbatim — see
``tokenization_utils_base.py`` ``_from_pretrained`` — the renumbering only
happens when added tokens are re-registered via ``_add_tokens`` against a fresh
backend.  BAGEL's tokenizer has only 22 added tokens immediately following a
151643 vocab, so its re-registration coincidentally lands on the same ids and
is never observed.)

The reference implementation (``SenseNova-Vision/inference/sensenova_vision.py``)
works because it *never* goes through ``from_pretrained``: it instantiates the
repo's own slow ``Qwen2Tokenizer`` directly from ``vocab.json``/``merges.txt``
and then appends the four control specials, which land at 151643-151646.

This class reproduces that behaviour for the vllm-omni integration: it is a
transformers **slow** tokenizer (``PythonBackend`` subclass, matching
transformers >= 5.14.1 where slow tokenizers no longer live in
``tokenization_utils.py``) that loads the base Qwen vocab itself and registers
the checkpoint's ``added_tokens_decoder`` ids **verbatim** — no renumbering, so
all 2033 added tokens keep ids <= 151664 and the controls stay in vocab.
``len(tokenizer)`` stays <= 152064 and ``vocab_size`` remains 152064, matching
the checkpoint rows exactly.
"""

from __future__ import annotations

import json
import unicodedata
from functools import lru_cache

import regex as re
from transformers import AddedToken
from transformers.tokenization_python import PythonBackend
from transformers.utils import logging

logger = logging.get_logger(__name__)

# Maximum LLM embedding/head rows (``llm_config.json`` vocab_size).  Every
# token id the model embeds must be strictly below this.
LLM_VOCAB_SIZE = 152064

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

# Matches the base Qwen2 tokenizer's pre-tokenization regex (same as the
# reference repo's modeling/qwen2/tokenization_qwen2.py).
PRETOKENIZE_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


@lru_cache
def bytes_to_unicode():
    """UTF-8 byte <-> unicode mapping used by the GPT2-style byte-level BPE."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word (GPT2-style BPE)."""
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class VLLMSenseNovaVisionTokenizer(PythonBackend):
    """Qwen2 byte-level BPE tokenizer for SenseNova-Vision that never renumbers.

    The tokenizer preserves the ids recorded in the checkpoint's
    ``tokenizer_config.json`` ``added_tokens_decoder`` exactly, keeping the four
    BAGEL control tokens (and all other added tokens) inside the LLM's 152064
    embedding rows.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token=None,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        clean_up_tokenization_spaces=False,
        split_special_tokens=False,
        **kwargs,
    ):
        # Special-token wrappers (same semantics as the reference slow tokenizer).
        bos_token = (
            AddedToken(bos_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(bos_token, str)
            else bos_token
        )
        eos_token = (
            AddedToken(eos_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(eos_token, str)
            else eos_token
        )
        unk_token = (
            AddedToken(unk_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(unk_token, str)
            else unk_token
        )
        pad_token = (
            AddedToken(pad_token, lstrip=False, rstrip=False, special=True, normalized=False)
            if isinstance(pad_token, str)
            else pad_token
        )

        # --- Base Qwen BPE vocab -------------------------------------------------
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        bpe_merges = []
        with open(merges_file, encoding="utf-8") as merges_handle:
            for i, line in enumerate(merges_handle):
                line = line.strip()
                if (i == 0 and line.startswith("#version:")) or not line:
                    continue
                bpe_merges.append(tuple(line.split()))
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        self.pat = re.compile(PRETOKENIZE_REGEX)

        # --- Register the checkpoint's added tokens verbatim ---------------------
        # `PythonBackend.__init__` seeds `_added_tokens_decoder` from the
        # `added_tokens_decoder` kwarg (which `_from_pretrained` loaded verbatim
        # from tokenizer_config.json). Filling it *before* the base-class init
        # below means those ids are already registered when the base class adds
        # special tokens: no re-registration against a fresh backend can occur,
        # so ids 149632-151664 stay exactly as recorded on disk.
        added_tokens_decoder = kwargs.pop("added_tokens_decoder", {})
        self._added_tokens_decoder: dict[int, AddedToken] = {}
        self._added_tokens_encoder: dict[str, int] = {}
        for idx, token in added_tokens_decoder.items():
            if isinstance(token, dict):
                token = AddedToken(**token)
            elif isinstance(token, str):
                token = AddedToken(token, special=True, normalized=False)
            self._added_tokens_decoder[int(idx)] = token
            self._added_tokens_encoder[str(token)] = int(idx)

        super().__init__(
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            split_special_tokens=split_special_tokens,
            vocab_file=vocab_file,
            merges_file=merges_file,
            **kwargs,
        )

    # ------------------------------------------------------------------ vocab ---
    @property
    def vocab_size(self) -> int:
        """Size of the base vocabulary (without added tokens): 151643."""
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder, **self._added_tokens_encoder)

    def __len__(self) -> int:
        """Size of the used token-id space = ``max_id + 1`` (sparse-aware).

        ``PythonBackend.__len__`` counts the keys of ``get_vocab()``
        (``_update_total_vocab_size``), i.e. 151643 base + 2033 added =
        153676.  But the checkpoint's id space is *sparse*: the 2033 added
        tokens occupy ids 149632-151664, which overlap the base vocab range
        only in numbering, not in content — there is an 11-id hole at
        149632-151642 (base vocab ends at 151642).  Counting keys therefore
        over-reports the true ``max_id + 1`` = 151665 and would make
        ``BagelPipeline.__init__`` inflate ``llm_config.vocab_size`` past the
        152064 checkpoint rows.

        Reporting the actual highest used id + 1 keeps ``len(tokenizer)`` a
        faithful upper bound for the ids the model embeds (151665 < 152064),
        so the BAGEL ``max(152064, len(tok), required_max_id + 1)`` still
        evaluates to the checkpoint's 152064.
        """
        max_id = max([len(self.encoder) - 1] + [idx for idx in self._added_tokens_decoder if idx >= 0])
        return max_id + 1

    def _convert_token_to_id(self, token):
        return self.encoder.get(token, self.encoder.get(self.unk_token or "<|endoftext|>"))

    def _convert_id_to_token(self, index):
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    # --------------------------------------------------------------- tokenize ----
    def _tokenize(self, text):
        """Tokenize a string (byte-level BPE, mirroring GPT2/Qwen2)."""
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)
        if not pairs:
            return token
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def prepare_for_tokenization(self, text, **kwargs):
        text = unicodedata.normalize("NFC", text)
        return (text, kwargs)

    # ------------------------------------------------------------- saving -------
    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str, str]:
        vocab_file = f"{save_directory}/{filename_prefix + '-' if filename_prefix else ''}vocab.json"
        merge_file = f"{save_directory}/{filename_prefix + '-' if filename_prefix else ''}merges.txt"
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if token_index is not None:
                    writer.write(" ".join(bpe_tokens) + "\n")
        return vocab_file, merge_file


_registered = False


def register_vllm_sensenova_vision_tokenizer() -> None:
    """Make ``AutoTokenizer`` resolve our class in-process (no remote-code file).

    ``AutoTokenizer.from_pretrained`` resolves ``tokenizer_class`` through
    ``tokenizer_class_from_name`` (transformers 5.x
    ``tokenization_auto.py``), which consults the ``REGISTERED_TOKENIZER_CLASSES``
    dict *before* any filesystem/dynamic-module lookup.  Registering our class
    there means the BAGEL core's ``AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True)`` instantiates
    ``VLLMSenseNovaVisionTokenizer`` entirely in-process — with no
    ``tokenization_sensenova_vision.py`` file copied into the checkpoint / patch
    directory and no ``auto_map`` in ``tokenizer_config.json``.

    We intentionally avoid ``AutoTokenizer.register`` (the public API): it also
    registers into the ``TOKENIZER_MAPPING`` keyed by a config class, and
    ``Qwen2Config`` is already mapped to the stock Qwen2 tokenizer, so that path
    raises ``ValueError: already used by a Transformers model``.  Writing
    directly into ``REGISTERED_TOKENIZER_CLASSES`` (the same table
    ``tokenizer_class_from_name`` reads at line 477) is the minimal, version-safe
    hook; the table is a plain module-level dict, so the write is idempotent.

    Must be called before the ``AutoTokenizer.from_pretrained`` that loads the
    checkpoint tokenizer (``SenseNovaVisionPipeline.__init__`` does this before
    delegating to the BAGEL core).
    """
    global _registered
    if _registered:
        return
    from transformers.models.auto.tokenization_auto import REGISTERED_TOKENIZER_CLASSES

    REGISTERED_TOKENIZER_CLASSES[VLLMSenseNovaVisionTokenizer.__name__] = VLLMSenseNovaVisionTokenizer
    _registered = True
