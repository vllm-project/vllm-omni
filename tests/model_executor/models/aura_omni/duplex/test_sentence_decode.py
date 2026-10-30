# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Growing Stage1 token ids must decode to the same text as one full decode."""

from __future__ import annotations

from vllm_omni.model_executor.models.aura_omni.duplex.sentence_tts import decode_growing_token_ids


def _assert_matches_full(decode, ids: list[int]) -> None:
    cache: dict[str, object] = {}
    for end in range(1, len(ids) + 1):
        got = decode_growing_token_ids(decode, ids[:end], cache)
        assert got == decode(ids[:end])


def test_growing_decode_matches_full_decode_for_concat_pieces() -> None:
    vocab = {1: "你", 2: "好", 3: "。", 4: "世", 5: "界"}

    def decode(token_ids: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(vocab[token_id] for token_id in token_ids)

    _assert_matches_full(decode, [1, 2, 3, 4, 5])


def test_growing_decode_matches_full_decode_when_the_previous_piece_changes() -> None:
    """Token 9 rewrites the previous character. The tail window must reject a bad splice."""

    def decode(token_ids: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        chars: list[str] = []
        for token_id in token_ids:
            if token_id == 9:
                if chars:
                    chars.pop()
                chars.append("X")
            else:
                chars.append(chr(96 + token_id))
        return "".join(chars)

    _assert_matches_full(decode, [1, 2, 9, 3])


def test_growing_decode_matches_wordpiece_spacing() -> None:
    vocab = {1: "hello", 2: "##world", 3: "。"}

    def decode(token_ids: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        parts: list[str] = []
        for index, token_id in enumerate(token_ids):
            piece = vocab[token_id]
            if piece.startswith("##"):
                parts.append(piece[2:])
            else:
                if index:
                    parts.append(" ")
                parts.append(piece)
        return "".join(parts)

    _assert_matches_full(decode, [1, 2, 3, 1])
