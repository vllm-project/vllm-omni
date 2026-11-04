# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-owned input ledger for Qwen turn deltas and snapshot fallback.

Keep the multimodal inputs on the ordinary processor path: its sender cache
must observe the same media accesses as the EngineCore receiver cache.
"""

from vllm.tokenizers import TokenizerLike

from vllm_omni.engine import OmniEngineCoreRequest

HISTORY_KEY = "_qwen_append_history"


class QwenIncrementalPrompt:
    """Append to accepted model tokens, or tokenize a changed history snapshot."""

    _boundary = "<|im_start|>"

    def __init__(self, tokenizer: TokenizerLike):
        self.tokenizer = tokenizer
        self._prefix = ""
        self._prefix_ids: list[int] = []
        self._history: list | None = None
        self._prompt_text = ""
        self._prompt_ids: list[int] = []
        self._output_ids: list[int] = []
        self._finished = False
        self._media_counts: dict[str, int] = {}
        self._append_safe = False

    def record_output(self, token_ids: list[int], finished: bool) -> None:
        self._output_ids.extend(token_ids)
        self._finished = self._finished or finished

    def prepare_turn(self, prompt: dict) -> tuple[dict, list[int] | None]:
        """Authorize an append from immutable history and actual sampled IDs.

        Only the new media visit the sender cache. The matching receiver thus
        sees precisely the same accesses; old materialized features stay owned
        by the resident EngineCore request.
        """
        history = prompt.get(HISTORY_KEY)
        end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if (
            not self._append_safe
            or "<|im_end|>" not in self.tokenizer.all_special_tokens
            or self._history is None
            or not isinstance(history, list)
            or history[: len(self._history)] != self._history
            or not self._finished
            or not self._output_ids
            or self._output_ids[-1] != end_id
            or self.tokenizer.encode("<|im_end|>", add_special_tokens=False) != [end_id]
        ):
            return prompt, None
        # Playback truncation, edited assistant text, tools inserted into old
        # context, or a changed system prompt must rebuild, not silently append.
        answer = self.tokenizer.decode(self._output_ids[:-1], skip_special_tokens=False)
        closed = self._prompt_text + answer + "<|im_end|>\n"
        text = prompt["prompt"]
        if not text.startswith(closed):
            return prompt, None
        media = prompt.get("multi_modal_data") or {}
        if any(not isinstance(items, list) for items in media.values()):
            return prompt, None
        if any(len(media.get(kind, [])) < count for kind, count in self._media_counts.items()):
            return prompt, None
        uuids = prompt.get("multi_modal_uuids")
        if uuids is not None and (
            not isinstance(uuids, dict)
            or any(
                kind not in media or not isinstance(ids, list) or len(ids) != len(media[kind])
                for kind, ids in uuids.items()
            )
        ):
            return prompt, None
        prepared = dict(prompt)
        # Upstream drops the final uncomputed sample. Re-inject that terminator
        # exactly once, followed by the new conversation content.
        prepared["prompt"] = "<|im_end|>\n" + text[len(closed) :]
        prepared["multi_modal_data"] = {
            kind: items[self._media_counts.get(kind, 0) :]
            for kind, items in media.items()
            if len(items) > self._media_counts.get(kind, 0)
        }
        if uuids is not None:
            # Replica scoping ran before this hook. Keep UUIDs aligned with the
            # sliced media, otherwise the new item could reuse an old cache key.
            prepared["multi_modal_uuids"] = {
                kind: ids[self._media_counts.get(kind, 0) :]
                for kind, ids in uuids.items()
                if kind in prepared["multi_modal_data"]
            }
        return prepared, self._prompt_ids + self._output_ids[:-1]

    def complete_turn(self, prompt: dict, request: OmniEngineCoreRequest, prefix: list[int] | None) -> None:
        if prefix is not None:
            request.append_prefix_length = len(prefix)
            request.prompt_token_ids = prefix + list(request.prompt_token_ids or ())
        self._history = prompt.get(HISTORY_KEY)
        self._prompt_text = prompt["prompt"]
        self._prompt_ids = list(request.prompt_token_ids or ())
        self._output_ids = []
        self._finished = False
        self._media_counts = {kind: len(items) for kind, items in (prompt.get("multi_modal_data") or {}).items()}

    def _encode_full(self, text: str) -> list[int]:
        ids = self.tokenizer.encode(text, add_special_tokens=True)
        # Tokenizers may omit chat delimiters from their special-token catalog
        # or omit AddedToken metadata. That disables text-prefix splitting,
        # but not continuation of the actual sampled token sequence.
        self._append_safe = ids == self.tokenizer.encode(text, add_special_tokens=False)
        return ids

    def tokenize(self, text: str) -> list[int]:
        tokenizer = self.tokenizer
        self._append_safe = False
        # Qwen's chat template embeds its special tokens explicitly. Refuse
        # the optimization if a tokenizer adds implicit BOS/EOS tokens.
        if self._boundary not in tokenizer.all_special_tokens:
            return self._encode_full(text)
        token = tokenizer.convert_tokens_to_ids(self._boundary)
        added = getattr(tokenizer, "added_tokens_decoder", {}).get(token)
        if added is None or added.lstrip or added.rstrip or added.single_word:
            # A token whose match depends on adjacent text is not a stable split.
            return self._encode_full(text)
        if self._prefix and text.startswith(self._prefix):
            ids = self._prefix_ids + tokenizer.encode(text[len(self._prefix) :], add_special_tokens=False)
        else:
            ids = self._encode_full(text)
            if not self._append_safe:
                self._prefix = ""
                self._prefix_ids = []
                return ids
        self._append_safe = True

        # End at an atomic special token, never an arbitrary text boundary:
        # BPE merges cannot cross it. Keep the generation header in the suffix
        # so assistant text from the previous turn is encoded canonically.
        end = text.rfind(self._boundary)
        if end >= 0 and token in ids:
            self._prefix = text[: end + len(self._boundary)]
            last = len(ids) - 1 - ids[::-1].index(token)
            self._prefix_ids = ids[: last + 1]
        else:
            self._prefix = ""
            self._prefix_ids = []
        return ids
