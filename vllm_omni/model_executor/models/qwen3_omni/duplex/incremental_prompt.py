# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-owned input ledger for Qwen turn deltas and snapshot fallback.

Keep the multimodal inputs on the ordinary processor path: its sender cache
must observe the same media accesses as the EngineCore receiver cache.
"""

from vllm.tokenizers import TokenizerLike

from vllm_omni.engine import OmniEngineCoreRequest

DELTA_KEY = "_qwen_turn_delta"


class QwenIncrementalPrompt:
    """Append to accepted model tokens, or tokenize a changed history snapshot."""

    def __init__(self, tokenizer: TokenizerLike):
        self.tokenizer = tokenizer
        self._prompt_ids: list[int] = []
        self._output_ids: list[int] = []
        self._finished = False
        self._append_safe = False

    def record_output(self, token_ids: list[int], finished: bool) -> None:
        self._output_ids.extend(token_ids)
        self._finished = self._finished or finished

    def can_append_turn(self, assistant_text: str) -> bool:
        end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        return bool(
            self._append_safe
            and self._finished
            and self._output_ids
            and self._output_ids[-1] == end_id
            and "<|im_end|>" in self.tokenizer.all_special_tokens
            and self.tokenizer.encode("<|im_end|>", add_special_tokens=False) == [end_id]
            and self.tokenizer.decode(self._output_ids[:-1], skip_special_tokens=False) == assistant_text
        )

    def prepare_turn(self, prompt: dict) -> tuple[dict, list[int] | None]:
        """Use the session-authorized delta, or process a replacement snapshot."""
        if prompt.get(DELTA_KEY):
            if not self._finished or not self._output_ids or not self._append_safe:
                raise ValueError("Turn delta has no completed resident token ledger")
            return prompt, self._prompt_ids + self._output_ids[:-1]
        return prompt, None

    def complete_turn(self, prompt: dict, request: OmniEngineCoreRequest, prefix: list[int] | None) -> None:
        if prefix is not None:
            request.append_prefix_length = len(prefix)
            request.prompt_token_ids = prefix + (request.prompt_token_ids or [])
        self._prompt_ids = list(request.prompt_token_ids or ())
        self._output_ids = []
        self._finished = False

    def tokenize(self, text: str) -> list[int]:
        # Deltas already contain only the new turn. Snapshots are encoded in
        # full; do not maintain a second, rendered-text prefix cache.
        self._append_safe = False
        ids = self.tokenizer.encode(text, add_special_tokens=True)
        # Implicit BOS/EOS tokens would break continuation at a turn boundary.
        self._append_safe = ids == self.tokenizer.encode(text, add_special_tokens=False)
        return ids
