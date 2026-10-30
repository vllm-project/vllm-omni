# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Route generated word identities into the frozen native TTS lexical head.

This is a discrete output interface, not a learned final-hidden-state readout.
Only the native assistant template is tokenized; reply words retain the
Thinker's segmentation. Control tokens must never be pronounced as words.
"""


class GeneratedTokenRouter:
    def __init__(self, source_tokenizer, target_tokenizer):
        self.source = source_tokenizer
        self.target = target_tokenizer
        target_vocab = target_tokenizer.get_vocab()
        source_special = set(source_tokenizer.all_special_ids)
        target_special = set(target_tokenizer.all_special_ids)
        self.mapping = {
            source_id: target_vocab[token]
            for token, source_id in source_tokenizer.get_vocab().items()
            if source_id not in source_special and token in target_vocab and target_vocab[token] not in target_special
        }
        self.stop_ids = {source_tokenizer.eos_token_id}
        # Qwen's document EOS is also an allowed generation stop, but never
        # silently remove arbitrary specials or interior EOS from a reply.
        endoftext = source_tokenizer.get_vocab().get("<|endoftext|>")
        if endoftext in source_special:
            self.stop_ids.add(endoftext)
        template = target_tokenizer.encode(
            "<|im_start|>assistant\nx<|im_end|>\n<|im_start|>assistant\n",
            add_special_tokens=False,
        )
        if len(template) != 9:
            raise ValueError("Unsupported native TTS assistant template")
        self.prefix, self.suffix = template[:3], template[-5:]

    def prepare_fixed(self, text):
        """Tokenize an explicitly fixed reply without claiming LLM provenance."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Fixed speech requires nonempty text")
        ids = self.target.encode(text, add_special_tokens=False)
        if not ids or any(i in self.target.all_special_ids for i in ids):
            raise ValueError("Fixed speech cannot contain control tokens")
        if self.target.decode(ids, skip_special_tokens=False) != text:
            raise ValueError("Fixed speech tokenization changed the reply")
        return self.prefix + ids + self.suffix

    def prepare(self, generated_ids, expected_text):
        if not isinstance(generated_ids, (list, tuple)) or not generated_ids:
            raise ValueError("Missing actual Thinker-generated token IDs")
        ids = list(generated_ids)
        if any(type(value) is not int for value in ids):
            raise ValueError("Generated token IDs must be integers")
        if ids[-1] in self.stop_ids:
            ids.pop()
        if not ids or any(value not in self.mapping for value in ids):
            raise ValueError("Empty, special, or unmapped generated lexical token")
        mapped = [self.mapping[value] for value in ids]
        source_text = self.source.decode(ids, skip_special_tokens=False)
        target_text = self.target.decode(mapped, skip_special_tokens=False)
        if source_text != expected_text or target_text != expected_text:
            raise ValueError("Generated tokens disagree with the displayed reply")
        return self.prefix + mapped + self.suffix
