from types import SimpleNamespace

import torch

from vllm_omni.model_executor.models.breeze_tts_2.prompt_builder import (
    BreezeTTS2PromptBuilder,
)


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = -1
    _special = {
        "<|AUDIO|>": 262144,
        "<|audio_eos|>": 262145,
    }

    def convert_tokens_to_ids(self, token):
        return self._special.get(token, self.unk_token_id)

    def __call__(self, texts, *, add_special_tokens=True, padding=False):
        if isinstance(texts, str):
            texts = [texts]
        ids = []
        for text in texts:
            values = [1000 + ord(char) % 1000 for char in text]
            if add_special_tokens:
                values = [2, *values, 1]
            ids.append(values)
        return {"input_ids": ids}

    def decode(self, ids, *, skip_special_tokens=False):
        del skip_special_tokens
        rendered = []
        for token_id in ids:
            if token_id == 262144:
                rendered.append("<|AUDIO|>")
            elif token_id == 262145:
                rendered.append("<|audio_eos|>")
            elif token_id in (2, 1):
                continue
            else:
                rendered.append(chr(int(token_id) - 1000))
        return "".join(rendered)


class _RenderingTokenizer(_Tokenizer):
    """Tiny tokenizer that preserves Breeze placeholders after decode."""

    def __call__(self, texts, *, add_special_tokens=True, padding=False):
        if isinstance(texts, str):
            texts = [texts]
        ids = []
        for text in texts:
            row = []
            cursor = 0
            while cursor < len(text):
                if text.startswith("<|AUDIO|>", cursor):
                    row.append(262144)
                    cursor += len("<|AUDIO|>")
                elif text.startswith("<|audio_eos|>", cursor):
                    row.append(262145)
                    cursor += len("<|audio_eos|>")
                else:
                    row.append(1000 + ord(text[cursor]) % 1000)
                    cursor += 1
            if add_special_tokens:
                row = [2, *row, 1]
            ids.append(row)
        return {"input_ids": ids}

class _ReferenceEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, audio, sample_rate=None):
        self.calls.append((audio, sample_rate))
        return torch.arange(32, dtype=torch.int16).reshape(2, 16)


def _config():
    return SimpleNamespace(
        audio_token_id=262144,
        audio_eos_token_id=262145,
        pad_token_id=0,
        backbone_config=SimpleNamespace(vocab_size=100000),
        num_codebooks=16,
        codec_config={"codebook_size": 2048},
    )


def test_build_instruction_prompt():
    builder = BreezeTTS2PromptBuilder(_Tokenizer(), _config())

    prompt = builder.build({"text": "hello", "instruction": "happy", "speaker": "S0"})

    assert prompt["prompt_token_ids"]
    info = prompt["additional_information"]
    assert "input_values" not in info
    assert set(prompt["prompt_token_ids"]) == {0}
    assert info["prompt_ids"].numel() == len(prompt["prompt_token_ids"])
    assert info["text_ids_mask"].dtype == torch.bool
    assert info["text_ids_len"].tolist() == [len(prompt["prompt_token_ids"])]
    assert bool(info["text_ids_mask"].all())


def test_build_reference_prompt_and_reuse_encoder():
    encoder = _ReferenceEncoder()
    builder = BreezeTTS2PromptBuilder(_RenderingTokenizer(), _config(), encoder)

    prompt = builder.build(
        {
            "template": "ref_edit_tata",
            "text": "target",
            "instruction": "calm",
            "ref_text": "reference",
            "ref_audio": "ref.wav",
            "ref_audio_sample_rate": 16000,
        }
    )

    info = prompt["additional_information"]
    codes = info["input_values"]
    assert tuple(codes.shape) == (2, 16)
    assert codes.dtype == torch.int16
    assert set(prompt["prompt_token_ids"]) == {0}
    assert info["prompt_ids"].tolist().count(262144) == 2
    assert info["prompt_ids"].tolist().count(262145) == 1
    assert int((~info["text_ids_mask"]).sum()) == 3
    assert info["text_ids_len"].numel() == 2
    assert encoder.calls == [("ref.wav", 16000)]


def test_build_reference_prompt_preserves_special_tokens_after_rendering():
    encoder = _ReferenceEncoder()
    builder = BreezeTTS2PromptBuilder(_RenderingTokenizer(), _config(), encoder)

    prompt = builder.build(
        {
            "template": "ref_clone_tata",
            "text": "target",
            "ref_text": "reference",
            "ref_audio": "ref.wav",
            "ref_audio_sample_rate": 16000,
        }
    )

    info = prompt["additional_information"]
    prompt_ids = info["prompt_ids"].tolist()
    assert prompt_ids.count(262144) == 2
    assert prompt_ids.count(262145) == 1
    assert int((~info["text_ids_mask"]).sum()) == 3
    assert len(info["text_ids_len"]) == 2
    assert sum(info["text_ids_len"].tolist()) == int(info["text_ids_mask"].sum())


def test_build_reference_prompt_accepts_preencoded_codes_mapping():
    builder = BreezeTTS2PromptBuilder(_RenderingTokenizer(), _config())
    codes = torch.arange(32, dtype=torch.int16).reshape(2, 16)

    prompt = builder.build(
        {
            "template": "ref_clone_tata",
            "text": "target",
            "ref_text": "reference",
            "codes": {"ref": codes},
        }
    )

    info = prompt["additional_information"]
    assert torch.equal(info["input_values"], codes)
    assert info["ref_code_len"] == 2
