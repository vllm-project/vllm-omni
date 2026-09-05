from types import SimpleNamespace

from vllm_omni.entrypoints.openai.tts_adapters.breeze_tts_2 import BreezeTTS2Adapter
from vllm_omni.model_executor.models.breeze_tts_2.prompt_builder import (
    BreezeTTS2PromptBuilder,
)


class _Tokenizer:
    unk_token_id = -1

    def convert_tokens_to_ids(self, token):
        return {"<|AUDIO|>": 262144, "<|audio_eos|>": 262145}.get(token, self.unk_token_id)

    def __call__(self, texts, *, add_special_tokens=True, padding=False):
        if isinstance(texts, str):
            texts = [texts]
        ids = [[1000 + ord(char) % 1000 for char in text] for text in texts]
        if add_special_tokens:
            ids = [[2, *row, 1] for row in ids]
        return {"input_ids": ids}

    def decode(self, ids, *, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(int(token_id) - 1000) for token_id in ids if token_id not in (1, 2))


def test_instruction_prompt_remains_one_independent_text_segment():
    builder = BreezeTTS2PromptBuilder(
        _Tokenizer(),
        SimpleNamespace(
            audio_token_id=262144,
            audio_eos_token_id=262145,
            pad_token_id=0,
            backbone_config=SimpleNamespace(vocab_size=100000),
            num_codebooks=16,
            codec_config={"codebook_size": 2048},
        ),
    )

    prompt = builder.build(
        {
            "template": "tts_instruction",
            "speaker": "S2",
            "instruction": "Use a bright and friendly delivery.",
            "text": "Hello from Breeze.",
        }
    )

    info = prompt["additional_information"]
    assert info["template"] == "tts_instruction"
    assert "input_values" not in info
    assert bool(info["text_ids_mask"].all())
    assert info["text_ids_len"].tolist() == [len(info["prompt_ids"])]


def _request(extra_params=None):
    return SimpleNamespace(
        input="Hello.",
        ref_text=None,
        ref_audio=None,
        task_type=None,
        speed=1.0,
        max_new_tokens=None,
        extra_params=extra_params or {},
    )


def _adapter() -> BreezeTTS2Adapter:
    adapter = object.__new__(BreezeTTS2Adapter)
    adapter.ctx = SimpleNamespace(server=SimpleNamespace())
    return adapter


def test_cfg_scale_one_is_explicitly_supported_without_companion_request():
    adapter = _adapter()

    assert adapter.validate(_request({"guidance_scale": 1.0})) is None
    assert adapter.validate(_request({"cfg_scale": 1.0})) is None


def test_non_unit_cfg_is_rejected_until_negative_branch_exists():
    adapter = _adapter()

    assert "guidance_scale=1.0" in adapter.validate(_request({"guidance_scale": 2.0}))
    assert "guidance_scale=1.0" in adapter.validate(_request({"cfg_scale": 0.5}))
