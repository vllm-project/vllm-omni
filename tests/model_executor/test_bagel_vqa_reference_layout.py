import torch

from vllm_omni.utils.bagel_vqa import (
    BAGEL_VLM_THINK_SYSTEM_PROMPT,
    bagel_vqa_reference_layout_enabled,
    build_bagel_vqa_image_spans,
    build_bagel_vqa_rope_positions,
    build_bagel_vqa_reference_prompt_token_ids,
)


class FakeTokenizer:
    vocab = {
        "<|im_start|>": 100,
        "<|im_end|>": 101,
        "<|image_pad|>": 102,
    }

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]

    def encode(self, text):
        return [ord(ch) for ch in text]


def test_bagel_vqa_serving_prompt_uses_reference_layout():
    messages = [
        {"role": "system", "content": BAGEL_VLM_THINK_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                {
                    "type": "text",
                    "text": f"{BAGEL_VLM_THINK_SYSTEM_PROMPT}\n\nquestion",
                },
                {"type": "text", "text": "briefly"},
            ],
        },
        {"role": "assistant", "content": "<think>\n"},
    ]

    ids, prompt, image_count = build_bagel_vqa_reference_prompt_token_ids(
        messages,
        FakeTokenizer(),
    )

    assert ids[0] == FakeTokenizer.vocab["<|im_start|>"]
    assert ids[-1] == FakeTokenizer.vocab["<|im_start|>"]
    assert FakeTokenizer.vocab["<|image_pad|>"] in ids
    assert prompt == "<img><|image_1|></img> question briefly"
    assert image_count == 1


def test_bagel_vqa_logical_rope_collapses_vision_block_and_decode_continues():
    states = {}
    input_ids = torch.tensor([1, 10, 2, 3, 11, 4])
    positions = torch.arange(6)

    rope = build_bagel_vqa_rope_positions(
        input_ids=input_ids,
        positions=positions,
        req_ids=["r0"],
        num_computed_tokens=[0],
        num_scheduled_tokens=[6],
        rope_states=states,
        start_of_image_id=10,
        end_of_image_id=11,
    )

    assert rope.tolist() == [0, 1, 1, 1, 1, 2]

    decode_rope = build_bagel_vqa_rope_positions(
        input_ids=torch.tensor([5]),
        positions=torch.tensor([6]),
        req_ids=["r0"],
        num_computed_tokens=[6],
        num_scheduled_tokens=[1],
        rope_states=states,
        start_of_image_id=10,
        end_of_image_id=11,
    )

    assert decode_rope.tolist() == [3]


def test_bagel_vqa_logical_rope_survives_chunked_image_prefill():
    states = {}

    first = build_bagel_vqa_rope_positions(
        input_ids=torch.tensor([1, 10, 2]),
        positions=torch.tensor([0, 1, 2]),
        req_ids=["r0"],
        num_computed_tokens=[0],
        num_scheduled_tokens=[3],
        rope_states=states,
        start_of_image_id=10,
        end_of_image_id=11,
    )
    second = build_bagel_vqa_rope_positions(
        input_ids=torch.tensor([3, 11, 4]),
        positions=torch.tensor([3, 4, 5]),
        req_ids=["r0"],
        num_computed_tokens=[3],
        num_scheduled_tokens=[3],
        rope_states=states,
        start_of_image_id=10,
        end_of_image_id=11,
    )

    assert first.tolist() == [0, 1, 1]
    assert second.tolist() == [1, 1, 2]


def test_bagel_reference_prefill_implies_reference_layout(monkeypatch):
    monkeypatch.delenv("BAGEL_VQA_REFERENCE_LAYOUT", raising=False)
    monkeypatch.setenv("BAGEL_VQA_REFERENCE_PREFILL", "1")

    assert bagel_vqa_reference_layout_enabled()


def test_bagel_vqa_image_spans_include_kv_end_before_future_text():
    spans = build_bagel_vqa_image_spans(
        input_ids=torch.tensor([1, 10, 2, 3, 11, 4, 5]),
        req_ids=["r0"],
        num_computed_tokens=[20],
        num_scheduled_tokens=[7],
        start_of_image_id=10,
        end_of_image_id=11,
    )

    assert spans == [
        {
            "req_idx": 0,
            "request_start": 0,
            "num_computed_tokens": 20,
            "q_start": 1,
            "q_end": 5,
            "kv_local_end": 5,
            "kv_end": 25,
        }
    ]
