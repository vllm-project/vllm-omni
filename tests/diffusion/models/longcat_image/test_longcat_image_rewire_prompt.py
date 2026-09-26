# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.models.longcat_image.pipeline_longcat_image import LongCatImagePipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _ProcessorInputs(dict):
    @property
    def input_ids(self):
        return self["input_ids"]

    def to(self, _device):
        return self


class _FakeProcessor:
    def apply_chat_template(self, *_args, **_kwargs):
        return "chat prompt"

    def __call__(self, text, **_kwargs):
        return _ProcessorInputs(input_ids=torch.tensor([[1, 2]] * len(text)))

    def batch_decode(self, sequences, **_kwargs):
        return [" ".join(str(int(token_id)) for token_id in sequence) for sequence in sequences]


class _StrictTextEncoder:
    """Stands in for the ``transformers`` prompt-rewriting encoder.

    ``GenerationMixin.generate()`` samples off the global RNG and rejects
    keyword arguments it does not know about, so both halves of that contract
    are enforced here.
    """

    _KNOWN_KWARGS = frozenset({"input_ids", "max_new_tokens"})

    def __init__(self):
        self.kwargs_seen = []

    def to(self, _device):
        return self

    def generate(self, **kwargs):
        unused = sorted(set(kwargs) - self._KNOWN_KWARGS)
        if unused:
            # transformers.GenerationMixin._validate_model_kwargs
            raise ValueError(f"The following model_kwargs are not used by the model: {unused}")
        self.kwargs_seen.append(kwargs)
        batch_size = kwargs["input_ids"].shape[0]
        return torch.multinomial(torch.ones(batch_size, 64), num_samples=6, replacement=True)


def _rewire_with(generator):
    pipe = LongCatImagePipeline.__new__(LongCatImagePipeline)
    pipe.text_processor = _FakeProcessor()
    pipe.text_encoder = _StrictTextEncoder()
    pipe.tokenizer_max_length = 32

    rewired = pipe.rewire_prompt(["a cat on a table", "桌上的一只猫"], torch.device("cpu"), generator=generator)

    return rewired, pipe.text_encoder


def test_rewire_prompt_is_reproducible_for_a_fixed_seed():
    first, encoder = _rewire_with(torch.Generator().manual_seed(1234))
    again, _ = _rewire_with(torch.Generator().manual_seed(1234))
    other, _ = _rewire_with(torch.Generator().manual_seed(4321))

    # generate() never takes the request's generator: _validate_model_kwargs
    # would reject it and rewire_prompt would raise instead of rewriting.
    assert encoder.kwargs_seen and "generator" not in encoder.kwargs_seen[0]
    assert first == again
    assert first != other


def test_rewire_prompt_accepts_one_generator_per_prompt():
    from_list, _ = _rewire_with([torch.Generator().manual_seed(1234), torch.Generator().manual_seed(4321)])
    from_single, _ = _rewire_with(torch.Generator().manual_seed(1234))

    assert from_list == from_single
