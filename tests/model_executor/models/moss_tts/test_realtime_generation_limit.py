# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_realtime_talker_stops_at_max_new_frames() -> None:
    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import (
        MossTTSRealtimeTalkerForGeneration,
    )

    talker = MossTTSRealtimeTalkerForGeneration.__new__(MossTTSRealtimeTalkerForGeneration)
    talker.n_vq = 4
    talker.audio_pad_token = 1024
    talker.text_vocab_size = 8
    talker.text_pad_id = 2
    talker.audio_eos_id = 3
    talker._batch_state = None
    talker._build_input_embeds = lambda input_ids, audio_codes: torch.zeros((input_ids.shape[0], 1))

    generated: list[None] = []

    def generate_frame(*args, **kwargs):
        generated.append(None)
        return torch.ones((1, talker.n_vq), dtype=torch.long)

    talker.local_transformer = SimpleNamespace(generate_frame=generate_frame)
    talker.local_lm_heads = []

    _, _, info = talker.preprocess(
        torch.tensor([1, 2]),
        None,
        max_new_frames=[2],
    )
    assert info["audio_state"]["max_new_frames"] == 2

    for _ in range(3):
        output = talker.make_omni_output(
            torch.ones((1, 1)),
            runtime_additional_information=[info],
            request_token_spans=[(0, 1)],
        )

    assert len(generated) == 2
    assert info["audio_state"]["step"] == 2
    assert info["audio_state"]["is_stopping"] is True
    assert output.multimodal_outputs == {}
    assert talker.compute_logits(torch.ones((1, 1))).argmax().item() == talker.audio_eos_id


@pytest.mark.parametrize("seed_source", ["additional_information", "tts_local_seed", "sampling_seed"])
def test_realtime_talker_owns_request_sampling_generator(seed_source: str) -> None:
    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import (
        MossTTSRealtimeTalkerForGeneration,
    )

    talker = MossTTSRealtimeTalkerForGeneration.__new__(MossTTSRealtimeTalkerForGeneration)
    talker.n_vq = 4
    talker.AUDIO_EOS = 1001
    talker.AUDIO_BOS = 1002
    talker.audio_pad_token = 1003
    talker._batch_state = None
    talker._batch_state_spans = None
    talker.local_lm_heads = []
    seen_generators: list[torch.Generator] = []

    def generate_frame_stub(hidden: torch.Tensor, *_args, generators=None, **_kwargs) -> torch.Tensor:
        assert generators is not None and len(generators) == hidden.shape[0]
        assert generators[0] is not None
        seen_generators.append(generators[0])
        return torch.randint(4, 100, (hidden.shape[0], talker.n_vq), generator=generators[0])

    class LocalTransformer:
        def generate_frame(self, *args, **kwargs):
            return generate_frame_stub(*args, **kwargs)

    talker.local_transformer = LocalTransformer()

    def make_request() -> tuple[dict[str, object], SamplingParams | None]:
        info: dict[str, object] = {"audio_state": {"is_stopping": False, "step": 0}}
        params = None
        if seed_source == "additional_information":
            info["seed"] = [17]
        elif seed_source == "tts_local_seed":
            params = SamplingParams(extra_args={"tts_local_seed": 17})
        else:
            params = SamplingParams(seed=17)
        return info, params

    def run_sequence(info: dict[str, object], params: SamplingParams | None) -> list[torch.Tensor]:
        outputs = []
        for _ in range(2):
            kwargs = {"request_sampling_params": [params]} if params is not None else {}
            result = talker.make_omni_output(
                torch.ones((1, 1)),
                model_intermediate_buffer=[info],
                request_token_spans=[(0, 1)],
                **kwargs,
            )
            outputs.append(result.multimodal_outputs["codes"]["audio"][0])
        return outputs

    with torch.random.fork_rng(devices=[]):
        first_info, first_params = make_request()
        torch.manual_seed(1)
        first_sequence = run_sequence(first_info, first_params)
        replay_info, replay_params = make_request()
        torch.manual_seed(999)
        replay_sequence = run_sequence(replay_info, replay_params)

    assert len(seen_generators) == 4
    assert seen_generators[0] is seen_generators[1]
    assert seen_generators[2] is seen_generators[3]
    assert seen_generators[0] is not seen_generators[2]
    first_state = first_info["audio_state"]
    replay_state = replay_info["audio_state"]
    assert isinstance(first_state, dict)
    assert isinstance(replay_state, dict)
    assert first_state["local_generator"] is seen_generators[0]
    assert replay_state["local_generator"] is seen_generators[2]
    for actual, expected in zip(first_sequence, replay_sequence, strict=True):
        torch.testing.assert_close(actual, expected)
