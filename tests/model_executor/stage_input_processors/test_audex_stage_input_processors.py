# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.audex import (
    thinker2code2wav_async_chunk,
    thinker2code2wav_full_payload,
    thinker2code2wav_token_only,
)

_OFFSET = 131077
_SIZE = 65536


def _request(request_id: str, output_token_ids: list[int], finished: bool = False):
    return SimpleNamespace(
        external_req_id=request_id,
        request_id=request_id,
        output_token_ids=output_token_ids,
        is_finished=lambda: finished,
    )


def _transfer_manager(*, chunk_frames: int = 4, initial_chunk_frames: int = 1):
    return SimpleNamespace(
        request_payload={},
        connector=SimpleNamespace(
            config={
                "extra": {
                    "codec_chunk_frames": chunk_frames,
                    "initial_codec_chunk_frames": initial_chunk_frames,
                    "codec_token_offset": _OFFSET,
                    "codec_vocab_size": _SIZE,
                }
            }
        ),
    )


def _codes(payload) -> list[int]:
    audio = payload.codes.audio
    assert audio.ndim == 2 and (audio.numel() == 0 or audio.shape[1] == 1)
    return audio.reshape(-1).tolist()


def test_async_chunk_initial_then_steady_with_holdback():
    tm = _transfer_manager(chunk_frames=4, initial_chunk_frames=1)
    # 2 codec tokens sampled: initial target 1, holdback keeps 1 pending.
    req = _request("r0", [_OFFSET + 5, _OFFSET + 6])
    payload = thinker2code2wav_async_chunk(tm, None, req)
    assert _codes(payload) == [5]
    assert not bool(payload.meta.stream_finished)
    assert payload.meta.req_id == ["r0"]

    # 4 more tokens: pending = 1 + 4 = 5 > steady target 4 -> emit 4, keep 1.
    req.output_token_ids = req.output_token_ids + [_OFFSET + i for i in (7, 8, 9, 10)]
    payload = thinker2code2wav_async_chunk(tm, None, req)
    assert _codes(payload) == [6, 7, 8, 9]

    # No new tokens and not finished: nothing to emit.
    assert thinker2code2wav_async_chunk(tm, None, req) is None


def test_async_chunk_terminal_carries_residual_and_finished():
    tm = _transfer_manager(chunk_frames=4, initial_chunk_frames=1)
    req = _request("r0", [_OFFSET + 1, _OFFSET + 2])
    first = thinker2code2wav_async_chunk(tm, None, req)
    assert _codes(first) == [1]

    req.is_finished = lambda: True
    terminal = thinker2code2wav_async_chunk(tm, None, req)
    assert _codes(terminal) == [2]
    assert bool(terminal.meta.stream_finished)
    assert bool(terminal.meta.finished)

    # Terminal payload is sent exactly once.
    assert thinker2code2wav_async_chunk(tm, None, req) is None


def test_async_chunk_terminal_never_empty_when_codes_exist():
    """The holdback guarantees the terminal chunk carries codes (an empty
    terminal chunk would never reach the decoder for the lookahead flush)."""
    tm = _transfer_manager(chunk_frames=2, initial_chunk_frames=2)
    # Exactly one steady chunk's worth of codes: without holdback the terminal
    # payload would be empty.
    req = _request("r0", [_OFFSET + i for i in range(2)])
    assert thinker2code2wav_async_chunk(tm, None, req) is None  # held back

    req.is_finished = lambda: True
    terminal = thinker2code2wav_async_chunk(tm, None, req)
    assert _codes(terminal) == [0, 1]
    assert bool(terminal.meta.stream_finished)


def test_async_chunk_filters_non_codec_tokens():
    tm = _transfer_manager(chunk_frames=2, initial_chunk_frames=1)
    # Text token, codec token, marker (<speechgen_end> = 131076 < offset).
    req = _request("r0", [42, _OFFSET + 3, _OFFSET - 1], finished=True)
    payload = thinker2code2wav_async_chunk(tm, None, req)
    assert _codes(payload) == [3]


def test_full_payload_ships_all_codes_with_finished():
    tm = _transfer_manager()
    req = _request("r1", [42, _OFFSET + 10, _OFFSET + 11], finished=True)
    payload = thinker2code2wav_full_payload(tm, None, req)
    assert payload["codes"]["audio"] == [10, 11]
    assert bool(payload["meta"]["finished"])
    assert payload["meta"]["req_id"] == ["r1"]


def test_token_only_strips_prompt_prefix():
    source = SimpleNamespace(
        request_id="r2",
        prompt_token_ids=[7, 8],
        finished=True,
        outputs=[SimpleNamespace(cumulative_token_ids=[7, 8, _OFFSET, _OFFSET + 1], multimodal_output=None)],
    )
    inputs = thinker2code2wav_token_only([source])
    assert len(inputs) == 1
    assert inputs[0]["prompt_token_ids"] == [_OFFSET, _OFFSET + 1]


def test_async_chunk_zero_codec_tokens_raises_on_terminal():
    """A request that finishes without ever producing a codec token must fail
    fast (the adapter logs it and the serving layer errors the request)."""
    tm = _transfer_manager(chunk_frames=2, initial_chunk_frames=1)
    req = _request("r0", [42, 43], finished=True)  # only non-codec tokens
    with pytest.raises(ValueError, match="no codec tokens"):
        thinker2code2wav_async_chunk(tm, None, req)
    # Terminal handling is one-shot: subsequent calls stay silent.
    assert thinker2code2wav_async_chunk(tm, None, req) is None


def test_full_payload_zero_codec_tokens_raises():
    tm = _transfer_manager()
    req = _request("r1", [42, 43], finished=True)
    with pytest.raises(ValueError, match="no codec tokens"):
        thinker2code2wav_full_payload(tm, None, req)


def test_meta_finished_tensor_dtype():
    tm = _transfer_manager(chunk_frames=2, initial_chunk_frames=1)
    req = _request("r3", [_OFFSET], finished=True)
    payload = thinker2code2wav_async_chunk(tm, None, req)
    assert isinstance(payload.meta.finished, torch.Tensor)
    assert payload.meta.finished.dtype == torch.bool
    assert isinstance(payload.meta.stream_finished, torch.Tensor)


# ---------------------------------------------------------------- CFG prompt expansion


def _cfg_params(extra_args):
    from vllm.sampling_params import SamplingParams

    return SamplingParams(extra_args=extra_args)


class TestExpandCfgPrompts:
    def _cond_args(self, **overrides):
        args = {
            "cfg_scale": 1.5,
            "cfg_role": "cond",
            "cfg_pair_id": "req-1",
            "cfg_null_prompt": "<unk><unk><unk>",
        }
        args.update(overrides)
        return args

    def test_no_extra_args_expands_to_nothing(self):
        from vllm_omni.model_executor.stage_input_processors.audex import expand_cfg_prompts

        assert expand_cfg_prompts({"prompt": "p"}, _cfg_params(None)) == []

    def test_scale_one_expands_to_nothing(self):
        from vllm_omni.model_executor.stage_input_processors.audex import expand_cfg_prompts

        params = _cfg_params(self._cond_args(cfg_scale=1.0))
        assert expand_cfg_prompts({"prompt": "p"}, params) == []

    def test_non_cond_role_expands_to_nothing(self):
        from vllm_omni.model_executor.stage_input_processors.audex import expand_cfg_prompts

        params = _cfg_params(self._cond_args(cfg_role="uncond"))
        assert expand_cfg_prompts({"prompt": "p"}, params) == []

    def test_malformed_scale_expands_to_nothing(self):
        from vllm_omni.model_executor.stage_input_processors.audex import expand_cfg_prompts

        params = _cfg_params(self._cond_args(cfg_scale="big"))
        assert expand_cfg_prompts({"prompt": "p"}, params) == []

    @pytest.mark.parametrize("missing", ["cfg_pair_id", "cfg_null_prompt"])
    def test_missing_pair_metadata_raises(self, missing):
        from vllm_omni.model_executor.stage_input_processors.audex import expand_cfg_prompts

        args = self._cond_args()
        del args[missing]
        with pytest.raises(ValueError, match="cfg_pair_id/cfg_null_prompt"):
            expand_cfg_prompts({"prompt": "p"}, _cfg_params(args))

    def test_companion_shape_for_dict_prompt(self):
        from vllm_omni.model_executor.stage_input_processors.audex import (
            CFG_UNCOND_SUFFIX,
            expand_cfg_prompts,
        )

        prompt = {"prompt": "cond text", "modalities": ["audio"], "prompt_token_ids": [1, 2]}
        params = _cfg_params(self._cond_args())
        (companion,) = expand_cfg_prompts(prompt, params)

        assert companion.role == "uncond"
        assert companion.request_id_suffix == CFG_UNCOND_SUFFIX
        assert companion.prompt["prompt"] == "<unk><unk><unk>"
        assert companion.prompt["modalities"] == ["audio"]
        assert "prompt_token_ids" not in companion.prompt

        patched, _ = companion.apply_overrides(params, [params])
        assert patched.extra_args == {"cfg_scale": 1.5, "cfg_role": "uncond", "cfg_pair_id": "req-1"}
        # The original cond params must not be mutated by the override.
        assert params.extra_args["cfg_role"] == "cond"

    def test_companion_shape_for_string_prompt(self):
        from vllm_omni.model_executor.stage_input_processors.audex import expand_cfg_prompts

        (companion,) = expand_cfg_prompts("cond text", _cfg_params(self._cond_args()))
        assert companion.prompt == "<unk><unk><unk>"
