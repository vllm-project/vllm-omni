# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm_omni.model_executor.stage_input_processors.indextts2 import (
    _strip_stop_token,
    talker2s2mel_full_payload,
    talker2s2mel_token_only,
)

STOP_MEL_TOKEN = 8193
LATENT_DIM = 16


def _make_talker_output(
    *,
    mel_codes: torch.Tensor,
    latent: torch.Tensor,
    s_ref: torch.Tensor | None = None,
    ref_mel: torch.Tensor | None = None,
    style: torch.Tensor | None = None,
    finished: bool = True,
):
    meta = {}
    if s_ref is not None:
        meta["S_ref"] = s_ref
    if ref_mel is not None:
        meta["ref_mel"] = ref_mel
    if style is not None:
        meta["style"] = style

    mm = {
        "codes": {"mel": mel_codes},
        "hidden_states": {"latent": latent},
        "meta": meta,
    }
    return SimpleNamespace(
        finished=finished,
        outputs=[SimpleNamespace(multimodal_output=mm)],
    )


# ── _strip_stop_token ──


def test_strip_stop_token_basic():
    codes = torch.tensor([10, 20, 30, STOP_MEL_TOKEN, 99])
    latent = torch.randn(5, LATENT_DIM)

    c, lat, lens = _strip_stop_token(codes, latent)

    assert c.shape == (1, 3)
    assert lat.shape == (1, 3, LATENT_DIM)
    assert lens.tolist() == [3]
    assert (c[0] == torch.tensor([10, 20, 30])).all()


def test_strip_stop_token_no_stop():
    codes = torch.tensor([10, 20, 30])
    latent = torch.randn(3, LATENT_DIM)

    c, lat, lens = _strip_stop_token(codes, latent)

    assert c.shape == (1, 3)
    assert lens.tolist() == [3]


def test_strip_stop_token_at_start():
    codes = torch.tensor([STOP_MEL_TOKEN, 10, 20])
    latent = torch.randn(3, LATENT_DIM)

    c, lat, lens = _strip_stop_token(codes, latent)

    assert c.shape == (1, 0)
    assert lat.shape == (1, 0, LATENT_DIM)
    assert lens.tolist() == [0]


def test_strip_stop_token_batch_with_padding():
    codes = torch.tensor(
        [
            [10, 20, STOP_MEL_TOKEN, 0],
            [30, 40, 50, STOP_MEL_TOKEN],
        ]
    )
    latent = torch.randn(2, 4, LATENT_DIM)

    c, lat, lens = _strip_stop_token(codes, latent)

    assert c.shape == (2, 3)
    assert lat.shape == (2, 3, LATENT_DIM)
    assert lens.tolist() == [2, 3]
    assert c[0, 2].item() == 0  # padded with a valid semantic code


def test_strip_stop_token_all_stop():
    codes = torch.tensor([STOP_MEL_TOKEN])
    latent = torch.randn(1, LATENT_DIM)

    c, lat, lens = _strip_stop_token(codes, latent)

    assert lens.tolist() == [0]
    assert c.shape[1] == 0
    assert lat.shape[1] == 0


# ── full-payload connector path ──


def _conditioning_tensors():
    return {
        "S_ref": torch.randn(1, 3, LATENT_DIM),
        "ref_mel": torch.randn(1, 80, 5),
        "style": torch.randn(1, 4),
    }


def test_talker2s2mel_token_only_placeholder_finished_only():
    finished = _make_talker_output(mel_codes=torch.tensor([1]), latent=torch.randn(1, LATENT_DIM), finished=True)
    unfinished = _make_talker_output(mel_codes=torch.tensor([2]), latent=torch.randn(1, LATENT_DIM), finished=False)

    outputs = talker2s2mel_token_only([unfinished, finished])

    assert len(outputs) == 1
    assert outputs[0]["prompt_token_ids"] == [0]
    assert outputs[0]["additional_information"] is None
    assert outputs[0]["multi_modal_data"] is None


def test_talker2s2mel_full_payload_flat_keys_builds_s2mel_contract():
    cond = _conditioning_tensors()
    latent = torch.randn(3, LATENT_DIM)
    payload = {
        "codes.mel": torch.tensor([[10], [20], [30]]),
        "hidden_states.latent": latent,
        "meta.S_ref": cond["S_ref"],
        "meta.ref_mel": cond["ref_mel"],
        "meta.style": cond["style"],
    }

    result = talker2s2mel_full_payload(None, payload, SimpleNamespace(request_id="r-flat"))

    assert result is not None
    assert result["mel_codes"].shape == (1, 3)
    assert result["mel_codes"].tolist() == [[10, 20, 30]]
    assert result["latent"].shape == (1, 3, LATENT_DIM)
    assert result["code_lens"].tolist() == [3]
    assert result["S_ref"].device.type == "cpu"
    assert result["ref_mel"].device.type == "cpu"
    assert result["style"].device.type == "cpu"
    assert result["S_ref"].data_ptr() != cond["S_ref"].data_ptr()


def test_talker2s2mel_full_payload_nested_fallback_input():
    cond = _conditioning_tensors()
    payload = {
        "codes": {"mel": torch.tensor([[4], [5]])},
        "hidden_states": {"latent": torch.randn(2, LATENT_DIM)},
        "meta": cond,
    }

    result = talker2s2mel_full_payload(None, payload, SimpleNamespace(request_id="r-nested"))

    assert result is not None
    assert result["mel_codes"].tolist() == [[4, 5]]
    assert result["latent"].shape == (1, 2, LATENT_DIM)


def test_talker2s2mel_full_payload_normalizes_one_by_t_mel_row():
    cond = _conditioning_tensors()
    payload = {
        "codes.mel": torch.tensor([[7, 8, 9]]),
        "hidden_states.latent": torch.randn(3, LATENT_DIM),
        "meta": cond,
    }

    result = talker2s2mel_full_payload(None, payload, SimpleNamespace(request_id="r-row"))

    assert result is not None
    assert result["mel_codes"].tolist() == [[7, 8, 9]]


def test_talker2s2mel_full_payload_trims_stop_token_and_matching_latent():
    cond = _conditioning_tensors()
    latent = torch.arange(4 * LATENT_DIM, dtype=torch.float32).reshape(4, LATENT_DIM)
    payload = {
        "codes.mel": torch.tensor([[1], [2], [STOP_MEL_TOKEN], [99]]),
        "hidden_states.latent": latent,
        "meta": cond,
    }

    result = talker2s2mel_full_payload(None, payload, SimpleNamespace(request_id="r-stop"))

    assert result is not None
    assert result["mel_codes"].tolist() == [[1, 2]]
    assert result["latent"].shape == (1, 2, LATENT_DIM)
    assert result["latent"][0, 0].tolist() == latent[0].tolist()
    assert result["latent"][0, 1].tolist() == latent[1].tolist()
    assert result["code_lens"].tolist() == [2]


def test_talker2s2mel_full_payload_crops_mel_latent_to_common_length():
    cond = _conditioning_tensors()
    payload = {
        "codes.mel": torch.tensor([[1], [2], [3], [4]]),
        "hidden_states.latent": torch.randn(2, LATENT_DIM),
        "meta": cond,
    }

    result = talker2s2mel_full_payload(None, payload, SimpleNamespace(request_id="r-crop"))

    assert result is not None
    assert result["mel_codes"].tolist() == [[1, 2]]
    assert result["latent"].shape == (1, 2, LATENT_DIM)
    assert result["code_lens"].tolist() == [2]


def test_talker2s2mel_full_payload_missing_required_fields_returns_none():
    assert talker2s2mel_full_payload(None, {"hidden_states.latent": torch.randn(1, LATENT_DIM)}, None) is None
    assert talker2s2mel_full_payload(None, {"codes.mel": torch.tensor([[1]])}, None) is None
