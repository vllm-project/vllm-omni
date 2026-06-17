# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_omni.model_executor.models.indextts2.indextts2_talker import IndexTTS2TalkerForConditionalGeneration


def _make_talker():
    talker = object.__new__(IndexTTS2TalkerForConditionalGeneration)
    talker.omni_payload_at_request_end = False
    return talker


def test_make_omni_output_exports_delta_mel_row_and_aligned_latent_row():
    talker = _make_talker()
    hidden = torch.randn(1, 4)
    latent_acc = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4)
    info = {
        "codes": {"mel": torch.tensor([11, 22, 33])},
        "hidden_states": {"latent": torch.full((3, 4), -1.0)},
        "meta": {"latent_acc": latent_acc},
    }

    out = talker.make_omni_output(hidden, model_intermediate_buffer=[info])

    mm = out.multimodal_outputs
    assert mm is not None
    assert mm["codes"]["mel"][0].shape == (1, 1)
    assert mm["codes"]["mel"][0].tolist() == [[33]]
    assert mm["hidden_states"]["latent"][0].shape == (1, 4)
    assert mm["hidden_states"]["latent"][0].tolist() == latent_acc[2:3].tolist()
    assert "meta" not in mm


def test_make_omni_output_emits_conditioning_only_on_first_useful_mel_step():
    talker = _make_talker()
    hidden = torch.randn(1, 4)
    s_ref = torch.randn(1, 2, 4)
    ref_mel = torch.randn(1, 80, 3)
    style = torch.randn(1, 4)

    first = talker.make_omni_output(
        hidden,
        model_intermediate_buffer=[
            {
                "codes": {"mel": torch.tensor([11])},
                "meta": {
                    "latent_acc": torch.randn(1, 4),
                    "S_ref": s_ref,
                    "ref_mel": ref_mel,
                    "style": style,
                },
            }
        ],
    )
    later = talker.make_omni_output(
        hidden,
        model_intermediate_buffer=[
            {
                "codes": {"mel": torch.tensor([11, 22])},
                "meta": {
                    "latent_acc": torch.randn(2, 4),
                    "S_ref": s_ref,
                    "ref_mel": ref_mel,
                    "style": style,
                },
            }
        ],
    )

    assert first.multimodal_outputs is not None
    assert first.multimodal_outputs["meta"][0]["S_ref"] is s_ref
    assert first.multimodal_outputs["meta"][0]["ref_mel"] is ref_mel
    assert first.multimodal_outputs["meta"][0]["style"] is style
    assert later.multimodal_outputs is not None
    assert "meta" not in later.multimodal_outputs


def test_make_omni_output_does_not_emit_meta_for_zero_mel_placeholder_in_mixed_batch():
    talker = _make_talker()
    hidden = torch.randn(2, 4)
    s_ref = torch.randn(1, 2, 4)
    ref_mel = torch.randn(1, 80, 3)
    style = torch.randn(1, 4)

    out = talker.make_omni_output(
        hidden,
        model_intermediate_buffer=[
            {
                "codes": {"mel": torch.zeros(0, dtype=torch.long)},
                "meta": {
                    "latent_acc": torch.zeros(0, 4),
                    "S_ref": s_ref,
                    "ref_mel": ref_mel,
                    "style": style,
                },
            },
            {
                "codes": {"mel": torch.tensor([22])},
                "meta": {"latent_acc": torch.randn(1, 4)},
            },
        ],
    )

    mm = out.multimodal_outputs
    assert mm is not None
    assert mm["codes"]["mel"][0].numel() == 0
    assert mm["codes"]["mel"][1].tolist() == [[22]]
    assert "meta" not in mm


def test_make_omni_output_uses_current_latent_fallback_as_delta_only():
    talker = _make_talker()
    hidden = torch.randn(1, 4)
    latent = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4)
    info = {
        "codes": {"mel": torch.tensor([44, 55])},
        "hidden_states": {"latent": latent},
        "meta": {},
    }

    out = talker.make_omni_output(hidden, model_intermediate_buffer=[info])

    mm = out.multimodal_outputs
    assert mm is not None
    assert mm["codes"]["mel"][0].tolist() == [[55]]
    assert mm["hidden_states"]["latent"][0].shape == (1, 4)
    assert mm["hidden_states"]["latent"][0].tolist() == latent[-1:].tolist()
