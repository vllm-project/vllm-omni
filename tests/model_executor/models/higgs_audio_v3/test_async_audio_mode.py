# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("enabled", [False, True])
def test_model_metadata_reset(enabled):
    from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_talker import (
        HiggsAudioV3TalkerForConditionalGeneration as C,
    )

    t = C.__new__(C)
    torch.nn.Module.__init__(t)
    t.config = SimpleNamespace(audio_async_prompt_mode=enabled)
    t._set_last_step_query_start_loc = lambda x: None
    t._sync_decode_state_with_batch = lambda x: None
    t.update_decode_step_metadata(audio_prompt_mode_rows=2)
    assert t._step_audio_mode_rows == (2 if enabled else 0)
    t.update_decode_step_metadata()
    assert t._step_audio_mode_rows == 0


@pytest.mark.parametrize("bad_state", [False, True])
def test_actual_state_guard_and_eos(bad_state):
    import importlib.util
    from pathlib import Path

    import vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_talker as mod

    root = Path(mod.__file__).parents[4]
    spec = importlib.util.spec_from_file_location(
        "helper_fixture", root / "tests/model_executor/models/higgs_audio_v3/test_higgs_audio_v3.py"
    )
    assert spec is not None and spec.loader is not None
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    t = helpers.TestSamplerMethods()._make_batched_sampler_talker(2)
    t._restore_terminal_audio_rows = (
        mod.HiggsAudioV3TalkerForConditionalGeneration._restore_terminal_audio_rows.__get__(t)
    )
    t.config = SimpleNamespace(audio_async_prompt_mode=True, audio_full_sample_graph=False)
    t._resolve_token_ids = lambda: None
    t._audio_continuation_id = 99999
    t._eos_token_id = 151671
    t._last_logits_hidden = torch.zeros(2, 16)
    t._last_step_input_ids = torch.tensor([99999, 12345 if bad_state else 151671])
    t._last_step_query_start_loc = None
    t._decode_has_codes = torch.tensor([True, False])
    t._decode_generation_done = torch.tensor([False, False])
    t._decode_delay_count = torch.zeros(2, dtype=torch.long)
    t._decode_eoc_countdown = torch.full((2,), -1, dtype=torch.long)
    t._fast_audio_direct_rows = 0
    t._step_audio_tail_rows = 0
    t._step_audio_mode_rows = 2
    t._fast_audio_sampler_gpu_fallback_reason = lambda **kw: None
    t._audio_codebook_logits_from_rows = lambda hidden, rows, all_rows=False: torch.zeros(2, 8, 1026)
    t._apply_delay_pattern_masking_batched = lambda *a, **kw: None
    t._sample_audio_codes = lambda logits, *a, **kw: torch.zeros(logits.shape[0], dtype=torch.long)
    seen_codes = {}
    t._update_delay_state_batched = lambda *a, **kw: seen_codes.update(kw)

    def call():
        return mod.HiggsAudioV3TalkerForConditionalGeneration.sample(
            t, torch.zeros(2, 200000), SimpleNamespace(no_penalties=True)
        )

    if bad_state:
        with pytest.raises(RuntimeError, match="tail mismatch"):
            call()
    else:
        assert call().sampled_token_ids.tolist() == [[99999], [151671]]
        assert seen_codes["code_row_mask"].tolist() == [True, False]
