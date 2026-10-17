# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Golden tests for the consolidated ``_common`` stage-input-processor helpers.

Each case encodes a documented semantic difference (None handling, tensor ->
list, ``ConstantList._x`` unwrapping, tuple handling, codec-frame validity
masks, delay-pattern strictness, ...).  Modules whose optional dependencies are
unavailable in the current environment are skipped and exercised on CI.
"""

import functools
import importlib
from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_PREFIX = "vllm_omni.model_executor.stage_input_processors."


def _import(name: str) -> Any:
    """Import a stage-input-processor module (or diffusion output_formatter)."""
    try:
        if name == "diffusion.output_formatter":
            return importlib.import_module("vllm_omni.diffusion.output_formatter")
        if name == "higgs_audio_v2":
            # higgs_audio_v2 keeps its own local _extract_last_frame.
            return importlib.import_module(_PREFIX + name)
        return importlib.import_module(_PREFIX + name)
    except Exception as exc:  # pragma: no cover - env dependent
        pytest.skip(f"module {name!r} not importable here: {type(exc).__name__}: {exc}")
        raise AssertionError("unreachable")


def _common_import() -> Any:
    """Import the canonical shared helper module."""
    try:
        import vllm_omni.model_executor.stage_input_processors._common as c

        return c
    except Exception as exc:  # pragma: no cover - env dependent
        pytest.skip(f"_common not importable here: {type(exc).__name__}: {exc}")


class _ConstantList:
    """Mimics the ``ConstantList._x`` attribute seen across modules."""

    def __init__(self, values):
        self._x = list(values)


def _norm(value: Any) -> Any:
    """Normalize a result for comparison: tensors -> nested lists."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _is_tensor(value: Any) -> bool:
    return isinstance(value, torch.Tensor)


# ===========================================================================
# _ensure_list
# ===========================================================================


_T2 = torch.tensor([1, 2])
_T22 = torch.tensor([[1, 2], [3, 4]])
_T24 = torch.tensor([1, 2, 3, 4])


def _ensure_list_cases():
    c = _common_import()
    mapping = {
        "qwen3_omni": c.ensure_list_unchanged,
        "ming_flash_omni": c.ensure_list_strict,
        "audex": c.ensure_list_flatten,
        "cosyvoice3": c.ensure_list,
        "step_audio2": c.ensure_list_preserve_none,
        "diffusion.output_formatter": _import("diffusion.output_formatter")._ensure_list,
    }
    # (module, callable, input, expected, expect_typeerror)
    return [
        pytest.param(mapping["qwen3_omni"], [1, 2], [1, 2], False, id="qwen3_omni-list"),
        pytest.param(mapping["qwen3_omni"], _ConstantList([1, 2]), [1, 2], False, id="qwen3_omni-constantlist"),
        pytest.param(mapping["qwen3_omni"], None, None, False, id="qwen3_omni-none"),
        pytest.param(mapping["qwen3_omni"], (1, 2), (1, 2), False, id="qwen3_omni-tuple"),
        pytest.param(mapping["qwen3_omni"], _T2, None, False, id="qwen3_omni-tensor"),
        pytest.param(mapping["ming_flash_omni"], [1, 2], [1, 2], False, id="ming_flash_omni-list"),
        pytest.param(
            mapping["ming_flash_omni"], _ConstantList([1, 2]), [1, 2], False, id="ming_flash_omni-constantlist"
        ),
        pytest.param(mapping["ming_flash_omni"], _T2, [1, 2], False, id="ming_flash_omni-tensor"),
        pytest.param(mapping["ming_flash_omni"], (1, 2), [1, 2], False, id="ming_flash_omni-tuple"),
        pytest.param(mapping["ming_flash_omni"], None, None, True, id="ming_flash_omni-none-raises"),
        pytest.param(mapping["audex"], None, [], False, id="audex-none"),
        pytest.param(mapping["audex"], _T24, [1, 2, 3, 4], False, id="audex-tensor"),
        pytest.param(mapping["audex"], [1, 2], [1, 2], False, id="audex-list"),
        pytest.param(mapping["audex"], (1, 2), [1, 2], False, id="audex-tuple"),
        pytest.param(mapping["audex"], 5, None, True, id="audex-scalar-raises"),
        pytest.param(mapping["cosyvoice3"], None, [], False, id="cosyvoice3-none"),
        pytest.param(mapping["cosyvoice3"], (1, 2), [1, 2], False, id="cosyvoice3-tuple"),
        pytest.param(mapping["cosyvoice3"], 5, [5], False, id="cosyvoice3-scalar"),
        pytest.param(mapping["cosyvoice3"], _ConstantList([1, 2]), [1, 2], False, id="cosyvoice3-constantlist"),
        pytest.param(mapping["cosyvoice3"], [1, 2], [1, 2], False, id="cosyvoice3-list"),
        pytest.param(mapping["step_audio2"], _ConstantList([1, 2]), [1, 2], False, id="step_audio2-constantlist"),
        pytest.param(mapping["step_audio2"], _T2, [1, 2], False, id="step_audio2-tensor"),
        pytest.param(mapping["step_audio2"], (1, 2), [1, 2], False, id="step_audio2-tuple"),
        pytest.param(mapping["step_audio2"], 5, [5], False, id="step_audio2-scalar"),
        pytest.param(mapping["step_audio2"], None, [None], False, id="step_audio2-none"),
        pytest.param(mapping["diffusion.output_formatter"], [1, 2], [1, 2], False, id="output_formatter-list"),
        pytest.param(mapping["diffusion.output_formatter"], None, [], False, id="output_formatter-none"),
        pytest.param(mapping["diffusion.output_formatter"], 5, [5], False, id="output_formatter-scalar"),
    ]


@pytest.mark.parametrize("fn,inp,expected,expect_typeerror", _ensure_list_cases())
def test_ensure_list_golden(fn, inp, expected, expect_typeerror):
    if expect_typeerror:
        with pytest.raises(TypeError):
            fn(inp)
        return
    result = fn(inp)
    if _is_tensor(result):
        # qwen3_omni may return a tensor unchanged.
        assert expected is None
        assert torch.equal(result, _T2)
        return
    assert _norm(result) == expected


def test_output_formatter_ensure_list_wrap_only():
    """output_formatter._ensure_list keeps wrap-only semantics.

    The diffusion formatter is NOT a stage-input helper: a non-list primary
    payload is wrapped as ``[x]`` verbatim — no tensor flattening, no dict-key
    iteration, no row-wise iteration of an iterable (e.g. a PIL ``Image``).
    ``_common.ensure_list`` (the processor flatten) must NOT be reused here.
    """
    mod = _import("diffusion.output_formatter")
    fn = getattr(mod, "_ensure_list")
    assert fn([1, 2]) == [1, 2]
    assert fn(None) == []
    assert fn(5) == [5]
    # A tensor is wrapped whole, NOT flattened via .tolist().
    t = torch.tensor([1, 2])
    (wrapped,) = fn(t)
    assert wrapped is t
    # A dict is wrapped whole, NOT iterated over its keys.
    d = {"a": 1}
    (wrapped_dict,) = fn(d)
    assert wrapped_dict is d

    # A row-iterable (PIL.Image-like) is wrapped whole, NOT walked row by row.
    class _RowIterable:
        def __iter__(self):
            yield from [[1, 2], [3, 4]]

    img = _RowIterable()
    (wrapped_img,) = fn(img)
    assert wrapped_img is img
    # Contrast: the canonical processor flatten would have flattened each.
    c = _common_import()
    assert c.ensure_list(t) == [1, 2]  # flattened, not [t]
    assert c.ensure_list(d) == ["a"]  # dict keys iterated
    assert c.ensure_list(img) == [[1, 2], [3, 4]]  # rows walked


# ===========================================================================
# _extract_last_frame
# ===========================================================================


def _codes_audio(tensor):
    return {"codes": {"audio": tensor}}


def _extract_last_frame_cases():
    """Legacy per-module behavior: one callable + input per table row."""
    c = _common_import()
    mapping = {
        # higgs_audio_v2 keeps its own local range-filtered helper (RFC refactor
        # removed the other per-module helpers; its range filter is not
        # expressible through _common.extract_last_codec_frame).
        "higgs_audio_v2": _import("higgs_audio_v2")._extract_last_frame,
        "fish_speech": functools.partial(
            c.extract_last_codec_frame,
            key_path=("audio_codes",),
            validate="valid_mask",
            to_cpu=True,
            to_long=True,
        ),
        "qwen3_tts": functools.partial(c.extract_last_codec_frame, validate="any"),
        # voxtral_tts: no validity gate, no cpu/long cast.  Under the
        # consolidated implementation a 2-D ``codes.audio`` tensor contributes
        # its LAST row only (legacy voxtral flattened the whole tensor; the
        # refactor narrows this to the last inter-stage frame, matching all
        # other modules).
        "voxtral_tts": functools.partial(c.extract_last_codec_frame, to_long=False),
    }
    higgs = mapping["higgs_audio_v2"]
    fish = mapping["fish_speech"]
    qwen3 = mapping["qwen3_tts"]
    voxtral = mapping["voxtral_tts"]
    return [
        # higgs_audio_v2: codes.audio key path; range filter [-0, _NUM_REAL_CODES).
        pytest.param(higgs, _codes_audio(torch.tensor([[0, 1], [2, 3], [4, 5]])), [4, 5], False, id="higgs-last-row"),
        # >= _NUM_REAL_CODES
        pytest.param(higgs, _codes_audio(torch.tensor([[0, 1], [2, 3], [4, 1024]])), None, False, id="higgs-range"),
        pytest.param(higgs, _codes_audio(torch.tensor([[0, 1], [2, 3], [-1, 5]])), None, False, id="higgs-negative"),
        pytest.param(higgs, _codes_audio(torch.tensor([1, 2, 3])), [1, 2, 3], False, id="higgs-1d"),
        pytest.param(higgs, _codes_audio(torch.empty(0, 4)), None, False, id="higgs-empty"),
        # 3-D -> ValueError
        pytest.param(higgs, _codes_audio(torch.zeros(2, 2, 2)), None, True, id="higgs-3d-raises"),
        pytest.param(higgs, {"codes": {}}, None, False, id="higgs-missing-audio"),
        # fish_speech: TOP-LEVEL audio_codes key; audio_code_valid mask; cpu+long.
        pytest.param(
            fish,
            {"audio_codes": torch.tensor([[0, 1], [2, 3], [4, 5]]), "audio_code_valid": torch.tensor([1, 1, 1])},
            [4, 5],
            False,
            id="fish-valid",
        ),
        pytest.param(
            fish,
            {"audio_codes": torch.tensor([[0, 1], [2, 3], [4, 5]]), "audio_code_valid": torch.tensor([1, 1, 0])},
            None,
            False,
            id="fish-invalid-last",
        ),
        pytest.param(
            fish,
            {"audio_codes": torch.tensor([[0, 1], [2, 3], [4, 5]]), "audio_code_valid": True},
            [4, 5],
            False,
            id="fish-valid-true",
        ),
        pytest.param(
            fish,
            {"audio_codes": torch.tensor([[0, 1], [2, 3], [4, 5]]), "audio_code_valid": False},
            None,
            False,
            id="fish-invalid-false",
        ),
        # no valid key -> frame.any() False
        pytest.param(fish, {"audio_codes": torch.tensor([[0, 1], [2, 3], [0, 0]])}, None, False, id="fish-zero-frame"),
        pytest.param(fish, {"audio_codes": torch.tensor([1, 2, 3])}, [1, 2, 3], False, id="fish-1d"),
        # 1-D semantics: flat frames bypass the validity/zero gate (amy-why-3459 review).
        pytest.param(fish, {"audio_codes": torch.zeros(8, dtype=torch.long)}, [0] * 8, False, id="fish-1d-zeros"),
        pytest.param(
            fish,
            {"audio_codes": torch.zeros(8, dtype=torch.long), "audio_code_valid": torch.tensor([False])},
            [0] * 8,
            False,
            id="fish-1d-zeros-invalid",
        ),
        pytest.param(
            fish,
            {"audio_codes": torch.zeros(8, dtype=torch.long), "audio_code_valid": False},
            [0] * 8,
            False,
            id="fish-1d-zeros-false",
        ),
        pytest.param(fish, {"audio_codes": torch.zeros(2, 2, 2)}, None, True, id="fish-3d-raises"),
        pytest.param(fish, {}, None, False, id="fish-missing-codes"),
        # qwen3_tts: codes.audio; frame.any() gate; long (no cpu).
        pytest.param(qwen3, _codes_audio(torch.tensor([[0, 1], [2, 3], [4, 5]])), [4, 5], False, id="qwen3-last-row"),
        pytest.param(qwen3, _codes_audio(torch.tensor([[0, 1], [2, 3], [0, 0]])), None, False, id="qwen3-zero-frame"),
        pytest.param(qwen3, _codes_audio(torch.tensor([1, 2, 3])), [1, 2, 3], False, id="qwen3-1d"),
        # 1-D semantics: flat frames bypass the zero gate (legacy qwen3_tts behavior).
        pytest.param(qwen3, _codes_audio(torch.zeros(8, dtype=torch.long)), [0] * 8, False, id="qwen3-1d-zeros"),
        pytest.param(qwen3, _codes_audio(torch.zeros(2, 2, 2)), None, True, id="qwen3-3d-raises"),
        pytest.param(qwen3, {"codes": {}}, None, False, id="qwen3-missing-audio"),
        # voxtral_tts: validated via _common.extract_last_codec_frame(to_long=False).
        pytest.param(
            voxtral, _codes_audio(torch.tensor([[0, 1], [2, 3], [4, 5]])), [4, 5], False, id="voxtral-last-row"
        ),
        pytest.param(
            voxtral,
            {"codes": {"audio": [torch.tensor([1, 2]), torch.tensor([3, 4])]}},
            [3, 4],
            False,
            id="voxtral-list-audio",
        ),
        pytest.param(voxtral, {"codes": {}}, None, False, id="voxtral-missing-audio"),
        # 3-D per-step frames [B, 1, C] (Voxtral compute_mm_logits L1075):
        # normalized via targeted squeeze(1) then the 2-D last-row semantics.
        pytest.param(
            voxtral,
            _codes_audio(torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)),
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            False,
            id="voxtral-3d-per-step",
        ),
        # Mirror the real payload: a list whose last element is 3-D (1, 1, C).
        pytest.param(
            voxtral,
            {"codes": {"audio": [torch.ones(1, 1, 4)]}},
            [1.0, 1.0, 1.0, 1.0],
            False,
            id="voxtral-3d-list-element",
        ),
        # Non per-step 3-D layouts keep raising ValueError (dim1 != 1).
        pytest.param(voxtral, _codes_audio(torch.zeros(1, 2, 3)), None, True, id="voxtral-3d-non-singleton-raises"),
        # Generic 3-D (aligned with the fish/qwen3/higgs contracts).
        pytest.param(voxtral, _codes_audio(torch.zeros(2, 2, 2)), None, True, id="voxtral-3d-generic-raises"),
    ]


@pytest.mark.parametrize("fn,payload,expected,expect_error", _extract_last_frame_cases())
def test_extract_last_frame_golden(fn, payload, expected, expect_error):
    if expect_error:
        with pytest.raises(ValueError):
            fn(payload)
        return
    assert _norm(fn(payload)) == expected


# ===========================================================================
# revert_delay_pattern / filter_real_code_frames
# ===========================================================================
# Behavior is locked by
# ``test_common_revert_delay_pattern_matches_golden`` and
# ``test_common_filter_real_code_frames_matches_golden`` below.

# ===========================================================================
# _to_cpu_tensor  (glm_tts)
# ===========================================================================


_T2F = torch.tensor([1.0, 2.0])


@pytest.mark.parametrize(
    "inp,expected",
    [
        pytest.param(_T2F, _T2F, id="tensor"),
        pytest.param([], None, id="empty-list"),
        pytest.param([_T2F], _T2F, id="list-tensor"),
        pytest.param([1, 2], None, id="list-int"),
        pytest.param(None, None, id="none"),
        pytest.param(5, None, id="int"),
    ],
)
def test_to_cpu_tensor_glm_tts_golden(inp, expected):
    result = _common_import().to_cpu_tensor(inp)
    if expected is None:
        assert result is None
    else:
        assert torch.equal(result, expected)


# ===========================================================================
# _to_token_id_list  (dynin_omni; cosyvoice3 recursive variant)
# ===========================================================================


@pytest.mark.parametrize(
    "inp,expected",
    [
        pytest.param(torch.tensor(5), [5], id="scalar-tensor"),
        pytest.param(torch.tensor([1, 2, 3]), [1, 2, 3], id="vec-tensor"),
        pytest.param(torch.tensor([[1, 2], [3, 4]]), [1, 2], id="mat-tensor"),  # 2-D -> first row only
        pytest.param([[1, 2], [3, 4]], [1, 2], id="nested-list"),  # nested list -> first nested row only
        pytest.param([1, 2, 3], [1, 2, 3], id="list"),
        pytest.param(None, [], id="none"),
        pytest.param(5, [5], id="int"),
    ],
)
def test_to_token_id_list_dynin_omni_golden(inp, expected):
    assert _common_import().to_token_id_list(inp) == expected


def test_to_token_id_list_cosyvoice3_recursive_golden():
    c = _common_import()
    fn = functools.partial(c.to_token_id_list, recursive=True)
    # cosyvoice3 recursively flattens ALL nesting (dynin takes first row only).
    assert fn([[1, 2], [3, 4]]) == [1, 2, 3, 4]
    assert fn(None) == []
    assert fn(torch.tensor([[1, 2], [3, 4]])) == [1, 2, 3, 4]
    assert fn(5) == [5]
    # Per-item normalization flattens a list containing a non-scalar tensor and
    # a plain tuple.
    assert fn([torch.tensor([[4, 5]])]) == [4, 5]
    assert fn((4, 5)) == [4, 5]


# ===========================================================================
# placeholder length helper (Qwen chat-template scan).
# ===========================================================================


def test_compute_talker_prompt_ids_length_golden():
    c = _common_import()
    # im_start markers at 0 and 6; user segment len 6, last assistant +9.
    prompt = [151644, 872, 10, 11, 12, 13, 151644, 77091, 20, 21]
    info = {"ids": {"all": prompt, "prompt": prompt}}
    assert c.compute_placeholder_prompt_len(ids_or_prompt=info, mode="full", device="cpu") == 15
    # stage0_only scans the flat stage-0 list (same 15 for the golden prompt).
    assert c.compute_placeholder_prompt_len(ids_or_prompt=prompt, mode="stage0_only") == 15


# ===========================================================================
# The shared _common.py helpers must reproduce the golden-locked behaviour.
# ===========================================================================


def _shim_active() -> bool:
    """True when the test-support import fallback (no real vllm) is active."""
    try:
        import vllm_omni  # noqa: F401

        return bool(getattr(vllm_omni, "_SHIM_ACTIVE", False))
    except Exception:
        return False


def test_common_ensure_list_matches_golden():
    c = _common_import()
    assert c.ensure_list([1, 2]) == [1, 2]
    assert c.ensure_list((1, 2)) == [1, 2]
    assert c.ensure_list(None) == []
    assert c.ensure_list(_ConstantList([1, 2])) == [1, 2]
    assert c.ensure_list(5) == [5]
    # tolist branch (2-D -> list-of-lists, no flattening).
    assert c.ensure_list(torch.tensor([[1, 2], [3, 4]])) == [[1, 2], [3, 4]]
    # Named variants.
    assert c.ensure_list_unchanged(None) is None
    assert c.ensure_list_unchanged((1, 2)) == (1, 2)
    assert c.ensure_list_unchanged([1, 2]) == [1, 2]
    # Strict: non-iterable scalar raises (ming_flash / audex legacy).
    with pytest.raises(TypeError):
        c.ensure_list_strict(5)


def test_common_to_cpu_tensor_matches_golden():
    c = _common_import()
    t = torch.tensor([1.0, 2.0])
    assert torch.equal(c.to_cpu_tensor(t), t)
    assert c.to_cpu_tensor([]) is None
    assert torch.equal(c.to_cpu_tensor([t]), t)
    assert c.to_cpu_tensor([1, 2]) is None
    assert c.to_cpu_tensor(None) is None


def test_common_to_token_id_list_matches_golden():
    c = _common_import()
    # dynin semantics: first nested row only (non-recursive).
    assert c.to_token_id_list(torch.tensor(5)) == [5]
    assert c.to_token_id_list(torch.tensor([1, 2, 3])) == [1, 2, 3]
    assert c.to_token_id_list(torch.tensor([[1, 2], [3, 4]])) == [1, 2]
    assert c.to_token_id_list([[1, 2], [3, 4]]) == [1, 2]
    assert c.to_token_id_list(None) == []
    assert c.to_token_id_list(5) == [5]
    # cosyvoice3 semantics: recursive flatten.
    assert c.to_token_id_list([[1, 2], [3, 4]], recursive=True) == [1, 2, 3, 4]
    assert c.to_token_id_list(torch.tensor([[1, 2], [3, 4]]), recursive=True) == [1, 2, 3, 4]
    # Per-item recursion normalizes a list of non-scalar tensors and a tuple.
    assert c.to_token_id_list([torch.tensor([[4, 5]])], recursive=True) == [4, 5]
    assert c.to_token_id_list((4, 5), recursive=True) == [4, 5]
    assert c.to_token_id_list((torch.tensor(4), torch.tensor([5, 6])), recursive=True) == [4, 5, 6]


def test_common_revert_delay_pattern_matches_golden():
    c = _common_import()
    inp = torch.tensor(
        [
            [10, 1, 2, 3, 4],
            [11, 12, 5, 6, 7],
            [13, 14, 15, 8, 9],
        ]
    )  # [Q=3, T=5]
    assert c.revert_delay_pattern(inp, allow_short=True).tolist() == [[10, 1, 2], [12, 5, 6], [15, 8, 9]]
    # Lenient (v2): T < Q returns input unchanged.
    short = torch.tensor([[1], [2]])
    assert torch.equal(c.revert_delay_pattern(short, allow_short=True), short)
    # Strict (v3): T < Q raises; wrong codebook count raises.
    with pytest.raises(ValueError):
        c.revert_delay_pattern(short, allow_short=False)
    with pytest.raises(ValueError):
        c.revert_delay_pattern(torch.zeros(5, 6), expected_codebooks=3)


def test_common_filter_real_code_frames_matches_golden():
    c = _common_import()
    nrc = 1024
    # frames-first (v2 layout).
    frames_first = torch.tensor(
        [
            [0, 1, 2],
            [nrc, 5, 6],
            [7, 8, 9],
            [-1, 0, 1],
        ]
    )
    assert c.filter_real_code_frames(frames_first, num_real_codes=nrc, layout="frames_first").tolist() == [
        [0, 1, 2],
        [7, 8, 9],
    ]
    # codebooks-first (v3 layout).
    codebooks_first = (
        torch.tensor(
            [
                [0, 1, 2],
                [nrc, 5, 6],
                [7, 8, 9],
                [-1, 0, 1],
            ]
        )
        .t()
        .contiguous()
    )  # [Q=3, frames=4]
    assert c.filter_real_code_frames(codebooks_first, num_real_codes=nrc, layout="codebooks_first").tolist() == [
        [0, 7],
        [1, 8],
        [2, 9],
    ]


def test_common_compute_placeholder_prompt_len_matches_golden():
    c = _common_import()
    prompt = [151644, 872, 10, 11, 12, 13, 151644, 77091, 20, 21]
    # full mode == the legacy Qwen chat-template scan golden (15).
    assert c.compute_placeholder_prompt_len(ids_or_prompt={"ids": {"all": prompt, "prompt": prompt}}, mode="full") == 15
    # stage0_only mode: the same scan on the flat stage-0 list (prewarm) -> 15,
    # so the builder and the inline fallback agree.
    assert c.compute_placeholder_prompt_len(ids_or_prompt=prompt, mode="stage0_only") == 15


def test_common_pack_placeholder_prompt():
    c = _common_import()
    prompt = c.pack_placeholder_prompt(prompt_len=4, voice_metadata={"speaker": 1})
    empty_meta = c.pack_placeholder_prompt(prompt_len=1)
    if _shim_active():
        # Under the import fallback, OmniTokensPrompt's real constructor is
        # bypassed, so field access returns fallback values. Field assertions
        # run where real vllm constructs the model properly.
        pytest.skip("OmniTokensPrompt field construction needs real vllm")
    # Real vllm (0.27) builds OmniTokensPrompt as a dict subclass (MRO
    # ['OmniTokensPrompt', 'dict', 'object']): attribute access is not
    # available, so read the packed fields via dict indexing.
    assert prompt["prompt_token_ids"] == [0, 0, 0, 0]
    assert prompt["additional_information"] == {"speaker": 1}
    assert empty_meta["prompt_token_ids"] == [0]
    # When additional_information is None the key may be absent from the
    # dict, so use .get() for the empty-metadata case.
    assert empty_meta.get("additional_information") is None


# Keep a reference to the case builder so linters don't flag dead code.
_EXTRA_ALIAS = _ensure_list_cases
