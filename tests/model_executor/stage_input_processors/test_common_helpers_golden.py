# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Golden tests that lock the *current* per-module behavior of duplicated
helper functions in ``stage_input_processors``.

Related to RFC #4872 (https://github.com/vllm-project/vllm-omni/issues/4872):
these cases capture the observable per-module semantics that were previously
duplicated across model processors, so a consolidated implementation in
``_common`` must reproduce the same behavior.  Each case encodes a
documented semantic difference between modules (None handling, tensor->list,
``ConstantList._x`` unwrapping, tuple handling, codec-frame validity masks,
delay-pattern strictness, ...).  The shared helpers in ``_common`` must pass
the same cases; where legacy variants disagreed, the consolidated behavior is
preserved through explicit named variants rather than silently normalized.

Modules whose optional dependencies are unavailable in the current
environment are skipped and exercised on CI, where the real dependencies
exist.
"""

import importlib
from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_PREFIX = "vllm_omni.model_executor.stage_input_processors."


def _import(name: str) -> Any:
    try:
        if name == "diffusion.output_formatter":
            # The real module lives under ``vllm_omni.diffusion``, not under
            # ``stage_input_processors`` (which only hosts the golden alias).
            return importlib.import_module("vllm_omni.diffusion.output_formatter")
        return importlib.import_module(_PREFIX + name)
    except Exception as exc:  # pragma: no cover - env dependent
        pytest.skip(f"module {name!r} not importable here: {type(exc).__name__}: {exc}")
        raise AssertionError("unreachable")  # keep type checkers happy


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


def _ensure_list_cases():
    return [
        # (module, input, expected-or-None-if-tensor, expect_typeerror)
        ("qwen3_omni", [1, 2], [1, 2], False),
        ("qwen3_omni", _ConstantList([1, 2]), [1, 2], False),
        # qwen3_omni returns non-list values *unchanged* (incl. None / tuple).
        ("qwen3_omni", None, None, False),
        ("qwen3_omni", (1, 2), (1, 2), False),
        # qwen3_omni has NO tolist() branch: tensor comes back unchanged.
        ("qwen3_omni", "TENSOR", None, False),
        ("ming_flash_omni", [1, 2], [1, 2], False),
        ("ming_flash_omni", _ConstantList([1, 2]), [1, 2], False),
        ("ming_flash_omni", "TENSOR", [1, 2], False),
        ("ming_flash_omni", (1, 2), [1, 2], False),
        # ming_flash_omni: list(None) raises TypeError.
        ("ming_flash_omni", None, None, True),
        ("audex", None, [], False),
        ("audex", "TENSOR", [1, 2, 3, 4], False),
        ("audex", [1, 2], [1, 2], False),
        ("audex", (1, 2), [1, 2], False),
        # audex: list(5) raises TypeError.
        ("audex", 5, None, True),
        ("cosyvoice3", None, [], False),
        ("cosyvoice3", (1, 2), [1, 2], False),
        ("cosyvoice3", 5, [5], False),  # TypeError -> [x]
        ("cosyvoice3", _ConstantList([1, 2]), [1, 2], False),
        ("cosyvoice3", [1, 2], [1, 2], False),
        ("step_audio2", _ConstantList([1, 2]), [1, 2], False),
        ("step_audio2", "TENSOR", [1, 2], False),
        ("step_audio2", (1, 2), [1, 2], False),
        # step_audio2: non-iterable scalar -> [x]; None is not iterable -> [None].
        ("step_audio2", 5, [5], False),
        ("step_audio2", None, [None], False),
        ("diffusion.output_formatter", [1, 2], [1, 2], False),
        ("diffusion.output_formatter", None, [], False),
        ("diffusion.output_formatter", 5, [5], False),
    ]


@pytest.mark.parametrize("module,inp,expected,expect_typeerror", _ensure_list_cases())
def test_ensure_list_golden(module, inp, expected, expect_typeerror):
    mod = _import(module)
    fn = getattr(mod, "_ensure_list")
    if inp == "TENSOR":
        inp = torch.tensor([[1, 2], [3, 4]]) if module == "audex" else torch.tensor([1, 2])
    if expect_typeerror:
        with pytest.raises(TypeError):
            fn(inp)
        return
    result = fn(inp)
    if _is_tensor(result):
        # qwen3_omni may return a tensor unchanged.
        assert _is_tensor(expected) or expected is None
        if expected is None:
            assert torch.equal(result, torch.tensor([1, 2]))
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


@pytest.mark.parametrize(
    "module,payload,expected,expect_error",
    [
        # higgs_audio_v2: codes.audio key path; range filter [-0, _NUM_REAL_CODES).
        ("higgs_audio_v2", _codes_audio(torch.tensor([[0, 1], [2, 3], [4, 5]])), [4, 5], False),
        (
            "higgs_audio_v2",
            _codes_audio(torch.tensor([[0, 1], [2, 3], [4, 1024]])),  # >= _NUM_REAL_CODES
            None,
            False,
        ),
        ("higgs_audio_v2", _codes_audio(torch.tensor([[0, 1], [2, 3], [-1, 5]])), None, False),
        ("higgs_audio_v2", _codes_audio(torch.tensor([1, 2, 3])), [1, 2, 3], False),
        ("higgs_audio_v2", _codes_audio(torch.empty(0, 4)), None, False),
        ("higgs_audio_v2", _codes_audio(torch.zeros(2, 2, 2)), None, True),  # 3-D -> ValueError
        ("higgs_audio_v2", {"codes": {}}, None, False),
        # fish_speech: TOP-LEVEL audio_codes key; audio_code_valid mask; cpu+long.
        (
            "fish_speech",
            {"audio_codes": torch.tensor([[0, 1], [2, 3], [4, 5]]), "audio_code_valid": torch.tensor([1, 1, 1])},
            [4, 5],
            False,
        ),
        (
            "fish_speech",
            {"audio_codes": torch.tensor([[0, 1], [2, 3], [4, 5]]), "audio_code_valid": torch.tensor([1, 1, 0])},
            None,
            False,
        ),
        (
            "fish_speech",
            {"audio_codes": torch.tensor([[0, 1], [2, 3], [4, 5]]), "audio_code_valid": True},
            [4, 5],
            False,
        ),
        (
            "fish_speech",
            {"audio_codes": torch.tensor([[0, 1], [2, 3], [4, 5]]), "audio_code_valid": False},
            None,
            False,
        ),
        (
            "fish_speech",
            {"audio_codes": torch.tensor([[0, 1], [2, 3], [0, 0]])},
            None,
            False,
        ),  # no valid key -> frame.any() False
        ("fish_speech", {"audio_codes": torch.tensor([1, 2, 3])}, [1, 2, 3], False),
        ("fish_speech", {"audio_codes": torch.zeros(2, 2, 2)}, None, True),
        ("fish_speech", {}, None, False),
        # qwen3_tts: codes.audio; frame.any() gate; long (no cpu).
        ("qwen3_tts", _codes_audio(torch.tensor([[0, 1], [2, 3], [4, 5]])), [4, 5], False),
        ("qwen3_tts", _codes_audio(torch.tensor([[0, 1], [2, 3], [0, 0]])), None, False),
        ("qwen3_tts", _codes_audio(torch.tensor([1, 2, 3])), [1, 2, 3], False),
        ("qwen3_tts", _codes_audio(torch.zeros(2, 2, 2)), None, True),
        ("qwen3_tts", {"codes": {}}, None, False),
        # voxtral_tts: for a single tensor it flattens the WHOLE tensor (not just
        # the last row); for a list of tensors it flattens the last tensor.
        ("voxtral_tts", _codes_audio(torch.tensor([[0, 1], [2, 3], [4, 5]])), [0, 1, 2, 3, 4, 5], False),
        ("voxtral_tts", {"codes": {"audio": [torch.tensor([1, 2]), torch.tensor([3, 4])]}}, [3, 4], False),
        ("voxtral_tts", {"codes": {}}, None, False),
    ],
)
def test_extract_last_frame_golden(module, payload, expected, expect_error):
    mod = _import(module)
    fn = getattr(mod, "_extract_last_frame")
    if expect_error:
        with pytest.raises(ValueError):
            fn(payload)
        return
    result = fn(payload)
    assert _norm(result) == expected


# ===========================================================================
# _revert_delay_pattern  (higgs_audio_v2 lenient vs higgs_audio_v3 strict)
# ===========================================================================


def test_revert_delay_pattern_higgs_v2_golden():
    mod = _import("higgs_audio_v2")
    fn = getattr(mod, "_revert_delay_pattern")
    inp = torch.tensor(
        [
            [10, 1, 2, 3, 4],
            [11, 12, 5, 6, 7],
            [13, 14, 15, 8, 9],
        ]
    )  # [Q=3, T=5], seq_len = 3
    out = fn(inp)
    assert out.shape == (3, 3)
    assert out.tolist() == [[10, 1, 2], [12, 5, 6], [15, 8, 9]]
    # Lenient: strictly T < Q returns input unchanged.
    short = torch.tensor([[1], [2]])  # Q=2, T=1 -> t < q True
    assert torch.equal(fn(short), short)
    # Non-lenient: T >= Q processes (even when the result is narrower).
    proc = torch.tensor([[1, 2], [3, 4]])  # Q=2, T=2 -> seq_len = 1
    assert fn(proc).tolist() == [[1], [4]]


def test_revert_delay_pattern_higgs_v3_golden():
    mod = _import("higgs_audio_v3")
    fn = getattr(mod, "_revert_delay_pattern")
    q = int(mod._NUM_CODEBOOKS)
    # Build a [Q, T] input with T = q + 2 (seq_len = 3).
    t = q + 2
    inp = torch.arange(q * t, dtype=torch.long).reshape(q, t)
    out = fn(inp)
    assert out.shape == (q, 3)
    expected = torch.cat([inp[i : i + 1, i : 3 + i] for i in range(q)], dim=0)
    assert torch.equal(out, expected)
    # Strict: wrong codebook count raises.
    with pytest.raises(ValueError):
        fn(torch.zeros(q + 1, t, dtype=torch.long))
    # Strict: T < Q raises.
    with pytest.raises(ValueError):
        fn(torch.zeros(q, q - 1, dtype=torch.long))


# ===========================================================================
# _filter_real_code_frames  (higgs v2 [frames,Q] vs v3 [Q,frames])
# ===========================================================================


def test_filter_real_code_frames_higgs_v2_golden():
    mod = _import("higgs_audio_v2")
    fn = getattr(mod, "_filter_real_code_frames")
    nrc = int(mod._NUM_REAL_CODES)
    inp = torch.tensor(
        [
            [0, 1, 2],
            [nrc, 5, 6],  # invalid (>= _NUM_REAL_CODES)
            [7, 8, 9],
            [-1, 0, 1],  # invalid (< 0)
        ]
    )
    out = fn(inp)
    assert out.tolist() == [[0, 1, 2], [7, 8, 9]]
    # Empty input passes through.
    empty = torch.empty(0, 3, dtype=torch.long)
    assert torch.equal(fn(empty), empty)


def test_filter_real_code_frames_higgs_v3_golden():
    mod = _import("higgs_audio_v3")
    fn = getattr(mod, "_filter_real_code_frames")
    nrc = int(mod._NUM_REAL_CODES)
    # Input is [Q, frames]: 4 frames x 3 codebooks -> transpose to [Q=3, frames=4].
    inp = (
        torch.tensor(
            [
                [0, 1, 2],  # frame 0 (valid)
                [nrc, 5, 6],  # frame 1 (invalid: >= _NUM_REAL_CODES)
                [7, 8, 9],  # frame 2 (valid)
                [-1, 0, 1],  # frame 3 (invalid: < 0)
            ]
        )
        .t()
        .contiguous()
    )
    out = fn(inp)  # keeps frames 0 and 2
    assert out.shape == (3, 2)
    assert out.tolist() == [[0, 7], [1, 8], [2, 9]]
    empty = torch.empty(3, 0, dtype=torch.long)
    assert torch.equal(fn(empty), empty)


# ===========================================================================
# _to_cpu_tensor  (glm_tts)
# ===========================================================================


@pytest.mark.parametrize(
    "inp,expected",
    [
        ("TENSOR", "TENSOR"),
        ([], None),
        ("LIST_TENSOR", "LIST_TENSOR"),
        ([1, 2], None),
        (None, None),
        (5, None),
    ],
)
def test_to_cpu_tensor_glm_tts_golden(inp, expected):
    mod = _import("glm_tts")
    fn = getattr(mod, "_to_cpu_tensor")
    t = torch.tensor([1.0, 2.0])
    if inp == "TENSOR":
        value = t
    elif inp == "LIST_TENSOR":
        value = [t]
    else:
        value = inp
    result = fn(value)
    if expected == "TENSOR":
        assert torch.equal(result, t)
    elif expected == "LIST_TENSOR":
        assert torch.equal(result, t)
    else:
        assert result is expected


# ===========================================================================
# _to_token_id_list  (dynin_omni; cosyvoice3 recursive variant)
# ===========================================================================


@pytest.mark.parametrize(
    "inp,expected",
    [
        ("SCALAR_TENSOR", [5]),
        ("VEC_TENSOR", [1, 2, 3]),
        ("MAT_TENSOR", [1, 2]),  # 2-D -> first row only
        ("NESTED_LIST", [1, 2]),  # nested list -> first nested row only
        ([1, 2, 3], [1, 2, 3]),
        (None, []),
        (5, [5]),
    ],
)
def test_to_token_id_list_dynin_omni_golden(inp, expected):
    mod = _import("dynin_omni")
    fn = getattr(mod, "_to_token_id_list")
    if inp == "SCALAR_TENSOR":
        value = torch.tensor(5)
    elif inp == "VEC_TENSOR":
        value = torch.tensor([1, 2, 3])
    elif inp == "MAT_TENSOR":
        value = torch.tensor([[1, 2], [3, 4]])
    elif inp == "NESTED_LIST":
        value = [[1, 2], [3, 4]]
    else:
        value = inp
    assert fn(value) == expected


def test_to_token_id_list_cosyvoice3_recursive_golden():
    mod = _import("cosyvoice3")
    fn = getattr(mod, "_to_token_id_list")
    # cosyvoice3 recursively flattens ALL nesting (dynin takes first row only).
    assert fn([[1, 2], [3, 4]]) == [1, 2, 3, 4]
    assert fn(None) == []
    assert fn(torch.tensor([[1, 2], [3, 4]])) == [1, 2, 3, 4]
    assert fn(5) == [5]
    # P2 deep-dive parity: per-item normalization flattens a list containing a
    # non-scalar tensor and a plain tuple.
    assert fn([torch.tensor([[4, 5]])]) == [4, 5]
    assert fn((4, 5)) == [4, 5]


# ===========================================================================
# P8a: placeholder length helper (Qwen chat-template scan).
# The legacy ``qwen3_omni._compute_talker_prompt_ids_length`` is dead after the
# P8b consolidation; the canonical ``_common.compute_placeholder_prompt_len``
# reproduces the golden result.
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


def _common_import() -> Any:
    try:
        import vllm_omni.model_executor.stage_input_processors._common as c

        return c
    except Exception as exc:  # pragma: no cover - env dependent
        pytest.skip(f"_common not importable here: {type(exc).__name__}: {exc}")


def _shim_active() -> bool:
    """True when the test-support import fallback (no real vllm) is active."""
    try:
        import vllm_omni  # noqa: F401

        return bool(getattr(vllm_omni, "_SHIM_ACTIVE", False))
    except Exception:
        return False


def test_common_ensure_list_matches_golden():
    c = _common_import()
    # Canonical default: list / tuple / None -> [] / scalar -> [x].
    assert c.ensure_list([1, 2]) == [1, 2]
    assert c.ensure_list((1, 2)) == [1, 2]
    assert c.ensure_list(None) == []
    assert c.ensure_list(_ConstantList([1, 2])) == [1, 2]
    assert c.ensure_list(5) == [5]
    # Tensor -> .tolist() preserves dims (ming_flash / step_audio2 semantics).
    assert c.ensure_list(torch.tensor([[1, 2], [3, 4]])) == [[1, 2], [3, 4]]
    # Legacy qwen3_omni: non-list returned unchanged (incl. None / tuple).
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
    # P2 deep-dive parity: per-item recursion normalizes a list of non-scalar
    # tensors and a tuple (previously ValueError / TypeError).
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
