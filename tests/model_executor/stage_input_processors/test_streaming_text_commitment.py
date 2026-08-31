# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass

import pytest

from vllm_omni.model_executor.stage_input_processors.streaming_text_commitment import (
    CommitmentState,
    StreamingTextCommitmentPolicy,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass(frozen=True)
class _Diagnostic:
    name: str
    packets: tuple[str, ...]
    pending_after_packet: tuple[str, ...]


# CPU-only ambiguity diagnostic set for the deterministic zh/en special-text
# profile in RFC #6496. These are original minimal strings rather than copied
# evaluation text; the referenced paper's diagnostic strings are not public.
_AMBIGUITY_DIAGNOSTICS = (
    _Diagnostic("integer_currency", ("金额为2026", "元"), ("2026", "2026元")),
    _Diagnostic("year", ("年份是2026", "年"), ("2026", "2026年")),
    _Diagnostic("multi_char_temperature", ("温度25", "摄氏", "度"), ("25", "25摄氏", "25摄氏度")),
    _Diagnostic("fahrenheit", ("温度80", "°", "F", "，记录"), ("80", "80°", "80°F", "")),
    _Diagnostic("percentage", ("完成率12", ".", "5", "%"), ("12", "12.", "12.5", "12.5%")),
    _Diagnostic("fullwidth_percentage", ("完成率１２", "．５", "％"), ("１２", "１２．５", "１２．５％")),
    _Diagnostic("version", ("版本v2", ".", "1", "发布"), ("v2", "v2.", "v2.1", "")),
    _Diagnostic(
        "ip_address",
        ("地址127.0", ".0", ".1", "，继续"),
        ("127.0", "127.0.0", "127.0.0.1", ""),
    ),
    _Diagnostic("clock_time", ("时间08", ":", "30", "开始"), ("08", "08:", "08:30", "")),
    _Diagnostic("score", ("比分78", ":96", "，结束"), ("78", "78:96", "")),
    _Diagnostic("phone_number", ("拨打123", "15", "咨询"), ("123", "12315", "")),
    _Diagnostic("dollar_decimal", ("价格$12", ".50", "，确认"), ("$12", "$12.50", "")),
    _Diagnostic("yuan_prefix", ("价格￥2026", "元"), ("￥2026", "￥2026元")),
    _Diagnostic("kilometres_zh", ("距离4", "公里"), ("4", "4公里")),
    _Diagnostic("decimal_kilometres_zh", ("距离0.", "5", "公里"), ("0.", "0.5", "0.5公里")),
    _Diagnostic("spaced_ascii_unit", ("重量25", " kg", "，记录"), ("25", "25 kg", "")),
    _Diagnostic("superscript_unit", ("面积100", " cm²", "，记录"), ("100", "100 cm²", "")),
    _Diagnostic("ordinal_third", ("排名第3", "rd", "，确认"), ("3", "3rd", "")),
    _Diagnostic("ordinal_first", ("名次1", "st", "，确认"), ("1", "1st", "")),
    _Diagnostic("mixed_alphanumeric_o2o", ("模式O2", "O", "服务"), ("O2", "O2O", "")),
    _Diagnostic("mixed_alphanumeric_b2b", ("采用B2", "B", "方案"), ("B2", "B2B", "")),
    _Diagnostic("ascii_abbreviation", ("调用AP", "I", "接口"), ("AP", "API", "")),
    _Diagnostic("numeric_range", ("比分1", "-2", "落后"), ("1", "1-2", "")),
    _Diagnostic("fraction", ("战成2", "/2", "平", "局"), ("2", "2/2", "2/2平", "")),
    _Diagnostic("multiplication", ("计算3", "×4", "得到"), ("3", "3×4", "")),
    _Diagnostic("email", ("邮箱a@b", ".com", "可用"), ("a@b", "a@b.com", "")),
    _Diagnostic("domain", ("访问vllm", ".ai", "网站"), ("vllm", "vllm.ai", "")),
    _Diagnostic("symbol_run", ("前文——", "后文"), ("——", "")),
    _Diagnostic("dotted_abbreviation", ("例如e.", "g.", " 中文"), ("e.", "e.g.", "")),
    _Diagnostic("fullwidth_integer_unit", ("长度４", "厘米"), ("４", "４厘米")),
)


@pytest.mark.parametrize("diagnostic", _AMBIGUITY_DIAGNOSTICS, ids=lambda case: case.name)
def test_ambiguity_diagnostics_never_release_an_open_atom(diagnostic: _Diagnostic) -> None:
    policy = StreamingTextCommitmentPolicy()
    committed = ""
    input_so_far = ""

    for packet, expected_pending in zip(diagnostic.packets, diagnostic.pending_after_packet, strict=True):
        input_so_far += packet
        update = policy.feed(packet)
        committed += update.committed_text
        assert update.pending_text == expected_pending
        assert committed + update.pending_text == input_so_far

    committed += policy.finish().committed_text
    assert committed == "".join(diagnostic.packets)
    assert policy.pending_text == ""
    assert policy.state is CommitmentState.FINISHED


@dataclass(frozen=True)
class _Trace:
    committed_text: str
    pending_text: str
    atoms: tuple[tuple[str, str], ...]
    strong_boundaries: tuple[int, ...]


def _trace(packets: tuple[str, ...], *, finish: bool) -> _Trace:
    policy = StreamingTextCommitmentPolicy()
    committed = ""
    atoms: list[tuple[str, str]] = []
    boundaries: list[int] = []

    def record(update) -> None:
        nonlocal committed
        for span in update.spans:
            committed += span.source_text
            if span.kind != "natural":
                atoms.append((span.kind, span.source_text))
            if span.boundary_after:
                boundaries.append(len(committed))

    for packet in packets:
        record(policy.feed(packet))
    if finish:
        record(policy.finish())
    return _Trace(committed, policy.pending_text, tuple(atoms), tuple(boundaries))


def _packetizations(text: str) -> tuple[tuple[str, ...], ...]:
    variants: set[tuple[str, ...]] = {(text,), tuple(text)}
    for first in range(1, len(text)):
        variants.add((text[:first], text[first:]))
        for second in range(first + 1, len(text)):
            variants.add((text[:first], text[first:second], text[second:]))
    return tuple(variants)


def _segments(trace: _Trace, source: str) -> tuple[str, ...]:
    start = 0
    segments = []
    for end in trace.strong_boundaries:
        segments.append(source[start:end])
        start = end
    if start < len(source):
        segments.append(source[start:])
    return tuple(segments)


@pytest.mark.parametrize(
    "text",
    (
        "年份2026年12月结束。",
        "费率25℃/h，继续。",
        "金额1元人民币到账。",
        "面积1米²，记录。",
        "功率3千瓦时，记录。",
        "The API returned 3rd-party data.",
        "例如e.g. 中文说明。",
    ),
)
def test_source_atoms_and_boundaries_are_invariant_to_packetization(text: str) -> None:
    online_baseline = _trace((text,), finish=False)
    final_baseline = _trace((text,), finish=True)

    for packets in _packetizations(text):
        assert _trace(packets, finish=False) == online_baseline
        assert _trace(packets, finish=True) == final_baseline


@pytest.mark.parametrize(
    ("packets", "expected_atom"),
    (
        (("年份2026年", "12月结束。"), ("special", "2026年12月")),
        (("费率25℃", "/h，继续。"), ("special", "25℃/h")),
        (("金额1元", "人民币到账。"), ("special", "1元人民币")),
        (("面积1米", "²，记录。"), ("special", "1米²")),
    ),
)
def test_unit_continuations_are_not_cut_at_transport_frontiers(
    packets: tuple[str, ...], expected_atom: tuple[str, str]
) -> None:
    trace = _trace(packets, finish=True)
    assert expected_atom in trace.atoms


@pytest.mark.parametrize(
    "packets",
    (
        ("按1", "\ufe0f\u20e3", "继续。"),
        ("按1\ufe0f", "\u20e3", "继续。"),
    ),
)
def test_digit_keycap_suffix_stays_with_the_number_across_every_seam(
    packets: tuple[str, ...],
) -> None:
    keycap = "1\ufe0f\u20e3"

    policy = StreamingTextCommitmentPolicy()
    first_update = policy.feed(packets[0])
    assert first_update.pending_text in {"1", "1\ufe0f"}
    assert policy.feed(packets[1]).pending_text == keycap
    update = policy.feed(packets[2])
    assert [(span.kind, span.source_text) for span in update.spans] == [
        ("special", keycap),
        ("natural", "继续"),
    ]
    assert update.pending_text == "。"
    assert _trace(packets, finish=True) == _trace(("按" + keycap + "继续。",), finish=True)


@pytest.mark.parametrize(
    ("packets", "abbreviation"),
    (
        (("例如e.", "g.", " 中文说明。"), "e.g."),
        (("缩写U.", "S.", " 中文说明。"), "U.S."),
    ),
)
def test_dotted_abbreviation_keeps_its_terminal_dot_without_a_sentence_boundary(
    packets: tuple[str, ...], abbreviation: str
) -> None:
    trace = _trace(packets, finish=True)
    full_text = "".join(packets)

    assert ("lexical", abbreviation) in trace.atoms
    assert trace.strong_boundaries == (len(full_text),)
    assert _trace(packets, finish=True) == _trace((full_text,), finish=True)


def test_raw_span_kinds_do_not_send_ordinary_words_to_special_normalization() -> None:
    trace = _trace(("The API costs 25元。",), finish=True)

    assert trace.atoms == (
        ("lexical", "The"),
        ("lexical", "API"),
        ("lexical", "costs"),
        ("special", "25元"),
    )
    assert trace.strong_boundaries == (len("The API costs 25元。"),)


def test_unit_prefix_backtracks_to_natural_text_after_a_mismatch() -> None:
    trace = _trace(("战成2/2", "平", "局。"), finish=True)

    assert trace.atoms == (("special", "2/2"),)
    assert trace.committed_text == "战成2/2平局。"


@pytest.mark.parametrize("line_break", ("\n", "\r\n", " \n ", "\t\n\t"))
def test_numeric_unit_lookahead_stops_at_newline_for_every_packetization(line_break: str) -> None:
    text = f"There are 3{line_break}More things."
    boundary = text.index("\n") + 1
    baseline = _trace((text,), finish=True)

    assert ("special", "3") in baseline.atoms
    assert all("\n" not in atom_text for _, atom_text in baseline.atoms)
    assert baseline.strong_boundaries == (boundary, len(text))
    assert _segments(baseline, text) == (text[:boundary], text[boundary:])
    for packets in _packetizations(text):
        assert _trace(packets, finish=True) == baseline


def test_numeric_unit_lookahead_still_accepts_non_newline_whitespace() -> None:
    trace = _trace(("Weight is 25", "\t kg, recorded."), finish=True)

    assert ("special", "25\t kg") in trace.atoms
    assert trace.strong_boundaries == (len("Weight is 25\t kg, recorded."),)


def test_only_confirmed_strong_sentence_ends_set_boundary_metadata() -> None:
    policy = StreamingTextCommitmentPolicy()
    update = policy.feed("值12.5，继续。下一句!")
    spans = [(span.source_text, span.boundary_after) for span in update.spans]

    assert ("，继续。", True) in spans
    assert update.pending_text == "!"
    assert ("下一句", False) in spans
    assert all(not boundary for text, boundary in spans if text == "12.5")
    final = policy.finish()
    assert [(span.source_text, span.boundary_after) for span in final.spans] == [("!", True)]


@pytest.mark.parametrize("decimal", (".5 seconds.", "．５秒。"))
def test_leading_decimal_is_one_special_atom_for_every_packetization(decimal: str) -> None:
    baseline = _trace((decimal,), finish=True)

    assert baseline.atoms[0][0] == "special"
    assert baseline.atoms[0][1].startswith(decimal[:2])
    assert baseline.strong_boundaries == (len(decimal),)
    assert _segments(baseline, decimal) == (decimal,)
    for packets in _packetizations(decimal):
        assert _trace(packets, finish=True) == baseline


@pytest.mark.parametrize("terminator", ("!", "?", "。", "！", "？", "…", "\n", "\r\n"))
def test_leading_decimal_after_non_dot_terminator_is_atomic_for_every_packetization(terminator: str) -> None:
    prefix = f"Value:{terminator}"
    decimal = ".5 seconds."
    text = prefix + decimal
    baseline = _trace((text,), finish=True)

    assert ("special", ".5 seconds") in baseline.atoms
    assert baseline.strong_boundaries == (len(prefix), len(text))
    assert _segments(baseline, text) == (prefix, decimal)
    for packets in _packetizations(text):
        assert _trace(packets, finish=True) == baseline


def test_decimal_point_at_packet_frontier_stays_pending() -> None:
    policy = StreamingTextCommitmentPolicy()

    first = policy.feed("Value is .")
    assert first.committed_text == "Value is "
    assert first.pending_text == "."

    second = policy.feed("5 seconds.")
    assert all(span.source_text != "." for span in second.spans)
    assert second.pending_text == ".5 seconds."
    final = policy.finish()
    assert second.committed_text + final.committed_text == ".5 seconds."


def test_maximal_terminator_run_takes_precedence_over_leading_decimal() -> None:
    text = "Wait...5 seconds."
    baseline = _trace((text,), finish=True)

    assert baseline.strong_boundaries == (len("Wait..."), len(text))
    assert _segments(baseline, text) == ("Wait...", "5 seconds.")
    assert all(atom[1] != ".5" for atom in baseline.atoms)
    for packets in _packetizations(text):
        assert _trace(packets, finish=True) == baseline


def test_fullwidth_leading_decimal_after_terminator_remains_special() -> None:
    text = "结束。．５秒。"
    baseline = _trace((text,), finish=True)

    assert ("special", "．５秒") in baseline.atoms
    assert baseline.strong_boundaries == (len("结束。"), len(text))
    for packets in _packetizations(text):
        assert _trace(packets, finish=True) == baseline


@pytest.mark.parametrize("text", ("Wait...", "What?!", "Really!!!"))
def test_consecutive_terminators_form_one_boundary_for_every_packetization(text: str) -> None:
    baseline = _trace((text,), finish=True)

    assert baseline.strong_boundaries == (len(text),)
    assert _segments(baseline, text) == (text,)
    for packets in _packetizations(text):
        trace = _trace(packets, finish=True)
        assert trace == baseline
        assert _segments(trace, text) == (text,)


@pytest.mark.parametrize(
    ("packets", "expected_first_segment"),
    (
        (("Wait.", ".. Next."), "Wait..."),
        (("What?", "! Next."), "What?!"),
    ),
)
def test_terminator_run_is_coalesced_across_packet_seam(packets: tuple[str, str], expected_first_segment: str) -> None:
    source = "".join(packets)
    trace = _trace(packets, finish=True)
    segments = _segments(trace, source)

    assert trace.committed_text == source
    assert segments[0] == expected_first_segment
    assert all(not segment or not all(ch in ".!?。！？…\n" for ch in segment.strip()) for segment in segments)


def test_feed_final_and_finish_produce_the_same_raw_atoms() -> None:
    text = "温度25摄氏度，调用API."
    direct = StreamingTextCommitmentPolicy().feed(text, final=True)
    streamed = _trace(("温度25", "摄氏度，调用", "API."), finish=True)

    direct_atoms = tuple((span.kind, span.source_text) for span in direct.spans if span.kind != "natural")
    assert direct.committed_text == streamed.committed_text == text
    assert direct_atoms == streamed.atoms
    assert direct.final


def test_end_of_input_flushes_an_unresolved_number() -> None:
    policy = StreamingTextCommitmentPolicy()

    update = policy.feed("数量为3")
    assert update.committed_text == "数量为"
    assert update.pending_text == "3"
    assert policy.finish().committed_text == "3"
    assert policy.state is CommitmentState.FINISHED


def test_pending_limit_failure_is_terminal_and_transactional() -> None:
    policy = StreamingTextCommitmentPolicy(max_pending_chars=3)
    policy.feed("12")

    with pytest.raises(ValueError, match="max_pending_chars=3"):
        policy.feed("34")

    assert policy.pending_text == "12"
    assert policy.state is CommitmentState.FAILED
    with pytest.raises(RuntimeError, match="failed"):
        policy.feed("元")
    with pytest.raises(RuntimeError, match="failed"):
        policy.finish()


def test_finished_policy_rejects_late_transport_chunks_and_second_finish() -> None:
    policy = StreamingTextCommitmentPolicy()
    policy.feed("完成。", final=True)

    with pytest.raises(RuntimeError, match="finished"):
        policy.feed("late")
    with pytest.raises(RuntimeError, match="finished"):
        policy.finish()
    assert not hasattr(policy, "reset")


@pytest.mark.parametrize("bad_value", (None, b"text", 1))
def test_non_string_transport_chunks_are_rejected_without_failing_state(bad_value: object) -> None:
    policy = StreamingTextCommitmentPolicy()

    with pytest.raises(TypeError, match="must be strings"):
        policy.feed(bad_value)  # type: ignore[arg-type]
    assert policy.state is CommitmentState.OPEN
    assert policy.pending_text == ""


def test_profile_is_explicit_and_unknown_profiles_are_rejected() -> None:
    policy = StreamingTextCommitmentPolicy(profile="zh_en_special_v1")
    assert policy.profile == "zh_en_special_v1"

    with pytest.raises(ValueError, match="unsupported.*profile"):
        StreamingTextCommitmentPolicy(profile="all_languages")
