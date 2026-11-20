# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from vllm_omni.benchmarks.data_modules.omniinteract_dataset import (
    OmniInteractCase,
    OmniInteractEvaluationOptions,
)
from vllm_omni.benchmarks.omniinteract import OmniInteractCaseResult
from vllm_omni.benchmarks.omniinteract_eval import (
    AlignedWord,
    TranscriptChunk,
    _summarize,
    build_slots,
    evaluate_batch,
    evaluate_case,
    evaluation_inputs_fingerprint,
    match_slots,
)
from vllm_omni.benchmarks.omniinteract_judge import (
    CoreJudgment,
    EarlyJudgment,
    OmniInteractJudge,
    PartialJudgment,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]


class _FixedJudge:
    def judge_early(
        self,
        slot: dict[str, object],
        full_context: str,
        actual_text: str,
    ) -> EarlyJudgment:
        del slot, full_context, actual_text
        return EarlyJudgment("neutral", 1.0, "ok", "{}", "llm_json")

    def judge_core(
        self,
        slot: dict[str, object],
        full_context: str,
        actual_text: str,
        future_answers: str,
    ) -> CoreJudgment:
        del slot, full_context, actual_text, future_answers
        return CoreJudgment(1.0, "answer", False, "ok", "{}", "llm_json")

    def judge_interrupted_partial(
        self,
        slot: dict[str, object],
        actual_text: str,
    ) -> PartialJudgment:
        del slot, actual_text
        return PartialJudgment(1.0, False, "ok", "{}", "llm_json")


class _GeneratedJudge(OmniInteractJudge):
    def __init__(self, responses: list[str]) -> None:
        super().__init__("http://judge", "judge-model")
        self.responses = iter(responses)

    def _generate(self, system_prompt: str, user_prompt: str) -> str:
        del system_prompt, user_prompt
        return next(self.responses)


def test_build_slots_and_split_word_aligned_chunk() -> None:
    slots = build_slots(
        [
            {
                "question_time": 0.0,
                "answer_time": 2.0,
                "question_text": "question",
                "answer_text": "answer",
                "question_type": "realtime",
            },
            {
                "question_time": 5.0,
                "answer_time": 6.0,
                "question_text": "next",
                "answer_text": "next answer",
                "question_type": "proactive",
            },
        ],
        "multi_turn",
    )
    chunk = TranscriptChunk(
        source_id=0,
        text="okay answer",
        start=0.5,
        end=2.5,
        aligned_words=(
            AlignedWord("okay", 0.5, 0.9),
            AlignedWord("answer", 2.1, 2.5),
        ),
    )

    matched, unmatched = match_slots(slots, [chunk])

    assert not unmatched
    assert matched[0].early_chunks[0].text == "okay"
    assert matched[0].core_chunks[0].text == "answer"
    assert matched[0].core_chunks[0].start == pytest.approx(2.1)


def test_judge_parses_json_wrapped_in_model_text() -> None:
    judge = _GeneratedJudge(
        [
            'result: {"flag":"FP_Hallucination","score":1,"rationale":"early"}',
            '```json\n{"score":0.8,"trigger_phrase":"answer","spoiler":false,"rationale":"good"}\n```',
            '{"score":0.7,"hallucination":false,"rationale":"partial"}',
        ]
    )
    slot = {"question_text": "q", "gt_answer": "answer"}

    early = judge.judge_early(slot, "context", "guess")
    core = judge.judge_core(slot, "context", "answer", "(none)")
    partial = judge.judge_interrupted_partial(slot, "answer")

    assert early.category == "hallucination"
    assert early.score == 0.0
    assert core.score == pytest.approx(0.8)
    assert core.trigger_phrase == "answer"
    assert partial.score == pytest.approx(0.7)


def test_evaluate_case_persists_accuracy_artifact(tmp_path: Path) -> None:
    annotation = tmp_path / "annotation.json"
    annotation.write_text(
        json.dumps(
            [
                {
                    "question_time": 0.0,
                    "answer_time": 1.0,
                    "question_text": "question",
                    "answer_text": "answer",
                    "question_type": "realtime",
                }
            ]
        )
    )
    output_dir = tmp_path / "case-output"
    output_dir.mkdir()
    (output_dir / "wav_transcript.json").write_text(
        json.dumps(
            {
                "chunks": [
                    {
                        "timestamp": [1.0, 1.5],
                        "text": "answer",
                        "aligned_words": [{"text": "answer", "start": 1.0, "end": 1.5}],
                    }
                ]
            }
        )
    )
    video = tmp_path / "video.mp4"
    video.touch()
    case = OmniInteractCase("1q1a", "video.mp4", video, annotation, "multi_turn")
    result = OmniInteractCaseResult(
        subset="1q1a",
        video=str(video),
        output_dir=str(output_dir),
        success=True,
        eligible_for_official_eval=True,
    )
    artifact = tmp_path / "evaluation.json"

    evaluation = evaluate_case(case, result, _FixedJudge(), artifact)

    assert evaluation["timing_precision"] == "word-aligned"
    assert evaluation["summary"]["IA_QTF1"] == pytest.approx(1.0)
    paper = evaluation["summary"]["paper_metrics"]["exp_f1"]
    assert paper["all_global"]["IA_QTF1"] == pytest.approx(1.0)
    assert paper["realtime"]["IA_QTF1"] == pytest.approx(1.0)
    assert json.loads(artifact.read_text())["status"] == "ok"
    fingerprint = evaluation["inputs_fingerprint"]
    assert fingerprint["annotation_sha256"]
    assert fingerprint["transcript_sha256"]
    assert fingerprint["protocol_source"]


def _realtime_eval_case(tmp_path: Path, *, transcript_text: str) -> tuple[OmniInteractCase, OmniInteractCaseResult]:
    annotation = tmp_path / "annotation.json"
    annotation.write_text(
        json.dumps(
            [
                {
                    "question_time": 0.0,
                    "answer_time": 1.0,
                    "question_text": "question",
                    "answer_text": "answer",
                    "question_type": "realtime",
                }
            ]
        )
    )
    output_dir = tmp_path / "case-output"
    output_dir.mkdir(exist_ok=True)
    chunks: list[dict[str, object]] = []
    if transcript_text:
        chunks.append({"timestamp": [1.0, 1.5], "text": transcript_text})
    (output_dir / "wav_transcript.json").write_text(json.dumps({"chunks": chunks}))
    video = tmp_path / "video.mp4"
    video.touch()
    case = OmniInteractCase("1q1a", "video.mp4", video, annotation, "multi_turn")
    result = OmniInteractCaseResult(
        subset="1q1a",
        video=str(video),
        output_dir=str(output_dir),
        success=True,
        eligible_for_official_eval=True,
    )
    return case, result


def _eval_options(tmp_path: Path, *, skip_existing: bool) -> OmniInteractEvaluationOptions:
    return OmniInteractEvaluationOptions(
        judge_base_url="http://127.0.0.1:9",
        judge_model="cached-judge",
        judge_api_key="EMPTY",
        judge_timeout_s=60.0,
        judge_max_tokens=32,
        workers=1,
        output_dir=tmp_path / "evaluation",
        skip_existing=skip_existing,
    )


class _ConfiguredFixedJudge(_FixedJudge):
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "EMPTY",
        timeout_s: float = 60.0,
        max_tokens: int = 512,
    ) -> None:
        del api_key, timeout_s
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens


def test_skip_existing_reuses_matching_fingerprint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vllm_omni.benchmarks.omniinteract_eval as eval_mod

    case, result = _realtime_eval_case(tmp_path, transcript_text="answer")
    options = _eval_options(tmp_path, skip_existing=False)
    monkeypatch.setattr(eval_mod, "OmniInteractJudge", _ConfiguredFixedJudge)
    first = evaluate_batch([case], [result], options)
    assert first["summary"]["IA_QTF1"] == pytest.approx(1.0)

    calls: list[str] = []

    def _forbidden(
        case: OmniInteractCase,
        result: OmniInteractCaseResult,
        judge: object,
        output_path: Path,
        config: object = None,
        inputs_fingerprint: object = None,
    ) -> dict[str, object]:
        del case, result, judge, output_path, config, inputs_fingerprint
        calls.append("evaluate_case")
        raise AssertionError("matching fingerprint must reuse the cached evaluation")

    monkeypatch.setattr(eval_mod, "evaluate_case", _forbidden)
    reused = evaluate_batch([case], [result], _eval_options(tmp_path, skip_existing=True))
    assert calls == []
    assert reused["summary"]["IA_QTF1"] == pytest.approx(1.0)


def test_skip_existing_rejects_stale_transcript(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vllm_omni.benchmarks.omniinteract_eval as eval_mod

    case, result = _realtime_eval_case(tmp_path, transcript_text="answer")
    options = _eval_options(tmp_path, skip_existing=True)
    fingerprint = evaluation_inputs_fingerprint(
        annotation_path=case.annotation_path,
        transcript_path=Path(result.output_dir) / "wav_transcript.json",
        judge_model=options.judge_model,
        judge_base_url=options.judge_base_url,
        judge_max_tokens=options.judge_max_tokens,
    )
    sample_id = f"{case.subset}__{Path(result.output_dir).name}"
    cache_path = options.output_dir / f"{sample_id}.unified_eval.json"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "sample_id": sample_id,
                "inputs_fingerprint": asdict(fingerprint),
                "summary": {
                    "IA_QTF1": 1.0,
                    "num_slots": 1,
                    "num_unmatched_chunks": 0,
                    "Global_TP": 1,
                    "Global_FP": 0,
                    "Global_FN": 0,
                },
                "slots": [],
            }
        )
    )
    (Path(result.output_dir) / "wav_transcript.json").write_text(json.dumps({"chunks": []}))
    monkeypatch.setattr(eval_mod, "OmniInteractJudge", _ConfiguredFixedJudge)
    fresh = evaluate_batch([case], [result], options)
    assert fresh["summary"]["IA_QTF1"] == pytest.approx(0.0)


class _MissingCoreScoreJudge(_ConfiguredFixedJudge):
    def judge_core(
        self,
        slot: dict[str, object],
        full_context: str,
        actual_text: str,
        future_answers: str,
    ) -> CoreJudgment:
        del slot, full_context, actual_text, future_answers
        return CoreJudgment(0.0, "", False, "Could not evaluate", "{}", "llm_parse_failed")


def test_evaluate_batch_reports_missing_score_as_parse_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vllm_omni.benchmarks.omniinteract_eval as eval_mod

    case, result = _realtime_eval_case(tmp_path, transcript_text="answer")
    options = _eval_options(tmp_path, skip_existing=False)
    monkeypatch.setattr(eval_mod, "OmniInteractJudge", _MissingCoreScoreJudge)
    evaluation = evaluate_batch([case], [result], options)
    assert evaluation["status"] == "failed"
    assert evaluation["evaluated"] == 0
    assert evaluation["failed"] == 1
    sample_id = f"{case.subset}__{Path(result.output_dir).name}"
    cached = json.loads((options.output_dir / f"{sample_id}.unified_eval.json").read_text())
    assert cached["status"] == "parse_failed"


class _CoreRationaleNumberJudge(OmniInteractJudge):
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "EMPTY",
        timeout_s: float = 60.0,
        max_tokens: int = 512,
    ) -> None:
        super().__init__(base_url, model, api_key=api_key, timeout_s=timeout_s, max_tokens=max_tokens)

    def _generate(self, system_prompt: str, user_prompt: str) -> str:
        del system_prompt, user_prompt
        return '{"rationale":"Cannot score: 1 required reference is missing."}'


def test_evaluate_batch_rejects_core_rationale_number_as_parse_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vllm_omni.benchmarks.omniinteract_eval as eval_mod

    case, result = _realtime_eval_case(tmp_path, transcript_text="answer")
    options = _eval_options(tmp_path, skip_existing=False)
    monkeypatch.setattr(eval_mod, "OmniInteractJudge", _CoreRationaleNumberJudge)
    evaluation = evaluate_batch([case], [result], options)
    assert evaluation["status"] == "failed"
    assert evaluation["evaluated"] == 0
    assert evaluation["failed"] == 1
    sample_id = f"{case.subset}__{Path(result.output_dir).name}"
    cached = json.loads((options.output_dir / f"{sample_id}.unified_eval.json").read_text())
    assert cached["status"] == "parse_failed"
    assert cached["slots"][0]["stage_core"]["parse_source"] == "llm_parse_failed"
    assert cached["slots"][0]["stage_core"]["S_core"] == 0.0


def test_skip_existing_rejects_parse_failed_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vllm_omni.benchmarks.omniinteract_eval as eval_mod

    case, result = _realtime_eval_case(tmp_path, transcript_text="answer")
    options = _eval_options(tmp_path, skip_existing=True)
    fingerprint = evaluation_inputs_fingerprint(
        annotation_path=case.annotation_path,
        transcript_path=Path(result.output_dir) / "wav_transcript.json",
        judge_model=options.judge_model,
        judge_base_url=options.judge_base_url,
        judge_max_tokens=options.judge_max_tokens,
    )
    sample_id = f"{case.subset}__{Path(result.output_dir).name}"
    cache_path = options.output_dir / f"{sample_id}.unified_eval.json"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(
        json.dumps(
            {
                "status": "parse_failed",
                "sample_id": sample_id,
                "inputs_fingerprint": asdict(fingerprint),
                "summary": {
                    "IA_QTF1": 0.0,
                    "num_slots": 1,
                    "num_unmatched_chunks": 0,
                    "Global_TP": 0,
                    "Global_FP": 0,
                    "Global_FN": 1,
                },
                "slots": [
                    {
                        "stage_core": {"parse_source": "llm_parse_failed", "S_core": 0.0},
                        "TP_n": 0.0,
                        "FP_delta": 0,
                        "FN_delta": 1,
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(eval_mod, "OmniInteractJudge", _ConfiguredFixedJudge)
    fresh = evaluate_batch([case], [result], options)
    assert fresh["evaluated"] == 1
    assert fresh["failed"] == 0
    assert fresh["summary"]["IA_QTF1"] == pytest.approx(1.0)
    assert json.loads(cache_path.read_text())["status"] == "ok"


def _slot_row(
    *,
    scene_type: str,
    question_type: str,
    tp_n: float,
    fp: int = 0,
    fn: int = 0,
    nested_role: str | None = None,
    nested_group_id: int | None = None,
    sample_id: str = "s",
    tp_core: float | None = None,
    answer_start: float | None = 1.0,
) -> dict[str, object]:
    core_tp = tp_n if tp_core is None else tp_core
    return {
        "sample_id": sample_id,
        "scene_type": scene_type,
        "question_type": question_type,
        "nested_role": nested_role,
        "nested_group_id": nested_group_id,
        "TP_n": tp_n,
        "TP_core": core_tp,
        "FP_delta": fp,
        "FN_delta": fn,
        "is_interrupted": False,
        "stage_core": {"answer_start": answer_start},
    }


def test_paper_metrics_exclude_nested_from_realtime_and_proactive() -> None:
    summary = _summarize(
        [
            _slot_row(scene_type="multi_turn", question_type="realtime", tp_n=1.0),
            _slot_row(
                scene_type="nested",
                question_type="realtime",
                tp_n=1.0,
                nested_role="inner",
                nested_group_id=1,
                answer_start=0.5,
            ),
            _slot_row(
                scene_type="nested",
                question_type="proactive",
                tp_n=1.0,
                nested_role="outer",
                nested_group_id=1,
                answer_start=1.5,
            ),
            _slot_row(scene_type="multi_turn", question_type="proactive", tp_n=1.0),
            _slot_row(scene_type="1QnA", question_type="step", tp_n=1.0, fn=0),
        ],
        unmatched=0,
    )
    paper = summary["paper_metrics"]["exp_f1"]

    assert paper["realtime"]["num_slots"] == 1
    assert paper["proactive"]["num_slots"] == 1
    assert paper["nested"]["num_slots"] == 2
    assert paper["one_q1a_global"]["num_slots"] == 4
    assert paper["one_qna"]["num_slots"] == 1
    assert paper["all_global"]["num_slots"] == 5
    # All Global includes 1QnA; 1Q1A Global does not.
    assert paper["all_global"]["Global_TP"] == pytest.approx(5.0)
    assert paper["one_q1a_global"]["Global_TP"] == pytest.approx(4.0)
    assert summary["nested"]["NCCS"] == pytest.approx(1.0)
    assert summary["nested"]["missing_q1_count"] == 0
    assert summary["paper_metrics"]["exp_nested"]["missed_outer"] == 0
    assert summary["paper_metrics"]["exp_nested"]["inner_IA_QTF1"] == pytest.approx(1.0)
    assert summary["paper_metrics"]["exp_nested"]["outer_IA_QTF1"] == pytest.approx(1.0)
    assert summary["scenario_case_counts"]["realtime"] == 1
    assert summary["scenario_case_counts"]["proactive"] == 1
    assert summary["scenario_case_counts"]["nested"] == 1
    assert summary["scenario_case_counts"]["1qna"] == 1


def test_nccs_allows_missing_answer_start() -> None:
    summary = _summarize(
        [
            _slot_row(
                scene_type="nested",
                question_type="realtime",
                tp_n=0.8,
                tp_core=0.8,
                nested_role="inner",
                nested_group_id=1,
                answer_start=None,
            ),
            _slot_row(
                scene_type="nested",
                question_type="proactive",
                tp_n=0.8,
                tp_core=0.8,
                nested_role="outer",
                nested_group_id=1,
                answer_start=None,
            ),
        ],
        unmatched=0,
    )
    assert summary["nested"]["success_pairs"] == 1
    assert summary["nested"]["NCCS"] == pytest.approx(0.8)


def test_missed_outer_counts_zero_outer_core() -> None:
    summary = _summarize(
        [
            _slot_row(
                scene_type="nested",
                question_type="realtime",
                tp_n=1.0,
                tp_core=1.0,
                nested_role="inner",
                nested_group_id=1,
                answer_start=0.5,
            ),
            _slot_row(
                scene_type="nested",
                question_type="proactive",
                tp_n=0.0,
                tp_core=0.0,
                fn=1,
                nested_role="outer",
                nested_group_id=1,
                answer_start=1.5,
            ),
        ],
        unmatched=0,
    )
    nested = summary["paper_metrics"]["exp_nested"]
    assert nested["missed_outer"] == 1
    assert nested["NCCS"] == pytest.approx(0.0)
    assert nested["inner_IA_QTF1"] == pytest.approx(1.0)
    assert nested["outer_IA_QTF1"] == pytest.approx(0.0)


def test_cross_boundary_core_sets_effective_start_hint() -> None:
    slots = build_slots(
        [
            {
                "question_time": 0.0,
                "answer_time": 2.0,
                "question_text": "question",
                "answer_text": "answer",
                "question_type": "realtime",
            }
        ],
        "multi_turn",
    )
    chunk = TranscriptChunk(
        source_id=0,
        text="okay answer",
        start=0.5,
        end=2.5,
        aligned_words=(
            AlignedWord("okay", 0.5, 0.9),
            AlignedWord("answer", 2.1, 2.5),
        ),
    )
    matched, unmatched = match_slots(slots, [chunk])
    assert not unmatched
    assert matched[0].core_chunks[0].effective_start_hint == pytest.approx(2.0)


class _NonNeutralEarlyJudge(_FixedJudge):
    def judge_early(
        self,
        slot: dict[str, object],
        full_context: str,
        actual_text: str,
    ) -> EarlyJudgment:
        del slot, full_context, actual_text
        return EarlyJudgment("other", 1.0, "ignored", "{}", "llm_json")


def test_early_tp_ack_only_for_neutral(tmp_path: Path) -> None:
    annotation = tmp_path / "annotation.json"
    annotation.write_text(
        json.dumps(
            [
                {
                    "question_time": 0.0,
                    "answer_time": 2.0,
                    "question_text": "question",
                    "answer_text": "answer",
                    "question_type": "realtime",
                }
            ]
        )
    )
    output_dir = tmp_path / "case-output"
    output_dir.mkdir()
    (output_dir / "wav_transcript.json").write_text(
        json.dumps(
            {
                "chunks": [
                    {
                        "timestamp": [0.5, 0.9],
                        "text": "okay",
                        "aligned_words": [{"text": "okay", "start": 0.5, "end": 0.9}],
                    },
                    {
                        "timestamp": [2.1, 2.5],
                        "text": "answer",
                        "aligned_words": [{"text": "answer", "start": 2.1, "end": 2.5}],
                    },
                ]
            }
        )
    )
    video = tmp_path / "video.mp4"
    video.touch()
    case = OmniInteractCase("1q1a", "video.mp4", video, annotation, "multi_turn")
    result = OmniInteractCaseResult(
        subset="1q1a",
        video=str(video),
        output_dir=str(output_dir),
        success=True,
        eligible_for_official_eval=True,
    )
    evaluation = evaluate_case(case, result, _NonNeutralEarlyJudge(), tmp_path / "evaluation.json")
    assert evaluation["slots"][0]["TP_ack"] == 0.0
    assert evaluation["slots"][0]["stage_early"]["category"] == "other"


def test_paper_metrics_unknown_inner_does_not_underflow_realtime() -> None:
    summary = _summarize(
        [
            _slot_row(scene_type="multi_turn", question_type="realtime", tp_n=0.5),
            _slot_row(
                scene_type="nested",
                question_type="unknown",
                tp_n=2.0,
                nested_role="inner",
                nested_group_id=1,
                answer_start=0.5,
            ),
            _slot_row(
                scene_type="nested",
                question_type="proactive",
                tp_n=1.0,
                nested_role="outer",
                nested_group_id=1,
                answer_start=1.5,
            ),
        ],
        unmatched=0,
    )
    paper = summary["paper_metrics"]["exp_f1"]
    assert paper["realtime"]["Global_TP"] == pytest.approx(0.5)
    assert paper["realtime"]["num_slots"] == 1
    assert paper["realtime"]["Precision"] >= 0.0
    assert paper["one_q1a_global"]["Global_TP"] == pytest.approx(3.5)


def test_core_score_float_ignores_leading_counts() -> None:
    from vllm_omni.benchmarks.omniinteract_judge import _score_float_from_text

    assert _score_float_from_text("The answer covers 3 of the 5 key points, score: 0.6") == pytest.approx(0.6)
    assert _score_float_from_text("score: 3") is None
    assert _score_float_from_text("no score here 0.9") is None

    judge = _GeneratedJudge(["The answer covers 3 of the 5 key points, score: 0.6"])
    core = judge.judge_core({"question_text": "q", "gt_answer": "a"}, "ctx", "answer", "(none)")
    assert core.score == pytest.approx(0.6)
    assert core.parse_source == "llm_float_text"


def test_core_missing_json_score_does_not_use_rationale_number() -> None:
    raw = '{"rationale":"Cannot score: 1 required reference is missing."}'
    core = _GeneratedJudge([raw]).judge_core({"question_text": "q", "gt_answer": "a"}, "ctx", "answer", "(none)")
    assert core.score == 0.0
    assert core.parse_source == "llm_parse_failed"


def test_early_missing_score_and_unparsed_flag() -> None:
    missing_score = _GeneratedJudge(['{"flag":"Neutral","rationale":"ok"}'])
    early = missing_score.judge_early({"question_text": "q", "gt_answer": "a"}, "ctx", "hi")
    assert early.category == "unparsed"
    assert early.score == 0.0
    assert early.parse_source == "llm_parse_failed"

    garbage = _GeneratedJudge(["not json and not a known flag"])
    unparsed = garbage.judge_early({"question_text": "q", "gt_answer": "a"}, "ctx", "hi")
    assert unparsed.category == "unparsed"
    assert unparsed.score == 0.0
    assert unparsed.parse_source == "llm_parse_failed"


@pytest.mark.parametrize(
    ("path", "response"),
    [
        ("early", '{"flag":"Neutral","rationale":"Could not evaluate this answer"}'),
        ("early", '{"flag":"Neutral","score":"invalid","rationale":"bad"}'),
        ("core", '{"rationale":"Could not evaluate this answer"}'),
        ("core", '{"rationale":"Cannot score: 1 required reference is missing."}'),
        ("core", '{"score":"invalid","trigger_phrase":"","spoiler":false,"rationale":"bad"}'),
        ("partial", '{"rationale":"Could not evaluate this answer"}'),
        ("partial", '{"score":"invalid","hallucination":false,"rationale":"bad"}'),
    ],
)
def test_missing_or_invalid_json_scores_are_parse_failures(path: str, response: str) -> None:
    slot = {"question_text": "q", "gt_answer": "a"}
    judge = _GeneratedJudge([response])
    if path == "early":
        judgment = judge.judge_early(slot, "ctx", "hi")
    elif path == "core":
        judgment = judge.judge_core(slot, "ctx", "answer", "(none)")
    else:
        judgment = judge.judge_interrupted_partial(slot, "answer")
    assert judgment.score == 0.0
    assert judgment.parse_source == "llm_parse_failed"


def test_explicit_zero_json_score_is_valid_llm_json() -> None:
    slot = {"question_text": "q", "gt_answer": "a"}
    early = _GeneratedJudge(['{"flag":"Neutral","score":0,"rationale":"zero"}']).judge_early(slot, "ctx", "hi")
    core = _GeneratedJudge(['{"score":0.0,"trigger_phrase":"","spoiler":false,"rationale":"zero"}']).judge_core(
        slot, "ctx", "answer", "(none)"
    )
    partial = _GeneratedJudge(['{"score":0,"hallucination":false,"rationale":"zero"}']).judge_interrupted_partial(
        slot, "answer"
    )
    assert early.category == "neutral"
    assert early.score == 0.0
    assert early.parse_source == "llm_json"
    assert core.score == 0.0
    assert core.parse_source == "llm_json"
    assert partial.score == 0.0
    assert partial.parse_source == "llm_json"


@pytest.mark.parametrize("score_literal", ['"NaN"', '"Infinity"', '"-Infinity"', "NaN", "Infinity", "-Infinity"])
def test_nonfinite_json_scores_are_parse_failures(score_literal: str) -> None:
    slot = {"question_text": "q", "gt_answer": "a"}
    early = _GeneratedJudge([f'{{"flag":"Neutral","score":{score_literal},"rationale":"bad"}}']).judge_early(
        slot, "ctx", "hi"
    )
    core = _GeneratedJudge(
        [f'{{"score":{score_literal},"trigger_phrase":"answer","spoiler":false,"rationale":"bad"}}']
    ).judge_core(slot, "ctx", "answer", "(none)")
    partial = _GeneratedJudge(
        [f'{{"score":{score_literal},"hallucination":false,"rationale":"bad"}}']
    ).judge_interrupted_partial(slot, "answer")

    assert early.score == 0.0
    assert early.parse_source == "llm_parse_failed"
    assert core.score == 0.0
    assert core.parse_source == "llm_parse_failed"
    assert core.trigger_phrase == ""
    assert partial.score == 0.0
    assert partial.parse_source == "llm_parse_failed"


def test_fingerprint_requires_evaluator_schema_version() -> None:
    from vllm_omni.benchmarks.omniinteract_eval import EvaluationInputsFingerprint

    payload = {
        "annotation_sha256": "a" * 64,
        "transcript_sha256": "b" * 64,
        "judge_model": "m",
        "judge_base_url": "http://127.0.0.1:9",
        "judge_max_tokens": 32,
        "protocol_source": "proto",
    }
    assert EvaluationInputsFingerprint.from_mapping(payload) is None
    payload["evaluator_schema_version"] = 1
    assert EvaluationInputsFingerprint.from_mapping(payload) is not None
    payload["evaluator_schema_version"] = 999
    matched = EvaluationInputsFingerprint.from_mapping(payload)
    assert matched is not None
    assert matched.evaluator_schema_version == 999


def test_summarize_counts_parse_sources() -> None:
    rows = [
        {
            **_slot_row(scene_type="multi_turn", question_type="realtime", tp_n=1.0),
            "stage_early": {"parse_source": "llm_json"},
            "stage_core": {"parse_source": "llm_parse_failed", "trigger_fallback": True},
        }
    ]
    summary = _summarize(rows, unmatched=0)
    assert summary["judge_parse"]["llm_json"] == 1
    assert summary["judge_parse"]["llm_parse_failed"] == 1
    assert summary["judge_parse"]["judge_calls"] == 2
    assert summary["trigger_fallback_slots"] == 1
