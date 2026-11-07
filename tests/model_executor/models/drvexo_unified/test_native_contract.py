# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exercise hybrid registration, temporal isolation and chunked head capture."""

from types import SimpleNamespace
from typing import TypedDict

import pytest
import torch

from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.entrypoints.openai.speech_usage import SpeechOutputTokenCounter
from vllm_omni.entrypoints.openai.tts_adapters import detect_tts_model_type
from vllm_omni.entrypoints.openai.tts_adapters.drvexo import DRVEXOTTSAdapter
from vllm_omni.entrypoints.openai.tts_adapters.qwen3_tts import Qwen3TTSAdapter
from vllm_omni.model_executor.models.drvexo_speech_8b.lexical import GeneratedTokenRouter, RetokenizedTextRouter
from vllm_omni.model_executor.models.drvexo_speech_8b.talker import (
    DRVEXOSpeech8BTalker,
    Qwen3TTSTalkerForConditionalGeneration,
)
from vllm_omni.model_executor.models.drvexo_unified import decision as decision_module
from vllm_omni.model_executor.models.drvexo_unified.decision import UnifiedClassifier, UnifiedPrefill
from vllm_omni.model_executor.models.drvexo_unified.processing import UnifiedProcessingInfo
from vllm_omni.model_executor.models.drvexo_unified.temporal_projector import TemporalInputProjector, project_utterances
from vllm_omni.model_executor.models.drvexo_unified.thinker import DRVEXOSpeechQwen35ForConditionalGeneration
from vllm_omni.transformers_utils.configs.drvexo import decision_token_limits
from vllm_omni.transformers_utils.configs.drvexo_unified import DRVEXOUnifiedConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _TextKwargs(TypedDict):
    padding: bool


class _ImageKwargs(TypedDict):
    size: dict[str, int]


class _AudioKwargs(TypedDict):
    sampling_rate: int


class _ProcessorKwargs(TypedDict):
    text_kwargs: _TextKwargs
    audio_kwargs: _AudioKwargs


def test_image_audio_kwargs_do_not_require_an_unadvertised_video_scope():
    processor = SimpleNamespace(
        valid_processor_kwargs=_ProcessorKwargs,
        image_processor=SimpleNamespace(valid_kwargs=_ImageKwargs),
    )
    info = SimpleNamespace(
        get_hf_processor=lambda: processor,
        supported_mm_limits={"image": 1, "audio": 1},
    )
    result = UnifiedProcessingInfo.get_supported_mm_processor_kwargs(info)
    assert result == {"text_kwargs": {"padding"}, "images_kwargs": {"size"}, "audio_kwargs": {"sampling_rate"}}


def test_family_registry_preserves_distinct_native_backbones():
    old = OMNI_PIPELINES["drvexo_speech_8b"]
    new = OMNI_PIPELINES["drvexo_speech_qwen35_4b"]
    assert old.model_arch == "DRVEXOSpeech8BForConditionalGeneration"
    assert new.model_arch == "DRVEXOSpeechQwen35ForConditionalGeneration"
    assert new.stages[1].model_arch == "DRVEXOUnifiedTalker"
    assert old.stages[1].model_arch == "DRVEXOSpeech8BTalker"
    assert new.stages[2] == old.stages[2]
    for arch in (new.stages[1].model_arch, old.stages[1].model_arch):
        assert detect_tts_model_type("qwen3_tts", arch) == "drvexo"
    assert detect_tts_model_type("qwen3_tts", "Qwen3TTSTalkerForConditionalGeneration") == "qwen3_tts"
    assert DRVEXOSpeechQwen35ForConditionalGeneration.supports_mm_device_do_normalize is False
    config = DRVEXOUnifiedConfig()
    assert config.get_text_config().model_type == "qwen3_5_text"
    assert "linear_attention" in config.text_config.layer_types
    assert config.vision_config.model_type == "qwen3_5_vision"
    assert decision_token_limits(old.model_type) == (1536, 384)
    assert decision_token_limits(new.model_type) == (4096, 384)
    with pytest.raises(ValueError, match="Unsupported"):
        decision_token_limits("other")


def test_direct_speech_cannot_claim_generated_token_provenance(monkeypatch):
    reference = [[{"audio": [0.1, 0.2]}, 16000]]
    monkeypatch.setattr(Qwen3TTSAdapter, "_build_tts_params", lambda self, request: {"ref_audio": reference})
    adapter = object.__new__(DRVEXOTTSAdapter)
    params = adapter._build_tts_params(None)
    assert params["drvexo_fixed_reply"] is True
    assert "drvexo_generated_token_ids" not in params
    assert params["ref_audio"] is reference


def test_direct_speech_budget_and_finish_reason_belong_to_talker(monkeypatch):
    thinker, talker, codec = object(), object(), object()
    replacement = object()

    def apply(self, params, request, prompt, request_id):
        assert params == [talker, codec]
        return [replacement, codec]

    monkeypatch.setattr(Qwen3TTSAdapter, "apply_sampling_overrides", apply)
    adapter = object.__new__(DRVEXOTTSAdapter)
    assert adapter.entry_stage_id == 1
    assert adapter.apply_sampling_overrides([thinker, talker, codec], None) == [thinker, replacement, codec]
    counter = SpeechOutputTokenCounter(entry_stage_id=adapter.entry_stage_id)
    counter.observe(
        SimpleNamespace(
            metrics={
                "stage_metrics": {
                    "0": {"finish_reason": "length", "num_tokens_out": 0},
                    "1": {"finish_reason": "stop", "num_tokens_out": 24},
                    "2": {"num_tokens_out": 0},
                }
            }
        )
    )
    assert counter.stage0_finish_reason == "stop"
    assert counter.total() == 24


@pytest.mark.parametrize("stage_count", [0, 2, 4])
def test_direct_speech_rejects_sampling_list_without_full_pipeline(stage_count):
    adapter = object.__new__(DRVEXOTTSAdapter)
    with pytest.raises(ValueError, match="full stage-0-indexed"):
        adapter.apply_sampling_overrides([object()] * stage_count, None)


def test_thinker_only_speech_reports_unavailable_before_sampling_overrides(mocker):
    adapter = object.__new__(DRVEXOTTSAdapter)
    overrides = mocker.patch.object(Qwen3TTSAdapter, "apply_sampling_overrides")
    with pytest.raises(ValueError, match="Speech output is unavailable.*Thinker-only"):
        adapter.apply_sampling_overrides([object()], None)
    overrides.assert_not_called()


def test_default_speaker_does_not_override_reference_or_leak_to_next_request(monkeypatch):
    talker = object.__new__(DRVEXOSpeech8BTalker)
    torch.nn.Module.__init__(talker)
    talker.output_conditioning = "hidden_bridge"
    talker.drvexo_speaker = torch.tensor([1.0, 2.0])
    monkeypatch.setattr(talker, "_ensure_quantized_pad_embed", lambda: None)
    monkeypatch.setattr(Qwen3TTSTalkerForConditionalGeneration, "preprocess_batch", lambda self, **kwargs: None)
    explicit = [{"ref_spk_embedding": torch.tensor([3.0, 4.0])}]
    buffers: dict[str, dict[str, object]] = {
        "custom": {"ref_audio": [[{"audio": [0.1, 0.2]}, 16000]]},
        "explicit": {"voice_clone_prompt": explicit},
        "default": {},
    }
    talker.preprocess_batch(req_ids=list(buffers), model_intermediate_buffer=buffers, device="cpu")
    assert "voice_clone_prompt" not in buffers["custom"]
    assert buffers["explicit"]["voice_clone_prompt"] is explicit
    default_prompt = buffers["default"]["voice_clone_prompt"]
    assert isinstance(default_prompt, list)
    assert default_prompt[0]["ref_spk_embedding"] is talker.drvexo_speaker


def test_temporal_convolutions_do_not_leak_across_utterances():
    torch.manual_seed(17)
    projector = TemporalInputProjector(8, 12, 4).eval()
    first, second = torch.randn(6, 8), torch.randn(9, 8)
    before = project_utterances(projector, torch.cat((first, second)), [6, 9])
    changed = project_utterances(projector, torch.cat((first, second + 50)), [6, 9])
    torch.testing.assert_close(before[0], projector(first), rtol=0, atol=0)
    torch.testing.assert_close(before[0], changed[0], rtol=0, atol=0)
    assert not torch.equal(before[1], changed[1])
    for sizes in ([5, 9], [0, 15], [391]):
        with pytest.raises(ValueError, match="lengths"):
            project_utterances(projector, torch.cat((first, second)), sizes)


class _Scorer:
    def score(self, hidden, task):
        return hidden[:1].float()


def test_head_captures_final_normalized_token_across_long_prefill_once():
    prefill = UnifiedPrefill(_Scorer())
    info = {"drvexo_general_decision": {"prompt_length": 4480, "task": "choice"}}
    first = prefill.process(torch.zeros(2048, 2560), [info], [(0, 2048)], [0])
    assert first["drvexo_general_status"][0].item() == 0
    assert first["drvexo_general_scores"][0].numel() == 0
    prefill.process(torch.zeros(2048, 2560), [info], [(0, 2048)], [2048])
    last = torch.zeros(384, 2560)
    last[-1, 0] = 37
    final = prefill.process(last, [info], [(0, 384)], [4096])
    assert final["drvexo_general_status"][0].item() == 1
    assert final["drvexo_general_scores"][0].item() == 37
    again = prefill.process(last, [info], [(0, 384)], [4096])
    assert again["drvexo_general_scores"][0].numel() == 0
    info["drvexo_general_decision"]["task"] = "noul"
    assert prefill.process(last, [info], [(0, 384)], [4096])["drvexo_general_status"][0].item() == -1


def test_nonfinite_head_failure_is_logged_and_next_request_can_score(mocker):
    scorer = mocker.Mock()
    scorer.score.side_effect = [ValueError("Nonfinite native candidate score"), torch.tensor([7.0])]
    debug = mocker.patch.object(decision_module.logger, "debug")
    warning = mocker.patch.object(decision_module.logger, "warning")
    prefill = UnifiedPrefill(scorer)
    broken = {"drvexo_general_decision": {"prompt_length": 1, "task": "choice"}}
    failed = prefill.process(torch.zeros(1, 2560), [broken], [(0, 1)], [0])
    assert failed["drvexo_general_status"][0].item() == -1
    assert failed["drvexo_general_scores"][0].numel() == 0
    warning.assert_called_once()
    assert "Nonfinite native candidate score" in str(warning.call_args)
    debug.assert_not_called()
    healthy = {"drvexo_general_decision": {"prompt_length": 1, "task": "choice"}}
    recovered = prefill.process(torch.zeros(1, 2560), [healthy], [(0, 1)], [0])
    assert recovered["drvexo_general_status"][0].item() == 1
    assert recovered["drvexo_general_scores"][0].item() == 7
    assert warning.call_count == 1


@pytest.mark.parametrize(
    "metadata",
    [{"prompt_length": "1", "task": "choice"}, {"task": "choice"}, {"prompt_length": 1, "task": "unknown"}],
)
def test_malformed_prefill_remains_debug_only_and_next_request_can_score(mocker, metadata):
    debug = mocker.patch.object(decision_module.logger, "debug")
    warning = mocker.patch.object(decision_module.logger, "warning")
    prefill = UnifiedPrefill(_Scorer())
    failed = prefill.process(torch.zeros(1, 2560), [{"drvexo_general_decision": metadata}], [(0, 1)], [0])
    assert failed["drvexo_general_status"][0].item() == -1
    assert failed["drvexo_general_scores"][0].numel() == 0
    debug.assert_called_once()
    warning.assert_not_called()
    healthy = {"drvexo_general_decision": {"prompt_length": 1, "task": "choice"}}
    recovered = prefill.process(torch.ones(1, 2560), [healthy], [(0, 1)], [0])
    assert recovered["drvexo_general_status"][0].item() == 1
    assert recovered["drvexo_general_scores"][0].item() == 1
    assert debug.call_count == 1
    warning.assert_not_called()


@pytest.mark.parametrize("components", [{}, {"general_decision_head": {}}, {"general_decision_head": {"metadata": {}}}])
def test_head_missing_metadata_is_a_package_validation_error(components):
    with pytest.raises(ValueError, match="2560-dimensional"):
        UnifiedClassifier(SimpleNamespace(components=components))


@pytest.mark.parametrize(
    "section,setting,value",
    [
        ("model_config", "enforce_eager", False),
        ("cache_config", "enable_prefix_caching", True),
        ("scheduler_config", "max_num_seqs", 2),
        ("scheduler_config", "enable_chunked_prefill", False),
        ("scheduler_config", "max_num_batched_tokens", 1024),
        ("model_config", "max_model_len", 2048),
        ("model_config", "dtype", torch.float32),
    ],
)
def test_scheduler_startup_error_names_the_incompatible_setting(section, setting, value):
    config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=True, max_model_len=4480, dtype=torch.bfloat16),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        scheduler_config=SimpleNamespace(max_num_seqs=1, enable_chunked_prefill=True, max_num_batched_tokens=2048),
    )
    UnifiedClassifier.validate_scheduler(config)
    setattr(getattr(config, section), setting, value)
    with pytest.raises(ValueError, match=setting):
        UnifiedClassifier.validate_scheduler(config)


def test_missing_scheduler_progress_fails_explicitly():
    model = SimpleNamespace(general_prefill=UnifiedPrefill(_Scorer()))
    with pytest.raises(ValueError, match="scheduler metadata"):
        DRVEXOSpeechQwen35ForConditionalGeneration.make_omni_output(
            model,
            torch.zeros(1, 2560),
            model_intermediate_buffer=[{"drvexo_general_decision": {"prompt_length": 1, "task": "choice"}}],
            request_token_spans=[(0, 1)],
        )


def test_source_heads_load_as_cpu_fp32_and_reject_nonfinite():
    class Checkpoint:
        components = {"general_decision_head": {"metadata": {"hidden_size": 2560, "projection_size": 0}}}

        def iter_component(self, component):
            for task in ("choice", "noul", "score"):
                yield f"heads.{task}.weight", torch.ones(1, 2560)
                yield f"heads.{task}.bias", torch.tensor([2.0])

    classifier = UnifiedClassifier(Checkpoint())
    assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in classifier.head.parameters())
    # A constant final token normalizes to zero, so the exact score is the bias.
    assert classifier.score(torch.full((2560,), 9.0), "choice").item() == 2.0
    with pytest.raises(ValueError, match="representation"):
        classifier.score(torch.full((2560,), float("nan")), "choice")


class _Tokenizer:
    all_special_ids = [9]
    eos_token_id = 9
    added_tokens_decoder: dict[int, object] = {}

    def __init__(self, source):
        self.source = source

    def get_vocab(self):
        return {"whole-word": 4, "<eos>": 9} if self.source else {"half1": 1, "half2": 2, "<eos>": 9}

    def encode(self, text, **kwargs):
        if text.startswith("<|im_start|>assistant"):
            return list(range(9))
        if text == "杭州":
            return [4] if self.source else [1, 2]
        raise ValueError(text)

    def decode(self, ids, **kwargs):
        if ids == ([4] if self.source else [1, 2]):
            return "杭州"
        return "wrong"


def test_different_vocabulary_retokenizes_verified_text_without_word_mapping():
    source, target = _Tokenizer(True), _Tokenizer(False)
    with pytest.raises(ValueError, match="unmapped"):
        GeneratedTokenRouter(source, target).prepare([4, 9], "杭州")
    router = RetokenizedTextRouter(source, target)
    assert router.prepare([4, 9], "杭州")[3:-5] == [1, 2]
    for ids, text in (([4], "上海"), ([4, 9, 4], "杭州"), ([True], "杭州"), ([100], "杭州")):
        with pytest.raises(ValueError):
            router.prepare(ids, text)
