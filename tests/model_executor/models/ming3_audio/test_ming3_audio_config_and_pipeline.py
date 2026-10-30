# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from pytest_mock import MockerFixture
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoConfig, BatchFeature, PreTrainedTokenizerFast
from vllm.config import VllmConfig
from vllm.config.speech_to_text import SpeechToTextParams
from vllm.exceptions import VLLMValidationError
from vllm.model_executor.models.interfaces import (
    is_hybrid,
    supports_multimodal,
    supports_pp,
    supports_transcription,
)
from vllm.multimodal.processing import InputProcessingContext
from vllm.tokenizers import cached_tokenizer_from_config

from vllm_omni.config import OmniModelConfig
from vllm_omni.config.pipeline_registry import resolve_pipeline_config
from vllm_omni.config.stage_config import StageExecutionType
from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.ming3_audio import (
    ming3_audio as ming3_audio_module,
)
from vllm_omni.model_executor.models.ming3_audio.ming3_audio import (
    BailingMoeV3AudioForConditionalGeneration,
    Ming3AudioStreamingPostProcessor,
)
from vllm_omni.model_executor.models.ming3_audio.processing import ASR_PROMPT
from vllm_omni.model_executor.models.registry import OmniModelRegistry
from vllm_omni.model_executor.stage_input_processors.ming3_audio import (
    thinker2talker_async_chunk,
    thinker2talker_token_only,
)
from vllm_omni.transformers_utils.configs.ming3_audio import (
    BailingMoeV3AudioConfig,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_TEST_CHAT_TEMPLATE = """\
{%- for message in messages -%}
{{- '<role>' + message['role']|upper + '</role>' -}}
{%- if message['content'] is string -%}
{{- message['content'] -}}
{%- else -%}
{%- for item in message['content'] -%}
{%- if item['type'] == 'audio' -%}
{{- '<|audio_start|><|audio_pad|><|audio_end|>' -}}
{%- elif item['type'] == 'text' -%}
{{- item['text'] -}}
{%- endif -%}
{%- endfor -%}
{%- endif -%}
{{- '<|role_end|>' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- '<role>ASSISTANT</role>' -}}
{%- endif -%}
"""


@pytest.fixture(scope="module")
def tiny_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a self-contained one-layer checkpoint layout for CPU CI."""
    model_path = tmp_path_factory.mktemp("ming3_audio")
    config = BailingMoeV3AudioConfig(
        architectures=["BailingMoeV3AudioForConditionalGeneration"],
        text_config={
            "model_type": "",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "vocab_size": 16,
            "max_position_embeddings": 512,
            "head_dim": 8,
            "qk_head_dim": 8,
            "qk_nope_head_dim": 4,
            "qk_rope_head_dim": 4,
            "v_head_dim": 8,
            "kv_lora_rank": 4,
        },
        audio_config={
            # Whisper's feature extractor supports 80- and 128-bin mels.
            "n_mels": 80,
            "n_audio_ctx": 128,
            "n_audio_state": 8,
            "n_audio_head": 2,
            "n_audio_layer": 1,
            "downsampler_kernel_size_list": [3, 3, 3],
            "downsampler_stride_list": [2, 2, 2],
            "downsampler_padding_list": [1, 1, 1],
            "whisper_chunk_size": 32,
        },
        audio_token_id=3,
    )
    config.save_pretrained(model_path)
    # Match the released checkpoint's unusual empty nested model_type. The
    # native adapter must normalize it without loading remote configuration.
    config_path = model_path / "config.json"
    serialized_config = json.loads(config_path.read_text())
    serialized_config["text_config"]["model_type"] = ""
    config_path.write_text(json.dumps(serialized_config))

    backend = Tokenizer(
        WordLevel(
            vocab={
                "[UNK]": 0,
                "<|audio_start|>": 1,
                "<|audio_end|>": 2,
                "<|audio_pad|>": 3,
                "test": 4,
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        chat_template=_TEST_CHAT_TEMPLATE,
        additional_special_tokens=[
            "<|audio_start|>",
            "<|audio_end|>",
            "<|audio_pad|>",
        ],
    )
    tokenizer.save_pretrained(model_path)
    return model_path


@pytest.fixture(scope="module")
def tiny_model_config(tiny_model_path: Path) -> OmniModelConfig:
    return OmniModelConfig(
        model=str(tiny_model_path),
        tokenizer=str(tiny_model_path),
        trust_remote_code=False,
        max_model_len=128,
        limit_mm_per_prompt={"audio": 1},
        model_arch="BailingMoeV3AudioForConditionalGeneration",
        model_stage="thinker",
        hf_config_name="text_config",
    )


@pytest.fixture(scope="module")
def tiny_processor(tiny_model_config: OmniModelConfig):
    ctx = InputProcessingContext(
        tiny_model_config,
        tokenizer=cached_tokenizer_from_config(tiny_model_config),
    )
    return BailingMoeV3AudioForConditionalGeneration._processor_factory.build_processor(ctx)


@pytest.fixture
def tiny_model(
    tiny_model_config: OmniModelConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> BailingMoeV3AudioForConditionalGeneration:
    class DummyLanguageModel(nn.Module):
        def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
            super().__init__()
            self.vllm_config = vllm_config
            self.prefix = prefix
            text_config = vllm_config.model_config.hf_config
            self.embedding = nn.Embedding(text_config.vocab_size, text_config.hidden_size)

        def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
            return self.embedding(input_ids)

        def make_empty_intermediate_tensors(self, *args: object, **kwargs: object):
            return None

    monkeypatch.setattr(
        ming3_audio_module,
        "BailingMoeV3ForCausalLM",
        DummyLanguageModel,
    )
    return BailingMoeV3AudioForConditionalGeneration(vllm_config=VllmConfig(model_config=tiny_model_config))


def test_native_config_loads_without_remote_code(tiny_model_path: Path):
    config = AutoConfig.from_pretrained(tiny_model_path, trust_remote_code=False)

    assert isinstance(config, BailingMoeV3AudioConfig)
    assert config.architectures == ["BailingMoeV3AudioForConditionalGeneration"]
    assert config.model_type == "bailing_moe_v3_audio"
    assert config.text_config.model_type == "bailing_hybrid"
    assert config.text_config.hidden_size == 16
    assert config.text_config.num_hidden_layers == 1
    assert config.audio_config.n_audio_layer == 1
    assert config.audio_config.downsampler_stride_list == [2, 2, 2]
    assert config.audio_token_id == 3
    assert config.norm_query_embeds is False
    assert config.get_text_config() is config.text_config


def test_speech_qa_uses_native_audio_preprocessing(
    tiny_model_config: OmniModelConfig,
    tiny_processor,
):
    assert tiny_model_config.use_mla is True
    assert tiny_model_config.get_head_size() == 8
    tokenizer = cached_tokenizer_from_config(tiny_model_config)
    system_prompt = "回答音频中的问题，回答不要使用markdown格式。"
    rendered_prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "audio"}]},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert system_prompt in rendered_prompt

    items = tiny_processor.info.parse_mm_data({"audio": (np.zeros(16000, dtype=np.float32), 16000)})
    output = tiny_processor(
        rendered_prompt,
        mm_items=items,
        hf_processor_mm_kwargs={},
    )
    mm_data = output["mm_kwargs"].get_data()

    assert output["prompt_token_ids"].count(tiny_model_config.hf_config.audio_token_id) == 7
    assert mm_data["audio_feats"].shape == (1, 100, 80)
    assert mm_data["audio_feats_lengths"].tolist() == [[100]]
    # The encoder reads lengths as Python ints; device copies would sync.
    fields = tiny_processor._get_mm_fields_config(BatchFeature({}), {})
    assert fields["audio_feats_lengths"].field.keep_on_cpu
    assert not fields["audio_feats"].field.keep_on_cpu


def test_partial_dummy_model_executes_real_audio_path(
    tiny_model: BailingMoeV3AudioForConditionalGeneration,
    monkeypatch: pytest.MonkeyPatch,
):
    assert len(tiny_model.audio_encoder.blocks) == 1
    assert tiny_model.language_model.prefix == "language_model"
    assert tiny_model.language_model.vllm_config.model_config.hf_config.num_hidden_layers == 1

    features = torch.zeros(1, 63, 80)
    lengths = torch.tensor([[63]])
    embeddings = tiny_model.embed_multimodal(audio_feats=features, audio_feats_lengths=lengths)

    assert len(embeddings) == 1
    assert embeddings[0].shape == (4, 16)
    assert torch.isfinite(embeddings[0]).all()
    monkeypatch.setattr(tiny_model.config, "norm_query_embeds", True)
    normalized = tiny_model.embed_multimodal(audio_feats=features, audio_feats_lengths=lengths)
    torch.testing.assert_close(normalized[0], torch.nn.functional.normalize(embeddings[0], dim=-1))


def test_partial_dummy_model_loads_mapped_audio_weights(
    tiny_model: BailingMoeV3AudioForConditionalGeneration,
):
    encoder_weight = torch.full_like(tiny_model.audio_encoder.conv1.weight, 0.25)
    projector_weight = torch.full_like(tiny_model.audio_projector.proj[0].weight, 0.5)

    loaded = tiny_model.load_weights(
        [
            ("model.audio.conv1.weight", encoder_weight),
            ("linear_proj_audio.0.weight", projector_weight),
            ("model.visual.blocks.0.weight", torch.ones(1)),
        ]
    )

    assert loaded == {
        "audio_encoder.conv1.weight",
        "audio_projector.proj.0.weight",
    }
    torch.testing.assert_close(tiny_model.audio_encoder.conv1.weight, encoder_weight)
    torch.testing.assert_close(tiny_model.audio_projector.proj[0].weight, projector_weight)


def test_ming3_model_is_registered():
    model_cls = OmniModelRegistry._try_load_model_cls("BailingMoeV3AudioForConditionalGeneration")
    assert model_cls is BailingMoeV3AudioForConditionalGeneration


def test_ming3_talker_reuses_shared_ming_implementation():
    from vllm_omni.model_executor.models.ming_flash_omni.ming_flash_omni_talker import (
        MingFlashOmniTalkerForConditionalGeneration,
    )

    assert OmniModelRegistry._try_load_model_cls("BailingTalker2") is MingFlashOmniTalkerForConditionalGeneration


def test_ming3_advertises_required_vllm_capabilities():
    model = BailingMoeV3AudioForConditionalGeneration
    assert supports_multimodal(model)
    assert supports_pp(model)
    assert is_hybrid(model)
    assert supports_transcription(model)


def test_checkpoint_prefix_mapping_reuses_bailing_loader():
    mapper = BailingMoeV3AudioForConditionalGeneration.hf_to_vllm_mapper
    names = mapper.apply_list(
        [
            "model.audio.conv1.weight",
            "linear_proj_audio.0.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "lm_head.weight",
            "model.visual.blocks.0.weight",
            "linear_proj.0.weight",
        ]
    )
    assert names == [
        "audio_encoder.conv1.weight",
        "audio_projector.proj.0.weight",
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.lm_head.weight",
    ]


def test_asr_service_does_not_split_model_context():
    config = BailingMoeV3AudioForConditionalGeneration.get_speech_to_text_config(None, "transcribe")
    assert config.sample_rate == 16000
    assert config.max_audio_clip_s is None
    assert config.min_energy_split_window_size is None
    assert config.allow_audio_chunking is False

    translation_init_config = BailingMoeV3AudioForConditionalGeneration.get_speech_to_text_config(None, "translate")
    assert translation_init_config.sample_rate == 16000


def test_translation_is_rejected_by_model(tiny_model_config: OmniModelConfig):
    stt_config = BailingMoeV3AudioForConditionalGeneration.get_speech_to_text_config(None, "translate")
    params = SpeechToTextParams(
        audio=np.zeros(160, dtype=np.float32),
        stt_config=stt_config,
        model_config=tiny_model_config,
        task_type="translate",
    )

    with pytest.raises(ValueError, match="does not support translation"):
        BailingMoeV3AudioForConditionalGeneration.get_generation_prompt(params)


def test_asr_generation_prompt_uses_checkpoint_chat_template(
    tiny_model_config: OmniModelConfig,
):
    audio = np.zeros(160, dtype=np.float32)
    stt_config = BailingMoeV3AudioForConditionalGeneration.get_speech_to_text_config(None, "transcribe")
    params = SpeechToTextParams(
        audio=audio,
        stt_config=stt_config,
        model_config=tiny_model_config,
        task_type="transcribe",
    )
    prompt = BailingMoeV3AudioForConditionalGeneration.get_generation_prompt(params)
    tokenizer = cached_tokenizer_from_config(tiny_model_config)
    expected_ids = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ASR_PROMPT},
                    {"type": "audio"},
                ],
            }
        ],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    assert prompt["prompt_token_ids"] == expected_ids
    assert prompt["multi_modal_data"]["audio"] is audio


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("English\thello world", "hello world"),
        ("hello world", "hello world"),
    ],
)
def test_language_prefix_post_processing(text: str, expected: str):
    assert BailingMoeV3AudioForConditionalGeneration.post_process_output(text) == expected


def test_streaming_language_prefix_post_processing():
    processor = Ming3AudioStreamingPostProcessor()
    assert processor.process_delta("English", finished=False) == ""
    assert processor.process_delta("\thello", finished=False) == "hello"
    assert processor.process_delta(" world", finished=True) == " world"


def test_ming3_talker_bridge_uses_generated_text_and_default_voice(
    mocker: MockerFixture,
):
    source = mocker.Mock(
        spec=["outputs"],
        outputs=[mocker.Mock(spec=["text"], text="abc def")],
    )
    prompt = OmniTokensPrompt(prompt_token_ids=[1], additional_information={"max_text_length": 42})

    [talker_input] = thinker2talker_token_only([source], prompt)
    info = talker_input["additional_information"]

    assert talker_input["prompt_token_ids"] == [0]
    assert info["ming_task"] == "omni"
    assert info["text"] == "abc def"
    assert info["voice_name"] is None
    assert info["voice_mode"] == "ip"
    assert info["ip_name"] == "灵小柔"
    assert info["prompt_wav"] is None
    assert info["reference_audio"] is None
    assert info["speaker_conditioning"] == "mute"
    assert "Target language: English." in info["prompt"]
    assert info["max_text_length"] == 42


def test_ming3_pipeline_supports_text_and_audio_outputs():
    pipeline = resolve_pipeline_config("bailing_moe_v3_audio")
    thinker_only = resolve_pipeline_config("bailing_moe_v3_audio_thinker_only")

    assert pipeline is not None and thinker_only is not None
    assert pipeline.model_arch == "BailingMoeV3AudioForConditionalGeneration"
    assert pipeline.default_deploy_config_name == "ming3_audio_flash.yaml"
    assert thinker_only.default_deploy_config_name == "ming3_audio_flash_thinker_only.yaml"
    assert len(pipeline.stages) == 2
    assert thinker_only.stages == pipeline.stages[:1]

    thinker = pipeline.get_stage(0)
    assert thinker.model_stage == "thinker"
    assert thinker.final_output is True
    assert thinker.final_output_type == "text"
    assert thinker.requires_multimodal_data is True
    assert thinker.hf_config_name == "text_config"
    assert thinker.sampling_constraints == {"detokenize": True}

    talker = pipeline.get_stage(1)
    assert talker.model_stage == "ming3_tts"
    assert talker.model_arch == "BailingTalker2"
    assert talker.execution_type is StageExecutionType.LLM_GENERATION
    assert talker.supports_pending_generation is True
    assert talker.input_sources == (0,)
    assert talker.final_output is True
    assert talker.final_output_type == "audio"
    assert talker.tokenizer_subdir == "llm"
    assert talker.sync_process_input_func == (
        f"{thinker2talker_token_only.__module__}.{thinker2talker_token_only.__name__}"
    )
    assert thinker.async_chunk_process_next_stage_input_func == (
        f"{thinker2talker_async_chunk.__module__}.{thinker2talker_async_chunk.__name__}"
    )


def test_ming3_async_chunk_emits_stable_sentences_and_terminal_tail(mocker: MockerFixture):
    class Tokenizer:
        is_fast = True

        def __len__(self):
            return 4

        def convert_ids_to_tokens(self, ids, **kwargs):
            return [{1: "这是一段包含足够多汉字的正常回答", 2: "。", 3: "世界"}[token_id] for token_id in ids]

        def convert_tokens_to_string(self, tokens):
            return "".join(tokens)

    tokenizer = Tokenizer()
    manager = mocker.Mock(spec=["_ming3_text_tokenizer", "_ming3_chunk_state"])
    manager._ming3_text_tokenizer = tokenizer
    manager._ming3_chunk_state = {}
    request = mocker.Mock(spec=["external_req_id", "prompt_token_ids", "all_token_ids", "additional_information"])
    request.external_req_id = "r1"
    request.prompt_token_ids = [99]
    request.additional_information = {"max_text_length": 42, "voice_name": "test_voice_b"}

    request.all_token_ids = [99, 1]
    assert thinker2talker_async_chunk(manager, None, request) is None
    request.all_token_ids = [99, 1, 2]
    first = thinker2talker_async_chunk(manager, None, request)
    assert first is not None
    assert first.additional_information["text"] == "这是一段包含足够多汉字的正常回答。"
    assert first.additional_information["ming_task"] == "omni"
    assert first.additional_information["voice_name"] == "test_voice_b"
    assert first.additional_information["max_text_length"] == 42
    assert first.codes.audio.tolist() == [0]
    assert first.meta.replace_runtime_additional_information is True
    assert thinker2talker_async_chunk(manager, None, request) is None

    request.all_token_ids = [99, 1, 2, 3]
    final = thinker2talker_async_chunk(manager, None, request, is_finished=True)
    assert final is not None
    assert final.additional_information["text"] == "世界。"
    assert manager._ming3_chunk_state == {}


def test_ming_talker_reads_async_chunk_metadata():
    from vllm_omni.model_executor.models.ming_flash_omni.ming_flash_omni_talker import (
        MingFlashOmniTalkerForConditionalGeneration,
    )

    info = {"additional_information": {"text": "你好。", "ming_task": "omni"}}
    assert (
        MingFlashOmniTalkerForConditionalGeneration._extract_additional_info([info]) == info["additional_information"]
    )


def test_ming3_standalone_tts_leaves_ming2_pipeline_unchanged():
    ming2 = resolve_pipeline_config("ming_flash_omni_tts")
    ming3 = resolve_pipeline_config("ming3_audio_tts")
    assert ming2 is not None and ming3 is not None
    old_stage = ming2.get_stage(0)
    assert old_stage.model_stage == "ming_tts"
    assert old_stage.hf_config_name == "talker_config"
    assert old_stage.tokenizer_subdir == "talker/llm"
    assert old_stage.model_path_resolver is None
    new_stage = ming3.get_stage(0)
    assert new_stage.model_stage == "ming3_tts"
    assert new_stage.model_arch == "BailingTalker2"
    assert new_stage.tokenizer_subdir == "llm"
    assert new_stage.model_path_resolver is not None


@pytest.fixture(params=[False, True], ids=["uncached", "cached"])
def history_processor(request, tiny_model_path: Path, monkeypatch: pytest.MonkeyPatch):
    from vllm.multimodal.cache import MultiModalProcessorOnlyCache

    config = OmniModelConfig(
        model=str(tiny_model_path),
        tokenizer=str(tiny_model_path),
        trust_remote_code=False,
        max_model_len=512,
        limit_mm_per_prompt={"audio": 16},
        model_arch="BailingMoeV3AudioForConditionalGeneration",
        model_stage="thinker",
        hf_config_name="text_config",
    )
    ctx = InputProcessingContext(config, tokenizer=cached_tokenizer_from_config(config))
    cache = MultiModalProcessorOnlyCache(config) if request.param else None
    processor = BailingMoeV3AudioForConditionalGeneration._processor_factory.build_processor(ctx)
    # Exercise the real Whisper frontend on CPU without depending on a CUDA
    # context; model-independent transfer is the only substituted operation.
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    return processor, cache


def _process_audio_history(processor, waveforms, *, cache=None):
    tokenizer = processor.info.get_tokenizer()
    messages: list[dict[str, object]] = [{"role": "system", "content": "test"}]
    for index in range(len(waveforms)):
        messages.append({"role": "user", "content": [{"type": "text", "text": "test"}, {"type": "audio"}]})
        if index + 1 < len(waveforms):
            messages.append({"role": "assistant", "content": "test"})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    items = processor.info.parse_mm_data({"audio": [(waveform, 16000) for waveform in waveforms]})
    return processor(prompt, mm_items=items, hf_processor_mm_kwargs={}, cache=cache)


@pytest.mark.parametrize("item_count", [1, 2, 13])
def test_audio_history_preserves_individual_features_and_placeholders(history_processor, item_count):
    history_processor, cache = history_processor
    from vllm_omni.model_executor.models.common.ming.audio_lengths import ming3_audio_token_count
    from vllm_omni.model_executor.models.ming3_audio.processing import _audio_config_args

    rng = np.random.RandomState(42)
    waveforms = [rng.uniform(-0.1, 0.1, size=1600 + 160 * index).astype(np.float32) for index in range(item_count)]
    if item_count > 2:
        waveforms[-1] = waveforms[0].copy()  # Duplicate items must retain both message positions.
    output = _process_audio_history(history_processor, waveforms, cache=cache)
    audio_items = output["mm_kwargs"]["audio"]
    placeholders = output["mm_placeholders"]["audio"]
    assert len(audio_items) == len(placeholders) == item_count
    for index, waveform in enumerate(waveforms):
        expected = history_processor.info.get_feature_extractor()([(waveform, 16000)], return_tensors="pt")
        actual = audio_items[index]
        torch.testing.assert_close(actual["audio_feats"].data, expected["audio_feats"][0])
        assert actual["audio_feats_lengths"].data.item() == expected["audio_feats_lengths"].item()
        count = ming3_audio_token_count(
            expected["audio_feats_lengths"].item(), **_audio_config_args(history_processor.info.get_hf_config())
        )
        assert placeholders[index].length == count
        offset = placeholders[index].offset
        assert output["prompt_token_ids"][offset : offset + count] == [3] * count
    # A second call covers processor-cache hits and uncached equivalence.
    repeated = _process_audio_history(history_processor, waveforms, cache=cache)
    assert repeated["prompt_token_ids"] == output["prompt_token_ids"]
    for previous, current in zip(audio_items, repeated["mm_kwargs"]["audio"], strict=True):
        torch.testing.assert_close(previous["audio_feats"].data, current["audio_feats"].data)


def test_audio_history_respects_configured_limit(history_processor):
    history_processor, cache = history_processor
    assert history_processor.info.get_supported_mm_limits() == {"audio": None}
    with pytest.raises(VLLMValidationError, match="16.*audio"):
        _process_audio_history(history_processor, [np.zeros(1600, dtype=np.float32)] * 17, cache=cache)


@pytest.mark.parametrize("item_count", [1, 2, 13])
def test_audio_history_dummy_shares_context_budget(history_processor, item_count):
    history_processor, cache = history_processor
    from vllm_omni.model_executor.models.common.ming.audio_lengths import ming3_audio_token_count
    from vllm_omni.model_executor.models.ming3_audio.processing import (
        AUDIO_PAD_TOKEN,
        _audio_config_args,
    )

    counts = {"audio": item_count}
    builder = history_processor.dummy_inputs
    text = builder.get_dummy_text(counts)
    assert text.count(AUDIO_PAD_TOKEN) == item_count
    data = builder.get_dummy_mm_data(128, counts)
    total_tokens = len(history_processor.info.get_tokenizer().encode(text)) - item_count
    for waveform in data["audio"]:
        total_tokens += ming3_audio_token_count(
            len(waveform) // 160, **_audio_config_args(history_processor.info.get_hf_config())
        )
    assert total_tokens <= 128


@pytest.mark.parametrize("lengths", [[0], [-1], [11], [1, 2]])
def test_audio_history_rejects_invalid_feature_lengths(lengths):
    with pytest.raises(ValueError, match="length|frame"):
        BailingMoeV3AudioForConditionalGeneration._normalize_audio_inputs(torch.zeros(1, 10, 80), lengths)


def test_audio_history_encoder_matches_individual_clips(tiny_model):
    features = [torch.zeros(17, 80), torch.ones(33, 80)]
    lengths = torch.tensor([[17], [33]])
    combined = tiny_model.embed_multimodal(audio_feats=features, audio_feats_lengths=lengths)
    assert len(combined) == 2
    for index, feature in enumerate(features):
        single = tiny_model.embed_multimodal(
            audio_feats=feature.unsqueeze(0), audio_feats_lengths=lengths[index : index + 1]
        )
        torch.testing.assert_close(combined[index], single[0])


def test_audio_history_normalization_keeps_single_frame_items():
    features, lengths = BailingMoeV3AudioForConditionalGeneration._normalize_audio_inputs(
        [torch.zeros(1, 80), torch.zeros(3, 80)], [[1], [3]]
    )
    assert [feature.shape for feature in features] == [(1, 80), (3, 80)]
    assert lengths == [1, 3]


def test_audio_history_partial_cache_hits_preserve_reordered_items(history_processor, mocker: MockerFixture):
    history_processor, cache = history_processor
    from vllm_omni.model_executor.models.common.ming.audio_lengths import ming3_audio_token_count
    from vllm_omni.model_executor.models.ming3_audio.processing import _audio_config_args

    rng = np.random.RandomState(17)
    waveforms = [rng.uniform(-0.1, 0.1, size=length).astype(np.float32) for length in (1600, 4000, 8000)]
    expected = [
        history_processor.info.get_feature_extractor()([(waveform, 16000)], return_tensors="pt")
        for waveform in waveforms
    ]
    frontend = mocker.spy(history_processor, "_call_hf_processor")
    for order in [(0,), (0, 1), (1, 0, 2)]:
        output = _process_audio_history(history_processor, [waveforms[index] for index in order], cache=cache)
        audio_items = output["mm_kwargs"]["audio"]
        placeholders = output["mm_placeholders"]["audio"]
        assert len(audio_items) == len(placeholders) == len(order)
        for item, placeholder, index in zip(audio_items, placeholders, order, strict=True):
            reference = expected[index]
            torch.testing.assert_close(item["audio_feats"].data, reference["audio_feats"][0])
            length = reference["audio_feats_lengths"].item()
            assert item["audio_feats_lengths"].data.item() == length
            count = ming3_audio_token_count(length, **_audio_config_args(history_processor.info.get_hf_config()))
            assert placeholder.length == count
            assert output["prompt_token_ids"][placeholder.offset : placeholder.offset + count] == [3] * count

    # A and then B must hit the same processor's cache while C is new, even
    # when earlier clips change order. The uncached fixture is the control.
    processed_counts = [len(call.args[1].get("audios", [])) for call in frontend.call_args_list]
    assert processed_counts == ([1, 1, 1] if cache is not None else [1, 2, 3])
