# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import yaml

from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.model_executor.models.indextts2.configuration_indextts2 import (
    IndexTTS2Config,
    IndexTTS25Config,
)
from vllm_omni.model_executor.models.indextts2.pipeline import INDEXTTS25_PIPELINE


def test_indextts25_defaults_are_distinct_from_v2():
    v2 = IndexTTS2Config()
    v25 = IndexTTS25Config()

    assert v2.model_type == "indextts2"
    assert v2.gpt["number_text_tokens"] == 12000
    assert v2.gpt["condition_num_latent"] == 32
    assert v2.use_gpt_latent is True
    assert v2.semantic_codec_type == "repcodec"
    assert v2.vocab_size == v2.gpt["number_mel_codes"]

    assert v25.model_type == "indextts2_5"
    assert v25.gpt["number_text_tokens"] == 60509
    assert v25.gpt["condition_num_latent"] == 32
    assert v25.use_gpt_latent is False
    assert v25.semantic_codec_type == "enhanced"
    assert v25.semantic_codec_checkpoint == "codec.pth"
    assert v25.tokenizer_file == "multilingual_zh_ja_yue_char_del.tiktoken"
    assert v25.vocab_size == v25.gpt["number_mel_codes"]


def test_indextts25_explicit_overrides_win_over_defaults():
    config = IndexTTS25Config(
        use_gpt_latent=True,
        gpt={
            "number_text_tokens": 70000,
            "condition_num_latent": 2,
        },
        vocab_size=9000,
    )

    assert config.use_gpt_latent is True
    assert config.gpt["number_text_tokens"] == 70000
    assert config.gpt["condition_num_latent"] == 2
    assert config.gpt["model_dim"] == config.hidden_size
    assert config.vocab_size == 9000


def test_indextts25_pipeline_is_registered_with_two_distinct_stages():
    assert OMNI_PIPELINES["indextts2_5"] is INDEXTTS25_PIPELINE
    assert INDEXTTS25_PIPELINE.default_deploy_config_name == "indextts2_5.yaml"
    assert [stage.model_stage for stage in INDEXTTS25_PIPELINE.stages] == [
        "indextts2_5_talker",
        "indextts2_5_s2mel_decoder",
    ]
    assert [stage.model_arch for stage in INDEXTTS25_PIPELINE.stages] == [
        None,
        "IndexTTS25S2MelDecoder",
    ]
    assert all("tokenizer" not in stage.extras for stage in INDEXTTS25_PIPELINE.stages)


def test_indextts25_default_deploy_selects_validated_triton_backend():
    deploy_path = Path(__file__).parents[4] / "vllm_omni" / "deploy" / "indextts2_5.yaml"
    config = yaml.safe_load(deploy_path.read_text(encoding="utf-8"))
    stage0 = next(stage for stage in config["stages"] if stage["stage_id"] == 0)

    assert stage0["attention_backend"] == "TRITON_ATTN"
    assert stage0["enable_chunked_prefill"] is False


def test_indextts25_deploy_configs_keep_latent_policy_in_sync():
    deploy_dir = Path(__file__).parents[4] / "vllm_omni" / "deploy"
    expected = {"indextts2_5.yaml": False}
    for filename, use_gpt_latent in expected.items():
        with open(deploy_dir / filename, encoding="utf-8") as deploy_file:
            config = yaml.safe_load(deploy_file)
        assert config["pipeline"] == "indextts2_5"
        assert config["async_chunk"] is False
        assert all("tokenizer" not in stage for stage in config["stages"])
        assert [stage["hf_overrides"]["use_gpt_latent"] for stage in config["stages"]] == [
            use_gpt_latent,
            use_gpt_latent,
        ]
