# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for standalone stage mode."""

from __future__ import annotations

import pytest

from vllm_omni.config.omni_config import OmniStageConnectorConfig, OmniStageModelConfig, VllmOmniARStageConfig
from vllm_omni.config.stage_config import StagePipelineConfig
from vllm_omni.entrypoints.cli.serve import OmniServeCommand
from vllm_omni.entrypoints.openai import serving_stage
from vllm_omni.entrypoints.openai.serving_stage import (
    _clean_codec_frames,
    _parse_codec_tokens,
    _to_json_safe,
    _validate_downstream_controls,
)
from vllm_omni.entrypoints.utils import extract_standalone_stage_config
from vllm_omni.utils.tracking_parser import TrackingArgumentParser

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_stage_configs():
    """Build a 2-stage TTS-like structured config list."""
    stage0 = VllmOmniARStageConfig(
        stage_pipeline_config=StagePipelineConfig(
            stage_id=0,
            model_stage="talker",
            engine_output_type="latent",
            final_output=False,
            custom_process_next_stage_input_func="some.module.talker2code2wav",
            async_chunk_process_next_stage_input_func="some.module.talker2code2wav_async",
            custom_process_input_func="some.module.entry_input",
        ),
        connector_config=OmniStageConnectorConfig(
            async_chunk=True,
            output_connectors={"to_stage_1": "connector_of_shared_memory"},
        ),
        model_config=OmniStageModelConfig(),
    )
    stage1 = VllmOmniARStageConfig(
        stage_pipeline_config=StagePipelineConfig(
            stage_id=1,
            model_stage="code2wav",
            engine_output_type="audio",
            final_output=True,
            final_output_type="audio",
            input_sources=(0,),
            custom_process_input_func="some.module.downstream_input",
        ),
        connector_config=OmniStageConnectorConfig(
            input_connectors={"from_stage_0": "connector_of_shared_memory"},
        ),
        model_config=OmniStageModelConfig(),
    )
    return [stage0, stage1]


class TestExtractStandaloneStageConfig:
    def test_downstream_stage_renumbered_to_zero(self):
        """Extracting stage 1 renumbers it to stage_id=0 and marks final_output."""
        configs = _make_stage_configs()
        result = extract_standalone_stage_config(configs, 1)
        cfg = result[0]

        assert cfg.stage_id == 0
        assert cfg.stage_pipeline_config.final_output is True

    def test_clears_all_connectors_and_input_sources(self):
        """Connectors and input_sources reference other stages that don't exist standalone."""
        configs = _make_stage_configs()

        r0 = extract_standalone_stage_config(configs, 0)
        assert r0[0].connector_config.output_connectors is None

        r1 = extract_standalone_stage_config(configs, 1)
        assert r1[0].connector_config.input_connectors is None
        assert r1[0].stage_pipeline_config.input_sources == ()

    def test_disables_async_chunk_and_strips_next_stage_transforms(self):
        """Async chunk and next-stage transforms require an orchestrator."""
        configs = _make_stage_configs()
        result = extract_standalone_stage_config(configs, 0)
        spc = result[0].stage_pipeline_config

        assert result[0].connector_config.async_chunk is False
        assert spc.custom_process_next_stage_input_func is None
        assert spc.async_chunk_process_next_stage_input_func is None

    def test_preserves_own_input_processor(self):
        """The stage's own custom_process_input_func must survive extraction."""
        configs = _make_stage_configs()

        r0 = extract_standalone_stage_config(configs, 0)
        assert r0[0].stage_pipeline_config.custom_process_input_func == "some.module.entry_input"

        r1 = extract_standalone_stage_config(configs, 1)
        assert r1[0].stage_pipeline_config.custom_process_input_func == "some.module.downstream_input"

    def test_infers_final_output_type_from_engine_output_type(self):
        """Stage 0 has no final_output_type — should infer from engine_output_type."""
        configs = _make_stage_configs()
        result = extract_standalone_stage_config(configs, 0)

        assert result[0].stage_pipeline_config.final_output_type == "latent"

    def test_preserves_existing_final_output_type(self):
        """Stage 1 already has final_output_type=audio — don't overwrite."""
        configs = _make_stage_configs()
        result = extract_standalone_stage_config(configs, 1)

        assert result[0].stage_pipeline_config.final_output_type == "audio"

    def test_invalid_stage_id_raises_with_available_list(self):
        configs = _make_stage_configs()
        with pytest.raises(ValueError, match=r"stage_id 99 not found.*available: \[0, 1\]"):
            extract_standalone_stage_config(configs, 99)

    def test_returns_structured_config(self):
        """Result should be usable by the engine."""
        configs = _make_stage_configs()
        result = extract_standalone_stage_config(configs, 0)

        assert len(result) == 1
        cfg = result[0]
        assert cfg.stage_pipeline_config.model_stage == "talker"
        assert isinstance(cfg, VllmOmniARStageConfig)

    def test_clears_omni_kv_config(self):
        """KV transfer has no peer in standalone mode."""
        from dataclasses import replace

        configs = _make_stage_configs()
        configs[0].stage_pipeline_config = replace(configs[0].stage_pipeline_config, omni_kv_config={"role": "sender"})
        result = extract_standalone_stage_config(configs, 0)

        assert result[0].stage_pipeline_config.omni_kv_config is None

    def test_unsupported_engine_output_type_raises(self):
        from dataclasses import replace

        configs = _make_stage_configs()
        configs[0].stage_pipeline_config = replace(configs[0].stage_pipeline_config, engine_output_type="image")
        with pytest.raises(ValueError, match="Unsupported engine_output_type"):
            extract_standalone_stage_config(configs, 0)


class TestLatentBroadening:
    """Verify the latent postprocess broadening does not affect co-located mode."""

    def test_latent_with_downstream_consumers_not_broadened(self):
        from types import SimpleNamespace

        from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

        runner = object.__new__(GPUARModelRunner)
        runner.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(engine_output_type="latent"),
        )
        runner._client_multimodal_output_keys = lambda: set()

        def _needs_downstream(req_id):
            return True

        runner._request_needs_downstream_stage_payload = _needs_downstream

        _, downstream = runner._resolve_pooler_payload_req_ids(["req-1", "req-2"])
        assert downstream == ["req-1", "req-2"]

    def test_latent_without_downstream_broadened(self):
        from types import SimpleNamespace

        from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

        runner = object.__new__(GPUARModelRunner)
        runner.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(engine_output_type="latent"),
        )
        runner._client_multimodal_output_keys = lambda: set()

        def _no_downstream(req_id):
            return False

        runner._request_needs_downstream_stage_payload = _no_downstream

        _, downstream = runner._resolve_pooler_payload_req_ids(["req-1"])
        assert downstream == ["req-1"]


class TestStandaloneCLIValidation:
    @pytest.fixture()
    def parser(self):
        parser = TrackingArgumentParser()
        subparsers = parser.add_subparsers(dest="subcommand")
        cmd = OmniServeCommand()
        cmd.subparser_init(subparsers)
        return parser

    def test_standalone_requires_stage_id(self, parser):
        args = parser.parse_args(["serve", "fake-model", "--omni", "--standalone"])
        cmd = OmniServeCommand()
        with pytest.raises(ValueError, match="--standalone requires --stage-id"):
            cmd.validate(args)

    def test_standalone_headless_mutually_exclusive(self, parser):
        args = parser.parse_args(
            [
                "serve",
                "fake-model",
                "--omni",
                "--standalone",
                "--headless",
                "--stage-id",
                "0",
                "--omni-master-address",
                "127.0.0.1",
                "--omni-master-port",
                "9999",
            ]
        )
        cmd = OmniServeCommand()
        with pytest.raises(ValueError, match="mutually exclusive"):
            cmd.validate(args)

    def test_standalone_with_stage_id_validates(self, parser):
        args = parser.parse_args(["serve", "fake-model", "--omni", "--standalone", "--stage-id", "0"])
        cmd = OmniServeCommand()
        cmd.validate(args)

    def test_stage_id_without_standalone_requires_master(self, parser):
        args = parser.parse_args(["serve", "fake-model", "--omni", "--stage-id", "0"])
        cmd = OmniServeCommand()
        with pytest.raises(ValueError, match="--omni-master-address"):
            cmd.validate(args)


class TestParseCodecTokens:
    def test_2d_flattens_codebook_major(self):
        """Rows are [frame][quantizer]; wire order is quantizer-major flat."""
        stage_output = {"codes": {"audio": [[1, 2], [3, 4], [5, 6]]}}

        assert _parse_codec_tokens(stage_output) == [1, 3, 5, 2, 4, 6]

    def test_flat_list_passes_through(self):
        stage_output = {"codes": {"audio": [7, 8, 9]}}

        assert _parse_codec_tokens(stage_output) == [7, 8, 9]

    def test_dotted_fallback_key(self):
        stage_output = {"codes.audio": [[1, 2]]}

        assert _parse_codec_tokens(stage_output) == [1, 2]

    def test_non_dict_stage_output_raises(self):
        with pytest.raises(ValueError, match="'stage_output' must be a JSON object"):
            _parse_codec_tokens("not-a-dict")

    def test_missing_codec_data_raises(self):
        with pytest.raises(ValueError, match="No codec data"):
            _parse_codec_tokens({"codes": {}})

    def test_non_list_audio_raises(self):
        with pytest.raises(ValueError, match="must be a list of frames"):
            _parse_codec_tokens({"codes": {"audio": {"frames": 1}}})

    def test_ragged_rows_raise(self):
        with pytest.raises(ValueError, match="Ragged codec data at frame 1"):
            _parse_codec_tokens({"codes": {"audio": [[1, 2], [3]]}})

    def test_zero_quantizers_raise(self):
        with pytest.raises(ValueError, match="zero quantizers"):
            _parse_codec_tokens({"codes": {"audio": [[], []]}})

    def test_non_int_element_raises(self):
        with pytest.raises(ValueError, match="must be an integer"):
            _parse_codec_tokens({"codes": {"audio": [[1, "x"], [3, 4]]}})

    def test_bool_element_rejected(self):
        with pytest.raises(ValueError, match="must be an integer"):
            _parse_codec_tokens({"codes": {"audio": [[True, 2]]}})

    def test_negative_token_raises(self):
        with pytest.raises(ValueError, match="Negative codec token"):
            _parse_codec_tokens({"codes": {"audio": [[-1, 2], [3, 4]]}})

    def test_oversize_raises(self, monkeypatch):
        monkeypatch.setattr(serving_stage, "MAX_CODEC_ELEMENTS", 4)
        with pytest.raises(ValueError, match="too large"):
            _parse_codec_tokens({"codes": {"audio": [[1, 2], [3, 4], [5, 6]]}})


class TestValidateDownstreamControls:
    def test_defaults(self):
        max_tokens, response_format, speed = _validate_downstream_controls({})

        assert max_tokens == 65536
        assert response_format == "wav"
        assert speed == 1.0

    def test_valid_override(self):
        max_tokens, response_format, speed = _validate_downstream_controls(
            {"max_tokens": 1024, "response_format": "mp3", "speed": 1.5}
        )

        assert (max_tokens, response_format, speed) == (1024, "mp3", 1.5)

    def test_non_int_max_tokens_raises(self):
        with pytest.raises(ValueError, match="'max_tokens' must be an integer"):
            _validate_downstream_controls({"max_tokens": "1024"})

    def test_bool_max_tokens_raises(self):
        with pytest.raises(ValueError, match="'max_tokens' must be an integer"):
            _validate_downstream_controls({"max_tokens": True})

    def test_zero_max_tokens_raises(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            _validate_downstream_controls({"max_tokens": 0})

    def test_bad_response_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported 'response_format'"):
            _validate_downstream_controls({"response_format": "ogg"})

    def test_speed_out_of_range_raises(self):
        with pytest.raises(ValueError, match="'speed' must be in"):
            _validate_downstream_controls({"speed": 100.0})

    def test_non_numeric_speed_raises(self):
        with pytest.raises(ValueError, match="'speed' must be a number"):
            _validate_downstream_controls({"speed": "fast"})


class TestCleanCodecFrames:
    def test_drops_invalid_rows_and_keeps_valid(self):
        torch = pytest.importorskip("torch")

        audio = torch.tensor(
            [
                [10, 20],  # valid
                [-1, 5],  # negative padding
                [0, 0],  # prefill/EOS padding
                [30, 40],  # valid
            ]
        )
        mm = {"codes": {"audio": audio}}

        out = _clean_codec_frames(mm)

        assert out["codes"]["audio"].tolist() == [[10, 20], [30, 40]]

    def test_out_of_range_rows_are_model_specific(self):
        """Generic filter keeps out-of-range ids; codebook bounds live in each
        model's stage input processor (documented standalone limitation)."""
        torch = pytest.importorskip("torch")

        audio = torch.tensor([[10, 20], [99999, 7]])
        mm = {"codes": {"audio": audio}}

        out = _clean_codec_frames(mm)

        assert out["codes"]["audio"].tolist() == [[10, 20], [99999, 7]]

    def test_missing_codes_passthrough(self):
        mm = {"meta": {}}

        assert _clean_codec_frames(mm) is mm

    def test_non_tensor_passthrough(self):
        mm = {"codes": {"audio": [[1, 2]]}}

        assert _clean_codec_frames(mm) is mm


class TestToJsonSafe:
    def test_to_json_safe_converts_tensors(self):
        torch = pytest.importorskip("torch")

        assert _to_json_safe(torch.tensor([1, 2])) == [1, 2]
        assert _to_json_safe({"a": (1, "x", None)}) == {"a": [1, "x", None]}
