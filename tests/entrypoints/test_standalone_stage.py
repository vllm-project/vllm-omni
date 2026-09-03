# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for standalone stage mode."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from vllm_omni.entrypoints.cli.serve import OmniServeCommand
from vllm_omni.entrypoints.utils import extract_standalone_stage_config
from vllm_omni.utils.tracking_parser import TrackingArgumentParser

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_stage_configs():
    """Build a 2-stage TTS-like OmegaConf config list."""
    return OmegaConf.create(
        [
            {
                "stage_id": 0,
                "max_num_seqs": 64,
                "final_output": False,
                "output_connectors": {"to_stage_1": "connector_of_shared_memory"},
                "engine_args": {
                    "model_stage": "talker",
                    "engine_output_type": "latent",
                    "async_chunk": True,
                    "custom_process_next_stage_input_func": "some.module.talker2code2wav",
                    "async_chunk_process_next_stage_input_func": "some.module.talker2code2wav_async",
                    "custom_process_input_func": "some.module.entry_input",
                },
            },
            {
                "stage_id": 1,
                "max_num_seqs": 64,
                "final_output": True,
                "final_output_type": "audio",
                "input_sources": [0],
                "input_connectors": {"from_stage_0": "connector_of_shared_memory"},
                "engine_args": {
                    "model_stage": "code2wav",
                    "engine_output_type": "audio",
                    "custom_process_input_func": "some.module.downstream_input",
                },
            },
        ]
    )


class TestExtractStandaloneStageConfig:
    def test_downstream_stage_renumbered_to_zero(self):
        """Extracting stage 1 renumbers it to stage_id=0 and marks final_output."""
        configs = _make_stage_configs()
        result = extract_standalone_stage_config(configs, 1)
        cfg = result[0]

        assert cfg.stage_id == 0
        assert cfg.final_output is True

    def test_clears_all_connectors_and_input_sources(self):
        """Connectors and input_sources reference other stages that don't exist standalone."""
        configs = _make_stage_configs()

        r0 = extract_standalone_stage_config(configs, 0)
        assert "output_connectors" not in r0[0]

        r1 = extract_standalone_stage_config(configs, 1)
        assert "input_connectors" not in r1[0]
        assert "input_sources" not in r1[0]

    def test_disables_async_chunk_and_strips_next_stage_transforms(self):
        """Async chunk and next-stage transforms require an orchestrator."""
        configs = _make_stage_configs()
        result = extract_standalone_stage_config(configs, 0)
        ea = result[0].engine_args

        assert ea.async_chunk is False
        assert "custom_process_next_stage_input_func" not in ea
        assert "async_chunk_process_next_stage_input_func" not in ea

    def test_preserves_own_input_processor(self):
        """The stage's own custom_process_input_func must survive extraction."""
        configs = _make_stage_configs()

        r0 = extract_standalone_stage_config(configs, 0)
        assert r0[0].engine_args.custom_process_input_func == "some.module.entry_input"

        r1 = extract_standalone_stage_config(configs, 1)
        assert r1[0].engine_args.custom_process_input_func == "some.module.downstream_input"

    def test_infers_final_output_type_from_engine_output_type(self):
        """Stage 0 has no final_output_type — should infer from engine_output_type."""
        configs = _make_stage_configs()
        result = extract_standalone_stage_config(configs, 0)

        assert result[0].final_output_type == "latent"

    def test_preserves_existing_final_output_type(self):
        """Stage 1 already has final_output_type=audio — don't overwrite."""
        configs = _make_stage_configs()
        result = extract_standalone_stage_config(configs, 1)

        assert result[0].final_output_type == "audio"

    def test_invalid_stage_id_raises_with_available_list(self):
        configs = _make_stage_configs()
        with pytest.raises(ValueError, match=r"stage_id 99 not found.*available: \[0, 1\]"):
            extract_standalone_stage_config(configs, 99)

    def test_returns_omegaconf_config(self):
        """Result should be usable by the engine (OmegaConf attribute access)."""
        configs = _make_stage_configs()
        result = extract_standalone_stage_config(configs, 0)

        assert len(result) == 1
        cfg = result[0]
        assert cfg.engine_args.model_stage == "talker"
        assert cfg.max_num_seqs == 64


class TestLatentBroadening:
    """Verify the latent postprocess broadening does not affect co-located mode."""

    def test_latent_with_downstream_consumers_not_broadened(self):
        """In co-located mode, latent stages have downstream consumers.

        The broadening (engine_output_type in ("audio", "latent") and not
        downstream_req_ids) only fires when downstream_req_ids is empty.
        With downstream consumers present, the broadening is skipped.
        This test verifies the condition logic is correct.
        """
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
        """In standalone mode, no downstream consumers exist.

        The broadening fires, giving the AR decode loop its hidden_states
        feedback for postprocess.
        """
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
