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


class TestClearConnectorsParam:
    def test_clear_connectors_false_preserves_stage_id_and_connectors(self):
        """With clear_connectors=False, original stage_id and connectors survive.

        This is required for connector key matching: keys include stage IDs,
        so renumbering breaks producer-consumer key agreement.
        """
        configs = _make_stage_configs()

        r0 = extract_standalone_stage_config(configs, 0, clear_connectors=False)
        assert r0[0].stage_id == 0
        assert "output_connectors" in r0[0]

        r1 = extract_standalone_stage_config(configs, 1, clear_connectors=False)
        assert r1[0].stage_id == 1
        assert "input_connectors" in r1[0]

    def test_clear_connectors_false_preserves_producer_hook(self):
        """Producer hook controls embedding accumulation for send_full_payload_outputs.

        Without it, the model runner skips embedding accumulation and the
        downstream stage never receives data.
        """
        configs = _make_stage_configs()
        result = extract_standalone_stage_config(configs, 0, clear_connectors=False)
        ea = result[0].engine_args

        assert ea.custom_process_next_stage_input_func == "some.module.talker2code2wav"
        assert ea.async_chunk_process_next_stage_input_func == "some.module.talker2code2wav_async"

    def test_clear_connectors_false_still_disables_async_chunk(self):
        """async_chunk requires orchestrator coordination regardless of connector mode."""
        configs = _make_stage_configs()
        result = extract_standalone_stage_config(configs, 0, clear_connectors=False)

        assert result[0].engine_args.async_chunk is False
        assert result[0].final_output is True


class TestCodecTokenParsing:
    """Test _parse_codec_tokens which is the critical data path for downstream stages."""

    def test_2d_codec_transposed_to_codebook_major(self):
        """2D [num_frames, Q] must be transposed to codebook-major [Q*num_frames].

        This ordering matches how the code2wav model expects flattened codec input.
        Wrong ordering produces garbage audio.
        """
        from vllm_omni.entrypoints.openai.serving_stage import _parse_codec_tokens

        result = _parse_codec_tokens({"codes": {"audio": [[1, 2], [3, 4], [5, 6]]}}, "test")
        assert result == [1, 3, 5, 2, 4, 6]

    def test_rejects_oversized_codec_data(self):
        """Prevent unbounded allocation from malicious/corrupt input."""
        from fastapi.responses import JSONResponse

        from vllm_omni.entrypoints.openai.serving_stage import _parse_codec_tokens

        huge = list(range(3 * 1024 * 1024))
        result = _parse_codec_tokens({"codes": {"audio": huge}}, "test")
        assert isinstance(result, JSONResponse)

    def test_rejects_missing_codec_data(self):
        from fastapi.responses import JSONResponse

        from vllm_omni.entrypoints.openai.serving_stage import _parse_codec_tokens

        result = _parse_codec_tokens({"no_codes": {}}, "test")
        assert isinstance(result, JSONResponse)


class TestConnectorPlumbing:
    """Test kv_transfer_params flow through the connector plumbing.

    Follows llm-d's pattern: mock internal components, verify data flows
    correctly without real GPUs or RDMA. Each test covers a real failure
    mode that would break omni disagg if the plumbing is wrong.
    """

    @pytest.fixture()
    def _mixin(self):
        """Create a minimal OmniConnectorModelRunnerMixin with recv state."""
        from unittest.mock import MagicMock

        from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin

        m = object.__new__(OmniConnectorModelRunnerMixin)
        m._stage_id = 1
        m._request_ids_mapping = {}
        m._per_request_sender = {}
        m._stage_recv_req_ids = set()
        m._chunk_stream_completed = set()
        m._pending_load_reqs = {}
        m._work_available = MagicMock()
        m._lock = __import__("threading").Lock()
        m._get_req_chunk = {}
        m._finished_load_reqs = set()
        m._chunk_ready_req_ids = set()
        m._chunk_finished_req_ids = set()
        m._full_payload_pending_broadcast_req_ids = set()
        m._async_chunk_updated_req_ids = set()
        m._local_stage_payload_cache = {}
        m._local_request_metadata = {}
        return m

    def test_per_request_sender_survives_to_poll(self, _mixin):
        """kv_transfer_params from register_chunk_recv must be readable by _poll_single_request.

        If the stash is lost between registration and polling, the recv thread
        falls back to the default sender (update_sender_info) which may point
        to a different thinker pod — wrong data for this request.
        """
        from types import SimpleNamespace

        request = SimpleNamespace(
            request_id="req-42",
            external_req_id="ext-42",
            kv_transfer_params={"source_host": "10.0.0.1", "source_port": 50051},
        )
        _mixin.register_chunk_recv(request)

        sender_meta = _mixin._per_request_sender.get("req-42")
        assert sender_meta is not None
        assert sender_meta["source_host"] == "10.0.0.1"
        assert sender_meta["source_port"] == 50051

    def test_orchestrator_managed_requests_use_default_sender(self, _mixin):
        """Requests without kv_transfer_params must not populate per-request sender.

        This preserves backward compatibility: orchestrator-managed deployments
        use update_sender_info (Path 3), not per-request metadata (Path 2).
        """
        from types import SimpleNamespace

        request = SimpleNamespace(request_id="req-99", external_req_id=None)
        _mixin.register_chunk_recv(request)

        assert _mixin._per_request_sender.get("req-99") is None

    def test_cleanup_prevents_sender_leak(self, _mixin):
        """Per-request sender must be cleaned up when the request finishes.

        Without cleanup, long-running talker processes accumulate stale entries
        for every completed request — unbounded memory growth.
        """
        _mixin._per_request_sender["req-1"] = {"source_host": "10.0.0.1"}
        _mixin._get_req_chunk["req-1"] = 0

        _mixin._clear_recv_delivery_state("req-1")

        assert "req-1" not in _mixin._per_request_sender

    def test_recv_methods_accept_metadata_kwarg(self):
        """All recv methods must accept metadata=None without breaking.

        This is the backward compatibility contract: existing orchestrator
        paths pass no metadata, and the connector falls back to Path 3
        (default sender). If the signature is wrong, every deployment breaks.
        """
        from unittest.mock import MagicMock

        from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin

        m = object.__new__(OmniConnectorModelRunnerMixin)
        mock_connector = MagicMock()
        mock_connector.get.return_value = ({"data": "test"}, 100)

        m._get_local_tp_group = lambda: None

        m._recv_ordinary_stage_result(mock_connector, "0", "1", "key-1", metadata=None)
        mock_connector.get.assert_called_with("0", "1", "key-1", metadata=None)

        m._recv_ordinary_stage_result(
            mock_connector,
            "0",
            "1",
            "key-2",
            metadata={"source_host": "10.0.0.1", "source_port": 50051},
        )
        mock_connector.get.assert_called_with(
            "0",
            "1",
            "key-2",
            metadata={"source_host": "10.0.0.1", "source_port": 50051},
        )

    def test_extra_args_kv_transfer_params_on_sampling_params(self):
        """kv_transfer_params set via SamplingParams.extra_args must survive.

        This is how the downstream handler passes connector info to the engine.
        If extra_args is dropped or ignored, the talker never learns where to
        pull embeddings from.
        """
        from vllm import SamplingParams

        sp = SamplingParams(max_tokens=100, detokenize=False)
        sp.extra_args = {"kv_transfer_params": {"source_host": "10.0.0.1"}}

        assert sp.extra_args["kv_transfer_params"]["source_host"] == "10.0.0.1"


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
