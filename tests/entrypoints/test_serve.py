"""Unit tests for the Omni serve CLI helpers."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest
from pytest_mock import MockerFixture

from vllm_omni.entrypoints.cli.serve import OmniServeCommand, run_headless
from vllm_omni.utils.tracking_parser import TrackingArgumentParser, TrackingNamespace

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_serve_parser_accepts_no_async_chunk_and_marks_it_explicit() -> None:
    """``--no-async-chunk`` should parse to ``async_chunk=False`` and mark the
    shared deploy-level dest as explicitly provided by the user."""
    parser = TrackingArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand")
    cmd = OmniServeCommand()
    cmd.subparser_init(subparsers)

    argv = ["serve", "fake-model", "--omni", "--no-async-chunk"]
    args = parser.parse_args(argv)
    assert args.async_chunk is False

    explicit = args.get_explicit_kwargs_dict()
    assert args.get_explicit_kwargs_dict()
    assert not explicit["async_chunk"]


def _make_headless_args(**kwargs) -> TrackingNamespace:
    defaults = {
        "model": "fake-model",
        "stage_id": 0,
        "replica_id": 0,
        "omni_master_address": "127.0.0.1",
        "omni_master_port": 26000,
        "omni_replica_address": None,
        "omni_dp_size_local": 1,
        "worker_backend": "multi_process",
        "stage_configs_path": None,
        "deploy_config": None,
        "log_stats": False,
        "disable_log_stats": False,
        "stage_init_timeout": 600,
        "tokenizer": None,
    }
    ns_kwargs = {**defaults, **kwargs}
    ns = argparse.Namespace(**ns_kwargs)
    return TrackingNamespace(
        unfiltered_ns=ns,
        explicit_keys=frozenset(ns.__dict__.keys()),
    )


def test_run_headless_requires_stage_id() -> None:
    args = _make_headless_args(stage_id=None)
    with pytest.raises(ValueError, match="--stage-id is required"):
        run_headless(args)


def test_run_headless_requires_master_address() -> None:
    args = _make_headless_args(omni_master_address=None)
    with pytest.raises(ValueError, match="--omni-master-address and --omni-master-port"):
        run_headless(args)


def test_run_headless_requires_master_port() -> None:
    args = _make_headless_args(omni_master_port=None)
    with pytest.raises(ValueError, match="--omni-master-address and --omni-master-port"):
        run_headless(args)


def test_run_headless_rejects_non_multiprocess_worker_backend() -> None:
    args = _make_headless_args(worker_backend="ray")
    with pytest.raises(ValueError, match="worker_backend=multi_process"):
        run_headless(args)


def test_run_headless_raises_when_stage_id_not_in_configs(mocker: MockerFixture) -> None:
    """Headless looks up its assigned stage_id in the loaded deploy YAML and
    fails fast when the launcher's --stage-id doesn't match any entry."""
    other_stage = SimpleNamespace(stage_id=99)
    mocker.patch(
        "vllm_omni.entrypoints.utils.load_and_resolve_stage_configs",
        return_value=("/fake/stages.yaml", [other_stage]),
    )

    args = _make_headless_args(stage_id=0)
    with pytest.raises(ValueError, match="No stage config found for stage_id=0"):
        run_headless(args)


# ---------------------------------------------------------------------------
# run_headless happy paths
# ---------------------------------------------------------------------------


def _make_stage_cfg(stage_id: int, stage_type: str) -> SimpleNamespace:
    """Build a stage config that satisfies every attribute run_headless reads.

    Notably ``engine_args`` is a real dict (not a Mock) so
    ``get_stage_devices_per_replica`` can call ``.get("tensor_parallel_size")``
    and feed the result through ``int()`` without TypeError.
    """
    return SimpleNamespace(
        stage_id=stage_id,
        stage_type=stage_type,
        # No "devices" key -> split_devices_for_replicas skipped, each replica
        # inherits the launcher's CUDA_VISIBLE_DEVICES.
        runtime=None,
        engine_args={},
    )


def test_run_headless_llm_registers_with_auto_assigned_replica_id(mocker: MockerFixture) -> None:
    """LLM headless: each loop iteration registers with auto-assigned
    replica_id (master picks a free slot) and spawns one
    ``OmniCoreEngineProcManager`` per local replica."""
    from vllm_omni.engine.stage_engine_startup import StageRegistrationResponse

    stage_cfg = _make_stage_cfg(0, stage_type="llm")
    parallel_config = SimpleNamespace(
        data_parallel_size_local=1,
        data_parallel_rank=0,
        data_parallel_rank_local=0,
        node_rank_within_dp=0,
    )
    vllm_config = SimpleNamespace(parallel_config=parallel_config, needs_dp_coordinator=False)
    engine_manager = mocker.Mock()

    mocker.patch(
        "vllm_omni.entrypoints.utils.load_and_resolve_stage_configs",
        return_value=("/fake/stages.yaml", [stage_cfg]),
    )
    mocker.patch("vllm_omni.engine.stage_init_utils.prepare_engine_environment")
    mocker.patch("vllm_omni.engine.stage_init_utils.load_omni_transfer_config_for_model", return_value=None)
    mocker.patch(
        "vllm_omni.distributed.omni_connectors.utils.initialization.resolve_omni_kv_config_for_stage",
        return_value=(None, None, None),
    )
    mocker.patch("vllm_omni.engine.stage_init_utils.get_stage_connector_spec", return_value={})
    mocker.patch("vllm_omni.engine.stage_init_utils.build_engine_args_dict", return_value={})
    mocker.patch(
        "vllm_omni.engine.stage_init_utils.build_vllm_config",
        return_value=(vllm_config, object),
    )
    mock_register = mocker.patch(
        "vllm_omni.engine.stage_engine_startup.register_stage_with_omni_master",
        return_value=StageRegistrationResponse(
            handshake_address="tcp://127.0.0.1:26001",
            input_address="tcp://127.0.0.1:26002",
            output_address="tcp://127.0.0.1:26003",
            replica_id=0,
            coordinator_router_address="tcp://127.0.0.1:26100",
        ),
    )
    mock_manager_cls = mocker.patch(
        "vllm_omni.engine.omni_core_engine_proc_manager.OmniCoreEngineProcManager",
        return_value=engine_manager,
    )
    mocker.patch("signal.signal")

    run_headless(_make_headless_args(stage_id=0))

    # The launcher must request auto-assignment (replica_id=None) and the
    # full response so it can wire the master-allocated coordinator into the
    # spawned subprocess. ``replica_binds_sockets=False`` is required for LLM
    # because the head binds all three sockets (handshake, input, output).
    assert mock_register.call_count == 1
    kwargs = mock_register.call_args.kwargs
    assert kwargs["omni_master_address"] == "127.0.0.1"
    assert kwargs["omni_master_port"] == 26000
    assert kwargs["omni_stage_id"] == 0
    assert kwargs["omni_stage_config"] is stage_cfg
    assert kwargs["replica_id"] is None
    assert kwargs["return_full_response"] is True
    assert kwargs["replica_binds_sockets"] is False

    assert mock_manager_cls.call_count == 1
    mgr_kwargs = mock_manager_cls.call_args.kwargs
    assert mgr_kwargs["local_engine_count"] == 1
    assert mgr_kwargs["local_client"] is False
    assert mgr_kwargs["handshake_address"] == "tcp://127.0.0.1:26001"
    assert mgr_kwargs["omni_stage_id"] == 0
    assert mgr_kwargs["omni_coordinator_address"] == "tcp://127.0.0.1:26100"
    assert mgr_kwargs["omni_replica_base_id"] == 0

    engine_manager.monitor_engine_liveness.assert_called_once_with()
    engine_manager.shutdown.assert_called_once_with()


def test_run_headless_llm_launches_one_manager_per_omni_dp_size_local(mocker: MockerFixture) -> None:
    """``--omni-dp-size-local=N`` must spawn N managers, each with its own
    master-assigned replica_id, and join all of them before returning."""
    from vllm_omni.engine.stage_engine_startup import StageRegistrationResponse

    stage_cfg = _make_stage_cfg(0, stage_type="llm")
    parallel_config = SimpleNamespace(
        data_parallel_size_local=1,
        data_parallel_rank=0,
        data_parallel_rank_local=0,
        node_rank_within_dp=0,
    )
    vllm_config = SimpleNamespace(parallel_config=parallel_config, needs_dp_coordinator=False)
    manager_a = mocker.Mock()
    manager_b = mocker.Mock()

    mocker.patch(
        "vllm_omni.entrypoints.utils.load_and_resolve_stage_configs",
        return_value=("/fake/stages.yaml", [stage_cfg]),
    )
    mocker.patch("vllm_omni.engine.stage_init_utils.prepare_engine_environment")
    mocker.patch("vllm_omni.engine.stage_init_utils.load_omni_transfer_config_for_model", return_value=None)
    mocker.patch(
        "vllm_omni.distributed.omni_connectors.utils.initialization.resolve_omni_kv_config_for_stage",
        return_value=(None, None, None),
    )
    mocker.patch("vllm_omni.engine.stage_init_utils.get_stage_connector_spec", return_value={})
    mocker.patch("vllm_omni.engine.stage_init_utils.build_engine_args_dict", return_value={})
    mocker.patch(
        "vllm_omni.engine.stage_init_utils.build_vllm_config",
        return_value=(vllm_config, object),
    )
    mocker.patch(
        "vllm_omni.engine.stage_engine_startup.register_stage_with_omni_master",
        side_effect=[
            StageRegistrationResponse(
                handshake_address=f"tcp://127.0.0.1:2700{idx}",
                input_address=f"tcp://127.0.0.1:2710{idx}",
                output_address=f"tcp://127.0.0.1:2720{idx}",
                replica_id=idx,
                coordinator_router_address=None,
            )
            for idx in (0, 1)
        ],
    )
    mock_manager_cls = mocker.patch(
        "vllm_omni.engine.omni_core_engine_proc_manager.OmniCoreEngineProcManager",
        side_effect=[manager_a, manager_b],
    )
    mocker.patch("signal.signal")

    run_headless(_make_headless_args(stage_id=0, omni_dp_size_local=2))

    assert mock_manager_cls.call_count == 2
    assigned_ids = [call.kwargs["omni_replica_base_id"] for call in mock_manager_cls.call_args_list]
    assert assigned_ids == [0, 1]

    # Multi-replica path joins the monitor threads instead of calling
    # ``monitor_engine_liveness`` synchronously on the main thread, but every
    # manager must still be shut down in the finally block.
    manager_a.shutdown.assert_called_once_with()
    manager_b.shutdown.assert_called_once_with()


def test_run_headless_diffusion_registers_and_spawns_proc(mocker: MockerFixture) -> None:
    """Diffusion headless: registers as auto-assign, spawns a single
    ``StageDiffusionProc`` per local replica, and waits for it via
    ``multiprocessing.connection.wait``."""
    from vllm_omni.engine.stage_engine_startup import StageRegistrationResponse

    stage_cfg = _make_stage_cfg(1, stage_type="diffusion")
    od_config = mocker.Mock()
    proc = mocker.Mock(sentinel=object(), exitcode=0)
    proc.is_alive.return_value = False

    mocker.patch(
        "vllm_omni.entrypoints.utils.load_and_resolve_stage_configs",
        return_value=("/fake/stages.yaml", [stage_cfg]),
    )
    mocker.patch("vllm_omni.engine.stage_init_utils.prepare_engine_environment")
    mocker.patch("vllm_omni.engine.stage_init_utils.load_omni_transfer_config_for_model", return_value=None)
    mocker.patch(
        "vllm_omni.distributed.omni_connectors.utils.initialization.resolve_omni_kv_config_for_stage",
        return_value=(None, None, None),
    )
    mocker.patch(
        "vllm_omni.engine.stage_init_utils.extract_stage_metadata",
        return_value=SimpleNamespace(stage_id=1, stage_type="diffusion"),
    )
    mock_inject = mocker.patch("vllm_omni.engine.stage_init_utils.inject_kv_stage_info")
    mocker.patch("vllm_omni.engine.stage_init_utils.build_diffusion_config", return_value=od_config)
    mock_register = mocker.patch(
        "vllm_omni.engine.stage_engine_startup.register_stage_with_omni_master",
        return_value=StageRegistrationResponse(
            handshake_address="tcp://127.0.0.1:26001",
            input_address="tcp://127.0.0.1:26002",
            output_address="tcp://127.0.0.1:26003",
            replica_id=0,
            coordinator_router_address="tcp://127.0.0.1:26100",
        ),
    )
    mock_spawn = mocker.patch(
        "vllm_omni.diffusion.stage_diffusion_proc.spawn_diffusion_proc",
        return_value=(proc, None, None, None),
    )
    mock_handshake = mocker.patch("vllm_omni.diffusion.stage_diffusion_proc.complete_diffusion_handshake")
    # Replace the blocking wait with one that returns the only proc's sentinel
    # immediately so the test does not hang.
    mocker.patch(
        "multiprocessing.connection.wait",
        side_effect=lambda sentinels: [sentinels[0]],
    )
    mocker.patch("vllm_omni.engine.stage_init_utils.terminate_alive_proc")
    mocker.patch("signal.signal")

    run_headless(_make_headless_args(stage_id=1))

    mock_inject.assert_called_once()
    assert mock_inject.call_args.args[0] is stage_cfg
    assert mock_inject.call_args.args[1] == 1
    assert mock_inject.call_args.args[2] == [stage_cfg]

    reg_kwargs = mock_register.call_args.kwargs
    assert reg_kwargs["omni_master_address"] == "127.0.0.1"
    assert reg_kwargs["omni_master_port"] == 26000
    assert reg_kwargs["omni_stage_id"] == 1
    assert reg_kwargs["omni_stage_config"] is stage_cfg
    assert reg_kwargs["replica_id"] is None
    assert reg_kwargs["return_full_response"] is True

    spawn_kwargs = mock_spawn.call_args.kwargs
    assert spawn_kwargs["handshake_address"] == "tcp://127.0.0.1:26001"
    assert spawn_kwargs["request_address"] == "tcp://127.0.0.1:26002"
    assert spawn_kwargs["response_address"] == "tcp://127.0.0.1:26003"
    assert spawn_kwargs["omni_coordinator_address"] == "tcp://127.0.0.1:26100"
    assert spawn_kwargs["omni_stage_id"] == 1
    assert spawn_kwargs["omni_replica_id"] == 0

    mock_handshake.assert_called_once_with(proc, "tcp://127.0.0.1:26001", 600)


def test_run_headless_diffusion_raises_on_nonzero_proc_exit(mocker: MockerFixture) -> None:
    """A diffusion replica that exits with a non-zero code must surface as a
    RuntimeError from ``run_headless`` (the head needs the signal to roll
    back its own stage init)."""
    from vllm_omni.engine.stage_engine_startup import StageRegistrationResponse

    stage_cfg = _make_stage_cfg(1, stage_type="diffusion")
    proc = mocker.Mock(sentinel=object(), exitcode=137, name="proc-stage1-rep0")
    proc.is_alive.return_value = False

    mocker.patch(
        "vllm_omni.entrypoints.utils.load_and_resolve_stage_configs",
        return_value=("/fake/stages.yaml", [stage_cfg]),
    )
    mocker.patch("vllm_omni.engine.stage_init_utils.prepare_engine_environment")
    mocker.patch("vllm_omni.engine.stage_init_utils.load_omni_transfer_config_for_model", return_value=None)
    mocker.patch(
        "vllm_omni.distributed.omni_connectors.utils.initialization.resolve_omni_kv_config_for_stage",
        return_value=(None, None, None),
    )
    mocker.patch(
        "vllm_omni.engine.stage_init_utils.extract_stage_metadata",
        return_value=SimpleNamespace(stage_id=1, stage_type="diffusion"),
    )
    mocker.patch("vllm_omni.engine.stage_init_utils.inject_kv_stage_info")
    mocker.patch("vllm_omni.engine.stage_init_utils.build_diffusion_config", return_value=mocker.Mock())
    mocker.patch(
        "vllm_omni.engine.stage_engine_startup.register_stage_with_omni_master",
        return_value=StageRegistrationResponse(
            handshake_address="tcp://127.0.0.1:26001",
            input_address="tcp://127.0.0.1:26002",
            output_address="tcp://127.0.0.1:26003",
            replica_id=0,
            coordinator_router_address=None,
        ),
    )
    mocker.patch(
        "vllm_omni.diffusion.stage_diffusion_proc.spawn_diffusion_proc",
        return_value=(proc, None, None, None),
    )
    mocker.patch("vllm_omni.diffusion.stage_diffusion_proc.complete_diffusion_handshake")
    mocker.patch(
        "multiprocessing.connection.wait",
        side_effect=lambda sentinels: [sentinels[0]],
    )
    mocker.patch("vllm_omni.engine.stage_init_utils.terminate_alive_proc")
    mocker.patch("signal.signal")

    with pytest.raises(RuntimeError, match=r"exited with code 137"):
        run_headless(_make_headless_args(stage_id=1))
