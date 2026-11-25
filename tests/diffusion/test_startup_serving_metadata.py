# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real ZeroMQ and upstream startup handshake, without workers or weights."""

import os
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import pytest
import zmq
from vllm.v1.engine.utils import CoreEngine, CoreEngineLaunch, EngineZmqAddresses, wait_for_engine_startup

from vllm_omni.diffusion.startup_metadata import DiffusionStartupSocket
from vllm_omni.request_capabilities import ImageGeometry, RuntimeServingMetadata, ServingPostProcess

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("local", [True, False])
@pytest.mark.parametrize("multiple", [None, 8, 16, 32])
def test_ready_metadata_uses_existing_handshake_and_preserves_socket_ownership(local, multiple):
    metadata = (
        RuntimeServingMetadata(image_geometry=ImageGeometry(state="known", multiple_of=multiple))
        if multiple
        else RuntimeServingMetadata()
    )
    with zmq.Context() as context:
        with context.socket(zmq.ROUTER) as original, context.socket(zmq.DEALER) as worker:
            original.bind("inproc://startup-metadata")
            worker.setsockopt(zmq.IDENTITY, (0).to_bytes(2, "little"))
            worker.connect("inproc://startup-metadata")
            receiver = DiffusionStartupSocket.shadow(original)
            errors = []

            def send_ready():
                try:
                    worker.send(msgspec.msgpack.encode({"status": "HELLO", "local": local, "headless": not local}))
                    assert worker.poll(5000)
                    worker.recv()
                    worker.send(
                        msgspec.msgpack.encode(
                            {
                                "status": "READY",
                                "local": local,
                                "headless": not local,
                                "serving_metadata": metadata.model_dump(exclude_none=True),
                            }
                        )
                    )
                except Exception as error:
                    errors.append(error)

            thread = threading.Thread(target=send_ready)
            thread.start()
            wait_for_engine_startup(
                receiver,
                [CoreEngine(index=0, local=local)],
                SimpleNamespace(
                    data_parallel_size_local=int(local), data_parallel_hybrid_lb=False, data_parallel_external_lb=False
                ),
                False,
                None,
                CoreEngineLaunch(
                    engine_manager=None,
                    coordinator=None,
                    addresses=EngineZmqAddresses(inputs=["inproc://input"], outputs=["inproc://output"]),
                    tensor_queue=None,
                ),
            )
            thread.join(5)
            assert not thread.is_alive() and not errors
            assert receiver.serving_metadata == metadata
            del receiver
            assert not original.closed


def test_postprocessor_facts_do_not_change_processing_or_assume_a_vae():
    metadata = RuntimeServingMetadata(image_geometry=ImageGeometry(state="known", multiple_of=24))
    hook = ServingPostProcess(lambda value: value + 1, metadata)
    assert hook(2) == 3
    assert hook.serving_metadata.image_geometry.multiple_of == 24
    assert RuntimeServingMetadata().image_geometry is None


def _worker_ready(address, metadata, local, errors):
    """Only the wire peer is synthetic; both startup loops are unchanged."""
    from vllm_omni.diffusion.stage_diffusion_proc import StageDiffusionProc

    try:
        with zmq.Context() as context, context.socket(zmq.DEALER) as socket:
            socket.setsockopt(zmq.IDENTITY, b"\x00\x00")
            socket.connect(address)
            socket.send(msgspec.msgpack.encode({"status": "HELLO", "local": local, "headless": not local}))
            assert socket.poll(5000)
            socket.recv()
            StageDiffusionProc._send_startup_ready(
                socket,
                local_client=local,
                headless=not local,
                serving_metadata=metadata.model_dump(exclude_none=True),
            )
    except Exception as error:
        errors.append(error)


@pytest.mark.parametrize("local", [True, False])
def test_completed_local_and_remote_startup_deliver_metadata_to_real_client(local, monkeypatch):
    from vllm.utils.network_utils import get_open_zmq_ipc_path

    from vllm_omni.diffusion.stage_diffusion_client import StageDiffusionClient
    from vllm_omni.diffusion.stage_diffusion_proc import StageDiffusionProcManager
    from vllm_omni.engine.stage_engine_startup import connect_remote_diffusion_proc
    from vllm_omni.engine.stage_init_utils import StageMetadata

    runtime = RuntimeServingMetadata(image_geometry=ImageGeometry(state="known", multiple_of=24))
    address = get_open_zmq_ipc_path()
    addresses = EngineZmqAddresses(inputs=[get_open_zmq_ipc_path()], outputs=[get_open_zmq_ipc_path()])
    errors: list[Exception] = []
    worker = threading.Thread(target=_worker_ready, args=(address, runtime, local, errors), daemon=True)
    manager = None
    read_fd, write_fd = os.pipe()
    try:
        if local:
            manager = object.__new__(StageDiffusionProcManager)
            # An untriggered real descriptor represents a live model process.
            manager.proc = SimpleNamespace(sentinel=read_fd, name="fixture-worker", exitcode=None)
            manager.addresses = addresses
            worker.start()
            manager._wait_until_started(address, 5)
            assert manager.serving_metadata == runtime
            client_metadata = None
        else:
            master = SimpleNamespace(
                get_zmq_addresses=lambda *a, **kw: addresses,
                get_allocation=lambda *a, **kw: SimpleNamespace(handshake_bind_address=address),
                release_route_port_reservations=Mock(),
            )
            with connect_remote_diffusion_proc(master, 0) as resources:
                assert resources.serving_metadata is None
                worker.start()
            # It becomes visible only after the context's READY wait returns.
            assert resources.serving_metadata == runtime
            client_metadata = resources.serving_metadata
            assert master.release_route_port_reservations.call_count == 2
        worker.join(5)
        assert not worker.is_alive() and not errors
        # Disable only the fake process sentinel's background monitor. The
        # actual client transport and metadata construction still run.
        monkeypatch.setattr(StageDiffusionClient, "_start_proc_monitor", lambda self: None)
        metadata = StageMetadata(
            stage_id=0,
            stage_type="diffusion",
            engine_output_type="image",
            is_comprehension=False,
            requires_multimodal_data=False,
            engine_input_source=[],
            final_output=True,
            final_output_type="image",
            default_sampling_params={},
            custom_process_input_func=None,
            model_stage=None,
            runtime_cfg=None,
        )
        client = StageDiffusionClient.from_addresses(
            metadata,
            addresses.inputs[0],
            addresses.outputs[0],
            proc_manager=manager,
            serving_metadata=client_metadata,
        )
        try:
            assert client.serving_metadata == runtime
        finally:
            client._request_socket.close(linger=0)
            client._response_socket.close(linger=0)
            client._zmq_ctx.term()
    finally:
        os.close(read_fd)
        os.close(write_fd)


@pytest.mark.parametrize("channels,multiple", [(2, 4), (5, 32)])
def test_klein_geometry_comes_from_existing_local_postprocessor_initialization(tmp_path, channels, multiple):
    import json

    from vllm_omni.diffusion.models.flux2_klein.pipeline_flux2_klein import get_flux2_klein_post_process_func

    vae = tmp_path / "vae"
    vae.mkdir()
    (vae / "config.json").write_text(json.dumps({"block_out_channels": [128] * channels}))
    hook = get_flux2_klein_post_process_func(SimpleNamespace(model=str(tmp_path)))
    assert hook.serving_metadata.image_geometry.multiple_of == multiple
