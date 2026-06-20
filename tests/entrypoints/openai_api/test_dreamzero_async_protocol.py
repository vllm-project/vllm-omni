import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import msgspec
import numpy as np
import pytest

from vllm_omni.entrypoints.openpi import dreamzero_async_protocol as protocol
from vllm_omni.entrypoints.openpi.connection import DreamZeroAsyncRealtimeConnection
from vllm_omni.entrypoints.openpi.dreamzero_async_runtime import (
    CacheEntryKey,
    DeterministicFakeDreamZeroRunner,
    DreamZeroAsyncScheduler,
    DreamZeroAsyncSession,
    ForwardResult,
    ForwardTask,
    RealObservationEntry,
    ServingDreamZeroAsyncRunner,
)
from vllm_omni.entrypoints.openpi.serving import PolicyServerConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeWebSocket:
    def __init__(self, messages):
        self._messages = list(messages)
        self.sent_bytes = []
        self.sent_texts = []
        self.accepted = False
        self.closed = False

    async def accept(self):
        self.accepted = True

    async def send_bytes(self, data):
        self.sent_bytes.append(data)

    async def send_text(self, data):
        self.sent_texts.append(data)

    async def receive(self):
        return self._messages.pop(0)

    async def close(self):
        self.closed = True


class WaitForActionChunksWebSocket(FakeWebSocket):
    def __init__(self, messages, *, action_chunk_count):
        super().__init__(messages)
        self._action_chunk_count = action_chunk_count

    async def receive(self):
        if self._messages:
            return self._messages.pop(0)
        for _ in range(100):
            sent = _decode_sent(self)
            action_chunks = [message for message in sent if message.get("type") == "action_chunk"]
            if len(action_chunks) >= self._action_chunk_count:
                return {"type": "websocket.disconnect"}
            await asyncio.sleep(0.01)
        return {"type": "websocket.disconnect"}


class FailingForwardRunner(DeterministicFakeDreamZeroRunner):
    async def run_forward(self, task):
        raise RuntimeError("boom")


class CleanupTrackingRunner(DeterministicFakeDreamZeroRunner):
    def __init__(self):
        super().__init__()
        self.closed_sessions = []

    async def close_session(self, session_id):
        self.closed_sessions.append(session_id)


class FakeAsyncEngineClient:
    def __init__(self, output):
        self.output = output

    async def generate(self, **kwargs):
        yield self.output


class FakeDreamZeroServing:
    def __init__(self, output):
        self.engine_client = FakeAsyncEngineClient(output)

    def build_dreamzero_async_forward_request(self, *args, **kwargs):
        return SimpleNamespace(
            prompts=["pick"],
            request_id="req-1",
            sampling_params=SimpleNamespace(),
        )

    def _extract_actions(self, result):
        return result.multimodal_output["actions"]


def _serving_mock():
    serving = MagicMock()
    serving.policy_server_config = PolicyServerConfig(
        {
            "image_resolution": (180, 320),
            "n_external_cameras": 2,
            "needs_wrist_camera": True,
            "needs_stereo_camera": False,
            "needs_session_id": True,
            "action_space": "joint_position",
            "action_horizon": 24,
            "action_dim": 8,
        }
    )
    return serving


def _ws_message(payload):
    return {"type": "websocket.receive", "bytes": protocol.pack_message(payload)}


def _decode_sent(websocket):
    return [protocol.unpack_message(item) for item in websocket.sent_bytes]


def test_async_metadata_advertises_protocol_and_model_shape():
    metadata = protocol.make_metadata(_serving_mock().policy_server_config.to_dict())

    assert metadata["protocol"] == "dreamzero_async"
    assert metadata["protocol_version"] == 1
    assert metadata["lookahead_mode"] == "real_rebased_one_step_simulated"
    assert metadata["model"] == {"action_horizon": 24, "action_dim": 8}
    assert metadata["action_space"] == "joint_position"


def test_validate_session_start_rejects_unsupported_protocol_version():
    with pytest.raises(protocol.ProtocolValidationError) as exc_info:
        protocol.validate_session_start(
            {
                "type": "session_start",
                "protocol_version": 2,
                "session_id": "session-a",
            }
        )

    assert exc_info.value.code == "unsupported_protocol_version"


def test_action_chunk_schema_round_trips_numpy_actions():
    actions = np.arange(16, dtype=np.float32).reshape(2, 8)
    payload = {
        "type": "action_chunk",
        "session_id": "session-a",
        "session_epoch": 1,
        "chunk_index": 1,
        "actions": actions,
        "provenance": {"real_observations": [1], "sim_observations": [], "sim_depth": 0},
        "monitoring": {"forward_finished_s": 123.0},
    }

    decoded = protocol.validate_action_chunk(protocol.unpack_message(protocol.pack_message(payload)))

    np.testing.assert_allclose(decoded["actions"], actions)
    assert decoded["provenance"]["real_observations"] == [1]
    assert decoded["monitoring"]["forward_finished_s"] == 123.0


def test_connection_session_start_reset_and_scheduler_placeholder():
    websocket = FakeWebSocket(
        [
            _ws_message(
                {
                    "type": "session_start",
                    "protocol_version": 1,
                    "session_id": "session-a",
                    "prompt": "pick up the object",
                }
            ),
            _ws_message(
                {
                    "type": "observation_real",
                    "session_id": "session-a",
                    "session_epoch": 1,
                    "observation_index": 1,
                    "timestamp_s": 1.25,
                    "robot_obs": {"observation/joint_position": np.zeros(7, dtype=np.float32)},
                }
            ),
            _ws_message({"type": "session_reset", "session_id": "session-a", "session_epoch": 1}),
            {"type": "websocket.disconnect"},
        ]
    )

    asyncio.run(DreamZeroAsyncRealtimeConnection(websocket, _serving_mock()).handle_connection())

    sent = _decode_sent(websocket)
    assert websocket.accepted is True
    assert sent[0]["protocol"] == "dreamzero_async"
    assert sent[1] == {
        "type": "session_started",
        "session_id": "session-a",
        "session_epoch": 1,
        "lookahead_mode": "real_rebased_one_step_simulated",
    }
    assert sent[2]["type"] == "error"
    assert sent[2]["code"] == "scheduler_unavailable"
    assert sent[2]["session_id"] == "session-a"
    assert sent[2]["session_epoch"] == 1
    assert sent[3] == {
        "type": "session_reset_ack",
        "session_id": "session-a",
        "session_epoch": 2,
    }
    assert websocket.sent_texts == []


def test_connection_rejects_malformed_payload_without_traceback():
    websocket = FakeWebSocket(
        [
            {"type": "websocket.receive", "bytes": msgspec.msgpack.encode(["bad"])},
            {"type": "websocket.disconnect"},
        ]
    )

    asyncio.run(DreamZeroAsyncRealtimeConnection(websocket, _serving_mock()).handle_connection())

    sent = _decode_sent(websocket)
    assert sent[1]["type"] == "error"
    assert sent[1]["code"] == "invalid_payload"
    assert "traceback" not in sent[1]["message"].lower()


def test_fake_scheduler_bootstrap_and_real_rebased_step_sequence():
    async def run_sequence():
        harness = DreamZeroAsyncSession(
            runner=DeterministicFakeDreamZeroRunner(action_horizon=2, action_dim=8)
        )
        await harness.start_session(session_id="session-a", session_epoch=1, prompt="pick")

        bootstrap = await harness.handle_observation(
            {
                "observation_index": 1,
                "robot_obs": {"observation/joint_position": np.zeros(7, dtype=np.float32)},
            }
        )
        after_real_o2 = await harness.handle_observation(
            {
                "observation_index": 2,
                "robot_obs": {"observation/joint_position": np.ones(7, dtype=np.float32)},
            }
        )
        return bootstrap, after_real_o2

    bootstrap, after_real_o2 = asyncio.run(run_sequence())

    assert [message["chunk_index"] for message in bootstrap] == [1, 2]
    assert bootstrap[0]["provenance"]["real_observations"] == [1]
    assert bootstrap[0]["provenance"]["sim_observations"] == []
    assert bootstrap[1]["provenance"]["real_observations"] == [1]
    assert bootstrap[1]["provenance"]["sim_observations"] == [2]
    assert [message["chunk_index"] for message in after_real_o2] == [3]
    assert after_real_o2[0]["provenance"]["real_observations"] == [1, 2]
    assert after_real_o2[0]["provenance"]["sim_observations"] == [3]


def test_fake_scheduler_submit_observation_does_not_wait_for_forward_completion():
    async def run_sequence():
        harness = DreamZeroAsyncSession(
            runner=DeterministicFakeDreamZeroRunner(
                action_horizon=2,
                action_dim=8,
                forward_delay_s=0.05,
            )
        )
        await harness.start_session(session_id="session-a", session_epoch=1, prompt="pick")

        await harness.submit_observation(
            {
                "observation_index": 1,
                "robot_obs": {"observation/joint_position": np.zeros(7, dtype=np.float32)},
            }
        )
        immediate = harness.drain_ready_outbound()
        await harness.wait_for_idle()
        finished = harness.drain_ready_outbound()
        return immediate, finished

    immediate, finished = asyncio.run(run_sequence())

    assert immediate == []
    assert [message["chunk_index"] for message in finished] == [1, 2]


def test_fake_scheduler_uses_late_sim_observation_for_missing_action():
    async def run_sequence():
        harness = DreamZeroAsyncSession(
            runner=DeterministicFakeDreamZeroRunner(
                action_horizon=2,
                action_dim=8,
                forward_delay_s=0.01,
            )
        )
        await harness.start_session(session_id="session-a", session_epoch=1, prompt="pick")

        await harness.handle_observation(
            {
                "observation_index": 1,
                "robot_obs": {"observation/joint_position": np.zeros(7, dtype=np.float32)},
            }
        )
        await harness.submit_observation(
            {
                "observation_index": 2,
                "robot_obs": {"observation/joint_position": np.ones(7, dtype=np.float32)},
            }
        )
        await harness.submit_observation(
            {
                "observation_index": 3,
                "robot_obs": {"observation/joint_position": np.full(7, 2.0, dtype=np.float32)},
            }
        )
        await harness.wait_for_idle()
        return harness.drain_ready_outbound()

    messages = asyncio.run(run_sequence())
    action_chunks = [message for message in messages if message["type"] == "action_chunk"]

    chunk3 = next(message for message in action_chunks if message["chunk_index"] == 3)
    assert chunk3["provenance"]["real_observations"] == [1, 2]
    assert chunk3["provenance"]["sim_observations"] == [3]


def test_fake_scheduler_caps_real_prefix_for_sim_conditioned_action():
    async def run_sequence():
        harness = DreamZeroAsyncSession(
            runner=DeterministicFakeDreamZeroRunner(action_horizon=2, action_dim=8)
        )
        await harness.start_session(session_id="session-a", session_epoch=1, prompt="pick")

        messages = []
        for observation_index in range(1, 6):
            messages.extend(
                await harness.handle_observation(
                    {
                        "observation_index": observation_index,
                        "robot_obs": {
                            "observation/joint_position": np.full(
                                7,
                                float(observation_index),
                                dtype=np.float32,
                            )
                        },
                    }
                )
            )
        return messages

    messages = asyncio.run(run_sequence())
    action_chunks = [message for message in messages if message["type"] == "action_chunk"]

    chunk5 = next(message for message in action_chunks if message["chunk_index"] == 5)
    assert chunk5["provenance"]["real_observations"] == [2, 3, 4]
    assert chunk5["provenance"]["sim_observations"] == [5]


def test_fake_scheduler_reports_forward_failure_without_hanging():
    async def run_sequence():
        harness = DreamZeroAsyncSession(runner=FailingForwardRunner())
        await harness.start_session(session_id="session-a", session_epoch=1, prompt="pick")

        await harness.submit_observation(
            {
                "observation_index": 1,
                "robot_obs": {"observation/joint_position": np.zeros(7, dtype=np.float32)},
            }
        )
        await harness.wait_for_idle()
        return harness.drain_ready_outbound()

    messages = asyncio.run(run_sequence())

    assert len(messages) == 1
    assert messages[0]["type"] == "error"
    assert messages[0]["code"] == "forward_failed"
    assert messages[0]["session_id"] == "session-a"
    assert messages[0]["session_epoch"] == 1


def test_fake_scheduler_close_cleans_up_runner_session():
    async def run_sequence():
        runner = CleanupTrackingRunner()
        harness = DreamZeroAsyncSession(runner=runner)
        await harness.start_session(session_id="session-a", session_epoch=1, prompt="pick")
        await harness.close()
        return runner.closed_sessions

    assert asyncio.run(run_sequence()) == ["session-a"]


def test_fake_scheduler_reset_cleans_up_old_runner_session():
    async def run_sequence():
        runner = CleanupTrackingRunner()
        harness = DreamZeroAsyncSession(runner=runner)
        await harness.start_session(session_id="session-a", session_epoch=1, prompt="pick")
        await harness.reset(session_epoch=2)
        return runner.closed_sessions, harness.scheduler

    closed_sessions, runtime = asyncio.run(run_sequence())

    assert closed_sessions == ["session-a"]
    assert runtime is not None
    assert runtime.session_id == "session-a"
    assert runtime.session_epoch == 2


def test_runtime_deduplicates_duplicate_observation_commits():
    runtime = DreamZeroAsyncScheduler(session_id="session-a", session_epoch=1, prompt="pick")
    first = runtime.on_observation_received(observation_index=1, robot_obs={})
    duplicate_while_committing = runtime.on_observation_received(observation_index=1, robot_obs={})

    assert len(first.observation_commits) == 1
    assert duplicate_while_committing.observation_commits == ()

    runtime.on_observation_committed(first.observation_commits[0])
    duplicate_after_commit = runtime.on_observation_received(observation_index=1, robot_obs={})

    assert duplicate_after_commit.observation_commits == ()


def test_runtime_discards_stale_epoch_forward_completion():
    runtime = DreamZeroAsyncScheduler(session_id="session-a", session_epoch=1, prompt="pick")
    key = CacheEntryKey(
        session_id="session-a",
        session_epoch=1,
        observation_index=1,
        sim_depth=0,
    )
    runtime.real_observations[1] = RealObservationEntry(
        robot_obs={},
        key=key,
        commit_in_flight=False,
        kv_committed=True,
    )
    task = ForwardTask(prefix_keys=(key,), send_action=True, commit_sim_kv=True)
    runtime.try_submit_forward(task)

    runtime.session_epoch = 2
    update = runtime.on_forward_done(
        task,
        ForwardResult(
            chunk_index=1,
            actions=np.zeros((2, 8), dtype=np.float32),
            produced_sim_observation=2,
        ),
    )

    assert update.outbound_messages == ()


def test_forward_task_engine_metadata_uses_plain_cache_key_dicts():
    o1 = CacheEntryKey("session-a", 1, 1, 0)
    task = ForwardTask(prefix_keys=(o1,), send_action=True, commit_sim_kv=True)

    assert task.engine_metadata() == {
        "prefix_keys": [],
        "input_key": {"session_id": "session-a", "session_epoch": 1, "observation_index": 1, "sim_depth": 0},
    }

    o2_sim = CacheEntryKey("session-a", 1, 2, 1)
    lookahead = ForwardTask(prefix_keys=(o1, o2_sim), send_action=True, commit_sim_kv=False)

    assert lookahead.engine_metadata() == {
        "prefix_keys": [{"session_id": "session-a", "session_epoch": 1, "observation_index": 1, "sim_depth": 0}],
        "input_key": {"session_id": "session-a", "session_epoch": 1, "observation_index": 2, "sim_depth": 1},
    }


def test_forward_task_engine_metadata_batches_owner_drops():
    o1 = CacheEntryKey("session-a", 1, 1, 0)
    o2 = CacheEntryKey("session-a", 1, 2, 0)
    o2_sim = CacheEntryKey("session-a", 1, 2, 1)
    task = ForwardTask(
        prefix_keys=(o2,),
        send_action=False,
        commit_sim_kv=True,
        drop_owner_key=o2_sim,
        drop_owner_keys=(o1,),
    )

    assert task.engine_metadata()["drop_owner_keys"] == [
        {"session_id": "session-a", "session_epoch": 1, "observation_index": 2, "sim_depth": 1},
        {"session_id": "session-a", "session_epoch": 1, "observation_index": 1, "sim_depth": 0},
    ]


def test_runtime_can_queue_action_and_real_materialization_forwards():
    runtime = DreamZeroAsyncScheduler(session_id="session-a", session_epoch=1, prompt="pick")
    o1 = runtime.on_observation_received(observation_index=1, robot_obs={}).observation_commits[0]
    update = runtime.on_observation_committed(o1)
    assert len(update.forwards) == 1

    o2_update = runtime.on_observation_received(observation_index=2, robot_obs={})
    assert len(o2_update.observation_commits) == 1
    assert o2_update.forwards == ()
    assert runtime.on_observation_committed(o2_update.observation_commits[0]).forwards == ()

    forward_done = runtime.on_forward_done(
        update.forwards[0],
        ForwardResult(
            chunk_index=1,
            actions=np.zeros((2, 8), dtype=np.float32),
            produced_sim_observation=2,
            produced_video=np.zeros((1, 16, 2, 1, 1), dtype=np.float32),
        ),
    )

    assert len(forward_done.forwards) == 2
    assert forward_done.forwards[0].prefix_keys[-1].observation_index == 2
    assert forward_done.forwards[0].prefix_keys[-1].sim_depth == 1
    assert forward_done.forwards[0].send_action is True
    assert forward_done.forwards[1].prefix_keys[-1].observation_index == 2
    assert forward_done.forwards[1].prefix_keys[-1].sim_depth == 0
    assert forward_done.forwards[1].send_action is False


def test_connection_with_fake_scheduler_publishes_bootstrap_actions():
    websocket = WaitForActionChunksWebSocket(
        [
            _ws_message(
                {
                    "type": "session_start",
                    "protocol_version": 1,
                    "session_id": "session-a",
                    "prompt": "pick up the object",
                }
            ),
            _ws_message(
                {
                    "type": "observation_real",
                    "session_id": "session-a",
                    "session_epoch": 1,
                    "observation_index": 1,
                    "timestamp_s": 1.25,
                    "robot_obs": {"observation/joint_position": np.zeros(7, dtype=np.float32)},
                }
            ),
        ],
        action_chunk_count=2,
    )
    scheduler = DreamZeroAsyncSession(
        runner=DeterministicFakeDreamZeroRunner(action_horizon=2, action_dim=8)
    )

    asyncio.run(
        DreamZeroAsyncRealtimeConnection(
            websocket,
            _serving_mock(),
            scheduler=scheduler,
        ).handle_connection()
    )

    sent = _decode_sent(websocket)
    action_chunks = [message for message in sent if message.get("type") == "action_chunk"]
    assert [message["chunk_index"] for message in action_chunks] == [1, 2]
    np.testing.assert_allclose(action_chunks[0]["actions"], np.ones((2, 8), dtype=np.float32))
    np.testing.assert_allclose(action_chunks[1]["actions"], np.full((2, 8), 2.0, dtype=np.float32))


def test_serving_runner_reads_async_video_from_custom_output():
    async def run_sequence():
        actions = np.ones((2, 8), dtype=np.float32)
        video = np.ones((1, 16, 2, 4, 4), dtype=np.float32)
        output = SimpleNamespace(
            multimodal_output={"actions": actions},
            custom_output={"dreamzero_async_video": video},
        )
        runner = ServingDreamZeroAsyncRunner(FakeDreamZeroServing(output))
        key = CacheEntryKey("session-a", 1, 1, 0)
        return await runner.run_forward(
            ForwardTask(
                prefix_keys=(key,),
                send_action=True,
                commit_sim_kv=True,
                robot_obs={"observation/joint_position": np.zeros(7, dtype=np.float32)},
            )
        )

    result = asyncio.run(run_sequence())

    assert result.produced_sim_observation == 2
    np.testing.assert_allclose(result.produced_video, np.ones((1, 16, 2, 4, 4), dtype=np.float32))
