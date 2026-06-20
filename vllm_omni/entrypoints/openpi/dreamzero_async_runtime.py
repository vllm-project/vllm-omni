# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Serving-session runtime harness for DreamZero W8 async control."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from vllm.logger import init_logger
from vllm_omni.experimental.bde.kv_cache import (
    BDECacheEntryKey as CacheEntryKey,
    bde_cache_entry_key_dict,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class ObservationCommitTask:
    key: CacheEntryKey
    robot_obs: dict[str, Any]


@dataclass(frozen=True)
class DropSimOwnerTask:
    key: CacheEntryKey


@dataclass(frozen=True)
class ForwardTask:
    prefix_keys: tuple[CacheEntryKey, ...]
    send_action: bool
    commit_sim_kv: bool
    robot_obs: dict[str, Any] | None = None
    latent_video: Any | None = None
    drop_owner_key: CacheEntryKey | None = None

    @property
    def session_epoch(self) -> int:
        return self.prefix_keys[-1].session_epoch

    @property
    def session_id(self) -> str:
        return self.prefix_keys[-1].session_id

    def engine_metadata(self) -> dict[str, Any]:
        prefix_keys = self.prefix_keys[:-1]
        input_key = self.prefix_keys[-1]
        metadata = {
            "prefix_keys": [bde_cache_entry_key_dict(key) for key in prefix_keys],
            "input_key": bde_cache_entry_key_dict(input_key),
        }
        if self.drop_owner_key is not None:
            metadata["drop_owner_key"] = bde_cache_entry_key_dict(self.drop_owner_key)
        return metadata


@dataclass(frozen=True)
class ForwardResult:
    chunk_index: int
    actions: Any
    produced_sim_observation: int
    produced_video: Any | None = None
    q_spec: dict[str, Any] | None = None
    monitoring: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeUpdate:
    observation_commits: tuple[ObservationCommitTask, ...] = ()
    drop_sim_owner_tasks: tuple[DropSimOwnerTask, ...] = ()
    forwards: tuple[ForwardTask, ...] = ()
    outbound_messages: tuple[dict[str, Any], ...] = ()


TaskKey = tuple[int, tuple[CacheEntryKey, ...], bool, bool]


class SessionRuntime:
    def __init__(self, *, session_id: str, session_epoch: int, prompt: str) -> None:
        self.session_id = session_id
        self.session_epoch = session_epoch
        self.prompt = prompt
        self.arrived_real_indices: set[int] = set()
        self.real_obs_by_index: dict[int, dict[str, Any]] = {}
        self.committed_real_keys: dict[int, CacheEntryKey] = {}
        self.committing_real_indices: set[int] = set()
        self.admitted_sim_by_index: dict[int, CacheEntryKey] = {}
        self.sim_obs_by_index: dict[int, dict[str, Any]] = {}
        self.sim_video_by_index: dict[int, Any] = {}
        self.dropped_sim_owner_keys: set[CacheEntryKey] = set()
        self.pending_drop_owner_by_index: dict[int, CacheEntryKey] = {}
        self.running_tasks: set[TaskKey] = set()
        self.completed_tasks: set[TaskKey] = set()
        self.published_chunks: set[int] = set()

    def on_observation_received(
        self,
        *,
        observation_index: int,
        robot_obs: dict[str, Any],
    ) -> RuntimeUpdate:
        self.arrived_real_indices.add(observation_index)
        self.real_obs_by_index[observation_index] = robot_obs
        drop_tasks: list[DropSimOwnerTask] = []
        sim_key = self.admitted_sim_by_index.get(observation_index)
        if sim_key is not None and sim_key not in self.dropped_sim_owner_keys:
            self.dropped_sim_owner_keys.add(sim_key)
            self.pending_drop_owner_by_index[observation_index] = sim_key
            drop_tasks.append(DropSimOwnerTask(sim_key))

        if observation_index in self.committed_real_keys or observation_index in self.committing_real_indices:
            return RuntimeUpdate(drop_sim_owner_tasks=tuple(drop_tasks))

        key = CacheEntryKey(
            session_id=self.session_id,
            session_epoch=self.session_epoch,
            observation_index=observation_index,
            sim_depth=0,
        )
        self.committing_real_indices.add(observation_index)
        return RuntimeUpdate(
            observation_commits=(ObservationCommitTask(key=key, robot_obs=robot_obs),),
            drop_sim_owner_tasks=tuple(drop_tasks),
        )

    def on_observation_committed(self, task: ObservationCommitTask) -> RuntimeUpdate:
        if self._is_stale_key(task.key):
            return RuntimeUpdate()
        index = task.key.observation_index
        self.committing_real_indices.discard(index)
        self.committed_real_keys[index] = task.key
        drop_owner_key = self.pending_drop_owner_by_index.pop(index, None)

        if index == 1:
            forward = ForwardTask(
                prefix_keys=(task.key,),
                send_action=True,
                commit_sim_kv=True,
                robot_obs=task.robot_obs,
                drop_owner_key=drop_owner_key,
            )
            return RuntimeUpdate(forwards=self.try_submit_forward(forward))

        real_prefix = self._real_prefix_through(index)
        if real_prefix is None:
            return RuntimeUpdate()
        forward = ForwardTask(
            prefix_keys=real_prefix,
            send_action=False,
            commit_sim_kv=True,
            robot_obs=task.robot_obs,
            drop_owner_key=drop_owner_key,
        )
        return RuntimeUpdate(forwards=self.try_submit_forward(forward))

    def on_forward_done(self, task: ForwardTask, result: ForwardResult) -> RuntimeUpdate:
        if task.session_epoch != self.session_epoch:
            return RuntimeUpdate()
        task_key = self._task_key(task)
        self.running_tasks.discard(task_key)
        self.completed_tasks.add(task_key)

        messages: list[dict[str, Any]] = []
        if task.send_action and result.chunk_index not in self.published_chunks:
            self.published_chunks.add(result.chunk_index)
            logger.info(
                "DreamZero async publishing action chunk=%d prefix=%s produced_sim=%d",
                result.chunk_index,
                self._prefix_label(task.prefix_keys),
                result.produced_sim_observation,
            )
            messages.append(self._action_chunk_message(task, result))

        if not task.commit_sim_kv:
            return RuntimeUpdate(outbound_messages=tuple(messages))

        sim_key = CacheEntryKey(
            session_id=self.session_id,
            session_epoch=self.session_epoch,
            observation_index=result.produced_sim_observation,
            sim_depth=1,
        )
        if sim_key.observation_index in self.arrived_real_indices:
            if sim_key not in self.dropped_sim_owner_keys:
                self.dropped_sim_owner_keys.add(sim_key)
                return RuntimeUpdate(
                    drop_sim_owner_tasks=(DropSimOwnerTask(sim_key),),
                    outbound_messages=tuple(messages),
                )
            return RuntimeUpdate(outbound_messages=tuple(messages))

        if result.produced_video is None:
            logger.info(
                "DreamZero async forward chunk=%d did not produce simulated video; no lookahead scheduled",
                result.chunk_index,
            )
            return RuntimeUpdate(outbound_messages=tuple(messages))

        self.admitted_sim_by_index[sim_key.observation_index] = sim_key
        self.sim_video_by_index[sim_key.observation_index] = result.produced_video
        self.sim_obs_by_index[sim_key.observation_index] = self._make_sim_robot_obs(task, result)
        real_prefix = self._real_prefix_through(result.chunk_index)
        if real_prefix is None:
            return RuntimeUpdate(outbound_messages=tuple(messages))
        forward = ForwardTask(
            prefix_keys=real_prefix + (sim_key,),
            send_action=True,
            commit_sim_kv=False,
            robot_obs=self.sim_obs_by_index[sim_key.observation_index],
            latent_video=result.produced_video,
        )
        return RuntimeUpdate(
            forwards=self.try_submit_forward(forward),
            outbound_messages=tuple(messages),
        )

    def on_observation_commit_failed(self, task: ObservationCommitTask, message: str) -> RuntimeUpdate:
        if self._is_stale_key(task.key):
            return RuntimeUpdate()
        self.committing_real_indices.discard(task.key.observation_index)
        return RuntimeUpdate(outbound_messages=(self._error_message("observation_commit_failed", message),))

    def on_forward_failed(self, task: ForwardTask, message: str) -> RuntimeUpdate:
        if task.session_epoch != self.session_epoch:
            return RuntimeUpdate()
        self.running_tasks.discard(self._task_key(task))
        return RuntimeUpdate(outbound_messages=(self._error_message("forward_failed", message),))

    def try_submit_forward(self, task: ForwardTask) -> tuple[ForwardTask, ...]:
        if not self._prefix_available(task.prefix_keys):
            return ()
        task_key = self._task_key(task)
        if task_key in self.running_tasks or task_key in self.completed_tasks:
            logger.debug(
                "DreamZero async skip duplicate forward prefix=%s send_action=%s commit_sim_kv=%s",
                self._prefix_label(task.prefix_keys),
                task.send_action,
                task.commit_sim_kv,
            )
            return ()
        self.running_tasks.add(task_key)
        logger.info(
            "DreamZero async schedule forward prefix=%s send_action=%s commit_sim_kv=%s",
            self._prefix_label(task.prefix_keys),
            task.send_action,
            task.commit_sim_kv,
        )
        return (task,)

    def _real_prefix_through(self, observation_index: int) -> tuple[CacheEntryKey, ...] | None:
        keys = []
        for index in range(1, observation_index + 1):
            key = self.committed_real_keys.get(index)
            if key is None:
                return None
            keys.append(key)
        return tuple(keys)

    def _prefix_available(self, prefix_keys: tuple[CacheEntryKey, ...]) -> bool:
        for key in prefix_keys:
            if self._is_stale_key(key):
                return False
            if key.sim_depth == 0 and self.committed_real_keys.get(key.observation_index) != key:
                return False
            if key.sim_depth == 1 and self.admitted_sim_by_index.get(key.observation_index) != key:
                return False
        return True

    def _is_stale_key(self, key: CacheEntryKey) -> bool:
        return key.session_id != self.session_id or key.session_epoch != self.session_epoch

    def _action_chunk_message(self, task: ForwardTask, result: ForwardResult) -> dict[str, Any]:
        real_observations = [key.observation_index for key in task.prefix_keys if key.sim_depth == 0]
        sim_observations = [key.observation_index for key in task.prefix_keys if key.sim_depth > 0]
        return {
            "type": "action_chunk",
            "session_id": self.session_id,
            "session_epoch": self.session_epoch,
            "chunk_index": result.chunk_index,
            "actions": result.actions,
            "provenance": {
                "real_observations": real_observations,
                "sim_observations": sim_observations,
                "sim_depth": max((key.sim_depth for key in task.prefix_keys), default=0),
                "q_spec": result.q_spec,
                "produced_sim_observation": result.produced_sim_observation,
            },
            "monitoring": dict(result.monitoring),
        }

    def _error_message(self, code: str, message: str) -> dict[str, Any]:
        return {
            "type": "error",
            "code": code,
            "message": message,
            "session_id": self.session_id,
            "session_epoch": self.session_epoch,
        }

    def _make_sim_robot_obs(self, task: ForwardTask, result: ForwardResult) -> dict[str, Any]:
        base = self.real_obs_by_index.get(result.chunk_index)
        if base is None:
            base = task.robot_obs or {}
        sim_obs = dict(base)
        actions = np.asarray(result.actions)
        if actions.ndim >= 2 and actions.shape[0] > 0:
            terminal = actions[-1]
            if terminal.shape[0] >= 7:
                sim_obs["observation/joint_position"] = np.asarray(terminal[:7], dtype=np.float32)
            if terminal.shape[0] >= 8:
                sim_obs["observation/gripper_position"] = np.asarray([terminal[7]], dtype=np.float32)
        return sim_obs

    @staticmethod
    def _task_key(task: ForwardTask) -> TaskKey:
        return (task.session_epoch, task.prefix_keys, task.send_action, task.commit_sim_kv)

    @staticmethod
    def _prefix_label(keys: tuple[CacheEntryKey, ...]) -> str:
        return ",".join(f"O{key.observation_index}:sim{key.sim_depth}" for key in keys)


class DeterministicFakeDreamZeroRunner:
    def __init__(
        self,
        *,
        action_horizon: int = 24,
        action_dim: int = 8,
        forward_delay_s: float = 0.0,
    ) -> None:
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.forward_delay_s = forward_delay_s

    async def commit_observation(self, task: ObservationCommitTask) -> ObservationCommitTask:
        await asyncio.sleep(0)
        return task

    async def drop_sim_owner(self, task: DropSimOwnerTask) -> DropSimOwnerTask:
        await asyncio.sleep(0)
        return task

    async def run_forward(self, task: ForwardTask) -> ForwardResult:
        started_s = time.monotonic()
        if self.forward_delay_s > 0:
            await asyncio.sleep(self.forward_delay_s)
        await asyncio.sleep(0)
        chunk_index = task.prefix_keys[-1].observation_index
        actions = np.full(
            (self.action_horizon, self.action_dim),
            float(chunk_index),
            dtype=np.float32,
        )
        q_spec = None
        if any(key.sim_depth > 0 for key in task.prefix_keys):
            q_spec = {
                "joint_position": actions[-1, :7].copy(),
                "gripper_position": actions[-1, 7].copy() if self.action_dim > 7 else None,
            }
        return ForwardResult(
            chunk_index=chunk_index,
            actions=actions,
            produced_sim_observation=chunk_index + 1,
            produced_video=np.zeros((1, 16, 4, 1, 1), dtype=np.float32),
            q_spec=q_spec,
            monitoring={
                "forward_started_s": started_s,
                "forward_finished_s": time.monotonic(),
            },
        )


class ServingDreamZeroAsyncRunner:
    def __init__(self, serving: Any) -> None:
        self.serving = serving

    async def close_session(self, session_id: str) -> None:
        await self.serving.cleanup_dreamzero_session(session_id)

    async def commit_observation(self, task: ObservationCommitTask) -> ObservationCommitTask:
        await asyncio.sleep(0)
        return task

    async def drop_sim_owner(self, task: DropSimOwnerTask) -> DropSimOwnerTask:
        await asyncio.sleep(0)
        return task

    async def run_forward(self, task: ForwardTask) -> ForwardResult:
        if task.robot_obs is None:
            raise RuntimeError("DreamZero async forward requires robot_obs")

        started_s = time.monotonic()
        input_key = task.prefix_keys[-1]
        request = self.serving.build_dreamzero_async_forward_request(
            task.robot_obs,
            session_id=task.session_id,
            reset=input_key.observation_index == 1 and input_key.sim_depth == 0,
            prefix_keys=task.prefix_keys[:-1],
            input_key=input_key,
            drop_owner_key=task.drop_owner_key,
            latent_video=task.latent_video,
        )

        result = None
        async for output in self.serving.engine_client.generate(
            prompt=request.prompts[0],
            request_id=request.request_id,
            sampling_params_list=[request.sampling_params],
        ):
            result = output
        if result is None:
            raise RuntimeError("DreamZero async forward produced no output")

        actions = self.serving._extract_actions(result)
        multimodal_output = getattr(result, "multimodal_output", {}) or {}
        custom_output = getattr(result, "custom_output", {}) or {}
        produced_video = None
        if isinstance(custom_output, dict):
            produced_video = custom_output.get("dreamzero_async_video")
        if produced_video is None and isinstance(multimodal_output, dict):
            produced_video = multimodal_output.get("video")
        actions_array = np.asarray(actions)
        q_spec = None
        if any(key.sim_depth > 0 for key in task.prefix_keys) and actions_array.ndim >= 2 and actions_array.shape[0] > 0:
            terminal = actions_array[-1]
            q_spec = {
                "joint_position": np.asarray(terminal[:7], dtype=np.float32),
                "gripper_position": np.asarray([terminal[7]], dtype=np.float32) if terminal.shape[0] > 7 else None,
            }
        return ForwardResult(
            chunk_index=input_key.observation_index,
            actions=actions,
            produced_sim_observation=input_key.observation_index + 1,
            produced_video=produced_video,
            q_spec=q_spec,
            monitoring={
                "forward_started_s": started_s,
                "forward_finished_s": time.monotonic(),
            },
        )


class DreamZeroAsyncSessionHarness:
    def __init__(
        self,
        *,
        runner: DeterministicFakeDreamZeroRunner | None = None,
    ) -> None:
        self.runner = runner or DeterministicFakeDreamZeroRunner()
        self.runtime: SessionRuntime | None = None
        self._outbound: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._tasks: set[asyncio.Task[None]] = set()

    async def start_session(self, *, session_id: str, session_epoch: int, prompt: str) -> None:
        await self._cancel_tasks()
        self._drain_outbound()
        self.runtime = SessionRuntime(
            session_id=session_id,
            session_epoch=session_epoch,
            prompt=prompt,
        )

    async def reset(self, *, session_epoch: int) -> None:
        if self.runtime is None:
            return
        old_session_id = self.runtime.session_id
        old_prompt = self.runtime.prompt
        await self._cancel_tasks()
        await self._close_runner_session(old_session_id)
        self._drain_outbound()
        self.runtime = SessionRuntime(
            session_id=old_session_id,
            session_epoch=session_epoch,
            prompt=old_prompt,
        )

    async def close(self) -> None:
        session_id = self.runtime.session_id if self.runtime is not None else None
        await self._cancel_tasks()
        if session_id is not None:
            await self._close_runner_session(session_id)
        self.runtime = None
        self._drain_outbound()

    async def submit_observation(self, request: dict[str, Any]) -> None:
        if self.runtime is None:
            return

        update = self.runtime.on_observation_received(
            observation_index=request["observation_index"],
            robot_obs=request["robot_obs"],
        )
        self._schedule_update(update)

    async def get_outbound(self) -> dict[str, Any]:
        return await self._outbound.get()

    async def handle_observation(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        await self.submit_observation(request)
        await self.wait_for_idle()
        return self.drain_ready_outbound()

    async def wait_for_idle(self) -> None:
        while self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)

    def drain_ready_outbound(self) -> list[dict[str, Any]]:
        outbound: list[dict[str, Any]] = []
        while not self._outbound.empty():
            outbound.append(self._outbound.get_nowait())
        return outbound

    def _schedule_update(self, update: RuntimeUpdate) -> None:
        for message in update.outbound_messages:
            self._outbound.put_nowait(message)
        for drop_task in update.drop_sim_owner_tasks:
            self._track(asyncio.create_task(self._run_drop_sim_owner(drop_task)))
        for commit_task in update.observation_commits:
            self._track(asyncio.create_task(self._run_observation_commit(commit_task)))
        for forward_task in update.forwards:
            self._track(asyncio.create_task(self._run_forward(forward_task)))

    async def _run_drop_sim_owner(self, task: DropSimOwnerTask) -> None:
        try:
            await self.runner.drop_sim_owner(task)
        except Exception:
            logger.exception("DreamZero async failed to release simulated cache owner")
            await self._publish_task_error("drop_sim_owner_failed", "DreamZero async failed to release simulated cache owner")

    async def _run_observation_commit(self, task: ObservationCommitTask) -> None:
        try:
            committed = await self.runner.commit_observation(task)
        except Exception:
            logger.exception("DreamZero async failed to commit real observation")
            if self.runtime is not None:
                self._schedule_update(
                    self.runtime.on_observation_commit_failed(
                        task,
                        "DreamZero async failed to commit real observation",
                    )
                )
            return
        if self.runtime is None:
            return
        self._schedule_update(self.runtime.on_observation_committed(committed))

    async def _run_forward(self, task: ForwardTask) -> None:
        try:
            result = await self.runner.run_forward(task)
        except Exception:
            logger.exception(
                "DreamZero async forward failed prefix=%s send_action=%s commit_sim_kv=%s",
                SessionRuntime._prefix_label(task.prefix_keys),
                task.send_action,
                task.commit_sim_kv,
            )
            if self.runtime is not None:
                self._schedule_update(
                    self.runtime.on_forward_failed(
                        task,
                        "DreamZero async forward failed",
                    )
                )
            return
        if self.runtime is None:
            return
        self._schedule_update(self.runtime.on_forward_done(task, result))

    async def _publish_task_error(self, code: str, message: str) -> None:
        if self.runtime is None:
            return
        self._outbound.put_nowait(
            {
                "type": "error",
                "code": code,
                "message": message,
                "session_id": self.runtime.session_id,
                "session_epoch": self.runtime.session_epoch,
            }
        )

    def _track(self, task: asyncio.Task[None]) -> None:
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _cancel_tasks(self) -> None:
        tasks = tuple(self._tasks)
        if not tasks:
            return
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _close_runner_session(self, session_id: str) -> None:
        close_session = getattr(self.runner, "close_session", None)
        if close_session is None:
            return
        try:
            await close_session(session_id)
        except Exception:
            logger.exception("DreamZero async failed to cleanup session %s", session_id)

    def _drain_outbound(self) -> None:
        while not self._outbound.empty():
            self._outbound.get_nowait()
