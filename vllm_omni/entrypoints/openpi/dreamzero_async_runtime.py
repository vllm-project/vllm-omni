# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Serving-session runtime harness for DreamZero W8 async control."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field, replace
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
class ForwardTask:
    prefix_keys: tuple[CacheEntryKey, ...]
    send_action: bool
    commit_sim_kv: bool
    robot_obs: dict[str, Any] | None = None
    latent_video: Any | None = None
    drop_owner_key: CacheEntryKey | None = None
    drop_owner_keys: tuple[CacheEntryKey, ...] = ()

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
        drop_owner_keys = self.drop_owner_keys
        if self.drop_owner_key is not None:
            drop_owner_keys = (self.drop_owner_key,) + drop_owner_keys
        if drop_owner_keys:
            metadata["drop_owner_keys"] = [bde_cache_entry_key_dict(key) for key in drop_owner_keys]
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
    forwards: tuple[ForwardTask, ...] = ()
    outbound_messages: tuple[dict[str, Any], ...] = ()


@dataclass
class RealObservationEntry:
    robot_obs: dict[str, Any]
    key: CacheEntryKey
    commit_in_flight: bool = True
    kv_committed: bool = False
    kv_materialized: bool = False


@dataclass(frozen=True)
class SimulatedObservationEntry:
    key: CacheEntryKey
    robot_obs: dict[str, Any]
    latent_video: Any


TaskKey = tuple[int, tuple[CacheEntryKey, ...], bool, bool]


class DreamZeroAsyncScheduler:
    # DROID's current DreamZero checkpoint uses local_attn_size=9 latent frames
    # and num_frame_per_block=2, so four observation entries cover the active
    # BDE window while leaving room for the current in-flight write entry.
    _MAX_PREFIX_ENTRIES = 4
    _MAX_REAL_PREFIX_ENTRIES_WITH_SIM = _MAX_PREFIX_ENTRIES - 1
    _MAX_QUEUED_FORWARDS = 2

    def __init__(self, *, session_id: str, session_epoch: int, prompt: str) -> None:
        self.session_id = session_id
        self.session_epoch = session_epoch
        self.prompt = prompt
        self.real_observations: dict[int, RealObservationEntry] = {}
        self.sim_observations: dict[int, SimulatedObservationEntry] = {}
        self.completed_sim_indices: set[int] = set()
        self.owner_release_candidates: set[CacheEntryKey] = set()
        self.running_tasks: set[TaskKey] = set()
        self.completed_tasks: set[TaskKey] = set()
        self.published_chunks: set[int] = set()

    def on_observation_received(
        self,
        *,
        observation_index: int,
        robot_obs: dict[str, Any],
    ) -> RuntimeUpdate:
        sim_entry = self.sim_observations.get(observation_index)
        if sim_entry is not None:
            self.owner_release_candidates.add(sim_entry.key)
            if not self._key_used_by_running_task(sim_entry.key):
                self.sim_observations.pop(observation_index, None)

        existing = self.real_observations.get(observation_index)
        if existing is not None and (existing.commit_in_flight or existing.kv_committed):
            return RuntimeUpdate(forwards=self._try_submit_next_forward())

        key = CacheEntryKey(
            session_id=self.session_id,
            session_epoch=self.session_epoch,
            observation_index=observation_index,
            sim_depth=0,
        )
        self.real_observations[observation_index] = RealObservationEntry(
            robot_obs=robot_obs,
            key=key,
            commit_in_flight=True,
        )
        return RuntimeUpdate(
            observation_commits=(ObservationCommitTask(key=key, robot_obs=robot_obs),),
            forwards=self._try_submit_next_forward(),
        )

    def on_observation_committed(self, task: ObservationCommitTask) -> RuntimeUpdate:
        if self._is_stale_key(task.key):
            return RuntimeUpdate()
        index = task.key.observation_index
        entry = self.real_observations.get(index)
        if entry is None or entry.key != task.key:
            return RuntimeUpdate()
        entry.commit_in_flight = False
        entry.kv_committed = True

        return RuntimeUpdate(forwards=self._try_submit_next_forward())

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

        input_key = task.prefix_keys[-1]
        if input_key.sim_depth == 0:
            entry = self.real_observations.get(input_key.observation_index)
            if entry is not None and entry.key == input_key:
                entry.kv_materialized = True

        for key in task.prefix_keys:
            if key.sim_depth > 0 and self._real_arrived(key.observation_index):
                self.owner_release_candidates.add(key)
                if not self._key_used_by_running_task(key):
                    self.sim_observations.pop(key.observation_index, None)

        if not task.commit_sim_kv:
            return RuntimeUpdate(
                forwards=self._try_submit_next_forward(),
                outbound_messages=tuple(messages),
            )
        self.completed_sim_indices.add(result.produced_sim_observation)

        sim_key = CacheEntryKey(
            session_id=self.session_id,
            session_epoch=self.session_epoch,
            observation_index=result.produced_sim_observation,
            sim_depth=1,
        )
        if result.produced_video is None:
            logger.info(
                "DreamZero async forward chunk=%d did not produce simulated video; no lookahead scheduled",
                result.chunk_index,
            )
            return RuntimeUpdate(outbound_messages=tuple(messages))

        self.sim_observations[sim_key.observation_index] = SimulatedObservationEntry(
            key=sim_key,
            robot_obs=self._make_sim_robot_obs(task, result),
            latent_video=result.produced_video,
        )
        if self._real_arrived(sim_key.observation_index):
            self.owner_release_candidates.add(sim_key)
        return RuntimeUpdate(
            forwards=self._try_submit_next_forward(),
            outbound_messages=tuple(messages),
        )

    def on_observation_commit_failed(self, task: ObservationCommitTask, message: str) -> RuntimeUpdate:
        if self._is_stale_key(task.key):
            return RuntimeUpdate()
        entry = self.real_observations.get(task.key.observation_index)
        if entry is not None and entry.key == task.key:
            entry.commit_in_flight = False
        return RuntimeUpdate(outbound_messages=(self._error_message("observation_commit_failed", message),))

    def on_forward_failed(self, task: ForwardTask, message: str) -> RuntimeUpdate:
        if task.session_epoch != self.session_epoch:
            return RuntimeUpdate()
        self.running_tasks.discard(self._task_key(task))
        self.completed_tasks.add(self._task_key(task))
        return RuntimeUpdate(outbound_messages=(self._error_message("forward_failed", message),))

    def try_submit_forward(self, task: ForwardTask) -> tuple[ForwardTask, ...]:
        if len(self.running_tasks) >= self._MAX_QUEUED_FORWARDS:
            return ()
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
        task = replace(task, drop_owner_keys=self._take_releasable_owner_keys(task.prefix_keys))
        self.running_tasks.add(task_key)
        logger.info(
            "DreamZero async schedule forward prefix=%s send_action=%s commit_sim_kv=%s",
            self._prefix_label(task.prefix_keys),
            task.send_action,
            task.commit_sim_kv,
        )
        return (task,)

    def _try_submit_next_forward(self) -> tuple[ForwardTask, ...]:
        if len(self.running_tasks) >= self._MAX_QUEUED_FORWARDS:
            return ()

        candidates: list[ForwardTask] = []
        if self._real_committed(1) and not self._real_materialized(1):
            real_prefix = self._real_forward_prefix_through(1)
            if real_prefix is None:
                return ()
            candidates.append(
                ForwardTask(
                    prefix_keys=real_prefix,
                    send_action=True,
                    commit_sim_kv=True,
                    robot_obs=self._real_robot_obs(1),
                )
            )

        for sim_index in sorted(self.sim_observations):
            if sim_index in self.published_chunks:
                continue
            real_prefix = self._materialized_real_prefix_through(
                sim_index - 1,
                max_entries=self._MAX_REAL_PREFIX_ENTRIES_WITH_SIM,
            )
            if real_prefix is None:
                continue
            sim_entry = self.sim_observations[sim_index]
            candidates.append(
                ForwardTask(
                    prefix_keys=real_prefix + (sim_entry.key,),
                    send_action=True,
                    commit_sim_kv=False,
                    robot_obs=sim_entry.robot_obs,
                    latent_video=sim_entry.latent_video,
                )
            )

        for index in self._committed_real_indices():
            if not self._real_materialized(index):
                real_prefix = self._real_forward_prefix_through(index)
                if real_prefix is None:
                    continue
                if any(candidate.prefix_keys == real_prefix and candidate.commit_sim_kv for candidate in candidates):
                    continue
                candidates.append(
                    ForwardTask(
                        prefix_keys=real_prefix,
                        send_action=False,
                        commit_sim_kv=True,
                        robot_obs=self._real_robot_obs(index),
                    )
                )
                break

        for index in self._materialized_real_indices():
            produced_index = index + 1
            if (
                self._real_arrived(produced_index)
                or produced_index in self.sim_observations
                or produced_index in self.completed_sim_indices
            ):
                continue
            real_prefix = self._materialized_real_prefix_through(index)
            if real_prefix is None:
                continue
            if any(candidate.prefix_keys == real_prefix and candidate.commit_sim_kv for candidate in candidates):
                continue
            candidates.append(
                ForwardTask(
                    prefix_keys=real_prefix,
                    send_action=False,
                    commit_sim_kv=True,
                    robot_obs=self._real_robot_obs(index),
                )
            )

        materialized_indices = self._materialized_real_indices()
        if materialized_indices:
            latest_index = max(materialized_indices)
            real_prefix = self._materialized_real_prefix_through(latest_index)
            if (
                real_prefix is not None
                and latest_index + 1 not in self.completed_sim_indices
                and not any(candidate.prefix_keys == real_prefix and candidate.commit_sim_kv for candidate in candidates)
            ):
                candidates.append(
                    ForwardTask(
                        prefix_keys=real_prefix,
                        send_action=False,
                        commit_sim_kv=True,
                        robot_obs=self._real_robot_obs(latest_index),
                    )
                )

        submitted_tasks: list[ForwardTask] = []
        for candidate in candidates:
            submitted = self.try_submit_forward(candidate)
            if submitted:
                submitted_tasks.extend(submitted)
            if len(self.running_tasks) >= self._MAX_QUEUED_FORWARDS:
                break
        return tuple(submitted_tasks)

    def _real_forward_prefix_through(self, observation_index: int) -> tuple[CacheEntryKey, ...] | None:
        keys = []
        for index in range(1, observation_index):
            if not self._real_materialized(index):
                return None
            key = self._real_key(index)
            if key is None:
                return None
            keys.append(key)
        key = self._real_key(observation_index)
        if key is None:
            return None
        keys.append(key)
        return tuple(keys[-self._MAX_PREFIX_ENTRIES :])

    def _materialized_real_prefix_through(
        self,
        observation_index: int,
        *,
        max_entries: int | None = None,
    ) -> tuple[CacheEntryKey, ...] | None:
        keys = []
        for index in range(1, observation_index + 1):
            if not self._real_materialized(index):
                return None
            key = self._real_key(index)
            if key is None:
                return None
            keys.append(key)
        limit = self._MAX_PREFIX_ENTRIES if max_entries is None else max_entries
        return tuple(keys[-limit:])

    def _prefix_available(self, prefix_keys: tuple[CacheEntryKey, ...]) -> bool:
        input_position = len(prefix_keys) - 1
        for position, key in enumerate(prefix_keys):
            if self._is_stale_key(key):
                return False
            if key.sim_depth == 0:
                if self._real_key(key.observation_index) != key:
                    return False
                if position != input_position and not self._real_materialized(key.observation_index):
                    return False
            if key.sim_depth == 1:
                sim_entry = self.sim_observations.get(key.observation_index)
                if sim_entry is None or sim_entry.key != key:
                    return False
        return True

    def _real_arrived(self, observation_index: int) -> bool:
        return observation_index in self.real_observations

    def _real_committed(self, observation_index: int) -> bool:
        entry = self.real_observations.get(observation_index)
        return entry is not None and entry.kv_committed

    def _real_materialized(self, observation_index: int) -> bool:
        entry = self.real_observations.get(observation_index)
        return entry is not None and entry.kv_materialized

    def _real_key(self, observation_index: int) -> CacheEntryKey | None:
        entry = self.real_observations.get(observation_index)
        if entry is None or not entry.kv_committed:
            return None
        return entry.key

    def _real_robot_obs(self, observation_index: int) -> dict[str, Any] | None:
        entry = self.real_observations.get(observation_index)
        if entry is None:
            return None
        return entry.robot_obs

    def _committed_real_indices(self) -> list[int]:
        return sorted(index for index, entry in self.real_observations.items() if entry.kv_committed)

    def _materialized_real_indices(self) -> list[int]:
        return sorted(index for index, entry in self.real_observations.items() if entry.kv_materialized)

    def _take_releasable_owner_keys(self, next_prefix_keys: tuple[CacheEntryKey, ...]) -> tuple[CacheEntryKey, ...]:
        excluded = set(next_prefix_keys)
        for task_key in self.running_tasks:
            excluded.update(task_key[1])

        committed_indices = self._committed_real_indices()
        latest_committed_index = committed_indices[-1] if committed_indices else 0
        for index, entry in self.real_observations.items():
            if not entry.kv_committed:
                continue
            key = entry.key
            if key in excluded:
                continue
            if index < latest_committed_index:
                self.owner_release_candidates.add(key)

        releasable = tuple(sorted(key for key in self.owner_release_candidates if key not in excluded))
        self.owner_release_candidates.difference_update(releasable)
        return releasable

    def _key_used_by_running_task(self, key: CacheEntryKey) -> bool:
        return any(key in task_key[1] for task_key in self.running_tasks)

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
        base = self._real_robot_obs(result.chunk_index)
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
            drop_owner_keys=task.drop_owner_keys,
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

class DreamZeroAsyncSession:
    def __init__(
        self,
        *,
        runner: DeterministicFakeDreamZeroRunner | None = None,
    ) -> None:
        self.runner = runner or DeterministicFakeDreamZeroRunner()
        self.scheduler: DreamZeroAsyncScheduler | None = None
        self._outbound: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._tasks: set[asyncio.Task[None]] = set()

    async def start_session(self, *, session_id: str, session_epoch: int, prompt: str) -> None:
        await self._cancel_tasks()
        self._drain_outbound()
        self.scheduler = DreamZeroAsyncScheduler(
            session_id=session_id,
            session_epoch=session_epoch,
            prompt=prompt,
        )

    async def reset(self, *, session_epoch: int) -> None:
        if self.scheduler is None:
            return
        old_session_id = self.scheduler.session_id
        old_prompt = self.scheduler.prompt
        await self._cancel_tasks()
        await self._close_runner_session(old_session_id)
        self._drain_outbound()
        self.scheduler = DreamZeroAsyncScheduler(
            session_id=old_session_id,
            session_epoch=session_epoch,
            prompt=old_prompt,
        )

    async def close(self) -> None:
        session_id = self.scheduler.session_id if self.scheduler is not None else None
        await self._cancel_tasks()
        if session_id is not None:
            await self._close_runner_session(session_id)
        self.scheduler = None
        self._drain_outbound()

    async def submit_observation(self, request: dict[str, Any]) -> None:
        if self.scheduler is None:
            return

        update = self.scheduler.on_observation_received(
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
        for commit_task in update.observation_commits:
            self._track(asyncio.create_task(self._run_observation_commit(commit_task)))
        for forward_task in update.forwards:
            self._track(asyncio.create_task(self._run_forward(forward_task)))

    async def _run_observation_commit(self, task: ObservationCommitTask) -> None:
        try:
            committed = await self.runner.commit_observation(task)
        except Exception:
            logger.exception("DreamZero async failed to commit real observation")
            if self.scheduler is not None:
                self._schedule_update(
                    self.scheduler.on_observation_commit_failed(
                        task,
                        "DreamZero async failed to commit real observation",
                    )
                )
            return
        if self.scheduler is None:
            return
        self._schedule_update(self.scheduler.on_observation_committed(committed))

    async def _run_forward(self, task: ForwardTask) -> None:
        try:
            result = await self.runner.run_forward(task)
        except Exception:
            logger.exception(
                "DreamZero async forward failed prefix=%s send_action=%s commit_sim_kv=%s",
                DreamZeroAsyncScheduler._prefix_label(task.prefix_keys),
                task.send_action,
                task.commit_sim_kv,
            )
            if self.scheduler is not None:
                self._schedule_update(
                    self.scheduler.on_forward_failed(
                        task,
                        "DreamZero async forward failed",
                    )
                )
            return
        if self.scheduler is None:
            return
        self._schedule_update(self.scheduler.on_forward_done(task, result))

    async def _publish_task_error(self, code: str, message: str) -> None:
        if self.scheduler is None:
            return
        self._outbound.put_nowait(
            {
                "type": "error",
                "code": code,
                "message": message,
                "session_id": self.scheduler.session_id,
                "session_epoch": self.scheduler.session_epoch,
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
