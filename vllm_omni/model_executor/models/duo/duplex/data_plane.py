# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Publish owned AV bytes; retain child ownership until the clean KV ACK."""

import time
from collections.abc import Mapping
from dataclasses import dataclass

from vllm_omni.engine.duplex.contracts import duplex_resource_request_belongs_to_session
from vllm_omni.engine.duplex.plugin import DuplexDataPlane, DuplexDataPlaneContext

from .av_output import encode_av_chunk
from .trace import trace


@dataclass
class PendingChunk:
    identity: tuple | None = None
    clean: bool = False
    decoded: dict | None = None
    codec_received: bool = False
    emitted: bool = False
    clean_metadata: dict | None = None
    codec_metadata: dict | None = None
    latent_metadata: dict | None = None


class DuoDataPlane(DuplexDataPlane):
    def __init__(self, *, realization_stage_id=1, codec_stage_id=2, defer_clean=False):
        self.realization_stage_id, self.codec_stage_id = realization_stage_id, codec_stage_id
        self.defer_clean = defer_clean
        self.requests, self.states = {}, {}

    def begin_request(self, request_id):
        self.requests.setdefault(request_id, PendingChunk())

    def bind_state(self, request_id, state):
        self.states[request_id] = state
        self.requests[request_id] = PendingChunk()

    def is_terminal(self, request_id):
        return request_id is not None and request_id not in self.requests

    def mark_terminal(self, request_id):
        self.close_stream(request_id)

    def owner_request_id(self, request_id):
        state = self.states.get(request_id)
        return getattr(state, "parent_request_id", request_id)

    def close_stream(self, request_id):
        self.requests.pop(request_id, None)
        state = self.states.get(request_id)
        if state is None:
            return
        if hasattr(state, "owner"):
            trace("retired_child", state)
            self.states.pop(request_id, None)
            return
        if state.busy or state.inflight:
            trace("stream_cancelled", state)
            state.close()
            for child_id, child in list(self.states.items()):
                if getattr(child, "owner", None) is state:
                    self.requests.pop(child_id, None)
                    self.states.pop(child_id, None)

    def close_session(self, session_id, *, active_request_id=None):
        for rid in list(set(self.requests) | set(self.states)):
            if duplex_resource_request_belongs_to_session(rid, session_id):
                self.close_stream(rid)
                state = self.states.pop(rid, None)
                if state is not None and hasattr(state, "close"):
                    state.close()

    def project(self, result, *, context=None):
        if not isinstance(result, Mapping):
            return
        context = context or DuplexDataPlaneContext()
        for output in result.get("data_plane_outputs", ()):
            pending = self.requests.get(output.request_id)
            state = self.states.get(output.request_id)
            if pending is None or state is None:
                continue
            owner = getattr(state, "owner", None)
            deferred = self.defer_clean and owner is not None and owner.schedule.defer_clean
            if pending.emitted and not (deferred and output.stage_id == self.realization_stage_id):
                continue
            mm = getattr(output, "multimodal_output", None)
            if not isinstance(mm, Mapping) or not isinstance(mm.get("metadata"), Mapping):
                continue
            meta = mm["metadata"]
            identity = (meta.get("session_id"), meta.get("fence"), meta.get("block"))
            if identity != (state.session_id, context.epoch, state.next_block):
                raise ValueError("Stale Realization AV identity")
            if pending.identity is None:
                pending.identity = identity
            if pending.identity != identity:
                raise ValueError("Realization clean/codec identity mismatch")
            if output.stage_id == self.realization_stage_id:
                if meta.get("phase") == "latent_ready":
                    pending.latent_metadata = dict(meta)
                    continue
                if meta.get("phase") != "clean_committed":
                    continue
                if pending.clean or meta.get("commit_nfe") != 1 or meta.get("clean_committed") is not True:
                    raise ValueError("Invalid/duplicate clean acknowledgement")
                pending.clean = True
                pending.clean_metadata = dict(meta)
                trace("clean_ack", state)
                if pending.emitted:
                    owner = state.owner
                    owner.inflight.pop(state.next_block)
                    owner.capacity.release()
                    state.busy = False
                    final = state.final and not owner.inflight and not owner.busy
                    yield dict(
                        supported=True,
                        data_plane_request_id=state.parent_request_id,
                        completed_request_ids=[output.request_id],
                        text="",
                        end_of_turn=final,
                        audio_complete=final,
                        model_turn_id=context.active_response_turn_id
                        if context.active_response_turn_id is not None
                        else context.turn_id,
                    )
                    continue
            elif output.stage_id == self.codec_stage_id:
                if pending.codec_received or pending.decoded is not None:
                    raise ValueError("Duplicate codec chunk")
                pending.codec_received = True
                # Official IPC attaches shared-memory ownership to tensor
                # storage. Encoding copies into bytes; no raw view escapes.
                payload = mm.get("payload", mm)
                if meta.get("codec_mode") == "duo_5_17":
                    owner = getattr(state, "owner", None)
                    if owner is None:
                        raise ValueError("Delayed codec requires a pipeline owner")
                    for packet in payload["packets"]:
                        clock = packet["metadata"]
                        block = clock["block"]
                        rid = owner.inflight.get(block)
                        target = self.requests.get(rid)
                        target_state = self.states.get(rid)
                        if (
                            target is None
                            or target_state is None
                            or target.emitted
                            or target.decoded is not None
                            or (clock["session_id"], clock["fence"]) != (state.session_id, state.fence)
                            or not 0 <= block <= state.next_block
                        ):
                            raise ValueError("Stale/duplicate delayed codec output")
                        # The codec rank already encoded owned JPEG/PCM bytes;
                        # keep the orchestration loop free for stage dispatch.
                        target.decoded = packet["payload"]
                        target.codec_metadata = dict(meta, **clock, decode_source_block=state.next_block)
                        trace("codec_complete", target_state)
                else:
                    pending.decoded = encode_av_chunk(payload, meta)
                    pending.codec_metadata = dict(meta)
                    trace("codec_complete", state)
            else:
                continue
            owner = getattr(state, "owner", None)
            if owner is None:
                ready_ids = [output.request_id]
            else:
                ready_ids = []
                block = owner.next_publish
                while block in owner.inflight:
                    rid = owner.inflight[block]
                    current = self.requests.get(rid)
                    if (
                        current is None
                        or current.emitted
                        or (not current.clean and not deferred)
                        or current.decoded is None
                    ):
                        break
                    ready_ids.append(rid)
                    block += 1
            for rid in ready_ids:
                current = self.requests[rid]
                if (not current.clean and not deferred) or current.decoded is None or current.emitted:
                    continue
                chunk = self.states[rid]
                current.emitted = True
                packet, current.decoded = current.decoded, None
                packet["video"]["timings"] = dict(
                    realization_latent=current.latent_metadata,
                    realization_clean=current.clean_metadata,
                    clean_pending=not current.clean,
                    codec=current.codec_metadata,
                    published_ns=time.perf_counter_ns(),
                )
                trace("published", chunk)
                chunk.busy = deferred and not current.clean
                if owner is None:
                    chunk.next_block += 1
                    end_of_turn = True
                    root_id = rid
                    completed = []
                else:
                    if not deferred or current.clean:
                        owner.inflight.pop(chunk.next_block)
                        owner.capacity.release()
                    owner.next_publish += 1
                    end_of_turn = not owner.inflight and not owner.busy and not result.get("pending_input_units", 0)
                    if current.codec_metadata.get("codec_mode") == "duo_5_17":
                        # A paced stream can be idle between observations.
                        # Only the model's explicit final block ends its response.
                        end_of_turn = end_of_turn and chunk.final
                    root_id = chunk.parent_request_id
                    completed = [] if deferred and not current.clean else [rid]
                    response_id = getattr(context, "active_response_id", None)
                    if getattr(owner, "response_clock_id", "") != response_id:
                        owner.response_clock_id = response_id
                        owner.response_clock_origin = packet["video"]["audio_start_sample"]
                    packet["audio_duration_ms"] = (
                        (packet["video"]["audio_stop_sample"] - owner.response_clock_origin) * 1000 // 32000
                    )
                    packet["audio_duration_is_cumulative"] = True
                packet["video"]["action"] = "wait" if chunk.current_text == "<wait>" else "speak"
                text = "" if chunk.current_text == "<wait>" else chunk.current_text
                yield dict(
                    supported=True,
                    data_plane_request_id=root_id,
                    completed_request_ids=completed,
                    text=text,
                    end_of_turn=end_of_turn,
                    audio_complete=end_of_turn,
                    model_turn_id=context.active_response_turn_id
                    if context.active_response_turn_id is not None
                    else context.turn_id,
                    runtime_impl="duo",
                    uses_model_runner_scheduler=True,
                    runner_kv_backed=True,
                    **packet,
                )
                if owner is not None and deferred:
                    # Runs after the session emitter consumed the AV packet.
                    # Keep the child registered until the later clean-only event.
                    owner.schedule.complete("published", chunk.next_block)
