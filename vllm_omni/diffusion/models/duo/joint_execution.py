# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Adjacent official StepRequestStates share clean + first-noisy execution."""

import torch

from .joint_graph import RealizationJointCycle


def execute_joint_pairs(pipeline, states):
    result = {}
    available = {s.request_id: s for s in states}
    for session in tuple(pipeline.sessions.values()):
        old, nxt = session.active_state, session.lookahead_state
        if old is None or nxt is None or any(s.request_id not in available for s in (old, nxt)):
            continue
        a, b = (s.extra["realization_condition"] for s in (old, nxt))
        if (
            old.extra["realization_ack"]
            or old.step_index != old.extra["realization_denoise_segments"]
            or nxt.step_index != 0
            or a.final
            or b.block != a.block + 1
            or a.fence != b.fence
            or a.session_id != b.session_id
            or a.tenant != b.tenant
        ):
            raise ValueError("Invalid official Realization joint state transition")
        if pipeline._graph_key(a.sequence, False) != pipeline._graph_key(b.sequence, False):
            # Keep the original two graph calls when body/refiner shape changes.
            continue
        clean, noisy = pipeline._sequence(old, True), pipeline._sequence(nxt)
        window = session.history_blocks
        key = pipeline._graph_key(a.sequence, False) + (
            nxt.extra["realization_sampling_stage"],
            nxt.extra["realization_flow_shift"],
            window,
        )
        slot = pipeline.joint_graphs.get(key)
        if slot is None:
            if len(pipeline.joint_graphs) >= pipeline.options.get("max_graphs", 8):
                torch.accelerator.synchronize(clean.device)
                pipeline.joint_graphs.pop(next(iter(pipeline.joint_graphs)))
            slot = RealizationJointCycle(
                pipeline.transformer,
                pipeline._pool(),
                clean,
                noisy,
                capacity=pipeline.options.get("kv_capacity", 65536),
                history_blocks=window,
                separate_projections=pipeline.options["joint_separate_projections"],
                sampling_stage=nxt.extra["realization_sampling_stage"],
                flow_shift=nxt.extra["realization_flow_shift"],
            )
            pipeline.joint_graphs[key] = slot
        try:
            early_ack = pipeline.options.get("joint_early_clean_ack", False)
            (video, audio), cache = slot.run(
                clean, noisy, session.cache, first_only=early_ack, noises=nxt.extra["realization_noises"]
            )
            session.cache = cache
            old.extra["realization_ack"] = True
            old.extra["realization_joint_clean"] = True
            nxt.extra["realization_joint_denoise"] = True
            if early_ack:
                # The first NFE already ran with clean; one remaining-chain
                # segment precedes this request's mandatory clean segment.
                nxt.extra["realization_joint_first_nfe"] = True
                nxt.extra["realization_joint_text_output"] = slot.pair.next_plan.native_runtime.text_output.clone()
                nxt.extra["realization_denoise_segments"] = 2
                nxt.timesteps = torch.arange(3, device=noisy.device)
            nxt.extra["realization_history_rows_before"] = cache.length
            result[old.request_id] = old.latents.clone()
            result[nxt.request_id] = torch.cat((video.flatten(), audio.flatten())).view(1, -1)
        except BaseException:
            pipeline.close_session(a.session_id, a.fence)
            raise
    return result
