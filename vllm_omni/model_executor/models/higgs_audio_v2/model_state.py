# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MRV2 model state for the Higgs Audio v2 talker.

The stock ``OmniModelState`` already owns this pipeline's MRV2 lifecycle
(``mrv2_custom_sampler`` dispatch, ``postprocess`` request walk); the single
addition is a per-forward realignment of the talker's batch-row-keyed audio
state view. vLLM rebuilds each step's batch layout after finished requests
are freed and prefill requests graduate into the decode block, so a view
materialized at sampling time goes stale before the next forward's
``_maybe_apply_audio_feedback`` lookup. ``run_preprocess`` runs before every
forward (all graph modes) with the current batch, which is the layout the
forward actually consumes.
"""

from __future__ import annotations

from typing import Any

from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState


class HiggsAudioV2ModelState(OmniModelState):
    def run_preprocess(
        self,
        input_batch: Any,
        model_inputs: dict[str, Any],
        req_states: Any = None,
        mtp_batch_descriptor_dispatcher: Any = None,
    ) -> None:
        """Realign the talker's audio-state view, then run the stock preprocess.

        The stock preprocess is a no-op while the talker declares no
        ``has_preprocess`` embedding path; the rebuild is the reason this
        override exists and must stay ahead of it.
        """
        rebuild = getattr(self.model, "_rebuild_audio_state_view", None)
        if callable(rebuild):
            rebuild(input_batch)
        return super().run_preprocess(input_batch, model_inputs, req_states, mtp_batch_descriptor_dispatcher)
