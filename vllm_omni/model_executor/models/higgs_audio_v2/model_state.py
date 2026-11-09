# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MRV2 model state for the Higgs Audio v2 talker.

The stock ``OmniModelState`` owns this pipeline's MRV2 lifecycle. The only
addition realigns the talker's batch-row-keyed audio state view before every
forward: vLLM rebuilds the batch layout between steps, so a view taken at
sampling time goes stale before the next lookup.
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

        The rebuild is the reason this override exists and must run ahead of
        the preprocess; the stock preprocess is a no-op for this talker.
        """
        rebuild = getattr(self.model, "_rebuild_audio_state_view", None)
        if callable(rebuild):
            rebuild(input_batch)
        return super().run_preprocess(input_batch, model_inputs, req_states, mtp_batch_descriptor_dispatcher)
