# SPDX-License-Identifier: Apache-2.0
"""Serving parameters exposed by the Cosmos3-Nano-Sim-Bimanual pipeline."""

COSMOS3_NANO_SIM_BIMANUAL_EXTRA_BODY_PARAMS = frozenset(
    {
        "action",
        "ar_diffusion_tick",
        "chunk_only",
        "close_session",
        "domain_id",
        "domain_name",
        "frame_idx",
        "initial_latent",
        "measure_tick_latency",
        "num_latent_frames",
        "reset",
        "session_id",
    }
)

COSMOS3_NANO_SIM_BIMANUAL_EXTRA_OUTPUT_PARAMS = frozenset()
