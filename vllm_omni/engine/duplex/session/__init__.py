# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""One engine-resident duplex session, and everything that runs it.

``DuplexEngineSession`` (``engine_session``) is the state -- the input,
response, playback and conversation ledgers, the lease and the identity fence.
``DuplexSessionRunner`` (``runner``) owns one of those on the orchestrator loop
and is the only writer of it; ``DuplexSessionManager`` (``manager``) owns the
runners and the admission slots.

The rest of the package is what the runner was split into, and each module
names the thing it owns rather than a layer:

* ``context``  -- the state a runner and its components share, and the only
  services a component may ask the runner for.
* ``emitter``  -- everything the session sends: the Realtime projection, the
  epoch filter, the domain effects a terminal event applies.
* ``model_channel`` -- the model side: submitting an append, projecting the
  stage output that comes back, continuing a turn with silence.
* ``control``  -- server VAD and the events that reconfigure the session.
* ``append_task`` -- one append in flight, and the rollback its failure owes.
* ``helpers``  -- the small pure reads and payload builders over a session.
* ``overlap_policy`` / ``commit_policy`` / ``playback_ledger`` -- the decisions
  a session makes about speech that arrives while the model is speaking, what a
  commit should do, and what the client has actually played.
* ``lease`` -- the idle/activity lease that decides when a session expires.

Imports here are the package's public surface; a module of this package is fair
game for anything inside it, but code outside ``engine.duplex`` should reach for
these three names.
"""

from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession
from vllm_omni.engine.duplex.session.manager import DuplexSessionManager
from vllm_omni.engine.duplex.session.runner import DuplexSessionRunner

__all__ = ["DuplexEngineSession", "DuplexSessionManager", "DuplexSessionRunner"]
