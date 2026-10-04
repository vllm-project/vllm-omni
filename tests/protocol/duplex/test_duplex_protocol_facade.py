# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""``protocol.duplex`` is the only door a duplex consumer uses.

The layering is a chain::

    protocol.realtime  <-  protocol.duplex  <-  engine / entrypoints / clients

The middle link only earns its keep if the last arrow is the *only* one. If the
engine may also reach past it straight into ``protocol.realtime``, then giving a
Tier 1 helper a duplex-specific version later means touching every call site
instead of one file --- which is exactly the change
``convert_input_audio_with_rate`` is expected to need (it resamples to
MiniCPM-o's 16 kHz, not the client's rate).

So this asserts two things: nobody skips the middle link, and the middle link is
complete enough that nobody has to.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

REPO_ROOT = Path(__file__).resolve().parents[3]

#: Every module that speaks the duplex protocol. These import
#: ``vllm_omni.protocol.duplex``; reaching into ``vllm_omni.protocol.realtime``
#: from here skips the extension point.
CONSUMER_PATHS = (
    "vllm_omni/engine/duplex",
    "vllm_omni/engine/duplex_omni_engine.py",
    "vllm_omni/entrypoints/duplex",
    "vllm_omni/entrypoints/duplex_omni.py",
    "vllm_omni/clients/duplex.py",
    "vllm_omni/clients/inline_duplex.py",
)


def _consumer_modules() -> list[Path]:
    found: list[Path] = []
    for entry in CONSUMER_PATHS:
        path = REPO_ROOT / entry
        found.extend(sorted(path.rglob("*.py")) if path.is_dir() else [path])
    assert found, "consumer list is stale"
    return found


@pytest.mark.parametrize("module", _consumer_modules(), ids=lambda p: p.name)
def test_a_duplex_consumer_does_not_skip_the_duplex_protocol_package(module: Path) -> None:
    # Source-level, so a lazy import inside a function body is caught too.
    tree = ast.parse(module.read_text())
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("vllm_omni.protocol.realtime"):
            offenders.append(f"line {node.lineno}: from {node.module} import ...")
        elif isinstance(node, ast.Import):
            offenders.extend(
                f"line {node.lineno}: import {alias.name}"
                for alias in node.names
                if alias.name.startswith("vllm_omni.protocol.realtime")
            )
    assert not offenders, (
        f"{module.relative_to(REPO_ROOT)} imports the Tier 1 package directly; "
        f"import from vllm_omni.protocol.duplex instead:\n  " + "\n  ".join(offenders)
    )


def test_the_duplex_package_carries_the_whole_command_vocabulary() -> None:
    from vllm_omni.engine.duplex import commands as engine_commands
    from vllm_omni.protocol.duplex import commands as duplex_wire

    # Every command the engine handles is nameable through the one door.
    engine_names = {
        name
        for name in engine_commands.__all__
        if isinstance(getattr(engine_commands, name), type)
        and issubclass(getattr(engine_commands, name), engine_commands.DuplexCommand)
        and getattr(engine_commands, name) is not engine_commands.DuplexCommand
    }
    assert engine_names <= set(duplex_wire.__all__)


def test_the_duplex_package_carries_the_whole_event_vocabulary() -> None:
    from vllm_omni.engine.duplex import events as engine_events
    from vllm_omni.protocol.duplex import events as duplex_wire

    engine_names = {
        name
        for name in engine_events.__all__
        if isinstance(getattr(engine_events, name), type)
        and issubclass(getattr(engine_events, name), engine_events.DuplexEvent)
    }
    assert engine_names <= set(duplex_wire.__all__)


@pytest.mark.parametrize(
    "module_pair",
    [("events", "events"), ("commands", "commands")],
    ids=["events", "commands"],
)
def test_a_reexported_tier1_name_is_the_tier1_object(module_pair: tuple[str, str]) -> None:
    """Re-export, not re-declaration: the Tier 1 class is one object."""
    import importlib

    duplex_mod = importlib.import_module(f"vllm_omni.protocol.duplex.{module_pair[0]}")
    realtime_mod = importlib.import_module(f"vllm_omni.protocol.realtime.{module_pair[1]}")

    shared = [n for n in duplex_mod.__all__ if hasattr(realtime_mod, n)]
    assert shared, "nothing re-exported --- the facade is not doing its job"
    for name in shared:
        duplex_obj = getattr(duplex_mod, name)
        realtime_obj = getattr(realtime_mod, name)
        if duplex_obj is realtime_obj:
            continue
        # The only permitted divergence is a Tier 2 class extending its twin.
        assert isinstance(duplex_obj, type) and issubclass(duplex_obj, realtime_obj), (
            f"{name} in protocol.duplex is neither the Tier 1 object nor a subclass of it"
        )


def test_duplex_event_is_an_alias_but_duplex_command_is_not() -> None:
    """The asymmetry is deliberate, and a later reader should not 'fix' it.

    An event has no engine-internal half --- what the session emits is what the
    socket sends --- so ``DuplexEvent`` is simply ``RealtimeEvent``. A command
    does: ``payload()`` renders the session runner's mailbox dictionary, and for
    four of the seventeen the mailbox channel is not even the client event type.
    """
    from vllm_omni.engine.duplex.commands import DuplexCommand
    from vllm_omni.protocol.duplex.events import DuplexEvent
    from vllm_omni.protocol.realtime.commands import RealtimeCommand
    from vllm_omni.protocol.realtime.events import RealtimeEvent

    assert DuplexEvent is RealtimeEvent

    assert DuplexCommand is not RealtimeCommand
    assert issubclass(DuplexCommand, RealtimeCommand)
    assert hasattr(DuplexCommand, "payload") and not hasattr(RealtimeCommand, "payload")
