# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Journal budget boundaries must survive incremental media accounting."""

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from vllm_omni.engine.duplex.session.context_history import DuplexContextHistory

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.asyncio]


def encoded_size(prompts):
    return len(json.dumps(prompts, default=lambda o: vars(o) if hasattr(o, "__dict__") else str(o)).encode())


@pytest.fixture
def history():
    async def idle(*args):
        pass

    policy = SimpleNamespace(
        max_bytes=1_000_000,
        token_count=lambda p: len(p["prompt_token_ids"]) + len(p.get("output_token_ids", [])),
        complete=lambda p, output, context: {**deepcopy(p), "output_token_ids": output},
    )
    ctx = SimpleNamespace(session=SimpleNamespace(epoch=0), services=SimpleNamespace(spawn=asyncio.create_task))
    return DuplexContextHistory(
        ctx, policy, out=None, model=None, max_tokens=1000, wait_for_append_tail=idle, close_from_runtime=idle
    )


def finish(history, output):
    context = SimpleNamespace(
        identity=SimpleNamespace(fence=SimpleNamespace(epoch=history.session.epoch)), segment_finished=True
    )
    return history.observe(0, history.pending_request, output, context)


async def test_exact_byte_budget_includes_unicode_escaping_and_metadata(history):
    first: dict[str, object] = {
        "prompt_token_ids": [1],
        "media": '中文🙂\n"\\',
        "fence": SimpleNamespace(epoch=0),
        "opaque": b"abc",
    }
    second: dict[str, object] = {"prompt_token_ids": [2], "media": "AQID"}
    expected = deepcopy([first, second])
    history.max_bytes = encoded_size(expected)
    history.record("a", first)
    first["media"] = "mutated after submission"
    history.record("b", second)
    assert encoded_size(history.prompts) == history.max_bytes
    assert history.prompts[0]["media"] == expected[0]["media"]
    with pytest.raises(ValueError, match="budget"):
        history.record("c", second)
    assert history.prompts == expected
    assert history.pending_request == "b"


@pytest.mark.parametrize("budget", ["bytes", "tokens"])
async def test_rejected_append_preserves_budget_for_next_input(history, budget):
    first = {"prompt_token_ids": [1], "media": "a"}
    rejected = {"prompt_token_ids": [2, 3], "media": "large"}
    accepted = {"prompt_token_ids": [4], "media": "b"}
    if budget == "bytes":
        history.max_bytes = encoded_size([first, rejected]) - 1
    else:
        history.max_tokens = 3  # Equality is rejected, not only overflow.
    history.record("a", first)
    with pytest.raises(ValueError, match="budget"):
        history.record("rejected", rejected)
    history.record("accepted", accepted)
    assert history.prompts == [first, accepted]


async def test_completion_replaces_only_last_unit_and_charges_its_output(history):
    first = {"prompt_token_ids": [1], "media": "old"}
    second = {"prompt_token_ids": [2], "media": "new"}
    history.record("a", first)
    finish(history, [3])
    await history.wait_applied()
    history.record("b", second)
    expected = [{**first, "output_token_ids": [3]}, {**second, "output_token_ids": [4, 5]}]
    history.max_bytes = encoded_size(expected)
    finish(history, [4, 5])
    await history.wait_applied()
    assert history.prompts == expected
    with pytest.raises(ValueError, match="budget"):
        history.record("c", second)
    history.max_bytes = 1_000_000
    history.max_tokens = 6  # Five settled tokens plus one input hits equality.
    with pytest.raises(ValueError, match="budget"):
        history.record("c", {"prompt_token_ids": [6]})


@pytest.mark.parametrize("replaying", [False, True])
async def test_oversized_completion_does_not_change_retained_journal(history, replaying, mocker):
    prompt = {"prompt_token_ids": [1]}
    history.record("a", prompt)
    history.max_tokens = 3
    history.replaying = replaying
    # The real close path is covered by session-runner tests; retain its task
    # boundary here so no background close can mask the failed completion.
    spawned = mocker.spy(history._ctx.services, "spawn")
    finish(history, [2, 3])
    with pytest.raises(ValueError, match="budget"):
        await history.wait_applied()
    assert history.prompts == [prompt]
    await spawned.spy_return
    history.record("b", {"prompt_token_ids": [4]})
    assert len(history.prompts) == 2


async def test_replay_completion_is_validated_but_does_not_charge_unretained_output(history):
    prompt = {"prompt_token_ids": [1]}
    history.max_tokens = 4
    history.record("a", prompt)
    history.replaying = True
    assert finish(history, [2, 3])
    await history.wait_applied()
    assert history.prompts == [prompt]
    history.replaying = False
    history.record("b", {"prompt_token_ids": [4, 5]})
    assert len(history.prompts) == 2


async def test_epoch_reset_releases_byte_and_token_budget(history):
    prompt = {"prompt_token_ids": [1], "media": "a"}
    history.max_bytes = encoded_size([prompt])
    history.max_tokens = 2
    history.record("old", prompt)
    pending = history.pending
    history.session.epoch += 1
    history.synchronize_epoch()
    assert pending.done() and pending.exception() is not None
    history.record("new", prompt)
    assert history.prompts == [prompt]
    assert history.pending_request == "new"


async def test_long_journal_append_and_completion_do_not_reencode_older_media(history, mocker):
    measure = mocker.spy(history, "_size")
    count = mocker.spy(history.policy, "token_count")
    for index in range(96):
        history.record(str(index), {"prompt_token_ids": [index], "media": "AQID" * 100})
        finish(history, [index + 1])
        await history.wait_applied()
    assert len(history.prompts) == 96
    assert measure.call_count == count.call_count == 192
    assert all(len(call.args[0]) == 1 for call in measure.call_args_list)
