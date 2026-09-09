# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib.util
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from unittest.mock import Mock

import pytest
from pytest_mock import MockerFixture

import vllm_omni.diffusion.worker.diffusion_worker as worker_module

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@dataclass(frozen=True)
class _FakeGroup:
    world_size: int = 1
    rank_in_group: int = 0


_GROUP_GETTERS = (
    "get_dp_group",
    "get_pp_group",
    "get_sp_group",
    "get_cfg_group",
    "get_tp_group",
    "get_fs_group",
    "get_hsdp_replicate_group",
    "get_ep_group",
)


def _patch_groups(
    mocker: MockerFixture,
    groups: dict[str, _FakeGroup],
) -> dict[str, Mock]:
    patched_getters = {}
    for getter_name in _GROUP_GETTERS:
        patched_getters[getter_name] = mocker.patch.object(
            worker_module,
            getter_name,
            return_value=groups.get(getter_name, _FakeGroup()),
        )
    return patched_getters


def test_setup_uses_default_name_before_model_parallel_init(
    mocker: MockerFixture,
) -> None:
    mocker.patch.object(
        worker_module,
        "model_parallel_is_initialized",
        return_value=False,
    )
    group_getters = _patch_groups(mocker, {})
    set_process_title = mocker.patch.object(worker_module, "set_process_title")
    decorate_logs = mocker.patch.object(worker_module, "decorate_logs")

    worker_module._setup_diffusion_worker_proc_title_and_log_prefix(enable_ep=True, use_hsdp=True)

    set_process_title.assert_called_once_with(
        name="DiffusionWorker",
        prefix="vLLM-Omni",
    )
    decorate_logs.assert_called_once_with("DiffusionWorker")
    for group_getter in group_getters.values():
        group_getter.assert_not_called()


@pytest.mark.parametrize(
    ("groups", "enable_ep", "use_hsdp", "hsdp_replicate_size", "expected_name"),
    [
        ({}, False, False, 1, "DiffusionWorker"),
        ({"get_tp_group": _FakeGroup(2, 1)}, False, False, 1, "DiffusionWorker_TP1"),
        ({"get_dp_group": _FakeGroup(4, 2)}, False, False, 1, "DiffusionWorker_DP2"),
        ({"get_pp_group": _FakeGroup(2, 1)}, False, False, 1, "DiffusionWorker_PP1"),
        (
            {
                "get_cfg_group": _FakeGroup(2, 1),
                "get_tp_group": _FakeGroup(2, 1),
            },
            False,
            False,
            1,
            "DiffusionWorker_CFG1_TP1",
        ),
        (
            {
                "get_dp_group": _FakeGroup(2, 1),
                "get_sp_group": _FakeGroup(4, 2),
                "get_tp_group": _FakeGroup(2, 1),
            },
            False,
            False,
            1,
            "DiffusionWorker_DP1_SP2_TP1",
        ),
        (
            {"get_fs_group": _FakeGroup(4, 3)},
            False,
            True,
            1,
            "DiffusionWorker_FS3",
        ),
        (
            {
                "get_sp_group": _FakeGroup(2, 1),
                "get_fs_group": _FakeGroup(4, 3),
            },
            False,
            True,
            1,
            "DiffusionWorker_SP1_FS3",
        ),
        (
            {
                "get_fs_group": _FakeGroup(1, 0),
                "get_hsdp_replicate_group": _FakeGroup(2, 1),
            },
            False,
            True,
            2,
            "DiffusionWorker_RP1",
        ),
        ({"get_ep_group": _FakeGroup(4, 2)}, True, False, 1, "DiffusionWorker_EP2"),
        ({}, True, False, 1, "DiffusionWorker"),
    ],
    ids=[
        "all-singleton",
        "tp-only",
        "dp-only",
        "pp-only",
        "tp-cfg",
        "dp-sp-tp",
        "hsdp-only",
        "hsdp-sp",
        "hsdp-replicated",
        "expert-parallel",
        "singleton-ep",
    ],
)
def test_setup_uses_initialized_parallel_groups(
    mocker: MockerFixture,
    groups: dict[str, _FakeGroup],
    enable_ep: bool,
    use_hsdp: bool,
    hsdp_replicate_size: int,
    expected_name: str,
) -> None:
    mocker.patch.object(
        worker_module,
        "model_parallel_is_initialized",
        return_value=True,
    )
    group_getters = _patch_groups(mocker, groups)
    set_process_title = mocker.patch.object(worker_module, "set_process_title")
    decorate_logs = mocker.patch.object(worker_module, "decorate_logs")

    worker_module._setup_diffusion_worker_proc_title_and_log_prefix(
        enable_ep=enable_ep,
        use_hsdp=use_hsdp,
        hsdp_replicate_size=hsdp_replicate_size,
    )

    set_process_title.assert_called_once_with(
        name=expected_name,
        prefix="vLLM-Omni",
    )
    decorate_logs.assert_called_once_with(expected_name)
    if use_hsdp:
        group_getters["get_fs_group"].assert_called_once_with()
        if hsdp_replicate_size > 1:
            group_getters["get_hsdp_replicate_group"].assert_called_once_with()
        else:
            group_getters["get_hsdp_replicate_group"].assert_not_called()
    else:
        group_getters["get_fs_group"].assert_not_called()
        group_getters["get_hsdp_replicate_group"].assert_not_called()
    if enable_ep:
        group_getters["get_ep_group"].assert_called_once_with()
    else:
        group_getters["get_ep_group"].assert_not_called()


def test_missing_setproctitle_is_non_fatal(mocker: MockerFixture) -> None:
    mocker.patch.object(
        worker_module,
        "model_parallel_is_initialized",
        return_value=False,
    )
    mocker.patch.dict(sys.modules, {"setproctitle": None})
    decorate_logs = mocker.patch.object(worker_module, "decorate_logs")

    worker_module._setup_diffusion_worker_proc_title_and_log_prefix(enable_ep=False, use_hsdp=False)

    decorate_logs.assert_called_once_with("DiffusionWorker")


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("ps") is None or importlib.util.find_spec("setproctitle") is None,
    reason="requires Linux ps and setproctitle",
)
def test_process_title_is_visible_through_ps() -> None:
    expected_name = "vLLM-Omni::DiffusionWorker_TP1"
    child_script = textwrap.dedent(
        """
        import time

        from vllm.utils.system_utils import set_process_title


        set_process_title(
            name="DiffusionWorker_TP1",
            prefix="vLLM-Omni",
        )
        time.sleep(30)
        """
    )
    process = subprocess.Popen(
        [sys.executable, "-c", child_script],
        stderr=subprocess.PIPE,
        text=True,
    )

    observed_title = ""
    try:
        for _ in range(100):
            if process.poll() is not None:
                stderr = process.stderr.read() if process.stderr is not None else ""
                pytest.fail(f"child process exited before title check: {stderr}")
            result = subprocess.run(
                ["ps", "-o", "args=", "-p", str(process.pid)],
                check=False,
                capture_output=True,
                text=True,
            )
            observed_title = result.stdout.strip()
            if expected_name in observed_title:
                break
            time.sleep(0.1)
        else:
            pytest.fail(f"expected {expected_name!r} in process title, got {observed_title!r}")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
