# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Guard the Ring correctness harness against exiting zero without evidence.

``bench_ring_flash_attn_opt.py`` is the evidence that the shipped ring forward
matches the upstream one, and its exit status is what makes it usable as a
regression gate. Three ways it could report success without having validated
the thing under test:

* a candidate that raises where the baseline succeeded being recorded as a
  skip instead of a failure,
* a sweep whose every case was skipped still exiting zero, and
* a sweep that passed on a backend or ring degree where the packed path never
  runs being read as evidence for packing.

All three are regressions in the harness itself, so each gets a test, and each
asserts the specific exit status rather than merely "nonzero": 0 validated with
packing exercised, 1 a case failed, 2 every case skipped, 3 ran but never
packed.

The harness is a torchrun script, not an importable module: it calls
``_preload_cuda13_runtime()`` at import time, which may ``os.execv`` to put the
CUDA-13 runtime on ``LD_LIBRARY_PATH`` before ``import torch``. Importing it
in-process would therefore re-exec the test runner, so these tests drive it as
a subprocess. torchrun then reports only its own status and collapses any
nonzero child exit to 1, so the driver prints the harness's real code and
``_harness_exit`` reads it back.
"""

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

_BENCH_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BENCH_DIR.parents[2]

try:
    from vllm_omni.diffusion.attention.backends.ring.ring_globals import HAS_FA3 as _HAS_FA3
except Exception:  # pragma: no cover - import shape differs on unsupported platforms
    _HAS_FA3 = False

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None, reason="requires NVIDIA CUDA"
)

# The TORCH/SDPA backend needs no FlashAttention build, so it runs anywhere --
# but it is NOT one of the packing backends, so a torch sweep exercises the
# harness's bookkeeping rather than packing itself. The packed-path tests below
# force FA3 at ring degree 3, the lowest degree the gate packs at, and skip
# where that build is absent.
_DRIVER = textwrap.dedent(
    """
    import sys
    # Keep argv[0]: the harness may os.execv itself to fix LD_LIBRARY_PATH, and
    # that re-exec replays argv[0] as the script to run.
    sys.argv = [sys.argv[0], "--attn-type", {attn_type!r}, "--quick", "--iters", "1"]
    sys.path.insert(0, {bench_dir!r})
    import bench_ring_flash_attn_opt as bench
    {injection}
    try:
        bench.main()
    except SystemExit as exc:
        # Surface the harness's own status: torchrun reports only its own, and
        # collapses every nonzero child exit to 1.
        print("HARNESS_EXIT=%d" % (exc.code or 0), flush=True)
        raise
    else:
        print("HARNESS_EXIT=0", flush=True)
    """
)

_INJECT_CANDIDATE_RAISES = """
def _boom(*args, **kwargs):
    raise RuntimeError("injected candidate failure")

bench.ring_flash_attn_func = _boom
"""

_INJECT_ONE_RANK_REF_OOM = """
import os
import torch

# Rank 1 only. Call through to the real oracle first so its internal
# all-gathers complete on every rank -- raising before them would strand the
# peer inside the collective, which is the timeout case the harness puts out
# of scope -- and only then report it as out of memory.
if int(os.environ["RANK"]) == 1:
    _real_manual, _real_sdpa = bench._ref_manual, bench._ref_sdpa

    def _oom_manual(*a, **kw):
        _real_manual(*a, **kw)
        raise torch.cuda.OutOfMemoryError("injected single-rank oracle OOM")

    def _oom_sdpa(*a, **kw):
        _real_sdpa(*a, **kw)
        raise torch.cuda.OutOfMemoryError("injected single-rank oracle OOM")

    bench._ref_manual, bench._ref_sdpa = _oom_manual, _oom_sdpa
"""


_INJECT_EVERYTHING_SKIPS = """
bench._build_cases = lambda quick: [
    dict(
        label="forced unsupported",
        batch=1,
        seqlen=64,
        hq=4,
        hkv=4,
        d=64,
        needs={"a-capability-this-backend-lacks"},
    )
]
"""


def _run_harness(tmp_path, injection, *, attn_type="torch", ranks=1, timeout=1800):
    driver = tmp_path / "driver.py"
    driver.write_text(_DRIVER.format(bench_dir=str(_BENCH_DIR), injection=injection, attn_type=attn_type))

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(_REPO_ROOT), env.get("PYTHONPATH", "")]))
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={ranks}",
            str(driver),
        ],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc, (proc.stdout + proc.stderr)


def _harness_exit(out, ranks):
    """The harness's own exit status, once per rank and identical on all of them.

    ``proc.returncode`` is torchrun's status, and torchrun collapses any
    nonzero child exit to 1 -- so it cannot tell "a case failed" (1) apart
    from "nothing exercised packing" (3), which is the distinction these tests
    exist to pin. The driver prints the real code instead.

    ``ranks`` is required rather than inferred: a set of the reported codes
    cannot tell "both ranks agreed on 1" from "only one rank ever reported".
    The second is reachable -- the elastic agent SIGTERMs surviving workers
    once one exits nonzero, and the driver prints from its ``except
    SystemExit`` handler, after ``destroy_process_group`` -- and it is exactly
    the shape an asymmetric-verdict regression would take, with the silent
    rank being the one whose disagreement should have failed the run.
    """
    reported = [int(m) for m in re.findall(r"^HARNESS_EXIT=(\d+)$", out, re.M)]
    assert len(reported) == ranks, f"expected one HARNESS_EXIT per rank ({ranks}), got {reported}\n{out[-4000:]}"
    assert len(set(reported)) == 1, f"ranks disagreed on the exit status: {reported}\n{out[-4000:]}"
    code = reported[0]

    # The harness also prints its status on rank 0 for CI to grep, since the
    # launcher's exit code cannot carry it. Hold the two to the same value so
    # that line cannot drift away from the real status.
    announced = {int(m) for m in re.findall(r"^HARNESS_STATUS=(\d+) ", out, re.M)}
    assert announced == {code}, f"HARNESS_STATUS {announced or 'missing'} != actual exit {code}\n{out[-4000:]}"
    return code


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@requires_cuda
def test_candidate_only_failure_fails_the_sweep(tmp_path):
    """A candidate that raises must fail, not skip.

    The baseline is left working, so this is exactly "a supported candidate
    raising after a successful baseline" -- the regression the harness exists
    to catch.
    """
    proc, out = _run_harness(tmp_path, _INJECT_CANDIDATE_RAISES)

    assert proc.returncode != 0, f"candidate failure exited 0\n{out[-4000:]}"
    assert _harness_exit(out, 1) == 1, f"expected exit 1 (a case failed)\n{out[-4000:]}"
    assert "candidate raised RuntimeError" in out, out[-4000:]
    # Delimited: the summary also prints "<n> PASS-with-packing", whose 0 value
    # on this non-packing sweep makes a bare "0 PASS" substring always true.
    assert " 0 PASS," in out, f"a case still reported PASS\n{out[-4000:]}"


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@requires_cuda
def test_all_skipped_is_not_correctness_evidence(tmp_path):
    """A sweep that compared nothing must not exit zero."""
    proc, out = _run_harness(tmp_path, _INJECT_EVERYTHING_SKIPS)

    assert proc.returncode != 0, f"all-skipped sweep exited 0\n{out[-4000:]}"
    assert _harness_exit(out, 1) == 2, f"expected exit 2 (nothing validated)\n{out[-4000:]}"
    assert "no case was validated" in out, out[-4000:]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@requires_cuda
def test_healthy_non_packing_sweep_is_not_packing_evidence(tmp_path):
    """A clean torch sweep must not exit 0.

    Positive control for the exit gate, and for the reviewer's criterion that an
    unchanged (non-packing) path is not treated as exercising packing. Nothing
    is injected here: the sweep genuinely passes its comparisons, and must still
    refuse to look like evidence for this change.
    """
    proc, out = _run_harness(tmp_path, "")

    assert "packed-KV path: NOT EXERCISED" in out, out[-4000:]
    assert proc.returncode != 0, f"a non-packing sweep exited 0\n{out[-4000:]}"
    assert _harness_exit(out, 1) == 3, f"expected exit 3 (ran but never packed)\n{out[-4000:]}"
    assert "NOT evidence for packed-KV" in out, out[-4000:]
    assert " 0 FAIL," in out, f"the control sweep should not have failed cases\n{out[-4000:]}"


@hardware_test(res={"cuda": "L4"}, num_cards=3)
@requires_cuda
@pytest.mark.skipif(not _HAS_FA3, reason="packing is gated to FA/FA3; needs a source-built FA3")
def test_packing_sweep_passes_and_coordinates_across_ranks(tmp_path):
    """The real positive control: FA3 at ring degree 3 exits 0.

    Degree 3 is the lowest degree the shipped gate packs at, so this is both
    the positive control and the lower boundary of the gate. Degree 2 is
    covered by ``test_degree_2_does_not_pack`` below, which pins the other side
    of that boundary.

    It executes the cross-rank verdict agreement, as do the two multi-rank
    tests below, but all three place every rank in the same state -- only the
    oracle-OOM one makes the reduction aggregate ranks that disagree.
    """
    if torch.accelerator.device_count() < 3:
        pytest.skip("needs 3 GPUs")
    proc, out = _run_harness(tmp_path, "", attn_type="fa3", ranks=3)

    assert "packed-KV path: EXERCISED" in out, out[-4000:]
    assert proc.returncode == 0, f"expected exit 0, got {proc.returncode}\n{out[-6000:]}"
    assert _harness_exit(out, 3) == 0, f"expected exit 0 on every rank\n{out[-6000:]}"
    assert " 0 FAIL," in out, out[-4000:]
    # The label is printed unconditionally, so pin a NONZERO count: the point
    # is that cases passed *with packing live*, not that the word appeared.
    packed_counts = [int(m) for m in re.findall(r"(\d+) PASS-with-packing", out)]
    assert packed_counts and all(n > 0 for n in packed_counts), (
        f"no case passed with packing active: {packed_counts}\n{out[-4000:]}"
    )


@hardware_test(res={"cuda": "L4"}, num_cards=2)
@requires_cuda
@pytest.mark.skipif(not _HAS_FA3, reason="packing is gated to FA/FA3; needs a source-built FA3")
def test_degree_2_does_not_pack(tmp_path):
    """FA3 at ring degree 2 must report the gate closed, not exit 0.

    The gate has a degree floor of 3 because degree 2 measured 0.98x on the
    full ring call -- a small but consistent regression, since its single hop
    cannot pay for the packing copy. This pins that floor from below: a change
    that lowers it back to ``world_size > 1`` makes this sweep pack, which
    turns status 3 into status 0 and fails here.

    It is also the one healthy multi-rank sweep that must NOT count as evidence
    for packing, which is the reviewer criterion the exit-3 status exists for.
    """
    if torch.accelerator.device_count() < 2:
        pytest.skip("needs 2 GPUs")
    proc, out = _run_harness(tmp_path, "", attn_type="fa3", ranks=2)

    assert "packed-KV path: NOT EXERCISED" in out, out[-4000:]
    assert proc.returncode != 0, f"a non-packing sweep exited 0\n{out[-4000:]}"
    assert _harness_exit(out, 2) == 3, f"expected exit 3 on every rank\n{out[-6000:]}"
    # Healthy, just not packing: the cases must still have passed.
    assert " 0 FAIL," in out, out[-4000:]
    packed_counts = [int(m) for m in re.findall(r"(\d+) PASS-with-packing", out)]
    assert packed_counts and all(n == 0 for n in packed_counts), (
        f"degree 2 reported packing: {packed_counts}\n{out[-4000:]}"
    )


@hardware_test(res={"cuda": "L4"}, num_cards=2)
@requires_cuda
@pytest.mark.skipif(not _HAS_FA3, reason="packing is gated to FA/FA3; needs a source-built FA3")
def test_candidate_failure_fails_every_rank(tmp_path):
    """One rank's candidate error must fail the whole multi-rank sweep.

    The injection raises on every rank, which is the symmetric case the MAX
    all-reduce is there to turn into one agreed verdict. An error that strands a
    single rank inside the ring's P2P still needs a collective timeout; that is
    out of scope here and called out in the harness.
    """
    if torch.accelerator.device_count() < 2:
        pytest.skip("needs 2 GPUs")
    proc, out = _run_harness(tmp_path, _INJECT_CANDIDATE_RAISES, attn_type="fa3", ranks=2)

    assert proc.returncode != 0, f"candidate failure exited 0\n{out[-4000:]}"
    assert _harness_exit(out, 2) == 1, f"every rank must report exit 1\n{out[-4000:]}"
    assert "candidate raised RuntimeError" in out, out[-4000:]
    assert " 0 PASS," in out, f"a case still reported PASS\n{out[-4000:]}"


@hardware_test(res={"cuda": "L4"}, num_cards=3)
@requires_cuda
@pytest.mark.skipif(not _HAS_FA3, reason="packing is gated to FA/FA3; needs a source-built FA3")
def test_one_rank_oracle_oom_drops_the_oracle_everywhere(tmp_path):
    """Disagreeing ranks must be agreed, not silently mixed.

    The other multi-rank tests put every rank in an identical state, so the
    MAX all-reduce in ``_agree_fail`` runs but never has to aggregate ranks
    that disagree -- deleting the reduction keeps them green. This makes
    exactly one rank report its fp32 oracle as out of memory, which is the
    asymmetry the harness documents as expected because free memory differs
    per device on a shared box.

    Without the agreement, the rank that still had memory keeps a real
    reference diff while the other holds the -1.0 sentinel; the MAX then
    selects the real value and the row prints an ``opt-vs-ref`` number, so the
    sweep publishes a reference comparison only some of its ranks performed.
    With it, every rank drops the oracle and the row says so.

    Runs at degree 3 so the sweep still reaches status 0 when the oracle
    degrades: at degree 2 the packing gate is closed and the status would be 3
    for an unrelated reason, which would blunt the assertion below.
    """
    if torch.accelerator.device_count() < 3:
        pytest.skip("needs 3 GPUs")
    proc, out = _run_harness(tmp_path, _INJECT_ONE_RANK_REF_OOM, attn_type="fa3", ranks=3)

    # Losing the oracle is a documented degradation, not a failure: opt-vs-base
    # still runs on every case, so the sweep is still valid correctness evidence.
    assert proc.returncode == 0, f"a one-rank oracle OOM must degrade, not fail\n{out[-6000:]}"
    assert _harness_exit(out, 3) == 0, out[-6000:]
    assert "oom" in out, f"no row degraded to the oom marker\n{out[-6000:]}"
    survivors = re.findall(r"opt-vs-ref\s+(\d\.\d+e[+-]\d+)", out)
    assert not survivors, f"oracle still reported on rows after a rank lost it: {survivors}\n{out[-6000:]}"
