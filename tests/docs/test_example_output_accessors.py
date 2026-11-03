# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard the public accessor that examples and docs use to read omni outputs.

``OmniRequestOutput`` used to hold a nested ``RequestOutput`` in a
``request_output`` field.  Since #5146 it *inherits* ``RequestOutput``, so the
generation content lives on the object itself and the nested field is gone.

Call sites that kept the old accessor only fail once a real model runs, which
means a GPU nightly job is the first thing to notice.  These checks pin both
halves of the contract on CPU instead: the field is really gone, and no doc,
example, test or tool still reaches for it.
"""

import re
from pathlib import Path

import pytest

from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


ROOT_DIR = Path(__file__).parents[2]
SCANNED_DIRS = ("docs", "examples", "tests", "tools")
SCANNED_SUFFIXES = (".py", ".md")

# ``.request_outputs`` (plural, an OutputProcessor batch) is a different and
# still-valid attribute, so require a non-identifier character after the name.
NESTED_ACCESSOR = re.compile(r"\.request_output(?![\w])")


def _iter_scanned_files():
    for directory in SCANNED_DIRS:
        for path in sorted((ROOT_DIR / directory).rglob("*")):
            if path.suffix in SCANNED_SUFFIXES and path.resolve() != Path(__file__).resolve():
                yield path


def test_omni_request_output_has_no_nested_request_output():
    assert not hasattr(OmniRequestOutput(request_id="req-accessor"), "request_output")


def test_no_source_reads_the_nested_request_output():
    offenders = []
    for path in _iter_scanned_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if NESTED_ACCESSOR.search(line):
                offenders.append(f"{path.relative_to(ROOT_DIR)}:{lineno}: {line.strip()}")

    assert not offenders, (
        "OmniRequestOutput no longer nests a RequestOutput; read the content off the "
        "output itself (e.g. `outputs[0].images`) instead of `.request_output`:\n" + "\n".join(offenders)
    )
