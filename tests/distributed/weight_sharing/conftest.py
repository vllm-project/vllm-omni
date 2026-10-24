# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from tests.distributed.weight_sharing.helpers import load_weight_sharing


@pytest.fixture(scope="session")
def weight_sharing():
    return load_weight_sharing()
