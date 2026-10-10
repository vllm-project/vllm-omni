# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import torch
import torch.nn as nn


def get_intmeanflow_time_mixer(dims):
    """ "
    Diagonal init as described in 3.3 https://arxiv.org/pdf/2510.07979
    """
    layer = nn.Linear(dims * 2, dims, bias=False)

    with torch.no_grad():
        target_weight = torch.zeros(dims, 2 * dims)
        target_weight[:, 0:dims] = torch.eye(dims)
        layer.weight.data = target_weight

    return layer
