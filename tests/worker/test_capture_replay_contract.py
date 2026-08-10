# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static coupling check for the ``has_preprocess`` capture/replay buffers.

For models with ``has_preprocess``, CUDA-graph *capture* (``_dummy_run``) and
*replay* (``_preprocess``) MUST read from the same pre-allocated buffers
(``self.input_ids.gpu`` + ``self.inputs_embeds.gpu``); otherwise replay reads a
buffer capture never wrote and produces garbage. This is a source-level check
(no GPU) that catches a one-sided edit to either branch.

The check is AST-scoped: it inspects ONLY the body of the ``has_preprocess``
guarded branch (not a line window), so a buffer read removed from that branch
cannot be masked by an adjacent ``else``/``elif`` that happens to read it. It
matches by attribute chain (``self.input_ids.gpu``), so local variable renames
do not break it.

Scope: the BASE runner (``OmniGPUModelRunner``) only. The generation runner is
KNOWN to lack the ``has_preprocess`` branch; extend this
check to the generation file when it adds the branch.
"""

import ast
import inspect
import textwrap

import pytest

from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_REQUIRED_BUFFERS = {"self.input_ids.gpu", "self.inputs_embeds.gpu"}


def _attr_chain(node: ast.AST) -> str | None:
    """Reconstruct a dotted ``self.x.y`` chain from an ast.Attribute node."""
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def _attr_chains_in(nodes: list[ast.stmt]) -> set[str]:
    chains: set[str] = set()
    for stmt in nodes:
        for inner in ast.walk(stmt):
            if isinstance(inner, ast.Attribute):
                chain = _attr_chain(inner)
                if chain is not None:
                    chains.add(chain)
    return chains


def _has_preprocess_branch_reads_both_buffers(method) -> bool:
    """True iff SOME ``if has_preprocess:`` branch body reads BOTH buffers.

    Only the guarded branch's own ``body`` is inspected (elif/else excluded),
    which is what makes a one-sided buffer removal fail.
    """
    src = textwrap.dedent(inspect.getsource(method))
    tree = ast.parse(src)
    guarded = [node for node in ast.walk(tree) if isinstance(node, ast.If) and "has_preprocess" in ast.dump(node.test)]
    assert guarded, f"{method.__qualname__}: has_preprocess branch removed entirely"
    return any(_REQUIRED_BUFFERS <= _attr_chains_in(node.body) for node in guarded)


@pytest.mark.parametrize("method_name", ["_dummy_run", "_preprocess"])
def test_has_preprocess_branch_reads_shared_buffers(method_name):
    method = getattr(OmniGPUModelRunner, method_name)
    assert _has_preprocess_branch_reads_both_buffers(method), (
        f"{OmniGPUModelRunner.__name__}.{method_name}: the has_preprocess branch no "
        "longer reads BOTH self.input_ids.gpu and self.inputs_embeds.gpu — capture "
        "and replay would diverge."
    )
