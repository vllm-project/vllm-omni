# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Duplex artifact checks shared by benchmark unit tests and the perf runner."""


def assert_duplex_artifact_publication(result: dict, params: dict, num_prompt: int) -> None:
    if params.get("dataset_name") == "omni-duplex-eval":
        summary = result.get("omni_duplex_eval")
        assert isinstance(summary, dict), "Omni-DuplexEval summary is missing"
        assert summary.get("published") == num_prompt, "Omni-DuplexEval artifacts are incomplete"
        assert summary.get("artifact_errors") == [], "Omni-DuplexEval artifact publication failed"
    elif params.get("dataset_name") == "omniinteract":
        summary = result.get("omniinteract")
        assert isinstance(summary, dict), "OmniInteract summary is missing"
        assert (summary.get("total"), summary.get("success"), summary.get("failed")) == (
            num_prompt,
            num_prompt,
            0,
        ), "OmniInteract requests did not all succeed"
        assert summary.get("artifacts_complete") is True, "OmniInteract artifacts are incomplete"
