# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CI overlay publication must not expose incomplete YAML to other workers."""

from pathlib import Path

import pytest
import yaml

from tests.helpers import stage_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_ci_overlay_publishes_complete_document(tmp_path, monkeypatch):
    monkeypatch.setattr(stage_config, "_CI_GENERATED_DIR", tmp_path)
    out = tmp_path / "bagel.yaml"
    previous = "stages: [{stage_id: 99}]\n"
    out.write_text(previous, encoding="utf-8")
    dump = yaml.safe_dump

    def check_reader_during_write(data, stream, **kwargs):
        # Readers must keep seeing the previous document until publication.
        assert out.read_text(encoding="utf-8") == previous
        result = dump(data, stream, **kwargs)
        stream.flush()
        assert out.read_text(encoding="utf-8") == previous
        return result

    monkeypatch.setattr(yaml, "safe_dump", check_reader_during_write)
    assert Path(stage_config.get_deploy_config_path("ci/bagel.yaml")) == out
    expected = dict(stage_config._CI_OVERLAYS["bagel"])
    expected["base_config"] = str(stage_config._DEPLOY_DIR / expected["base_config"])
    assert yaml.safe_load(out.read_text(encoding="utf-8")) == expected
    assert list(tmp_path.iterdir()) == [out]
