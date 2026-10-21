# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from vllm_omni.engine.duplex.commands import UpdateSession
from vllm_omni.engine.duplex.realtime_commands import translate_realtime_command

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("profile_name", "config"),
    [
        ("joyai-turn", {"speechInput": True, "turnMode": "manual"}),
        ("joyai-turn", {"speechInput": True, "turnMode": "vad"}),
        ("aura-ptt", {}),
        ("qwen3-turn", {"turnMode": "vad"}),
        ("qwen3-turn", {"turnMode": "semantic_vad"}),
    ],
)
def test_browser_profile_session_update_passes_public_validator(profile_name, config):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to generate the real browser profile payload")
    assert node is not None
    profiles = Path(__file__).resolve().parents[2] / "examples/online_serving/realtime_web/app/static/profiles"
    result = subprocess.run(
        [
            node,
            "-e",
            """
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const [root, name, configJson] = process.argv.slice(1);
const context = vm.createContext({ URL });
for (const file of ['common', name.replaceAll('-', '_')]) {
  vm.runInContext(fs.readFileSync(path.join(root, `${file}.js`), 'utf8'), context);
}
const config = JSON.parse(configJson);
const profile = context.OmniRealtimeProfiles[name](config);
process.stdout.write(JSON.stringify(profile.initialMessages(config, 'Watch the camera.')));
""",
            str(profiles),
            profile_name,
            json.dumps(config),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    for message in json.loads(result.stdout):
        command = translate_realtime_command(message)
        assert isinstance(command, UpdateSession)
        assert command.patch == message["session"]
