# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for the MiniCPM-o 4.5 examples' request construction.

Each test runs the example's own code and hands its output to the code that
consumes it in a real run:

- the offline ``end2end.py`` ``main()``, with ``Omni`` replaced by a stand-in
  whose ``generate`` resolves the list through ``OmniBase``;
- the curl script, with ``curl`` replaced by a stub that records the request
  body, which then goes through the chat server's ``_to_sampling_params_list``;
- the SDK client's response handling, with a fake OpenAI client;
- the deploy name in ``barge_in_serve.sh``, resolved from the directory
  ``barge_in_client.py`` tells users to run it from.

Stage defaults and constraints come from the shipped ``minicpmo_4_5.yaml`` and
``MINICPMO_4_5_PIPELINE``.
"""

from __future__ import annotations

import base64
import contextlib
import importlib.util
import json
import shlex
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml
from vllm import SamplingParams

from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.entrypoints.omni_base import OmniBase
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO_4_5_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.example]

REPO_ROOT = Path(__file__).resolve().parents[2]
ONLINE_DIR = REPO_ROOT / "examples" / "online_serving"
MINICPMO_ONLINE_DIR = ONLINE_DIR / "minicpmo"
MINICPMO_OFFLINE_DIR = REPO_ROOT / "examples" / "offline_inference" / "minicpmo"
DEPLOY_YAML = REPO_ROOT / "vllm_omni" / "deploy" / "minicpmo_4_5.yaml"


def _deploy_stage_defaults() -> list[dict]:
    stages = yaml.safe_load(DEPLOY_YAML.read_text())["stages"]
    return [dict(stage["default_sampling_params"]) for stage in sorted(stages, key=lambda s: s["stage_id"])]


def _assert_keeps_deploy_defaults(params: SamplingParams, defaults: dict) -> None:
    # Compare normalized values: SamplingParams turns e.g. top_k=-1 into 0.
    expected = SamplingParams(**defaults)
    for key in defaults:
        assert getattr(params, key) == getattr(expected, key), key


@contextlib.contextmanager
def _load_example(name: str, path: Path) -> Iterator[ModuleType]:
    saved_path = list(sys.path)
    saved_modules = set(sys.modules)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path[:] = saved_path
        for added in set(sys.modules) - saved_modules:
            sys.modules.pop(added, None)


class _RecordingOmni:
    """Stands in for ``Omni``: resolves the caller's list the way ``Omni.generate`` does."""

    instances: list[_RecordingOmni] = []

    def __init__(self, **kwargs) -> None:
        self.num_stages = len(MINICPMO_4_5_PIPELINE.stages)
        self.default_sampling_params_list = [SamplingParams(**d) for d in _deploy_stage_defaults()]
        self.sampling_constraints_list = OmniBase._get_sampling_constraints_list(MINICPMO_4_5_PIPELINE.stages)
        self.resolved: list | None = None
        _RecordingOmni.instances.append(self)

    _apply_sampling_constraints = staticmethod(OmniBase._apply_sampling_constraints)
    resolve_sampling_params_list = OmniBase.resolve_sampling_params_list

    def generate(self, prompts, sampling_params_list, py_generator=False):
        self.resolved = list(self.resolve_sampling_params_list(sampling_params_list))
        return iter(())

    def start_profile(self, stages=None) -> None:
        pass

    def stop_profile(self, stages=None) -> None:
        pass

    def close(self) -> None:
        pass


def test_offline_end2end_passes_one_params_per_stage_and_keeps_talker_defaults(tmp_path, monkeypatch):
    with _load_example("_minicpmo_offline_end2end", MINICPMO_OFFLINE_DIR / "end2end.py") as end2end:
        monkeypatch.setattr(end2end, "Omni", _RecordingOmni)
        monkeypatch.setattr(
            sys,
            "argv",
            ["end2end.py", "--model", "openbmb/MiniCPM-o-4_5", "--query-type", "text", "--output-dir", str(tmp_path)],
        )
        _RecordingOmni.instances.clear()
        end2end.main(end2end.parse_args())

    (omni,) = _RecordingOmni.instances
    resolved = omni.resolved
    assert resolved is not None and len(resolved) == 3
    defaults = _deploy_stage_defaults()
    # The Talker samples codec tokens; its deploy limits must survive the example.
    _assert_keeps_deploy_defaults(resolved[1], defaults[1])
    _assert_keeps_deploy_defaults(resolved[2], defaults[2])


def _run_curl_example(tmp_path: Path, *args: str) -> dict:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    body_file = tmp_path / "body.json"
    (bin_dir / "curl").write_text(
        "#!/usr/bin/env bash\n"
        'while [ $# -gt 0 ]; do if [ "$1" = "-d" ]; then printf "%s" "$2" > "$BODY_FILE"; shift; fi; shift; done\n'
        "echo '{\"choices\": []}'\n"
    )
    (bin_dir / "jq").write_text("#!/usr/bin/env bash\ncat > /dev/null\necho 0\n")
    for stub in bin_dir.iterdir():
        stub.chmod(0o755)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "BODY_FILE": str(body_file), "HOST": "127.0.0.1", "PORT": "8099"}
    subprocess.run(
        ["bash", str(MINICPMO_ONLINE_DIR / "run_curl_multimodal_generation.sh"), *args],
        cwd=MINICPMO_ONLINE_DIR,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(body_file.read_text())


@pytest.mark.parametrize("args", [("text",), ("use_audio", '["text"]')], ids=["text_audio_out", "text_only"])
def test_curl_payload_resolves_against_deploy_defaults(tmp_path, args):
    body = _run_curl_example(tmp_path, *args)
    defaults = _deploy_stage_defaults()
    stage_configs = [
        # The stage config marks the tokenizer-owning stage as the comprehension stage.
        {"stage_type": "llm", "is_comprehension": stage.owns_tokenizer, "default_sampling_params": dict(d)}
        for stage, d in zip(MINICPMO_4_5_PIPELINE.stages, defaults, strict=True)
    ]
    serving = object.__new__(OmniOpenAIServingChat)
    serving.engine_client = SimpleNamespace(
        stage_configs=stage_configs,
        default_sampling_params_list=[SamplingParams(**d) for d in defaults],
    )
    resolved = serving._to_sampling_params_list(body["sampling_params_list"])
    assert len(resolved) == 3
    _assert_keeps_deploy_defaults(resolved[1], defaults[1])


def test_sdk_client_prints_text_and_saves_audio_from_one_choice(tmp_path, monkeypatch, capsys):
    wav = b"RIFF$\x00\x00\x00WAVEfmt fake"
    choice = SimpleNamespace(
        message=SimpleNamespace(
            content="Hello from MiniCPM-o", audio=SimpleNamespace(data=base64.b64encode(wav).decode())
        )
    )
    completion = SimpleNamespace(id="chatcmpl-test", choices=[choice])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: completion)))
    args = SimpleNamespace(
        model="openbmb/MiniCPM-o-4_5",
        query_type="text",
        prompt="Say hello.",
        prompts=None,
        modalities=None,
        num_concurrent_requests=1,
        stream=False,
        video_path=None,
        image_path=None,
        audio_path=None,
    )
    monkeypatch.chdir(tmp_path)
    path = MINICPMO_ONLINE_DIR / "openai_chat_completion_client_for_multimodal_generation.py"
    with _load_example("_minicpmo_sdk_client", path) as client_example:
        client_example.run_multimodal_generation(args, client)

    out = capsys.readouterr().out
    assert "Hello from MiniCPM-o" in out
    saved = list(tmp_path.glob("audio_*.wav"))
    assert len(saved) == 1 and saved[0].read_bytes() == wav


def test_barge_in_serve_deploy_config_resolves_from_examples_online_serving(monkeypatch):
    script = (MINICPMO_ONLINE_DIR / "barge_in_serve.sh").read_text()
    tokens = shlex.split(script.split("exec vllm-omni serve", 1)[1].replace("\\\n", " ").splitlines()[0])
    deploy_config = tokens[tokens.index("--deploy-config") + 1]
    # barge_in_client.py documents ``bash minicpmo/barge_in_serve.sh`` from examples/online_serving.
    monkeypatch.chdir(ONLINE_DIR)
    assert StageConfigFactory._load_user_deploy_config(deploy_config) is not None
