# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Contracts for the standalone audio-to-score/YuE2 handoff tool."""

import json
import sys
from pathlib import Path

import pytest

from tools import sheetsage2_transcribe as cli

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def arguments(tmp_path):
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"audio is decoded by the upstream transcriber")
    return cli.create_parser().parse_args([str(audio), "--output-dir", str(tmp_path / "score"), "--trust-remote-code"])


@pytest.fixture
def backend(mocker):
    model = mocker.Mock()
    model.config = mocker.Mock(
        spec_set=["model_type", "base_model_revision"], model_type="sheetsage2", base_model_revision="parent-sha"
    )
    model.eval.return_value = model
    model.to.return_value = model
    model.transcribe.return_value = {"abc": "X:1\nM:4/4\nK:C\nC4|\n", "midi": b"MThd\x00\x00\x00\x06"}
    loader = mocker.Mock(return_value=model)
    torch = mocker.Mock(
        float32=object(),
        __version__="test",
        get_num_threads=lambda: 8,
        set_num_threads=mocker.Mock(),
        accelerator=mocker.Mock(is_available=lambda: False),
    )
    mocker.patch.dict(
        sys.modules,
        {"torch": torch, "transformers": mocker.Mock(AutoModel=mocker.Mock(from_pretrained=loader))},
    )
    return mocker.Mock(model=model, loader=loader, torch=torch)


@pytest.mark.parametrize("melody_only,expected_cot", [(True, "melody"), (False, "full")])
def test_score_handoff_preserves_unicode_and_selects_matching_cot(arguments, backend, melody_only, expected_cot):
    arguments.lyrics_file = arguments.audio.with_suffix(".txt")
    arguments.lyrics_file.write_text("[Verse]\n江边的清晨", encoding="utf-8")
    arguments.style = "  Chinese folk, bamboo flute  "
    arguments.yue2_model = "music-server"
    arguments.melody_only = melody_only
    arguments.seed = 123

    output = cli.transcribe(arguments)

    request = json.loads((output / "yue2_request.json").read_text())
    assert request == {
        "model": "music-server",
        "input": "[Verse]\n江边的清晨",
        "instructions": "Chinese folk, bamboo flute",
        "response_format": "wav",
        "stream": False,
        "seed": 123,
        "extra_params": {"cot": expected_cot, "abc": backend.model.transcribe.return_value["abc"]},
    }
    assert backend.model.transcribe.call_args.kwargs["melody_only"] == melody_only
    assert "max_new_tokens" not in request  # Leave natural song length to the server.


def test_default_model_pins_code_and_loads_adapters_in_fp32(arguments, backend):
    output = cli.transcribe(arguments)
    options = backend.loader.call_args.kwargs
    assert options["revision"] == options["code_revision"] == cli.DEFAULT_REVISION
    assert options["torch_dtype"] is backend.torch.float32
    backend.model.to.assert_called_once_with("cpu")
    assert not (output / "yue2_request.json").exists()


def test_local_offline_checkpoint_forwards_parent_without_default_hub_revision(arguments, backend, tmp_path, mocker):
    arguments.model = str(tmp_path / "SheetSage2")
    snapshot = Path(arguments.model)
    snapshot.mkdir()
    source = snapshot / "modeling_sheetsage2.py"
    direct = snapshot / "exports_sheetsage2.py"
    transitive = snapshot / "chord_spelling_sheetsage2.py"
    source.write_text("from .exports_sheetsage2 import export\n")
    direct.write_text("from .chord_spelling_sheetsage2 import chord\n")
    transitive.write_text("chord = 'C'\n")
    dynamic_modules = mocker.Mock()
    dynamic_modules.get_relative_import_files.return_value = [str(direct), str(transitive)]
    mocker.patch.dict(sys.modules, {"transformers.dynamic_module_utils": dynamic_modules})
    calls = mocker.Mock()
    calls.attach_mock(dynamic_modules.get_cached_module_file, "prime")
    calls.attach_mock(backend.loader, "load")
    arguments.base_model_path = tmp_path / "MERT"
    arguments.base_model_path.mkdir()
    arguments.local_files_only = True
    cli.transcribe(arguments)
    dynamic_modules.get_relative_import_files.assert_called_once_with(str(source))
    dynamic_modules.get_cached_module_file.assert_has_calls(
        [
            mocker.call(arguments.model, direct.name, local_files_only=True),
            mocker.call(arguments.model, transitive.name, local_files_only=True),
        ]
    )
    assert [call[0] for call in calls.mock_calls] == ["prime", "prime", "load"]
    backend.model.transcribe.assert_called_once()
    options = backend.loader.call_args.kwargs
    assert options["revision"] is None
    assert options["local_files_only"] is True
    assert options["base_model_path"] == str(arguments.base_model_path)


@pytest.mark.parametrize("limit", [0, -1, float("inf"), float("nan")])
def test_invalid_duration_fails_before_model_loading(arguments, backend, limit):
    arguments.max_seconds = limit
    with pytest.raises(ValueError, match="finite and positive"):
        cli.transcribe(arguments)
    backend.loader.assert_not_called()
    assert not arguments.output_dir.exists()


def test_nonempty_output_is_not_reused(arguments, backend):
    arguments.output_dir.mkdir()
    old = arguments.output_dir / "yue2_request.json"
    old.write_text("old request")
    with pytest.raises(ValueError, match="new or empty"):
        cli.transcribe(arguments)
    assert old.read_text() == "old request"
    backend.loader.assert_not_called()


def test_explicit_code_trust_is_required(arguments, backend):
    arguments.trust_remote_code = False
    with pytest.raises(ValueError, match="--trust-remote-code"):
        cli.transcribe(arguments)
    backend.loader.assert_not_called()


@pytest.mark.parametrize(
    "result,message",
    [
        ({"abc": "", "midi": b"MThd"}, "ABC export failed"),
        ({"abc": "X:1", "abc_error": "invalid beat grid", "midi": b"MThd"}, "invalid beat grid"),
        ({"abc": "X:1", "midi": b"invalid"}, "MIDI export failed"),
    ],
)
def test_failed_export_never_writes_a_yue2_request(arguments, backend, result, message):
    arguments.lyrics_file = arguments.audio.with_suffix(".txt")
    arguments.lyrics_file.write_text("lyrics")
    arguments.style = "folk"
    backend.model.transcribe.return_value = result
    with pytest.raises(RuntimeError, match=message):
        cli.transcribe(arguments)
    assert not (arguments.output_dir / "yue2_request.json").exists()
    assert not (arguments.output_dir / "manifest.json").exists()


def test_style_requires_lyrics_before_loading(arguments, backend):
    arguments.style = "folk"
    with pytest.raises(ValueError, match="supplied together"):
        cli.transcribe(arguments)
    backend.loader.assert_not_called()


def test_seed_requires_request_context_before_loading(arguments, backend):
    arguments.seed = 123
    with pytest.raises(ValueError, match="supply --lyrics-file and --style"):
        cli.transcribe(arguments)
    backend.loader.assert_not_called()
    assert not arguments.output_dir.exists()


@pytest.mark.parametrize("seed", [-1, 2**63])
def test_out_of_range_seed_fails_before_loading(arguments, backend, seed):
    arguments.lyrics_file = arguments.audio.with_suffix(".txt")
    arguments.lyrics_file.write_text("lyrics", encoding="utf-8")
    arguments.style = "folk"
    arguments.seed = seed
    with pytest.raises(ValueError, match="between 0 and"):
        cli.transcribe(arguments)
    backend.loader.assert_not_called()
    assert not arguments.output_dir.exists()


def test_cli_reports_export_failure_as_nonzero(arguments, backend, capsys):
    backend.model.transcribe.return_value = {"abc": "", "midi": b"MThd"}
    with pytest.raises(SystemExit) as exc:
        cli.main([str(arguments.audio), "--output-dir", str(arguments.output_dir), "--trust-remote-code"])
    assert exc.value.code == 1
    assert "ABC export failed" in capsys.readouterr().err
