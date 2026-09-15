# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "examples" / "online_serving" / "realtime_web"
APP_ROOT = ROOT / "app"
STATIC_ROOT = APP_ROOT / "static"

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_page_exposes_focused_call_conversation_and_log_surfaces():
    html = (APP_ROOT / "index.html").read_text(encoding="utf-8")

    assert 'id="callButton"' in html
    assert 'id="muteButton"' in html
    assert 'id="connectionState"' in html
    assert 'id="modelState"' in html
    assert 'id="conversation"' in html
    assert 'id="eventLog"' in html
    assert "<details" in html
    assert "Automatic barge-in" not in html
    assert "Server VAD" not in html


def test_shared_shell_delegates_wire_contract_to_profiles():
    source = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    assert "profile.url(config, window.location.href)" in source
    assert "profile.initialMessages(config, instructions)" in source
    assert "profile.append(" in source
    assert "profile.mapEvent(event)" in source
    assert "native_duplex" not in source
    assert "server_vad" not in source


def test_native_profile_preserves_session_and_capture_policy():
    source = (STATIC_ROOT / "profiles/minicpm_native.js").read_text(encoding="utf-8")
    assert "url.searchParams.set('native_duplex', '1')" in source
    assert "auto_response: true" in source
    assert "if (config.refAudio) session.ref_audio = config.refAudio;" in source
    assert "halfDuplex: false" in source
    assert "commitMessages: () => []" in source
    assert "video_frames" in source
    assert "playback.ack" in source


def test_client_has_transactional_cleanup_and_visible_event_logging():
    source = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    assert "async function cleanupSession" in source
    assert "waitForSessionClosed" in source
    assert "track.stop()" in source
    assert "clearInterval(sendTimer)" in source
    assert "appendEventLog(event)" in source
    assert "stopSession({ terminal: false })" in source


def test_audio_worklets_define_capture_and_playback_processors():
    capture = (STATIC_ROOT / "pcm_worklet.js").read_text(encoding="utf-8")
    playback = (STATIC_ROOT / "playback_worklet.js").read_text(encoding="utf-8")

    assert "registerProcessor('fullduplex-pcm-capture'" in capture
    assert "Int16Array" in capture
    assert "registerProcessor('fullduplex-pcm-playback'" in playback
    assert "playback-drained" in playback
    assert "clear" in playback


def test_audio_worklet_urls_use_the_static_asset_version():
    server = (ROOT / "server.py").read_text(encoding="utf-8")
    app = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert '"appVersion": app_version' in server
    assert 'STATIC_DIR / "playback_worklet.js"' in server
    assert 'STATIC_DIR / "pcm_worklet.js"' in server
    assert "staticAssetUrl('static/playback_worklet.js')" in app
    assert "staticAssetUrl('static/pcm_worklet.js')" in app


def test_playback_worklet_buffers_first_400ms_and_reports_underruns():
    app = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    playback = (STATIC_ROOT / "playback_worklet.js").read_text(encoding="utf-8")

    assert "INITIAL_PLAYBACK_BUFFER_MS = 400" in app
    assert "initialBufferMs" in app
    assert "responseId" in app
    assert "playback-underrun" in app
    assert "underrunMs" in app
    assert "initialBufferFrames" in playback
    assert "playback-underrun" in playback
    assert "underrunFrames" in playback
    assert "underrunMs" in playback


def test_playback_worklet_waits_before_playing_and_rebuffers_after_underrun():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required for the AudioWorklet regression test")
    assert node is not None

    script = textwrap.dedent(
        """
        const fs = require('fs');
        const vm = require('vm');

        global.sampleRate = 1000;
        global.AudioWorkletProcessor = class {
          constructor() {
            this.port = { onmessage: null, postMessage: () => {} };
          }
        };
        let Processor = null;
        global.registerProcessor = (_name, processor) => { Processor = processor; };
        vm.runInThisContext(fs.readFileSync(process.argv[1], 'utf8'));

        const processor = new Processor();
        const render = () => {
          const output = new Float32Array(100);
          processor.process([], [[output]]);
          return output;
        };
        const assert = (condition, message) => {
          if (!condition) throw new Error(message);
        };

        const first = new Int16Array(150);
        first.fill(16384);
        processor.handleMessage({
          type: 'audio',
          pcm: first,
          responseId: 'response-1',
          initialBufferMs: 400,
        });
        assert(!processor.started, 'large first delta must not bypass wall-clock prebuffer');
        assert(render().every((sample) => sample === 0), 'first prebuffer render must be silent');
        assert(render().every((sample) => sample === 0), 'second prebuffer render must be silent');
        assert(render().every((sample) => sample === 0), 'third prebuffer render must be silent');
        assert(render().every((sample) => sample === 0), 'fourth prebuffer render must be silent');
        const firstPlayback = render();
        assert(firstPlayback.some((sample) => sample !== 0), 'playback must start after prebuffer');

        const underrun = render();
        assert(!processor.started, 'an empty queue must return to buffering');
        assert(underrun[underrun.length - 1] === 0, 'underrun boundary must fade to zero');

        const resumed = new Int16Array(300);
        resumed.fill(8192);
        processor.handleMessage({
          type: 'audio',
          pcm: resumed,
          responseId: 'response-1',
          initialBufferMs: 400,
        });
        assert(render().every((sample) => sample === 0), 'resume must rebuild jitter buffer');
        assert(render().every((sample) => sample === 0), 'resume must keep rebuilding jitter buffer');
        assert(render().every((sample) => sample === 0), 'resume must keep waiting');
        assert(render().every((sample) => sample === 0), 'resume must wait the full buffer interval');
        const resumedPlayback = render();
        assert(resumedPlayback.some((sample) => sample !== 0), 'playback must resume after rebuffer');
        assert(
          Math.abs(resumedPlayback[0]) < Math.abs(resumedPlayback[50]),
          'resumed playback must fade in instead of jumping from silence',
        );
        """
    )
    subprocess.run(
        [node, "-e", script, str(STATIC_ROOT / "playback_worklet.js")],
        check=True,
        capture_output=True,
        text=True,
    )


def test_playback_worklet_fades_terminal_drain_to_zero():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required for the AudioWorklet regression test")
    assert node is not None

    script = textwrap.dedent(
        """
        const fs = require('fs');
        const vm = require('vm');

        global.sampleRate = 1000;
        const messages = [];
        global.AudioWorkletProcessor = class {
          constructor() {
            this.port = {
              onmessage: null,
              postMessage: (message) => messages.push(message),
            };
          }
        };
        let Processor = null;
        global.registerProcessor = (_name, processor) => { Processor = processor; };
        vm.runInThisContext(fs.readFileSync(process.argv[1], 'utf8'));

        const processor = new Processor();
        const pcm = new Int16Array(100);
        pcm.fill(16384);
        processor.handleMessage({
          type: 'audio',
          pcm,
          responseId: 'response-terminal',
          initialBufferMs: 400,
        });
        processor.handleMessage({
          type: 'drain',
          responseId: 'response-terminal',
        });

        const output = new Float32Array(100);
        processor.process([], [[output]]);
        const assert = (condition, message) => {
          if (!condition) throw new Error(message);
        };

        assert(output.some((sample) => sample !== 0), 'terminal audio must still play');
        assert(output[output.length - 1] === 0, 'terminal drain must fade the final sample to zero');
        const tail = Array.from(output.slice(-5), Math.abs);
        assert(
          tail.every((sample, index) => index === 0 || sample <= tail[index - 1]),
          'terminal drain fade must decrease monotonically',
        );
        assert(
          messages.filter((message) => message.type === 'playback-drained').length === 1,
          'terminal drain must be reported exactly once after playback',
        );
        """
    )
    subprocess.run(
        [node, "-e", script, str(STATIC_ROOT / "playback_worklet.js")],
        check=True,
        capture_output=True,
        text=True,
    )
