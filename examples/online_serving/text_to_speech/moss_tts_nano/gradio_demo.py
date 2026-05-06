"""Gradio demo for MOSS-TTS-Nano with gapless streaming audio playback.

Uses a custom AudioWorklet-based player (adapted from the Qwen3-TTS demo)
for gap-free streaming.  Audio is streamed from the vLLM-Omni server
through a same-origin proxy and played via the Web Audio API's AudioWorklet.

MOSS-TTS-Nano outputs 48 kHz stereo (2-channel interleaved L,R,L,R,...).
The AudioWorklet downmixes stereo to mono during playback.

Also supports non-streaming mode (full audio download) via gr.Audio.

Usage:
    # Start the server first (see run_server.sh), then:
    python gradio_demo.py --api-base http://localhost:8091

    # Or use run_gradio_demo.sh to start both server and demo together.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging

try:
    import gradio as gr
except ImportError:
    raise ImportError("gradio is required to run this demo. Install it with: pip install 'vllm-omni[demo]'") from None

import httpx
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

logger = logging.getLogger(__name__)

# MOSS-TTS-Nano outputs 48 kHz stereo (2-channel interleaved).
PCM_SAMPLE_RATE = 48000
PCM_CHANNELS = 2
DEFAULT_API_BASE = "http://localhost:8091"

# MOSS-TTS-Nano is voice-cloning-only — no built-in presets. Users must
# upload a reference audio clip + transcript in the UI.

# ── AudioWorklet processor (loaded in browser via Blob URL) ──────────
# Downmixes stereo-interleaved int16 PCM to mono float32 for playback.
WORKLET_JS = r"""
class TTSPlaybackProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.queue = [];
        this.buf = null;
        this.pos = 0;
        this.playing = false;
        this.played = 0;
        this.port.onmessage = (e) => {
            if (e.data && e.data.type === 'clear') {
                this.queue = []; this.buf = null; this.pos = 0; this.played = 0;
                if (this.playing) { this.playing = false; this.port.postMessage({type:'stopped'}); }
                return;
            }
            // Input: Int16Array of stereo-interleaved samples [L0,R0,L1,R1,...]
            // Convert to mono float32 by averaging L+R channels.
            const pcm = e.data;
            const monoLen = Math.floor(pcm.length / 2);
            const mono = new Float32Array(monoLen);
            for (let i = 0; i < monoLen; i++) {
                mono[i] = (pcm[i * 2] + pcm[i * 2 + 1]) / 65536.0;
            }
            this.queue.push(mono);
        };
    }
    process(inputs, outputs) {
        const out = outputs[0][0];
        for (let i = 0; i < out.length; i++) {
            if (!this.buf || this.pos >= this.buf.length) {
                if (this.queue.length > 0) {
                    this.buf = this.queue.shift(); this.pos = 0;
                } else {
                    for (let j = i; j < out.length; j++) out[j] = 0;
                    if (this.playing) { this.playing = false; this.port.postMessage({type:'stopped', played:this.played}); }
                    return true;
                }
            }
            out[i] = this.buf[this.pos++];
            this.played++;
        }
        if (!this.playing) { this.playing = true; this.port.postMessage({type:'started'}); }
        return true;
    }
}
registerProcessor('tts-playback-processor', TTSPlaybackProcessor);
"""

# ── Player HTML (container with metric cards) ────────────────────────
PLAYER_HTML = """
<div id="tts-player">
  <div style="display:flex; align-items:center; gap:10px;">
    <div id="tts-status-dot" style="width:10px;height:10px;border-radius:50%;background:#ccc;flex-shrink:0;"></div>
    <span id="tts-status" style="font-weight:600;font-size:1.05em;">Ready</span>
    <button id="tts-stop-btn" onclick="window.ttsStop()"
      style="display:none; margin-left:auto; padding:5px 16px; border-radius:6px; border:1px solid #EF5552;
             background:#fff; color:#EF5552; cursor:pointer; font-size:0.85em; transition:all 0.15s;">Stop</button>
  </div>
  <div id="tts-metrics" style="display:none; grid-template-columns:repeat(4,1fr); gap:10px; margin-top:12px;">
    <div style="background:#f8f9fa;border-radius:6px;padding:8px 10px;text-align:center;">
      <div style="font-size:0.7em;text-transform:uppercase;color:#888;letter-spacing:0.5px;margin-bottom:2px;">TTFP</div>
      <div id="tts-m-ttfp" style="font-size:1.2em;font-weight:700;color:#333;">—</div>
    </div>
    <div style="background:#f8f9fa;border-radius:6px;padding:8px 10px;text-align:center;">
      <div style="font-size:0.7em;text-transform:uppercase;color:#888;letter-spacing:0.5px;margin-bottom:2px;">RTF</div>
      <div id="tts-m-rtf" style="font-size:1.2em;font-weight:700;color:#333;">—</div>
    </div>
    <div style="background:#f8f9fa;border-radius:6px;padding:8px 10px;text-align:center;">
      <div style="font-size:0.7em;text-transform:uppercase;color:#888;letter-spacing:0.5px;margin-bottom:2px;">Audio</div>
      <div id="tts-m-dur" style="font-size:1.2em;font-weight:700;color:#333;">—</div>
    </div>
    <div style="background:#f8f9fa;border-radius:6px;padding:8px 10px;text-align:center;">
      <div style="font-size:0.7em;text-transform:uppercase;color:#888;letter-spacing:0.5px;margin-bottom:2px;">Speed</div>
      <div id="tts-m-speed" style="font-size:1.2em;font-weight:700;color:#333;">—</div>
    </div>
  </div>
  <div id="tts-rtf-bar-wrap" style="display:none; background:#e8ecf1; border-radius:4px; height:20px; overflow:hidden; position:relative; margin-top:10px;">
    <div id="tts-rtf-bar" style="height:100%; border-radius:4px; transition:width 0.3s ease, background 0.3s ease; width:0%;"></div>
    <span id="tts-rtf-label" style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:0.75em;font-weight:600;color:#444;"></span>
  </div>
  <div id="tts-elapsed" style="display:none; margin-top:6px; font-size:0.8em; color:#999; text-align:right;"></div>
</div>
"""


def _build_player_js(sample_rate: int) -> str:
    """Build the JavaScript that powers the AudioWorklet player."""
    return f"""
    <script>
    const SR = {sample_rate};
    const WC = {json.dumps(WORKLET_JS)};
    let ctx = null, node = null, abort = null, gen = false, st = {{}};

    async function init() {{
        if (ctx) return;
        ctx = new AudioContext({{ sampleRate: SR }});
        const b = new Blob([WC], {{ type: 'application/javascript' }});
        const u = URL.createObjectURL(b);
        await ctx.audioWorklet.addModule(u);
        URL.revokeObjectURL(u);
        node = new AudioWorkletNode(ctx, 'tts-playback-processor');
        node.connect(ctx.destination);
        node.port.onmessage = (e) => {{
            if (e.data.type === 'started') setStatus('Playing...', '#2E7D32');
            else if (e.data.type === 'stopped' && !gen) {{
                setStatus('Done', '#2E7D32'); showStats(true);
                const btn = document.getElementById('tts-stop-btn');
                if (btn) btn.style.display = 'none';
            }}
        }};
    }}

    function setStatus(text, color) {{
        const s = document.getElementById('tts-status');
        const d = document.getElementById('tts-status-dot');
        if (s) s.textContent = text;
        if (d) d.style.background = color || '#ccc';
    }}

    function showStats(fin) {{
        if (!st.t0) return;
        const elapsed = (fin && st.streamEnd ? (st.streamEnd - st.t0) : (performance.now() - st.t0)) / 1000;
        // st.samples counts stereo int16 samples; mono frames = samples / 2
        const dur = (st.samples / 2) / SR;
        const mTtfp = document.getElementById('tts-m-ttfp');
        const mRtf = document.getElementById('tts-m-rtf');
        const mDur = document.getElementById('tts-m-dur');
        const mSpeed = document.getElementById('tts-m-speed');
        const bar = document.getElementById('tts-rtf-bar');
        const barLabel = document.getElementById('tts-rtf-label');
        const elapsedEl = document.getElementById('tts-elapsed');

        if (mTtfp && st.ttfp != null) mTtfp.textContent = st.ttfp.toFixed(0) + 'ms';
        if (mDur) mDur.textContent = dur.toFixed(1) + 's';

        if (dur > 0 && elapsed > 0) {{
            const rtf = elapsed / dur;
            const speed = 1 / rtf;
            if (mRtf) {{
                mRtf.textContent = rtf.toFixed(2) + 'x';
                mRtf.style.color = rtf < 1 ? '#2E7D32' : rtf < 1.5 ? '#e8a317' : '#EF5552';
            }}
            if (mSpeed) {{
                mSpeed.textContent = speed.toFixed(1) + 'x';
                mSpeed.style.color = speed > 1 ? '#2E7D32' : speed > 0.7 ? '#e8a317' : '#EF5552';
            }}
            if (bar) {{
                const pct = Math.min(speed / 3 * 100, 100);
                bar.style.width = pct + '%';
                bar.style.background = speed > 1 ? 'linear-gradient(90deg,#66BB6A,#2E7D32)' : speed > 0.7 ? 'linear-gradient(90deg,#e8a317,#f0b866)' : 'linear-gradient(90deg,#EF5552,#f87171)';
            }}
            if (barLabel) barLabel.textContent = speed.toFixed(1) + 'x realtime';
        }}
        if (elapsedEl) {{
            elapsedEl.style.display = 'block';
            elapsedEl.textContent = fin ? 'Completed in ' + elapsed.toFixed(1) + 's  (' + st.chunks + ' chunks)' : elapsed.toFixed(1) + 's elapsed  (' + st.chunks + ' chunks)';
        }}
    }}

    window.ttsStop = function() {{
        if (abort) abort.abort();
        if (node) node.port.postMessage({{ type: 'clear' }});
        gen = false;
        setStatus('Stopped', '#999');
        const btn = document.getElementById('tts-stop-btn');
        if (btn) btn.style.display = 'none';
    }};

    window.ttsGenerate = async function(payload) {{
        try {{ await init(); if (ctx.state === 'suspended') await ctx.resume(); }}
        catch (e) {{ const s = document.getElementById('tts-status'); if (s) s.textContent = 'Audio init error: ' + e.message; return; }}

        if (abort) abort.abort();
        node.port.postMessage({{ type: 'clear' }});
        await new Promise(r => setTimeout(r, 50));
        node.port.postMessage({{ type: 'clear' }});

        gen = true;
        st = {{ t0: null, chunks: 0, samples: 0, ttfp: null }};
        setStatus('Connecting...', '#2E7D32');
        const bEl = document.getElementById('tts-stop-btn');
        if (bEl) bEl.style.display = 'inline-block';
        const mp = document.getElementById('tts-metrics');
        if (mp) {{ mp.style.display = 'grid'; ['tts-m-ttfp','tts-m-rtf','tts-m-dur','tts-m-speed'].forEach(id => {{ const e = document.getElementById(id); if(e) {{ e.textContent = '\\u2014'; e.style.color = '#333'; }} }}); }}
        const bw = document.getElementById('tts-rtf-bar-wrap');
        if (bw) bw.style.display = 'block';
        const bar = document.getElementById('tts-rtf-bar');
        if (bar) bar.style.width = '0%';
        const bl = document.getElementById('tts-rtf-label');
        if (bl) bl.textContent = '';
        const ee = document.getElementById('tts-elapsed');
        if (ee) {{ ee.style.display = 'none'; ee.textContent = ''; }}
        abort = new AbortController();

        try {{
            st.t0 = performance.now();
            const r = await fetch('/proxy/v1/audio/speech', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify(payload),
                signal: abort.signal,
            }});
            if (!r.ok) {{ const t = await r.text(); throw new Error('Server ' + r.status + ': ' + t.slice(0, 200)); }}
            setStatus('Streaming...', '#43A047');

            const reader = r.body.getReader();
            let left = new Uint8Array(0);
            while (true) {{
                const {{ done, value }} = await reader.read();
                if (done) break;
                let raw;
                if (left.length > 0) {{
                    raw = new Uint8Array(left.length + value.length);
                    raw.set(left); raw.set(value, left.length);
                }} else {{ raw = value; }}
                const usable = raw.length - (raw.length % 2);
                left = usable < raw.length ? raw.slice(usable) : new Uint8Array(0);
                if (usable > 0) {{
                    const ab = new ArrayBuffer(usable);
                    new Uint8Array(ab).set(raw.subarray(0, usable));
                    const pcm = new Int16Array(ab);
                    node.port.postMessage(pcm);
                    st.chunks++;
                    st.samples += pcm.length;
                    if (st.ttfp == null) st.ttfp = performance.now() - st.t0;
                    showStats(false);
                }}
            }}
        }} catch (e) {{
            if (e.name !== 'AbortError') {{
                setStatus('Error: ' + e.message, '#EF5552');
                console.error('TTS error:', e);
            }}
        }} finally {{
            st.streamEnd = performance.now();
            showStats(true);
            gen = false;
            if (st.samples > 0) setStatus('Finishing playback...', '#2E7D32');
            else {{
                setStatus('No audio received', '#999');
                if (bEl) bEl.style.display = 'none';
            }}
        }}
    }};
    </script>
"""


def encode_audio_to_base64(audio_data: tuple) -> str:
    """Encode Gradio audio input (sample_rate, numpy_array) to base64 data URL."""
    sample_rate, audio_np = audio_data
    if audio_np.dtype != np.int16:
        if audio_np.dtype in (np.float32, np.float64):
            audio_np = np.clip(audio_np, -1.0, 1.0)
            audio_np = (audio_np * 32767).astype(np.int16)
        else:
            audio_np = audio_np.astype(np.int16)
    buf = io.BytesIO()
    sf.write(buf, audio_np, sample_rate, format="WAV")
    wav_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:audio/wav;base64,{wav_b64}"


def build_payload(
    text: str,
    ref_audio: tuple | None,
    ref_text: str,
    response_format: str = "pcm",
    stream: bool = True,
) -> dict:
    """Build the /v1/audio/speech request payload.

    The server uses upstream's voice_clone mode and ignores ``ref_text``,
    so we drop it from the payload entirely. The textbox is kept in the
    UI for users coming from voice-cloning systems that do consume a
    transcript (e.g. Qwen3-TTS), so the same UX habits transfer.
    """
    if not text or not text.strip():
        raise gr.Error("Please enter text to synthesize.")
    if ref_audio is None:
        raise gr.Error("Reference audio is required. Upload a 10-30 s clip in the Reference Audio panel.")
    del ref_text  # accepted in the form but not forwarded

    return {
        "input": text.strip(),
        "ref_audio": encode_audio_to_base64(ref_audio),
        "response_format": "pcm" if stream else response_format,
        "stream": stream,
    }


def generate_speech(
    api_base: str,
    text: str,
    ref_audio: tuple | None,
    ref_text: str,
    response_format: str,
) -> tuple:
    """Non-streaming: call /v1/audio/speech and return full audio as (sr, np_array)."""
    payload = build_payload(text, ref_audio, ref_text, response_format, stream=False)

    try:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(
                f"{api_base}/v1/audio/speech",
                json=payload,
                headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
            )
    except httpx.TimeoutException:
        raise gr.Error("Request timed out. The server may be busy.")
    except httpx.ConnectError:
        raise gr.Error(f"Cannot connect to server at {api_base}. Is the server running?")

    if resp.status_code != 200:
        raise gr.Error(f"Server error ({resp.status_code}): {resp.text}")

    try:
        if response_format == "pcm":
            # PCM is stereo interleaved; downmix to mono for Gradio.
            samples = np.frombuffer(resp.content, dtype=np.int16)
            n = len(samples) - (len(samples) % 2)
            left = samples[:n:2].astype(np.float32)
            right = samples[1:n:2].astype(np.float32)
            mono = (left + right) / 65536.0
            return (PCM_SAMPLE_RATE, mono)
        # WAV/MP3/FLAC: soundfile handles header parsing.
        audio_np, sample_rate = sf.read(io.BytesIO(resp.content))
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        return (int(sample_rate), audio_np.astype(np.float32))
    except Exception as e:
        raise gr.Error(f"Failed to decode audio: {e}")


def create_app(api_base: str):
    """Create the FastAPI app with streaming proxy + Gradio UI."""
    fastapi_app = FastAPI()

    # ── Streaming proxy (same-origin, no CORS issues) ────────────
    @fastapi_app.post("/proxy/v1/audio/speech")
    async def proxy_speech(request: Request):
        body = await request.json()
        logger.info(
            "Proxy request: input=%r voice=%s stream=%s",
            body.get("input", "")[:50],
            body.get("voice"),
            body.get("stream"),
        )
        try:
            client = httpx.AsyncClient(timeout=300)
            resp = await client.send(
                client.build_request(
                    "POST",
                    f"{api_base}/v1/audio/speech",
                    json=body,
                    headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"},
                ),
                stream=True,
            )
        except Exception as exc:
            logger.exception("Proxy connection error")
            await client.aclose()
            return Response(content=str(exc), status_code=502)

        if resp.status_code != 200:
            content = await resp.aread()
            logger.error("Proxy upstream error %d: %s", resp.status_code, content[:200])
            await resp.aclose()
            await client.aclose()
            return Response(content=content, status_code=resp.status_code)

        async def relay():
            total = 0
            try:
                async for chunk in resp.aiter_bytes():
                    total += len(chunk)
                    yield chunk
            except Exception:
                logger.exception("Proxy relay error after %d bytes", total)
            finally:
                logger.info("Proxy relay done: %d bytes", total)
                await resp.aclose()
                await client.aclose()

        return StreamingResponse(relay(), media_type="application/octet-stream")

    # ── Gradio UI ────────────────────────────────────────────────
    css = """
    #generate-btn button { width: 100%; }
    #streaming-player { border: 1px solid var(--border-color-primary) !important; border-radius: var(--block-radius) !important; padding: var(--block-padding) !important; }
    """

    theme = gr.themes.Default(
        primary_hue=gr.themes.Color(
            c50="#E8F5E9",
            c100="#C8E6C9",
            c200="#A5D6A7",
            c300="#81C784",
            c400="#66BB6A",
            c500="#4CAF50",
            c600="#43A047",
            c700="#388E3C",
            c800="#2E7D32",
            c900="#1B5E20",
            c950="#0D3B10",
        ),
    )

    with gr.Blocks(title="MOSS-TTS-Nano Demo") as demo:
        gr.HTML(f"""
        <div style="display:flex; align-items:center; gap:16px; margin-bottom:8px;">
          <img src="https://raw.githubusercontent.com/vllm-project/vllm-omni/main/docs/source/logos/vllm-omni-logo.png"
               alt="vLLM-Omni" style="height:42px;">
          <div>
            <h1 style="margin:0; font-size:1.5em;">MOSS-TTS-Nano Streaming Demo</h1>
            <span style="font-size:0.85em; color:#666;">
              0.1B AR LM + MOSS-Audio-Tokenizer-Nano &middot; 48 kHz &middot; ZH / EN / JA
              &nbsp;&middot;&nbsp; Served by
              <a href="https://github.com/vllm-project/vllm-omni" target="_blank"
              style="color:#43A047; text-decoration:none; font-weight:600;">vLLM-Omni</a>
              &nbsp;&middot;&nbsp; <code style="background:#eef2f7; padding:2px 6px; border-radius:4px; font-size:0.9em;">{api_base}</code>
            </span>
          </div>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="Text to Synthesize",
                    placeholder="Enter text in Chinese, English, or Japanese...",
                    lines=4,
                )

                gr.Markdown(
                    "Upload a 10-30 s reference audio clip. MOSS-TTS-Nano "
                    "uses upstream's voice_clone mode, which does not "
                    "consume a transcript — the transcript box below is "
                    "kept for UX consistency with other TTS systems but is "
                    "not sent to the model."
                )
                ref_audio = gr.Audio(
                    label="Reference Audio (required)",
                    type="numpy",
                    sources=["upload", "microphone"],
                )
                ref_text = gr.Textbox(
                    label="Reference Audio Transcript (ignored by MOSS-TTS-Nano)",
                    placeholder="Optional. The MOSS-TTS-Nano voice_clone mode does not use a transcript.",
                    lines=2,
                )

                with gr.Row():
                    response_format = gr.Dropdown(
                        choices=["wav", "mp3", "flac", "pcm"],
                        value="pcm",
                        label="Audio Format",
                        interactive=False,
                        scale=1,
                    )
                    stream_checkbox = gr.Checkbox(
                        label="Stream (gapless)",
                        value=True,
                        info="AudioWorklet streaming (recommended)",
                        scale=1,
                    )

                with gr.Row():
                    generate_btn = gr.Button(
                        "Generate Speech",
                        variant="primary",
                        size="lg",
                        elem_id="generate-btn",
                        scale=3,
                    )
                    reset_btn = gr.Button(
                        "Reset",
                        variant="secondary",
                        size="lg",
                        scale=1,
                    )

            with gr.Column(scale=2):
                player_html = gr.HTML(
                    value=PLAYER_HTML,
                    visible=True,
                    label="streaming player",
                    elem_id="streaming-player",
                )
                audio_output = gr.Audio(
                    label="Generated Audio",
                    interactive=False,
                    autoplay=True,
                    visible=False,
                )
                gr.Examples(
                    examples=[
                        ["Hello, this is MOSS-TTS-Nano speaking."],
                        ["The quick brown fox jumps over the lazy dog."],
                        ["MOSS-TTS-Nanoは、軽量なテキスト読み上げモデルです。"],
                    ],
                    inputs=[text_input],
                    label="Example Texts (upload reference audio above first)",
                )
                gr.HTML("""
                <div style="text-align:center; padding:8px 0; margin-top:4px;">
                  <a href="https://github.com/vllm-project/vllm-omni" target="_blank">
                    <img src="https://raw.githubusercontent.com/vllm-project/vllm-omni/main/docs/source/logos/vllm-omni-logo.png"
                         alt="vLLM-Omni" style="height:28px; opacity:0.7;">
                  </a>
                </div>
                """)

        # Hidden textbox to pass payload from Python -> JavaScript
        hidden_payload = gr.Textbox(visible=False, elem_id="tts-payload")

        # ── Event wiring ─────────────────────────────────────────
        def on_stream_change(stream: bool):
            if stream:
                return (
                    gr.update(value="pcm", interactive=False),
                    gr.update(visible=True),  # player
                    gr.update(visible=False),  # audio
                )
            return (
                gr.update(value="wav", interactive=True),
                gr.update(visible=False),
                gr.update(visible=True),
            )

        stream_checkbox.change(
            fn=on_stream_change,
            inputs=[stream_checkbox],
            outputs=[response_format, player_html, audio_output],
        )

        def on_reset():
            return (
                "",  # text
                None,  # audio_output
                "",  # hidden_payload
                PLAYER_HTML,  # reset player
            )

        reset_btn.click(
            fn=on_reset,
            outputs=[text_input, audio_output, hidden_payload, player_html],
            js="() => { if (window.ttsStop) window.ttsStop(); }",
        )

        all_inputs = [text_input, ref_audio, ref_text, response_format]

        def on_generate(stream_enabled, *args):
            import time as _time

            text, ref_a, ref_t, _fmt = args
            if stream_enabled:
                payload = build_payload(text, ref_a, ref_t, stream=True)
                payload["_nonce"] = int(_time.time() * 1000)
                return json.dumps(payload), gr.update()
            else:
                audio = generate_speech(api_base, text, ref_a, ref_t, _fmt)
                return "", audio

        generate_btn.click(
            fn=on_generate,
            inputs=[stream_checkbox] + all_inputs,
            outputs=[hidden_payload, audio_output],
        ).then(
            fn=lambda p: p,
            inputs=[hidden_payload],
            outputs=[hidden_payload],
            js="(p) => { if (p && p.trim()) { const d = JSON.parse(p); delete d._nonce; window.ttsGenerate(d); } return p; }",
        )

        demo.queue()

    return gr.mount_gradio_app(
        fastapi_app,
        demo,
        path="/",
        css=css,
        theme=theme,
        head=_build_player_js(PCM_SAMPLE_RATE),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Gradio demo for MOSS-TTS-Nano with gapless AudioWorklet streaming.",
    )
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help=f"API base URL (default: {DEFAULT_API_BASE})")
    parser.add_argument("--host", default="0.0.0.0", help="Gradio host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Gradio port (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Share publicly via Gradio tunnel")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    print(f"Connecting to vLLM-Omni server at: {args.api_base}")

    app = create_app(args.api_base)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
