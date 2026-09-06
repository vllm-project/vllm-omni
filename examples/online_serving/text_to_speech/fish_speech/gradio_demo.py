"""Gradio demo for Fish Speech S2 Pro online serving via /v1/audio/speech API.

Streaming uses a custom AudioWorklet-based player for gap-free playback,
inspired by github.com/KoljaB/RealtimeVoiceChat. Audio is streamed from the
vLLM server through a same-origin proxy and played via the Web Audio API's
AudioWorklet, which maintains a FIFO buffer queue and plays samples at the
audio clock rate — eliminating the inter-chunk gaps inherent in Gradio's
built-in streaming audio component.

Supports:
  - Text-to-speech synthesis
  - Voice cloning from reference audio (upload or URL)
  - Streaming (gapless AudioWorklet) and non-streaming modes

Usage:
    # Start the server first (see run_server.sh), then:
    python gradio_demo.py --api-base http://localhost:8091

    # Or use run_gradio_demo.sh to start both server and demo together.
"""

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

PCM_SAMPLE_RATE = 44100
DEFAULT_API_BASE = "http://localhost:8091"


# ── AudioWorklet processor (loaded in browser via Blob URL) ──────────
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
            this.queue.push(e.data);
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
            out[i] = this.buf[this.pos++] / 32768;
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
  <div id="tts-final" style="display:none; margin-top:10px;"></div>
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
            if (e.data.type === 'started') setStatus('Playing...', '#64dd17');
            else if (e.data.type === 'stopped' && !gen) {{
                setStatus('Done', '#64dd17'); showStats(true);
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
        const dur = st.samples / SR;
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
                mRtf.style.color = rtf < 1 ? '#64dd17' : rtf < 1.5 ? '#e8a317' : '#EF5552';
            }}
            if (mSpeed) {{
                mSpeed.textContent = speed.toFixed(1) + 'x';
                mSpeed.style.color = speed > 1 ? '#64dd17' : speed > 0.7 ? '#e8a317' : '#EF5552';
            }}
            if (bar) {{
                const pct = Math.min(speed / 3 * 100, 100);
                bar.style.width = pct + '%';
                bar.style.background = speed > 1 ? 'linear-gradient(90deg,#4A90D9,#64dd17)' : speed > 0.7 ? 'linear-gradient(90deg,#e8a317,#f0b866)' : 'linear-gradient(90deg,#EF5552,#f87171)';
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

        // Abort previous request and clear worklet buffer
        if (abort) abort.abort();
        node.port.postMessage({{ type: 'clear' }});
        // Wait for worklet to process clear before sending new data
        await new Promise(r => setTimeout(r, 50));
        node.port.postMessage({{ type: 'clear' }});

        gen = true;
        st = {{ t0: null, chunks: 0, samples: 0, ttfp: null }};
        window._ttsChunks = [];
        setStatus('Connecting...', '#4A90D9');
        const bEl = document.getElementById('tts-stop-btn');
        if (bEl) bEl.style.display = 'inline-block';
        const mp = document.getElementById('tts-metrics');
        if (mp) {{ mp.style.display = 'grid'; ['tts-m-ttfp','tts-m-rtf','tts-m-dur','tts-m-speed'].forEach(id => {{ const e = document.getElementById(id); if(e) {{ e.textContent = '—'; e.style.color = '#333'; }} }}); }}
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
            setStatus('Streaming...', '#4A90D9');

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
                    window._ttsChunks.push(pcm);
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
            // Freeze RTF at stream-end time (before playback finishes)
            st.streamEnd = performance.now();
            showStats(true);
            gen = false;
            if (st.samples > 0) {{
                setStatus('Finishing playback...', '#64dd17');
                showFinalAudio();
            }} else {{
                setStatus('No audio received', '#999');
                if (bEl) bEl.style.display = 'none';
            }}
        }}
    }};

    // Assemble the accumulated PCM chunks into a WAV blob and render an
    // <audio controls> player + download link, so the full clip stays
    // playable/downloadable after the live stream ends.
    function showFinalAudio() {{
        const chunks = window._ttsChunks || [];
        if (!chunks.length) return;
        let total = 0;
        for (const c of chunks) total += c.length;
        const pcm = new Int16Array(total);
        let off = 0;
        for (const c of chunks) {{ pcm.set(c, off); off += c.length; }}

        // Build a 44-byte WAV header + PCM body (mono, 16-bit, SR).
        const bytesPerSample = 2, numCh = 1;
        const dataLen = pcm.length * bytesPerSample;
        const buf = new ArrayBuffer(44 + dataLen);
        const dv = new DataView(buf);
        const wr = (o, s) => {{ for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); }};
        wr(0, 'RIFF'); dv.setUint32(4, 36 + dataLen, true); wr(8, 'WAVE');
        wr(12, 'fmt '); dv.setUint32(16, 16, true); dv.setUint16(20, 1, true);
        dv.setUint16(22, numCh, true); dv.setUint32(24, SR, true);
        dv.setUint32(28, SR * numCh * bytesPerSample, true);
        dv.setUint16(32, numCh * bytesPerSample, true); dv.setUint16(34, 16, true);
        wr(36, 'data'); dv.setUint32(40, dataLen, true);
        new Int16Array(buf, 44).set(pcm);

        const blob = new Blob([buf], {{ type: 'audio/wav' }});
        const url = URL.createObjectURL(blob);
        const el = document.getElementById('tts-final');
        if (!el) return;
        if (window._ttsFinalUrl) URL.revokeObjectURL(window._ttsFinalUrl);
        window._ttsFinalUrl = url;
        el.style.display = 'block';
        el.innerHTML =
            '<audio controls src="' + url + '" style="width:100%; margin-bottom:6px;"></audio>' +
            '<a href="' + url + '" download="fish_speech.wav" ' +
            'style="font-size:0.85em; color:#4A90D9; text-decoration:none;">⬇ Download WAV</a>';
    }}
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
    ref_audio_url: str,
    ref_text: str,
    response_format: str = "wav",
    stream: bool = False,
) -> dict:
    """Build the /v1/audio/speech request payload.

    Fish Speech has no built-in speakers; omit ``voice`` so the model
    generates with a random timbre (controllable via text tags like
    ``[cool]``, ``[warm]``, etc.).  When ``ref_audio`` is supplied we
    enter voice-cloning mode instead.
    """
    if not text or not text.strip():
        raise gr.Error("Please enter text to synthesize.")

    payload: dict = {
        "input": text.strip(),
        "response_format": "pcm" if stream else response_format,
        "stream": stream,
    }
    if stream:
        payload["stream_format"] = "audio"

    # Voice cloning: ref_audio takes priority over URL.
    ref_url_stripped = ref_audio_url.strip() if ref_audio_url else ""
    if ref_audio is not None:
        payload["ref_audio"] = encode_audio_to_base64(ref_audio)
    elif ref_url_stripped:
        payload["ref_audio"] = ref_url_stripped

    if "ref_audio" in payload:
        if not ref_text or not ref_text.strip():
            raise gr.Error("Voice cloning requires a transcript of the reference audio.")
        payload["ref_text"] = ref_text.strip()

    return payload


def generate_speech(api_base: str, text: str, ref_audio, ref_audio_url, ref_text, response_format):
    """Non-streaming: call /v1/audio/speech and return full audio."""
    payload = build_payload(text, ref_audio, ref_audio_url, ref_text, response_format, stream=False)

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

    content_type = resp.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            raise gr.Error(f"Server error: {resp.json()}")
        except ValueError:
            pass

    try:
        if response_format == "pcm":
            audio_np = np.frombuffer(resp.content, dtype=np.int16).astype(np.float32) / 32767.0
            return (PCM_SAMPLE_RATE, audio_np)
        audio_np, sample_rate = sf.read(io.BytesIO(resp.content))
        if audio_np.ndim > 1:
            audio_np = audio_np[:, 0]
        return (sample_rate, audio_np.astype(np.float32))
    except Exception as e:
        raise gr.Error(f"Failed to decode audio: {e}")


def create_app(api_base: str):
    """Create the FastAPI app with a streaming proxy + Gradio UI."""
    fastapi_app = FastAPI()

    # Server-side payload store: streaming payloads (esp. base64 ref_audio) are
    # too large to route through the Gradio textbox -> JS -> fetch pipeline, so
    # we build them in Python, stash them here, and hand the browser only a
    # short request id to fetch by.
    _pending_payloads: dict[str, dict] = {}

    # ── Streaming proxy (same-origin, no CORS issues) ────────────
    @fastapi_app.post("/proxy/v1/audio/speech")
    async def proxy_speech(request: Request):
        body = await request.json()
        req_id = body.get("_req_id")
        if req_id and req_id in _pending_payloads:
            body = _pending_payloads.pop(req_id)
        body.pop("_nonce", None)
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

        # Use audio/pcm to bypass browser (e.g. Chromium) MIME sniffing, which
        # buffers application/octet-stream responses before delivering chunks to
        # fetch().body.getReader(), inflating time-to-first-playback.
        return StreamingResponse(relay(), media_type="audio/pcm")

    # ── Gradio UI ────────────────────────────────────────────────
    with gr.Blocks(title="Fish Speech S2 Pro Demo") as demo:
        gr.Markdown("# Fish Speech S2 Pro - Text to Speech")
        gr.Markdown(f"**Server:** `{api_base}` | **Model:** fishaudio/s2-pro | **Output:** 44.1kHz")

        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="Text to Synthesize",
                    placeholder="Enter text here...",
                    lines=4,
                )

                with gr.Accordion("Voice Cloning (optional)", open=False):
                    gr.Markdown(
                        "Upload or link a short reference audio (10-30s) and provide its transcript to clone the voice. "
                        "Leave empty for random timbre; use text tags like `[cool]`, `[warm]`, `[happy]` to steer the output."
                    )
                    ref_audio = gr.Audio(
                        label="Reference Audio",
                        type="numpy",
                        sources=["upload", "microphone"],
                    )
                    ref_audio_url = gr.Textbox(
                        label="Reference Audio URL (alternative to upload)",
                        placeholder="https://example.com/reference.wav",
                        lines=1,
                    )
                    ref_text = gr.Textbox(
                        label="Reference Audio Transcript (required for cloning)",
                        placeholder="Exact transcript of the reference audio...",
                        lines=2,
                    )

                with gr.Row():
                    response_format = gr.Dropdown(
                        choices=["wav", "mp3", "flac", "pcm"],
                        value="wav",
                        label="Audio Format",
                        scale=1,
                    )
                    stream_checkbox = gr.Checkbox(
                        label="Stream output (gapless)",
                        value=False,
                        info="AudioWorklet streaming",
                        scale=1,
                    )

                generate_btn = gr.Button("Generate Speech", variant="primary", size="lg")

            with gr.Column(scale=2):
                player_html = gr.HTML(
                    value=PLAYER_HTML,
                    visible=False,
                    label="Streaming player",
                )
                audio_output = gr.Audio(
                    label="Generated Audio",
                    interactive=False,
                    autoplay=True,
                    visible=True,
                )
                gr.Markdown(
                    "### About\n"
                    "- **Fish Speech S2 Pro** by FishAudio: 4B dual-AR model\n"
                    "- **Voice cloning**: upload 10-30s reference + transcript\n"
                    "- **Streaming**: gapless real-time PCM via AudioWorklet\n"
                    "- **44.1kHz** output via DAC codec"
                )

        # Hidden textbox to pass the streaming payload from Python -> JavaScript.
        hidden_payload = gr.Textbox(visible=False, elem_id="tts-payload")

        def on_stream_change(stream: bool):
            """Lock format to PCM when streaming; swap player vs gr.Audio."""
            if stream:
                return (
                    gr.update(value="pcm", interactive=False),
                    gr.update(visible=True),  # player_html
                    gr.update(visible=False),  # audio_output
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

        all_inputs = [text_input, ref_audio, ref_audio_url, ref_text, response_format]

        def on_generate(stream_enabled, *args):
            # Streaming path: build payload in Python, stash it, hand the
            # browser a short request id. The .then() JS calls window.ttsGenerate
            # which fetches /proxy and feeds PCM into the AudioWorklet player.
            if stream_enabled:
                import time as _time

                text, ref_a, ref_url, ref_t, _fmt = args
                payload = build_payload(text, ref_a, ref_url, ref_t, "pcm", stream=True)
                req_id = f"req-{int(_time.time() * 1000)}"
                _pending_payloads[req_id] = payload
                browser_payload = {"_req_id": req_id, "_nonce": int(_time.time() * 1000)}
                return json.dumps(browser_payload), gr.update()
            # Non-streaming path: return the full clip via gr.Audio.
            audio = generate_speech(api_base, *args)
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
        head=_build_player_js(PCM_SAMPLE_RATE),
    )


def main():
    parser = argparse.ArgumentParser(description="Gradio demo for Fish Speech S2 Pro")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help=f"API base URL (default: {DEFAULT_API_BASE})")
    parser.add_argument("--host", default="0.0.0.0", help="Gradio host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Gradio port (default: 7860)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    print(f"Connecting to vLLM server at: {args.api_base}")

    app = create_app(args.api_base)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
