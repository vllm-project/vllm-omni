# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Serve SheetSage2 audio-to-score transcription over HTTP.

``POST /v1/audio/transcriptions`` follows the OpenAI multipart upload shape and
returns the recording's ABC score as ``text``, ready for YuE2's
``extra_params.abc``. ``melody_only`` and ``max_seconds`` are extra form fields
with the same meaning as the CLI flags. The model loads once at startup and
requests run one at a time on its device.

Run it in the separate environment documented in
recipes/m-a-p/SheetSage2-H200.md. Like the CLI, it uses the upstream
transcriber, not an Omni engine.
"""

# No `from __future__ import annotations`: FastAPI reads the endpoint's
# annotations at definition time, and they name locally imported types.
import argparse
import asyncio
import logging
import math
import tempfile
from pathlib import Path

try:
    from tools.sheetsage2_transcribe import add_model_arguments, checked_score, load_model
except ModuleNotFoundError:  # Run as a script: tools/ itself is on sys.path.
    from sheetsage2_transcribe import add_model_arguments, checked_score, load_model

UPLOAD_CHUNK = 1 << 20
logger = logging.getLogger("sheetsage2_serve")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--served-model-name", help="Model name clients send; defaults to --model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--max-upload-mb", type=int, default=200, help="Largest accepted request body, including the recording"
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.trust_remote_code:
        raise ValueError(
            "SheetSage2 uses custom model code; pass --trust-remote-code after reviewing the model repository"
        )
    if args.base_model_path is not None and not args.base_model_path.is_dir():
        raise ValueError(f"MERT-v2 directory does not exist: {args.base_model_path}")
    if args.max_upload_mb <= 0:
        raise ValueError("--max-upload-mb must be positive")


def limit_body(app, max_bytes: int):
    """Reject a request body above ``max_bytes`` while it arrives.

    Multipart parsing spools the whole upload to disk before the endpoint runs,
    so the endpoint itself cannot bound it. The raised 413 surfaces through
    FastAPI's body parsing.
    """
    from starlette.exceptions import HTTPException

    too_large = HTTPException(413, f"Request body exceeds {max_bytes // (1 << 20)} MiB")

    async def limited_app(scope, receive, send):
        if scope["type"] != "http":
            return await app(scope, receive, send)
        declared = dict(scope["headers"]).get(b"content-length", b"")
        received = 0

        async def limited_receive():
            nonlocal received
            # The server holds the body to its Content-Length, so a declared size is rejected before any read.
            if declared.isdigit() and int(declared) > max_bytes:
                raise too_large
            message = await receive()
            received += len(message.get("body", b""))
            if received > max_bytes:
                raise too_large
            return message

        await app(scope, limited_receive, send)

    return limited_app


def create_app(model, served_model_name: str, dtype: str, max_upload_bytes: int):
    from fastapi import FastAPI, Form, UploadFile
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse, PlainTextResponse
    from starlette.exceptions import HTTPException

    app = FastAPI(title="SheetSage2")
    app.add_middleware(limit_body, max_bytes=max_upload_bytes)
    # One model on one device: transcriptions queue instead of sharing it.
    lock = asyncio.Lock()

    def error(status: int, message: str, headers: dict[str, str] | None = None) -> JSONResponse:
        kind = "invalid_request_error" if status < 500 else "server_error"
        body = {"error": {"message": message, "type": kind, "code": status}}
        return JSONResponse(body, status_code=status, headers=headers)

    # Form validation and body parsing fail before the endpoint runs; keep the OpenAI error shape.
    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc: RequestValidationError) -> JSONResponse:
        problems = (f"{'.'.join(map(str, e['loc'][1:])) or e['loc'][0]}: {e['msg']}" for e in exc.errors())
        return error(400, "; ".join(problems))

    @app.exception_handler(HTTPException)
    async def http_error(request, exc: HTTPException) -> JSONResponse:
        return error(exc.status_code, str(exc.detail), exc.headers)

    @app.get("/health")
    async def health() -> PlainTextResponse:
        return PlainTextResponse("")

    @app.get("/v1/models")
    async def models() -> dict:
        return {"object": "list", "data": [{"id": served_model_name, "object": "model", "owned_by": "m-a-p"}]}

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        file: UploadFile,
        model_name: str | None = Form(None, alias="model"),
        response_format: str = Form("json"),
        melody_only: bool = Form(False),
        max_seconds: float | None = Form(None),
    ):
        if model_name not in (None, "", served_model_name):
            return error(404, f"The model {model_name!r} does not exist; this server serves {served_model_name!r}")
        if response_format not in ("json", "text"):
            return error(400, f"response_format must be json or text; got {response_format!r}")
        if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
            return error(400, "max_seconds must be finite and positive")
        with tempfile.TemporaryDirectory(prefix="sheetsage2-") as work:
            # Keep the extension: FFmpeg probes the container, but some demuxers need the hint.
            audio = Path(work) / f"input{Path(file.filename or '').suffix.lower()[:8]}"
            size = 0
            with audio.open("wb") as out:
                while chunk := await file.read(UPLOAD_CHUNK):
                    size += len(chunk)
                    out.write(chunk)
            if size == 0:
                return error(400, "file is empty")
            output = Path(work) / "score"
            output.mkdir()
            try:
                async with lock:
                    result = await asyncio.to_thread(
                        model.transcribe,
                        str(audio),
                        output_dir=str(output),
                        melody_only=melody_only,
                        dtype=dtype,
                        max_seconds=max_seconds,
                    )
                abc = checked_score(result)
            # Details stay in the server log: upstream messages can name server paths.
            except ValueError:  # Upstream rejects recordings it cannot decode with ValueError.
                logger.warning("Cannot transcribe %r", file.filename, exc_info=True)
                return error(400, "SheetSage2 cannot decode this recording")
            except Exception:  # noqa: BLE001 - report other upstream failures as server errors
                logger.exception("Transcription of %r failed", file.filename)
                return error(500, "SheetSage2 transcription failed; see the server log")
        if response_format == "text":
            return PlainTextResponse(abc)
        return {"text": abc}

    return app


def main(argv: list[str] | None = None) -> None:
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        import uvicorn

        model, device, _ = load_model(args)
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        parser.exit(1, f"SheetSage2: {exc}\nSee recipes/m-a-p/SheetSage2-H200.md for the isolated environment setup.\n")
    name = args.served_model_name or args.model
    app = create_app(model, name, args.dtype, args.max_upload_mb << 20)
    print(f"Serving {name} on {device} at http://{args.host}:{args.port}/v1/audio/transcriptions")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
