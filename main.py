"""HTTP entrypoint exposing the video rendering API."""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

if not hasattr(Image, "ANTIALIAS"):
    # Pillow 10+ uses Resampling.LANCZOS instead of ANTIALIAS.
    Image.ANTIALIAS = Image.Resampling.LANCZOS

from ffmpeg_engine.engine import VideoEngine
from ffmpeg_engine.models import RenderRequest, RenderResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FFmpeg Engine", version="1.0.0")
engine = VideoEngine(workspace=Path("renders"))
render_lock = asyncio.Lock()


def _resolve_download_target(filename: str) -> Path:
    workspace = engine.workspace.resolve()
    target = (workspace / filename).resolve()
    try:
        target.relative_to(workspace)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid download path") from exc
    return target


def _build_json_payload(request: Request, result: RenderResult, detail_answer: bool) -> dict:
    output_path = result.output.resolve()
    payload = {
        "status": result.status,
        "duration": result.duration,
        "output": {
            "filename": output_path.name,
            "absolute_path": str(output_path),
            "download_url": str(request.url_for("download_render", filename=output_path.name)),
        },
    }

    if detail_answer:
        timeline_clips = result.timeline.clips if result.timeline else []
        payload["timeline"] = {
            "clips": [
                {
                    "index": clip.index,
                    "source": str(clip.source),
                    "start": round(clip.start, 1),
                    "end": round(clip.end, 1),
                    "auto_placed": clip.auto_placed,
                }
                for clip in timeline_clips
            ],
            "total_duration": round(result.duration, 1),
        }

    return payload


@app.get("/")
async def root() -> dict:
    return {"status": "ok", "message": "FFmpeg engine is running"}


@app.get("/downloads/")
async def list_downloads(request: Request) -> dict:
    workspace = engine.workspace.resolve()
    items = []
    if workspace.exists():
        for file_path in sorted(workspace.glob("*")):
            if not file_path.is_file():
                continue
            items.append(
                {
                    "filename": file_path.name,
                    "absolute_path": str(file_path.resolve()),
                    "download_url": str(request.url_for("download_render", filename=file_path.name)),
                }
            )
    return {"items": items}


@app.get("/downloads/{filename:path}", name="download_render")
async def download_render(filename: str) -> Response:
    target = _resolve_download_target(filename)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Render file not found")
    return FileResponse(target, filename=target.name)


@app.post("/render")
async def render_endpoint(
    request: Request,
    detail_answer: bool = Query(False, description="Return timeline placement details in JSON response"),
    payload: RenderRequest = Body(..., description="Render instructions"),
):
    logger.info("Incoming render request from %s", request.client)
    async with render_lock:
        result = await run_in_threadpool(engine.render, payload)
    if result.status != "ok":
        raise HTTPException(status_code=400, detail=result.message or "Rendering failed")

    need_detail = detail_answer or payload.detail_answer
    accept = request.headers.get("accept", "application/json")
    json_payload = _build_json_payload(request, result, need_detail)

    if need_detail or "application/json" in accept:
        return JSONResponse(json_payload)
    if result.output.suffix.lower() in {".mp4", ".mov", ".webm", ".mkv"}:
        return FileResponse(result.output)
    return JSONResponse(json_payload)


@app.post("/render/raw")
async def render_raw_endpoint(
    request: Request,
    detail_answer: bool = Query(False, description="Return timeline placement details in JSON response"),
) -> Response:
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc
    try:
        parsed = RenderRequest.model_validate(payload)
    except AttributeError:  # pragma: no cover - pydantic v1 fallback
        parsed = RenderRequest.parse_obj(payload)
    return await render_endpoint(request=request, detail_answer=detail_answer, payload=parsed)


@app.get("/templates")
async def templates() -> dict:
    from ffmpeg_engine import templates as template_module

    return {
        "items": [
            {
                "name": template.name,
                "description": template.description,
                "resolution": template.resolution,
                "aspect_ratio": template.aspect_ratio,
            }
            for template in template_module.TEMPLATES.values()
        ]
    }
