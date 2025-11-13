"""HTTP entrypoint exposing the video rendering API."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse

from ffmpeg_engine.engine import VideoEngine
from ffmpeg_engine.models import RenderRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FFmpeg Engine", version="1.0.0")
engine = VideoEngine(workspace=Path("renders"))


@app.get("/")
async def root() -> dict:
    return {"status": "ok", "message": "FFmpeg engine is running"}


@app.post("/render")
async def render_endpoint(
    request: Request,
    payload: RenderRequest = Body(..., description="Render instructions"),
):
    logger.info("Incoming render request from %s", request.client)
    result = engine.render(payload)
    if result.status != "ok":
        raise HTTPException(status_code=400, detail=result.message or "Rendering failed")

    accept = request.headers.get("accept", "application/json")
    if "application/json" in accept:
        return JSONResponse(
            {
                "status": result.status,
                "duration": result.duration,
                "output": str(result.output),
            }
        )
    if result.output.suffix.lower() in {".mp4", ".mov", ".webm", ".mkv"}:
        return FileResponse(result.output)
    return JSONResponse(
        {
            "status": result.status,
            "duration": result.duration,
            "output": str(result.output),
        }
    )


@app.post("/render/raw")
async def render_raw_endpoint(request: Request) -> Response:
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc
    try:
        parsed = RenderRequest.model_validate(payload)
    except AttributeError:  # pragma: no cover - pydantic v1 fallback
        parsed = RenderRequest.parse_obj(payload)
    return await render_endpoint(request, parsed)


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

