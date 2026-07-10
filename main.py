"""HTTP entrypoint exposing the video rendering API."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

if not hasattr(Image, "ANTIALIAS"):
    # Pillow 10+ uses Resampling.LANCZOS instead of ANTIALIAS.
    Image.ANTIALIAS = Image.Resampling.LANCZOS

from ffmpeg_engine.engine import VideoEngine
from ffmpeg_engine.models import ProcessingMode, RenderRequest, RenderResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FFmpeg Engine", version="1.0.0")
engine = VideoEngine(workspace=Path("renders"))
VALIDATED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv"}


class CleanupRequest(BaseModel):
    dry_run: bool = True
    older_than_hours: int = Field(default=24, ge=0)
    include_media: bool = True


def _get_public_base_url() -> str | None:
    raw_value = os.getenv("PUBLIC_BASE_URL", "").strip()
    if not raw_value:
        return None
    return raw_value.rstrip("/")


def _get_cors_allow_origins() -> list[str]:
    raw_value = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
    if not raw_value:
        return ["*"]
    origins = [item.strip() for item in raw_value.split(",") if item.strip()]
    return origins or ["*"]


def _get_server_host() -> str:
    return os.getenv("API_HOST", "").strip() or "0.0.0.0"


def _get_server_port() -> int:
    raw_value = os.getenv("API_PORT", "").strip()
    if not raw_value:
        return 8000
    try:
        port = int(raw_value)
    except ValueError:
        logger.warning("Invalid API_PORT=%r; falling back to 8000", raw_value)
        return 8000
    return max(port, 1)


def _get_render_mode_concurrency() -> int:
    raw_value = os.getenv("RENDER_MODE_CONCURRENCY", "").strip()
    if raw_value:
        try:
            return max(int(raw_value), 1)
        except ValueError:
            logger.warning("Invalid RENDER_MODE_CONCURRENCY=%r; falling back to 4", raw_value)
            return 4
    legacy_value = os.getenv("RENDER_CONCURRENCY", "").strip()
    if legacy_value:
        try:
            return max(int(legacy_value), 1)
        except ValueError:
            logger.warning("Invalid RENDER_CONCURRENCY=%r; falling back to 4 for render mode", legacy_value)
    return 4


def _get_concat_mode_concurrency() -> int:
    raw_value = os.getenv("CONCAT_NORMALIZE_CONCURRENCY", "").strip()
    if not raw_value:
        return 1
    try:
        return max(int(raw_value), 1)
    except ValueError:
        logger.warning("Invalid CONCAT_NORMALIZE_CONCURRENCY=%r; falling back to 1", raw_value)
        return 1


def _get_disconnect_poll_seconds() -> float:
    raw_value = os.getenv("REQUEST_DISCONNECT_POLL_SECONDS", "").strip()
    if not raw_value:
        return 1.0
    try:
        value = float(raw_value)
    except ValueError:
        logger.warning("Invalid REQUEST_DISCONNECT_POLL_SECONDS=%r; falling back to 1.0", raw_value)
        return 1.0
    return max(value, 0.1)


def _cancel_render_on_client_disconnect() -> bool:
    raw_value = os.getenv("CANCEL_RENDER_ON_CLIENT_DISCONNECT", "").strip().lower()
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    return False


render_mode_semaphore = asyncio.Semaphore(_get_render_mode_concurrency())
concat_mode_semaphore = asyncio.Semaphore(_get_concat_mode_concurrency())
output_lock_registry_guard = asyncio.Lock()
output_locks: dict[str, asyncio.Lock] = {}
cors_allow_origins = _get_cors_allow_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins,
    allow_credentials=cors_allow_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _resolve_download_target(filename: str) -> Path:
    workspace = engine.workspace.resolve()
    target = (workspace / filename).resolve()
    try:
        target.relative_to(workspace)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid download path") from exc
    return target


def _path_size(path: Path) -> int:
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def _latest_mtime(path: Path) -> float:
    try:
        latest = path.stat().st_mtime
    except OSError:
        return 0.0
    if path.is_dir():
        for child in path.rglob("*"):
            try:
                latest = max(latest, child.stat().st_mtime)
            except OSError:
                continue
    return latest


def _delete_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _remove_output_file(path: Path) -> None:
    try:
        if path.exists() and path.is_file():
            path.unlink()
    except OSError as exc:
        logger.warning("Failed to remove render artifact %s: %s", path, exc)


def _is_valid_render_media(path: Path) -> bool:
    if path.suffix.lower() not in VALIDATED_VIDEO_EXTENSIONS:
        return True
    try:
        process = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_type:format=duration,format_name",
                "-of",
                "json",
                str(path),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=30.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if process.returncode != 0:
        return False
    try:
        payload = json.loads(process.stdout or "{}")
        duration = float((payload.get("format") or {}).get("duration") or 0.0)
        streams = payload.get("streams") or []
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return duration > 0.0 and any(stream.get("codec_type") == "video" for stream in streams)


def _is_locked_output_path(path: Path) -> bool:
    try:
        key = str(path.resolve())
    except OSError:
        key = str(path)
    lock = output_locks.get(key)
    return bool(lock and lock.locked())


def _iter_cleanup_roots(include_media: bool) -> list[tuple[str, Path]]:
    roots: list[tuple[str, Path]] = [("renders", engine.workspace.resolve())]
    if include_media:
        roots.append(("media", Path("media").resolve()))
    return roots


def _resolve_output_target_key(payload: RenderRequest) -> str:
    filename = Path(payload.output.filename).name
    extension = payload.output.format.lower().lstrip(".")
    if not filename:
        filename = "render"
    if Path(filename).suffix.lower() != f".{extension}":
        filename = f"{Path(filename).stem}.{extension}"
    return str((engine.workspace.resolve() / filename).resolve())


async def _get_output_lock(payload: RenderRequest) -> asyncio.Lock:
    output_key = _resolve_output_target_key(payload)
    async with output_lock_registry_guard:
        output_lock = output_locks.get(output_key)
        if output_lock is None:
            output_lock = asyncio.Lock()
            output_locks[output_key] = output_lock
    return output_lock


def _build_public_url(request: Request, route_name: str, **path_params: str) -> str:
    route_url = request.url_for(route_name, **path_params)
    public_base_url = _get_public_base_url()
    if not public_base_url:
        return str(route_url)
    path = route_url.path
    if route_url.query:
        return f"{public_base_url}{path}?{route_url.query}"
    return f"{public_base_url}{path}"


def _build_json_payload(request: Request, result: RenderResult, detail_answer: bool) -> dict:
    output_path = result.output.resolve()
    payload = {
        "status": result.status,
        "duration": result.duration,
        "output": {
            "filename": output_path.name,
            "absolute_path": str(output_path),
            "download_url": _build_public_url(request, "download_render", filename=output_path.name),
        },
    }

    if detail_answer:
        timeline_clips = result.timeline.clips if result.timeline else []
        timeline_attachments = result.timeline.attachments if result.timeline else []
        timeline_inserts = result.timeline.inserts if result.timeline else []
        timeline_channels = result.timeline.channels if result.timeline else []

        def serialize_timeline_clip(clip):
            payload = {
                "index": clip.index,
                "source": str(clip.source),
                "start": round(clip.start, 1),
                "end": round(clip.end, 1),
                "auto_placed": clip.auto_placed,
            }
            if clip.source_label:
                payload["source_label"] = clip.source_label
            if clip.source_type:
                payload["source_type"] = clip.source_type
            if clip.source_resolution is not None:
                payload["source_resolution"] = clip.source_resolution
            if clip.quality_label:
                payload["quality_label"] = clip.quality_label
            if clip.channel_id is not None:
                payload["channel_id"] = clip.channel_id
            if clip.kind:
                payload["kind"] = clip.kind
            return payload

        payload["timeline"] = {
            "clips": [serialize_timeline_clip(clip) for clip in timeline_clips],
            "attachments": [serialize_timeline_clip(clip) for clip in timeline_attachments],
            "inserts": [serialize_timeline_clip(clip) for clip in timeline_inserts],
            "channels": [
                {
                    "channel_id": channel.channel_id,
                    "clips": [serialize_timeline_clip(clip) for clip in channel.clips],
                }
                for channel in timeline_channels
            ],
            "total_duration": round(result.duration, 1),
        }

    return payload


def _model_to_jsonable(model) -> dict:
    model_dump = getattr(model, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return json.loads(model.json())


def _parse_render_result(payload: dict) -> RenderResult:
    try:
        return RenderResult.model_validate(payload)
    except AttributeError:  # pragma: no cover - pydantic v1 fallback
        return RenderResult.parse_obj(payload)


def _get_mode_semaphore(mode: ProcessingMode) -> asyncio.Semaphore:
    return concat_mode_semaphore if mode == ProcessingMode.CONCAT_NORMALIZE else render_mode_semaphore


async def _acquire_semaphore_with_disconnect(
    semaphore: asyncio.Semaphore,
    request: Request,
    wait_label: str,
) -> None:
    poll_seconds = _get_disconnect_poll_seconds()
    while True:
        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail=f"Client disconnected while waiting for {wait_label}")
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=poll_seconds)
            return
        except asyncio.TimeoutError:
            continue


async def _acquire_lock_with_disconnect(
    lock: asyncio.Lock,
    request: Request,
    wait_label: str,
) -> None:
    poll_seconds = _get_disconnect_poll_seconds()
    while True:
        if await request.is_disconnected():
            raise HTTPException(status_code=499, detail=f"Client disconnected while waiting for {wait_label}")
        try:
            await asyncio.wait_for(lock.acquire(), timeout=poll_seconds)
            return
        except asyncio.TimeoutError:
            continue


async def _pump_stream(stream: asyncio.StreamReader | None) -> bytes:
    if stream is None:
        return b""
    chunks: list[bytes] = []
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def _trim_stderr_tail(stderr_bytes: bytes, max_lines: int = 12) -> str:
    text = stderr_bytes.decode("utf-8", errors="replace").strip()
    if not text:
        return ""
    return " | ".join(text.splitlines()[-max_lines:])


async def _terminate_render_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return

    try:
        if os.name == "nt":
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/PID",
                str(process.pid),
                "/T",
                "/F",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(killer.wait(), timeout=5.0)
        else:
            os.killpg(process.pid, signal.SIGTERM)
            await asyncio.wait_for(process.wait(), timeout=5.0)
    except Exception:
        try:
            if process.returncode is None:
                if os.name == "nt":
                    killer = await asyncio.create_subprocess_exec(
                        "taskkill",
                        "/PID",
                        str(process.pid),
                        "/T",
                        "/F",
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    await killer.wait()
                else:
                    os.killpg(process.pid, signal.SIGKILL)
                    await process.wait()
        except Exception:
            logger.exception("Failed to terminate render worker pid=%s", process.pid)


async def _run_render_worker(payload: RenderRequest, request: Request) -> RenderResult:
    workspace = engine.workspace.resolve()
    request_data = _model_to_jsonable(payload)

    with tempfile.TemporaryDirectory(prefix="ffmpeg_engine_job_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        request_path = temp_dir / "request.json"
        result_path = temp_dir / "result.json"
        request_path.write_text(json.dumps(request_data, ensure_ascii=False), encoding="utf-8")

        command = [
            sys.executable,
            "-m",
            "ffmpeg_engine.render_worker",
            "--workspace",
            str(workspace),
            "--request-file",
            str(request_path),
            "--result-file",
            str(result_path),
        ]

        creation_kwargs = {}
        if os.name == "nt":
            creation_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            creation_kwargs["start_new_session"] = True

        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            **creation_kwargs,
        )
        stderr_task = asyncio.create_task(_pump_stream(process.stderr))

        try:
            poll_seconds = _get_disconnect_poll_seconds()
            cancel_on_disconnect = _cancel_render_on_client_disconnect()
            disconnected_logged = False
            while process.returncode is None:
                if await request.is_disconnected():
                    if cancel_on_disconnect:
                        await _terminate_render_process(process)
                        raise HTTPException(status_code=499, detail="Client disconnected; render cancelled")
                    if not disconnected_logged:
                        logger.warning(
                            "Client disconnected; render worker pid=%s will continue until completion",
                            process.pid,
                        )
                        disconnected_logged = True
                try:
                    await asyncio.wait_for(process.wait(), timeout=poll_seconds)
                except asyncio.TimeoutError:
                    continue
        finally:
            # A descendant can inherit stderr and keep the pipe open after the
            # render worker itself has exited. Do not let that orphaned pipe
            # keep the HTTP response and the per-output lock alive forever.
            try:
                stderr_bytes = await asyncio.wait_for(stderr_task, timeout=5.0)
            except asyncio.TimeoutError:
                stderr_task.cancel()
                try:
                    await stderr_task
                except asyncio.CancelledError:
                    pass
                logger.warning("Render worker pid=%s exited but stderr did not close; continuing", process.pid)
                stderr_bytes = b""

        if not result_path.exists():
            stderr_tail = _trim_stderr_tail(stderr_bytes)
            message = "Render worker exited without a result payload"
            if stderr_tail:
                message = f"{message}: {stderr_tail}"
            raise HTTPException(status_code=500, detail=message)

        try:
            result_data = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception as exc:
            stderr_tail = _trim_stderr_tail(stderr_bytes)
            message = f"Failed to parse render worker result: {exc}"
            if stderr_tail:
                message = f"{message}. Worker stderr: {stderr_tail}"
            raise HTTPException(status_code=500, detail=message) from exc

        result = _parse_render_result(result_data)
        if process.returncode not in {0, None} and result.status == "ok":
            stderr_tail = _trim_stderr_tail(stderr_bytes)
            message = "Render worker exited with a non-zero code despite ok result"
            if stderr_tail:
                message = f"{message}: {stderr_tail}"
            raise HTTPException(status_code=500, detail=message)

        return result


@app.get("/")
async def root() -> dict:
    return {"status": "ok", "message": "FFmpeg engine is running"}


@app.post("/cleanup")
async def cleanup(payload: CleanupRequest) -> dict[str, Any]:
    cutoff = time.time() - payload.older_than_hours * 3600
    removed: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    removed_bytes = 0

    for root_name, root in _iter_cleanup_roots(payload.include_media):
        if not root.exists():
            skipped.append({"path": root_name, "reason": "missing_root"})
            continue
        for child in root.iterdir():
            relative_name = f"{root_name}/{child.name}"
            if child.name == ".gitkeep":
                skipped.append({"path": relative_name, "reason": "reserved"})
                continue
            if root_name == "renders" and _is_locked_output_path(child):
                skipped.append({"path": relative_name, "reason": "active_output"})
                continue
            if _latest_mtime(child) > cutoff:
                skipped.append({"path": relative_name, "reason": "too_new"})
                continue

            size = _path_size(child)
            removed.append({"path": relative_name, "bytes": size})
            removed_bytes += size
            if payload.dry_run:
                continue
            try:
                _delete_path(child)
            except OSError as exc:
                skipped.append({"path": relative_name, "reason": f"delete_failed: {exc}"})

    return {
        "dry_run": payload.dry_run,
        "older_than_hours": payload.older_than_hours,
        "include_media": payload.include_media,
        "removed_count": len(removed),
        "removed_bytes": removed_bytes,
        "removed": removed,
        "skipped": skipped,
    }


@app.get("/downloads/")
async def list_downloads(request: Request) -> dict:
    workspace = engine.workspace.resolve()
    items = []
    if workspace.exists():
        for file_path in sorted(workspace.glob("*")):
            if not file_path.is_file():
                continue
            if not _is_valid_render_media(file_path):
                _remove_output_file(file_path)
                continue
            items.append(
                {
                    "filename": file_path.name,
                    "absolute_path": str(file_path.resolve()),
                    "download_url": _build_public_url(request, "download_render", filename=file_path.name),
                }
            )
    return {"items": items}


@app.get("/downloads/{filename:path}", name="download_render")
async def download_render(filename: str) -> Response:
    target = _resolve_download_target(filename)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Render file not found")
    if _is_locked_output_path(target):
        raise HTTPException(status_code=409, detail="Render file is still being produced")
    if not _is_valid_render_media(target):
        _remove_output_file(target)
        raise HTTPException(status_code=404, detail="Render file is incomplete or invalid and was removed")
    return FileResponse(target, filename=target.name)


@app.delete("/downloads/{filename:path}")
async def delete_render(filename: str) -> dict[str, Any]:
    target = _resolve_download_target(filename)
    if _is_locked_output_path(target):
        raise HTTPException(status_code=409, detail="Render file is still being produced")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Render file not found")
    removed_bytes = _path_size(target)
    target.unlink()
    return {"filename": filename, "removed": True, "removed_bytes": removed_bytes}


@app.post("/render")
async def render_endpoint(
    request: Request,
    detail_answer: bool = Query(False, description="Return timeline placement details in JSON response"),
    payload: RenderRequest = Body(..., description="Render instructions"),
):
    output_key = _resolve_output_target_key(payload)
    output_lock = await _get_output_lock(payload)
    mode_semaphore = _get_mode_semaphore(payload.mode)
    logger.info("Incoming render request from %s target=%s", request.client, output_key)
    lock_acquired = False
    slot_acquired = False
    result: RenderResult | None = None
    try:
        await _acquire_lock_with_disconnect(output_lock, request, f"output lock for {Path(output_key).name}")
        lock_acquired = True
        _remove_output_file(Path(output_key))
        await _acquire_semaphore_with_disconnect(mode_semaphore, request, f"{payload.mode.value} slot")
        slot_acquired = True
        result = await _run_render_worker(payload, request)
    finally:
        if lock_acquired and (result is None or result.status != "ok"):
            _remove_output_file(Path(output_key))
        if slot_acquired:
            mode_semaphore.release()
        if lock_acquired:
            output_lock.release()

    assert result is not None
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=_get_server_host(),
        port=_get_server_port(),
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
