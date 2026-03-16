"""Subprocess worker for isolated render execution."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

from .engine import VideoEngine
from .models import RenderRequest, RenderResult

logger = logging.getLogger(__name__)


def _load_request(request_path: Path) -> RenderRequest:
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    try:
        return RenderRequest.model_validate(payload)
    except AttributeError:  # pragma: no cover - pydantic v1 fallback
        return RenderRequest.parse_obj(payload)


def _result_to_jsonable(result: RenderResult) -> dict:
    model_dump = getattr(result, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return json.loads(result.json())


def _write_result(result_path: Path, result: RenderResult) -> None:
    result_path.write_text(
        json.dumps(_result_to_jsonable(result), ensure_ascii=False),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a single render request in an isolated subprocess.")
    parser.add_argument("--workspace", required=True, help="Render workspace path")
    parser.add_argument("--request-file", required=True, help="Path to request JSON")
    parser.add_argument("--result-file", required=True, help="Path to result JSON")
    args = parser.parse_args(argv)

    workspace = Path(args.workspace)
    request_path = Path(args.request_file)
    result_path = Path(args.result_file)

    try:
        request = _load_request(request_path)
        engine = VideoEngine(workspace=workspace)
        result = engine.render(request)
        _write_result(result_path, result)
        return 0
    except BaseException as exc:  # pragma: no cover - subprocess safety net
        logger.exception("Render worker failed: %s", exc)
        fallback_result = RenderResult(
            status="error",
            duration=0.0,
            output=Path(""),
            message=str(exc),
        )
        try:
            _write_result(result_path, fallback_result)
        except Exception:
            traceback.print_exc()
            return 1
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
