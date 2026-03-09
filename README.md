# FFmpeg Engine Python

HTTP-based video rendering microservice powered by FastAPI, MoviePy, and FFmpeg. The service consumes JSON instructions describing clips, transitions, overlays, and audio layers, then produces a rendered video or a structured JSON response.

## Features

- Aspect ratio templates for popular platforms (YouTube, TikTok, Instagram, Stories).
- Timeline instructions supporting clip trimming, auto-fit modes (cover/contain), background blur or solid color, and optional chroma key.
- Auto-append clip placement when `start` and `end` are omitted in clip instructions.
- Crossfade and fade-to-black transitions.
- Color adjustments (brightness, contrast, saturation, hue).
- Audio mixing with precise start/end trimming, fades, and volume controls.
- Text and PNG overlays with positioning, animation, and basic keyframe scaling.
- JSON detail mode (`detail_answer`) with per-clip timeline start/end values.
- Download endpoint (`/downloads/...`) and absolute output file paths in JSON responses.
- Safe defaults for incomplete output settings (`format=mp4`, `fps=30`, datetime filename, `1920x1080` fallback resolution).
- Optional fast processing mode (`mode=concat_normalize`) for normalize+concat via pure FFmpeg (video + audio, faster than full composition render).

## Quick Start

1. Install dependencies:
   ```bash
   pip install -e .
   ```
2. Launch the API server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
3. Send a render request:
   ```bash
   curl -X POST http://localhost:8000/render \
      -H "Content-Type: application/json" \
      -d @examples/test1.json
   ```
4. Download a render from the server:
   ```bash
   curl -O http://localhost:8000/downloads/<filename>.mp4
   ```

See [API_REFERENCE.md](API_REFERENCE.md) for the full payload specification and the `examples/` directory for ready-to-use request bodies.

## Run In Docker

1. Build and start the service:
   ```bash
   docker compose up --build -d
   ```
   If port `8000` is busy:
   ```bash
   HOST_PORT=8011 docker compose up --build -d
   ```
2. Send a render request with `repro_whip.json`:
   ```bash
   curl -X POST http://localhost:8000/render/raw \
      -H "Content-Type: application/json" \
      -H "Accept: application/json" \
      --data-binary "@repro_whip.json"
   ```
   Replace `8000` with your `HOST_PORT` if you started compose with a custom port.
3. Resulting video will be available at:
   - local path: `renders/whip_test.mp4`
   - HTTP download: `http://localhost:8000/downloads/whip_test.mp4`

The compose file mounts:
- `./media` -> `/app/media` (read-only source assets)
- `${EXTRA_MEDIA_HOST_PATH:-./media}` -> `${EXTRA_MEDIA_CONTAINER_PATH:-/external_media}` (optional external source root for host absolute paths)
- `./renders` -> `/app/renders` (render output)

Performance knobs:
- `RENDER_CONCURRENCY` limits how many render requests the API runs in parallel. Default in compose: `8`.
- `FFMPEG_USE_GPU=1` enables NVENC for output encoding when available. This is already enabled in the current container.
- `FFMPEG_THREADS` controls FFmpeg encoder/worker threads. Default in compose: `8`.
- `FFMPEG_FILTER_THREADS` controls FFmpeg filter graph threads. Default in compose: `8`.
- Requests targeting the same output filename are still serialized to avoid two renders writing to the same file at once.
- For CPU-heavy MoviePy timelines (motion blur / whip pan / custom Python frame filters), the main bottleneck is often Python frame generation, so raising FFmpeg thread counts alone will not fully saturate all CPU cores.

Absolute paths in Docker:
- Containers cannot directly read host paths like `C:/...` unless that host folder is mounted.
- Use `MEDIA_PATH_MAPPINGS` to remap host absolute prefixes to container prefixes.
- Example:
  - `EXTRA_MEDIA_HOST_PATH=C:/Users/Rinzler/Desktop/Video-pipeline/Video-pipeline/tests/STEP_3_MEDIA`
  - `EXTRA_MEDIA_CONTAINER_PATH=/external_media`
  - `MEDIA_PATH_MAPPINGS=C:/Users/Rinzler/Desktop/Video-pipeline/Video-pipeline/tests/STEP_3_MEDIA=/external_media`

