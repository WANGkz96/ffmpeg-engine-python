# FFmpeg Engine API Reference

## Overview

The FFmpeg Engine is an HTTP service that renders videos using `moviepy` and `ffmpeg`. It accepts JSON instructions describing the output format, clip timeline, overlays, and audio design. The `/render` endpoint validates input, renders the video, and can return either a JSON response (with download URL + absolute path) or the binary file.

Base URL: `http://<host>:<port>/`

## Endpoints

### `GET /`
Simple health check returning `{"status": "ok"}`.

### `GET /templates`
Returns the registered aspect-ratio templates:

```json
{
  "items": [
    {
      "name": "tiktok_9_16",
      "description": "Portrait video for TikTok/Reels (1080x1920).",
      "resolution": [1080, 1920],
      "aspect_ratio": "9:16"
    }
  ]
}
```

### `POST /render`
Consumes a structured JSON payload (see below).

- Default (`application/json`): returns status, duration, absolute output path, and `/downloads/...` URL.
- Video MIME type (`video/mp4`, `video/webm`, ...): streams the produced file if its extension matches the MIME type.
- Query flag `detail_answer=true` (or body field `detail_answer: true`) adds timeline details per clip (`start`/`end` rounded to tenths).
- Concurrency is controlled per mode on the server side. Typical production setup: `mode=render` runs in a limited worker pool, while `mode=concat_normalize` is processed through a smaller queue.

### `POST /render/raw`
Accepts arbitrary JSON and validates it against the schema at runtime. Useful when the caller cannot send typed JSON bodies (e.g., curl from shell).

### `GET /downloads/`
Returns the list of rendered files with absolute paths and downloadable URLs.

### `GET /downloads/{filename}`
Downloads a previously rendered file from the `renders/` workspace.

Network note:

- The API can be exposed to LAN / ZeroTier clients when the server listens on `0.0.0.0`.
- Returned `download_url` values are built from the incoming request host by default.
- Set `PUBLIC_BASE_URL` to force an externally reachable base URL in JSON responses, for example `http://192.168.1.50:8000` or `http://10.147.20.5:8000`.
- Cross-origin browser access can be controlled through `CORS_ALLOW_ORIGINS` (`*` or comma-separated origins).

## Instruction Schema

High-level structure:

```json
{
  "mode": "render",
  "output": { ... },
  "clips": [ ... ],
  "audio": [ ... ],
  "texts": [ ... ],
  "images": [ ... ],
  "detail_answer": false
}
```

### Processing Mode

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `mode` | enum | `render` | `render` = current MoviePy render pipeline. `concat_normalize` = fast FFmpeg pipeline: normalize each clip to target FPS+resolution and concatenate. |

#### `mode=concat_normalize` behavior

- Uses FFmpeg directly (no MoviePy composition).
- Purpose: fast video preparation pipeline (`normalize -> concat`).
- Uses `output.fps` and `output.resolution` as normalization target.
- Keeps aspect ratio and pads to target canvas (black bars if needed, no crop).
- `output.format` in this mode: `mp4`, `mov`, `mkv`.
- Preserves audio by normalizing it to AAC 48kHz stereo before concatenation.
- If a clip has no audio stream, silent audio is generated for that clip to keep concat compatibility.
- Ignores transitions/text/images and other composition effects.
- Still returns the same response structure and supports `detail_answer`.

### Output Settings

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `template` | string | `null` | One of `youtube_16_9`, `tiktok_9_16`, `instagram_square`, `story_4_5`, `story_16_9`. |
| `resolution` | object | `{ "width": 1920, "height": 1080 }`* | Overrides template. |
| `format` | string | `mp4` | Supported: `mp4`, `mov`, `webm`, `mkv`, etc. |
| `filename` | string | current datetime (e.g. `2026-03-01_17-39-20`) | Final file name stored under `renders/`. |
| `fps` | integer | `30` | Target frame rate. |
| `bitrate` | string | `null` | Optional ffmpeg bitrate string, e.g., `"6M"`. If omitted for `mp4`/`mov`, the engine falls back to quality-based H.264 encoding (`CRF/CQ`) instead of an arbitrary bitrate. |

\* If `template` is explicitly set and `resolution` is omitted, template resolution is used.

Default `mp4` / `mov` export profile:

- Video: H.264 / AVC (`libx264` or `h264_nvenc`, depending on `FFMPEG_USE_GPU`)
- Pixel format: `yuv420p`
- MP4 tag: `avc1`
- Container flag: `+faststart`
- Audio: AAC-LC, 48 kHz, stereo, 128 kbps

These defaults are chosen for broad compatibility with browsers, mobile devices, and common video platforms.

### Clip Instructions

Each entry represents a source video fragment.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `source` | string (path) | required | Source video path. Supports relative paths and absolute paths (`C:/...`, `/...`). In Docker, host absolute paths require mount + `MEDIA_PATH_MAPPINGS` remap. |
| `start` | float | `0.0` | Cut-in timestamp in seconds. |
| `end` | float | `null` | Cut-out timestamp. Ignored if before `start` or beyond video duration. |
| `fit_mode` | enum | `contain` | `cover` (fill & crop) or `contain` (fit within frame). |
| `background_mode` | enum | `blur` | When `fit_mode=contain`: `blur` or `color`. |
| `background_color` | RGBA | dark gray | Used for padding background or blur intensity alpha. |
| `reframe` | object | `null` | Optional source-level reframing before `fit_mode`/background. Currently supports `{ "mode": "center_zoom", "zoom_percent": 0.08 }`. |
| `internal_zoom` | float | `0.0` | Alias for `reframe.center_zoom`. `0.08` means about 8% center zoom-in. |
| `transitions_before` | array | `[]` | Transitions applied to the *start* of this clip (Intro). |
| `transitions_after` | array | `[]` | Transitions applied *after* this clip (Between clips or Outro). |
| `chroma_key` | object | disabled | `{ "enabled": true, "color": {"r":0,"g":255,"b":0}, "threshold":0.1, "softness":0.0 }`. |
| `adjustments` | object | zeros | Fine tuning for brightness, contrast, saturation, hue (`-1.0`..`1.0`). |
| `playback_rate` | float | `1.0` | Speed multiplier. |
| `volume` | float | `1.0` | Linear multiplier. |

#### Optional Source Reframe

- Applied to the source frame before `fit_mode` and background composition.
- Supported now: `center_zoom` only.
- `zoom_percent=0.08` means roughly 8% zoom-in from the clip center.
- Omitted or `0` keeps current behavior unchanged.
- Hard-clamped by the engine to `0.25` max.
- Supported in both `mode=render` and `mode=concat_normalize`.

If both `start` and `end` are omitted in a clip object, the clip is appended automatically to the timeline in request order:
- first clip starts at `0.0`
- each next clip starts where the previous visible clip ends

Important:
- `clips[].start` / `clips[].end` are source trim markers (what part of each source file to use), not absolute timeline coordinates.

Absolute host paths in Docker:
- A container cannot read host filesystem paths (`C:/...`) unless that host directory is mounted into the container.
- Use compose vars:
  - `EXTRA_MEDIA_HOST_PATH` (host folder to mount)
  - `EXTRA_MEDIA_CONTAINER_PATH` (container mount point, e.g. `/external_media`)
  - `MEDIA_PATH_MAPPINGS` for path remap (`SRC_PREFIX=DST_PREFIX`, multiple entries via `;`)
- Example:
  - `EXTRA_MEDIA_HOST_PATH=C:/Users/Rinzler/Desktop/Video-pipeline/Video-pipeline/tests/STEP_3_MEDIA`
  - `EXTRA_MEDIA_CONTAINER_PATH=/external_media`
  - `MEDIA_PATH_MAPPINGS=C:/Users/Rinzler/Desktop/Video-pipeline/Video-pipeline/tests/STEP_3_MEDIA=/external_media`

### Transition Settings

Used in `transitions_before` and `transitions_after`.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `type` | enum | `none` | `crossfade`, `fade_black`, `whip_pan`, `motion_blur`. |
| `duration` | float | `0.5` | Duration of the transition in seconds. |
| `direction` | enum | `left` | `left`, `right`, `top`, `bottom`. Used by `whip_pan`. |
| `blur_strength`| float | `300.0` | Intensity of motion blur for `whip_pan` and `motion_blur`. |
| `fade` | bool | `false` | If `true`, adds an opacity fade to the transition. |
| `glow` | bool | `false` | If `true`, adds a dynamic brightness boost (glow) during the transition. |

#### Transition Types Detail

- **`crossfade`**: Standard opacity overlap between clips.
- **`fade_black`**: Fades the outgoing clip to black, then fades the incoming clip from black.
- **`whip_pan`**: A fast "camera pan" effect with global motion blur. Clips move in the specified `direction`.
- **`motion_blur`**: Applies a directional blur without moving the clips. Can be combined with `fade` and `glow`.
    - If `fade=false`, the transition is a sequential cut at the peak of the blur.
    - If `fade=true`, it performs a crossfade while blurring.

### Audio Instructions

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `source` | string | required | Path to audio file. |
| `start` | float | `0.0` | Timeline start. |
| `end` | float | clip duration | Trim point. |
| `volume` | float | `1.0` | Linear multiplier. |
| `fade_in` | float | `0.0` | Fade-in duration (seconds). |
| `fade_out` | float | `0.0` | Fade-out duration. |

### Text Overlays

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `content` | string | required | Text contents. |
| `start` | float | `0.0` | Start time. |
| `end` | float | video duration | End time. |
| `position` | enum | `center` | `center`, `top`, `bottom`, `left`, `right`, `top_left`, `top_right`, `bottom_left`, `bottom_right`. |
| `font` | string | `DejaVu-Sans` | Must be available on host. |
| `font_size` | int | `48` | Font size in px. |
| `color` | RGBA | white | Text color. |
| `stroke_color` | RGBA | `null` | Outline color. |
| `stroke_width` | int | `0` | Outline width. |
| `shadow` | bool | `false` | If `true`, apply drop shadow. |
| `glow` | bool | `false` | If `true`, apply glow (requires font raster support). |
| `max_width` | int | `null` | Force wrapping width. |
| `animation` | object | defaults | `{"fade_in":0.3,"fade_out":0.3,"letter_spacing":null}`. |

### Image Overlays (PNG etc.)

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `source` | string | required | Path to the image. PNG with alpha recommended. |
| `start` | float | `0.0` | Start time. |
| `end` | float | video duration | End time. |
| `position` | enum | `center` | Same set as text. |
| `size` | object | `null` | `{ "width": 400, "height": 400 }` to force resize. |
| `animation` | object | defaults | `{"fade_in":0.2,"fade_out":0.2,"keyframes":[]}`. Keyframes accept `{"time":1.0,"duration":0.5,"scale":1.1}`. |

## Response Format

Default JSON response:

```json
{
  "status": "ok",
  "duration": 8.23,
  "output": {
    "filename": "2026-03-01_17-39-20.mp4",
    "absolute_path": "C:/.../renders/2026-03-01_17-39-20.mp4",
    "download_url": "http://localhost:8000/downloads/2026-03-01_17-39-20.mp4"
  }
}
```

Detailed response (`detail_answer=true`):

```json
{
  "status": "ok",
  "duration": 8.23,
  "output": {
    "filename": "2026-03-01_17-39-20.mp4",
    "absolute_path": "C:/.../renders/2026-03-01_17-39-20.mp4",
    "download_url": "http://localhost:8000/downloads/2026-03-01_17-39-20.mp4"
  },
  "timeline": {
    "clips": [
      {"index": 0, "source": "C:/.../a.mp4", "start": 0.0, "end": 3.0, "auto_placed": true},
      {"index": 1, "source": "C:/.../b.mp4", "start": 3.0, "end": 6.0, "auto_placed": true}
    ],
    "total_duration": 8.2
  }
}
```

`timeline.clips[].source` echoes the original source path from the request. It does not expose internal Docker remap paths such as `/external_media/...`.

## Error Handling

- Invalid or missing fields fall back to safe defaults whenever possible.
- Cutting ranges outside the media duration are ignored instead of raising an error.
- Missing clip sources fail the render with HTTP 400.
- Missing image/audio sources are skipped (warning only).
- Fatal errors return `{ "status": "error", "message": "..." }` with HTTP 400.

## Running the Service

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Rendered files are stored under the `renders/` directory.

