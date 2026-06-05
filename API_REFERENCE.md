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
  "timeline": { "channels": [ ... ] },
  "zoom_border": { ... },
  "show_source": { ... },
  "attachments": [ ... ],
  "inserts": [ ... ],
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
- Supports the legacy `clips[]` sequence and global `texts[]`/`zoom_border`/`show_source`.
- Does not implement arbitrary `timeline.channels`, `attachments`, `inserts`, images, or transition composition. Use `mode=render` for full channel composition.
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
| `include_audio` | boolean | `true` | If `false`, the final render is exported without audio. If `true`, clip audio is preserved unless explicit `audio[]` tracks are provided, in which case `audio[]` becomes the final mix. |

\* If `template` is explicitly set and `resolution` is omitted, template resolution is used.

Default `mp4` / `mov` export profile:

- Video: H.264 / AVC (`libx264` or `h264_nvenc`, depending on `FFMPEG_USE_GPU`)
- Pixel format: `yuv420p`
- MP4 tag: `avc1`
- Container flag: `+faststart`
- Audio: AAC-LC, 48 kHz, stereo, 128 kbps

These defaults are chosen for broad compatibility with browsers, mobile devices, and common video platforms.

### Zoom Border

`zoom_border` is an optional global overlay that draws a centered frame showing the area left after applying the same zoom factor as `internal_zoom`. It is applied on top of the final video in `mode=render` and baked into normalized clips in `mode=concat_normalize`.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `width` | integer | `4` | Border line width in pixels. |
| `color` | RGBA object or string | `red` | Border color. Accepts names such as `"red"`, hex strings such as `"#ff0000"`, or `{ "r": 255, "g": 0, "b": 0 }`. |
| `zoom` | float/string | `0.2` | Zoom value using the same semantics as `internal_zoom`. Accepts `0.2`, `20`, or `"20%"`; all mean a visible rectangle sized to `1 / 1.2 = 83.33%` of output width and height. |

Example:

```json
"zoom_border": {
  "width": 4,
  "color": "red",
  "zoom": "20%"
}
```

### Show Source

`show_source` is an optional debug overlay that displays the current source clip name in the upper-left corner, for example `Source: 8348370-uhd_4096_2160_30fps.mp4`.

It works in `mode=render` and `mode=concat_normalize`. In `mode=render`, it follows channel `1` / `clips[]`. In `mode=concat_normalize`, it follows the normalized concat timeline.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `font` | string | `DejaVu-Sans` | Font family/style name or file name resolved through the same system-font lookup as `texts[].font`. |
| `color` | RGBA object or string | `red` | Text color. Accepts names, hex strings, or RGBA objects. |
| `size` | integer | `22` | Font size in pixels. |

Example:

```json
"show_source": {
  "font": "IBM Plex Sans",
  "color": "#ff2d2d",
  "size": 22
}
```

The label uses a small fixed top-left margin and a dark stroke for readability. Long file names are shortened to fit the output width.

### Clip Instructions

Each entry represents a source video fragment.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `source` | string (path) | required | Source video path. Supports relative paths and absolute paths (`C:/...`, `/...`). In Docker, host absolute paths require mount + `MEDIA_PATH_MAPPINGS` remap. |
| `start` | float | `0.0` | Cut-in timestamp in seconds. |
| `end` | float | `null` | Cut-out timestamp. Ignored if before `start` or beyond video duration. |
| `at` | float | `null` | Timeline position in seconds for `mode=render`. If omitted, the clip is placed immediately after the current end of its channel. |
| `fit_mode` | enum | `contain` | `cover` (fill & crop) or `contain` (fit within frame). |
| `background_mode` | enum | `blur` | When `fit_mode=contain`: `blur` or `color`. |
| `background_color` | RGBA | dark gray | Used for padding background or blur intensity alpha. |
| `reframe` | object | `null` | Optional source-level reframing before `fit_mode`/background. Currently supports `{ "mode": "center_zoom", "zoom_percent": 0.08 }`. |
| `internal_zoom` | float | `0.0` | Alias for `reframe.center_zoom`. `0.08` means about 8% center zoom-in. |
| `mirror_horizontal` | bool | `false` | Optional left/right mirror before `fit_mode`/background. Equivalent to ffmpeg `hflip`. |
| `transitions_before` | array | `[]` | Transitions applied to the *start* of this clip (Intro). |
| `transitions_after` | array | `[]` | Transitions applied *after* this clip (Between clips or Outro). |
| `chroma_key` | object | disabled | Removes a color background by generating an alpha mask. See "Chroma Key Settings". |
| `adjustments` | object | zeros | Fine tuning for brightness, contrast, saturation, hue (`-1.0`..`1.0`). |
| `playback_rate` | float | `1.0` | Speed multiplier. |
| `volume` | float | `1.0` | Linear multiplier. |

#### Optional Source Reframe

- Applied to the source frame before `fit_mode` and background composition.
- Supported now: `center_zoom` only.
- `zoom_percent=0.08` means roughly 8% zoom-in from the clip center.
- Omitted or `0` keeps current behavior unchanged.
- Hard-clamped by the engine to `1.0` max.
- Supported in both `mode=render` and `mode=concat_normalize`.

Timeline placement in `mode=render`:
- `clips[]` are placed on channel `1`.
- If `at` is omitted, the clip is appended automatically after the current end of that channel.
- If `at` is present, the clip starts at that timeline position, unless the previous clip's transition creates an overlap.
- Transition overlaps (`crossfade`, `whip_pan`, `motion_blur` with `fade=true`) start the affected clip earlier for the effect, but compensate the overlap by extending that clip. The scheduled end time stays unchanged, so later `at` values keep their original timeline positions.
- To compensate the overlap, the engine first tries to extend the source trim backward, then forward, then slows the prepared clip enough to reach the required duration if the source has no spare media.

Important:
- `clips[].start` / `clips[].end` are source trim markers (what part of each source file to use), not absolute timeline coordinates.
- `clips[].at` is the timeline coordinate.

### Timeline Channels

`mode=render` internally renders video as ordered channels. Higher `channel_id` values are composited above lower channel values.

Backward-compatible mapping:

| Request field | Channel |
| --- | --- |
| `clips[]` | `1` |
| `attachments[]` | `10` |
| `inserts[]` | `11` |

Future/new channel syntax:

```json
"timeline": {
  "channels": [
    {
      "channel_id": 3,
      "clips": [
        {
          "source": "media/overlay.mp4",
          "at": 1.2,
          "start": 0,
          "end": 4,
          "fit_mode": "contain",
          "chroma_key": { "enabled": true, "color": "#00ff00" }
        }
      ]
    }
  ]
}
```

Single-channel shorthand is also accepted:

```json
"timeline": {
  "channel_id": 3,
  "clips": [ ... ]
}
```

Channel behavior:
- Clips within the same channel keep request order for z-order if they overlap.
- A clip without `at` starts at the current end of that channel.
- A clip with `at` can overlap previous clips on the same channel.
- `transitions_after` applies between adjacent clips on the same channel; for the last scheduled clip it behaves as an outro.
- Global `texts[]` and `images[]` are drawn above channels below `10` and below legacy `attachments[]`/`inserts[]`. `zoom_border` is drawn last.

#### Chroma Key Settings

`chroma_key` works on both `clips[]` and `attachments[]` / `inserts[]`. The controls follow FFmpeg `colorkey` semantics: `similarity` defines the fully transparent color radius, and `blend` defines the soft alpha falloff outside that radius.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | bool | `false` | Enables color-key transparency. |
| `color` | RGBA object or string | `green` | Key color to remove. Accepts `{ "r": 0, "g": 255, "b": 0 }`, `"#00ff00"`, `"0x00ff00"`, or common names such as `"green"`, `"blue"`, `"red"`. |
| `similarity` | float `0.0..1.0` | `0.1` | FFmpeg-like color radius. Lower values are stricter; higher values remove more colors around the key color. |
| `blend` | float `0.0..1.0` | `0.04` | Soft alpha falloff outside `similarity`. Higher values create smoother semi-transparent edges. |
| `edge_blur` | float | `0.75` | Gaussian blur radius in pixels for the generated alpha mask. Use small values (`0.5..2`) to smooth jagged edges. |
| `spill` | float `0.0..1.0` | `0.0` | Optional reduction of key-color spill on semi-transparent edge pixels. Useful for green/blue fringes. |
| `threshold` | float `0.0..1.0` | `null` | Backward-compatible alias for `similarity`. |
| `softness` | float `0.0..1.0` | `null` | Backward-compatible alias for `blend`. |

Example:

```json
{
  "source": "media/Chromakey_subscribe.mp4",
  "at": 0.4,
  "placement": "time",
  "fit_mode": "contain",
  "volume": 0,
  "chroma_key": {
    "enabled": true,
    "color": "green",
    "similarity": 0.12,
    "blend": 0.06,
    "edge_blur": 1.0,
    "spill": 0.15
  }
}
```

When `fit_mode="contain"` and `chroma_key.enabled=true`, the engine keeps the padded canvas transparent instead of adding the normal blurred/color background. This prevents keyed inserts from carrying a green or blurred source-background rectangle into the final composite.

### Attachment / Insert Overlays

`attachments` and `inserts` are backward-compatible overlay arrays. Internally they are rendered as normal timeline channels:

- `attachments[]` -> channel `10`
- `inserts[]` -> channel `11`

These entries inherit the same source-level options as `clips`:

- `source`, `start`, `end`
- `fit_mode`, `background_mode`, `background_color`
- `reframe`, `internal_zoom`
- `transitions_before`, `transitions_after`
- `chroma_key`, `adjustments`
- `playback_rate`, `volume`

Additional fields:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `at` | float | `null` | Timeline position where the insert starts when `placement="time"`. If omitted, it is auto-placed after the current end of its channel. |
| `placement` | enum | `time` | `time`, `start`, `end`. |

Behavior:

- Attachments/inserts are composited above `clips[]` because their channels have higher numbers.
- Explicit `at` values are not shifted by `clips[]` transitions. Transition compensation keeps the lower-channel duration stable, so overlays stay synchronized against the original calculated timeline.
- `placement="start"` ignores `at` and starts at timeline `0.0`.
- `placement="end"` ignores `at` and aligns the insert end with the lower-channel timeline duration.
- `placement="time"` uses `at` as the timeline start.
- Legacy attachments/inserts are trimmed to the lower-channel timeline duration so they do not extend old-style renders unexpectedly.
- Transitions between adjacent attachments/inserts on the same channel use the same channel scheduler as `clips[]`.

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
| `font` | string | `DejaVu-Sans` | Font family/style name or file name. The engine resolves direct file names, installed system fonts from `FFMPEG_FONT_DIRS` / default OS font directories, and bundled fallbacks. |
| `font_size` | int | `48` | Font size in px. |
| `bottom_offset_px` | int | `null` | Bottom offset in px for `bottom`, `bottom_left`, `bottom_right`. When omitted, the default bottom margin is used. |
| `color` | RGBA | white | Text color. |
| `stroke_color` | RGBA | `null` | Outline color. |
| `stroke_width` | int | `0` | Outline width. |
| `shadow` | bool | `false` | If `true`, apply drop shadow. |
| `glow` | bool | `false` | If `true`, apply glow (requires font raster support). |
| `max_width` | int | `null` | Force wrapping width. |
| `animation` | object | defaults | `{"fade_in":0.3,"fade_out":0.3,"letter_spacing":null}`. |

Font resolution notes:

- Request fonts by name, for example `"font": "DejaVu-Sans"`, `"font": "Arial"`, `"font": "Arial Bold"`, or `"font": "Comic Sans MS"`.
- On Windows host runs, the engine scans `C:/Windows/Fonts` and `%LOCALAPPDATA%/Microsoft/Windows/Fonts`.
- In Docker Compose, `C:/Windows/Fonts` and `%LOCALAPPDATA%/Microsoft/Windows/Fonts` are mounted read-only to `/system_fonts/windows` and `/system_fonts/windows_user` by default and included in `FFMPEG_FONT_DIRS`.
- To use another host font folder in Docker, set `SYSTEM_FONTS_HOST_PATH` or override `FFMPEG_FONT_DIRS`.

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
      {"index": 0, "source": "C:/.../a.mp4", "start": 0.0, "end": 3.0, "auto_placed": true, "channel_id": 1, "kind": "clip"},
      {"index": 1, "source": "C:/.../b.mp4", "start": 3.0, "end": 6.0, "auto_placed": true, "channel_id": 1, "kind": "clip"}
    ],
    "attachments": [
      {"index": 0, "source": "C:/.../intro.mp4", "start": 0.0, "end": 1.2, "auto_placed": true, "channel_id": 10, "kind": "attachment"}
    ],
    "inserts": [],
    "channels": [
      {
        "channel_id": 1,
        "clips": [
          {"index": 0, "source": "C:/.../a.mp4", "start": 0.0, "end": 3.0, "auto_placed": true, "channel_id": 1, "kind": "clip"}
        ]
      },
      {
        "channel_id": 10,
        "clips": [
          {"index": 0, "source": "C:/.../intro.mp4", "start": 0.0, "end": 1.2, "auto_placed": true, "channel_id": 10, "kind": "attachment"}
        ]
      }
    ],
    "total_duration": 8.2
  }
}
```

`timeline.*[].source` echoes the original source path from the request. It does not expose internal Docker remap paths such as `/external_media/...`.

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

