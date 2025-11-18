# FFmpeg Engine Python

HTTP-based video rendering microservice powered by FastAPI, MoviePy, and FFmpeg. The service consumes JSON instructions describing clips, transitions, overlays, and audio layers, then produces a rendered video or a structured JSON response.

## Features

- Aspect ratio templates for popular platforms (YouTube, TikTok, Instagram, Stories).
- Timeline instructions supporting clip trimming, auto-fit modes (cover/contain), background blur or solid color, and optional chroma key.
- Crossfade and fade-to-black transitions.
- Color adjustments (brightness, contrast, saturation, hue).
- Audio mixing with precise start/end trimming, fades, and volume controls.
- Text and PNG overlays with positioning, animation, and basic keyframe scaling.
- Safe defaults to avoid runtime crashes when instructions are incomplete.

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

See [API_REFERENCE.md](API_REFERENCE.md) for the full payload specification and the `examples/` directory for ready-to-use request bodies.

