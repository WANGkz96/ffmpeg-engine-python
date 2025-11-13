"""Aspect ratio templates and helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Template:
    name: str
    description: str
    resolution: Tuple[int, int]
    aspect_ratio: str


TEMPLATES: Dict[str, Template] = {
    "youtube_16_9": Template(
        name="youtube_16_9",
        description="Landscape video for YouTube (1920x1080).",
        resolution=(1920, 1080),
        aspect_ratio="16:9",
    ),
    "tiktok_9_16": Template(
        name="tiktok_9_16",
        description="Portrait video for TikTok/Reels (1080x1920).",
        resolution=(1080, 1920),
        aspect_ratio="9:16",
    ),
    "instagram_square": Template(
        name="instagram_square",
        description="Square video for Instagram feed (1080x1080).",
        resolution=(1080, 1080),
        aspect_ratio="1:1",
    ),
    "story_4_5": Template(
        name="story_4_5",
        description="Portrait 4:5 for Instagram posts (1080x1350).",
        resolution=(1080, 1350),
        aspect_ratio="4:5",
    ),
    "story_16_9": Template(
        name="story_16_9",
        description="Landscape story 16:9 (1280x720).",
        resolution=(1280, 720),
        aspect_ratio="16:9",
    ),
}


DEFAULT_TEMPLATE = TEMPLATES["tiktok_9_16"]


def resolve_template(name: str | None) -> Template:
    if name and name in TEMPLATES:
        return TEMPLATES[name]
    return DEFAULT_TEMPLATE

