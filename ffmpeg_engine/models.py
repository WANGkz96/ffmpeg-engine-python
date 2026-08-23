"""Data models for the ffmpeg engine instructions."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, root_validator, validator

MAX_INTERNAL_ZOOM = 1.0


def clamp_internal_zoom(value) -> float:
    if value is None:
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(numeric, MAX_INTERNAL_ZOOM))


def parse_bool_flag(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "on"}


def parse_percent_value(value, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        is_percent = text.endswith("%")
        if is_percent:
            text = text[:-1].strip()
        try:
            numeric = float(text)
        except ValueError:
            return default
        if is_percent or numeric > 1.0:
            numeric /= 100.0
        return numeric
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if numeric > 1.0:
        numeric /= 100.0
    return numeric


class FitMode(str, Enum):
    """How to fit the source clip inside the template frame."""

    COVER = "cover"  # fill the frame, cropping if necessary
    CONTAIN = "contain"  # fit within the frame and optionally pad/blur


class TransitionType(str, Enum):
    NONE = "none"
    CROSSFADE = "crossfade"
    FADE_BLACK = "fade_black"
    WHIP_PAN = "whip_pan"
    MOTION_BLUR = "motion_blur"


class BackgroundMode(str, Enum):
    BLUR = "blur"
    COLOR = "color"


class InsertPlacement(str, Enum):
    TIME = "time"
    START = "start"
    END = "end"


class ProcessingMode(str, Enum):
    RENDER = "render"
    CONCAT_NORMALIZE = "concat_normalize"
    EDITOR_PROXY = "editor_proxy"


class TransitionDirection(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"


class ReframeMode(str, Enum):
    CENTER_ZOOM = "center_zoom"


class ColorModel(BaseModel):
    r: int = Field(128, ge=0, le=255)
    g: int = Field(128, ge=0, le=255)
    b: int = Field(128, ge=0, le=255)
    a: float = Field(1.0, ge=0.0, le=1.0)

    def as_tuple(self) -> tuple:
        return (self.r, self.g, self.b)


def _parse_color_value(value: Any) -> Any:
    if isinstance(value, ColorModel) or isinstance(value, dict):
        return value
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        payload = {"r": value[0], "g": value[1], "b": value[2]}
        if len(value) >= 4:
            payload["a"] = value[3]
        return payload
    if not isinstance(value, str):
        return value

    color_text = value.strip().lower()
    named_colors = {
        "black": (0, 0, 0),
        "blue": (0, 0, 255),
        "cyan": (0, 255, 255),
        "green": (0, 255, 0),
        "lime": (0, 255, 0),
        "magenta": (255, 0, 255),
        "red": (255, 0, 0),
        "white": (255, 255, 255),
        "yellow": (255, 255, 0),
    }
    if color_text in named_colors:
        r, g, b = named_colors[color_text]
        return {"r": r, "g": g, "b": b, "a": 1.0}

    if color_text.startswith("#"):
        color_text = color_text[1:]
    elif color_text.startswith("0x"):
        color_text = color_text[2:]

    if len(color_text) in {6, 8}:
        try:
            r = int(color_text[0:2], 16)
            g = int(color_text[2:4], 16)
            b = int(color_text[4:6], 16)
            payload = {"r": r, "g": g, "b": b}
            if len(color_text) == 8:
                payload["a"] = int(color_text[6:8], 16) / 255.0
            return payload
        except ValueError:
            return value

    return value


class ResolutionModel(BaseModel):
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)


def default_output_filename() -> str:
    # Datetime-based filename to avoid collisions when output block is omitted.
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def default_output_resolution() -> ResolutionModel:
    return ResolutionModel(width=1920, height=1080)


class TransitionInstruction(BaseModel):
    type: TransitionType = TransitionType.NONE
    duration: float = Field(0.5, ge=0.0)
    direction: TransitionDirection = TransitionDirection.LEFT
    blur_strength: float = Field(300.0, ge=0.0, description="Motion blur intensity for whip_pan transitions")
    fade: bool = Field(False, description="Combine with fade opacity")
    glow: bool = Field(False, description="Combine with glow effect")


class ChromaKeyInstruction(BaseModel):
    enabled: bool = False
    color: ColorModel = Field(default_factory=lambda: ColorModel(r=0, g=255, b=0, a=1.0))
    similarity: float = Field(
        0.1,
        ge=0.0,
        le=1.0,
        description="FFmpeg-like key color radius. 0.01 is strict, 1.0 matches everything.",
    )
    blend: float = Field(
        0.04,
        ge=0.0,
        le=1.0,
        description="Soft alpha falloff outside similarity radius.",
    )
    edge_blur: float = Field(
        0.75,
        ge=0.0,
        description="Gaussian blur radius for the generated alpha mask, in pixels.",
    )
    spill: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Optional key-color spill reduction.",
    )
    threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Deprecated alias for similarity.",
    )
    softness: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Deprecated alias for blend.",
    )

    @root_validator(pre=True)
    def normalize_legacy_aliases(cls, values):
        if not isinstance(values, dict):
            return values
        if "threshold" in values and "similarity" not in values:
            values["similarity"] = values["threshold"]
        if "softness" in values and "blend" not in values:
            values["blend"] = values["softness"]
        return values

    @validator("color", pre=True)
    def parse_color(cls, value):
        return _parse_color_value(value)

    @property
    def effective_similarity(self) -> float:
        value = self.threshold if self.threshold is not None else self.similarity
        return max(0.0, min(float(value), 1.0))

    @property
    def effective_blend(self) -> float:
        value = self.softness if self.softness is not None else self.blend
        return max(0.0, min(float(value), 1.0))


class AdjustmentInstruction(BaseModel):
    brightness: float = Field(0.0, ge=-1.0, le=1.0)
    contrast: float = Field(0.0, ge=-1.0, le=1.0)
    saturation: float = Field(0.0, ge=-1.0, le=1.0)
    hue: float = Field(0.0, ge=-1.0, le=1.0)


class ReframeInstruction(BaseModel):
    mode: ReframeMode = ReframeMode.CENTER_ZOOM
    zoom_percent: float = Field(0.0, ge=0.0)

    @validator("zoom_percent", pre=True, always=True)
    def clamp_zoom_percent(cls, v):
        return clamp_internal_zoom(v)


class ClipInstruction(BaseModel):
    source: Path
    source_label: Optional[str] = None
    source_type: Optional[str] = None
    source_resolution: Optional[Any] = None
    quality_label: Optional[str] = None
    start: float = Field(0.0, ge=0.0)
    end: Optional[float] = Field(None, gt=0.0)
    at: Optional[float] = Field(None, ge=0.0)
    fit_mode: FitMode = FitMode.CONTAIN
    background_mode: BackgroundMode = BackgroundMode.BLUR
    background_color: ColorModel = Field(default_factory=lambda: ColorModel(r=16, g=16, b=16, a=1.0))
    reframe: Optional[ReframeInstruction] = None
    internal_zoom: float = 0.0
    mirror_horizontal: bool = False

    # НОВОЕ: переходы, которые применяются к самому клипу "в начале"
    transitions_before: List[TransitionInstruction] = Field(default_factory=list)

    # СТАРОЕ: переходы, которые применяются "после этого клипа" (к следующему)
    transitions_after: List[TransitionInstruction] = Field(default_factory=list)

    chroma_key: ChromaKeyInstruction = Field(default_factory=ChromaKeyInstruction)
    adjustments: AdjustmentInstruction = Field(default_factory=AdjustmentInstruction)
    playback_rate: float = Field(1.0, gt=0.0)
    volume: float = Field(1.0, ge=0.0)

    @root_validator(pre=True)
    def normalize_clip_aliases(cls, values):
        if isinstance(values, dict) and "playback_rate" not in values and "playbackRate" in values:
            values = {**values, "playback_rate": values.get("playbackRate")}
        if isinstance(values, dict) and "mirror_horizontal" not in values:
            for alias in ("mirrorHorizontal", "horizontal_flip", "horizontalFlip", "hflip"):
                if alias in values:
                    values = {**values, "mirror_horizontal": parse_bool_flag(values.get(alias))}
                    break
        if isinstance(values, dict):
            alias_map = {
                "source_label": ("sourceLabel", "label"),
                "source_type": ("sourceType",),
                "source_resolution": ("sourceResolution",),
                "quality_label": ("qualityLabel", "quality"),
            }
            for target, aliases in alias_map.items():
                if target not in values:
                    for alias in aliases:
                        if alias in values:
                            values = {**values, target: values.get(alias)}
                            break
        return values

    @validator("internal_zoom", pre=True, always=True)
    def clamp_alias_internal_zoom(cls, v):
        return clamp_internal_zoom(v)

    @validator("end")
    def validate_end(cls, v, values):
        start = values.get("start", 0.0)
        if v is not None and v <= start:
            # ignore invalid end marker
            return None
        return v

    @property
    def effective_internal_zoom(self) -> float:
        reframe_zoom = 0.0
        if self.reframe and self.reframe.mode == ReframeMode.CENTER_ZOOM:
            reframe_zoom = clamp_internal_zoom(self.reframe.zoom_percent)
        return max(reframe_zoom, clamp_internal_zoom(self.internal_zoom))



class InsertInstruction(ClipInstruction):
    at: Optional[float] = Field(None, ge=0.0)
    placement: InsertPlacement = InsertPlacement.TIME


class AudioVolumeKeyframe(BaseModel):
    time: float = Field(..., ge=0.0)
    volume: float = Field(..., ge=0.0, le=1.0)


class AudioEffectInstruction(BaseModel):
    type: Literal["reverb"] = "reverb"
    start: float = Field(0.0, ge=0.0)
    end: float = Field(..., gt=0.0)
    mix: float = Field(0.16, ge=0.0, le=0.5)


class VideoEffectInstruction(BaseModel):
    type: Literal["grayscale"] = "grayscale"
    start: float = Field(0.0, ge=0.0)
    end: float = Field(..., gt=0.0)
    fade_in: float = Field(1.0, ge=0.0)
    fade_out: float = Field(1.0, ge=0.0)


class AudioInstruction(BaseModel):
    source: Path
    start: float = Field(0.0, ge=0.0)
    end: Optional[float] = Field(None, gt=0.0)
    at: Optional[float] = Field(None, ge=0.0)
    timeline_start: Optional[float] = Field(None, ge=0.0)
    volume: float = Field(1.0, ge=0.0)
    fade_in: float = Field(0.0, ge=0.0)
    fade_out: float = Field(0.0, ge=0.0)
    volume_keyframes: List[AudioVolumeKeyframe] = Field(default_factory=list)
    effects: List[AudioEffectInstruction] = Field(default_factory=list)

    @root_validator(pre=True)
    def normalize_audio_aliases(cls, values):
        if not isinstance(values, dict):
            return values
        if "timeline_start" not in values:
            for alias in ("timelineStart", "timeline_at", "timelineAt"):
                if alias in values:
                    values = {**values, "timeline_start": values.get(alias)}
                    break
        if "at" not in values and "timeline_start" in values:
            values = {**values, "at": values.get("timeline_start")}
        return values


class TextAnimationInstruction(BaseModel):
    fade_in: float = Field(0.3, ge=0.0)
    fade_out: float = Field(0.3, ge=0.0)
    letter_spacing: Optional[float] = None
    scale_from: float = Field(1.0, gt=0.0)
    scale_to: float = Field(1.0, gt=0.0)
    zoom_duration: Optional[float] = Field(None, ge=0.0)


class TextInstruction(BaseModel):
    content: str
    start: float = Field(0.0, ge=0.0)
    end: Optional[float] = Field(None, gt=0.0)
    position: Literal[
        "center", "top", "bottom", "left", "right",
        "top_left", "top_right", "bottom_left", "bottom_right"
    ] = "center"
    font: str = "DejaVu-Sans"
    font_size: int = Field(48, gt=0)
    bottom_offset_px: Optional[int] = Field(None, ge=0)
    color: ColorModel = Field(default_factory=lambda: ColorModel(r=255, g=255, b=255, a=1.0))
    stroke_color: Optional[ColorModel] = None
    stroke_width: int = Field(0, ge=0)
    shadow: bool = False
    glow: bool = False
    max_width: Optional[int] = None
    animation: TextAnimationInstruction = Field(default_factory=TextAnimationInstruction)
    font_path: Optional[Path] = None



class ImageAnimationInstruction(BaseModel):
    fade_in: float = Field(0.2, ge=0.0)
    fade_out: float = Field(0.2, ge=0.0)
    keyframes: List[dict] = Field(default_factory=list)


class ImageInstruction(BaseModel):
    source: Path
    start: float = Field(0.0, ge=0.0)
    end: Optional[float] = Field(None, gt=0.0)
    position: Literal["center", "top", "bottom", "left", "right", "top_left", "top_right", "bottom_left", "bottom_right"] = "center"
    size: Optional[ResolutionModel] = None
    animation: ImageAnimationInstruction = Field(default_factory=ImageAnimationInstruction)


class ZoomBorderInstruction(BaseModel):
    width: int = Field(4, ge=1)
    color: ColorModel = Field(default_factory=lambda: ColorModel(r=255, g=0, b=0, a=1.0))
    zoom: float = Field(0.2, ge=0.0, lt=1.0)

    @validator("color", pre=True)
    def parse_color(cls, value):
        return _parse_color_value(value)

    @validator("zoom", pre=True, always=True)
    def parse_zoom(cls, value):
        return max(0.0, min(parse_percent_value(value, default=0.2), 0.95))


class ShowSourceInstruction(BaseModel):
    font: str = "DejaVu-Sans"
    color: ColorModel = Field(default_factory=lambda: ColorModel(r=255, g=0, b=0, a=1.0))
    size: int = Field(22, gt=0)

    @validator("color", pre=True)
    def parse_color(cls, value):
        return _parse_color_value(value)


class OutputInstruction(BaseModel):
    template: Optional[str] = None
    resolution: Optional[ResolutionModel] = None
    format: str = "mp4"
    filename: str = Field(default_factory=default_output_filename)
    fps: int = Field(30, gt=0)
    bitrate: Optional[str] = None
    include_audio: bool = True

    @validator("resolution", always=True)
    def apply_default_resolution(cls, v, values):
        if v is not None:
            return v
        if values.get("template"):
            return None
        return default_output_resolution()


class TimelineChannelInstruction(BaseModel):
    channel_id: int = Field(1, ge=0)
    clips: List[ClipInstruction] = Field(default_factory=list)

    @root_validator(pre=True)
    def normalize_channel_id_aliases(cls, values):
        if not isinstance(values, dict):
            return values
        if "channel_id" not in values:
            if "id" in values:
                values["channel_id"] = values["id"]
            elif "channel" in values:
                values["channel_id"] = values["channel"]
        return values


class TimelineInstruction(BaseModel):
    channels: List[TimelineChannelInstruction] = Field(default_factory=list)


class EditorProxyInstruction(BaseModel):
    """Low-bandwidth proxy settings used by a remote video editor preview."""

    source: Path
    short_side: int = Field(480, ge=144, le=2160)
    fps: int = Field(15, ge=1, le=60)
    quality: int = Field(30, ge=0, le=51)
    include_audio: bool = True
    audio_bitrate: str = "64k"

    @root_validator(pre=True)
    def normalize_short_side_alias(cls, values):
        if isinstance(values, dict) and "short_side" not in values and "max_height" in values:
            values["short_side"] = values["max_height"]
        return values


class RenderRequest(BaseModel):
    mode: ProcessingMode = ProcessingMode.RENDER
    output: OutputInstruction = Field(default_factory=OutputInstruction)
    editor_proxy: Optional[EditorProxyInstruction] = None
    clips: List[ClipInstruction] = Field(default_factory=list)
    attachments: List[InsertInstruction] = Field(default_factory=list)
    inserts: List[InsertInstruction] = Field(default_factory=list)
    timeline: Optional[TimelineInstruction] = None
    audio: List[AudioInstruction] = Field(default_factory=list)
    video_effects: List[VideoEffectInstruction] = Field(default_factory=list)
    texts: List[TextInstruction] = Field(default_factory=list)
    images: List[ImageInstruction] = Field(default_factory=list)
    zoom_border: Optional[ZoomBorderInstruction] = None
    show_source: Optional[ShowSourceInstruction] = None
    detail_answer: bool = False

    @root_validator(pre=True)
    def normalize_insert_aliases(cls, values):
        if not isinstance(values, dict):
            return values

        timeline = values.get("timeline")
        if isinstance(timeline, list):
            values["timeline"] = {"channels": timeline}
        elif isinstance(timeline, dict) and "channels" not in timeline and "clips" in timeline:
            values["timeline"] = {"channels": [timeline]}

        return values

    @root_validator(skip_on_failure=True)
    def require_editor_proxy_settings(cls, values):
        if values.get("mode") == ProcessingMode.EDITOR_PROXY and values.get("editor_proxy") is None:
            raise ValueError("editor_proxy settings are required when mode=editor_proxy")
        return values


class TimelineClipModel(BaseModel):
    index: int
    source: Path
    source_label: Optional[str] = None
    source_type: Optional[str] = None
    source_resolution: Optional[Any] = None
    quality_label: Optional[str] = None
    start: float
    end: float
    auto_placed: bool = False
    channel_id: Optional[int] = None
    kind: Optional[str] = None


class TimelineChannelModel(BaseModel):
    channel_id: int
    clips: List[TimelineClipModel] = Field(default_factory=list)


class TimelineDetailModel(BaseModel):
    clips: List[TimelineClipModel] = Field(default_factory=list)
    attachments: List[TimelineClipModel] = Field(default_factory=list)
    inserts: List[TimelineClipModel] = Field(default_factory=list)
    channels: List[TimelineChannelModel] = Field(default_factory=list)


class RenderResult(BaseModel):
    status: Literal["ok", "error"]
    duration: float
    output: Path
    message: Optional[str] = None
    timeline: Optional[TimelineDetailModel] = None

