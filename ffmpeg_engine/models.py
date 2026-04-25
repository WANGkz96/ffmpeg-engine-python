"""Data models for the ffmpeg engine instructions."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, root_validator, validator

MAX_INTERNAL_ZOOM = 0.25


def clamp_internal_zoom(value) -> float:
    if value is None:
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(numeric, MAX_INTERNAL_ZOOM))


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
    threshold: float = Field(0.1, ge=0.0)
    softness: float = Field(0.0, ge=0.0)


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
    start: float = Field(0.0, ge=0.0)
    end: Optional[float] = Field(None, gt=0.0)
    fit_mode: FitMode = FitMode.CONTAIN
    background_mode: BackgroundMode = BackgroundMode.BLUR
    background_color: ColorModel = Field(default_factory=lambda: ColorModel(r=16, g=16, b=16, a=1.0))
    reframe: Optional[ReframeInstruction] = None
    internal_zoom: float = 0.0

    # НОВОЕ: переходы, которые применяются к самому клипу "в начале"
    transitions_before: List[TransitionInstruction] = Field(default_factory=list)

    # СТАРОЕ: переходы, которые применяются "после этого клипа" (к следующему)
    transitions_after: List[TransitionInstruction] = Field(default_factory=list)

    chroma_key: ChromaKeyInstruction = Field(default_factory=ChromaKeyInstruction)
    adjustments: AdjustmentInstruction = Field(default_factory=AdjustmentInstruction)
    playback_rate: float = Field(1.0, gt=0.0)
    volume: float = Field(1.0, ge=0.0)

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
    at: float = Field(0.0, ge=0.0)
    placement: InsertPlacement = InsertPlacement.TIME


class AudioInstruction(BaseModel):
    source: Path
    start: float = Field(0.0, ge=0.0)
    end: Optional[float] = Field(None, gt=0.0)
    volume: float = Field(1.0, ge=0.0)
    fade_in: float = Field(0.0, ge=0.0)
    fade_out: float = Field(0.0, ge=0.0)


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


class RenderRequest(BaseModel):
    mode: ProcessingMode = ProcessingMode.RENDER
    output: OutputInstruction = Field(default_factory=OutputInstruction)
    clips: List[ClipInstruction] = Field(default_factory=list)
    attachments: List[InsertInstruction] = Field(default_factory=list)
    audio: List[AudioInstruction] = Field(default_factory=list)
    texts: List[TextInstruction] = Field(default_factory=list)
    images: List[ImageInstruction] = Field(default_factory=list)
    detail_answer: bool = False

    @root_validator(pre=True)
    def normalize_insert_aliases(cls, values):
        if not isinstance(values, dict):
            return values

        attachments = values.get("attachments")
        inserts = values.get("inserts")

        normalized_attachments = attachments if isinstance(attachments, list) else []
        normalized_inserts = inserts if isinstance(inserts, list) else []

        if normalized_attachments or normalized_inserts:
            values["attachments"] = [*normalized_attachments, *normalized_inserts]

        return values


class TimelineClipModel(BaseModel):
    index: int
    source: Path
    start: float
    end: float
    auto_placed: bool = False


class TimelineDetailModel(BaseModel):
    clips: List[TimelineClipModel] = Field(default_factory=list)
    attachments: List[TimelineClipModel] = Field(default_factory=list)


class RenderResult(BaseModel):
    status: Literal["ok", "error"]
    duration: float
    output: Path
    message: Optional[str] = None
    timeline: Optional[TimelineDetailModel] = None

