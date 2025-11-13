"""Data models for the ffmpeg engine instructions."""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, validator


class FitMode(str, Enum):
    """How to fit the source clip inside the template frame."""

    COVER = "cover"  # fill the frame, cropping if necessary
    CONTAIN = "contain"  # fit within the frame and optionally pad/blur


class TransitionType(str, Enum):
    NONE = "none"
    CROSSFADE = "crossfade"
    FADE_BLACK = "fade_black"


class BackgroundMode(str, Enum):
    BLUR = "blur"
    COLOR = "color"


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


class TransitionInstruction(BaseModel):
    type: TransitionType = TransitionType.NONE
    duration: float = Field(0.5, ge=0.0)


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


class ClipInstruction(BaseModel):
    source: Path
    start: float = Field(0.0, ge=0.0)
    end: Optional[float] = Field(None, gt=0.0)
    fit_mode: FitMode = FitMode.CONTAIN
    background_mode: BackgroundMode = BackgroundMode.BLUR
    background_color: ColorModel = Field(default_factory=lambda: ColorModel(r=16, g=16, b=16, a=1.0))
    transitions_after: List[TransitionInstruction] = Field(default_factory=list)
    chroma_key: ChromaKeyInstruction = Field(default_factory=ChromaKeyInstruction)
    adjustments: AdjustmentInstruction = Field(default_factory=AdjustmentInstruction)
    playback_rate: float = Field(1.0, gt=0.0)

    @validator("end")
    def validate_end(cls, v, values):
        start = values.get("start", 0.0)
        if v is not None and v <= start:
            # ignore invalid end marker
            return None
        return v


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


class TextInstruction(BaseModel):
    content: str
    start: float = Field(0.0, ge=0.0)
    end: Optional[float] = Field(None, gt=0.0)
    position: Literal["center", "top", "bottom", "left", "right", "top_left", "top_right", "bottom_left", "bottom_right"] = "center"
    font: str = "DejaVu-Sans"
    font_size: int = Field(48, gt=0)
    color: ColorModel = Field(default_factory=lambda: ColorModel(r=255, g=255, b=255, a=1.0))
    stroke_color: Optional[ColorModel] = None
    stroke_width: int = Field(0, ge=0)
    shadow: bool = False
    glow: bool = False
    max_width: Optional[int] = None
    animation: TextAnimationInstruction = Field(default_factory=TextAnimationInstruction)


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
    filename: str = "rendered.mp4"
    fps: int = Field(30, gt=0)
    bitrate: Optional[str] = None


class RenderRequest(BaseModel):
    output: OutputInstruction = Field(default_factory=OutputInstruction)
    clips: List[ClipInstruction] = Field(default_factory=list)
    audio: List[AudioInstruction] = Field(default_factory=list)
    texts: List[TextInstruction] = Field(default_factory=list)
    images: List[ImageInstruction] = Field(default_factory=list)


class RenderResult(BaseModel):
    status: Literal["ok", "error"]
    duration: float
    output: Path
    message: Optional[str] = None

