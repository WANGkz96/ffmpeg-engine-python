"""Core rendering engine built on top of moviepy/ffmpeg."""
from __future__ import annotations

import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import logging
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# Monkey patch Image.ANTIALIAS for Pillow 10+
from PIL import Image
if not hasattr(Image, "ANTIALIAS"):
    if hasattr(Image, "Resampling"):
        Image.ANTIALIAS = Image.Resampling.LANCZOS
    else:
        Image.ANTIALIAS = Image.LANCZOS

import moviepy.editor as mpe
from moviepy.video.fx import all as vfx
from moviepy.config import change_settings
import numpy as np
from PIL import ImageDraw, ImageFilter, ImageFont

change_settings({
    "IMAGEMAGICK_BINARY": "magick",
    "FFMPEG_BINARY": "ffmpeg"
})

# Polyfill for gaussian_blur if missing in moviepy
if not hasattr(vfx, "gaussian_blur"):
    def gaussian_blur(clip, sigma=5):
        def filter_frame(get_frame, t):
            frame = get_frame(t)
            img = Image.fromarray(frame)
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
            return np.array(img)
        return clip.fl(filter_frame)
    vfx.gaussian_blur = gaussian_blur

logger = logging.getLogger(__name__)

# --- Monkey patch or helper functions ---

ADJACENT_INSERT_TOLERANCE_SEC = 0.05

SUPPORTED_FONT_EXTENSIONS = {".ttf", ".otf", ".ttc", ".otc"}


def _normalize_font_lookup_name(value: str) -> str:
    name = (value or "").strip().replace("\\", "/").split("/")[-1]
    suffix = Path(name).suffix.lower()
    if suffix in SUPPORTED_FONT_EXTENSIONS:
        name = Path(name).stem
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _split_font_dirs(value: str) -> list[str]:
    if not value:
        return []
    separator = ";" if ";" in value else os.pathsep
    return [item.strip() for item in value.split(separator) if item.strip()]


def _iter_system_font_dirs() -> list[Path]:
    raw_dirs = _split_font_dirs(os.getenv("FFMPEG_FONT_DIRS", ""))
    default_dirs: list[str] = []
    if os.name == "nt":
        windir = os.getenv("WINDIR", r"C:\Windows")
        local_app_data = os.getenv("LOCALAPPDATA", "")
        default_dirs.append(str(Path(windir) / "Fonts"))
        if local_app_data:
            default_dirs.append(str(Path(local_app_data) / "Microsoft" / "Windows" / "Fonts"))
    else:
        default_dirs.extend(
            [
                "/system_fonts/windows",
                "/system_fonts/windows_user",
                "/usr/share/fonts",
                "/usr/local/share/fonts",
                str(Path.home() / ".fonts"),
                str(Path.home() / ".local" / "share" / "fonts"),
            ]
        )

    seen: set[Path] = set()
    font_dirs: list[Path] = []
    for raw_dir in [*raw_dirs, *default_dirs]:
        font_dir = Path(raw_dir).expanduser()
        try:
            resolved = font_dir.resolve()
        except Exception:
            resolved = font_dir
        if resolved in seen or not font_dir.exists() or not font_dir.is_dir():
            continue
        seen.add(resolved)
        font_dirs.append(font_dir)
    return font_dirs


def _font_style_priority(style: str) -> int:
    normalized = _normalize_font_lookup_name(style)
    if normalized in {"regular", "normal", "book", "roman"}:
        return 0
    if "regular" in normalized or "normal" in normalized:
        return 1
    return 10


def _register_font_candidate(
    index: dict[str, tuple[int, tuple[str, int]]],
    key: str,
    candidate: tuple[str, int],
    priority: int,
) -> None:
    normalized_key = _normalize_font_lookup_name(key)
    if not normalized_key:
        return
    current = index.get(normalized_key)
    if current is None or priority < current[0]:
        index[normalized_key] = (priority, candidate)


@lru_cache(maxsize=1)
def _build_system_font_index() -> dict[str, tuple[str, int]]:
    indexed: dict[str, tuple[int, tuple[str, int]]] = {}
    for font_dir in _iter_system_font_dirs():
        try:
            font_paths = list(font_dir.rglob("*"))
        except Exception:
            continue
        for font_path in font_paths:
            if not font_path.is_file() or font_path.suffix.lower() not in SUPPORTED_FONT_EXTENSIONS:
                continue
            face_indexes = range(0, 16) if font_path.suffix.lower() in {".ttc", ".otc"} else range(0, 1)
            for face_index in face_indexes:
                try:
                    font = ImageFont.truetype(str(font_path), 12, index=face_index)
                    family, style = font.getname()
                except Exception:
                    if face_index == 0:
                        continue
                    break

                candidate = (str(font_path), face_index)
                style_priority = _font_style_priority(style)
                _register_font_candidate(indexed, family, candidate, style_priority)
                _register_font_candidate(indexed, f"{family} {style}", candidate, 0)
                _register_font_candidate(indexed, f"{family}-{style}", candidate, 0)
                _register_font_candidate(indexed, font_path.stem, candidate, 5 + style_priority)

    return {key: value for key, (_, value) in indexed.items()}

def _normalize_motion_blur_kernel(strength: float) -> int:
    k_size = int(strength)
    if k_size < 2:
        return 0
    if k_size % 2 == 0:
        k_size += 1
    return k_size


def _apply_motion_blur_to_frame(frame, k_size: int, direction: str = "horizontal", max_value: float = 255.0):
    if k_size < 2:
        return frame

    img_float = frame.astype(float)
    squeezed = False
    if img_float.ndim == 2:
        img_float = img_float[:, :, None]
        squeezed = True

    radius = k_size // 2
    if direction == "horizontal":
        axis = 1
        pad_width = ((0, 0), (radius, radius), (0, 0))
    else:
        axis = 0
        pad_width = ((radius, radius), (0, 0), (0, 0))

    padded = np.pad(img_float, pad_width, mode="edge")
    cumsum = np.cumsum(padded, axis=axis)

    if axis == 1:
        zeros = np.zeros((cumsum.shape[0], 1, cumsum.shape[2]))
        cumsum_padded = np.hstack((zeros, cumsum))
        upper = cumsum_padded[:, k_size : k_size + img_float.shape[1], :]
        lower = cumsum_padded[:, 0 : img_float.shape[1], :]
    else:
        zeros = np.zeros((1, cumsum.shape[1], cumsum.shape[2]))
        cumsum_padded = np.vstack((zeros, cumsum))
        upper = cumsum_padded[k_size : k_size + img_float.shape[0], :, :]
        lower = cumsum_padded[0 : img_float.shape[0], :, :]

    result = np.clip((upper - lower) / k_size, 0.0, max_value)
    if squeezed:
        result = result[:, :, 0]

    if np.issubdtype(frame.dtype, np.integer):
        return result.astype(frame.dtype)
    return result.astype(frame.dtype, copy=False)


def dynamic_motion_blur(clip, strength_func, direction: str = "horizontal", blur_mask: bool = False):
    """
    Motion blur с динамической силой (зависит от времени).
    strength_func(t) -> float (размер ядра, px)
    direction: "horizontal" | "vertical"
    """
    if blur_mask and clip.mask is not None:
        def filter_frame(get_frame, t):
            rgb = get_frame(t).astype(float)
            alpha = np.clip(clip.mask.get_frame(t).astype(float), 0.0, 1.0)
            k_size = _normalize_motion_blur_kernel(strength_func(t))
            try:
                if rgb.ndim == 2:
                    rgb = rgb[:, :, None]

                alpha_3d = alpha[:, :, None] if alpha.ndim == 2 else alpha
                premultiplied = rgb * alpha_3d
                blurred_rgb = _apply_motion_blur_to_frame(premultiplied, k_size, direction=direction, max_value=255.0)
                blurred_alpha = _apply_motion_blur_to_frame(alpha, k_size, direction=direction, max_value=1.0)
                if blurred_rgb.ndim == 2:
                    blurred_rgb = blurred_rgb[:, :, None]
                if blurred_alpha.ndim == 2:
                    blurred_alpha_3d = blurred_alpha[:, :, None]
                else:
                    blurred_alpha_3d = blurred_alpha

                with np.errstate(divide="ignore", invalid="ignore"):
                    rgb_out = np.where(
                        blurred_alpha_3d > 1e-6,
                        blurred_rgb / blurred_alpha_3d,
                        0.0,
                    )
                rgb_out = np.clip(rgb_out, 0.0, 255.0)
                if get_frame(t).ndim == 2:
                    return rgb_out[:, :, 0].astype(np.uint8)
                return rgb_out.astype(np.uint8)
            except Exception as e:
                logger.warning(f"Motion blur failed: {e}")
                return get_frame(t)

        def filter_mask(get_frame, t):
            frame = get_frame(t)
            k_size = _normalize_motion_blur_kernel(strength_func(t))
            try:
                return _apply_motion_blur_to_frame(frame, k_size, direction=direction, max_value=1.0)
            except Exception as e:
                logger.warning(f"Motion blur mask failed: {e}")
                return frame

        return clip.fl(filter_frame).set_mask(clip.mask.fl(filter_mask))

    def filter_frame(get_frame, t):
        frame = get_frame(t)
        k_size = _normalize_motion_blur_kernel(strength_func(t))
        try:
            return _apply_motion_blur_to_frame(frame, k_size, direction=direction, max_value=255.0)
        except Exception as e:
            logger.warning(f"Motion blur failed: {e}")
            return frame

    return clip.fl(filter_frame)

def dynamic_brightness(clip, factor_func):
    """
    Changes the brightness of the clip dynamically.
    factor_func(t) -> float (multiplier, e.g. 1.0 = original, 1.5 = 50% brighter)
    """
    def filter_frame(get_frame, t):
        frame = get_frame(t)
        factor = factor_func(t)
        # Multiply and clip. Ensure float calculation.
        return np.clip(frame.astype(float) * factor, 0, 255).astype(np.uint8)
    return clip.fl(filter_frame)

from .models import (
    AdjustmentInstruction,
    AudioInstruction,
    BackgroundMode,
    ClipInstruction,
    FitMode,
    ImageInstruction,
    InsertInstruction,
    InsertPlacement,
    ProcessingMode,
    RenderRequest,
    RenderResult,
    ShowSourceInstruction,
    TimelineChannelModel,
    TimelineClipModel,
    TimelineDetailModel,
    TextInstruction,
    TransitionInstruction,
    TransitionType,
    TransitionDirection,
    ZoomBorderInstruction,
)
from .templates import resolve_template


class VideoEngineError(Exception):
    """Raised when the rendering pipeline fails."""


class VideoEngine:
    def __init__(self, workspace: Path | str = "renders") -> None:
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _get_thread_count(env_name: str, default: Optional[int] = None) -> int:
        fallback = default if default is not None else (os.cpu_count() or 4)
        raw_value = os.getenv(env_name, "").strip()
        if not raw_value:
            return max(int(fallback), 1)
        try:
            thread_count = int(raw_value)
        except ValueError:
            logger.warning("Invalid %s=%r; using fallback thread count %s", env_name, raw_value, fallback)
            return max(int(fallback), 1)
        return max(thread_count, 1)

    @staticmethod
    def _get_bool_env(env_name: str, default: bool = False) -> bool:
        raw_value = os.getenv(env_name, "").strip().lower()
        if not raw_value:
            return default
        return raw_value in {"1", "true", "yes", "on"}

    @staticmethod
    def _get_timeout_seconds(env_name: str, default: float) -> Optional[float]:
        raw_value = os.getenv(env_name, "").strip()
        if not raw_value:
            return default if default > 0 else None
        try:
            timeout_seconds = float(raw_value)
        except ValueError:
            logger.warning("Invalid %s=%r; using default timeout %.1fs", env_name, raw_value, default)
            return default
        if timeout_seconds <= 0:
            return None
        return timeout_seconds

    @staticmethod
    def _get_positive_int_env(env_name: str, default: int) -> int:
        raw_value = os.getenv(env_name, "").strip()
        if not raw_value:
            return max(int(default), 1)
        try:
            value = int(raw_value)
        except ValueError:
            logger.warning("Invalid %s=%r; using default value %s", env_name, raw_value, default)
            return max(int(default), 1)
        return max(value, 1)

    def _close_clip_resources(self, resources: Iterable[object]) -> None:
        seen: set[int] = set()
        for resource in resources:
            self._close_clip_resource(resource, seen)

    def _close_clip_resource(self, resource: object, seen: set[int]) -> None:
        if resource is None:
            return

        resource_id = id(resource)
        if resource_id in seen:
            return
        seen.add(resource_id)

        clips = getattr(resource, "clips", None)
        if clips:
            for child in clips:
                self._close_clip_resource(child, seen)

        for attr_name in ("audio", "mask", "bg"):
            child = getattr(resource, attr_name, None)
            if child is not None:
                self._close_clip_resource(child, seen)

        reader = getattr(resource, "reader", None)
        if reader is not None:
            close_proc = getattr(reader, "close_proc", None)
            if callable(close_proc):
                try:
                    close_proc()
                except Exception:
                    pass
            close_reader = getattr(reader, "close", None)
            if callable(close_reader):
                try:
                    close_reader()
                except Exception:
                    pass

        close_resource = getattr(resource, "close", None)
        if callable(close_resource):
            try:
                close_resource()
            except Exception:
                pass

    @staticmethod
    def _is_windows_absolute(path_value: str) -> bool:
        return len(path_value) > 2 and path_value[1] == ":" and path_value[2] in {"\\", "/"}

    @staticmethod
    def _normalize_path_text(path_text: str) -> str:
        return path_text.strip().replace("\\", "/")

    def _map_absolute_path(self, source_text: str) -> Optional[Path]:
        """
        Map absolute host paths to container-visible paths.

        MEDIA_PATH_MAPPINGS format:
          SRC_PREFIX=DST_PREFIX;SRC2=DST2

        Example:
          C:/Users/Rinzler/Desktop/Video-pipeline/Video-pipeline/tests/STEP_3_MEDIA=/external_media
        """
        mappings_raw = os.getenv("MEDIA_PATH_MAPPINGS", "").strip()
        if not mappings_raw:
            return None

        source_norm = self._normalize_path_text(source_text)
        source_is_windows = self._is_windows_absolute(source_norm)
        source_cmp = source_norm.lower() if source_is_windows else source_norm

        for item in mappings_raw.split(";"):
            entry = item.strip()
            if not entry or "=" not in entry:
                continue
            src_prefix_raw, dst_prefix_raw = entry.split("=", 1)
            src_prefix = self._normalize_path_text(src_prefix_raw)
            dst_prefix = self._normalize_path_text(dst_prefix_raw)
            if not src_prefix or not dst_prefix:
                continue

            prefix_is_windows = self._is_windows_absolute(src_prefix)
            cmp_prefix = src_prefix.lower() if prefix_is_windows else src_prefix
            cmp_prefix = cmp_prefix.rstrip("/")
            if source_cmp != cmp_prefix and not source_cmp.startswith(f"{cmp_prefix}/"):
                continue

            relative_part = source_norm[len(src_prefix):].lstrip("/")
            mapped = Path(dst_prefix)
            if relative_part:
                mapped = mapped / Path(relative_part)
            return mapped
        return None

    def _resolve_media_path(self, source: Path) -> Path:
        source_text = str(source).strip()
        path_obj = Path(source_text).expanduser()
        if path_obj.is_absolute():
            return path_obj
        if self._is_windows_absolute(source_text):
            mapped = self._map_absolute_path(source_text)
            if mapped is not None:
                return mapped
            return path_obj
        return (Path.cwd() / path_obj).resolve()

    @staticmethod
    def _get_instruction_fields_set(instruction: ClipInstruction) -> set[str]:
        fields_set = getattr(instruction, "model_fields_set", None)
        if fields_set is None:
            fields_set = getattr(instruction, "__fields_set__", set())
        return set(fields_set or set())

    @classmethod
    def _has_explicit_at(cls, instruction: ClipInstruction) -> bool:
        return "at" in cls._get_instruction_fields_set(instruction) and getattr(instruction, "at", None) is not None

    @classmethod
    def _is_auto_placed_instruction(cls, instruction: ClipInstruction) -> bool:
        placement = getattr(instruction, "placement", InsertPlacement.TIME)
        if placement != InsertPlacement.TIME:
            return True
        return not cls._has_explicit_at(instruction)

    # --- ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ДЛЯ WHIP PAN ---

    def _get_slide_vector(self, direction: TransitionDirection, width: int, height: int) -> Tuple[int, int]:
        """Возвращает вектор смещения (dx, dy) для уходящего клипа."""
        if direction == TransitionDirection.LEFT:
            return (-width, 0)
        elif direction == TransitionDirection.RIGHT:
            return (width, 0)
        elif direction == TransitionDirection.TOP:
            return (0, -height)
        elif direction == TransitionDirection.BOTTOM:
            return (0, height)
        return (-width, 0)

    def _apply_whip_pan_intro(
            self,
            clip: mpe.VideoClip,
            transition: TransitionInstruction,
            resolution: Tuple[int, int],
            transparent_background: bool = False,
    ) -> mpe.VideoClip:
        """Whip pan в начале: влетает в кадр (с черного фона)."""
        duration = min(transition.duration, clip.duration)
        if duration <= 0:
            return clip

        w, h = resolution
        dx, dy = self._get_slide_vector(transition.direction, w, h)

        # Делим клип
        head = clip.subclip(0, duration)
        rest = clip.subclip(duration) if clip.duration > duration else None

        # --- GLOBAL COMPOSITE LOGIC (Similar to _apply_whip_pan_between) ---
        def get_progress(t):
            return min(max(t / duration, 0.0), 1.0)

        # Используем только вторую половину ease-in-out (deceleration)
        # Но для простоты и единообразия используем ту же логику движения,
        # как если бы мы были во второй половине перехода "между" клипами.
        # То есть мы "прилетаем" из (-dx, -dy) в (0,0).
        
        def ease_out_cubic(p):
             return 1 - pow(1 - p, 3)

        # Движение
        def pos_prev(t):
            # Черный фон улетает так же, как улетал бы предыдущий клип
            # Но здесь мы моделируем только фазу "прилета" (deceleration)
            # Если мы хотим полную симметрию с "between", то intro - это как бы вторая половина перехода.
            # Но проще сделать просто ease_out для влета.
            p = get_progress(t)
            eased = ease_out_cubic(p)
            # prev улетает от (0,0) к (dx, dy)
            return (int(dx * eased), int(dy * eased))

        def pos_head(t):
            p = get_progress(t)
            eased = ease_out_cubic(p)
            # head летит от (-dx, -dy) к (0,0)
            return (int(-dx + dx * eased), int(-dy + dy * eased))

        head = head.set_position(pos_head)
        transition_layers: List[mpe.VideoClip] = []
        if not transparent_background:
            prev_clip = mpe.ColorClip(size=resolution, color=(0, 0, 0), duration=duration)
            prev_clip = prev_clip.set_position(pos_prev)
            transition_layers.append(prev_clip)
        transition_layers.append(head)
        transition_clip = mpe.CompositeVideoClip(transition_layers, size=resolution).set_duration(duration)

        max_blur = transition.blur_strength
        blur_direction = "horizontal" if abs(dx) > abs(dy) else "vertical"

        def blur_func(t):
            p = get_progress(t)
            # Blur убывает от максимума к 0
            # Derivative of ease_out_cubic (1 - (1-p)^3) is 3(1-p)^2
            # Normalized: (1-p)^2
            velocity_factor = (1 - p) ** 2
            return max_blur * velocity_factor

        transition_blurred = dynamic_motion_blur(
            transition_clip,
            blur_func,
            direction=blur_direction,
            blur_mask=transparent_background,
        )

        clips = [transition_blurred]
        if rest:
            clips.append(rest.set_start(duration))
        
        return mpe.CompositeVideoClip(clips, size=resolution).set_duration(clip.duration)

    def _apply_whip_pan_outro(
            self,
            clip: mpe.VideoClip,
            transition: TransitionInstruction,
            resolution: Tuple[int, int],
            transparent_background: bool = False,
    ) -> mpe.VideoClip:
        """Whip pan в конце: улетает из кадра (в черный фон)."""
        duration = min(transition.duration, clip.duration)
        if duration <= 0:
            return clip

        w, h = resolution
        dx, dy = self._get_slide_vector(transition.direction, w, h)
        total = clip.duration

        body = clip.subclip(0, total - duration) if total > duration else None
        tail = clip.subclip(max(total - duration, 0), total)

        # --- GLOBAL COMPOSITE LOGIC ---
        def get_progress(t):
            return min(max(t / duration, 0.0), 1.0)

        def ease_in_cubic(p):
            return p * p * p

        # Движение
        def pos_tail(t):
            p = get_progress(t)
            eased = ease_in_cubic(p)
            # tail улетает от (0,0) к (dx, dy)
            return (int(dx * eased), int(dy * eased))

        def pos_next(t):
            p = get_progress(t)
            eased = ease_in_cubic(p)
            # next прилетает от (-dx, -dy) к (0,0)
            # Но так как это outro, мы просто уводим tail, а next (черный) занимает его место
            # next должен двигаться синхронно с tail, находясь слева/справа/сверху/снизу
            return (int(-dx + dx * eased), int(-dy + dy * eased))

        tail = tail.set_position(pos_tail)
        transition_layers: List[mpe.VideoClip] = [tail]
        if not transparent_background:
            next_clip = mpe.ColorClip(size=resolution, color=(0, 0, 0), duration=duration)
            next_clip = next_clip.set_position(pos_next)
            transition_layers.append(next_clip)
        transition_clip = mpe.CompositeVideoClip(transition_layers, size=resolution).set_duration(duration)

        max_blur = transition.blur_strength
        blur_direction = "horizontal" if abs(dx) > abs(dy) else "vertical"

        def blur_func(t):
            p = get_progress(t)
            # Blur нарастает от 0 к максимуму
            # Derivative of ease_in_cubic (p^3) is 3p^2
            # Normalized: p^2
            velocity_factor = p ** 2
            return max_blur * velocity_factor

        transition_blurred = dynamic_motion_blur(
            transition_clip,
            blur_func,
            direction=blur_direction,
            blur_mask=transparent_background,
        )

        clips = []
        if body:
            clips.append(body)
            # transition_blurred начинается сразу после body
            transition_blurred = transition_blurred.set_start(body.duration)
        
        clips.append(transition_blurred)
        return mpe.CompositeVideoClip(clips, size=resolution).set_duration(clip.duration)

    def _apply_whip_pan_between(
            self,
            prev_clip: mpe.VideoClip,
            next_clip: mpe.VideoClip,
            transition: TransitionInstruction,
            resolution: Tuple[int, int]
    ) -> tuple[mpe.VideoClip, mpe.VideoClip, float]:
        """
        Whip pan между двумя клипами с глобальным размытием.
        Создает единый композит перехода, чтобы размытие применялось ко всей сцене целиком.
        """
        d = min(transition.duration, prev_clip.duration, next_clip.duration)
        if d <= 0:
            return prev_clip, next_clip, 0.0

        w, h = resolution
        dx, dy = self._get_slide_vector(transition.direction, w, h)
        
        # --- PREV CLIP (Уходит) ---
        total_prev = prev_clip.duration
        body_prev = prev_clip.subclip(0, total_prev - d) if total_prev > d else None
        tail_prev = prev_clip.subclip(max(total_prev - d, 0), total_prev)

        # --- NEXT CLIP (Приходит) ---
        total_next = next_clip.duration
        head_next = next_clip.subclip(0, d)
        rest_next = next_clip.subclip(d) if total_next > d else None

        # --- GLOBAL COMPOSITE ---
        # Используем единую функцию прогресса для синхронного движения
        def get_progress(t):
            return min(max(t / d, 0.0), 1.0)

        # Ease-in-out cubic для плавного разгона и торможения
        def ease_in_out_cubic(p):
            return 4 * p * p * p if p < 0.5 else 1 - pow(-2 * p + 2, 3) / 2

        # Оба клипа движутся как единое целое
        def pos_tail(t):
            p = get_progress(t)
            eased = ease_in_out_cubic(p)
            return (int(dx * eased), int(dy * eased))

        def pos_head(t):
            p = get_progress(t)
            eased = ease_in_out_cubic(p)
            # head смещен относительно tail на (-dx, -dy)
            return (int(-dx + dx * eased), int(-dy + dy * eased))

        tail_prev = tail_prev.set_position(pos_tail)
        head_next = head_next.set_position(pos_head)

        # Создаем композит перехода, где оба клипа рендерятся вместе
        transition_clip = mpe.CompositeVideoClip([tail_prev, head_next], size=resolution).set_duration(d)

        # Глобальный блюр применяется к композиту
        max_blur = transition.blur_strength
        blur_direction = "horizontal" if abs(dx) > abs(dy) else "vertical"

        def blur_func(t):
            p = get_progress(t)
            # Blur strength proportional to velocity (derivative of easing)
            # This ensures blur fades out exactly as movement stops
            if p < 0.5:
                # Derivative of 4*p^3 is 12*p^2. Normalized to peak at 1.0 (at p=0.5) -> 4*p^2
                velocity_factor = 4 * p * p
            else:
                # Derivative of 1 - (-2p + 2)^3 / 2 is 3*(-2p+2)^2. Normalized -> 4*(1-p)^2
                velocity_factor = 4 * (1 - p) ** 2
            
            return max_blur * velocity_factor

        transition_blurred = dynamic_motion_blur(transition_clip, blur_func, direction=blur_direction)

        # --- СБОРКА РЕЗУЛЬТАТА ---
        
        # new_prev - это часть предыдущего клипа ДО перехода
        # Если клип полностью ушел в переход, создаем пустой клип (техническая заглушка)
        new_prev = body_prev if body_prev else mpe.ColorClip(size=resolution, color=(0,0,0), duration=0)

        # new_next - это переход + остаток следующего клипа
        next_parts = [transition_blurred]
        if rest_next:
            next_parts.append(rest_next.set_start(d))
        
        new_next = mpe.CompositeVideoClip(next_parts, size=resolution).set_duration(next_clip.duration)

        return new_prev, new_next, d

    def _apply_motion_blur_intro(
            self,
            clip: mpe.VideoClip,
            transition: TransitionInstruction,
            resolution: Tuple[int, int]
    ) -> mpe.VideoClip:
        """Motion Blur в начале: появляется из размытия."""
        duration = min(transition.duration, clip.duration)
        if duration <= 0:
            return clip

        # Делим клип
        head = clip.subclip(0, duration)
        rest = clip.subclip(duration) if clip.duration > duration else None

        max_blur = transition.blur_strength
        # Blur убывает
        def blur_func(t):
            progress = t / duration
            return max_blur * ((1 - progress) ** 2)

        head = dynamic_motion_blur(head, blur_func, direction="horizontal")

        if transition.fade:
            head = head.fadein(duration)
        
        if transition.glow:
             # Simple glow: boost brightness
             # Factor 1.5 -> 1.0
             head = dynamic_brightness(head, lambda t: 1.0 + 0.5 * ((1 - t/duration)**2))

        clips = [head]
        if rest:
            clips.append(rest.set_start(duration))
        
        return mpe.CompositeVideoClip(clips, size=resolution).set_duration(clip.duration)

    def _apply_motion_blur_outro(
            self,
            clip: mpe.VideoClip,
            transition: TransitionInstruction,
            resolution: Tuple[int, int]
    ) -> mpe.VideoClip:
        """Motion Blur в конце: уходит в размытие."""
        duration = min(transition.duration, clip.duration)
        if duration <= 0:
            return clip

        total = clip.duration
        body = clip.subclip(0, total - duration) if total > duration else None
        tail = clip.subclip(max(total - duration, 0), total)

        max_blur = transition.blur_strength
        # Blur нарастает
        def blur_func(t):
            progress = t / duration
            return max_blur * (progress ** 2)

        tail = dynamic_motion_blur(tail, blur_func, direction="horizontal")

        if transition.fade:
            tail = tail.fadeout(duration)

        if transition.glow:
             # Simple glow: boost brightness
             # Factor 1.0 -> 1.5
             tail = dynamic_brightness(tail, lambda t: 1.0 + 0.5 * (t/duration)**2)

        clips = []
        if body:
            clips.append(body)
            tail = tail.set_start(body.duration)
        
        clips.append(tail)
        return mpe.CompositeVideoClip(clips, size=resolution).set_duration(clip.duration)

    def _apply_motion_blur_between(
            self,
            prev_clip: mpe.VideoClip,
            next_clip: mpe.VideoClip,
            transition: TransitionInstruction,
            resolution: Tuple[int, int]
    ) -> tuple[mpe.VideoClip, mpe.VideoClip, float]:
        """Motion Blur между клипами."""
        d = min(transition.duration, prev_clip.duration, next_clip.duration)
        if d <= 0:
            return prev_clip, next_clip, 0.0

        max_blur = transition.blur_strength

        # --- PREV CLIP ---
        total_prev = prev_clip.duration
        body_prev = prev_clip.subclip(0, total_prev - d) if total_prev > d else None
        tail_prev = prev_clip.subclip(max(total_prev - d, 0), total_prev)

        def blur_out(t):
            progress = min(t / d, 1.0)
            return max_blur * (progress ** 2)

        tail_prev = dynamic_motion_blur(tail_prev, blur_out, direction="horizontal")
        
        if transition.glow:
             tail_prev = dynamic_brightness(tail_prev, lambda t: 1.0 + 0.5 * (t/d)**2)

        prev_parts = []
        if body_prev:
            prev_parts.append(body_prev)
            tail_prev = tail_prev.set_start(body_prev.duration)
        prev_parts.append(tail_prev)
        new_prev = mpe.CompositeVideoClip(prev_parts, size=resolution).set_duration(prev_clip.duration)

        # --- NEXT CLIP ---
        head_next = next_clip.subclip(0, d)
        rest_next = next_clip.subclip(d) if next_clip.duration > d else None

        def blur_in(t):
            progress = min(t / d, 1.0)
            return max_blur * ((1 - progress) ** 2)

        head_next = dynamic_motion_blur(head_next, blur_in, direction="horizontal")

        if transition.fade:
            # Crossfade: next fades in over prev
            head_next = head_next.fadein(d)
        
        if transition.glow:
             head_next = dynamic_brightness(head_next, lambda t: 1.0 + 0.5 * ((1 - t/d)**2))

        next_parts = [head_next]
        if rest_next:
            next_parts.append(rest_next.set_start(d))
        new_next = mpe.CompositeVideoClip(next_parts, size=resolution).set_duration(next_clip.duration)

        # If not fading (crossfade), we want a sequential cut at the peak of the blur.
        # So overlap should be 0.
        overlap = d if transition.fade else 0.0

        return new_prev, new_next, overlap

    # --- MAIN RENDER LOGIC ---

    def render(self, request: RenderRequest) -> RenderResult:
        if request.mode == ProcessingMode.CONCAT_NORMALIZE:
            return self._render_concat_normalize(request)

        template = resolve_template(request.output.template)
        target_resolution = request.output.resolution.size if request.output.resolution else template.resolution
        logger.info("Using target resolution %s", target_resolution)
        fast_result = self._try_render_fast_path(request, target_resolution)
        if fast_result is not None:
            return fast_result
        return self._render_channels(request, target_resolution)

        clips = []
        final_clip = None
        video = None
        audio = None
        overlay: List[mpe.VideoClip] = []
        timeline = TimelineDetailModel(clips=[], attachments=[])
        try:
            # Подготовка всех клипов (ресайз, эффекты, хромакей)
            for clip_instruction in request.clips:
                clip = self._prepare_clip(clip_instruction, target_resolution, request.output.fps)
                clips.append(clip)

            if not clips:
                raise VideoEngineError("No clips provided in the request")

            # Сборка видео с переходами
            video, timeline_clips = self._concatenate_with_transitions(clips, request.clips, target_resolution)
            timeline = TimelineDetailModel(clips=timeline_clips, attachments=[])

            # Наложение текста и картинок
            overlay, attachment_timeline = self._build_overlay(
                video.duration,
                request,
                target_resolution,
                request.output.fps,
            )
            timeline.attachments = attachment_timeline

            # Финальный композит
            final_clip = mpe.CompositeVideoClip([video, *overlay], size=target_resolution)

            # Аудио
            if request.output.include_audio:
                audio = self._build_audio(request, final_clip.duration)
                if audio is not None:
                    final_clip = final_clip.set_audio(audio)
            else:
                final_clip = final_clip.without_audio()

            output_path = self._export(final_clip, request)
        except Exception as exc:
            logger.exception("Rendering failed: %s", exc)
            return RenderResult(status="error", duration=0.0, output=Path(""), message=str(exc))
        finally:
            self._close_clip_resources([final_clip, audio, video, *overlay, *clips])
            gc.collect()

        return RenderResult(status="ok", duration=final_clip.duration, output=output_path, timeline=timeline)

    def _render_channels(self, request: RenderRequest, target_resolution: tuple[int, int]) -> RenderResult:
        final_clip = None
        audio = None
        overlay: List[mpe.VideoClip] = []
        channel_layers: List[mpe.VideoClip] = []
        channel_resources: List[mpe.VideoClip] = []
        output_path = Path("")
        timeline = TimelineDetailModel(clips=[], attachments=[], inserts=[], channels=[])
        try:
            channels = self._collect_timeline_channels(request)
            if not channels:
                raise VideoEngineError("No clips provided in the request")

            lower_duration = 0.0
            for channel_id in sorted(channels):
                channel_clip, channel_timeline, resources = self._build_timeline_channel(
                    channel_id,
                    channels[channel_id],
                    target_resolution,
                    request.output.fps,
                    reference_duration=lower_duration,
                )
                channel_resources.extend(resources)
                if channel_clip is None:
                    continue

                channel_layers.append(channel_clip)
                lower_duration = max(lower_duration, float(channel_clip.duration or 0.0))
                timeline.channels.append(TimelineChannelModel(channel_id=channel_id, clips=channel_timeline))
                for timeline_item in channel_timeline:
                    if timeline_item.kind == "clip":
                        timeline.clips.append(timeline_item)
                    elif timeline_item.kind == "attachment":
                        timeline.attachments.append(timeline_item)
                    elif timeline_item.kind == "insert":
                        timeline.inserts.append(timeline_item)

            if not channel_layers:
                raise VideoEngineError("No clips provided in the request")

            timeline_duration = max((float(layer.duration or 0.0) for layer in channel_layers), default=0.0)
            text_image_overlay, zoom_overlay = self._build_global_overlays(
                timeline_duration,
                request,
                target_resolution,
                timeline,
            )
            overlay = [*text_image_overlay, *zoom_overlay]

            final_layers = self._order_final_layers(channel_layers, text_image_overlay, zoom_overlay)
            final_clip = mpe.CompositeVideoClip(final_layers, size=target_resolution).set_duration(timeline_duration)

            if request.output.include_audio:
                audio = self._build_audio(request, final_clip.duration)
                if audio is not None:
                    final_clip = final_clip.set_audio(audio)
            else:
                final_clip = final_clip.without_audio()

            output_path = self._export(final_clip, request)
        except Exception as exc:
            logger.exception("Rendering failed: %s", exc)
            return RenderResult(status="error", duration=0.0, output=Path(""), message=str(exc))
        finally:
            self._close_clip_resources([final_clip, audio, *overlay, *channel_layers, *channel_resources])
            gc.collect()

        return RenderResult(status="ok", duration=final_clip.duration, output=output_path, timeline=timeline)

    def _prepare_clip(
        self,
        instruction: ClipInstruction,
        target_resolution: tuple[int, int],
        fps: int,
        transition_padding: float = 0.0,
    ) -> mpe.VideoClip:
        source_path = self._resolve_media_path(instruction.source)
        if not source_path.exists():
            raise VideoEngineError(f"Clip source not found: {source_path}")

        clip = mpe.VideoFileClip(str(source_path))
        start = max(instruction.start, 0.0)
        end = instruction.end if instruction.end else clip.duration
        end = min(end, clip.duration)
        if end <= start:
            end = clip.duration

        playback_rate = max(float(instruction.playback_rate or 1.0), 1e-6)
        transition_padding = max(float(transition_padding or 0.0), 0.0)
        base_output_duration = max(end - start, 0.0) / playback_rate
        target_output_duration = base_output_duration + transition_padding

        raw_padding_needed = transition_padding * playback_rate
        extend_before = min(raw_padding_needed, start)
        raw_padding_remaining = max(raw_padding_needed - extend_before, 0.0)
        extend_after = min(raw_padding_remaining, max(clip.duration - end, 0.0))
        trim_start = max(start - extend_before, 0.0)
        trim_end = min(end + extend_after, clip.duration)

        clip = clip.subclip(trim_start, trim_end)

        if playback_rate != 1.0:
            clip = clip.fx(vfx.speedx, playback_rate)
        if transition_padding > 0.0 and clip.duration < target_output_duration - 1e-6:
            speed_factor = max(clip.duration / target_output_duration, 1e-6)
            clip = clip.fx(vfx.speedx, speed_factor)

        clip = self._apply_horizontal_mirror(clip, instruction)
        clip = self._apply_internal_zoom(clip, instruction)
        clip = self._apply_fit_mode(clip, instruction, target_resolution)
        clip = self._apply_adjustments(clip, instruction.adjustments)

        if instruction.volume != 1.0 and clip.audio is not None:
            clip = clip.volumex(instruction.volume)

        clip = clip.set_fps(fps)
        return clip

    @staticmethod
    def _apply_horizontal_mirror(clip: mpe.VideoClip, instruction: ClipInstruction) -> mpe.VideoClip:
        if not bool(getattr(instruction, "mirror_horizontal", False)):
            return clip
        return clip.fx(vfx.mirror_x).set_duration(clip.duration)

    @staticmethod
    def _apply_internal_zoom(clip: mpe.VideoClip, instruction: ClipInstruction) -> mpe.VideoClip:
        zoom_percent = max(float(getattr(instruction, "effective_internal_zoom", 0.0) or 0.0), 0.0)
        if zoom_percent <= 0.0:
            return clip

        source_w, source_h = clip.size
        zoom_factor = 1.0 + zoom_percent
        zoomed = clip.resize(zoom_factor)
        reframed = zoomed.crop(
            width=source_w,
            height=source_h,
            x_center=zoomed.w / 2,
            y_center=zoomed.h / 2,
        )
        return reframed.set_duration(clip.duration)

    @staticmethod
    def _build_center_zoom_filter(zoom_percent: float) -> Optional[str]:
        zoom_value = max(float(zoom_percent or 0.0), 0.0)
        if zoom_value <= 0.0:
            return None

        zoom_factor = 1.0 + zoom_value
        crop_w = f"trunc(iw/{zoom_factor:.6f}/2)*2"
        crop_h = f"trunc(ih/{zoom_factor:.6f}/2)*2"
        return f"crop={crop_w}:{crop_h}:(iw-ow)/2:(ih-oh)/2"

    @staticmethod
    def _color_to_ffmpeg(color) -> str:
        if not color:
            return "red@1"
        alpha = max(min(float(getattr(color, "a", 1.0)), 1.0), 0.0)
        return f"0x{int(color.r):02x}{int(color.g):02x}{int(color.b):02x}@{alpha:.3f}"

    def _build_zoom_border_filter(
        self,
        instruction: Optional[ZoomBorderInstruction],
        target_resolution: tuple[int, int],
    ) -> Optional[str]:
        if instruction is None:
            return None
        x, y, box_w, box_h = self._resolve_zoom_border_box(instruction, target_resolution)
        line_width = max(int(instruction.width), 1)
        color = self._color_to_ffmpeg(instruction.color)
        return f"drawbox=x={x}:y={y}:w={box_w}:h={box_h}:color={color}:t={line_width}"

    def _apply_fit_mode(self, clip: mpe.VideoClip, instruction: ClipInstruction, target_resolution: tuple[int, int]) -> mpe.VideoClip:
        target_w, target_h = target_resolution
        clip_w, clip_h = clip.size

        if instruction.fit_mode == FitMode.COVER:
            scale = max(target_w / clip_w, target_h / clip_h)
            resized = clip.resize(scale)
            fitted = resized.crop(width=target_w, height=target_h, x_center=resized.w / 2, y_center=resized.h / 2)
            if instruction.chroma_key.enabled:
                fitted = self._apply_chroma_key(fitted, instruction)
            return fitted

        # contain
        scale = min(target_w / clip_w, target_h / clip_h)
        resized = clip.resize(scale)

        if instruction.chroma_key.enabled:
            keyed = self._apply_chroma_key(resized, instruction)
            positioned = keyed.set_position("center")
            composite = mpe.CompositeVideoClip([positioned], size=target_resolution)
            return composite.set_duration(resized.duration)

        # Для contain нам нужно создать подложку (CompositeVideoClip), чтобы заполнить пустоты
        if instruction.background_mode == BackgroundMode.COLOR:
            background = mpe.ColorClip(size=target_resolution, color=instruction.background_color.as_tuple(), duration=resized.duration)
        else:
            # Blur background
            # Берем клип, ресайзим до cover, блюрим
            bg_scale = max(target_w / clip_w, target_h / clip_h)
            background = clip.resize(bg_scale)
            background = background.crop(width=target_w, height=target_h, x_center=background.w/2, y_center=background.h/2)
            background = background.fx(vfx.gaussian_blur, sigma=max(instruction.background_color.a * 25, 5))
            background = background.set_duration(resized.duration)

        positioned = resized.set_position("center")
        composite = mpe.CompositeVideoClip([background, positioned], size=target_resolution)
        return composite.set_duration(resized.duration)

    def _apply_adjustments(self, clip: mpe.VideoClip, adjustments: AdjustmentInstruction) -> mpe.VideoClip:
        if adjustments.brightness != 0:
            clip = clip.fx(vfx.colorx, 1 + adjustments.brightness)
        if adjustments.contrast != 0:
            clip = clip.fx(vfx.lum_contrast, contrast=adjustments.contrast * 100)
        if adjustments.saturation != 0:
            try:
                clip = clip.fx(vfx.saturation, 1 + adjustments.saturation)
            except Exception:
                pass
        if adjustments.hue != 0:
            try:
                clip = clip.fx(vfx.hue, adjustments.hue)
            except Exception:
                pass
        return clip

    @staticmethod
    def _build_chroma_alpha(
        frame: np.ndarray,
        key_color: tuple[int, int, int],
        similarity: float,
        blend: float,
        edge_blur: float,
    ) -> np.ndarray:
        if frame.ndim == 2:
            rgb = np.repeat(frame[:, :, None], 3, axis=2)
        else:
            rgb = frame[:, :, :3]

        max_value = (
            1.0
            if np.issubdtype(rgb.dtype, np.floating) and float(np.nanmax(rgb)) <= 1.0
            else 255.0
        )
        rgb_norm = rgb.astype(np.float32) / max_value
        key_norm = np.array(key_color, dtype=np.float32) / 255.0
        distance = np.linalg.norm(rgb_norm - key_norm, axis=2) / math.sqrt(3.0)

        similarity = max(float(similarity), 0.0)
        blend = max(float(blend), 0.0)
        if blend <= 1e-6:
            alpha = (distance > similarity).astype(np.float32)
        else:
            alpha = np.clip((distance - similarity) / blend, 0.0, 1.0).astype(np.float32)

        if edge_blur > 0:
            alpha_image = Image.fromarray(np.uint8(np.clip(alpha, 0.0, 1.0) * 255), mode="L")
            alpha = (
                np.asarray(alpha_image.filter(ImageFilter.GaussianBlur(radius=edge_blur))).astype(np.float32)
                / 255.0
            )
        return np.clip(alpha, 0.0, 1.0)

    @staticmethod
    def _apply_chroma_spill_reduction(
        frame: np.ndarray,
        alpha: np.ndarray,
        key_color: tuple[int, int, int],
        spill: float,
    ) -> np.ndarray:
        spill = max(0.0, min(float(spill), 1.0))
        if spill <= 0.0 or frame.ndim < 3:
            return frame

        rgb = frame[:, :, :3].astype(np.float32)
        key_channel = int(np.argmax(np.array(key_color, dtype=np.float32)))
        other_channels = [channel for channel in range(3) if channel != key_channel]
        other_max = np.max(rgb[:, :, other_channels], axis=2)
        excess = np.maximum(rgb[:, :, key_channel] - other_max, 0.0)
        proximity = (1.0 - np.clip(alpha, 0.0, 1.0)) * spill
        rgb[:, :, key_channel] = np.maximum(rgb[:, :, key_channel] - excess * proximity, 0.0)

        result = frame.copy()
        result[:, :, :3] = np.clip(rgb, 0.0, 255.0).astype(frame.dtype)
        return result

    def _apply_chroma_key(self, clip: mpe.VideoClip, instruction: ClipInstruction) -> mpe.VideoClip:
        key = instruction.chroma_key
        try:
            key_color = key.color.as_tuple()
            similarity = key.effective_similarity
            blend = key.effective_blend
            edge_blur = float(key.edge_blur or 0.0)
            base_mask = clip.mask
            source_clip = clip

            def make_mask_frame(t):
                frame = source_clip.get_frame(t)
                alpha = self._build_chroma_alpha(frame, key_color, similarity, blend, edge_blur)
                if base_mask is not None:
                    existing_alpha = base_mask.get_frame(t)
                    if existing_alpha.ndim == 3:
                        existing_alpha = existing_alpha[:, :, 0]
                    alpha = alpha * np.clip(existing_alpha.astype(np.float32), 0.0, 1.0)
                return alpha

            mask = mpe.VideoClip(make_frame=make_mask_frame, ismask=True).set_duration(clip.duration)
            keyed = clip.set_mask(mask)
            if key.spill > 0.0:

                def filter_frame(get_frame, t):
                    frame = get_frame(t)
                    alpha = mask.get_frame(t)
                    return self._apply_chroma_spill_reduction(frame, alpha, key_color, key.spill)

                keyed = keyed.fl(filter_frame)
            return keyed
        except Exception as exc:
            logger.warning("Chroma key failed for %s: %s", instruction.source, exc)
            return clip

    def _apply_intro_transition(
        self,
        clip: mpe.VideoClip,
        instruction: ClipInstruction,
        target_resolution: tuple[int, int],
        transparent_background: bool = False,
    ) -> mpe.VideoClip:
        if not instruction.transitions_before:
            return clip

        intro = instruction.transitions_before[0]
        if intro.type == TransitionType.CROSSFADE:
            return clip.crossfadein(min(intro.duration, clip.duration))
        if intro.type == TransitionType.FADE_BLACK:
            return clip.fadein(min(intro.duration, clip.duration))
        if intro.type == TransitionType.WHIP_PAN:
            return self._apply_whip_pan_intro(
                clip,
                intro,
                target_resolution,
                transparent_background=transparent_background,
            )
        if intro.type == TransitionType.MOTION_BLUR:
            return self._apply_motion_blur_intro(clip, intro, target_resolution)
        return clip

    def _apply_outro_transition(
        self,
        clip: mpe.VideoClip,
        instruction: ClipInstruction,
        target_resolution: tuple[int, int],
        transparent_background: bool = False,
    ) -> mpe.VideoClip:
        if not instruction.transitions_after:
            return clip

        outro = instruction.transitions_after[0]
        if outro.type == TransitionType.CROSSFADE:
            return clip.crossfadeout(min(outro.duration, clip.duration))
        if outro.type == TransitionType.FADE_BLACK:
            return clip.fadeout(min(outro.duration, clip.duration))
        if outro.type == TransitionType.WHIP_PAN:
            return self._apply_whip_pan_outro(
                clip,
                outro,
                target_resolution,
                transparent_background=transparent_background,
            )
        if outro.type == TransitionType.MOTION_BLUR:
            return self._apply_motion_blur_outro(clip, outro, target_resolution)
        return clip

    @staticmethod
    def _resolve_attachment_start(instruction: InsertInstruction, timeline_duration: float, clip_duration: float) -> float:
        if instruction.placement == InsertPlacement.START:
            return 0.0
        if instruction.placement == InsertPlacement.END:
            return timeline_duration - clip_duration
        return max(instruction.at or 0.0, 0.0)

    @staticmethod
    def _trim_attachment_to_timeline(
        clip: mpe.VideoClip,
        timeline_start: float,
        timeline_duration: float,
    ) -> tuple[Optional[mpe.VideoClip], float]:
        trim_start = 0.0
        if timeline_start < 0.0:
            trim_start = -timeline_start
            timeline_start = 0.0

        remaining_duration = max(timeline_duration - timeline_start, 0.0)
        if remaining_duration <= 0.0 or trim_start >= clip.duration:
            return None, timeline_start

        trim_end = min(trim_start + remaining_duration, clip.duration)
        if trim_end <= trim_start:
            return None, timeline_start

        if trim_start > 0.0 or trim_end < clip.duration:
            clip = clip.subclip(trim_start, trim_end)
        return clip, timeline_start

    @staticmethod
    def _is_auto_placed_attachment(instruction: InsertInstruction) -> bool:
        if instruction.placement != InsertPlacement.TIME:
            return True
        return not VideoEngine._has_explicit_at(instruction)

    def _collect_timeline_channels(self, request: RenderRequest) -> dict[int, list[tuple[str, int, ClipInstruction, bool]]]:
        channels: dict[int, list[tuple[str, int, ClipInstruction, bool]]] = {}

        def add(channel_id: int, kind: str, index: int, instruction: ClipInstruction, trim_to_reference: bool) -> None:
            channels.setdefault(int(channel_id), []).append((kind, index, instruction, trim_to_reference))

        for index, instruction in enumerate(request.clips):
            add(1, "clip", index, instruction, False)
        for index, instruction in enumerate(request.attachments):
            add(10, "attachment", index, instruction, True)
        for index, instruction in enumerate(request.inserts):
            add(11, "insert", index, instruction, True)

        if request.timeline:
            for channel in request.timeline.channels:
                for index, instruction in enumerate(channel.clips):
                    add(channel.channel_id, "timeline", index, instruction, False)

        return {channel_id: entries for channel_id, entries in channels.items() if entries}

    @staticmethod
    def _resolve_channel_base_start(
        instruction: ClipInstruction,
        cursor: float,
        clip_duration: float,
        reference_duration: float,
    ) -> float:
        placement = getattr(instruction, "placement", InsertPlacement.TIME)
        if placement == InsertPlacement.START:
            return 0.0
        if placement == InsertPlacement.END:
            return max(reference_duration - clip_duration, 0.0)
        if VideoEngine._has_explicit_at(instruction):
            return max(float(instruction.at or 0.0), 0.0)
        return max(cursor, 0.0)

    def _apply_between_transition(
        self,
        scheduled: list[tuple[mpe.VideoClip, float]],
        clip: mpe.VideoClip,
        transition: Optional[TransitionInstruction],
        base_start: float,
        target_resolution: tuple[int, int],
        transition_overlap: float = 0.0,
    ) -> tuple[mpe.VideoClip, float]:
        if not transition:
            return clip, base_start

        if transition.type == TransitionType.CROSSFADE:
            duration = min(transition_overlap or transition.duration, clip.duration / 2)
            clip = clip.crossfadein(duration)
            start = max(base_start - duration, 0.0)
            return clip, start

        if transition.type == TransitionType.FADE_BLACK:
            duration = min(transition.duration, clip.duration)
            if scheduled:
                prev_clip, prev_start = scheduled[-1]
                scheduled[-1] = (prev_clip.fadeout(duration), prev_start)
            return clip.fadein(duration), base_start

        if transition.type == TransitionType.WHIP_PAN and scheduled:
            prev_clip, prev_start = scheduled[-1]
            new_prev, new_clip, overlap = self._apply_whip_pan_between(prev_clip, clip, transition, target_resolution)
            scheduled[-1] = (new_prev, prev_start)
            start = max(base_start - overlap, 0.0)
            return new_clip, start

        if transition.type == TransitionType.MOTION_BLUR and scheduled:
            prev_clip, prev_start = scheduled[-1]
            new_prev, new_clip, overlap = self._apply_motion_blur_between(prev_clip, clip, transition, target_resolution)
            scheduled[-1] = (new_prev, prev_start)
            start = max(base_start - overlap, 0.0)
            return new_clip, start

        return clip, base_start

    @staticmethod
    def _trim_scheduled_clip_to_reference(
        clip: mpe.VideoClip,
        start: float,
        reference_duration: float,
    ) -> Optional[mpe.VideoClip]:
        if reference_duration <= 0.0:
            return None
        if start >= reference_duration:
            return None
        visible_duration = min(float(clip.duration or 0.0), reference_duration - start)
        if visible_duration <= 0.0:
            return None
        if visible_duration < float(clip.duration or 0.0):
            return clip.subclip(0, visible_duration)
        return clip

    def _get_instruction_output_duration(self, instruction: ClipInstruction) -> float:
        source_path = self._resolve_media_path(instruction.source)
        source_duration = self._probe_duration_seconds(source_path)
        start = max(float(instruction.start or 0.0), 0.0)
        end = instruction.end if instruction.end else source_duration
        end = min(float(end), source_duration)
        if end <= start:
            end = source_duration
        raw_duration = max(end - start, 0.0)
        return raw_duration / max(float(instruction.playback_rate or 1.0), 1e-6)

    def _get_transition_padding(
        self,
        transition: Optional[TransitionInstruction],
        previous_clip: Optional[mpe.VideoClip],
        instruction: ClipInstruction,
    ) -> float:
        if not transition or previous_clip is None:
            return 0.0
        requested = max(float(transition.duration or 0.0), 0.0)
        if requested <= 0.0:
            return 0.0

        previous_duration = max(float(previous_clip.duration or 0.0), 0.0)
        if transition.type == TransitionType.CROSSFADE:
            # crossfadein() is capped to half of the resulting clip duration, so
            # base_duration is the largest overlap that remains stable after padding.
            base_duration = self._get_instruction_output_duration(instruction)
            return max(min(requested, previous_duration, base_duration), 0.0)
        if transition.type == TransitionType.WHIP_PAN:
            return max(min(requested, previous_duration), 0.0)
        if transition.type == TransitionType.MOTION_BLUR and transition.fade:
            return max(min(requested, previous_duration), 0.0)
        return 0.0

    def _resolve_instruction_timeline_start_for_adjacency(
        self,
        instruction: ClipInstruction,
        cursor: float,
        output_duration: float,
        reference_duration: float,
    ) -> float:
        placement = getattr(instruction, "placement", InsertPlacement.TIME)
        if placement == InsertPlacement.START:
            return 0.0
        if placement == InsertPlacement.END:
            return max(reference_duration - max(output_duration, 0.0), 0.0)
        if self._has_explicit_at(instruction):
            return max(float(instruction.at or 0.0), 0.0)
        return max(cursor, 0.0)

    def _suppress_adjacent_insert_intro_transition(
        self,
        entries: list[tuple[str, int, ClipInstruction, bool]],
        entry_index: int,
        cursor: float,
        reference_duration: float,
    ) -> None:
        kind, _source_index, instruction, _trim_to_reference = entries[entry_index]
        if kind not in {"attachment", "insert"} or not instruction.transitions_before:
            return

        current_duration = self._get_instruction_output_duration(instruction)
        current_start = self._resolve_instruction_timeline_start_for_adjacency(
            instruction,
            cursor,
            current_duration,
            reference_duration,
        )
        if current_start <= ADJACENT_INSERT_TOLERANCE_SEC:
            instruction.transitions_before = []
            return
        if entry_index <= 0:
            return

        previous_kind, _previous_source_index, previous_instruction, _previous_trim = entries[entry_index - 1]
        if previous_kind != kind:
            return

        previous_duration = self._get_instruction_output_duration(previous_instruction)
        previous_start = self._resolve_instruction_timeline_start_for_adjacency(
            previous_instruction,
            max(cursor - previous_duration, 0.0),
            previous_duration,
            reference_duration,
        )
        previous_end = previous_start + previous_duration

        if abs(current_start - previous_end) <= ADJACENT_INSERT_TOLERANCE_SEC:
            instruction.transitions_before = []

    def _build_timeline_channel(
        self,
        channel_id: int,
        entries: list[tuple[str, int, ClipInstruction, bool]],
        target_resolution: tuple[int, int],
        fps: int,
        reference_duration: float,
    ) -> tuple[Optional[mpe.VideoClip], list[TimelineClipModel], list[mpe.VideoClip]]:
        scheduled: list[tuple[mpe.VideoClip, float]] = []
        scheduled_instructions: list[ClipInstruction] = []
        timeline: list[TimelineClipModel] = []
        resources: list[mpe.VideoClip] = []
        cursor = 0.0

        for entry_index, (kind, source_index, instruction, trim_to_reference) in enumerate(entries):
            self._suppress_adjacent_insert_intro_transition(
                entries,
                entry_index,
                cursor,
                reference_duration,
            )
            transition = None
            if entry_index > 0:
                prev_instruction = entries[entry_index - 1][2]
                if prev_instruction.transitions_after:
                    transition = prev_instruction.transitions_after[0]
            previous_clip = scheduled[-1][0] if scheduled else None
            transition_padding = self._get_transition_padding(transition, previous_clip, instruction)

            clip = self._prepare_clip(instruction, target_resolution, fps, transition_padding=transition_padding)
            resources.append(clip)

            if instruction.transitions_before:
                clip = self._apply_intro_transition(
                    clip,
                    instruction,
                    target_resolution,
                    transparent_background=channel_id > 1,
                )

            base_start = self._resolve_channel_base_start(
                instruction,
                cursor,
                max(float(clip.duration or 0.0) - transition_padding, 0.0),
                reference_duration,
            )

            clip, start = self._apply_between_transition(
                scheduled,
                clip,
                transition,
                base_start,
                target_resolution,
                transition_overlap=transition_padding,
            )

            if trim_to_reference:
                trimmed = self._trim_scheduled_clip_to_reference(clip, start, reference_duration)
                if trimmed is None:
                    continue
                if trimmed is not clip:
                    clip = trimmed
                    resources.append(clip)

            end = start + float(clip.duration or 0.0)
            scheduled.append((clip, start))
            scheduled_instructions.append(instruction)
            cursor = max(cursor, end)
            timeline.append(
                TimelineClipModel(
                    index=source_index,
                    source=instruction.source,
                    source_label=instruction.source_label,
                    source_type=instruction.source_type,
                    source_resolution=instruction.source_resolution,
                    quality_label=instruction.quality_label,
                    start=round(max(start, 0.0), 1),
                    end=round(max(end, 0.0), 1),
                    auto_placed=self._is_auto_placed_instruction(instruction),
                    channel_id=channel_id,
                    kind=kind,
                )
            )

        if scheduled:
            last_instruction = scheduled_instructions[-1]
            if last_instruction.transitions_after:
                outro = last_instruction.transitions_after[0]
                last_clip, last_start = scheduled[-1]
                last_clip = self._apply_outro_transition(
                    last_clip,
                    last_instruction,
                    target_resolution,
                    transparent_background=channel_id > 1,
                )
                scheduled[-1] = (last_clip, last_start)
                if timeline:
                    timeline[-1].end = round(max(last_start + float(last_clip.duration or 0.0), 0.0), 1)
                cursor = max(cursor, last_start + float(last_clip.duration or 0.0))

        if not scheduled:
            return None, [], resources

        layered = [clip.set_start(start) for clip, start in scheduled]
        channel_clip = mpe.CompositeVideoClip(layered, size=target_resolution).set_duration(cursor)
        setattr(channel_clip, "_ffmpeg_engine_channel_id", channel_id)
        return channel_clip, timeline, resources

    def _build_attachment_clip(
        self,
        index: int,
        instruction: InsertInstruction,
        timeline_duration: float,
        target_resolution: tuple[int, int],
        fps: int,
    ) -> tuple[Optional[mpe.VideoClip], Optional[TimelineClipModel]]:
        clip = self._prepare_clip(instruction, target_resolution, fps)
        timeline_start = self._resolve_attachment_start(instruction, timeline_duration, clip.duration)
        clip, timeline_start = self._trim_attachment_to_timeline(clip, timeline_start, timeline_duration)
        if clip is None:
            return None, None

        clip = self._apply_intro_transition(
            clip,
            instruction,
            target_resolution,
            transparent_background=True,
        )
        clip = self._apply_outro_transition(
            clip,
            instruction,
            target_resolution,
            transparent_background=True,
        )

        timeline_end = min(timeline_start + clip.duration, timeline_duration)
        clip = clip.set_start(timeline_start).set_end(timeline_end)
        timeline_item = TimelineClipModel(
            index=index,
            source=instruction.source,
            source_label=instruction.source_label,
            source_type=instruction.source_type,
            source_resolution=instruction.source_resolution,
            quality_label=instruction.quality_label,
            start=round(max(timeline_start, 0.0), 1),
            end=round(max(timeline_end, 0.0), 1),
            auto_placed=self._is_auto_placed_attachment(instruction),
        )
        return clip, timeline_item

    def _concatenate_with_transitions(
            self,
            clips: List[mpe.VideoClip],
            clip_instructions: List[ClipInstruction],
            target_resolution: tuple[int, int],
    ) -> tuple[mpe.VideoClip, List[TimelineClipModel]]:
        """
        Собирает клипы в один таймлайн, обрабатывая наложения (transitions).
        """
        scheduled: List[tuple[mpe.VideoClip, float]] = []
        cursor = 0.0

        for index, clip in enumerate(clips):
            instr = clip_instructions[index]

            # 1. Intro Transitions (применяются к самому клипу)
            intro: Optional[TransitionInstruction] = None
            if instr.transitions_before:
                intro = instr.transitions_before[0]

            if intro:
                if intro.type == TransitionType.FADE_BLACK:
                    d = min(intro.duration, clip.duration)
                    clip = clip.fadein(d)
                elif intro.type == TransitionType.WHIP_PAN:
                    clip = self._apply_whip_pan_intro(clip, intro, target_resolution)
                elif intro.type == TransitionType.MOTION_BLUR:
                    clip = self._apply_motion_blur_intro(clip, intro, target_resolution)

            # 2. Transition from Previous Clip (переход между клипами)
            transition: Optional[TransitionInstruction] = None
            if index > 0:
                prev_instr = clip_instructions[index - 1]
                if prev_instr.transitions_after:
                    transition = prev_instr.transitions_after[0]

            # Обработка переходов
            if transition and transition.type == TransitionType.CROSSFADE:
                duration = min(transition.duration, clip.duration / 2)
                clip = clip.crossfadein(duration)
                # Сдвигаем курсор назад, чтобы было наложение
                start = max(cursor - duration, 0.0)
                cursor = start + clip.duration

            elif transition and transition.type == TransitionType.FADE_BLACK:
                duration = min(transition.duration, clip.duration)
                # Предыдущий уходит в черное
                if scheduled:
                    prev_clip, prev_start = scheduled[-1]
                    scheduled[-1] = (prev_clip.fadeout(duration), prev_start)
                # Текущий выходит из черного
                clip = clip.fadein(duration)
                # Здесь нет наложения по времени (или минимальное), они стыкуются
                start = cursor
                cursor += clip.duration

            elif transition and transition.type == TransitionType.WHIP_PAN:
                if scheduled:
                    prev_clip, prev_start = scheduled[-1]
                    # Вызываем специальную логику, которая вернет обновленные клипы и длительность нахлеста
                    new_prev, new_clip, overlap = self._apply_whip_pan_between(
                        prev_clip, clip, transition, target_resolution
                    )

                    # Обновляем предыдущий клип в расписании
                    scheduled[-1] = (new_prev, prev_start)

                    # Текущий клип теперь new_clip
                    clip = new_clip

                    # Рассчитываем старт с учетом нахлеста
                    start = max(cursor - overlap, 0.0)
                    cursor = start + clip.duration

            elif transition and transition.type == TransitionType.MOTION_BLUR:
                if scheduled:
                    prev_clip, prev_start = scheduled[-1]
                    new_prev, new_clip, overlap = self._apply_motion_blur_between(
                        prev_clip, clip, transition, target_resolution
                    )
                    scheduled[-1] = (new_prev, prev_start)
                    clip = new_clip
                    start = max(cursor - overlap, 0.0)
                    cursor = start + clip.duration
                else:
                    start = cursor
                    cursor += clip.duration
            else:
                # Нет перехода
                start = cursor
                cursor += clip.duration

            scheduled.append((clip, start))

        # 3. Outro Transitions (для самого последнего клипа)
        if scheduled and clip_instructions:
            last_instr = clip_instructions[-1]
            if last_instr.transitions_after:
                outro = last_instr.transitions_after[0]
                last_clip, last_start = scheduled[-1]

                if outro.type == TransitionType.FADE_BLACK:
                    d = min(outro.duration, last_clip.duration)
                    last_clip = last_clip.fadeout(d)
                    scheduled[-1] = (last_clip, last_start)

                elif outro.type == TransitionType.WHIP_PAN:
                    last_clip = self._apply_whip_pan_outro(last_clip, outro, target_resolution)
                    scheduled[-1] = (last_clip, last_start)
                
                elif outro.type == TransitionType.MOTION_BLUR:
                    last_clip = self._apply_motion_blur_outro(last_clip, outro, target_resolution)
                    scheduled[-1] = (last_clip, last_start)

        # Собираем финальный композит
        # set_start устанавливает время начала клипа на глобальном таймлайне
        layered = [c.set_start(s) for c, s in scheduled]

        # size=target_resolution важен, чтобы canvas не скакал
        base = mpe.CompositeVideoClip(layered, size=target_resolution)
        base = base.set_duration(cursor)
        timeline = []
        for index, ((scheduled_clip, scheduled_start), instruction) in enumerate(zip(scheduled, clip_instructions)):
            clip_end = scheduled_start + scheduled_clip.duration
            timeline.append(
                TimelineClipModel(
                    index=index,
                    source=instruction.source,
                    source_label=instruction.source_label,
                    source_type=instruction.source_type,
                    source_resolution=instruction.source_resolution,
                    quality_label=instruction.quality_label,
                    start=round(max(scheduled_start, 0.0), 1),
                    end=round(max(clip_end, 0.0), 1),
                    auto_placed=self._is_auto_placed_instruction(instruction),
                )
            )
        return base, timeline

    def _build_overlay(
            self,
            duration: float,
            request: RenderRequest,
            target_resolution: tuple[int, int],
            fps: int,
    ) -> tuple[List[mpe.VideoClip], List[TimelineClipModel]]:
        overlays: List[mpe.VideoClip] = []
        attachment_timeline: List[TimelineClipModel] = []
        for text in request.texts:
            clip = self._build_text_clip(text, duration, target_resolution)
            if clip:
                overlays.append(clip)
        for image in request.images:
            clip = self._build_image_clip(image, duration, target_resolution)
            if clip:
                overlays.append(clip)
        for index, attachment in enumerate(request.attachments):
            clip, timeline_item = self._build_attachment_clip(index, attachment, duration, target_resolution, fps)
            if clip:
                overlays.append(clip)
            if timeline_item:
                attachment_timeline.append(timeline_item)
        zoom_border = self._build_zoom_border_clip(request.zoom_border, duration, target_resolution)
        if zoom_border:
            overlays.append(zoom_border)
        return overlays, attachment_timeline

    def _build_global_overlays(
        self,
        duration: float,
        request: RenderRequest,
        target_resolution: tuple[int, int],
        timeline: Optional[TimelineDetailModel] = None,
    ) -> tuple[list[mpe.VideoClip], list[mpe.VideoClip]]:
        text_image_overlays: list[mpe.VideoClip] = []
        zoom_overlays: list[mpe.VideoClip] = []
        for text in request.texts:
            clip = self._build_text_clip(text, duration, target_resolution)
            if clip:
                text_image_overlays.append(clip)
        for image in request.images:
            clip = self._build_image_clip(image, duration, target_resolution)
            if clip:
                text_image_overlays.append(clip)
        show_source_clips = self._build_show_source_clips(request.show_source, timeline, duration, target_resolution)
        zoom_overlays.extend(show_source_clips)
        zoom_border = self._build_zoom_border_clip(request.zoom_border, duration, target_resolution)
        if zoom_border:
            zoom_overlays.append(zoom_border)
        return text_image_overlays, zoom_overlays

    @staticmethod
    def _order_final_layers(
        channel_layers: list[mpe.VideoClip],
        text_image_overlays: list[mpe.VideoClip],
        zoom_overlays: list[mpe.VideoClip],
    ) -> list[mpe.VideoClip]:
        below_text = []
        above_text = []
        for layer in channel_layers:
            channel_id = int(getattr(layer, "_ffmpeg_engine_channel_id", 1))
            if channel_id < 10:
                below_text.append(layer)
            else:
                above_text.append(layer)
        return [*below_text, *text_image_overlays, *above_text, *zoom_overlays]

    @staticmethod
    def _collect_show_source_timeline(timeline: Optional[TimelineDetailModel]) -> list[TimelineClipModel]:
        if timeline is None:
            return []
        for channel in timeline.channels:
            if channel.channel_id == 1 and channel.clips:
                return list(channel.clips)
        return list(timeline.clips)

    def _build_show_source_clips(
        self,
        instruction: Optional[ShowSourceInstruction],
        timeline: Optional[TimelineDetailModel],
        duration: float,
        target_resolution: tuple[int, int],
    ) -> list[mpe.VideoClip]:
        if instruction is None or duration <= 0:
            return []

        overlays: list[mpe.VideoClip] = []
        margin_x = 14
        margin_y = 8
        font_probe = TextInstruction(content="source", font=instruction.font, font_size=instruction.size)
        font = self._resolve_pillow_font(font_probe)
        stroke_width = max(1, int(round(instruction.size * 0.08)))
        max_width = max(target_resolution[0] - margin_x * 2, 1)
        fill = self._color_to_rgba(instruction.color, (255, 0, 0, 255))
        stroke_fill = (0, 0, 0, 220)
        probe = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        probe_draw = ImageDraw.Draw(probe)

        def fit_label(raw_label: str) -> tuple[str, tuple[int, int, int, int]]:
            label = raw_label
            bbox = probe_draw.textbbox((0, 0), label or " ", font=font, stroke_width=stroke_width)
            if bbox[2] - bbox[0] <= max_width:
                return label, bbox

            prefix = "Source: "
            basename = raw_label[len(prefix):] if raw_label.startswith(prefix) else raw_label
            suffix = Path(basename).suffix
            stem = Path(basename).stem or basename
            for keep in range(len(stem), 3, -1):
                candidate = f"{prefix}{stem[:keep]}...{suffix}"
                bbox = probe_draw.textbbox((0, 0), candidate, font=font, stroke_width=stroke_width)
                if bbox[2] - bbox[0] <= max_width:
                    return candidate, bbox
            fallback = f"{prefix}...{suffix}" if suffix else f"{prefix}..."
            bbox = probe_draw.textbbox((0, 0), fallback, font=font, stroke_width=stroke_width)
            return fallback, bbox

        for item in self._collect_show_source_timeline(timeline):
            start = max(float(item.start), 0.0)
            end = min(max(float(item.end), start), duration)
            if end <= start:
                continue
            source_label = str(item.source_label or "").strip() or Path(item.source).name
            label, bbox = fit_label(f"Source: {source_label}")
            pad = stroke_width + 2
            label_w = max(bbox[2] - bbox[0], 1)
            label_h = max(bbox[3] - bbox[1], 1)
            image = Image.new("RGBA", (label_w + pad * 2, label_h + pad * 2), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            draw.text(
                (pad - bbox[0], pad - bbox[1]),
                label,
                font=font,
                fill=fill,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill,
            )
            frame = np.array(image)
            clip = (
                mpe.ImageClip(frame, transparent=True)
                .set_duration(max(end - start, 0.001))
                .set_start(start)
                .set_end(end)
                .set_position((margin_x, margin_y))
            )
            overlays.append(clip)
        return overlays

    @staticmethod
    def _color_to_rgba(color, default: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        if not color:
            return default
        alpha = int(max(min(getattr(color, "a", 1.0), 1.0), 0.0) * 255)
        return (int(color.r), int(color.g), int(color.b), alpha)

    @staticmethod
    def _resolve_zoom_border_box(
        instruction: ZoomBorderInstruction,
        target_resolution: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        target_w, target_h = target_resolution
        zoom = max(0.0, min(float(instruction.zoom or 0.0), 0.95))
        zoom_factor = 1.0 + zoom
        box_w = max(int(round(target_w / zoom_factor)), int(instruction.width))
        box_h = max(int(round(target_h / zoom_factor)), int(instruction.width))
        box_w = min(box_w, target_w)
        box_h = min(box_h, target_h)
        x = max((target_w - box_w) // 2, 0)
        y = max((target_h - box_h) // 2, 0)
        return x, y, box_w, box_h

    def _build_zoom_border_clip(
        self,
        instruction: Optional[ZoomBorderInstruction],
        duration: float,
        target_resolution: tuple[int, int],
    ) -> Optional[mpe.VideoClip]:
        if instruction is None or duration <= 0:
            return None

        target_w, target_h = target_resolution
        x, y, box_w, box_h = self._resolve_zoom_border_box(instruction, target_resolution)
        line_width = max(int(instruction.width), 1)
        image = Image.new("RGBA", (target_w, target_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        color = self._color_to_rgba(instruction.color, (255, 0, 0, 255))
        half = line_width // 2
        x0 = min(max(x + half, 0), target_w - 1)
        y0 = min(max(y + half, 0), target_h - 1)
        x1 = min(max(x + box_w - 1 - half, 0), target_w - 1)
        y1 = min(max(y + box_h - 1 - half, 0), target_h - 1)
        if x1 <= x0 or y1 <= y0:
            return None

        draw.rectangle((x0, y0, x1, y1), outline=color, width=line_width)
        frame = np.array(image)
        return mpe.ImageClip(frame).set_duration(duration).set_position((0, 0))

    def _resolve_pillow_font(self, instruction: TextInstruction) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        font_candidates: List[object] = []
        bundled_fonts_dir = Path(__file__).resolve().parent / "fonts"
        comic_regular = bundled_fonts_dir / "ComicNeue-Regular.ttf"
        comic_bold = bundled_fonts_dir / "ComicNeue-Bold.ttf"
        comic_ms_regular = bundled_fonts_dir / "ComicSansMS-Regular.ttf"
        comic_ms_bold = bundled_fonts_dir / "ComicSansMS-Bold.ttf"

        if getattr(instruction, "font_path", None):
            font_candidates.append(str(instruction.font_path))

        font_aliases = {
            "DejaVu-Sans": "DejaVuSans.ttf",
            "DejaVu Sans": "DejaVuSans.ttf",
            "DejaVuSans": "DejaVuSans.ttf",
            "DejaVu Sans Bold": "DejaVuSans-Bold.ttf",
            "DejaVu-Sans-Bold": "DejaVuSans-Bold.ttf",
            "DejaVu Sans Mono": "DejaVuSansMono.ttf",
            "DejaVu-Sans-Mono": "DejaVuSansMono.ttf",
            "DejaVu Sans Mono Bold": "DejaVuSansMono-Bold.ttf",
            "DejaVu-Sans-Mono-Bold": "DejaVuSansMono-Bold.ttf",
            "DejaVu Serif": "DejaVuSerif.ttf",
            "DejaVu-Serif": "DejaVuSerif.ttf",
            "DejaVu Serif Bold": "DejaVuSerif-Bold.ttf",
            "DejaVu-Serif-Bold": "DejaVuSerif-Bold.ttf",
            "Comic Sans": str(comic_ms_regular),
            "Comic Sans MS": str(comic_ms_bold),
            "ComicSansMS": str(comic_ms_bold),
            "Comic Neue": str(comic_regular),
            "Comic Neue Bold": str(comic_bold),
        }
        font_name = (instruction.font or "").strip()
        if font_name:
            font_candidates.append(font_name)
            system_font = _build_system_font_index().get(_normalize_font_lookup_name(font_name))
            if system_font:
                font_candidates.append(system_font)
            alias = font_aliases.get(font_name)
            if alias:
                font_candidates.append(alias)
            lowered = font_name.lower()
            if "comic sans" in lowered or lowered == "comicsans":
                font_candidates.extend(
                    [
                        str(comic_ms_bold),
                        str(comic_ms_regular),
                        str(comic_bold),
                        str(comic_regular),
                    ]
                )

        for candidate in font_candidates:
            if not candidate:
                continue
            try:
                if isinstance(candidate, tuple):
                    return ImageFont.truetype(candidate[0], instruction.font_size, index=candidate[1])
                return ImageFont.truetype(str(candidate), instruction.font_size)
            except Exception as exc:
                logger.debug("Unable to load font candidate %r: %s", candidate, exc)
                continue

        try:
            return ImageFont.truetype("DejaVuSans.ttf", instruction.font_size)
        except Exception:
            return ImageFont.load_default()

    @staticmethod
    def _measure_text(draw: ImageDraw.ImageDraw, text: str, font, stroke_width: int) -> tuple[int, int]:
        bbox = draw.textbbox((0, 0), text or " ", font=font, stroke_width=stroke_width)
        return max(bbox[2] - bbox[0], 1), max(bbox[3] - bbox[1], 1)

    @staticmethod
    def _measure_multiline_text(
            draw: ImageDraw.ImageDraw,
            text: str,
            font,
            stroke_width: int,
            spacing: int,
    ) -> tuple[int, int, tuple[int, int, int, int]]:
        raw_bbox = draw.multiline_textbbox(
            (0, 0),
            text or " ",
            font=font,
            align="center",
            spacing=spacing,
            stroke_width=stroke_width,
        )
        bbox = (
            int(math.floor(raw_bbox[0])),
            int(math.floor(raw_bbox[1])),
            int(math.ceil(raw_bbox[2])),
            int(math.ceil(raw_bbox[3])),
        )
        width = max(bbox[2] - bbox[0], 1)
        height = max(bbox[3] - bbox[1], 1)
        return width, height, bbox

    def _wrap_text_lines(self, text: str, font, max_width: int, stroke_width: int) -> List[str]:
        probe = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        draw = ImageDraw.Draw(probe)
        if max_width <= 0:
            return [text]

        def split_long_token(token: str) -> List[str]:
            parts: List[str] = []
            current = ""
            for ch in token:
                candidate = f"{current}{ch}"
                width, _ = self._measure_text(draw, candidate, font, stroke_width)
                if width <= max_width or not current:
                    current = candidate
                else:
                    parts.append(current)
                    current = ch
            if current:
                parts.append(current)
            return parts or [token]

        wrapped: List[str] = []
        paragraphs = text.splitlines() if text else [""]
        for paragraph in paragraphs:
            words = paragraph.split(" ")
            if not words:
                wrapped.append("")
                continue
            current_line = ""
            for word in words:
                token = word if current_line == "" else f"{current_line} {word}"
                width, _ = self._measure_text(draw, token, font, stroke_width)
                if width <= max_width:
                    current_line = token
                    continue
                if current_line:
                    wrapped.append(current_line)
                token_parts = split_long_token(word)
                current_line = token_parts.pop() if token_parts else ""
                wrapped.extend(token_parts)
            wrapped.append(current_line)
        return wrapped or [text]

    def _build_text_clip_with_pillow(
        self, instruction: TextInstruction, target_resolution: tuple[int, int]
    ) -> Optional[mpe.VideoClip]:
        font = self._resolve_pillow_font(instruction)
        stroke_width = max(int(instruction.stroke_width), 0)
        max_text_width = instruction.max_width if instruction.max_width else int(target_resolution[0] * 0.9)
        max_text_width = max(1, min(max_text_width, int(target_resolution[0] * 0.95)))

        lines = self._wrap_text_lines(instruction.content, font, max_text_width, stroke_width)
        probe = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        probe_draw = ImageDraw.Draw(probe)
        line_spacing = max(6, int(instruction.font_size * 0.22))
        text_block = "\n".join(lines if lines else [" "])
        text_w, text_h, text_bbox = self._measure_multiline_text(
            probe_draw,
            text_block,
            font,
            stroke_width,
            line_spacing,
        )

        scale_from = getattr(instruction.animation, "scale_from", 1.0) if instruction.animation else 1.0
        scale_to = getattr(instruction.animation, "scale_to", 1.0) if instruction.animation else 1.0
        scale_guard = max(scale_from, scale_to, 1.0)
        scale_margin = max(0, int(max(text_w, text_h) * (scale_guard - 1.0) * 0.55))

        effect_margin = 0
        if instruction.glow:
            effect_margin = max(effect_margin, int(instruction.font_size * 0.22))
        if instruction.shadow:
            effect_margin = max(effect_margin, int(instruction.font_size * 0.14) + 6)

        pad_x = max(16, int(instruction.font_size * 0.32) + stroke_width * 2 + scale_margin + effect_margin)
        pad_y = max(14, int(instruction.font_size * 0.34) + stroke_width * 2 + scale_margin + effect_margin)
        img_w = max(1, text_w + pad_x * 2)
        img_h = max(1, text_h + pad_y * 2)

        image = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
        text_color = self._color_to_rgba(instruction.color, (255, 255, 255, 255))
        stroke_color = self._color_to_rgba(instruction.stroke_color, (0, 0, 0, 255))
        render_stroke_width = stroke_width if stroke_color[3] > 0 else 0
        text_origin = (
            int((img_w - text_w) / 2 - text_bbox[0]),
            int((img_h - text_h) / 2 - text_bbox[1]),
        )

        if instruction.shadow:
            shadow_layer = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
            shadow_draw = ImageDraw.Draw(shadow_layer)
            shadow_offset = max(2, int(instruction.font_size * 0.06))
            shadow_alpha = max(90, int(text_color[3] * 0.65))
            shadow_draw.multiline_text(
                (text_origin[0] + shadow_offset, text_origin[1] + shadow_offset),
                text_block,
                font=font,
                fill=(0, 0, 0, shadow_alpha),
                align="center",
                spacing=line_spacing,
                stroke_width=stroke_width,
                stroke_fill=(0, 0, 0, shadow_alpha),
            )
            shadow_blur = max(1, int(instruction.font_size * 0.035))
            image.alpha_composite(shadow_layer.filter(ImageFilter.GaussianBlur(radius=shadow_blur)))

        if instruction.glow:
            glow_mask = Image.new("L", (img_w, img_h), 0)
            glow_draw = ImageDraw.Draw(glow_mask)
            glow_stroke = max(stroke_width, 1)
            glow_draw.multiline_text(
                text_origin,
                text_block,
                font=font,
                fill=255,
                align="center",
                spacing=line_spacing,
                stroke_width=glow_stroke,
                stroke_fill=255,
            )
            glow_color = (
                text_color[0],
                text_color[1],
                text_color[2],
                max(120, int(text_color[3] * 0.75)),
            )
            blur_radius = max(3, int(instruction.font_size * 0.12))
            spread_radius = max(2, int(instruction.font_size * 0.05))
            glow_alpha = glow_mask.filter(ImageFilter.MaxFilter(size=spread_radius * 2 + 1))
            glow_alpha = glow_alpha.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            glow_layer = Image.new("RGBA", (img_w, img_h), glow_color)
            image.paste(glow_layer, (0, 0), glow_alpha)

        draw = ImageDraw.Draw(image)
        draw.multiline_text(
            text_origin,
            text_block,
            font=font,
            fill=text_color,
            align="center",
            spacing=line_spacing,
            stroke_width=render_stroke_width,
            stroke_fill=stroke_color,
        )

        frame = np.array(image)
        return mpe.ImageClip(frame, transparent=True)

    def _build_text_clip(self, instruction: TextInstruction, duration: float, target_resolution: tuple[int, int]) -> Optional[mpe.VideoClip]:
        end_time = instruction.end if instruction.end else duration
        if end_time <= instruction.start:
            return None

        try:
            clip = self._build_text_clip_with_pillow(instruction, target_resolution)
        except Exception as exc:
            logger.warning("Pillow text rendering failed (%s); trying TextClip fallback", exc)
            font_arg = str(instruction.font_path) if getattr(instruction, "font_path", None) else instruction.font
            try:
                clip = mpe.TextClip(
                    instruction.content,
                    fontsize=instruction.font_size,
                    font=font_arg,
                    color=self._color_to_hex(instruction.color),
                    stroke_color=self._color_to_hex(instruction.stroke_color) if instruction.stroke_color else None,
                    stroke_width=instruction.stroke_width,
                    method="caption" if instruction.max_width else "label",
                    size=(instruction.max_width, None) if instruction.max_width else None,
                )
            except Exception as fallback_exc:
                logger.warning("TextClip fallback rendering failed (%s); skipping", fallback_exc)
                return None
        if clip is None:
            return None

        clip_duration = max(end_time - instruction.start, 0.001)
        clip = clip.set_duration(clip_duration)

        anim = instruction.animation

        # Scale animation
        if (getattr(anim, "scale_from", 1.0) != 1.0) or (getattr(anim, "scale_to", 1.0) != 1.0):
            scale_from = getattr(anim, "scale_from", 1.0)
            scale_to = getattr(anim, "scale_to", 1.0)

            zoom_dur = anim.zoom_duration if anim.zoom_duration else (end_time - instruction.start)
            if zoom_dur > 0:
                def scale_func(t):
                    progress = min(max(t / zoom_dur, 0.0), 1.0)
                    return scale_from + (scale_to - scale_from) * progress
                clip = clip.resize(scale_func)
            else:
                clip = clip.resize(scale_to)

        # Dynamic resize can drop duration metadata on some clip types.
        clip = clip.set_duration(clip_duration)

        clip = self._apply_text_opacity_fade(
            clip=clip,
            clip_duration=clip_duration,
            fade_in=getattr(anim, "fade_in", 0.0),
            fade_out=getattr(anim, "fade_out", 0.0),
        )

        clip = clip.set_start(instruction.start).set_end(end_time)
        clip = clip.set_position(
            self._resolve_position(
                instruction.position,
                target_resolution,
                clip.size,
                bottom_offset_px=instruction.bottom_offset_px,
            )
        )

        return clip

    @staticmethod
    def _apply_text_opacity_fade(
        clip: mpe.VideoClip,
        clip_duration: float,
        fade_in: float,
        fade_out: float,
    ) -> mpe.VideoClip:
        fin = max(float(fade_in or 0.0), 0.0)
        fout = max(float(fade_out or 0.0), 0.0)
        if fin <= 0.0 and fout <= 0.0:
            return clip

        if clip.mask is None:
            try:
                clip = clip.add_mask()
            except Exception:
                full_mask = mpe.ColorClip(size=clip.size, color=1.0, ismask=True).set_duration(clip_duration)
                clip = clip.set_mask(full_mask)

        mask = clip.mask.set_duration(clip_duration)
        if fin > 0.0:
            mask = mask.fadein(min(fin, clip_duration))
        if fout > 0.0:
            mask = mask.fadeout(min(fout, clip_duration))
        return clip.set_mask(mask)

    def _build_image_clip(self, instruction: ImageInstruction, duration: float, target_resolution: tuple[int, int]) -> Optional[mpe.VideoClip]:
        end_time = instruction.end if instruction.end else duration
        if end_time <= instruction.start:
            return None
        source_path = self._resolve_media_path(instruction.source)
        if not source_path.exists():
            logger.warning("Image source not found: %s", source_path)
            return None
        try:
            clip = mpe.ImageClip(str(source_path))
        except OSError as exc:
            logger.warning("Image loading failed (%s): %s", source_path, exc)
            return None

        if instruction.size:
            clip = clip.resize(newsize=instruction.size.size)
        clip = clip.set_start(instruction.start).set_end(end_time)
        clip = clip.set_position(self._resolve_position(instruction.position, target_resolution, clip.size))
        if instruction.animation.fade_in:
            clip = clip.fadein(instruction.animation.fade_in)
        if instruction.animation.fade_out:
            clip = clip.fadeout(instruction.animation.fade_out)

        if instruction.animation.keyframes:
            clip = clip.set_make_frame(self._apply_keyframes(clip, instruction.animation.keyframes))
        return clip

    def _apply_keyframes(self, clip: mpe.VideoClip, keyframes: List[dict]):
        def make_frame(t):
            frame = clip.get_frame(t)
            for keyframe in keyframes:
                start = keyframe.get("time", 0.0)
                duration = keyframe.get("duration", 0.0)
                if start <= t <= start + duration:
                    scale = keyframe.get("scale")
                    if scale:
                        frame_clip = mpe.ImageClip(frame).resize(scale)
                        frame = frame_clip.get_frame(0)
            return frame
        return make_frame

    def _build_audio(self, request: RenderRequest, duration: float) -> Optional[mpe.AudioClip]:
        tracks = []
        for instruction in request.audio:
            source_path = self._resolve_media_path(instruction.source)
            if not source_path.exists():
                logger.warning("Audio source not found: %s", source_path)
                continue
            if not self._has_audio_stream(source_path):
                logger.warning("Audio source has no audio stream: %s", source_path)
                continue
            try:
                clip = mpe.AudioFileClip(str(source_path))
            except OSError as exc:
                logger.warning("Audio loading failed (%s): %s", source_path, exc)
                continue
            source_start = max(float(instruction.start or 0.0), 0.0)
            source_end = float(instruction.end) if instruction.end else float(clip.duration or 0.0)
            source_end = min(source_end, float(clip.duration or 0.0))
            if source_end <= source_start:
                clip.close()
                continue

            timeline_start = instruction.timeline_start
            if timeline_start is None:
                timeline_start = instruction.at
            if timeline_start is None:
                timeline_start = source_start
            timeline_start = max(float(timeline_start or 0.0), 0.0)

            remaining_duration = max(duration - timeline_start, 0.0)
            if remaining_duration <= 0.0:
                clip.close()
                continue

            source_end = min(source_end, source_start + remaining_duration)
            if source_end <= source_start:
                clip.close()
                continue

            clip = clip.subclip(source_start, source_end)
            clip = clip.volumex(instruction.volume)
            if instruction.fade_in:
                clip = clip.audio_fadein(instruction.fade_in)
            if instruction.fade_out:
                clip = clip.audio_fadeout(instruction.fade_out)
            clip = self._guard_audio_clip_bounds(clip)
            clip = clip.set_start(timeline_start)
            tracks.append(clip)

        tracks.extend(self._build_insert_audio_tracks(request.attachments, duration, "attachment"))
        tracks.extend(self._build_insert_audio_tracks(request.inserts, duration, "insert"))

        if not tracks:
            return None
        return mpe.CompositeAudioClip(tracks).set_duration(duration)

    def _build_insert_audio_tracks(
        self,
        instructions: Iterable[InsertInstruction],
        timeline_duration: float,
        label: str,
    ) -> List[mpe.AudioClip]:
        tracks: List[mpe.AudioClip] = []
        for index, instruction in enumerate(instructions):
            volume = float(instruction.volume or 0.0)
            if volume <= 0.0:
                continue

            source_path = self._resolve_media_path(instruction.source)
            if not source_path.exists():
                logger.warning("%s audio source not found: %s", label.title(), source_path)
                continue
            if not self._has_audio_stream(source_path):
                logger.info("%s audio source has no audio stream: %s", label.title(), source_path)
                continue

            try:
                audio_clip = mpe.AudioFileClip(str(source_path))
            except OSError as exc:
                logger.warning("%s audio loading failed (%s): %s", label.title(), source_path, exc)
                continue

            source_start = max(float(instruction.start or 0.0), 0.0)
            source_end = float(instruction.end) if instruction.end else float(audio_clip.duration or 0.0)
            source_end = min(source_end, float(audio_clip.duration or 0.0))
            if source_end <= source_start:
                audio_clip.close()
                continue

            source_duration = source_end - source_start
            timeline_start = self._resolve_attachment_start(instruction, timeline_duration, source_duration)
            if timeline_start < 0.0:
                source_start += -timeline_start
                timeline_start = 0.0

            remaining_duration = max(timeline_duration - timeline_start, 0.0)
            if remaining_duration <= 0.0 or source_start >= source_end:
                audio_clip.close()
                continue

            source_end = min(source_end, source_start + remaining_duration)
            if source_end <= source_start:
                audio_clip.close()
                continue

            try:
                track = audio_clip.subclip(source_start, source_end).volumex(volume)
                if instruction.fade_in:
                    track = track.audio_fadein(instruction.fade_in)
                if instruction.fade_out:
                    track = track.audio_fadeout(instruction.fade_out)
                track = self._guard_audio_clip_bounds(track).set_start(timeline_start)
            except Exception as exc:
                audio_clip.close()
                logger.warning("%s audio track failed (%s #%s): %s", label.title(), source_path, index, exc)
                continue
            tracks.append(track)

        return tracks

    @staticmethod
    def _guard_audio_clip_bounds(clip: mpe.AudioClip) -> mpe.AudioClip:
        """Return silence when MoviePy probes slightly outside a clip.

        CompositeAudioClip can request a timestamp vector that crosses the start
        of a delayed insert. AudioFileClip readers then receive small negative
        local timestamps and crash before the inactive samples are masked out.
        """
        duration = max(float(clip.duration or 0.0), 0.0)
        if duration <= 0.0:
            return clip

        fps = getattr(clip, "fps", None) or 44100
        probe_time = min(duration / 2.0, max(duration - 1e-6, 0.0))
        try:
            probe = np.asarray(clip.get_frame(probe_time))
            channels = int(probe.shape[-1]) if probe.ndim > 0 else 1
        except Exception:
            channels = int(getattr(clip, "nchannels", 2) or 2)

        def make_frame(t):
            times = np.asarray(t)
            max_time = max(duration - 1e-6, 0.0)
            if times.ndim == 0:
                value = float(times)
                if value < 0.0 or value >= duration:
                    return np.zeros(channels, dtype=float)
                return clip.get_frame(min(max(value, 0.0), max_time))

            valid = (times >= 0.0) & (times < duration)
            safe_times = np.clip(times, 0.0, max_time)
            frames = np.asarray(clip.get_frame(safe_times)).copy()
            if frames.ndim == 1:
                frames[~valid] = 0.0
            else:
                frames[~valid, ...] = 0.0
            return frames

        guarded = mpe.AudioClip(make_frame=make_frame, duration=duration, fps=fps)
        try:
            guarded.nchannels = channels
        except Exception:
            pass
        return guarded

    def _resolve_output_path(self, request: RenderRequest) -> Path:
        self.workspace.mkdir(parents=True, exist_ok=True)
        filename = Path(request.output.filename).name
        extension = request.output.format.lower().lstrip(".")
        if not filename:
            filename = "render"
        if Path(filename).suffix.lower() != f".{extension}":
            filename = f"{Path(filename).stem}.{extension}"
        return (self.workspace / filename).resolve()

    def _build_ffmpeg_runtime_args(self) -> List[str]:
        thread_count = self._get_thread_count("FFMPEG_THREADS")
        filter_thread_count = self._get_thread_count("FFMPEG_FILTER_THREADS", thread_count)
        complex_thread_count = self._get_thread_count("FFMPEG_FILTER_COMPLEX_THREADS", filter_thread_count)
        return [
            "-threads",
            str(thread_count),
            "-filter_threads",
            str(filter_thread_count),
            "-filter_complex_threads",
            str(complex_thread_count),
        ]

    def _build_fast_video_codec_args(self, bitrate: Optional[str], extension: Optional[str] = None) -> List[str]:
        use_gpu = os.getenv("FFMPEG_USE_GPU", "0").lower() in {"1", "true", "yes", "on"}
        gpu_preset = os.getenv("FFMPEG_GPU_PRESET", "p4")
        thread_count = self._get_thread_count("FFMPEG_THREADS")
        if use_gpu:
            codec_args = ["-c:v", "h264_nvenc", "-preset", gpu_preset, "-profile:v", "high"]
            if bitrate:
                codec_args += ["-b:v", bitrate]
            else:
                codec_args += ["-rc:v", "vbr", "-cq:v", os.getenv("FFMPEG_NVENC_CQ", "23")]
        else:
            x264_preset = os.getenv("FFMPEG_X264_PRESET_FAST", os.getenv("FFMPEG_X264_PRESET", "veryfast"))
            codec_args = ["-c:v", "libx264", "-preset", x264_preset, "-profile:v", "high"]
            if bitrate:
                codec_args += ["-b:v", bitrate]
            else:
                codec_args += ["-crf", os.getenv("FFMPEG_X264_CRF", "23")]
        codec_args += ["-threads", str(thread_count)]
        codec_args += ["-pix_fmt", "yuv420p"]
        if extension in {"mp4", "mov"}:
            codec_args += ["-tag:v", "avc1"]
        return codec_args

    def _build_moviepy_mp4_export_settings(
        self,
        bitrate: Optional[str],
        filter_thread_count: int,
    ) -> tuple[str, str, str, int, str, List[str]]:
        use_gpu = os.getenv("FFMPEG_USE_GPU", "0").lower() in {"1", "true", "yes", "on"}
        if use_gpu:
            codec = "h264_nvenc"
            preset = os.getenv("FFMPEG_GPU_PRESET", "p4")
        else:
            codec = "libx264"
            preset = os.getenv("FFMPEG_X264_PRESET", "medium")

        audio_sample_rate = self._get_positive_int_env("FFMPEG_AAC_SAMPLE_RATE", 48000)
        audio_bitrate = os.getenv("FFMPEG_AAC_BITRATE", "").strip() or "128k"
        ffmpeg_params = [
            "-filter_threads",
            str(filter_thread_count),
            "-pix_fmt",
            "yuv420p",
            "-profile:v",
            "high",
            "-tag:v",
            "avc1",
            "-movflags",
            "+faststart",
        ]
        if bitrate:
            return codec, "aac", preset, audio_sample_rate, audio_bitrate, ffmpeg_params
        if use_gpu:
            ffmpeg_params += ["-rc:v", "vbr", "-cq:v", os.getenv("FFMPEG_NVENC_CQ", "23")]
        else:
            ffmpeg_params += ["-crf", os.getenv("FFMPEG_X264_CRF", "23")]
        return codec, "aac", preset, audio_sample_rate, audio_bitrate, ffmpeg_params

    @staticmethod
    def _build_fast_audio_codec_args(audio_bitrate: Optional[str] = None) -> List[str]:
        bitrate = str(audio_bitrate or os.getenv("FFMPEG_AAC_BITRATE", "").strip() or "128k").strip() or "128k"
        return ["-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", bitrate]

    def _run_external_command(
        self,
        command: List[str],
        stage: str,
        timeout_env: str = "FFMPEG_COMMAND_TIMEOUT_SECONDS",
        default_timeout_seconds: float = 0.0,
    ) -> None:
        timeout_seconds = self._get_timeout_seconds(timeout_env, default_timeout_seconds)
        try:
            process = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            timeout_label = f"{timeout_seconds:.1f}s" if timeout_seconds is not None else "disabled"
            raise VideoEngineError(f"{stage} timed out after {timeout_label}") from exc
        if process.returncode == 0:
            return
        # Keep enough FFmpeg context to identify the failing filter; ten final
        # lines often omit the actual reinitialization error.
        stderr_tail = (process.stderr or process.stdout or "").strip().splitlines()[-60:]
        details = " | ".join(stderr_tail)
        if details:
            raise VideoEngineError(f"{stage} failed: {details}")
        raise VideoEngineError(f"{stage} failed with exit code {process.returncode}")

    def _probe_duration_seconds(self, path: Path) -> float:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        timeout_seconds = self._get_timeout_seconds("FFPROBE_COMMAND_TIMEOUT_SECONDS", 60.0)
        try:
            process = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            timeout_label = f"{timeout_seconds:.1f}s" if timeout_seconds is not None else "disabled"
            raise VideoEngineError(f"ffprobe timed out after {timeout_label} for {path}") from exc
        if process.returncode != 0:
            details = (process.stderr or process.stdout or "").strip()
            raise VideoEngineError(f"ffprobe failed for {path}: {details}")
        try:
            return max(float((process.stdout or "").strip()), 0.0)
        except ValueError as exc:
            raise VideoEngineError(f"Unable to parse duration for {path}") from exc

    def _has_audio_stream(self, path: Path) -> bool:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
        timeout_seconds = self._get_timeout_seconds("FFPROBE_COMMAND_TIMEOUT_SECONDS", 60.0)
        try:
            process = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            timeout_label = f"{timeout_seconds:.1f}s" if timeout_seconds is not None else "disabled"
            raise VideoEngineError(f"ffprobe audio stream check timed out after {timeout_label} for {path}") from exc
        if process.returncode != 0:
            details = (process.stderr or process.stdout or "").strip()
            raise VideoEngineError(f"ffprobe audio stream check failed for {path}: {details}")
        return bool((process.stdout or "").strip())

    def _try_render_fast_path(
        self,
        request: RenderRequest,
        target_resolution: tuple[int, int],
    ) -> Optional[RenderResult]:
        if not self._get_bool_env("FFMPEG_RENDER_FAST_PATH", False):
            return None

        strict = self._get_bool_env("FFMPEG_RENDER_FAST_PATH_STRICT", True)
        rejection_reason = self._fast_render_rejection_reason(request)
        if rejection_reason:
            logger.info("Fast render path skipped: %s", rejection_reason)
            if strict:
                raise VideoEngineError(f"Fast render path unavailable: {rejection_reason}")
            return None

        try:
            logger.info("Using FFmpeg fast render path for %s", request.output.filename)
            if self._requires_fast_overlay_timeline(request):
                return self._render_fast_overlay_timeline(request, target_resolution)
            return self._render_fast_linear(request, target_resolution)
        except Exception as exc:
            logger.warning("Fast render path failed; falling back to MoviePy render: %s", exc)
            if strict:
                raise VideoEngineError(f"Fast render path failed: {exc}") from exc
            return None

    def _fast_render_rejection_reason(self, request: RenderRequest) -> Optional[str]:
        extension = request.output.format.lower().lstrip(".")
        if extension not in {"mp4", "mov", "mkv"}:
            return f"unsupported output format {request.output.format!r}"
        if not request.clips:
            return "no clips"
        if request.timeline or request.images or request.show_source:
            return "request uses layered/timeline features"
        if request.zoom_border:
            return "zoom_border is not supported by fast render path yet"

        approximate = self._get_bool_env("FFMPEG_RENDER_APPROXIMATE_TRANSITIONS", False)
        for index, instruction in enumerate(request.clips):
            if instruction.chroma_key.enabled:
                return "chroma_key is not supported by fast render path"
            if any(
                abs(float(value or 0.0)) > 1e-9
                for value in (
                    instruction.adjustments.brightness,
                    instruction.adjustments.contrast,
                    instruction.adjustments.saturation,
                    instruction.adjustments.hue,
                )
            ):
                return "clip adjustments are not supported by fast render path"
            if len(instruction.transitions_before) > 1 or len(instruction.transitions_after) > 1:
                return "multiple transitions per clip are not supported by fast render path"
            for transition in [*instruction.transitions_before, *instruction.transitions_after]:
                if transition.type in {TransitionType.NONE, TransitionType.CROSSFADE, TransitionType.FADE_BLACK}:
                    continue
                if transition.type in {TransitionType.WHIP_PAN, TransitionType.MOTION_BLUR} and approximate:
                    continue
                return f"transition {transition.type.value!r} requires MoviePy exact render"
            if index == len(request.clips) - 1 and instruction.transitions_after:
                outro = instruction.transitions_after[0]
                if outro.type not in {TransitionType.NONE, TransitionType.FADE_BLACK}:
                    return "last clip outro transition requires MoviePy exact render"

        for label, instructions in (("attachment", request.attachments), ("insert", request.inserts)):
            for instruction in instructions:
                if any(
                    abs(float(value or 0.0)) > 1e-9
                    for value in (
                        instruction.adjustments.brightness,
                        instruction.adjustments.contrast,
                        instruction.adjustments.saturation,
                        instruction.adjustments.hue,
                    )
                ):
                    return f"{label} adjustments are not supported by fast render path"
                if abs(float(instruction.playback_rate or 1.0) - 1.0) > 1e-9:
                    return f"{label} playback_rate is not supported by fast render path"
                if len(instruction.transitions_before) > 1 or len(instruction.transitions_after) > 1:
                    return f"multiple transitions per {label} are not supported by fast render path"
                for transition in [*instruction.transitions_before, *instruction.transitions_after]:
                    if transition.type in {TransitionType.NONE, TransitionType.CROSSFADE, TransitionType.FADE_BLACK}:
                        continue
                    if transition.type in {TransitionType.WHIP_PAN, TransitionType.MOTION_BLUR} and approximate:
                        continue
                    return f"{label} transition {transition.type.value!r} requires MoviePy exact render"

        for text in request.texts:
            anim = text.animation
            if abs(float(getattr(anim, "scale_from", 1.0) or 1.0) - 1.0) > 1e-9:
                return "text scale animation is not supported by fast render path"
            if abs(float(getattr(anim, "scale_to", 1.0) or 1.0) - 1.0) > 1e-9:
                return "text scale animation is not supported by fast render path"

        return None

    @staticmethod
    def _requires_fast_overlay_timeline(request: RenderRequest) -> bool:
        if request.attachments or request.inserts:
            return True
        return any(VideoEngine._has_explicit_at(instruction) for instruction in request.clips)

    @staticmethod
    def _fast_xfade_name(transition: TransitionInstruction) -> str:
        if transition.type == TransitionType.CROSSFADE:
            return "fade"
        if transition.type == TransitionType.FADE_BLACK:
            return "fadeblack"
        if transition.type == TransitionType.WHIP_PAN:
            direction = transition.direction.value if hasattr(transition.direction, "value") else str(transition.direction)
            return {
                "left": "slideleft",
                "right": "slideright",
                "top": "slideup",
                "bottom": "slidedown",
            }.get(direction, "slideleft")
        if transition.type == TransitionType.MOTION_BLUR:
            return "fade"
        return "fade"

    @staticmethod
    def _fast_transition_uses_xfade(transition: Optional[TransitionInstruction]) -> bool:
        if not transition:
            return False
        return transition.type in {
            TransitionType.CROSSFADE,
            TransitionType.WHIP_PAN,
            TransitionType.MOTION_BLUR,
        }

    def _fast_transition_overlap(
        self,
        transition: Optional[TransitionInstruction],
        previous_duration: float,
        current_base_duration: float,
    ) -> float:
        if not self._fast_transition_uses_xfade(transition):
            return 0.0
        if transition.type == TransitionType.MOTION_BLUR and not transition.fade:
            return 0.0
        requested = max(float(transition.duration or 0.0), 0.0)
        return max(min(requested, previous_duration, current_base_duration), 0.0)

    @staticmethod
    def _fast_transition_edge_filters(
        transition: Optional[TransitionInstruction],
        clip_duration: float,
        phase: str,
        fps: int = 24,
    ) -> list[str]:
        if transition is None or transition.type not in {TransitionType.WHIP_PAN, TransitionType.MOTION_BLUR}:
            return []
        duration = min(max(float(transition.duration or 0.0), 0.0), max(float(clip_duration), 0.0))
        if duration <= 0.0:
            return []

        # The legacy MoviePy implementation animated a directional box blur on
        # every frame. avgblur does not accept a time expression for its kernel,
        # so approximate the same quadratic velocity curve with short windows.
        # This is deliberately applied after the whip-pan scene is composed.
        max_strength = max(min(float(transition.blur_strength or 0.0), 1023.0), 0.0)
        if max_strength < 2.0:
            return []
        direction = transition.direction.value if hasattr(transition.direction, "value") else str(transition.direction)
        vertical = transition.type == TransitionType.WHIP_PAN and direction in {"top", "bottom"}
        # Use one blur-strength window per output frame. This preserves the
        # legacy quadratic curve without visible eight-step jumps.
        window_count = max(int(duration * max(int(fps), 1) + 0.999999), 1)
        start_offset = 0.0 if phase in {"intro", "between"} else max(float(clip_duration) - duration, 0.0)
        result: list[str] = []
        for window_index in range(window_count):
            window_start = start_offset + duration * window_index / window_count
            window_end = start_offset + duration * (window_index + 1) / window_count
            progress = (window_index + 0.5) / window_count
            if phase == "intro":
                factor = (1.0 - progress) ** 2
            elif phase == "outro":
                factor = progress ** 2
            else:
                factor = 4.0 * progress**2 if progress < 0.5 else 4.0 * (1.0 - progress) ** 2
            kernel = int(round(max_strength * factor))
            if kernel < 2:
                continue
            if kernel % 2 == 0:
                kernel += 1
            size_x, size_y = (1, kernel) if vertical else (kernel, 1)
            enable = f"gte(t\\,{window_start:.6f})*lt(t\\,{window_end:.6f})"
            result.append(f"avgblur=sizeX={size_x}:sizeY={size_y}:enable='{enable}'")
            # Glow belongs to motion_blur only in the legacy path. It followed
            # the same quadratic curve as the blur strength.
            if transition.type == TransitionType.MOTION_BLUR and transition.glow:
                brightness = min(0.5 * factor, 0.5)
                result.append(f"eq=brightness={brightness:.6f}:enable='{enable}'")
        return result

    @staticmethod
    def _fast_transition_uses_alpha_fade(transition: Optional[TransitionInstruction]) -> bool:
        if transition is None:
            return False
        if transition.type in {TransitionType.CROSSFADE, TransitionType.FADE_BLACK}:
            return True
        return transition.type == TransitionType.MOTION_BLUR and transition.fade

    @staticmethod
    def _fast_overlay_position_expressions(
        item: dict[str, object],
        target_resolution: tuple[int, int],
    ) -> tuple[str, str]:
        width, height = target_resolution
        start = max(float(item["start"]), 0.0)
        duration = max(float(item["duration"]), 0.001)
        end = start + duration
        x_terms: list[str] = []
        y_terms: list[str] = []

        def append_motion(transition: Optional[TransitionInstruction], phase: str) -> None:
            if transition is None or transition.type != TransitionType.WHIP_PAN:
                return
            transition_duration = min(max(float(transition.duration or 0.0), 0.0), duration)
            if transition_duration <= 0.0:
                return
            direction = transition.direction.value if hasattr(transition.direction, "value") else str(transition.direction)
            if phase == "intro":
                local_start = start
                progress = f"(t-{local_start:.6f})/{transition_duration:.6f}"
                enabled = f"between(t\\,{local_start:.6f}\\,{local_start + transition_duration:.6f})"
                eased = f"(1-pow(1-({progress})\\,3))"
                values = {
                    "left": (f"{width}-{eased}*{width}", None),
                    "right": (f"-{width}+{eased}*{width}", None),
                    "top": (None, f"{height}-{eased}*{height}"),
                    "bottom": (None, f"-{height}+{eased}*{height}"),
                }
            else:
                local_start = max(end - transition_duration, start)
                progress = f"(t-{local_start:.6f})/{transition_duration:.6f}"
                enabled = f"between(t\\,{local_start:.6f}\\,{end:.6f})"
                eased = f"pow(({progress})\\,3)"
                values = {
                    "left": (f"-{eased}*{width}", None),
                    "right": (f"{eased}*{width}", None),
                    "top": (None, f"-{eased}*{height}"),
                    "bottom": (None, f"{eased}*{height}"),
                }
            x_value, y_value = values.get(direction, values["left"])
            if x_value is not None:
                x_terms.append(f"if({enabled}\\,{x_value}\\,0)")
            if y_value is not None:
                y_terms.append(f"if({enabled}\\,{y_value}\\,0)")

        append_motion(item.get("intro_transition"), "intro")
        append_motion(item.get("outro_transition"), "outro")
        return ("+".join(x_terms) or "0", "+".join(y_terms) or "0")

    def _build_fast_whip_pan_between_filters(
        self,
        previous_label: str,
        next_label: str,
        output_label: str,
        sequence_index: int,
        transition: TransitionInstruction,
        previous_duration: float,
        next_duration: float,
        overlap: float,
        target_resolution: tuple[int, int],
        fps: int,
    ) -> list[str]:
        """Build the legacy whip-pan scene in FFmpeg without reducing it to xfade."""
        width, height = target_resolution
        duration = max(overlap, 0.001)
        body_duration = max(previous_duration - duration, 0.0)
        rest_duration = max(next_duration - duration, 0.0)
        direction = transition.direction.value if hasattr(transition.direction, "value") else str(transition.direction)
        dx, dy = {
            "left": (-width, 0),
            "right": (width, 0),
            "top": (0, -height),
            "bottom": (0, height),
        }.get(direction, (-width, 0))
        progress = f"(t/{duration:.6f})"
        eased = (
            f"if(lt({progress}\\,0.5)\\,4*pow({progress}\\,3)\\,"
            f"1-pow(-2*{progress}+2\\,3)/2)"
        )
        previous_x = f"({dx})*({eased})"
        previous_y = f"({dy})*({eased})"
        next_x = f"(-({dx}))+({dx})*({eased})"
        next_y = f"(-({dy}))+({dy})*({eased})"

        body_label = f"vwhipbody{sequence_index}"
        tail_label = f"vwhiptail{sequence_index}"
        head_label = f"vwhiphead{sequence_index}"
        rest_label = f"vwhiprest{sequence_index}"
        body_source_label = f"vwhipbodysrc{sequence_index}"
        tail_source_label = f"vwhiptailsrc{sequence_index}"
        head_source_label = f"vwhipheadsrc{sequence_index}"
        rest_source_label = f"vwhiprestsrc{sequence_index}"
        black_label = f"vwhipblack{sequence_index}"
        moved_previous_label = f"vwhipprev{sequence_index}"
        scene_label = f"vwhipscene{sequence_index}"
        transition_label = f"vwhiptransition{sequence_index}"
        filters: list[str] = []
        parts: list[str] = []

        if body_duration > 1e-6:
            filters.append(
                f"[{previous_label}]split=2[{body_source_label}][{tail_source_label}]"
            )
            filters.append(
                f"[{body_source_label}]trim=duration={body_duration:.6f},"
                f"setpts=PTS-STARTPTS[{body_label}]"
            )
            parts.append(body_label)
        else:
            tail_source_label = previous_label
        filters.append(
            f"[{tail_source_label}]trim=start={body_duration:.6f}:end={previous_duration:.6f},"
            f"setpts=PTS-STARTPTS[{tail_label}]"
        )
        if rest_duration > 1e-6:
            filters.append(
                f"[{next_label}]split=2[{head_source_label}][{rest_source_label}]"
            )
        else:
            head_source_label = next_label
        filters.append(
            f"[{head_source_label}]trim=duration={duration:.6f},setpts=PTS-STARTPTS[{head_label}]"
        )
        filters.append(f"color=c=black:s={width}x{height}:r={fps}:d={duration:.6f}[{black_label}]")
        filters.append(
            f"[{black_label}][{tail_label}]overlay=x='{previous_x}':y='{previous_y}':"
            f"eval=frame:shortest=1[{moved_previous_label}]"
        )
        filters.append(
            f"[{moved_previous_label}][{head_label}]overlay=x='{next_x}':y='{next_y}':"
            f"eval=frame:shortest=1[{scene_label}]"
        )
        blur_filters = self._fast_transition_edge_filters(transition, duration, "between", fps)
        if blur_filters:
            filters.append(f"[{scene_label}]{','.join(blur_filters)},format=yuv420p[{transition_label}]")
        else:
            filters.append(f"[{scene_label}]format=yuv420p[{transition_label}]")
        parts.append(transition_label)
        if rest_duration > 1e-6:
            filters.append(
                f"[{rest_source_label}]trim=start={duration:.6f},setpts=PTS-STARTPTS[{rest_label}]"
            )
            parts.append(rest_label)

        if len(parts) == 1:
            filters.append(f"[{parts[0]}]fps={fps},settb=AVTB[{output_label}]")
        else:
            filters.append(
                f"{''.join(f'[{part}]' for part in parts)}concat=n={len(parts)}:v=1:a=0,"
                f"fps={fps},settb=AVTB[{output_label}]"
            )
        return filters

    def _fast_clip_trim_window(
        self,
        instruction: ClipInstruction,
        source_duration: float,
        transition_padding: float,
    ) -> tuple[float, float, float]:
        start = max(float(instruction.start or 0.0), 0.0)
        end = float(instruction.end) if instruction.end else source_duration
        end = min(end, source_duration)
        if end <= start:
            end = source_duration

        playback_rate = max(float(instruction.playback_rate or 1.0), 1e-6)
        base_duration = max(end - start, 0.0) / playback_rate
        transition_padding = max(float(transition_padding or 0.0), 0.0)
        raw_padding_needed = transition_padding * playback_rate
        extend_before = min(raw_padding_needed, start)
        raw_padding_remaining = max(raw_padding_needed - extend_before, 0.0)
        extend_after = min(raw_padding_remaining, max(source_duration - end, 0.0))
        trim_start = max(start - extend_before, 0.0)
        trim_end = min(end + extend_after, source_duration)
        target_duration = max(base_duration + transition_padding, trim_end - trim_start)
        return trim_start, trim_end, target_duration

    def _build_fast_clip_filter_graph(
        self,
        instruction: ClipInstruction,
        target_resolution: tuple[int, int],
        target_fps: int,
        output_duration: float,
        intro_fade_black: float,
        outro_fade_black: float,
        timing_stretch: float = 1.0,
    ) -> str:
        target_w, target_h = target_resolution
        source_filters = ["setpts=PTS-STARTPTS"]
        playback_rate = max(float(instruction.playback_rate or 1.0), 1e-6)
        if abs(playback_rate - 1.0) > 1e-9:
            source_filters.append(f"setpts=PTS/{playback_rate:.9f}")
        if timing_stretch > 1.0 + 1e-9:
            source_filters.append(f"setpts=PTS*{timing_stretch:.9f}")
        if instruction.mirror_horizontal:
            source_filters.append("hflip")
        zoom_filter = self._build_center_zoom_filter(instruction.effective_internal_zoom)
        if zoom_filter:
            source_filters.append(zoom_filter)
        source_chain = ",".join(source_filters)

        if instruction.fit_mode == FitMode.COVER:
            graph = (
                f"[0:v]{source_chain},"
                f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase:force_divisible_by=2,"
                f"crop={target_w}:{target_h}:(iw-ow)/2:(ih-oh)/2,"
                "setsar=1"
            )
        elif instruction.background_mode == BackgroundMode.COLOR or instruction.chroma_key.enabled:
            if instruction.chroma_key.enabled:
                key_color = instruction.chroma_key.color.as_tuple()
                color = f"0x{key_color[0]:02x}{key_color[1]:02x}{key_color[2]:02x}"
            else:
                color = f"0x{instruction.background_color.r:02x}{instruction.background_color.g:02x}{instruction.background_color.b:02x}"
            graph = (
                f"[0:v]{source_chain},"
                f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease:force_divisible_by=2,"
                "setsar=1[fg];"
                f"color=c={color}:s={target_w}x{target_h}:r={target_fps}:d={output_duration:.6f}[bg];"
                "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1"
            )
        else:
            sigma = max(float(instruction.background_color.a or 0.0) * 25.0, 5.0)
            graph = (
                f"[0:v]{source_chain},split=2[fgsrc][bgsrc];"
                f"[bgsrc]scale={target_w}:{target_h}:force_original_aspect_ratio=increase:force_divisible_by=2,"
                f"crop={target_w}:{target_h}:(iw-ow)/2:(ih-oh)/2,"
                f"gblur=sigma={sigma:.3f},setsar=1[bg];"
                f"[fgsrc]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease:force_divisible_by=2,"
                "setsar=1[fg];"
                "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1"
            )

        tail_filters = [f"fps={target_fps}", "format=yuv420p"]
        if intro_fade_black > 0.0:
            tail_filters.append(f"fade=t=in:st=0:d={intro_fade_black:.6f}")
        if outro_fade_black > 0.0:
            fade_start = max(output_duration - outro_fade_black, 0.0)
            tail_filters.append(f"fade=t=out:st={fade_start:.6f}:d={outro_fade_black:.6f}")
        tail_filters.append("setpts=PTS-STARTPTS")
        return f"{graph},{','.join(tail_filters)}[vout]"

    def _normalize_clip_for_fast_render(
        self,
        instruction: ClipInstruction,
        source_path: Path,
        output_path: Path,
        target_resolution: tuple[int, int],
        target_fps: int,
        bitrate: Optional[str],
        transition_padding: float,
        intro_fade_black: float,
        outro_fade_black: float,
    ) -> float:
        source_duration = self._probe_duration_seconds(source_path)
        trim_start, trim_end, output_duration = self._fast_clip_trim_window(
            instruction,
            source_duration,
            transition_padding,
        )
        if trim_end <= trim_start:
            raise VideoEngineError(f"Clip has no renderable duration: {source_path}")

        filter_graph = self._build_fast_clip_filter_graph(
            instruction,
            target_resolution,
            target_fps,
            output_duration,
            intro_fade_black,
            outro_fade_black,
            timing_stretch=max(
                output_duration
                / max((trim_end - trim_start) / max(float(instruction.playback_rate or 1.0), 1e-6), 1e-6),
                1.0,
            ),
        )
        command = [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            *self._build_ffmpeg_runtime_args(),
        ]
        if trim_start > 0:
            command += ["-ss", f"{trim_start:.6f}"]
        command += ["-i", str(source_path), "-t", f"{max(trim_end - trim_start, 0.001):.6f}"]
        command += [
            "-an",
            "-filter_complex",
            filter_graph,
            "-map",
            "[vout]",
            "-dn",
            "-map_metadata",
            "-1",
            *self._build_fast_video_codec_args(bitrate, output_path.suffix.lower().lstrip(".")),
            "-t",
            f"{output_duration:.6f}",
        ]
        if output_path.suffix.lower().lstrip(".") in {"mp4", "mov"}:
            command += ["-movflags", "+faststart"]
        command.append(str(output_path))
        self._run_external_command(command, f"FFmpeg fast normalize clip {source_path.name}")
        return self._probe_duration_seconds(output_path)

    def _resolve_fast_overlay_position(
        self,
        position: str,
        target_resolution: tuple[int, int],
        clip_size: tuple[int, int],
        bottom_offset_px: int | None = None,
    ) -> tuple[int, int]:
        x, y = self._resolve_position(position, target_resolution, clip_size, bottom_offset_px=bottom_offset_px)
        target_w, target_h = target_resolution
        clip_w, clip_h = clip_size
        if x == "center":
            x = (target_w - clip_w) / 2
        if y == "center":
            y = (target_h - clip_h) / 2
        return max(int(round(float(x))), 0), max(int(round(float(y))), 0)

    @staticmethod
    def _ass_timestamp(seconds: float) -> str:
        centiseconds = max(int(round(float(seconds) * 100.0)), 0)
        hours, remainder = divmod(centiseconds, 360000)
        minutes, remainder = divmod(remainder, 6000)
        whole_seconds, fraction = divmod(remainder, 100)
        return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{fraction:02d}"

    @staticmethod
    def _ass_color_parts(color, default: tuple[int, int, int, int]) -> tuple[str, str]:
        if color is None:
            r, g, b, alpha = default
        else:
            r = int(color.r)
            g = int(color.g)
            b = int(color.b)
            alpha = int(round(max(min(float(color.a), 1.0), 0.0) * 255.0))
        ass_alpha = 255 - alpha
        return f"&H{b:02X}{g:02X}{r:02X}&", f"&H{ass_alpha:02X}&"

    @staticmethod
    def _escape_ass_text(value: str) -> str:
        return (
            str(value or "")
            .replace("\\", r"\\")
            .replace("{", r"\{")
            .replace("}", r"\}")
            .replace("\r\n", r"\N")
            .replace("\n", r"\N")
            .replace("\r", r"\N")
        )

    @staticmethod
    def _ass_position(
        position: str,
        target_resolution: tuple[int, int],
        bottom_offset_px: int | None,
    ) -> tuple[int, int, int]:
        width, height = target_resolution
        margin_x = max(int(round(width * 0.05)), 1)
        margin_y = max(int(round(height * 0.05)), 1)
        bottom_margin = margin_y if bottom_offset_px is None else max(int(bottom_offset_px), 0)
        positions = {
            "center": (5, width // 2, height // 2),
            "top": (8, width // 2, margin_y),
            "bottom": (2, width // 2, max(height - bottom_margin, 0)),
            "left": (4, margin_x, height // 2),
            "right": (6, max(width - margin_x, 0), height // 2),
            "top_left": (7, margin_x, margin_y),
            "top_right": (9, max(width - margin_x, 0), margin_y),
            "bottom_left": (1, margin_x, max(height - bottom_margin, 0)),
            "bottom_right": (3, max(width - margin_x, 0), max(height - bottom_margin, 0)),
        }
        return positions.get(position, positions["center"])

    def _build_fast_ass_subtitles(
        self,
        texts: list[TextInstruction],
        target_resolution: tuple[int, int],
        duration: float,
        temp_dir: Path,
    ) -> Optional[tuple[Path, Optional[Path]]]:
        width, height = target_resolution
        events: list[str] = []
        fonts_dir = temp_dir / "fonts"
        copied_fonts: set[Path] = set()
        for instruction in texts:
            start = max(float(instruction.start or 0.0), 0.0)
            end = min(float(instruction.end) if instruction.end else duration, duration)
            if end <= start:
                continue

            font = self._resolve_pillow_font(instruction)
            try:
                resolved_family = str(font.getname()[0]).strip()
            except Exception:
                resolved_family = ""
            font_source_value = getattr(font, "path", None)
            if font_source_value:
                font_source = Path(str(font_source_value))
                if font_source.exists() and font_source not in copied_fonts:
                    fonts_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(font_source, fonts_dir / font_source.name)
                    copied_fonts.add(font_source)
            stroke_width = max(int(instruction.stroke_width), 0)
            max_width = instruction.max_width or int(width * 0.9)
            max_width = max(1, min(int(max_width), int(width * 0.95)))
            lines = self._wrap_text_lines(instruction.content, font, max_width, stroke_width)
            content = self._escape_ass_text("\n".join(lines))

            alignment, x, y = self._ass_position(
                instruction.position,
                target_resolution,
                instruction.bottom_offset_px,
            )
            font_name = resolved_family or (instruction.font or "DejaVu Sans").replace("-", " ")
            font_name = font_name.replace("{", "").replace("}", "").replace("\\", "")
            primary_color, primary_alpha = self._ass_color_parts(
                instruction.color,
                (255, 255, 255, 255),
            )
            outline_color, outline_alpha = self._ass_color_parts(
                instruction.stroke_color,
                (0, 0, 0, 255),
            )
            border = max(stroke_width, 3 if instruction.glow else 0)
            shadow = max(2, int(round(instruction.font_size * 0.06))) if instruction.shadow else 0
            clip_duration = max(end - start, 0.001)
            fade_in_ms = int(round(min(float(instruction.animation.fade_in or 0.0), clip_duration) * 1000.0))
            fade_out_ms = int(round(min(float(instruction.animation.fade_out or 0.0), clip_duration) * 1000.0))
            overrides = (
                f"\\an{alignment}\\pos({x},{y})"
                f"\\fn{font_name}\\fs{int(instruction.font_size)}"
                f"\\1c{primary_color}\\1a{primary_alpha}"
                f"\\3c{outline_color}\\3a{outline_alpha}"
                f"\\bord{border}\\shad{shadow}"
                f"\\4c&H000000&\\4a&H40&"
                f"\\fad({fade_in_ms},{fade_out_ms})"
            )
            events.append(
                "Dialogue: 0,"
                f"{self._ass_timestamp(start)},{self._ass_timestamp(end)},"
                f"Default,,0,0,0,,{{{overrides}}}{content}"
            )

        if not events:
            return None

        ass_path = temp_dir / "fast_texts.ass"
        header = [
            "[Script Info]",
            "ScriptType: v4.00+",
            f"PlayResX: {width}",
            f"PlayResY: {height}",
            "ScaledBorderAndShadow: yes",
            "WrapStyle: 2",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
            "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
            "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            "Style: Default,DejaVu Sans,48,&H00FFFFFF,&H00FFFFFF,&H00000000,&H40000000,"
            "0,0,0,0,100,100,0,0,1,0,0,2,0,0,0,1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
        ass_path.write_text("\n".join([*header, *events, ""]), encoding="utf-8")
        return ass_path, (fonts_dir if fonts_dir.exists() else None)

    def _build_fast_text_overlays(
        self,
        texts: list[TextInstruction],
        target_resolution: tuple[int, int],
        duration: float,
        temp_dir: Path,
    ) -> list[dict[str, object]]:
        overlays: list[dict[str, object]] = []
        for index, text in enumerate(texts):
            end_time = min(float(text.end) if text.end else duration, duration)
            start_time = max(float(text.start or 0.0), 0.0)
            if end_time <= start_time:
                continue
            clip = None
            try:
                clip = self._build_text_clip_with_pillow(text, target_resolution)
                if clip is None:
                    continue
                frame = np.asarray(clip.get_frame(0))
                if np.issubdtype(frame.dtype, np.floating):
                    frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8, copy=False)

                if frame.ndim == 2:
                    frame = np.repeat(frame[:, :, None], 3, axis=2)
                rgb = frame[:, :, :3]
                if frame.ndim == 3 and frame.shape[2] >= 4:
                    alpha = frame[:, :, 3]
                elif getattr(clip, "mask", None) is not None:
                    alpha_frame = np.asarray(clip.mask.get_frame(0))
                    if np.issubdtype(alpha_frame.dtype, np.floating):
                        alpha = np.clip(alpha_frame * 255.0, 0, 255).astype(np.uint8)
                    else:
                        alpha = alpha_frame.astype(np.uint8, copy=False)
                    if alpha.ndim == 3:
                        alpha = alpha[:, :, 0]
                else:
                    alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)

                rgba = np.dstack([rgb, alpha])
                path = temp_dir / f"text_{index:04d}.png"
                Image.fromarray(rgba, mode="RGBA").save(path)
                x, y = self._resolve_fast_overlay_position(
                    text.position,
                    target_resolution,
                    clip.size,
                    bottom_offset_px=text.bottom_offset_px,
                )
                overlays.append(
                    {
                        "path": path,
                        "start": start_time,
                        "end": end_time,
                        "duration": max(end_time - start_time, 0.001),
                        "x": int(x),
                        "y": int(y),
                        "fade_in": max(float(getattr(text.animation, "fade_in", 0.0) or 0.0), 0.0),
                        "fade_out": max(float(getattr(text.animation, "fade_out", 0.0) or 0.0), 0.0),
                    }
                )
            finally:
                if clip is not None:
                    clip.close()
        return overlays

    def _collect_fast_audio_tracks(
        self,
        request: RenderRequest,
        duration: float,
    ) -> list[dict[str, object]]:
        tracks: list[dict[str, object]] = []
        if not request.output.include_audio:
            return tracks

        for instruction in request.audio:
            source_path = self._resolve_media_path(instruction.source)
            if not source_path.exists():
                logger.warning("Audio source not found: %s", source_path)
                continue
            if not self._has_audio_stream(source_path):
                logger.warning("Audio source has no audio stream: %s", source_path)
                continue

            source_duration = self._probe_duration_seconds(source_path)
            source_start = max(float(instruction.start or 0.0), 0.0)
            source_end = float(instruction.end) if instruction.end else source_duration
            source_end = min(source_end, source_duration)
            if source_end <= source_start:
                continue

            timeline_start = instruction.timeline_start
            if timeline_start is None:
                timeline_start = instruction.at
            if timeline_start is None:
                timeline_start = source_start
            timeline_start = max(float(timeline_start or 0.0), 0.0)

            remaining_duration = max(duration - timeline_start, 0.0)
            if remaining_duration <= 0.0:
                continue
            source_end = min(source_end, source_start + remaining_duration)
            if source_end <= source_start:
                continue

            tracks.append(
                {
                    "path": source_path,
                    "source_start": source_start,
                    "source_end": source_end,
                    "timeline_start": timeline_start,
                    "volume": max(float(instruction.volume or 0.0), 0.0),
                    "fade_in": max(float(instruction.fade_in or 0.0), 0.0),
                    "fade_out": max(float(instruction.fade_out or 0.0), 0.0),
                }
            )
        return tracks

    def _collect_fast_insert_audio_tracks(
        self,
        instructions: Iterable[InsertInstruction],
        timeline_duration: float,
    ) -> list[dict[str, object]]:
        tracks: list[dict[str, object]] = []
        previous_end: Optional[float] = None
        for instruction in instructions:
            volume = float(instruction.volume or 0.0)
            if volume <= 0.0:
                continue
            source_path = self._resolve_media_path(instruction.source)
            if not source_path.exists() or not self._has_audio_stream(source_path):
                continue
            source_duration = self._probe_duration_seconds(source_path)
            source_start = max(float(instruction.start or 0.0), 0.0)
            source_end = float(instruction.end) if instruction.end else source_duration
            source_end = min(source_end, source_duration)
            if source_end <= source_start:
                continue
            track_duration = source_end - source_start
            timeline_start = self._resolve_attachment_start(instruction, timeline_duration, track_duration)
            if timeline_start < 0.0:
                source_start += -timeline_start
                timeline_start = 0.0
            remaining_duration = max(timeline_duration - timeline_start, 0.0)
            if remaining_duration <= 0.0:
                continue
            source_end = min(source_end, source_start + remaining_duration)
            if source_end <= source_start:
                continue
            rendered_duration = source_end - source_start
            suppress_intro = timeline_start <= ADJACENT_INSERT_TOLERANCE_SEC
            if previous_end is not None:
                suppress_intro = suppress_intro or abs(timeline_start - previous_end) <= ADJACENT_INSERT_TOLERANCE_SEC
            tracks.append(
                {
                    "path": source_path,
                    "source_start": source_start,
                    "source_end": source_end,
                    "timeline_start": timeline_start,
                    "volume": volume,
                    "fade_in": max(float(instruction.transitions_before[0].duration), 0.0)
                    if instruction.transitions_before and not suppress_intro
                    else 0.0,
                    "fade_out": max(float(instruction.transitions_after[0].duration), 0.0)
                    if instruction.transitions_after
                    else 0.0,
                }
            )
            previous_end = timeline_start + rendered_duration
        return tracks

    @staticmethod
    def _fast_transition_fade_duration(transition: Optional[TransitionInstruction], duration: float) -> float:
        if not transition or transition.type == TransitionType.NONE:
            return 0.0
        return max(min(float(transition.duration or 0.0), max(duration, 0.0)), 0.0)

    def _build_fast_overlay_items(
        self,
        request: RenderRequest,
        reference_duration: float,
    ) -> tuple[list[dict[str, object]], list[TimelineClipModel], float]:
        items: list[dict[str, object]] = []
        timeline_entries: list[TimelineClipModel] = []
        cursor = 0.0

        for index, instruction in enumerate(request.clips):
            base_duration = self._get_instruction_output_duration(instruction)
            previous_transition = None
            if index > 0 and request.clips[index - 1].transitions_after:
                previous_transition = request.clips[index - 1].transitions_after[0]
            previous_duration = max(float(items[-1]["duration"]), 0.0) if items else 0.0
            overlap = self._fast_transition_overlap(previous_transition, previous_duration, base_duration)
            if self._has_explicit_at(instruction):
                start = max(float(instruction.at or 0.0) - overlap, 0.0)
            else:
                start = max(cursor - overlap, 0.0)
            output_duration = max(base_duration + overlap, 0.001)
            intro_fade = self._fast_transition_fade_duration(previous_transition, output_duration)
            outro = instruction.transitions_after[0] if instruction.transitions_after else None
            outro_fade = self._fast_transition_fade_duration(outro, output_duration)
            end = start + output_duration
            item = {
                "kind": "clip",
                "channel_id": 1,
                "index": index,
                "instruction": instruction,
                "start": start,
                "duration": output_duration,
                "transition_padding": overlap,
                "intro_fade": intro_fade,
                "outro_fade": outro_fade,
                "intro_transition": previous_transition,
                "outro_transition": outro,
                "trim_to_reference": False,
            }
            items.append(item)
            cursor = max(cursor, end)
            timeline_entries.append(
                TimelineClipModel(
                    index=index,
                    source=instruction.source,
                    source_label=instruction.source_label,
                    source_type=instruction.source_type,
                    source_resolution=instruction.source_resolution,
                    quality_label=instruction.quality_label,
                    start=round(max(start, 0.0), 1),
                    end=round(max(end, 0.0), 1),
                    auto_placed=self._is_auto_placed_instruction(instruction),
                    channel_id=1,
                    kind="clip",
                )
            )

        timeline_duration = max(reference_duration, cursor)

        for channel_id, kind, instructions in (
            (10, "attachment", request.attachments),
            (11, "insert", request.inserts),
        ):
            insert_cursor = 0.0
            previous_end: Optional[float] = None
            for index, instruction in enumerate(instructions):
                base_duration = self._get_instruction_output_duration(instruction)
                start = self._resolve_attachment_start(instruction, timeline_duration, base_duration)
                if not self._has_explicit_at(instruction) and instruction.placement == InsertPlacement.TIME:
                    start = max(insert_cursor, 0.0)
                source_offset = 0.0
                if start < 0.0:
                    source_offset = -start
                    start = 0.0
                if start >= timeline_duration:
                    continue
                visible_duration = min(base_duration - source_offset, timeline_duration - start)
                if visible_duration <= 0.0:
                    continue
                suppress_intro = start <= ADJACENT_INSERT_TOLERANCE_SEC
                if previous_end is not None:
                    suppress_intro = suppress_intro or abs(start - previous_end) <= ADJACENT_INSERT_TOLERANCE_SEC
                intro = (
                    instruction.transitions_before[0]
                    if instruction.transitions_before and not suppress_intro
                    else None
                )
                outro = instruction.transitions_after[0] if instruction.transitions_after else None
                item = {
                    "kind": kind,
                    "channel_id": channel_id,
                    "index": index,
                    "instruction": instruction,
                    "start": start,
                    "duration": max(visible_duration, 0.001),
                    "transition_padding": 0.0,
                    "intro_fade": self._fast_transition_fade_duration(intro, visible_duration),
                    "outro_fade": self._fast_transition_fade_duration(outro, visible_duration),
                    "intro_transition": intro,
                    "outro_transition": outro,
                    "trim_to_reference": True,
                    "source_offset": source_offset,
                }
                items.append(item)
                previous_end = start + visible_duration
                insert_cursor = max(insert_cursor, previous_end)
                timeline_entries.append(
                    TimelineClipModel(
                        index=index,
                        source=instruction.source,
                        source_label=instruction.source_label,
                        source_type=instruction.source_type,
                        source_resolution=instruction.source_resolution,
                        quality_label=instruction.quality_label,
                        start=round(max(start, 0.0), 1),
                        end=round(max(start + visible_duration, 0.0), 1),
                        auto_placed=self._is_auto_placed_instruction(instruction),
                        channel_id=channel_id,
                        kind=kind,
                    )
                )

        items.sort(key=lambda item: (int(item["channel_id"]), float(item["start"]), int(item["index"])))
        timeline_entries.sort(key=lambda item: (item.channel_id or 1, item.start, item.index))
        return items, timeline_entries, timeline_duration

    def _normalize_fast_overlay_item(
        self,
        item: dict[str, object],
        temp_dir: Path,
        target_resolution: tuple[int, int],
        target_fps: int,
        bitrate: Optional[str],
    ) -> Path:
        instruction = item["instruction"]
        assert isinstance(instruction, ClipInstruction)
        model_copy = getattr(instruction, "model_copy", None)
        normalized_instruction = model_copy(deep=True) if callable(model_copy) else instruction.copy(deep=True)
        transition_padding = max(float(item.get("transition_padding", 0.0) or 0.0), 0.0)
        source_offset = max(float(item.get("source_offset", 0.0) or 0.0), 0.0)
        if source_offset > 0.0:
            normalized_instruction.start = max(float(normalized_instruction.start or 0.0) + source_offset, 0.0)
        duration = max(float(item["duration"]), 0.001)
        if item.get("trim_to_reference"):
            normalized_instruction.end = normalized_instruction.start + duration
        output_path = temp_dir / f"fast_overlay_{int(item['channel_id']):02d}_{int(item['index']):04d}.mp4"
        source_path = self._resolve_media_path(normalized_instruction.source)
        if not source_path.exists():
            raise VideoEngineError(f"Clip source not found: {source_path}")
        self._normalize_clip_for_fast_render(
            instruction=normalized_instruction,
            source_path=source_path,
            output_path=output_path,
            target_resolution=target_resolution,
            target_fps=target_fps,
            bitrate=bitrate,
            transition_padding=transition_padding,
            intro_fade_black=0.0,
            outro_fade_black=0.0,
        )
        return output_path

    def _render_fast_overlay_timeline(
        self,
        request: RenderRequest,
        target_resolution: tuple[int, int],
    ) -> RenderResult:
        output_path = self._resolve_output_path(request)
        extension = output_path.suffix.lower().lstrip(".")
        base_duration = max((self._get_instruction_output_duration(instruction) for instruction in request.clips), default=0.0)
        items, timeline_entries, timeline_duration = self._build_fast_overlay_items(request, base_duration)
        if not items:
            raise VideoEngineError("No renderable timeline items")

        with tempfile.TemporaryDirectory(prefix="ffmpeg_fast_overlay_", dir=str(self.workspace)) as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            max_workers = self._get_positive_int_env("FFMPEG_FAST_RENDER_NORMALIZE_CONCURRENCY", 4)
            max_workers = min(max_workers, max(len(items), 1))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_pos = {
                    executor.submit(
                        self._normalize_fast_overlay_item,
                        item,
                        temp_dir,
                        target_resolution,
                        request.output.fps,
                        request.output.bitrate,
                    ): position
                    for position, item in enumerate(items)
                }
                for future in as_completed(future_to_pos):
                    items[future_to_pos[future]]["path"] = future.result()

            filters: list[str] = []
            target_w, target_h = target_resolution
            main_positions = [
                (position, item)
                for position, item in enumerate(items)
                if int(item["channel_id"]) == 1
            ]
            overlay_positions = [
                (position, item)
                for position, item in enumerate(items)
                if int(item["channel_id"]) != 1
            ]

            if main_positions:
                main_labels: list[tuple[str, dict[str, object]]] = []
                for position, item in main_positions:
                    instruction = item["instruction"]
                    assert isinstance(instruction, ClipInstruction)
                    duration = max(float(item["duration"]), 0.001)
                    label = f"vmain{position}"
                    main_filters = [
                        f"[{position}:v]fps={request.output.fps}",
                        "format=yuv420p",
                        "settb=AVTB",
                    ]
                    instruction_index = int(item["index"])
                    previous_transition = None
                    if instruction_index > 0 and request.clips[instruction_index - 1].transitions_after:
                        previous_transition = request.clips[instruction_index - 1].transitions_after[0]
                    # A whip-pan between main clips is composed as one moving
                    # scene below. Applying blur to each source here would blur
                    # the seam twice and reintroduce the old visible line.
                    if previous_transition and previous_transition.type == TransitionType.MOTION_BLUR:
                        main_filters.extend(
                            self._fast_transition_edge_filters(
                                previous_transition, duration, "intro", request.output.fps
                            )
                        )
                    outro_transition = (
                        instruction.transitions_after[0]
                        if instruction.transitions_after
                        else None
                    )
                    if outro_transition and (
                        outro_transition.type == TransitionType.MOTION_BLUR
                        or instruction_index == len(request.clips) - 1
                    ):
                        main_filters.extend(
                            self._fast_transition_edge_filters(
                                outro_transition, duration, "outro", request.output.fps
                            )
                        )
                    if previous_transition and previous_transition.type == TransitionType.FADE_BLACK:
                        fade_in = min(float(previous_transition.duration or 0.0), duration)
                        if fade_in > 0.0:
                            main_filters.append(f"fade=t=in:st=0:d={fade_in:.6f}")
                    if outro_transition and outro_transition.type == TransitionType.FADE_BLACK:
                        fade_out = min(float(outro_transition.duration or 0.0), duration)
                        if fade_out > 0.0:
                            main_filters.append(
                                f"fade=t=out:st={max(duration - fade_out, 0.0):.6f}:d={fade_out:.6f}"
                            )
                    filters.append(",".join(main_filters) + f"[{label}]")
                    main_labels.append((label, item))

                current_label, first_item = main_labels[0]
                current_duration = max(float(first_item["duration"]), 0.001)
                first_start = max(float(first_item["start"]), 0.0)
                if first_start > 1e-6:
                    filters.append(
                        f"color=c=black:s={target_w}x{target_h}:r={request.output.fps}:"
                        f"d={first_start:.6f}[vgap0]"
                    )
                    filters.append(f"[vgap0][{current_label}]concat=n=2:v=1:a=0[vbase0]")
                    current_label = "vbase0"
                    current_duration += first_start

                for sequence_index, (next_input_label, item) in enumerate(main_labels[1:], start=1):
                    desired_start = max(float(item["start"]), 0.0)
                    next_duration = max(float(item["duration"]), 0.001)
                    next_label = f"vbase{sequence_index}"
                    overlap = max(current_duration - desired_start, 0.0)
                    previous_item = main_labels[sequence_index - 1][1]
                    previous_instruction = previous_item["instruction"]
                    assert isinstance(previous_instruction, ClipInstruction)
                    transition = (
                        previous_instruction.transitions_after[0]
                        if previous_instruction.transitions_after
                        else None
                    )
                    if overlap > 1e-6 and transition and transition.type == TransitionType.WHIP_PAN:
                        filters.extend(
                            self._build_fast_whip_pan_between_filters(
                                previous_label=current_label,
                                next_label=next_input_label,
                                output_label=next_label,
                                sequence_index=sequence_index,
                                transition=transition,
                                previous_duration=current_duration,
                                next_duration=next_duration,
                                overlap=overlap,
                                target_resolution=target_resolution,
                                fps=request.output.fps,
                            )
                        )
                        current_duration = max(current_duration + next_duration - overlap, desired_start + next_duration)
                    elif overlap > 1e-6:
                        overlap = min(overlap, current_duration, next_duration)
                        transition_name = self._fast_xfade_name(transition) if transition else "fade"
                        offset = max(current_duration - overlap, 0.0)
                        filters.append(
                            f"[{current_label}][{next_input_label}]"
                            f"xfade=transition={transition_name}:duration={overlap:.6f}:"
                            f"offset={offset:.6f},format=yuv420p[{next_label}]"
                        )
                        current_duration = max(current_duration + next_duration - overlap, desired_start + next_duration)
                    else:
                        gap = max(desired_start - current_duration, 0.0)
                        if gap > 1e-6:
                            gap_label = f"vgap{sequence_index}"
                            joined_label = f"vjoined{sequence_index}"
                            filters.append(
                                f"color=c=black:s={target_w}x{target_h}:r={request.output.fps}:"
                                f"d={gap:.6f}[{gap_label}]"
                            )
                            filters.append(
                                f"[{current_label}][{gap_label}]concat=n=2:v=1:a=0[{joined_label}]"
                            )
                            current_label = joined_label
                            current_duration += gap
                        filters.append(
                            f"[{current_label}][{next_input_label}]concat=n=2:v=1:a=0[{next_label}]"
                        )
                        current_duration += next_duration
                    current_label = next_label

                tail_duration = max(timeline_duration - current_duration, 0.0)
                if tail_duration > 1e-6:
                    filters.append(
                        f"color=c=black:s={target_w}x{target_h}:r={request.output.fps}:"
                        f"d={tail_duration:.6f}[vmaintail]"
                    )
                    filters.append(f"[{current_label}][vmaintail]concat=n=2:v=1:a=0[vbasefinal]")
                    current_label = "vbasefinal"
            else:
                filters.append(
                    f"color=c=black:s={target_w}x{target_h}:r={request.output.fps}:"
                    f"d={timeline_duration:.6f}[vbase]"
                )
                current_label = "vbase"

            for position, item in overlay_positions:
                instruction = item["instruction"]
                assert isinstance(instruction, ClipInstruction)
                input_label = f"vitem{position}"
                video_filters = [f"[{position}:v]fps={request.output.fps}", "format=rgba"]
                if instruction.chroma_key.enabled:
                    key = instruction.chroma_key
                    key_color = key.color.as_tuple()
                    video_filters.append(
                        "colorkey="
                        f"0x{key_color[0]:02x}{key_color[1]:02x}{key_color[2]:02x}:"
                        f"{key.effective_similarity:.6f}:{key.effective_blend:.6f}"
                    )
                duration = max(float(item["duration"]), 0.001)
                intro_transition = item.get("intro_transition")
                outro_transition = item.get("outro_transition")
                video_filters.extend(
                    self._fast_transition_edge_filters(
                        intro_transition, duration, "intro", request.output.fps
                    )
                )
                video_filters.extend(
                    self._fast_transition_edge_filters(
                        outro_transition, duration, "outro", request.output.fps
                    )
                )
                intro_fade = min(max(float(item.get("intro_fade", 0.0) or 0.0), 0.0), duration)
                outro_fade = min(max(float(item.get("outro_fade", 0.0) or 0.0), 0.0), duration)
                if intro_fade > 0.0 and self._fast_transition_uses_alpha_fade(intro_transition):
                    video_filters.append(f"fade=t=in:st=0:d={intro_fade:.6f}:alpha=1")
                if outro_fade > 0.0 and self._fast_transition_uses_alpha_fade(outro_transition):
                    fade_start = max(duration - outro_fade, 0.0)
                    video_filters.append(f"fade=t=out:st={fade_start:.6f}:d={outro_fade:.6f}:alpha=1")
                start = max(float(item["start"]), 0.0)
                end = min(start + duration, timeline_duration)
                video_filters.append(f"setpts=PTS-STARTPTS+{start:.6f}/TB")
                filters.append(",".join(video_filters) + f"[{input_label}]")
                next_label = f"vover{position}"
                overlay_x, overlay_y = self._fast_overlay_position_expressions(item, target_resolution)
                filters.append(
                    f"[{current_label}][{input_label}]"
                    f"overlay=x='{overlay_x}':y='{overlay_y}':"
                    f"enable='between(t\\,{start:.6f}\\,{end:.6f})':"
                    f"shortest=0:format=auto[{next_label}]"
                )
                current_label = next_label

            ass_assets = self._build_fast_ass_subtitles(
                request.texts,
                target_resolution,
                timeline_duration,
                temp_dir,
            )
            video_input_count = len(items)
            if ass_assets is not None:
                ass_path, ass_fonts_dir = ass_assets
                escaped_ass_path = ass_path.as_posix().replace(":", r"\:").replace("'", r"\'")
                ass_filter = f"ass=filename='{escaped_ass_path}':original_size={target_w}x{target_h}"
                if ass_fonts_dir is not None:
                    escaped_fonts_dir = ass_fonts_dir.as_posix().replace(":", r"\:").replace("'", r"\'")
                    ass_filter += f":fontsdir='{escaped_fonts_dir}'"
                filters.append(f"[{current_label}]{ass_filter}[vtext]")
                current_label = "vtext"

            audio_tracks = self._collect_fast_audio_tracks(request, timeline_duration)
            audio_tracks.extend(self._collect_fast_insert_audio_tracks(request.attachments, timeline_duration))
            audio_tracks.extend(self._collect_fast_insert_audio_tracks(request.inserts, timeline_duration))
            audio_input_start = video_input_count
            audio_labels: list[str] = []
            for audio_index, track in enumerate(audio_tracks):
                input_index = audio_input_start + audio_index
                label = f"aud{audio_index}"
                source_start = float(track["source_start"])
                source_end = float(track["source_end"])
                track_duration = max(source_end - source_start, 0.001)
                audio_filters = [
                    f"atrim=start={source_start:.6f}:end={source_end:.6f}",
                    "asetpts=PTS-STARTPTS",
                    f"volume={float(track['volume']):.6f}",
                ]
                fade_in = min(float(track["fade_in"]), track_duration)
                fade_out = min(float(track["fade_out"]), track_duration)
                if fade_in > 0.0:
                    audio_filters.append(f"afade=t=in:st=0:d={fade_in:.6f}")
                if fade_out > 0.0:
                    fade_start = max(track_duration - fade_out, 0.0)
                    audio_filters.append(f"afade=t=out:st={fade_start:.6f}:d={fade_out:.6f}")
                delay_ms = max(int(round(float(track["timeline_start"]) * 1000.0)), 0)
                if delay_ms > 0:
                    audio_filters.append(f"adelay={delay_ms}:all=1")
                filters.append(f"[{input_index}:a]{','.join(audio_filters)}[{label}]")
                audio_labels.append(label)

            final_audio_label = None
            if len(audio_labels) == 1:
                final_audio_label = "aout"
                filters.append(
                    f"[{audio_labels[0]}]atrim=0:{timeline_duration:.6f},"
                    f"asetpts=PTS-STARTPTS[{final_audio_label}]"
                )
            elif len(audio_labels) > 1:
                final_audio_label = "aout"
                inputs = "".join(f"[{label}]" for label in audio_labels)
                filters.append(
                    f"{inputs}amix=inputs={len(audio_labels)}:duration=longest:dropout_transition=0,"
                    f"atrim=0:{timeline_duration:.6f},asetpts=PTS-STARTPTS[{final_audio_label}]"
                )

            command = [
                "ffmpeg",
                "-y",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                *self._build_ffmpeg_runtime_args(),
            ]
            for item in items:
                command += ["-i", str(item["path"])]
            for track in audio_tracks:
                command += ["-i", str(track["path"])]
            command += [
                "-filter_complex",
                ";".join(filters),
                "-map",
                f"[{current_label}]",
            ]
            if final_audio_label:
                command += ["-map", f"[{final_audio_label}]", *self._build_fast_audio_codec_args()]
            else:
                command += ["-an"]
            command += [
                "-dn",
                "-map_metadata",
                "-1",
                *self._build_fast_video_codec_args(request.output.bitrate, extension),
                "-t",
                f"{timeline_duration:.6f}",
            ]
            if extension in {"mp4", "mov"}:
                command += ["-movflags", "+faststart"]
            staged_output = temp_dir / f"final_output.{extension}"
            command.append(str(staged_output))
            self._run_external_command(command, "FFmpeg fast overlay final render")
            final_duration = self._probe_duration_seconds(staged_output)
            if final_duration <= 0.0:
                raise VideoEngineError("FFmpeg fast overlay final render produced an empty media file")
            os.replace(staged_output, output_path)

        timeline = TimelineDetailModel(
            clips=[entry for entry in timeline_entries if entry.channel_id == 1],
            attachments=[entry for entry in timeline_entries if entry.kind == "attachment"],
            inserts=[entry for entry in timeline_entries if entry.kind == "insert"],
            channels=[TimelineChannelModel(channel_id=channel_id, clips=[entry for entry in timeline_entries if entry.channel_id == channel_id]) for channel_id in sorted({entry.channel_id or 1 for entry in timeline_entries})],
        )
        return RenderResult(status="ok", duration=final_duration, output=output_path, timeline=timeline)

    def _render_fast_linear(
        self,
        request: RenderRequest,
        target_resolution: tuple[int, int],
    ) -> RenderResult:
        output_path = self._resolve_output_path(request)
        extension = output_path.suffix.lower().lstrip(".")
        timeline_entries: list[TimelineClipModel] = []

        with tempfile.TemporaryDirectory(prefix="ffmpeg_fast_render_", dir=str(self.workspace)) as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            normalized_files: list[Path] = []
            normalized_durations: list[float] = []
            base_durations: list[float] = []

            for instruction in request.clips:
                base_durations.append(self._get_instruction_output_duration(instruction))

            for index, instruction in enumerate(request.clips):
                source_path = self._resolve_media_path(instruction.source)
                if not source_path.exists():
                    raise VideoEngineError(f"Clip source not found: {source_path}")

                previous_transition = None
                previous_duration = 0.0
                if index > 0 and request.clips[index - 1].transitions_after:
                    previous_transition = request.clips[index - 1].transitions_after[0]
                    previous_duration = normalized_durations[index - 1]

                transition_padding = self._fast_transition_overlap(
                    previous_transition,
                    previous_duration,
                    base_durations[index],
                )
                intro_fade_black = 0.0
                if previous_transition and previous_transition.type == TransitionType.FADE_BLACK:
                    intro_fade_black = min(float(previous_transition.duration or 0.0), base_durations[index])

                outro_fade_black = 0.0
                if instruction.transitions_after:
                    current_transition = instruction.transitions_after[0]
                    if current_transition.type == TransitionType.FADE_BLACK:
                        outro_fade_black = min(float(current_transition.duration or 0.0), base_durations[index])

                normalized_path = temp_dir / f"fast_norm_{index:04d}.mp4"
                normalized_duration = self._normalize_clip_for_fast_render(
                    instruction=instruction,
                    source_path=source_path,
                    output_path=normalized_path,
                    target_resolution=target_resolution,
                    target_fps=request.output.fps,
                    bitrate=request.output.bitrate,
                    transition_padding=transition_padding,
                    intro_fade_black=intro_fade_black,
                    outro_fade_black=outro_fade_black,
                )
                normalized_files.append(normalized_path)
                normalized_durations.append(normalized_duration)

            filters: list[str] = []
            for index in range(len(normalized_files)):
                filters.append(f"[{index}:v]fps={request.output.fps},settb=AVTB[v{index}]")

            current_label = "v0"
            current_duration = normalized_durations[0]
            timeline_entries.append(
                TimelineClipModel(
                    index=0,
                    source=request.clips[0].source,
                    source_label=request.clips[0].source_label,
                    source_type=request.clips[0].source_type,
                    source_resolution=request.clips[0].source_resolution,
                    quality_label=request.clips[0].quality_label,
                    start=0.0,
                    end=round(max(current_duration, 0.0), 1),
                    auto_placed=self._is_auto_placed_instruction(request.clips[0]),
                    channel_id=1,
                    kind="clip",
                )
            )

            for index in range(1, len(normalized_files)):
                transition = None
                if request.clips[index - 1].transitions_after:
                    transition = request.clips[index - 1].transitions_after[0]

                overlap = self._fast_transition_overlap(
                    transition,
                    current_duration,
                    normalized_durations[index],
                )
                next_label = f"vfast{index}"
                if overlap > 0.0:
                    offset = max(current_duration - overlap, 0.0)
                    xfade_name = self._fast_xfade_name(transition)
                    filters.append(
                        f"[{current_label}][v{index}]"
                        f"xfade=transition={xfade_name}:duration={overlap:.6f}:offset={offset:.6f},"
                        f"format=yuv420p[{next_label}]"
                    )
                    clip_start = offset
                    current_duration = max(current_duration + normalized_durations[index] - overlap, 0.0)
                else:
                    filters.append(f"[{current_label}][v{index}]concat=n=2:v=1:a=0[{next_label}]")
                    clip_start = current_duration
                    current_duration = max(current_duration + normalized_durations[index], 0.0)

                current_label = next_label
                timeline_entries.append(
                    TimelineClipModel(
                        index=index,
                        source=request.clips[index].source,
                        source_label=request.clips[index].source_label,
                        source_type=request.clips[index].source_type,
                        source_resolution=request.clips[index].source_resolution,
                        quality_label=request.clips[index].quality_label,
                        start=round(max(clip_start, 0.0), 1),
                        end=round(max(clip_start + normalized_durations[index], 0.0), 1),
                        auto_placed=self._is_auto_placed_instruction(request.clips[index]),
                        channel_id=1,
                        kind="clip",
                    )
                )

            text_overlays = self._build_fast_text_overlays(
                request.texts,
                target_resolution,
                current_duration,
                temp_dir,
            )
            final_video_label = current_label
            video_input_count = len(normalized_files)

            for text_index, overlay in enumerate(text_overlays):
                input_index = video_input_count + text_index
                text_label = f"text{text_index}"
                fade_filters = ["format=rgba"]
                text_duration = float(overlay["duration"])
                fade_in = min(float(overlay["fade_in"]), text_duration)
                fade_out = min(float(overlay["fade_out"]), text_duration)
                if fade_in > 0.0:
                    fade_filters.append(f"fade=t=in:st=0:d={fade_in:.6f}:alpha=1")
                if fade_out > 0.0:
                    fade_start = max(text_duration - fade_out, 0.0)
                    fade_filters.append(f"fade=t=out:st={fade_start:.6f}:d={fade_out:.6f}:alpha=1")
                fade_filters.append(f"setpts=PTS-STARTPTS+{float(overlay['start']):.6f}/TB")
                filters.append(f"[{input_index}:v]{','.join(fade_filters)}[{text_label}]")

                next_label = f"vtext{text_index}"
                filters.append(
                    f"[{final_video_label}][{text_label}]"
                    f"overlay=x={int(overlay['x'])}:y={int(overlay['y'])}:"
                    f"enable='between(t\\,{float(overlay['start']):.6f}\\,{float(overlay['end']):.6f})':"
                    f"format=auto[{next_label}]"
                )
                final_video_label = next_label

            audio_tracks = self._collect_fast_audio_tracks(request, current_duration)
            audio_input_start = video_input_count + len(text_overlays)
            audio_labels: list[str] = []
            for audio_index, track in enumerate(audio_tracks):
                input_index = audio_input_start + audio_index
                label = f"aud{audio_index}"
                source_start = float(track["source_start"])
                source_end = float(track["source_end"])
                track_duration = max(source_end - source_start, 0.001)
                audio_filters = [
                    f"atrim=start={source_start:.6f}:end={source_end:.6f}",
                    "asetpts=PTS-STARTPTS",
                    f"volume={float(track['volume']):.6f}",
                ]
                fade_in = min(float(track["fade_in"]), track_duration)
                fade_out = min(float(track["fade_out"]), track_duration)
                if fade_in > 0.0:
                    audio_filters.append(f"afade=t=in:st=0:d={fade_in:.6f}")
                if fade_out > 0.0:
                    fade_start = max(track_duration - fade_out, 0.0)
                    audio_filters.append(f"afade=t=out:st={fade_start:.6f}:d={fade_out:.6f}")
                delay_ms = max(int(round(float(track["timeline_start"]) * 1000.0)), 0)
                if delay_ms > 0:
                    audio_filters.append(f"adelay={delay_ms}:all=1")
                filters.append(f"[{input_index}:a]{','.join(audio_filters)}[{label}]")
                audio_labels.append(label)

            final_audio_label = None
            if len(audio_labels) == 1:
                final_audio_label = "aout"
                filters.append(
                    f"[{audio_labels[0]}]atrim=0:{current_duration:.6f},"
                    f"asetpts=PTS-STARTPTS[{final_audio_label}]"
                )
            elif len(audio_labels) > 1:
                final_audio_label = "aout"
                inputs = "".join(f"[{label}]" for label in audio_labels)
                filters.append(
                    f"{inputs}amix=inputs={len(audio_labels)}:duration=longest:dropout_transition=0,"
                    f"atrim=0:{current_duration:.6f},asetpts=PTS-STARTPTS[{final_audio_label}]"
                )

            command = [
                "ffmpeg",
                "-y",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                *self._build_ffmpeg_runtime_args(),
            ]
            for normalized_file in normalized_files:
                command += ["-i", str(normalized_file)]
            for overlay in text_overlays:
                command += ["-loop", "1", "-t", f"{float(overlay['duration']):.6f}", "-i", str(overlay["path"])]
            for track in audio_tracks:
                command += ["-i", str(track["path"])]

            command += [
                "-filter_complex",
                ";".join(filters),
                "-map",
                f"[{final_video_label}]",
            ]
            if final_audio_label:
                command += ["-map", f"[{final_audio_label}]", *self._build_fast_audio_codec_args()]
            else:
                command += ["-an"]
            command += [
                "-dn",
                "-map_metadata",
                "-1",
                *self._build_fast_video_codec_args(request.output.bitrate, extension),
                "-t",
                f"{current_duration:.6f}",
            ]
            if extension in {"mp4", "mov"}:
                command += ["-movflags", "+faststart"]
            command.append(str(output_path))

            self._run_external_command(command, "FFmpeg fast final render")

        final_duration = self._probe_duration_seconds(output_path)
        timeline = TimelineDetailModel(clips=timeline_entries, channels=[
            TimelineChannelModel(channel_id=1, clips=timeline_entries)
        ])
        return RenderResult(status="ok", duration=final_duration, output=output_path, timeline=timeline)

    def _normalize_clip_for_concat(
        self,
        instruction: ClipInstruction,
        source_path: Path,
        output_path: Path,
        target_resolution: tuple[int, int],
        target_fps: int,
        bitrate: Optional[str],
        include_audio: bool,
        zoom_border: Optional[ZoomBorderInstruction] = None,
    ) -> None:
        target_w, target_h = target_resolution
        start = max(instruction.start, 0.0)
        end = instruction.end if instruction.end else None
        if end is not None and end <= start:
            end = None

        vf_parts = []
        if instruction.mirror_horizontal:
            vf_parts.append("hflip")
        zoom_filter = self._build_center_zoom_filter(instruction.effective_internal_zoom)
        if zoom_filter:
            vf_parts.append(zoom_filter)
        vf_parts.append(f"fps={target_fps}")
        if instruction.fit_mode == FitMode.COVER:
            vf_parts.extend(
                [
                    f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase:force_divisible_by=2",
                    f"crop={target_w}:{target_h}:(iw-ow)/2:(ih-oh)/2",
                    "setsar=1",
                ]
            )
        else:
            vf_parts.extend(
                [
                    f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease:force_divisible_by=2",
                    f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black",
                    "setsar=1",
                ]
            )
        zoom_border_filter = self._build_zoom_border_filter(zoom_border, target_resolution)
        if zoom_border_filter:
            vf_parts.append(zoom_border_filter)
        vf_parts.append("format=yuv420p")
        vf_chain = ",".join(vf_parts)
        source_has_audio = include_audio and self._has_audio_stream(source_path)
        command = [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            *self._build_ffmpeg_runtime_args(),
            "-i",
            str(source_path),
        ]
        if include_audio and not source_has_audio:
            command += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
        if start > 0:
            command += ["-ss", f"{start:.6f}"]
        if end is not None:
            duration = max(end - start, 0.0)
            if duration > 0:
                command += ["-t", f"{duration:.6f}"]
        command += [
            "-map",
            "0:v:0",
        ]
        if include_audio:
            command += [
                "-map",
                "0:a:0" if source_has_audio else "1:a:0",
            ]
        command += [
            "-dn",
            "-map_metadata",
            "-1",
            "-vf",
            vf_chain,
            *self._build_fast_video_codec_args(bitrate, output_path.suffix.lower().lstrip(".")),
        ]
        if include_audio:
            command += self._build_fast_audio_codec_args()
        else:
            command += ["-an"]
        command += [
            "-movflags",
            "+faststart",
            "-shortest",
            str(output_path),
        ]
        self._run_external_command(command, f"FFmpeg normalize clip {source_path.name}")

    def _render_concat_normalize(self, request: RenderRequest) -> RenderResult:
        template = resolve_template(request.output.template)
        target_resolution = request.output.resolution.size if request.output.resolution else template.resolution
        output_path = self._resolve_output_path(request)
        extension = output_path.suffix.lower().lstrip(".")
        if extension not in {"mp4", "mov", "mkv"}:
            return RenderResult(
                status="error",
                duration=0.0,
                output=Path(""),
                message="concat_normalize supports output.format: mp4, mov, mkv",
            )
        if not request.clips:
            return RenderResult(status="error", duration=0.0, output=Path(""), message="No clips provided in the request")

        logger.info("Using concat_normalize mode with target resolution %s and fps %s", target_resolution, request.output.fps)
        try:
            timeline_entries: List[TimelineClipModel] = []
            cursor = 0.0
            with tempfile.TemporaryDirectory(prefix="ffmpeg_concat_", dir=str(self.workspace)) as temp_dir_name:
                temp_dir = Path(temp_dir_name)
                normalized_files: List[Path] = []
                for index, instruction in enumerate(request.clips):
                    source_path = self._resolve_media_path(instruction.source)
                    if not source_path.exists():
                        raise VideoEngineError(f"Clip source not found: {source_path}")
                    normalized_path = temp_dir / f"norm_{index:04d}.mp4"
                    self._normalize_clip_for_concat(
                        instruction=instruction,
                        source_path=source_path,
                        output_path=normalized_path,
                        target_resolution=target_resolution,
                        target_fps=request.output.fps,
                        bitrate=request.output.bitrate,
                        include_audio=request.output.include_audio,
                        zoom_border=request.zoom_border,
                    )
                    clip_duration = self._probe_duration_seconds(normalized_path)
                    clip_start = cursor
                    clip_end = clip_start + clip_duration
                    timeline_entries.append(
                        TimelineClipModel(
                            index=index,
                            source=instruction.source,
                            source_label=instruction.source_label,
                            source_type=instruction.source_type,
                            source_resolution=instruction.source_resolution,
                            quality_label=instruction.quality_label,
                            start=round(max(clip_start, 0.0), 1),
                            end=round(max(clip_end, 0.0), 1),
                            auto_placed=self._is_auto_placed_instruction(instruction),
                        )
                    )
                    cursor = clip_end
                    normalized_files.append(normalized_path)

                concat_file = temp_dir / "list.ffconcat"
                with concat_file.open("w", encoding="utf-8") as handle:
                    handle.write("ffconcat version 1.0\n")
                    for normalized_file in normalized_files:
                        escaped = normalized_file.resolve().as_posix().replace("'", r"'\''")
                        handle.write(f"file '{escaped}'\n")

                concat_copy_command = [
                    "ffmpeg",
                    "-y",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    *self._build_ffmpeg_runtime_args(),
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_file),
                    "-map",
                    "0:v:0",
                    "-dn",
                    "-map_metadata",
                    "-1",
                    "-c",
                    "copy",
                ]
                concat_has_audio = (
                    request.output.include_audio
                    and bool(normalized_files)
                    and self._has_audio_stream(normalized_files[0])
                )
                if concat_has_audio:
                    concat_copy_command += [
                        "-map",
                        "0:a:0?",
                    ]
                if extension in {"mp4", "mov"}:
                    concat_copy_command += ["-movflags", "+faststart"]
                concat_copy_command.append(str(output_path))

                try:
                    self._run_external_command(concat_copy_command, "FFmpeg concat (stream copy)")
                except VideoEngineError as exc:
                    logger.warning("Concat stream copy failed; fallback to concat re-encode. Reason: %s", exc)
                    concat_encode_command = [
                        "ffmpeg",
                        "-y",
                        "-nostdin",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        *self._build_ffmpeg_runtime_args(),
                        "-f",
                        "concat",
                        "-safe",
                        "0",
                        "-i",
                        str(concat_file),
                        "-map",
                        "0:v:0",
                        "-dn",
                        "-map_metadata",
                        "-1",
                        *self._build_fast_video_codec_args(request.output.bitrate, extension),
                    ]
                    if concat_has_audio:
                        concat_encode_command += [
                            "-map",
                            "0:a:0?",
                            *self._build_fast_audio_codec_args(),
                        ]
                    else:
                        concat_encode_command += ["-an"]
                    if extension in {"mp4", "mov"}:
                        concat_encode_command += ["-movflags", "+faststart"]
                    concat_encode_command.append(str(output_path))
                    self._run_external_command(concat_encode_command, "FFmpeg concat fallback encode")

            timeline = TimelineDetailModel(clips=timeline_entries)
            final_duration = self._probe_duration_seconds(output_path)
            if request.texts or request.show_source:
                self._apply_concat_text_overlays(output_path, request, target_resolution, final_duration, timeline)
                final_duration = self._probe_duration_seconds(output_path)
            return RenderResult(status="ok", duration=final_duration, output=output_path, timeline=timeline)
        except Exception as exc:
            logger.exception("concat_normalize failed: %s", exc)
            return RenderResult(status="error", duration=0.0, output=Path(""), message=str(exc))

    def _apply_concat_text_overlays(
        self,
        output_path: Path,
        request: RenderRequest,
        target_resolution: tuple[int, int],
        duration: float,
        timeline: Optional[TimelineDetailModel] = None,
    ) -> None:
        overlays: list[mpe.VideoClip] = []
        for text in request.texts:
            clip = self._build_text_clip(text, duration, target_resolution)
            if clip:
                overlays.append(clip)
        overlays.extend(self._build_show_source_clips(request.show_source, timeline, duration, target_resolution))
        if not overlays:
            return

        source_clip = mpe.VideoFileClip(str(output_path))
        final_clip: mpe.VideoClip | None = None
        temp_output = output_path.with_name(f"{output_path.stem}.text_overlay.tmp{output_path.suffix}")
        try:
            final_clip = mpe.CompositeVideoClip([source_clip, *overlays], size=target_resolution).set_duration(duration)
            if source_clip.audio is not None:
                final_clip = final_clip.set_audio(source_clip.audio)

            model_copy = getattr(request, "model_copy", None)
            overlay_request = model_copy(deep=True) if callable(model_copy) else request.copy(deep=True)
            overlay_request.output.filename = temp_output.name
            self._export(final_clip, overlay_request)
        finally:
            if final_clip is not None:
                final_clip.close()
            source_clip.close()
            for overlay in overlays:
                overlay.close()

        os.replace(temp_output, output_path)

    def _export(self, clip: mpe.VideoClip, request: RenderRequest) -> Path:
        output_path = self._resolve_output_path(request)
        extension = request.output.format.lower().lstrip(".")
        thread_count = self._get_thread_count("FFMPEG_THREADS")
        filter_thread_count = self._get_thread_count("FFMPEG_FILTER_THREADS", thread_count)

        preset: Optional[str] = None
        audio_fps: Optional[int] = None
        audio_bitrate: Optional[str] = None
        if extension in {"mp4", "mov"}:
            codec, audio_codec, preset, audio_fps, audio_bitrate, ffmpeg_params = self._build_moviepy_mp4_export_settings(
                request.output.bitrate,
                filter_thread_count,
            )
        else:
            codec = None
            audio_codec = None
            ffmpeg_params = ["-filter_threads", str(filter_thread_count)]

        write_kwargs = {
            "fps": request.output.fps,
            "codec": codec,
            "audio_codec": audio_codec,
            "bitrate": request.output.bitrate,
            "threads": thread_count,
            "ffmpeg_params": ffmpeg_params,
        }
        if preset:
            write_kwargs["preset"] = preset
        if audio_fps:
            write_kwargs["audio_fps"] = audio_fps
        if audio_bitrate:
            write_kwargs["audio_bitrate"] = audio_bitrate

        # Suppress MoviePy progress bars in API mode; worker processes return structured status instead.
        write_kwargs["logger"] = None
        clip.write_videofile(str(output_path), **write_kwargs)
        return output_path

    @staticmethod
    def _resolve_position(
        position: str,
        target_resolution: tuple[int, int],
        clip_size: tuple[int, int],
        bottom_offset_px: int | None = None,
    ):
        w, h = target_resolution
        cw, ch = clip_size
        edge_margin_x = 0.05 * w
        edge_margin_y = 0.05 * h
        bottom_margin = edge_margin_y if bottom_offset_px is None else max(int(bottom_offset_px), 0)
        mapping = {
            "center": ("center", "center"),
            "top": ("center", edge_margin_y),
            "bottom": ("center", h - ch - bottom_margin),
            "left": (edge_margin_x, "center"),
            "right": (w - cw - edge_margin_x, "center"),
            "top_left": (edge_margin_x, edge_margin_y),
            "top_right": (w - cw - edge_margin_x, edge_margin_y),
            "bottom_left": (edge_margin_x, h - ch - bottom_margin),
            "bottom_right": (w - cw - edge_margin_x, h - ch - bottom_margin),
        }
        return mapping.get(position, ("center", "center"))

    @staticmethod
    def _color_to_hex(color) -> str:
        if not color:
            return "white"
        return "#%02x%02x%02x" % (color.r, color.g, color.b)
