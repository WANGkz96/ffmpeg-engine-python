"""Core rendering engine built on top of moviepy/ffmpeg."""
from __future__ import annotations

import gc
import logging
import math
import os
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
from PIL import ImageChops, ImageDraw, ImageFilter, ImageFont

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

def dynamic_motion_blur(clip, strength_func, direction: str = "horizontal"):
    """
    Motion blur с динамической силой (зависит от времени).
    strength_func(t) -> float (размер ядра, px)
    direction: "horizontal" | "vertical"
    """
    def filter_frame(get_frame, t):
        frame = get_frame(t)
        k_size = int(strength_func(t))
        
        if k_size < 2:
            return frame
            
        # Force odd kernel size for symmetry
        if k_size % 2 == 0:
            k_size += 1
            
        radius = k_size // 2
        
        # Convert to float to avoid overflow during accumulation
        img_float = frame.astype(float)
        
        if direction == "horizontal":
            axis = 1
            pad_width = ((0, 0), (radius, radius), (0, 0))
        else:
            axis = 0
            pad_width = ((radius, radius), (0, 0), (0, 0))
            
        try:
            # Pad with edge replication
            padded = np.pad(img_float, pad_width, mode='edge')
            
            # Cumsum along axis
            cumsum = np.cumsum(padded, axis=axis)
            
            # Pad cumsum with one zero slice at the beginning of the axis
            if axis == 1:
                zeros = np.zeros((cumsum.shape[0], 1, cumsum.shape[2]))
                cumsum_padded = np.hstack((zeros, cumsum))
            else:
                zeros = np.zeros((1, cumsum.shape[1], cumsum.shape[2]))
                cumsum_padded = np.vstack((zeros, cumsum))
                
            # Compute moving sum: sum[i] = cumsum[i+k] - cumsum[i]
            if axis == 1:
                upper = cumsum_padded[:, k_size : k_size + img_float.shape[1], :]
                lower = cumsum_padded[:, 0 : img_float.shape[1], :]
            else:
                upper = cumsum_padded[k_size : k_size + img_float.shape[0], :, :]
                lower = cumsum_padded[0 : img_float.shape[0], :, :]
                
            result = (upper - lower) / k_size
            
            return np.clip(result, 0, 255).astype(np.uint8)
            
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
    ProcessingMode,
    RenderRequest,
    RenderResult,
    TimelineClipModel,
    TimelineDetailModel,
    TextInstruction,
    TransitionInstruction,
    TransitionType,
    TransitionDirection,
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
    def _get_timeout_seconds(env_name: str, default: float) -> Optional[float]:
        raw_value = os.getenv(env_name, "").strip()
        if not raw_value:
            return default
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
    def _is_auto_placed_instruction(instruction: ClipInstruction) -> bool:
        fields_set = getattr(instruction, "model_fields_set", None)
        if fields_set is None:
            fields_set = getattr(instruction, "__fields_set__", set())
        return "start" not in fields_set and "end" not in fields_set

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
            resolution: Tuple[int, int]
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

        # Создаем "черный клип" как предыдущий
        prev_clip = mpe.ColorClip(size=resolution, color=(0,0,0), duration=duration)

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

        prev_clip = prev_clip.set_position(pos_prev)
        head = head.set_position(pos_head)

        transition_clip = mpe.CompositeVideoClip([prev_clip, head], size=resolution).set_duration(duration)

        max_blur = transition.blur_strength
        blur_direction = "horizontal" if abs(dx) > abs(dy) else "vertical"

        def blur_func(t):
            p = get_progress(t)
            # Blur убывает от максимума к 0
            # Derivative of ease_out_cubic (1 - (1-p)^3) is 3(1-p)^2
            # Normalized: (1-p)^2
            velocity_factor = (1 - p) ** 2
            return max_blur * velocity_factor

        transition_blurred = dynamic_motion_blur(transition_clip, blur_func, direction=blur_direction)

        clips = [transition_blurred]
        if rest:
            clips.append(rest.set_start(duration))
        
        return mpe.CompositeVideoClip(clips, size=resolution).set_duration(clip.duration)

    def _apply_whip_pan_outro(
            self,
            clip: mpe.VideoClip,
            transition: TransitionInstruction,
            resolution: Tuple[int, int]
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

        # Создаем "черный клип" как следующий
        next_clip = mpe.ColorClip(size=resolution, color=(0,0,0), duration=duration)

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
        next_clip = next_clip.set_position(pos_next)

        transition_clip = mpe.CompositeVideoClip([tail, next_clip], size=resolution).set_duration(duration)

        max_blur = transition.blur_strength
        blur_direction = "horizontal" if abs(dx) > abs(dy) else "vertical"

        def blur_func(t):
            p = get_progress(t)
            # Blur нарастает от 0 к максимуму
            # Derivative of ease_in_cubic (p^3) is 3p^2
            # Normalized: p^2
            velocity_factor = p ** 2
            return max_blur * velocity_factor

        transition_blurred = dynamic_motion_blur(transition_clip, blur_func, direction=blur_direction)

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

        clips = []
        final_clip = None
        video = None
        audio = None
        overlay: List[mpe.VideoClip] = []
        timeline = TimelineDetailModel(clips=[])
        try:
            # Подготовка всех клипов (ресайз, эффекты, хромакей)
            for clip_instruction in request.clips:
                clip = self._prepare_clip(clip_instruction, target_resolution, request.output.fps)
                clips.append(clip)

            if not clips:
                raise VideoEngineError("No clips provided in the request")

            # Сборка видео с переходами
            video, timeline_clips = self._concatenate_with_transitions(clips, request.clips, target_resolution)
            timeline = TimelineDetailModel(clips=timeline_clips)

            # Наложение текста и картинок
            overlay = self._build_overlay(video.duration, request, target_resolution)

            # Финальный композит
            final_clip = mpe.CompositeVideoClip([video, *overlay], size=target_resolution)

            # Аудио
            audio = self._build_audio(request, final_clip.duration)
            if audio is not None:
                final_clip = final_clip.set_audio(audio)

            output_path = self._export(final_clip, request)
        except Exception as exc:
            logger.exception("Rendering failed: %s", exc)
            return RenderResult(status="error", duration=0.0, output=Path(""), message=str(exc))
        finally:
            self._close_clip_resources([final_clip, audio, video, *overlay, *clips])
            gc.collect()

        return RenderResult(status="ok", duration=final_clip.duration, output=output_path, timeline=timeline)

    def _prepare_clip(self, instruction: ClipInstruction, target_resolution: tuple[int, int], fps: int) -> mpe.VideoClip:
        source_path = self._resolve_media_path(instruction.source)
        if not source_path.exists():
            raise VideoEngineError(f"Clip source not found: {source_path}")

        clip = mpe.VideoFileClip(str(source_path))
        start = max(instruction.start, 0.0)
        end = instruction.end if instruction.end else None
        if end is not None:
            end = min(end, clip.duration)
        if end and end <= start:
            end = None

        if end:
            clip = clip.subclip(start, end)
        else:
            clip = clip.subclip(start)

        if instruction.playback_rate != 1.0:
            clip = clip.fx(vfx.speedx, instruction.playback_rate)

        clip = self._apply_internal_zoom(clip, instruction)
        clip = self._apply_fit_mode(clip, instruction, target_resolution)
        clip = self._apply_adjustments(clip, instruction.adjustments)
        if instruction.chroma_key.enabled:
            clip = self._apply_chroma_key(clip, instruction)

        if instruction.volume != 1.0 and clip.audio is not None:
            clip = clip.volumex(instruction.volume)

        clip = clip.set_fps(fps)
        return clip

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

    def _apply_fit_mode(self, clip: mpe.VideoClip, instruction: ClipInstruction, target_resolution: tuple[int, int]) -> mpe.VideoClip:
        target_w, target_h = target_resolution
        clip_w, clip_h = clip.size

        if instruction.fit_mode == FitMode.COVER:
            scale = max(target_w / clip_w, target_h / clip_h)
            resized = clip.resize(scale)
            return resized.crop(width=target_w, height=target_h, x_center=resized.w / 2, y_center=resized.h / 2)

        # contain
        scale = min(target_w / clip_w, target_h / clip_h)
        resized = clip.resize(scale)

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

    def _apply_chroma_key(self, clip: mpe.VideoClip, instruction: ClipInstruction) -> mpe.VideoClip:
        key = instruction.chroma_key
        try:
            mask = clip.fx(vfx.mask_color, color=key.color.as_tuple(), thr=key.threshold, s=key.softness)
            return clip.set_mask(mask.mask)
        except Exception:
            return clip

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
                    start=round(max(scheduled_start, 0.0), 1),
                    end=round(max(clip_end, 0.0), 1),
                    auto_placed=self._is_auto_placed_instruction(instruction),
                )
            )
        return base, timeline

    def _build_overlay(
            self, duration: float, request: RenderRequest, target_resolution: tuple[int, int]
    ) -> List[mpe.VideoClip]:
        overlays: List[mpe.VideoClip] = []
        for text in request.texts:
            clip = self._build_text_clip(text, duration, target_resolution)
            if clip:
                overlays.append(clip)
        for image in request.images:
            clip = self._build_image_clip(image, duration, target_resolution)
            if clip:
                overlays.append(clip)
        return overlays

    @staticmethod
    def _color_to_rgba(color, default: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        if not color:
            return default
        alpha = int(max(min(getattr(color, "a", 1.0), 1.0), 0.0) * 255)
        return (int(color.r), int(color.g), int(color.b), alpha)

    def _resolve_pillow_font(self, instruction: TextInstruction) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        font_candidates: List[str] = []
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
            alias = font_aliases.get(font_name)
            if alias:
                font_candidates.append(alias)
            lowered = font_name.lower()
            if "comic sans" in lowered or lowered == "comicsans":
                font_candidates.extend([str(comic_ms_bold), str(comic_ms_regular), str(comic_bold), str(comic_regular)])

        for candidate in font_candidates:
            if not candidate:
                continue
            try:
                return ImageFont.truetype(candidate, instruction.font_size)
            except Exception:
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
            0,
            line_spacing,
        )

        scale_from = getattr(instruction.animation, "scale_from", 1.0) if instruction.animation else 1.0
        scale_to = getattr(instruction.animation, "scale_to", 1.0) if instruction.animation else 1.0
        scale_guard = max(scale_from, scale_to, 1.0)
        scale_margin = max(0, int(max(text_w, text_h) * (scale_guard - 1.0) * 0.55))

        pad_x = max(16, int(instruction.font_size * 0.32) + stroke_width * 2 + scale_margin)
        pad_y = max(14, int(instruction.font_size * 0.34) + stroke_width * 2 + scale_margin)
        img_w = max(1, text_w + pad_x * 2)
        img_h = max(1, text_h + pad_y * 2)

        image = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        text_color = self._color_to_rgba(instruction.color, (255, 255, 255, 255))
        stroke_color = self._color_to_rgba(instruction.stroke_color, (0, 0, 0, 255))
        text_origin = (
            int((img_w - text_w) / 2 - text_bbox[0]),
            int((img_h - text_h) / 2 - text_bbox[1]),
        )

        if instruction.glow:
            glow_layer = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
            glow_draw = ImageDraw.Draw(glow_layer)
            glow_color = (
                min(text_color[0] + 20, 255),
                min(text_color[1] + 20, 255),
                min(text_color[2] + 20, 255),
                max(120, int(text_color[3] * 0.55)),
            )
            glow_draw.multiline_text(
                text_origin,
                text_block,
                font=font,
                fill=glow_color,
                align="center",
                spacing=line_spacing,
                stroke_width=stroke_width + 2,
                stroke_fill=glow_color,
            )
            blur_radius = max(2, int(instruction.font_size * 0.12))
            image.alpha_composite(glow_layer.filter(ImageFilter.GaussianBlur(radius=blur_radius)))

        if instruction.shadow:
            shadow_alpha = max(90, int(text_color[3] * 0.6))
            draw.multiline_text(
                (text_origin[0] + 3, text_origin[1] + 3),
                text_block,
                font=font,
                fill=(0, 0, 0, shadow_alpha),
                align="center",
                spacing=line_spacing,
                stroke_width=0,
            )

        if stroke_width > 0 and stroke_color[3] > 0:
            fill_mask = Image.new("L", (img_w, img_h), 0)
            fill_draw = ImageDraw.Draw(fill_mask)
            fill_draw.multiline_text(
                text_origin,
                text_block,
                font=font,
                fill=255,
                align="center",
                spacing=line_spacing,
                stroke_width=0,
            )
            kernel_size = max(3, stroke_width * 2 + 1)
            max_kernel = max(3, min(img_w, img_h))
            if max_kernel % 2 == 0:
                max_kernel -= 1
            kernel_size = min(kernel_size, max_kernel)
            if kernel_size % 2 == 0:
                kernel_size = max(3, kernel_size - 1)
            dilated_mask = fill_mask.filter(ImageFilter.MaxFilter(size=kernel_size))
            outer_stroke_mask = ImageChops.subtract(dilated_mask, fill_mask)
            stroke_layer = Image.new("RGBA", (img_w, img_h), stroke_color)
            image.paste(stroke_layer, (0, 0), outer_stroke_mask)

        draw.multiline_text(
            text_origin,
            text_block,
            font=font,
            fill=text_color,
            align="center",
            spacing=line_spacing,
            stroke_width=0,
        )

        frame = np.array(image)
        return mpe.ImageClip(frame, transparent=True)

    def _build_text_clip(self, instruction: TextInstruction, duration: float, target_resolution: tuple[int, int]) -> Optional[mpe.VideoClip]:
        end_time = instruction.end if instruction.end else duration
        if end_time <= instruction.start:
            return None

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
        except Exception as exc:
            logger.warning("TextClip rendering failed (%s); trying Pillow fallback", exc)
            try:
                clip = self._build_text_clip_with_pillow(instruction, target_resolution)
            except Exception as fallback_exc:
                logger.warning("Pillow text rendering failed (%s); skipping", fallback_exc)
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
            try:
                clip = mpe.AudioFileClip(str(source_path))
            except OSError as exc:
                logger.warning("Audio loading failed (%s): %s", source_path, exc)
                continue
            start = max(instruction.start, 0.0)
            end = instruction.end if instruction.end else clip.duration
            end = min(end, clip.duration)
            if end <= start:
                clip.close()
                continue
            clip = clip.subclip(start, end)
            clip = clip.volumex(instruction.volume)
            if instruction.fade_in:
                clip = clip.audio_fadein(instruction.fade_in)
            if instruction.fade_out:
                clip = clip.audio_fadeout(instruction.fade_out)
            clip = clip.set_start(start)
            tracks.append(clip)

        if not tracks:
            return None
        return mpe.CompositeAudioClip(tracks).set_duration(duration)

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
        return ["-threads", str(thread_count), "-filter_threads", str(filter_thread_count)]

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
        default_timeout_seconds: float = 900.0,
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
        stderr_tail = (process.stderr or process.stdout or "").strip().splitlines()[-10:]
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

    def _normalize_clip_for_concat(
        self,
        instruction: ClipInstruction,
        source_path: Path,
        output_path: Path,
        target_resolution: tuple[int, int],
        target_fps: int,
        bitrate: Optional[str],
        include_audio: bool,
    ) -> None:
        target_w, target_h = target_resolution
        start = max(instruction.start, 0.0)
        end = instruction.end if instruction.end else None
        if end is not None and end <= start:
            end = None

        vf_parts = []
        zoom_filter = self._build_center_zoom_filter(instruction.effective_internal_zoom)
        if zoom_filter:
            vf_parts.append(zoom_filter)
        vf_parts.extend(
            [
                f"fps={target_fps}",
                f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease:force_divisible_by=2",
                f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black",
                "setsar=1",
                "format=yuv420p",
            ]
        )
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
                    )
                    clip_duration = self._probe_duration_seconds(normalized_path)
                    clip_start = cursor
                    clip_end = clip_start + clip_duration
                    timeline_entries.append(
                        TimelineClipModel(
                            index=index,
                            source=instruction.source,
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
                    "-map",
                    "0:a:0",
                    "-dn",
                    "-map_metadata",
                    "-1",
                    "-c",
                    "copy",
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
                        "-map",
                        "0:a:0",
                        "-dn",
                        "-map_metadata",
                        "-1",
                        *self._build_fast_video_codec_args(request.output.bitrate, extension),
                        *self._build_fast_audio_codec_args(),
                    ]
                    if extension in {"mp4", "mov"}:
                        concat_encode_command += ["-movflags", "+faststart"]
                    concat_encode_command.append(str(output_path))
                    self._run_external_command(concat_encode_command, "FFmpeg concat fallback encode")

            final_duration = self._probe_duration_seconds(output_path)
            timeline = TimelineDetailModel(clips=timeline_entries)
            return RenderResult(status="ok", duration=final_duration, output=output_path, timeline=timeline)
        except Exception as exc:
            logger.exception("concat_normalize failed: %s", exc)
            return RenderResult(status="error", duration=0.0, output=Path(""), message=str(exc))

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
