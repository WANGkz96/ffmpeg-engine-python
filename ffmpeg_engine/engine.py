"""Core rendering engine built on top of moviepy/ffmpeg."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

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
from PIL import ImageFilter

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
    def _is_windows_absolute(path_value: str) -> bool:
        return len(path_value) > 2 and path_value[1] == ":" and path_value[2] in {"\\", "/"}

    def _resolve_media_path(self, source: Path) -> Path:
        source_text = str(source).strip()
        path_obj = Path(source_text).expanduser()
        if path_obj.is_absolute() or self._is_windows_absolute(source_text):
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
        template = resolve_template(request.output.template)
        target_resolution = request.output.resolution.size if request.output.resolution else template.resolution
        logger.info("Using target resolution %s", target_resolution)

        clips = []
        final_clip = None
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
            for clip in clips:
                # Аккуратно закрываем ресурсы
                try:
                    if clip: clip.close()
                except: pass
            if final_clip:
                try:
                    final_clip.close()
                except: pass

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

        clip = self._apply_fit_mode(clip, instruction, target_resolution)
        clip = self._apply_adjustments(clip, instruction.adjustments)
        if instruction.chroma_key.enabled:
            clip = self._apply_chroma_key(clip, instruction)

        if instruction.volume != 1.0 and clip.audio is not None:
            clip = clip.volumex(instruction.volume)

        clip = clip.set_fps(fps)
        return clip

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
                    source=self._resolve_media_path(instruction.source),
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
            logger.warning("Text rendering failed (%s); skipping", exc)
            return None

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

        if anim.fade_in and anim.fade_in > 0:
            clip = clip.fadein(anim.fade_in)
        if anim.fade_out and anim.fade_out > 0:
            clip = clip.fadeout(anim.fade_out)

        clip = clip.set_start(instruction.start).set_end(end_time)
        clip = clip.set_position(self._resolve_position(instruction.position, target_resolution, clip.size))

        return clip

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

    def _export(self, clip: mpe.VideoClip, request: RenderRequest) -> Path:
        output_dir = self.workspace
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = Path(request.output.filename).name
        extension = request.output.format.lower().lstrip(".")
        if not filename:
            filename = "render"
        if Path(filename).suffix.lower() != f".{extension}":
            filename = f"{Path(filename).stem}.{extension}"
        output_path = (output_dir / filename).resolve()

        use_gpu = os.getenv("FFMPEG_USE_GPU", "0").lower() in {"1", "true", "yes", "on"}
        gpu_preset = os.getenv("FFMPEG_GPU_PRESET", "p4")
        if use_gpu and extension in {"mp4", "mov"}:
            codec = "h264_nvenc"
            audio_codec = "aac"
            ffmpeg_params = ["-preset", gpu_preset]
        else:
            codec = "libx264" if extension in {"mp4", "mov"} else None
            audio_codec = "aac" if extension in {"mp4", "mov"} else None
            ffmpeg_params = []

        clip.write_videofile(
            str(output_path),
            fps=request.output.fps,
            codec=codec,
            audio_codec=audio_codec,
            bitrate=request.output.bitrate,
            threads=4,
            ffmpeg_params=ffmpeg_params,
        )
        return output_path

    @staticmethod
    def _resolve_position(position: str, target_resolution: tuple[int, int], clip_size: tuple[int, int]):
        w, h = target_resolution
        cw, ch = clip_size
        mapping = {
            "center": ("center", "center"),
            "top": ("center", 0.05 * h),
            "bottom": ("center", h - ch - 0.05 * h),
            "left": (0.05 * w, "center"),
            "right": (w - cw - 0.05 * w, "center"),
            "top_left": (0.05 * w, 0.05 * h),
            "top_right": (w - cw - 0.05 * w, 0.05 * h),
            "bottom_left": (0.05 * w, h - ch - 0.05 * h),
            "bottom_right": (w - cw - 0.05 * w, h - ch - 0.05 * h),
        }
        return mapping.get(position, ("center", "center"))

    @staticmethod
    def _color_to_hex(color) -> str:
        if not color:
            return "white"
        return "#%02x%02x%02x" % (color.r, color.g, color.b)
