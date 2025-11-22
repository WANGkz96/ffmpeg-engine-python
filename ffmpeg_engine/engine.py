"""Core rendering engine built on top of moviepy/ffmpeg."""
from __future__ import annotations

import logging
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
        
        # Debug print to verify blur is working
        # logger.info(f"Blurring with k_size={k_size} at t={t}")
        
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

from .models import (
    AdjustmentInstruction,
    AudioInstruction,
    BackgroundMode,
    ClipInstruction,
    FitMode,
    ImageInstruction,
    RenderRequest,
    RenderResult,
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
        """Whip pan в начале: влетает в кадр."""
        duration = min(transition.duration, clip.duration)
        if duration <= 0:
            return clip

        w, h = resolution
        dx, dy = self._get_slide_vector(transition.direction, w, h)

        # Делим клип
        head = clip.subclip(0, duration)
        rest = clip.subclip(duration) if clip.duration > duration else None

        # Анимация влета: от (-dx, -dy) к (0, 0)
        # Deceleration: замедляемся при входе
        def intro_pos(t):
            progress = t / duration
            # ease-out cubic: 1 - (1-p)^3
            p_eased = 1 - (1 - progress) ** 3
            return (
                int(-dx * (1 - p_eased)),
                int(-dy * (1 - p_eased))
            )

        # Динамический Motion Blur: от сильного к 0
        max_blur = 300 
        blur_direction = "horizontal" if abs(dx) > abs(dy) else "vertical"
        
        def blur_func(t):
            progress = t / duration
            factor = (1 - progress) ** 2
            return max_blur * factor

        head = dynamic_motion_blur(head, blur_func, direction=blur_direction).set_position(intro_pos)

        clips = [head]
        if rest:
            clips.append(rest.set_start(duration))
        
        return mpe.CompositeVideoClip(clips, size=resolution).set_duration(clip.duration)

    def _apply_whip_pan_outro(
            self,
            clip: mpe.VideoClip,
            transition: TransitionInstruction,
            resolution: Tuple[int, int]
    ) -> mpe.VideoClip:
        """Whip pan в конце: улетает из кадра."""
        duration = min(transition.duration, clip.duration)
        if duration <= 0:
            return clip

        w, h = resolution
        dx, dy = self._get_slide_vector(transition.direction, w, h)
        total = clip.duration

        body = clip.subclip(0, total - duration) if total > duration else None
        tail = clip.subclip(max(total - duration, 0), total)

        # Анимация вылета: от (0,0) к (dx, dy)
        # Acceleration: разгоняемся
        def outro_pos(t):
            progress = t / duration
            # ease-in cubic: p^3
            p_eased = progress ** 3
            return (int(dx * p_eased), int(dy * p_eased))

        # Динамический Motion Blur: от 0 к сильному
        max_blur = 300
        blur_direction = "horizontal" if abs(dx) > abs(dy) else "vertical"
        
        def blur_func(t):
            progress = t / duration
            factor = progress ** 2
            return max_blur * factor

        tail = dynamic_motion_blur(tail, blur_func, direction=blur_direction).set_position(outro_pos)

        clips = []
        if body:
            clips.append(body)
            tail = tail.set_start(body.duration)
        
        clips.append(tail)
        return mpe.CompositeVideoClip(clips, size=resolution).set_duration(clip.duration)

    def _apply_whip_pan_between(
            self,
            prev_clip: mpe.VideoClip,
            next_clip: mpe.VideoClip,
            transition: TransitionInstruction,
            resolution: Tuple[int, int]
    ) -> tuple[mpe.VideoClip, mpe.VideoClip, float]:
        """
        Whip pan между двумя клипами.
        Возвращает (модифицированный_prev, модифицированный_next, overlap_duration).
        """
        d = min(transition.duration, prev_clip.duration, next_clip.duration)
        if d <= 0:
            return prev_clip, next_clip, 0.0

        w, h = resolution
        dx, dy = self._get_slide_vector(transition.direction, w, h)
        max_blur = 300
        blur_direction = "horizontal" if abs(dx) > abs(dy) else "vertical"

        # --- PREV CLIP (Уходит) ---
        total_prev = prev_clip.duration
        body_prev = prev_clip.subclip(0, total_prev - d) if total_prev > d else None
        tail_prev = prev_clip.subclip(max(total_prev - d, 0), total_prev)

        # Движение от (0,0) к (dx, dy) (Acceleration)
        def pos_out(t):
            progress = min(t / d, 1.0)
            p_eased = progress ** 3
            return (int(dx * p_eased), int(dy * p_eased))

        # Блюр нарастает
        def blur_out(t):
            progress = min(t / d, 1.0)
            return max_blur * (progress ** 2)

        tail_prev = dynamic_motion_blur(tail_prev, blur_out, direction=blur_direction).set_position(pos_out)

        prev_parts = []
        if body_prev:
            prev_parts.append(body_prev)
            tail_prev = tail_prev.set_start(body_prev.duration)
        prev_parts.append(tail_prev)
        
        new_prev = mpe.CompositeVideoClip(prev_parts, size=resolution).set_duration(prev_clip.duration)

        # --- NEXT CLIP (Приходит) ---
        total_next = next_clip.duration
        head_next = next_clip.subclip(0, d)
        rest_next = next_clip.subclip(d) if total_next > d else None

        # Движение от (-dx, -dy) к (0,0) (Deceleration)
        def pos_in(t):
            progress = min(t / d, 1.0)
            p_eased = 1 - (1 - progress) ** 3
            start_x, start_y = -dx, -dy
            return (
                int(start_x * (1 - p_eased)),
                int(start_y * (1 - p_eased))
            )

        # Блюр убывает
        def blur_in(t):
            progress = min(t / d, 1.0)
            return max_blur * ((1 - progress) ** 2)

        head_next = dynamic_motion_blur(head_next, blur_in, direction=blur_direction).set_position(pos_in)

        next_parts = [head_next]
        if rest_next:
            next_parts.append(rest_next.set_start(d))
        
        new_next = mpe.CompositeVideoClip(next_parts, size=resolution).set_duration(next_clip.duration)

        return new_prev, new_next, d

    # --- MAIN RENDER LOGIC ---

    def render(self, request: RenderRequest) -> RenderResult:
        template = resolve_template(request.output.template)
        target_resolution = request.output.resolution.size if request.output.resolution else template.resolution
        logger.info("Using target resolution %s", target_resolution)

        clips = []
        final_clip = None
        try:
            # Подготовка всех клипов (ресайз, эффекты, хромакей)
            for clip_instruction in request.clips:
                clip = self._prepare_clip(clip_instruction, target_resolution, request.output.fps)
                clips.append(clip)

            if not clips:
                raise VideoEngineError("No clips provided in the request")

            # Сборка видео с переходами
            video = self._concatenate_with_transitions(clips, request.clips, target_resolution)

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

        return RenderResult(status="ok", duration=final_clip.duration, output=output_path)

    def _prepare_clip(self, instruction: ClipInstruction, target_resolution: tuple[int, int], fps: int) -> mpe.VideoClip:
        clip = mpe.VideoFileClip(str(instruction.source))
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
    ) -> mpe.VideoClip:
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
                else:
                    # Если это первый клип (странно, но бывает), просто ставим
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

        # Собираем финальный композит
        # set_start устанавливает время начала клипа на глобальном таймлайне
        layered = [c.set_start(s) for c, s in scheduled]

        # size=target_resolution важен, чтобы canvas не скакал
        base = mpe.CompositeVideoClip(layered, size=target_resolution)
        base = base.set_duration(cursor)
        return base

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
        try:
            clip = mpe.ImageClip(str(instruction.source))
        except OSError:
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
            try:
                clip = mpe.AudioFileClip(str(instruction.source))
            except OSError:
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
        filename = request.output.filename
        extension = request.output.format
        if not filename.endswith(f".{extension}"):
            filename = f"{Path(filename).stem}.{extension}"
        output_path = output_dir / filename

        use_gpu = False  # Set True if NVENC available
        if use_gpu and extension in {"mp4", "mov"}:
            codec = "h264_nvenc"
            audio_codec = "aac"
            ffmpeg_params = ["-preset", "p4"]
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