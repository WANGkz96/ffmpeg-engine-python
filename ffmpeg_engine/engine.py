"""Core rendering engine built on top of moviepy/ffmpeg."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import moviepy.editor as mpe
from moviepy.video.fx import all as vfx

import moviepy.editor as mpe
from moviepy.video.fx import all as vfx
from moviepy.video.compositing import transitions as transfx

from moviepy.config import change_settings

change_settings({
    "IMAGEMAGICK_BINARY": "magick",
    "FFMPEG_BINARY": "ffmpeg"
})

if not hasattr(vfx, "gaussian_blur"):
    import numpy as np
    from PIL import Image, ImageFilter

    def gaussian_blur(clip, sigma: float = 5.0):
        """
        Простой Gaussian blur поверх всего кадра.
        Чтобы работало: clip.fx(vfx.gaussian_blur, sigma=...)
        """
        def _blur_frame(frame):
            # frame: np.ndarray (H, W, 3)
            img = Image.fromarray(frame)
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
            return np.array(img)

        return clip.fl_image(_blur_frame)

    # подмешиваем в пространство эффектов MoviePy
    vfx.gaussian_blur = gaussian_blur

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

logger = logging.getLogger(__name__)


class VideoEngineError(Exception):
    """Raised when the rendering pipeline fails."""


class VideoEngine:
    def __init__(self, workspace: Path | str = "renders") -> None:
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)

    def _apply_whip_pan_intro(
            self,
            clip: mpe.VideoClip,
            transition: TransitionInstruction,
    ) -> mpe.VideoClip:
        """Whip pan в начале клипа: залетает с края + размытие."""
        duration = min(transition.duration, clip.duration)
        if duration <= 0:
            return clip

        side = transition.direction.value  # "left" / "right" / "top" / "bottom"

        # Делим клип: первая часть с эффектом, остальное как есть
        head = clip.subclip(0, duration)
        rest = clip.subclip(duration) if clip.duration > duration else None

        head = head.fx(transfx.slide_in, duration, side)
        # Добавим размытие для ощущения "whip"
        head = head.fx(vfx.gaussian_blur, sigma=8)

        if rest:
            return mpe.concatenate_videoclips([head, rest])
        return head

    def _apply_whip_pan_outro(
            self,
            clip: mpe.VideoClip,
            transition: TransitionInstruction,
    ) -> mpe.VideoClip:
        """Whip pan в конце клипа: вылетает за край + размытие."""
        duration = min(transition.duration, clip.duration)
        if duration <= 0:
            return clip

        side = transition.direction.value
        total = clip.duration

        body = clip.subclip(0, total - duration) if total > duration else None
        tail = clip.subclip(max(total - duration, 0), total)

        tail = tail.fx(transfx.slide_out, duration, side)
        tail = tail.fx(vfx.gaussian_blur, sigma=8)

        if body:
            return mpe.concatenate_videoclips([body, tail])
        return tail

    def _apply_whip_pan_between(
            self,
            prev_clip: mpe.VideoClip,
            next_clip: mpe.VideoClip,
            transition: TransitionInstruction,
    ) -> tuple[mpe.VideoClip, mpe.VideoClip, float]:
        """
        Whip pan между двумя клипами.
        Возвращает (модифицированный_prev, модифицированный_next, overlap_duration).
        """
        d = min(
            transition.duration,
            prev_clip.duration,
            next_clip.duration,
        )
        if d <= 0:
            return prev_clip, next_clip, 0.0

        side = transition.direction.value

        # Хвост предыдущего клипа: вылетает + blur
        total_prev = prev_clip.duration
        body_prev = prev_clip.subclip(0, total_prev - d) if total_prev > d else None
        tail_prev = prev_clip.subclip(max(total_prev - d, 0), total_prev)
        tail_prev = tail_prev.fx(transfx.slide_out, d, side)
        tail_prev = tail_prev.fx(vfx.gaussian_blur, sigma=8)
        new_prev = (
            mpe.concatenate_videoclips([body_prev, tail_prev]) if body_prev else tail_prev
        )

        # Начало следующего клипа: влетает + blur
        total_next = next_clip.duration
        head_next = next_clip.subclip(0, d)
        head_next = head_next.fx(transfx.slide_in, d, side)
        head_next = head_next.fx(vfx.gaussian_blur, sigma=8)
        rest_next = next_clip.subclip(d) if total_next > d else None
        new_next = (
            mpe.concatenate_videoclips([head_next, rest_next]) if rest_next else head_next
        )

        return new_prev, new_next, d

    def render(self, request: RenderRequest) -> RenderResult:
        template = resolve_template(request.output.template)
        target_resolution = request.output.resolution.size if request.output.resolution else template.resolution
        logger.info("Using target resolution %s", target_resolution)

        clips = []
        final_clip = None
        transitions: List[TransitionInstruction] = []
        try:
            for clip_instruction in request.clips:
                clip = self._prepare_clip(clip_instruction, target_resolution, request.output.fps)
                clips.append(clip)
                transitions.extend(clip_instruction.transitions_after or [])

            if not clips:
                raise VideoEngineError("No clips provided in the request")

            video = self._concatenate_with_transitions(clips, request.clips, target_resolution)
            overlay = self._build_overlay(video.duration, request, target_resolution)
            final_clip = mpe.CompositeVideoClip([video, *overlay], size=target_resolution)

            audio = self._build_audio(request, final_clip.duration)
            if audio is not None:
                final_clip = final_clip.set_audio(audio)

            output_path = self._export(final_clip, request)
        except Exception as exc:  # pragma: no cover - best effort error shield
            logger.exception("Rendering failed: %s", exc)
            return RenderResult(status="error", duration=0.0, output=Path(""), message=str(exc))
        finally:
            for clip in clips:
                clip.close()
            if final_clip:
                try:
                    final_clip.close()
                except Exception:  # pragma: no cover
                    pass

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
        if instruction.background_mode == BackgroundMode.COLOR:
            background = mpe.ColorClip(size=target_resolution, color=instruction.background_color.as_tuple(), duration=resized.duration)
        else:
            background = clip.resize(max(target_w / clip_w, target_h / clip_h)).fx(
                vfx.gaussian_blur, sigma=max(instruction.background_color.a * 25, 5)
            )
            background = background.crop(
                width=target_w, height=target_h, x_center=background.w / 2, y_center=background.h / 2
            )
            background = background.set_duration(resized.duration)
        positioned = resized.set_position(("center", "center"))
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
            except Exception:  # pragma: no cover - depends on moviepy build
                logger.warning("Saturation adjustment failed; skipping")
        if adjustments.hue != 0:
            try:
                clip = clip.fx(vfx.hue, adjustments.hue)
            except Exception:  # pragma: no cover
                logger.warning("Hue adjustment failed; skipping")
        return clip

    def _apply_chroma_key(self, clip: mpe.VideoClip, instruction: ClipInstruction) -> mpe.VideoClip:
        key = instruction.chroma_key
        try:
            mask = clip.fx(vfx.mask_color, color=key.color.as_tuple(), thr=key.threshold, s=key.softness)
            return clip.set_mask(mask.mask)
        except Exception:  # pragma: no cover
            logger.warning("Chroma key failed; returning original clip")
            return clip

    def _concatenate_with_transitions(
            self,
            clips: List[mpe.VideoClip],
            clip_instructions: List[ClipInstruction],
            target_resolution: tuple[int, int],
    ) -> mpe.VideoClip:
        scheduled: List[tuple[mpe.VideoClip, float]] = []
        cursor = 0.0

        for index, clip in enumerate(clips):
            instr = clip_instructions[index]

            # ---------- ВХОДНЫЕ ПЕРЕХОДЫ (ДЛЯ САМОГО КЛИПА) ----------
            intro: Optional[TransitionInstruction] = None
            if instr.transitions_before:
                intro = instr.transitions_before[0]

            if intro:
                if intro.type == TransitionType.FADE_BLACK:
                    d = min(intro.duration, clip.duration)
                    clip = clip.fadein(d)
                elif intro.type == TransitionType.WHIP_PAN:
                    clip = self._apply_whip_pan_intro(clip, intro)

            # ---------- ПЕРЕХОД ИЗ ПРЕДЫДУЩЕГО КЛИПА В ЭТОТ ----------
            transition: Optional[TransitionInstruction] = None
            if index > 0:
                prev_instr = clip_instructions[index - 1]
                if prev_instr.transitions_after:
                    transition = prev_instr.transitions_after[0]

            if transition and transition.type == TransitionType.CROSSFADE:
                duration = min(transition.duration, clip.duration / 2)
                clip = clip.crossfadein(duration)
                start = max(cursor - duration, 0.0)
                cursor = start + clip.duration

            elif transition and transition.type == TransitionType.FADE_BLACK:
                duration = min(transition.duration, clip.duration)
                if scheduled:
                    prev_clip, prev_start = scheduled[-1]
                    scheduled[-1] = (prev_clip.fadeout(duration), prev_start)
                clip = clip.fadein(duration)
                start = cursor
                cursor += clip.duration

            elif transition and transition.type == TransitionType.WHIP_PAN:
                # Whip pan между предыдущим клипом и текущим
                if scheduled:
                    prev_clip, prev_start = scheduled[-1]
                    new_prev, new_clip, overlap = self._apply_whip_pan_between(
                        prev_clip, clip, transition
                    )
                    scheduled[-1] = (new_prev, prev_start)
                    clip = new_clip
                    start = max(cursor - overlap, 0.0)
                    cursor = start + clip.duration
                else:
                    # На всякий случай fallback, если почему-то нет предыдущего
                    start = cursor
                    cursor += clip.duration

            else:
                # Без перехода с предыдущим
                start = cursor
                cursor += clip.duration

            scheduled.append((clip, start))

        # ---------- АУТРО ДЛЯ ПОСЛЕДНЕГО КЛИПА ----------
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
                    last_clip = self._apply_whip_pan_outro(last_clip, outro)
                    scheduled[-1] = (last_clip, last_start)

        layered = [clip.set_start(start) for clip, start in scheduled]
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

        # выбираем, что передавать в параметр font
        if getattr(instruction, "font_path", None):
            font_arg = str(instruction.font_path)
        else:
            font_arg = instruction.font

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
        except Exception as exc:  # pragma: no cover - requires ImageMagick
            logger.warning("Text rendering failed (%s); skipping", exc)
            return None

        anim = instruction.animation

        # --- ЗУМ ПРИ ПОЯВЛЕНИИ ---
        # Делаем zoom от scale_from до scale_to в течение zoom_duration секунд (локальное время клипа t=0..)
        if (getattr(anim, "scale_from", 1.0) != 1.0) or (getattr(anim, "scale_to", 1.0) != 1.0):
            scale_from = getattr(anim, "scale_from", 1.0)
            scale_to = getattr(anim, "scale_to", 1.0)

            # Длительность зума: zoom_duration > fade_in > вся длина клипа
            zoom_duration = (
                anim.zoom_duration
                if anim.zoom_duration is not None
                else (anim.fade_in if anim.fade_in > 0 else (end_time - instruction.start))
            )

            if zoom_duration > 0:
                def scale_func(t):
                    # t — локальное время клипа, начиная с 0
                    progress = min(max(t / zoom_duration, 0.0), 1.0)
                    return scale_from + (scale_to - scale_from) * progress

                clip = clip.resize(scale_func)
            else:
                # если по какой-то причине длительность 0 — просто ставим финальный масштаб
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
        except OSError as exc:  # pragma: no cover - file missing etc.
            logger.warning("Unable to load image %s: %s", instruction.source, exc)
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
        """Return a make_frame function with rudimentary keyframe animation."""

        def make_frame(t):  # pragma: no cover - complex animation path
            frame = clip.get_frame(t)
            for keyframe in keyframes:
                start = keyframe.get("time", 0.0)
                duration = keyframe.get("duration", 0.0)
                if not (start <= t <= start + duration):
                    continue
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
            except OSError as exc:  # pragma: no cover
                logger.warning("Failed to load audio %s: %s", instruction.source, exc)
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

        # например, грубый флажок “использовать GPU”
        use_gpu = True

        if use_gpu and extension in {"mp4", "mov"}:
            codec = "h264_nvenc"
            audio_codec = "aac"
            ffmpeg_params = ["-preset", "p4"]  # подбирается по вкусу
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
        mapping = {
            "center": ("center", "center"),
            "top": ("center", 0.05 * target_resolution[1]),
            "bottom": ("center", target_resolution[1] - clip_size[1] - 0.05 * target_resolution[1]),
            "left": (0.05 * target_resolution[0], "center"),
            "right": (target_resolution[0] - clip_size[0] - 0.05 * target_resolution[0], "center"),
            "top_left": (0.05 * target_resolution[0], 0.05 * target_resolution[1]),
            "top_right": (target_resolution[0] - clip_size[0] - 0.05 * target_resolution[0], 0.05 * target_resolution[1]),
            "bottom_left": (0.05 * target_resolution[0], target_resolution[1] - clip_size[1] - 0.05 * target_resolution[1]),
            "bottom_right": (
                target_resolution[0] - clip_size[0] - 0.05 * target_resolution[0],
                target_resolution[1] - clip_size[1] - 0.05 * target_resolution[1],
            ),
        }
        return mapping.get(position, ("center", "center"))

    @staticmethod
    def _color_to_hex(color) -> str:
        if not color:
            return "white"
        return "#%02x%02x%02x" % (color.r, color.g, color.b)

