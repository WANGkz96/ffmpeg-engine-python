import logging
from pathlib import Path
import sys
import os

# Add current directory to sys.path
sys.path.append(os.getcwd())

from ffmpeg_engine.engine import VideoEngine
from ffmpeg_engine.models import RenderRequest, ClipInstruction, TransitionInstruction, TransitionType, TransitionDirection, OutputInstruction, ResolutionModel

logging.basicConfig(level=logging.INFO)

engine = VideoEngine()

# Ensure media files exist
file1 = Path("media/14705086_1080_1920_30fps.mp4").absolute()
file2 = Path("media/5512609-hd_1080_1920_25fps.mp4").absolute()

if not file1.exists() or not file2.exists():
    print("Media files not found!")
    sys.exit(1)

req = RenderRequest(
    clips=[
        ClipInstruction(
            source=file1,
            start=0,
            end=3,
            transitions_before=[
                TransitionInstruction(
                    type=TransitionType.MOTION_BLUR,
                    duration=1.0,
                    blur_strength=100,
                    fade=True  # Intro: Blur + Fade In
                )
            ],
            transitions_after=[
                TransitionInstruction(
                    type=TransitionType.MOTION_BLUR,
                    duration=1.0,
                    blur_strength=100,
                    glow=True # Between: Blur + Glow
                )
            ]
        ),
        ClipInstruction(
            source=file2,
            start=0,
            end=3,
            transitions_after=[
                TransitionInstruction(
                    type=TransitionType.MOTION_BLUR,
                    duration=1.0,
                    blur_strength=100,
                    fade=True # Outro: Blur + Fade Out
                )
            ]
        )
    ],
    output=OutputInstruction(
        filename="motion_blur_test.mp4",
        format="mp4",
        fps=30,
        resolution=ResolutionModel(width=1080, height=1920)
    )
)

print("Starting render...")
result = engine.render(req)
print(f"Result: {result.status}, Output: {result.output}, Duration: {result.duration}")
