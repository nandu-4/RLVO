"""
Adaptive frame sampling for Video RLVO (Python side).

Reads the SAME shared/video-sampling.json as src/lib/videoSampling.ts. The frame counts used to be
duplicated — SUMMARY_FRAMES/TIMECAPSULE_FRAMES here, separate literals in the React page — so
changing one silently left the other behind. There is now one place to change them.

Consistency with the browser
----------------------------
The browser seeks by TIME (`duration`), this module selects by FRAME INDEX (`CAP_PROP_FRAME_COUNT`).
Both sample the same fractional positions i/(n-1) across the video, so for constant-framerate
material they land on the same moments. They can differ on variable-framerate recordings, where a
frame index is not proportional to elapsed time — an inherent property of the two APIs, not a
disagreement about policy.
"""

import json
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "shared" / "video-sampling.json"

with _CONFIG_PATH.open(encoding="utf-8") as _f:
    _CONFIG = json.load(_f)

TIERS = _CONFIG["tiers"]
END_EPSILON_SECONDS = float(_CONFIG["endEpsilonSeconds"])

MODES = tuple(TIERS.keys())


def frame_count_for(mode: str, duration_seconds: float) -> int:
    """Frames to extract for a video of this length, per the shared tiers."""
    tiers = TIERS.get(mode)
    if not tiers:
        raise ValueError(f'No sampling tiers configured for mode "{mode}". Known: {", ".join(MODES)}')

    # An unknown or zero duration falls back to the shortest tier: fewer frames still produces a
    # usable result, where raising would lose the whole request.
    if not duration_seconds or duration_seconds <= 0:
        return int(tiers[0]["frames"])

    for tier in tiers:
        cap = tier["maxSeconds"]
        if cap is None or duration_seconds <= cap:
            return int(tier["frames"])
    return int(tiers[-1]["frames"])


def frame_indices_for(total_frames: int, count: int) -> list:
    """
    Evenly spaced frame indices, first at 0 and last at the final frame.

    Dividing by `count - 1` rather than `count` is what makes the last sample reach the end of the
    video. The previous `i * (total // count)` stepping stopped short, so the closing portion of
    every video went unsampled.
    """
    if count <= 0 or total_frames <= 0:
        return []
    last = total_frames - 1
    if count == 1:
        return [0]
    return [min(round(i * last / (count - 1)), last) for i in range(count)]


def sample_timestamps(duration_seconds: float, count: int) -> list:
    """Time-based equivalent of frame_indices_for — mirrors the browser exactly."""
    if count <= 0:
        return []
    duration = duration_seconds if duration_seconds and duration_seconds > 0 else 0.0
    end = max(0.0, duration - END_EPSILON_SECONDS)
    if count == 1:
        return [0.0]
    return [min(i * duration / (count - 1), end) for i in range(count)]
