"""Color classification for Woody Sort ball detection.

Classifies ball colors using HSV color-space analysis.  Each colour is
defined by one or more HSV ranges (to handle hue wrapping, e.g. red).
Classification works by scoring each colour based on the fraction of
pixels in a sampled region that fall within its range(s).
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

# ── Supported colour names ───────────────────────────────────────────────────
COLOR_NAMES = [
    "Red",
    "Orange",
    "Yellow",
    "Green",
    "Cyan",
    "Blue",
    "Purple",
    "Pink",
    "Violet",
    "Brown",
    "White",
    "Black",
]

# ── HSV ranges for each colour ───────────────────────────────────────────────
# OpenCV HSV:  H 0-179,  S 0-255,  V 0-255.
# Each colour maps to a list of ``(lower_hsv, upper_hsv)`` tuples.
# ``cv2.inRange`` treats both bounds as **inclusive**.
HSV_RANGES: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "Red": [
        ((0, 100, 60), (6, 255, 255)),  # low-hue reds
        ((170, 100, 60), (179, 255, 255)),  # high-hue reds (wrap-around)
    ],
    "Orange": [
        ((7, 120, 130), (16, 255, 255)),
    ],
    "Yellow": [
        ((17, 80, 140), (36, 255, 255)),
    ],
    "Green": [
        ((37, 60, 60), (77, 255, 255)),
    ],
    "Cyan": [
        ((78, 60, 100), (95, 255, 255)),
    ],
    "Blue": [
        ((96, 60, 60), (126, 255, 255)),
    ],
    "Purple": [
        ((127, 40, 40), (138, 255, 255)),
    ],
    "Pink": [
        ((139, 30, 100), (155, 255, 255)),
    ],
    "Violet": [
        ((156, 30, 100), (169, 255, 255)),
    ],
    "Brown": [
        ((8, 60, 30), (24, 200, 130)),
    ],
    "White": [
        ((0, 0, 190), (179, 35, 255)),
    ],
    "Black": [
        ((0, 0, 0), (179, 60, 40)),
    ],
}

# ── Empty-slot detection thresholds ──────────────────────────────────────────
# A slot is considered empty when *both* the median saturation and median
# value are below these thresholds (dark, unsaturated background).
EMPTY_MAX_SATURATION = 50
EMPTY_MAX_VALUE = 80

# Minimum fraction of pixels that must match a colour for it to be valid.
# Below this, the slot is treated as empty.
MIN_MATCH_FRACTION = 0.15


def classify_color(region_bgr: np.ndarray) -> Optional[str]:
    """Classify the dominant colour of a BGR image region.

    Parameters
    ----------
    region_bgr : np.ndarray
        A small BGR patch sampled from the centre of a ball slot.

    Returns
    -------
    str or None
        The colour name (e.g. ``"Red"``, ``"Green"``) or ``None`` if the
        slot is empty.
    """
    if region_bgr.size == 0:
        return None

    region_hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)

    # Quick empty-slot check: dark *and* desaturated → background
    median_s = float(np.median(region_hsv[:, :, 1]))
    median_v = float(np.median(region_hsv[:, :, 2]))

    if median_s < EMPTY_MAX_SATURATION and median_v < EMPTY_MAX_VALUE:
        return None

    # Score each colour by the fraction of pixels within its HSV range(s)
    total_pixels = region_hsv.shape[0] * region_hsv.shape[1]
    best_color: Optional[str] = None
    best_score: float = 0.0

    for color_name, ranges in HSV_RANGES.items():
        combined_mask = np.zeros(region_hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            lower_arr = np.array(lower, dtype=np.uint8)
            upper_arr = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(region_hsv, lower_arr, upper_arr)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        score = float(np.count_nonzero(combined_mask)) / total_pixels
        if score > best_score:
            best_score = score
            best_color = color_name

    if best_score < MIN_MATCH_FRACTION:
        return None

    return best_color
