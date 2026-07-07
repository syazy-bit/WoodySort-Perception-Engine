"""Tube and ball detection for Woody Sort game screenshots.

Detection strategy
------------------
1.  Find tubes by masking their distinctive cyan or gray outlines in HSV.
2.  Filter contours by aspect ratio (tall, narrow) to eliminate UI elements.
3.  Merge overlapping bounding boxes (NMS-style).
4.  Sort tubes into rows (top → bottom) and columns (left → right).
5.  Sample equally-spaced vertical slots inside each tube.
6.  Classify each slot's colour using the pixel-counting HSV analyser.

No hardcoded pixel coordinates.  Works for any tube count and screen size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from colors import HSV_RANGES, MIN_MATCH_FRACTION, classify_color

# Ball colors only (excludes tube-outline colors if they conflict, but Cyan is a valid ball color)
BALL_COLORS = {"Red", "Orange", "Yellow", "Green", "Cyan", "Purple", "Pink", "Violet", "Blue"}

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Tube:
    """A detected tube on the game board."""

    bbox: Tuple[int, int, int, int]  # (x, y, w, h) bounding box
    center: Tuple[int, int]  # (cx, cy) centre point
    row: int  # 0-indexed row (top = 0)
    col: int  # 0-indexed column within row
    index: int  # 0-indexed sequential order

# ─────────────────────────────────────────────────────────────────────────────
# Tube-outline detection constants
# ─────────────────────────────────────────────────────────────────────────────

# HSV range for the bright cyan / turquoise tube-outline glow.
TUBE_OUTLINE_H_LOWER = 80
TUBE_OUTLINE_H_UPPER = 110
TUBE_OUTLINE_S_LOWER = 80
TUBE_OUTLINE_S_UPPER = 255
TUBE_OUTLINE_V_LOWER = 140
TUBE_OUTLINE_V_UPPER = 255

# Morphological-closing kernel (fraction of image width) to bridge small gaps
MORPH_KERNEL_FRACTION = 0.02
MORPH_KERNEL_MIN = 5

# Tube-shape constraints.
MIN_TUBE_ASPECT_RATIO = 1.2
MAX_TUBE_ASPECT_RATIO = 5.0
MIN_TUBE_AREA_FRACTION = 0.003

# NMS IoU threshold for merging duplicate detections.
MERGE_IOU_THRESHOLD = 0.3

# ─────────────────────────────────────────────────────────────────────────────
# Ball-slot detection constants
# ─────────────────────────────────────────────────────────────────────────────

TUBE_TOP_MARGIN_FRAC = 0.12  
TUBE_BOTTOM_MARGIN_FRAC = 0.02
TUBE_SIDE_MARGIN_FRAC = 0.13

SAMPLE_RADIUS_FRACTION = 0.30
SAMPLE_RADIUS_MIN = 4

# ─────────────────────────────────────────────────────────────────────────────
# Row clustering
# ─────────────────────────────────────────────────────────────────────────────

ROW_GAP_FRACTION = 0.5

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class BoardDetector:
    """Detects tubes and classifies ball colours from a Woody Sort screenshot."""

    def detect_board(self, image: np.ndarray) -> Tuple[List[Tube], List[List[Optional[str]]], List[int]]:
        tubes = self.detect_tubes(image)
        capacities = [self._calculate_capacity(tube) for tube in tubes]
        board = [
            self.detect_balls(image, tube, cap)
            for tube, cap in zip(tubes, capacities)
        ]
        return tubes, board, capacities

    def detect_tubes(self, image: np.ndarray) -> List[Tube]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        cyan_lower = np.array([TUBE_OUTLINE_H_LOWER, TUBE_OUTLINE_S_LOWER, TUBE_OUTLINE_V_LOWER], dtype=np.uint8)
        cyan_upper = np.array([TUBE_OUTLINE_H_UPPER, TUBE_OUTLINE_S_UPPER, TUBE_OUTLINE_V_UPPER], dtype=np.uint8)
        cyan_mask = cv2.inRange(hsv, cyan_lower, cyan_upper)

        gray_lower = np.array([0, 0, 150], dtype=np.uint8)
        gray_upper = np.array([180, 50, 255], dtype=np.uint8)
        gray_mask = cv2.inRange(hsv, gray_lower, gray_upper)

        mask = cv2.bitwise_or(cyan_mask, gray_mask)

        ks = max(int(image.shape[1] * MORPH_KERNEL_FRACTION), MORPH_KERNEL_MIN)
        if ks % 2 == 0: ks += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = image.shape[0] * image.shape[1] * MIN_TUBE_AREA_FRACTION

        candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w == 0 or w * h < min_area: continue
            aspect = h / w
            if aspect < MIN_TUBE_ASPECT_RATIO or aspect > MAX_TUBE_ASPECT_RATIO: continue
            candidates.append((x, y, w, h))

        merged = _merge_overlapping_boxes(candidates, MERGE_IOU_THRESHOLD)
        return _assign_tube_positions(merged)

    def _calculate_capacity(self, tube: Tube) -> int:
        x, y, w, h = tube.bbox
        side_margin = int(w * TUBE_SIDE_MARGIN_FRAC)
        top_margin = int(h * TUBE_TOP_MARGIN_FRAC)
        bottom_margin = int(h * TUBE_BOTTOM_MARGIN_FRAC)
        inner_w = max(w - 2 * side_margin, 1)
        inner_h = max(h - top_margin - bottom_margin, 1)
        cap = round(inner_h / inner_w)
        return max(2, min(cap, 6))

    def detect_balls(self, image: np.ndarray, tube: Tube, capacity: int) -> List[Optional[str]]:
        x, y, w, h = tube.bbox
        side_margin = int(w * TUBE_SIDE_MARGIN_FRAC)
        top_margin = int(h * TUBE_TOP_MARGIN_FRAC)
        bottom_margin = int(h * TUBE_BOTTOM_MARGIN_FRAC)

        inner_x = x + side_margin
        inner_w = max(w - 2 * side_margin, 1)
        inner_y = y + top_margin
        inner_h = max(h - top_margin - bottom_margin, 1)

        sample_radius = max(int(inner_w * SAMPLE_RADIUS_FRACTION), SAMPLE_RADIUS_MIN)
        slot_height = inner_w
        colors = []

        for si in range(capacity):
            cy = inner_y + inner_h - int((si + 0.5) * slot_height)
            cx = inner_x + inner_w // 2
            y1, y2 = max(0, cy - sample_radius), min(image.shape[0], cy + sample_radius)
            x1, x2 = max(0, cx - sample_radius), min(image.shape[1], cx + sample_radius)
            region = image[y1:y2, x1:x2]

            if region.size == 0:
                break
            color = self._classify_ball_color(region)
            if color is not None:
                colors.append(color)
            else:
                break



        full = [None] * (capacity - len(colors)) + list(reversed(colors))
        return full

    def _classify_ball_color(self, region_bgr: np.ndarray) -> Optional[str]:
        color = classify_color(region_bgr)
        if color is not None and color in BALL_COLORS:
            return color

        hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
        total_pixels = hsv.shape[0] * hsv.shape[1]
        if total_pixels == 0:
            return None

        max_count = 0
        best_color = None
        for color_name, ranges in HSV_RANGES.items():
            if color_name not in BALL_COLORS:
                continue
            combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in ranges:
                m = cv2.inRange(
                    hsv,
                    np.array(lower, dtype=np.uint8),
                    np.array(upper, dtype=np.uint8),
                )
                combined = cv2.bitwise_or(combined, m)
            count = cv2.countNonZero(combined)
            if count > max_count:
                max_count = count
                best_color = color_name

        if best_color and max_count / total_pixels > MIN_MATCH_FRACTION:
            return best_color
        return None

    # ── debug visualisation ──────────────────────────────────────────────

    def draw_debug_image(
        self, image: np.ndarray, tubes: List[Tube], board: List[List[Optional[str]]]
    ) -> np.ndarray:
        """Draw bounding boxes and detected colours on a copy of the image."""
        debug = image.copy()
        for tube, contents in zip(tubes, board):
            x, y, w, h = tube.bbox
            cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                debug,
                str(tube.index),
                (x + w // 2 - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
            
            capacity = len(contents)
            inner_w = max(w - 2 * int(w * TUBE_SIDE_MARGIN_FRAC), 1)
            slot_height = inner_w
            inner_y = y + int(h * TUBE_TOP_MARGIN_FRAC)
            inner_h = max(h - int(h * TUBE_TOP_MARGIN_FRAC) - int(h * TUBE_BOTTOM_MARGIN_FRAC), 1)
            
            for si, color in enumerate(reversed(contents)):
                if color is not None:
                    cy = inner_y + inner_h - int((si + 0.5) * slot_height)
                    cx = x + w // 2
                    cv2.putText(
                        debug,
                        color,
                        (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
        return debug

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _merge_overlapping_boxes(boxes: List[Tuple[int, int, int, int]], iou_threshold: float) -> List[Tuple[int, int, int, int]]:
    if not boxes: return []
    rects = np.array([[x, y, w, h] for (x, y, w, h) in boxes], dtype=np.float32)
    scores = np.ones(len(rects), dtype=np.float32)
    indices = cv2.dnn.NMSBoxes(rects.tolist(), scores.tolist(), 0.0, iou_threshold)
    if len(indices) == 0: return []
    return [boxes[i] for i in indices.flatten()]

def _assign_tube_positions(boxes: List[Tuple[int, int, int, int]]) -> List[Tube]:
    if not boxes: return []
    avg_h = sum(h for _, _, _, h in boxes) / len(boxes)
    max_gap = avg_h * ROW_GAP_FRACTION
    sorted_by_y = sorted(boxes, key=lambda b: b[1] + b[3] / 2)
    rows = []
    current_row = [sorted_by_y[0]]
    for box in sorted_by_y[1:]:
        last_cy = current_row[-1][1] + current_row[-1][3] / 2
        cy = box[1] + box[3] / 2
        if cy - last_cy <= max_gap:
            current_row.append(box)
        else:
            rows.append(current_row)
            current_row = [box]
    if current_row:
        rows.append(current_row)
    tubes = []
    index = 0
    for r_idx, row in enumerate(rows):
        row_sorted = sorted(row, key=lambda b: b[0])
        for c_idx, box in enumerate(row_sorted):
            x, y, w, h = box
            tubes.append(Tube(
                bbox=box,
                center=(x + w // 2, y + h // 2),
                row=r_idx,
                col=c_idx,
                index=index,
            ))
            index += 1
    return tubes
