"""Woody Sort game solver — main pipeline.

Pipeline:
1. Capture screenshot
2. Detect board
3. If puzzle solved, stop
4. Solve puzzle
5. If no moves, stop
6. Execute ONLY the first move
7. Wait for animation
8. Capture verification screenshot
9. Detect verification board
10. Compare expected vs actual board
11. If PASS, continue loop. Else, stop immediately.

Logs are saved to the logs/YYYY-MM-DD_HH-MM-SS/ directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np

from adb import ADBError, get_devices
from board import Board
from capture import CaptureError, capture_screenshot
from detect import BoardDetector, Tube
from executor import POST_MOVE_DELAY_S, execute_move
from solver import solve


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

BASE_LOG_DIR = Path("logs")
BASE_LOG_DIR.mkdir(exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
SESSION_DIR = BASE_LOG_DIR / TIMESTAMP
SESSION_DIR.mkdir(exist_ok=True)
LOG_FILE = SESSION_DIR / "run.log"
BOARDS_JSON_FILE = SESSION_DIR / "boards.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.ERROR)
logging.getLogger().addHandler(console_handler)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_detector = BoardDetector()
_boards_log_data: List[Dict[str, Any]] = []


def _capture(path: Path) -> None:
    capture_screenshot(str(path))


def _load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Failed to read image from {path}")
    return image


def _detect(
    image: np.ndarray,
) -> Tuple[List[Tube], List[List[Optional[str]]], List[int]]:
    tubes, raw_board, capacities = _detector.detect_board(image)
    if not tubes:
        raise RuntimeError("No tubes detected in the screenshot")
    return tubes, raw_board, capacities


def _raw_board_to_solver_board(
    raw_board: List[List[Optional[str]]],
    capacities: List[int],
) -> Board:
    tubes_for_solver: List[List[str]] = []
    for slot_colors in raw_board:
        filled = [c for c in slot_colors if c is not None]
        filled.reverse()
        tubes_for_solver.append(filled)
    return Board(tubes_for_solver, capacities=tuple(capacities))


def _print_board(board: Board) -> None:
    for idx, tube in enumerate(board.tubes):
        log.info("Tube %d", idx + 1)
        if not tube:
            log.info("  [empty]")
        else:
            for color in reversed(tube):
                log.info("  %s", color)


def _save_debug_image(
    image: np.ndarray,
    tubes: List[Tube],
    raw_board: List[List[Optional[str]]],
    path: Path,
) -> None:
    debug = _detector.draw_debug_image(image, tubes, raw_board)
    cv2.imwrite(str(path), debug)


def _predict_board_after_move(
    board: Board,
    move: Tuple[int, int],
) -> Optional[Board]:
    return board.move(move[0], move[1])


def _save_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _progress_callback(visited_count: int, depth: int) -> None:
    if visited_count % 10_000 == 0:
        log.info("Solver: explored %d states (depth %d)...", visited_count, depth)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="AutoWoody AI — Continuous Solver")
    parser.add_argument("--dry-run", action="store_true", help="Solve once, no execution")
    parser.add_argument("--device", type=str, default=None, help="ADB device serial")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        console_handler.setLevel(logging.DEBUG)

    log.info("AutoWoody AI starting (dry_run=%s)", args.dry_run)
    log.info("Session directory: %s", SESSION_DIR.resolve())

    try:
        devices = get_devices()
    except ADBError as exc:
        print(f"\nExecution stopped\nReason: ADB error: {exc}")
        log.error("ADB error: %s", exc)
        return 1

    if not devices:
        print("\nExecution stopped\nReason: No Android device detected.")
        log.error("No Android device detected.")
        return 1

    device_serial = args.device or devices[0]
    log.info("Using device: %s", device_serial)

    move_count = 1
    start_time = time.time()

    while True:
        log.info("=== Starting iteration %d ===", move_count)

        # ── 1. Capture ──
        before_path = SESSION_DIR / f"move{move_count:03d}_before.png"
        try:
            _capture(before_path)
        except CaptureError as exc:
            print(f"\nExecution stopped\nReason: Capture failed: {exc}")
            log.error("Capture failed: %s", exc)
            return 1

        image = _load_image(before_path)

        # ── 2. Detect ──
        try:
            tubes, raw_board, capacities = _detect(image)
        except RuntimeError as exc:
            print(f"\nExecution stopped\nReason: Detection error: {exc}")
            log.error("Detection error: %s", exc)
            return 1

        log.info("Detected %d tubes", len(tubes))
        _save_debug_image(image, tubes, raw_board, SESSION_DIR / f"move{move_count:03d}_detected.png")

        board = _raw_board_to_solver_board(raw_board, capacities)

        # ── 3. Check if solved ──
        if board.is_solved():
            elapsed = time.time() - start_time
            print(f"\nPuzzle solved!\nTotal moves: {move_count - 1}\nElapsed time: {elapsed:.1f} seconds")
            log.info("Board is solved! Total moves: %d, Time: %.1fs", move_count - 1, elapsed)
            return 0

        # ── 4. Solve ──
        moves = solve(board, progress_callback=_progress_callback)
        if not moves:
            print("\nExecution stopped\nReason: Solver found no solution for this board.")
            log.error("Solver found no solution for this board.")
            return 1

        first_move = moves[0]
        src, dst = first_move[0], first_move[1]
        
        # Calculate total moves: current move + remaining moves from solver
        total_moves = (move_count - 1) + len(moves)

        print(f"\nMove {move_count} / {total_moves}")
        print(f"Detected {len(tubes)} tubes\n")
        print("Solver:")
        print(f"Tube {src + 1} -> Tube {dst + 1}\n")

        if args.dry_run:
            print(f"Dry-run mode. First move is Tube {src + 1} -> Tube {dst + 1}. Exiting.")
            log.info("Dry-run finished.")
            return 0

        # ── 5. Execute move ──
        execute_move(tubes[src], tubes[dst])
        time.sleep(POST_MOVE_DELAY_S)

        # ── 6. Verification with Retries ──
        expected_board = _predict_board_after_move(board, first_move)
        if expected_board is None:
            print("\nExecution stopped\nReason: Predicted move was invalid.")
            log.error("Predicted move was invalid.")
            return 1

        # If the expected board is solved, the game will instantly show a "Level Completed" 
        # popup and an ad, hiding the board. We must skip image verification.
        if expected_board.is_solved():
            elapsed = time.time() - start_time
            print("\nVerification:\nPASS (Final Move Assumed)\n")
            print(f"Puzzle solved!\nTotal moves: {move_count}\nElapsed time: {elapsed:.1f} seconds")
            log.info("Board is solved! Total moves: %d, Time: %.1fs", move_count, elapsed)
            return 0

        verification_passed = False
        max_retries = 3
        
        for attempt in range(1, max_retries + 1):
            after_path = SESSION_DIR / f"move{move_count:03d}_after_try{attempt}.png"
            try:
                _capture(after_path)
            except CaptureError as exc:
                print(f"\nExecution stopped\nReason: Verification capture failed: {exc}")
                log.error("Verification capture failed: %s", exc)
                return 1

            verify_image = _load_image(after_path)

            try:
                verify_tubes, verify_raw_board, verify_capacities = _detect(verify_image)
                verify_board = _raw_board_to_solver_board(verify_raw_board, verify_capacities)
                _save_debug_image(verify_image, verify_tubes, verify_raw_board, SESSION_DIR / f"move{move_count:03d}_after_detected_try{attempt}.png")
                
                if verify_board.tubes == expected_board.tubes:
                    verification_passed = True
                    break
                else:
                    log.warning("Verification failed on attempt %d", attempt)
                    if attempt < max_retries:
                        time.sleep(1.0)
                        
            except RuntimeError as exc:
                log.warning("Verification detection failed on attempt %d: %s", attempt, exc)
                if attempt < max_retries:
                    time.sleep(1.0)

        # Log move data
        move_data = {
            "move_number": move_count,
            "board_before": board.tubes,
            "solver_move": [src + 1, dst + 1],
            "expected_board": expected_board.tubes,
            "detected_board": verify_board.tubes if 'verify_board' in locals() else None,
            "verification_pass": verification_passed
        }
        _boards_log_data.append(move_data)
        _save_json(BOARDS_JSON_FILE, _boards_log_data)

        if verification_passed:
            print("Verification:\nPASS\n\nContinuing...")
            log.info("Verification PASS for move %d", move_count)
            move_count += 1
        else:
            print("Verification:\nFAIL\n")
            print("Execution stopped\nReason: Verification failed after 3 attempts")
            log.error("Verification FAIL for move %d after %d attempts", move_count, max_retries)
            
            # Print specific diff for debugging in logs
            log.error("Expected:")
            _print_board(expected_board)
            if 'verify_board' in locals():
                log.error("Detected:")
                _print_board(verify_board)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
