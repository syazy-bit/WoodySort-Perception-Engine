# WoodySort-Perception-Engine

An autonomous, computer-vision-driven solver and physical execution engine for Woody Sort (and similar color-sorting puzzle games). 

This project bridges pixel-perfect perception, graph-based heuristic solving, and direct hardware actuation (via ADB) into a fully closed-loop automation system. It is capable of visually interpreting dynamic puzzle states, computing the mathematically optimal move sequence, and autonomously playing the game directly on a physical Android device.

## Core Architecture

The engine is built on three primary pillars:

1. **Computer Vision & Perception (`detect.py`, `colors.py`)**
   - Employs OpenCV to process raw screenshots from the device.
   - Detects dynamic tube capacities and geometry without relying on hardcoded coordinates, allowing it to adapt to any level layout or screen resolution.
   - Uses precise HSV color-space masking to classify individual balls (handling up to 9 distinct colors including edge-case hues like Violet and Cyan).

2. **Graph-Based Solver (`solver.py`, `board.py`)**
   - Models the puzzle as a directed graph of immutable board states.
   - Implements an optimized Depth-First Search (DFS) with state memoization to calculate the fastest path to fully consolidated color tubes.
   - Computes the entire move sequence in milliseconds before issuing the first command.

3. **Autonomous Actuation Loop (`main.py`, `executor.py`)**
   - Communicates with the physical device via the Android Debug Bridge (ADB).
   - Executes a closed-loop `while True:` cycle: `Capture -> Detect -> Solve -> Actuate -> Verify`.
   - Features an adaptive verification mechanism that continuously samples the screen post-move, recovering gracefully from device lag or in-game animations.

## Prerequisites & Setup

1. **The Game**: Install the game **[Woody Sort](https://play.google.com/store/apps/details?id=com.unicostudio.balltubes)** (or similar compatible color-sorting game) on your Android device. 
2. **Device Connection**: Connect your Android device via USB to your computer and ensure **USB Debugging** is enabled in Developer Options.
3. **Environment Setup**: 
   Ensure you have Python 3.9+ installed. Then, set up a virtual environment and install the required dependencies:

   **Create the virtual environment:**
   ```bash
   python -m venv .venv
   ```

   **Activate the virtual environment:**
   - **Windows (PowerShell):**
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - **Windows (Command Prompt):**
     ```cmd
     .venv\Scripts\activate.bat
     ```
   - **macOS / Linux:**
     ```bash
     source .venv/bin/activate
     ```

   **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Open the **Woody Sort** game on your connected Android device and start a level.
2. Run the main orchestration script from your terminal:

```bash
python main.py
```

The engine will automatically capture the screen, detect the tubes and balls, compute the best sequence of moves, and begin tapping your screen autonomously until the level is solved!

*Note: Use the `--dry-run` flag to run the perception and solver logic to preview the first move without issuing physical tap commands to the device.*

## Logging & Analytics

The engine maintains comprehensive session logs in the `logs/` directory. Each run generates a timestamped folder containing:
- Before/After detection screenshots (`moveXXX_detected.png`).
- A `boards.json` file structured with complete state transitions for every move.
- Verbose execution logs (`run.log`) detailing solver depth, explored states, and HSV boundary debugging.
