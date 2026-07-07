from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from adb import resolve_adb_path


class CaptureError(RuntimeError):
    pass


def capture_screenshot(output_path: Optional[str] = None) -> Optional[bytes]:
    try:
        adb_path = resolve_adb_path()
        command = [adb_path, "exec-out", "screencap", "-p"]
        result = subprocess.run(command, capture_output=True, check=True)
    except FileNotFoundError as exc:
        raise CaptureError("ADB is not available on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise CaptureError(exc.stderr.decode("utf-8", errors="ignore")) from exc
    except Exception as exc:
        raise CaptureError(str(exc)) from exc

    data = result.stdout
    if output_path:
        Path(output_path).write_bytes(data)
    return data
