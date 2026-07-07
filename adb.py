from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import List


_DEVICE_ID_PATTERN = re.compile(r"^[\w\-:.]+$")


class ADBError(RuntimeError):
    pass


def resolve_adb_path() -> str:
    explicit = os.environ.get("ADB_PATH")
    if explicit:
        return explicit

    which_result = shutil.which("adb")
    if which_result and Path(which_result).name.lower() != "adb.py":
        return which_result

    candidates = []
    for env_name in ["ANDROID_HOME", "ANDROID_SDK_ROOT"]:
        root = os.environ.get(env_name)
        if root:
            candidates.append(Path(root) / "platform-tools" / "adb")
            candidates.append(Path(root) / "platform-tools" / "adb.exe")

    candidates.extend(
        [
            Path(os.path.expanduser("~"))
            / "AppData"
            / "Local"
            / "Android"
            / "Sdk"
            / "platform-tools"
            / "adb.exe",
            Path(os.path.expanduser("~"))
            / "AppData"
            / "Local"
            / "Android"
            / "Sdk"
            / "platform-tools"
            / "adb",
        ]
    )

    for candidate in candidates:
        if candidate.exists() and candidate.name.lower() not in {"adb.py"}:
            return str(candidate)

    raise ADBError("ADB is not available on PATH")


def run_adb(args: List[str]) -> str:
    adb_path = resolve_adb_path()
    command = [adb_path, *args]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise ADBError("ADB is not available on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise ADBError(exc.stderr or exc.stdout or "ADB command failed") from exc
    return result.stdout.strip()


def get_devices() -> List[str]:
    output = run_adb(["devices"])
    devices = []
    for line in output.splitlines()[1:]:
        if line.strip() and "device" in line:
            device_id = line.split()[0]
            if _DEVICE_ID_PATTERN.match(device_id):
                devices.append(device_id)
    return devices


def tap(x: int, y: int) -> None:
    run_adb(["shell", "input", "tap", str(x), str(y)])
