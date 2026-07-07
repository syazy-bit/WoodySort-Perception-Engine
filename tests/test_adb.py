from pathlib import Path

from adb import ADBError, resolve_adb_path


def test_resolve_adb_path_prefers_explicit_env(monkeypatch, tmp_path):
    adb_path = tmp_path / "adb.exe"
    adb_path.write_text("placeholder")
    monkeypatch.setenv("ADB_PATH", str(adb_path))

    assert resolve_adb_path() == str(adb_path)


def test_resolve_adb_path_accepts_batch_wrappers(monkeypatch, tmp_path):
    adb_path = tmp_path / "adb.bat"
    adb_path.write_text("@echo off")
    monkeypatch.setenv("ADB_PATH", str(adb_path))

    assert resolve_adb_path() == str(adb_path)


def test_resolve_adb_path_ignores_local_module_name(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    local_module = tmp_path / "adb.py"
    local_module.write_text("print('not adb')")
    monkeypatch.delenv("ADB_PATH", raising=False)
    monkeypatch.delenv("ANDROID_HOME", raising=False)
    monkeypatch.delenv("ANDROID_SDK_ROOT", raising=False)

    try:
        resolved = resolve_adb_path()
    except ADBError as exc:
        assert "ADB is not available" in str(exc)
    else:
        assert Path(resolved).name.lower() != "adb.py"
