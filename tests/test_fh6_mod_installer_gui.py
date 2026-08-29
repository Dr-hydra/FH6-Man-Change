from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "install_fh6_mod_gui", SCRIPTS / "install_fh6_mod_gui.py"
)
assert SPEC is not None and SPEC.loader is not None
GUI = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GUI
SPEC.loader.exec_module(GUI)


def make_zip(path: Path, names: list[str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            archive.writestr(name, b"{}")


def test_archive_manifest_detection(tmp_path: Path) -> None:
    valid = tmp_path / "valid.zip"
    invalid = tmp_path / "invalid.zip"
    duplicate = tmp_path / "duplicate.zip"
    make_zip(valid, ["package/mod.json", "package/data.bin"])
    make_zip(invalid, ["readme.txt"])
    make_zip(duplicate, ["one/mod.json", "two/mod.json"])

    assert GUI.archive_has_single_manifest(valid)
    assert not GUI.archive_has_single_manifest(invalid)
    assert not GUI.archive_has_single_manifest(duplicate)


def test_adjacent_package_requires_one_candidate(tmp_path: Path) -> None:
    first = tmp_path / "first.zip"
    make_zip(first, ["mod.json"])
    assert GUI.discover_adjacent_mod_zip(tmp_path) == first

    make_zip(tmp_path / "second.zip", ["nested/mod.json"])
    assert GUI.discover_adjacent_mod_zip(tmp_path) is None


def test_result_message_reports_backups() -> None:
    report = {
        "apply": True,
        "replacements": [
            {"already_installed": False, "backup": "one.bak"},
            {"already_installed": True},
        ],
        "characters_zip": {
            "backup": "characters.zip.bak",
            "already_present": False,
            "replace_before": 4,
            "append_before": 11,
            "replaced": 4,
            "added": 11,
        },
    }
    result = GUI.InstallResult(
        dry_run=False,
        mod_name="Test Mod",
        media_directory=Path("media"),
        report_path=Path("report.json"),
        report=report,
    )

    message = GUI.result_message(result)
    assert "Installation completed successfully" in message
    assert "Original files backed up: 2" in message
