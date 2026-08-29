#!/usr/bin/env python3
"""Build the generic FH6 Mod Installer as a standalone Windows executable."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--name", default="FH6ModInstaller")
    parser.add_argument(
        "--no-uac",
        action="store_true",
        help="Do not request administrator privileges in the Windows manifest.",
    )
    args = parser.parse_args()

    scripts = Path(__file__).resolve().parent
    entry = scripts / "install_fh6_mod_gui.py"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="fh6-gui-build-") as temporary:
        temporary_root = Path(temporary)
        command = [
            sys.executable,
            "-m",
            "PyInstaller",
            "--onefile",
            "--windowed",
            "--clean",
            "--noconfirm",
            "--name",
            args.name,
            "--paths",
            str(scripts),
            "--distpath",
            str(output),
            "--workpath",
            str(temporary_root / "work"),
            "--specpath",
            str(temporary_root / "spec"),
        ]
        if not args.no_uac:
            command.append("--uac-admin")
        command.append(str(entry))
        subprocess.run(command, cwd=scripts.parent, check=True)

    executable = output / f"{args.name}.exe"
    if not executable.is_file():
        raise FileNotFoundError(f"PyInstaller did not create {executable}")
    print(executable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
