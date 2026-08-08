#!/usr/bin/env python3
"""Dry-run, deploy, or restore a three-archive Si Display package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rebuild_fh6_aligned_zip import verify


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parent
DEFAULT_GAME_ROOT = Path(r"D:\SteamLibrary\steamapps\common\ForzaHorizon6")
DEFAULT_PACKAGE = (
    WORKSPACE
    / "work"
    / "si"
    / "fbx-source"
    / "milestone-06-display-package-lod0-v001"
    / "archives"
    / "si-display-lod0-v001.package.json"
)
DEFAULT_REPORT_DIR = (
    WORKSPACE
    / "work"
    / "si"
    / "fbx-source"
    / "milestone-06-display-package-lod0-v001"
    / "deployment"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-root", type=Path, default=DEFAULT_GAME_ROOT)
    parser.add_argument("--package-manifest", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--report", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Deploy after all dry-run gates pass")
    mode.add_argument(
        "--restore",
        type=Path,
        metavar="DEPLOYMENT_REPORT",
        help="Restore timestamped backups recorded by a prior apply report",
    )
    return parser.parse_args()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_path(path)}


def inside_game_root(game_root: Path, target: Path) -> Path:
    resolved = target.resolve()
    try:
        resolved.relative_to(game_root)
    except ValueError as exc:
        raise ValueError(f"Target escapes game root: {resolved}") from exc
    return resolved


def expected_entries(record: dict[str, Any]) -> int:
    if "entries" in record:
        return int(record["entries"])
    return int(record["entries_after"])


def inspect_package(game_root: Path, package_path: Path) -> tuple[dict, list[dict]]:
    package = json.loads(package_path.read_text(encoding="utf-8"))
    if package["validation"].get("game_directory_modified") is not False:
        raise ValueError("Package manifest is not an offline, undeployed build")
    assets = []
    for key, record in package["archives"].items():
        candidate = Path(record["path"]).resolve(strict=True)
        candidate_hash = sha256_path(candidate)
        if candidate_hash != record["sha256"]:
            raise ValueError(f"Package candidate hash changed: {candidate}")
        archive_validation = verify(candidate, {}, expected_entries(record))
        target = inside_game_root(game_root, game_root / record["game_relative_path"])
        if not target.is_file():
            raise FileNotFoundError(target)
        target_hash = sha256_path(target)
        assets.append(
            {
                "key": key,
                "candidate": str(candidate),
                "candidate_bytes": candidate.stat().st_size,
                "candidate_sha256": candidate_hash,
                "target": str(target),
                "target_bytes": target.stat().st_size,
                "target_sha256": target_hash,
                "already_deployed": target_hash == candidate_hash,
                "archive_validation": archive_validation,
            }
        )
    return package, assets


def atomic_deploy(assets: list[dict], timestamp: str) -> None:
    changed = [item for item in assets if not item["already_deployed"]]
    staged: list[tuple[dict, Path]] = []
    installed: list[dict] = []
    try:
        for item in changed:
            target = Path(item["target"])
            temporary = target.with_name(target.name + f".tmp-si-{timestamp}")
            backup = target.with_name(target.name + f".bak-si-{timestamp}")
            if temporary.exists() or backup.exists():
                raise FileExistsError(
                    f"Refusing to overwrite deployment staging or backup for {target}"
                )
            shutil.copy2(item["candidate"], temporary)
            if sha256_path(temporary) != item["candidate_sha256"]:
                raise ValueError(f"Staged candidate hash mismatch: {temporary}")
            item["backup"] = str(backup)
            item["staged"] = str(temporary)
            staged.append((item, temporary))

        for item, _temporary in staged:
            shutil.copy2(item["target"], item["backup"])
            if sha256_path(Path(item["backup"])) != item["target_sha256"]:
                raise ValueError(f"Backup hash mismatch: {item['backup']}")

        for item, temporary in staged:
            os.replace(temporary, item["target"])
            installed.append(item)
            if sha256_path(Path(item["target"])) != item["candidate_sha256"]:
                raise ValueError(f"Deployed hash mismatch: {item['target']}")
            item["deployed_sha256"] = item["candidate_sha256"]
    except Exception:
        for item in reversed(installed):
            rollback = Path(item["target"]).with_name(
                Path(item["target"]).name + f".rollback-si-{timestamp}"
            )
            shutil.copy2(item["backup"], rollback)
            os.replace(rollback, item["target"])
        raise
    finally:
        for _item, temporary in staged:
            if temporary.exists():
                temporary.unlink()


def restore_deployment(game_root: Path, deployment_path: Path, timestamp: str) -> list[dict]:
    deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
    if deployment.get("mode") != "apply" or not deployment.get("completed"):
        raise ValueError("Restore requires a completed apply deployment report")
    assets = []
    for original in deployment["assets"]:
        if original.get("already_deployed"):
            continue
        target = inside_game_root(game_root, Path(original["target"]))
        backup = Path(original["backup"]).resolve(strict=True)
        if sha256_path(backup) != original["target_sha256"]:
            raise ValueError(f"Recorded backup hash changed: {backup}")
        assets.append(
            {
                "key": original["key"],
                "target": str(target),
                "backup": str(backup),
                "expected_restored_sha256": original["target_sha256"],
                "pre_restore_sha256": sha256_path(target),
            }
        )

    staged: list[tuple[dict, Path, Path]] = []
    restored: list[dict] = []
    try:
        for item in assets:
            target = Path(item["target"])
            temporary = target.with_name(target.name + f".tmp-restore-si-{timestamp}")
            pre_restore = target.with_name(target.name + f".bak-prerestore-si-{timestamp}")
            if temporary.exists() or pre_restore.exists():
                raise FileExistsError(f"Restore staging path already exists for {target}")
            shutil.copy2(item["backup"], temporary)
            shutil.copy2(target, pre_restore)
            item["pre_restore_backup"] = str(pre_restore)
            staged.append((item, temporary, pre_restore))
        for item, temporary, _pre_restore in staged:
            os.replace(temporary, item["target"])
            restored.append(item)
            actual = sha256_path(Path(item["target"]))
            if actual != item["expected_restored_sha256"]:
                raise ValueError(f"Restored hash mismatch: {item['target']}")
            item["restored_sha256"] = actual
    except Exception:
        for item in reversed(restored):
            rollback = Path(item["pre_restore_backup"])
            os.replace(rollback, item["target"])
        raise
    finally:
        for _item, temporary, _pre_restore in staged:
            if temporary.exists():
                temporary.unlink()
    return assets


def main() -> int:
    args = arguments()
    game_root = args.game_root.resolve(strict=True)
    timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
    mode = "restore" if args.restore else "apply" if args.apply else "dry-run"
    report_path = (
        args.report.resolve()
        if args.report
        else DEFAULT_REPORT_DIR / f"{mode}-{timestamp}.json"
    )
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite report {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if args.restore:
        deployment_path = args.restore.resolve(strict=True)
        assets = restore_deployment(game_root, deployment_path, timestamp)
        report = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
            "mode": "restore",
            "game_root": str(game_root),
            "deployment_report": str(deployment_path),
            "assets": assets,
            "completed": True,
        }
    else:
        package_path = args.package_manifest.resolve(strict=True)
        _package, assets = inspect_package(game_root, package_path)
        if args.apply:
            atomic_deploy(assets, timestamp)
        report = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
            "mode": mode,
            "game_root": str(game_root),
            "package_manifest": str(package_path),
            "assets": assets,
            "changed_assets": sum(not item["already_deployed"] for item in assets),
            "completed": True,
        }

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "SI_DISPLAY_PACKAGE_DEPLOY="
        + json.dumps(
            {
                "mode": mode,
                "report": str(report_path),
                "assets": len(report["assets"]),
                "completed": True,
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
