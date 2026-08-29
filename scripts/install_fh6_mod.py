#!/usr/bin/env python3
"""Install a packaged FH6 Mod from a game directory and distribution ZIP."""

from __future__ import annotations

import argparse
import json
import shutil
import stat
import sys
import tempfile
import zipfile
from collections import deque
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from deploy_fh6_mod_package import deploy_package, load_manifest


MAX_ARCHIVE_FILES = 256
MAX_UNCOMPRESSED_BYTES = 4 * 1024 * 1024 * 1024
MAX_MEDIA_SEARCH_DEPTH = 4


def normalized_member_path(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    relative = PurePosixPath(normalized)
    if (
        not normalized
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or (relative.parts and ":" in relative.parts[0])
    ):
        raise ValueError(f"压缩包包含不安全路径: {name!r}")
    return relative


def extract_package(archive_path: Path, destination: Path) -> Path:
    """Safely extract a Mod ZIP and return its unique mod.json directory."""
    if not zipfile.is_zipfile(archive_path):
        raise ValueError(f"不是有效的 ZIP 压缩包: {archive_path}")

    extracted_files: list[Path] = []
    seen: set[str] = set()
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if len(members) > MAX_ARCHIVE_FILES:
            raise ValueError(
                f"压缩包文件数过多: {len(members)} > {MAX_ARCHIVE_FILES}"
            )
        total_size = sum(member.file_size for member in members)
        if total_size > MAX_UNCOMPRESSED_BYTES:
            raise ValueError(
                f"压缩包解压后过大: {total_size} > {MAX_UNCOMPRESSED_BYTES}"
            )

        for member in members:
            relative = normalized_member_path(member.filename)
            key = relative.as_posix().casefold()
            if key in seen:
                raise ValueError(f"压缩包包含重复路径: {member.filename}")
            seen.add(key)
            mode = member.external_attr >> 16
            if mode and stat.S_ISLNK(mode):
                raise ValueError(f"压缩包不允许符号链接: {member.filename}")

            output = destination.joinpath(*relative.parts)
            resolved = output.resolve()
            try:
                resolved.relative_to(destination.resolve())
            except ValueError as exc:
                raise ValueError(f"压缩包路径越界: {member.filename}") from exc
            if member.is_dir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, output.open("xb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
            if output.stat().st_size != member.file_size:
                raise ValueError(f"解压后的文件大小不匹配: {member.filename}")
            extracted_files.append(output)

    manifests = [path for path in extracted_files if path.name.casefold() == "mod.json"]
    if len(manifests) != 1:
        raise ValueError(f"压缩包必须且只能包含一个 mod.json，实际为 {len(manifests)} 个")
    package_root = manifests[0].parent
    load_manifest(package_root)
    return package_root


def required_media_targets(manifest: dict[str, Any]) -> list[PurePosixPath]:
    targets = [item["game_target"] for item in manifest["replacements"]]
    targets.extend(item["game_target"] for item in manifest.get("xml_patches", []))
    targets.append(manifest["characters_zip_injection"]["game_target"])
    result = []
    for raw in targets:
        path = PurePosixPath(raw)
        if not path.parts or path.parts[0].casefold() != "media" or ".." in path.parts:
            raise ValueError(f"mod.json 包含无效游戏目标: {raw}")
        result.append(PurePosixPath(*path.parts[1:]))
    return result


def media_matches(media: Path, required: list[PurePosixPath]) -> bool:
    if not media.is_dir() or media.name.casefold() != "media":
        return False
    return all(media.joinpath(*relative.parts).is_file() for relative in required)


def find_media_directory(
    supplied_directory: Path, manifest: dict[str, Any]
) -> Path:
    """Find the unique media directory at or below the supplied directory."""
    supplied = supplied_directory.resolve(strict=True)
    if not supplied.is_dir():
        raise NotADirectoryError(f"游戏路径不是目录: {supplied}")
    required = required_media_targets(manifest)

    if media_matches(supplied, required):
        return supplied

    candidates: list[Path] = []

    queue: deque[tuple[Path, int]] = deque([(supplied, 0)])
    visited = {str(supplied).casefold()}
    while queue:
        current, depth = queue.popleft()
        if depth >= MAX_MEDIA_SEARCH_DEPTH:
            continue
        try:
            children = sorted(
                (child for child in current.iterdir() if child.is_dir()),
                key=lambda child: child.name.casefold(),
            )
        except OSError:
            continue
        for child in children:
            resolved = child.resolve()
            key = str(resolved).casefold()
            if key in visited:
                continue
            visited.add(key)
            if child.name.casefold() == "media":
                if media_matches(child, required):
                    candidates.append(resolved)
                continue
            queue.append((resolved, depth + 1))

    unique = {str(path).casefold(): path for path in candidates}
    if not unique:
        raise FileNotFoundError(
            "未找到包含全部目标资源的 media 目录；请提供 ForzaHorizon6 游戏目录或其 media 目录"
        )
    if len(unique) > 1:
        choices = "\n  - ".join(str(path) for path in unique.values())
        raise ValueError(f"找到多个可用 media 目录，请提供更精确的路径:\n  - {choices}")
    return next(iter(unique.values()))


def default_report_path(archive_path: Path, dry_run: bool) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    mode = "dry-run" if dry_run else "install"
    return archive_path.with_name(f"{archive_path.stem}.{mode}-{stamp}.json")


def summary(report: dict[str, Any]) -> dict[str, Any]:
    replacements = report["replacements"]
    xml_patches = report.get("xml_patches", [])
    characters = report["characters_zip"]
    return {
        "mode": "installed" if report["apply"] else "dry-run",
        "resource_packages_changed": sum(
            1 for item in replacements if "backup" in item
        ),
        "resource_packages_to_change": sum(
            1 for item in replacements if not item["already_installed"]
        ),
        "resource_packages_already_installed": sum(
            1 for item in replacements if item["already_installed"]
        ),
        "xml_files_changed": sum(1 for item in xml_patches if "backup" in item),
        "xml_files_to_change": sum(
            1 for item in xml_patches if not item["already_patched"]
        ),
        "xml_files_already_patched": sum(
            1 for item in xml_patches if item["already_patched"]
        ),
        "swatches_replaced": characters["replaced"],
        "swatches_added": characters["added"],
        "swatches_to_replace": characters["replace_before"],
        "swatches_to_add": characters["append_before"],
        "characters_zip_already_patched": characters["already_present"],
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="安装 FH6 Mod：自动识别 media，预检后备份并替换资源。"
    )
    parser.add_argument("game_directory", type=Path, help="游戏目录或 media 目录")
    parser.add_argument("mod_zip", type=Path, help="Mod 分发 ZIP")
    parser.add_argument(
        "--dry-run", action="store_true", help="只检查，不修改游戏文件"
    )
    parser.add_argument("--report", type=Path, help="安装报告 JSON 输出路径")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    try:
        archive_path = args.mod_zip.resolve(strict=True)
        if not archive_path.is_file():
            raise FileNotFoundError(f"Mod 压缩包不是文件: {archive_path}")
        report_path = (
            args.report.resolve()
            if args.report is not None
            else default_report_path(archive_path, args.dry_run).resolve()
        )
        with tempfile.TemporaryDirectory(prefix="fh6-mod-installer-") as temporary:
            package_root = extract_package(archive_path, Path(temporary))
            manifest = load_manifest(package_root)
            media = find_media_directory(args.game_directory, manifest)
            game_root = media.parent
            mod_name = manifest["mod"].get("name_zh_cn") or manifest["mod"]["name"]
            print(f"Mod: {mod_name}")
            print(f"Media: {media}")
            print("模式: 仅预检" if args.dry_run else "模式: 备份并安装")
            report = deploy_package(
                package_root,
                game_root,
                apply=not args.dry_run,
                report_path=report_path,
            )

        result = summary(report)
        print("安装完成。" if not args.dry_run else "预检通过，未修改游戏文件。")
        print(f"报告: {report_path}")
        print("FH6_MOD_INSTALLER=" + json.dumps(result, ensure_ascii=False, separators=(",", ":")))
        return 0
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        print(f"安装失败: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
