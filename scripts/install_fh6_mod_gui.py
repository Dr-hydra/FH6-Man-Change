#!/usr/bin/env python3
"""Small English GUI for installing packaged FH6 Mods on Windows."""

from __future__ import annotations

import os
import queue
import sys
import tempfile
import threading
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import StringVar, Tk, filedialog, messagebox
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from typing import Callable

from deploy_fh6_mod_package import deploy_package, load_manifest
from install_fh6_mod import extract_package, find_media_directory, summary


APP_NAME = "FH6 Mod Installer"
WINDOW_SIZE = "760x560"


@dataclass(frozen=True)
class InstallResult:
    dry_run: bool
    mod_name: str
    media_directory: Path
    report_path: Path
    report: dict


def application_directory() -> Path:
    """Return the executable folder, or the current folder in source mode."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path.cwd().resolve()


def archive_has_single_manifest(path: Path) -> bool:
    """Return whether a ZIP looks like one packaged FH6 Mod."""
    try:
        with zipfile.ZipFile(path) as archive:
            manifests = [
                item
                for item in archive.infolist()
                if not item.is_dir() and Path(item.filename).name.casefold() == "mod.json"
            ]
        return len(manifests) == 1
    except (OSError, zipfile.BadZipFile):
        return False


def discover_adjacent_mod_zip(directory: Path) -> Path | None:
    """Select a Mod ZIP automatically when exactly one is beside the app."""
    candidates = [
        path
        for path in sorted(directory.glob("*.zip"), key=lambda item: item.name.casefold())
        if archive_has_single_manifest(path)
    ]
    return candidates[0] if len(candidates) == 1 else None


def report_directory() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    root = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
    return root / "FH6ModInstaller" / "Reports"


def allocate_report_path(mod_id: str, dry_run: bool) -> Path:
    root = report_directory()
    root.mkdir(parents=True, exist_ok=True)
    mode = "verification" if dry_run else "installation"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = root / f"{mod_id}.{mode}-{stamp}.json"
    if not base.exists():
        return base
    for suffix in range(1, 1000):
        candidate = base.with_name(f"{base.stem}-{suffix}{base.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError("Could not allocate a unique installation report name.")


def friendly_error(exc: Exception) -> str:
    """Convert common core errors into concise English GUI messages."""
    text = str(exc)
    translations = (
        ("不是有效的 ZIP 压缩包", "The selected Mod package is not a valid ZIP archive."),
        ("压缩包必须且只能包含一个 mod.json", "The Mod ZIP must contain exactly one mod.json file."),
        ("未找到包含全部目标资源的 media 目录", "No compatible FH6 media folder was found under the selected game directory."),
        ("找到多个可用 media 目录", "More than one compatible media folder was found. Select the exact FH6 game folder."),
        ("游戏路径不是目录", "The selected game path is not a directory."),
        ("压缩包包含不安全路径", "The Mod ZIP contains an unsafe file path."),
        ("压缩包包含重复路径", "The Mod ZIP contains duplicate file paths."),
        ("压缩包不允许符号链接", "The Mod ZIP contains a symbolic link, which is not allowed."),
        ("压缩包路径越界", "The Mod ZIP contains a path outside its package directory."),
        ("压缩包文件数过多", "The Mod ZIP contains too many files."),
        ("压缩包解压后过大", "The Mod ZIP is larger than the installer safety limit."),
    )
    for source, replacement in translations:
        if source in text:
            return replacement
    if isinstance(exc, PermissionError):
        return "Access was denied. Close the game and run the installer as administrator."
    if isinstance(exc, FileNotFoundError):
        return "A required file or folder was not found. Check both selected paths."
    if isinstance(exc, NotADirectoryError):
        return "The selected game path is not a directory."
    if isinstance(exc, zipfile.BadZipFile):
        return "The selected Mod package is not a valid ZIP archive."
    return text or exc.__class__.__name__


def install_package(
    game_directory: Path,
    mod_zip: Path,
    *,
    dry_run: bool,
    notify: Callable[[str], None] | None = None,
) -> InstallResult:
    """Run the shared installer core and return a GUI-friendly result."""
    emit = notify or (lambda _message: None)
    archive_path = mod_zip.expanduser().resolve(strict=True)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Mod package is not a file: {archive_path}")

    emit("Reading and validating the Mod package...")
    with tempfile.TemporaryDirectory(prefix="fh6-mod-installer-") as temporary:
        package_root = extract_package(archive_path, Path(temporary))
        manifest = load_manifest(package_root)
        mod_data = manifest["mod"]
        mod_name = mod_data.get("name") or mod_data["id"]
        report_path = allocate_report_path(mod_data["id"], dry_run)

        emit("Finding the FH6 media directory...")
        media = find_media_directory(game_directory.expanduser(), manifest)
        emit("Checking game files..." if dry_run else "Backing up and installing files...")
        report = deploy_package(
            package_root,
            media.parent,
            apply=not dry_run,
            report_path=report_path,
        )

    return InstallResult(
        dry_run=dry_run,
        mod_name=mod_name,
        media_directory=media,
        report_path=report_path,
        report=report,
    )


def result_message(result: InstallResult) -> str:
    values = summary(result.report)
    if result.dry_run:
        return (
            "Verification passed. No game files were changed.\n\n"
            f"Resource packages to install: {values['resource_packages_to_change']}\n"
            f"Customization XML files to patch: {values['xml_files_to_change']}\n"
            f"Texture entries to replace: {values['swatches_to_replace']}\n"
            f"Texture entries to add: {values['swatches_to_add']}"
        )

    backup_count = sum(
        1 for item in result.report["replacements"] if item.get("backup")
    ) + sum(
        1 for item in result.report.get("xml_patches", []) if item.get("backup")
    ) + int(bool(result.report["characters_zip"].get("backup")))
    changed = values["resource_packages_changed"]
    if (
        changed == 0
        and values["xml_files_changed"] == 0
        and not result.report["characters_zip"].get("backup")
    ):
        return "The Mod is already installed. No game files were changed."
    return (
        "Installation completed successfully.\n\n"
        f"Resource packages installed: {changed}\n"
        f"Customization XML files patched: {values['xml_files_changed']}\n"
        f"Texture entries replaced: {values['swatches_replaced']}\n"
        f"Texture entries added: {values['swatches_added']}\n"
        f"Original files backed up: {backup_count}"
    )


class InstallerWindow:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.busy = False
        self.game_path = StringVar()
        self.mod_path = StringVar()
        self.status = StringVar(value="Ready")

        self.root.title(APP_NAME)
        self.root.geometry(WINDOW_SIZE)
        self.root.minsize(680, 500)
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self._configure_style()
        self._build_ui()
        self._select_adjacent_package()
        self.root.after(100, self._poll_events)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10), foreground="#4a4a4a")
        style.configure("Status.TLabel", font=("Segoe UI", 10, "bold"))

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=(24, 20, 24, 20))
        frame.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(7, weight=1)

        ttk.Label(frame, text=APP_NAME, style="Title.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(
            frame,
            text="Install a packaged character Mod with validation and automatic backups.",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 18))

        ttk.Label(frame, text="Forza Horizon 6 folder").grid(row=2, column=0, sticky="w")
        game_row = ttk.Frame(frame)
        game_row.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(4, 12))
        game_row.columnconfigure(0, weight=1)
        self.game_entry = ttk.Entry(game_row, textvariable=self.game_path)
        self.game_entry.grid(row=0, column=0, sticky="ew")
        self.game_button = ttk.Button(game_row, text="Browse...", command=self._browse_game)
        self.game_button.grid(row=0, column=1, padx=(8, 0))

        ttk.Label(frame, text="Mod package (.zip)").grid(row=4, column=0, sticky="w")
        mod_row = ttk.Frame(frame)
        mod_row.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(4, 14))
        mod_row.columnconfigure(0, weight=1)
        self.mod_entry = ttk.Entry(mod_row, textvariable=self.mod_path)
        self.mod_entry.grid(row=0, column=0, sticky="ew")
        self.mod_button = ttk.Button(mod_row, text="Browse...", command=self._browse_mod)
        self.mod_button.grid(row=0, column=1, padx=(8, 0))

        status_row = ttk.Frame(frame)
        status_row.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        status_row.columnconfigure(1, weight=1)
        ttk.Label(status_row, textvariable=self.status, style="Status.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.progress = ttk.Progressbar(status_row, mode="indeterminate", length=180)
        self.progress.grid(row=0, column=1, sticky="e")

        self.log = ScrolledText(
            frame,
            height=10,
            wrap="word",
            font=("Consolas", 9),
            state="disabled",
            background="#ffffff",
            foreground="#202020",
        )
        self.log.grid(row=7, column=0, columnspan=2, sticky="nsew")

        actions = ttk.Frame(frame)
        actions.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        actions.columnconfigure(0, weight=1)
        self.reports_button = ttk.Button(
            actions, text="Open Reports", command=self._open_reports
        )
        self.reports_button.grid(row=0, column=0, sticky="w")
        self.verify_button = ttk.Button(
            actions, text="Verify Package", command=lambda: self._start(dry_run=True)
        )
        self.verify_button.grid(row=0, column=1, padx=(8, 0))
        self.install_button = ttk.Button(
            actions, text="Install Mod", command=lambda: self._start(dry_run=False)
        )
        self.install_button.grid(row=0, column=2, padx=(8, 0))

    def _select_adjacent_package(self) -> None:
        package = discover_adjacent_mod_zip(application_directory())
        if package is not None:
            self.mod_path.set(str(package))
            self._append_log(f"Found Mod package: {package.name}")

    def _browse_game(self) -> None:
        selected = filedialog.askdirectory(title="Select the Forza Horizon 6 folder")
        if selected:
            self.game_path.set(selected)

    def _browse_mod(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select a packaged FH6 Mod",
            filetypes=(("ZIP archives", "*.zip"), ("All files", "*.*")),
        )
        if selected:
            self.mod_path.set(selected)

    def _open_reports(self) -> None:
        directory = report_directory()
        directory.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(directory)  # type: ignore[attr-defined]
        except OSError as exc:
            messagebox.showerror(APP_NAME, friendly_error(exc), parent=self.root)

    def _start(self, *, dry_run: bool) -> None:
        if self.busy:
            return
        game_text = self.game_path.get().strip().strip('"')
        mod_text = self.mod_path.get().strip().strip('"')
        if not game_text or not mod_text:
            messagebox.showwarning(
                APP_NAME,
                "Select both the Forza Horizon 6 folder and the Mod ZIP.",
                parent=self.root,
            )
            return
        game_directory = Path(game_text)
        mod_zip = Path(mod_text)
        if not game_directory.is_dir():
            messagebox.showerror(APP_NAME, "The selected game folder does not exist.", parent=self.root)
            return
        if not mod_zip.is_file():
            messagebox.showerror(APP_NAME, "The selected Mod ZIP does not exist.", parent=self.root)
            return
        if not dry_run:
            confirmed = messagebox.askyesno(
                APP_NAME,
                "Close Forza Horizon 6 before continuing.\n\n"
                "The installer will validate the package, back up each original file, "
                "and then install the Mod. Continue?",
                icon="warning",
                parent=self.root,
            )
            if not confirmed:
                return

        self._set_busy(True)
        self._append_log("Starting package verification..." if dry_run else "Starting installation...")
        worker = threading.Thread(
            target=self._worker,
            args=(game_directory, mod_zip, dry_run),
            daemon=True,
        )
        worker.start()

    def _worker(self, game_directory: Path, mod_zip: Path, dry_run: bool) -> None:
        try:
            result = install_package(
                game_directory,
                mod_zip,
                dry_run=dry_run,
                notify=lambda message: self.events.put(("log", message)),
            )
            self.events.put(("success", result))
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            self.events.put(("error", exc))
        except Exception as exc:  # Keep the window alive for unexpected packaging faults.
            self.events.put(("error", exc))

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self.status.set(str(payload))
                    self._append_log(str(payload))
                elif kind == "success":
                    result = payload
                    assert isinstance(result, InstallResult)
                    message = result_message(result)
                    self.status.set("Verification passed" if result.dry_run else "Installation completed")
                    self._append_log(message.replace("\n\n", "\n"))
                    self._append_log(f"Report: {result.report_path}")
                    self._set_busy(False)
                    messagebox.showinfo(APP_NAME, message, parent=self.root)
                elif kind == "error":
                    assert isinstance(payload, Exception)
                    message = friendly_error(payload)
                    self.status.set("Operation failed")
                    self._append_log(f"ERROR: {message}")
                    self._set_busy(False)
                    messagebox.showerror(APP_NAME, message, parent=self.root)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        state = "disabled" if busy else "normal"
        for widget in (
            self.game_entry,
            self.game_button,
            self.mod_entry,
            self.mod_button,
            self.verify_button,
            self.install_button,
        ):
            widget.configure(state=state)
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()

    def _close(self) -> None:
        if self.busy:
            messagebox.showwarning(
                APP_NAME,
                "Please wait for the current operation to finish before closing the installer.",
                parent=self.root,
            )
            return
        self.root.destroy()


def main() -> int:
    root = Tk()
    InstallerWindow(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
