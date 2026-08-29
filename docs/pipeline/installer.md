# FH6 Mod Installer

`scripts/install_fh6_mod.py` is the shared CLI entry point. It accepts either
the game root or its `media` directory, detects the media path, validates one
unique `mod.json`, checks target hashes and swatch expectations, creates a
timestamped backup, and applies the six replacements plus the characters
injection.

The optional `xml_patches` manifest section supports narrowly scoped
`clone_element` and `replace_element` operations. The installer validates the
source selectors and final XML before changing any file, preserves all unrelated
XML text byte-for-byte, and backs up each XML target beside the original. Mods
must use these targeted operations instead of distributing a complete game XML.

`scripts/install_fh6_mod_gui.py` is the shared English Tkinter GUI. It exposes
the same validation and deployment core through game-folder and Mod-ZIP
pickers, a read-only verification action, an explicit install confirmation,
an activity log, and reports under `%LOCALAPPDATA%\FH6ModInstaller\Reports`.
When exactly one compatible Mod ZIP is beside the executable, the GUI selects
it automatically.

Build the standalone Windows executable with:

```powershell
python scripts/build_fh6_mod_gui_installer.py --output-dir releases/installer
```

The default build uses PyInstaller `--onefile --windowed --uac-admin`, because
Steam libraries under Program Files may require elevation. A Nexus upload can
place the executable, the unchanged Mod package ZIP, and a plain-English README
inside one outer archive. Package-specific Nexus assembly scripts belong under
`mods/<mod-id>/scripts/installer/`.

The installer supports `--dry-run`. Restore is a separate explicit operation;
the installer must never silently overwrite an existing backup or infer a
different game target.

If a Mod needs installer behavior beyond the shared contract, copy the installer
script into that Mod's own `scripts/installer/` directory and modify the copy.
