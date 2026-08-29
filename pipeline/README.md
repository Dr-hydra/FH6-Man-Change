# FH6 FBX-to-Mod Pipeline

This repository keeps `scripts/` as the shared, reusable script library. Shared
scripts are treated as stable APIs: a Mod must copy a shared script into its own
`mods/<mod-id>/scripts/` directory before making a project-specific change.

The reusable workflow is split into four lines:

1. `docs/pipeline/fbx-to-display.md`
2. `docs/pipeline/display-to-driver.md`
3. `docs/pipeline/package-build.md`
4. `docs/pipeline/installer.md`

The `pipeline/` directory contains indexes and policy only. It is deliberately
not a second copy of the shared Python implementation.
