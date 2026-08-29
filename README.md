# FH6 Character Mod Pipeline

This repository contains only the reusable FH6 model-processing pipeline:
general documentation, scripts, tools, tests, and Skills. Mod source code,
character assets, game files, and finished Mod packages are not part of the
open-source repository.

## Workflow

1. [FBX to Display](docs/pipeline/fbx-to-display.md)
2. [Display to Driver](docs/pipeline/display-to-driver.md)
3. [Package build](docs/pipeline/package-build.md)
4. [Package and deployment contract](docs/pipeline/installer.md)

Shared script ownership and the entry points for each line are recorded in
`pipeline/scripts.manifest.json`.

## Repository scope

- `scripts/`: shared, character-agnostic helpers and stable entry points
- `tools/`: reusable command-line tools
- `tests/`: tests using synthetic or redacted fixtures
- `docs/`: general format, pipeline, and validation documentation
- `skills/`: reusable FH6 Skills for the pipeline
- `pipeline/`: script ownership and workflow policy

Mod-specific sources, scripts, baselines, archives, and release payloads are
kept outside this public repository. Finished Mods are distributed separately
through the project's Release artifacts.

The user-facing installer and deployment workflow are maintained by the
separate `FH6Tools` project (local development checkout:
`E:\Dr.Hydra\FH6Tools`). This repository documents the package contract that
the installer consumes; it does not own the installer application.

## License and attribution

Unless a file states otherwise, the documentation and Skills in this
repository are provided under [CC BY-NC 4.0](LICENSE-DOCS), and reusable code,
scripts, tools, and tests are provided under the same non-commercial terms in
[LICENSE](LICENSE). Redistributions must retain the author attribution,
project URL, and license notice, and must describe material changes.

This repository does not grant rights to Forza Horizon 6 files, extracted game
assets, or third-party character assets. Those materials are intentionally
excluded from the public source tree and remain subject to their own terms.
