# Shared Tests

Tests in this directory must use only shared scripts and `samples/` fixtures.
Tests that assert a character's names, Swatch GUIDs, material profile, donor
choice, or repair milestone belong under that Mod's `tests/` directory.

The current Si-specific tests are in
`mods/si-sakura-female-display-driver/tests/`. Python `__pycache__` directories
are disposable runtime output and are not test sources.
