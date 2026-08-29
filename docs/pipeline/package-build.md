# Mod Package Build

Package one completed Mod as:

- six replacement ZIP files: Display Helmet/Outfit/Alice head and Driver
  Helmet/Outfit/Alice head;
- one aligned `characters.zip` injection payload;
- one `mod.json` describing source character, target character, clothing names,
  targets, hashes, and material dependencies.

When a custom item needs a dedicated runtime asset or must bypass retail
material overrides, declare targeted `xml_patches` inside `mod.json`. Keep the
physical package at the same eight files; do not distribute full copies of game
customization XML files.

The builder must operate on copied workspace assets, preserve ZIP alignment and
extra fields, and emit hashes and a package manifest. A package build never
modifies the game directory.

Package-specific scripts belong under the Mod directory. Shared modelbin, MatI,
Swatch, ZIP, and inspection helpers remain unchanged in `scripts/`.
