# FBX to Display Character

Use the native FBX as the source for a Display-character model. Keep the FBX
component and LOD boundaries, import in REST pose, retarget to the FH6 donor
skeleton, and export intermediate vertex/index/Skin buffers before patching a
donor modelbin.

The line produces a Display Head/Hair carrier, Body/Garment carrier, material
swatches, a modelbin contract report, and a copied ZIP candidate. PMX is only a
compatibility/reference input when FBX evidence is insufficient.

Project-specific repairs belong under `mods/<mod-id>/scripts/fbx-to-display/`.
Shared import, inspection, and modelbin helpers remain under `scripts/`.
