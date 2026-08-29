# Display to Driver Character

Start from a game-verified Display baseline. Select the matching Driver Helmet
and Outfit donor containers, remap the Display component to the Driver-local
skeleton, preserve validated materials where possible, and suppress the native
Driver head only in the target Alice head asset.

Display and Driver may share geometry when their donor contracts and coordinate
frames are identical, but each target ZIP remains a separate package and is
validated against its own modelbin contract.

Driver-only scripts belong under the Mod directory. The shared retarget and
container helpers in `scripts/` are not edited in place.
