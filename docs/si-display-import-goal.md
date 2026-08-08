# Si Display character import: delivery goal

## Outcome

Build a complete, reproducible, and reversible FH6 **Display-character** import
package from the native Si FBX. The package is complete only when all logical
components, all supported LODs, runtime materials, patched modelbins, copied
Forza archives, verification evidence, and install/restore scripts are present
and the character passes an in-game Display test.

`_Driver` assets are outside this milestone. PMX is reference-only and must not
provide new geometry, bind pose, skin weights, or LODs.

The game directory remains read-only until the user explicitly authorizes a
deployment run.

## Locked inputs and invariants

- Native source: `sources/si/source.config.json`.
- FBX import: `global_scale=100`; every geometric and skeletal measurement is
  evaluated in armature `REST` state.
- Preserve the immutable FBX, texture set, donor modelbins, donor ZIPs, and
  baseline `.blend` files. Work only on versioned copies.
- Preserve native source component and LOD boundaries through retargeting.
- Logical runtime roles remain separate: `Head`, `Hair`, `Body`, `Garment`.
- Each exported vertex/index domain contains at most 65,535 vertices.
- Every exported vertex has one to four valid component-local bone influences;
  weights are pruned, quantized, and normalized.
- The first reliable modelbin writer retains the selected donor `Skel`, `VLay`,
  and `MatI`. It changes only the buffers and the metadata that must follow them.

## Required package contents

The final versioned package must contain:

1. A source-lock manifest with hashes, import settings, coordinate convention,
   selected donor versions, and tool versions.
2. Immutable FBX and donor baselines plus versioned Blender milestones.
3. Authoritative source-to-donor bone maps with `"source_format": "fbx"`,
   explicit core, facial, twist, corrective, secondary, and fallback mappings.
4. Head, Hair, Body, and Garment export manifests for LOD0-LOD3, including
   source mesh provenance and every intentionally omitted VFX/shadow proxy.
5. Inspectable position/attribute vertex buffers, index buffers, skin buffers,
   draw ranges, material assignments, bounds, bone palettes, and LOD metadata.
6. Converted textures and a material manifest describing the source channel,
   color space, Alpha handling, FH6 template, swizzle, and runtime slot.
7. Patched Display-component modelbins and structural round-trip reports.
8. Copied/rebuilt Forza archives with their required ZIP extra fields intact.
9. REST and pose reports, seam metrics, validation renders, and a defect matrix.
10. Idempotent install and restore scripts, a deployment manifest, backups, and
    a concise in-game Display test checklist.

## Component and donor contract

The audit must freeze one physical package/container and one semantic reference
for each logical role before retargeting begins. Current candidates are:

| Logical role | Planned physical container | Semantic references |
| --- | --- | --- |
| Head | `Helmet_Race_Modern` package | `DRV_BA_F_01_Face`, `Driver_Alice_F` |
| Hair and head accessories | `Helmet_Race_Modern` package | compatible Hair shader/container plus Head rig |
| Body | `Outfit_Race_Suit_Modern_F` package | `Female`, `Driver_Alice_F` |
| Garment, gloves, footwear | `Outfit_Race_Suit_Modern_F` package | race-suit corrective weights and draw layout |

Logical components stay independently inspectable even if Head/Hair or
Body/Garment are ultimately packed into the same replaceable archive.

## Hard defect gates

### Eyes, eyelids, and face

Known causes are incorrect dark sclera material, flattened iris Alpha, rigid
sclera-to-eye weighting, and incomplete eyelid/facial mapping.

Required solution:

- keep sclera, iris, eyelashes, eye shadow, teeth, and face as distinct draw
  groups where their shaders or Alpha modes differ;
- use an opaque neutral sclera material and an iris material that preserves the
  original texture and Alpha distribution;
- bind eye rotation to correctly fitted eye pivots while transferring eyelid,
  socket, brow, cheek, jaw, and lash weights from the dense face donor;
- verify open, blink, look-left/right, and look-up/down poses.

Pass condition: both eyes render in all required poses; no black socket, missing
iris, flattened Alpha, obvious eye/socket void, or eyelash separation is visible,
and measured socket clearance does not regress beyond the chosen donor baseline.

### Hair, forehead, and head ornaments

Hair roots and ornaments must be assigned to the Head/Hair package. Hair should
not be globally translated merely to hide an oversized forehead.

Required solution:

- fit the hairline in donor head rest space, then weight roots to Head and map
  driven secondary chains where available;
- map ornaments to compatible helmet/head accessory bones, or document a
  deliberate stable Head fallback when no driven secondary bone exists;
- keep opaque hair, transparent cards, hair shadow, and ornaments in compatible
  draw/material passes.

Pass condition: hairline coverage is correct from front and side views, all
ornaments render, and head rotation causes no scalp gap, floating parts, or
detached transparent cards.

### Neck and face/body junction

The previous 12% radial expansion tore shared boundary points. It is prohibited.

Required solution:

- fit face and upper neck without radial inflation;
- weld truly shared boundaries and create a real two-to-four-ring neck bridge
  where the components do not share topology;
- match ring position, normals, tangents, UV continuity where applicable, and
  blend Head/Neck/Spine weights by topological distance;
- place the lower bridge safely beneath the collar rather than leaving an empty
  strip.

Pass condition: REST ring mismatch is below the project seam tolerance, and
idle, turn, tilt, and look-up poses show no visible hole, lighting seam, or
collar exposure. The tolerance and donor baseline are recorded in the report.

### Wrists, hands, ankles, and feet

The source and old donor bind centers differed by roughly 156 mm, while old
conversion paths collapsed twist/corrective semantics. Simple reassignment to
Hand or Foot is prohibited as a final fix.

Required solution:

- fit shoulder-elbow-wrist-finger and hip-knee-ankle-toe chains in REST space;
- explicitly map forearm/leg twist, wrist up/down corrective, foot/toe, and any
  compatible ankle corrective weights;
- synchronize paired skin/sleeve/glove and skin/trouser/shoe seam rings;
- prune only after the fitted transfer, keeping the strongest four normalized
  influences and logging discarded influence mass.

Pass condition: full elbow/wrist/finger and knee/ankle/toe pose suites show no
large bend, corkscrew deformation, detached seam, or foot inversion; seam and
volume metrics do not regress beyond the selected production donor baseline.

### Materials, textures, Alpha, and plastic appearance

Replacing diffuse alone is prohibited. The conversion must consume the source
`D/N/P/E/ST/HN/RS` semantics and original Alpha.

Required solution:

- record every source-to-runtime channel, swizzle, color space, and Alpha mode;
- use shader/template classes appropriate to Skin, Hair, Iris, Sclera,
  Hair Shadow, opaque cloth, and Alpha cloth;
- verify normal-map orientation before use and construct a neutral roughness /
  specular baseline before restoring channels one at a time;
- remove all unintended donor diffuse, logo, normal, mask, specular, detail, and
  tint references. The old constant suit roughness near `0.041` is not accepted.

Pass condition: no donor texture leakage remains, source colors and Alpha are
reproduced, skin/hair/cloth respond distinctly to light, and comparison renders
show neither the race-suit palette nor the former uniform plastic response.

## LOD and buffer requirements

- Produce traceable Head/Hair/Body/Garment mappings for native LOD0-LOD3.
- LOD counts must decrease monotonically unless a documented source exception
  exists; material/draw semantics and bone palettes remain compatible.
- Excluded effect meshes and shadow proxies are recorded rather than silently
  discarded.
- Each intermediate buffer has a manifest containing format, stride, count,
  byte size, hash, component, LOD, draw range, source mesh, and target modelbin
  block/range.
- Bounds are recomputed from exported geometry and checked in both local and
  package space.

## Validation ladder

1. **Source gate:** hashes and source metadata match the locked baseline.
2. **Data gate:** zero hard probe errors, zero invalid indices, zero unresolved
   bone indices, zero vertices over four influences, no missing required UVs or
   material slots, and negligible REST bind reconstruction error.
3. **Seam gate:** eye socket, face/neck, wrist, and ankle metrics are emitted for
   REST and the pose suite and compared with donor baselines.
4. **Pose gate:** saved tests cover face/eyes, neck, spine, shoulders, elbows,
   wrists, fingers, hips, knees, ankles, toes, hair roots, and garment motion.
5. **Visual gate:** front/back/side, face, both eyes, neck, both hands, both feet,
   Alpha edges, and neutral-light material renders are saved for inspection.
6. **Round-trip gate:** every patched modelbin parses; offsets, sizes, counts,
   mesh ranges, bounds, skeleton references, and unchanged block equivalence are
   verified. Every copied ZIP passes structural and extra-field checks.
7. **Game gate:** after explicit deployment authorization, the Display page
   loads without an infinite spinner; all parts, textures, Alpha, LODs, and poses
   render correctly. Offline gates do not substitute for this test.

## Definition of done

The goal is not complete when a mesh merely loads. It is complete only when the
versioned import package and restore package exist, all offline gates pass, and
the user confirms the in-game Display gate with:

- correct face, visible eyes, eyelashes, hair, and head ornaments;
- no neck hole or visible face/body seam;
- stable hands, wrists, feet, and ankles throughout animation;
- correct source colors, textures, transparency, and non-plastic material
  response;
- no donor race-suit/helmet texture leakage and no infinite loading.

