# Si FBX-first workflow

The complete Display-package scope and acceptance gates are defined in
`docs/si-display-import-goal.md`.

The primary Si source is the native FBX export configured in
`sources/si/source.config.json`. PMX remains available only as a compatibility
and comparison source.

## Source policy

- Import FBX with `global_scale=100`.
- Force every imported armature to `REST` pose before inspecting or copying
  geometry. The FBX contains non-identity pose transforms that must not become
  the conversion bind pose.
- Preserve all native LODs and shadow proxies in the immutable baseline.
- Build FH6 working components from native LOD0-LOD3 meshes. Retarget LOD0
  first, but keep each lower LOD independently traceable and probe-clean.
- Exclude `vfxpart`, effect-only `cloth_04` through `cloth_09`, and shadow
  proxies from the first FH6 modelbin milestone.
- Keep PMX scripts for legacy inspection. Do not use PMX-derived bones or
  weights for new retargeting work.

## Build the source milestones

```powershell
& .\scripts\build_si_fbx_source.ps1 -AllLods
```

This command creates:

- an immutable FBX baseline containing all 80 meshes and the original rig;
- four component source scenes containing the native LOD0-LOD3 Head, Hair,
  Body, and Garment meshes;
- probe and metadata reports for the baseline and every component scene.

The command refuses to overwrite any existing milestone. Pass `-ReuseBaseline`
and `-ReuseComponents` to verify and preserve configured existing outputs while
building only missing LOD milestones.

After the source scenes exist, freeze the Display donor/container contract:

```powershell
python .\scripts\build_si_display_contract.py
```

The contract records the two retained physical skeletons, semantic-only donors,
all four LOD manifests, required block counts, skeleton topology hashes, and
the first-writer block policy. A nonzero hard-error count stops the next stage.

## Component mapping

| FBX source | FH6 role |
| --- | --- |
| `body_01` | Body |
| `face_01`, `iris_01`, `brow_01`, `eyeshadow_01` | Head |
| `hair_01`, `hairshadow_01` | Hair |
| `cloth_01`, `cloth_02`, `cloth_03` | Garment |
| `vfxpart_*`, `cloth_04` through `cloth_09` | Deferred effects |
| `shadowProxyDesktop` | Reference-only shadow geometry |

## Retargeting contract

The FBX source rig is not an FH6 skeleton. Align each source limb chain to the
selected FH6 donor in rest space before transferring weights. Preserve the FBX
twist and corrective semantics when mapping wrists and ankles. Keep head,
hair, body, and garment in separate donor-local skeleton domains.
Every new bone-map JSON must declare `"source_format": "fbx"`; the retargeter
rejects legacy maps that do not identify their source format.

The native FBX materials are also not FH6 materials. Use the accompanying
`D/N/P/E/ST/HN/RS` textures as semantic inputs and build swatches from matching
FH6 material templates. Do not treat Blender's automatically connected FBX
nodes as the runtime material definition.

## PMX compatibility

The legacy PMX baseline is retained under `sources/si/v1/original`. It is useful
for its MMD expression morphs and historical material names, but new geometry,
bind-pose, LOD, and skin-weight work must originate from the FBX baseline.
