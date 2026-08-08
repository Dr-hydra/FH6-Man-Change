# FH6 character model format notes

These notes describe structures verified against retail FH6 character assets. The files were read
directly from `media/Cinematic_Assets/Characters`; no game files were changed.

## Archive layer

Character assets are ZIP archives containing one `.modelbin` entry. Tested archives use normal
Deflate compression and can be extracted with the Python standard library or .NET `ZipArchive`.
They include a large custom ZIP extra field, but the tested payloads are not encrypted.

The separate ForzaCryptoTool Method 22 path is therefore not needed for these tested character
archives. Do not assume this applies to every FH6 ZIP category.

Example:

```powershell
python scripts/extract_modelbin.py `
  "D:\SteamLibrary\steamapps\common\ForzaHorizon6\media\Cinematic_Assets\Characters\Head\DRV_BA_F_01.zip" `
  samples\modelbin
```

## Model container

The `.modelbin` starts with a little-endian `Grub` bundle. All tested files use bundle version 1.1.
The bundle contains a fixed-size table of typed blobs whose data and metadata use absolute file
offsets.

Observed blob tags:

| Tag | Purpose |
| --- | --- |
| `Skel` | Bone names, hierarchy, and 4x4 transforms |
| `MatI` | Material instance |
| `Mesh` | Draw range, LOD, render pass, buffer bindings |
| `IndB` | Index buffer |
| `VLay` | D3D12 input layout |
| `VerB` | Vertex buffer |
| `Skin` | Skin weights and bone indices |
| `Modl` | Model-level counts and LOD flags |

Run the standalone inspector with:

```powershell
python scripts/inspect_modelbin.py samples\modelbin\DRV_BA_F_01.modelbin
python scripts/inspect_modelbin.py samples\modelbin\DRV_BA_F_01.modelbin --bones
python scripts/inspect_modelbin.py samples\modelbin\DRV_BA_F_01.modelbin --json
```

## Verified samples

| Sample | Bones | Meshes | Vertex layouts | Vertex count | Index count |
| --- | ---: | ---: | ---: | ---: | ---: |
| `DRV_BA_F_01` head | 332 | 6 | 3 | 8,607 positions | 50,202 |
| `Average_Female_Driver` body | 328 | 2 | 1 | 11,928 | 128,952 |
| `Upper_Shirt_Tucked_N_Driver` | 197 | 3 | 1 | 6,151 | 31,341 |
| `Hair_Bald_Driver` | 195 | 1 | 1 | 4 | 6 |

The head mesh names are `EyeAO`, `Teeth`, `Face`, `Eyelashes`, `Eyelashes_flipped`, and `Eyes`.
The body and shirt contain separate `LODS` and `LOD0` draw ranges.

## Vertex data

Positions use input slot 0 with `DXGI_FORMAT_R16G16B16A16_SNORM` (format 13). Other attributes are
interleaved in slot 1. Observed formats include:

| Semantic | DXGI format |
| --- | --- |
| `POSITION0` | 13, `R16G16B16A16_SNORM` |
| `NORMAL0` | 37, `R16G16_SNORM` |
| `TEXCOORD0..4` | 35, `R16G16_UNORM` |
| `TANGENT0..4` | 24, `R10G10B10A2_UNORM` |
| `COLOR0` | 28, `R8G8B8A8_UNORM` |

Index buffers use format 57 (`R16_UINT`) and a two-byte stride in all tested samples.

Mesh descriptors contain per-mesh position scale and translation vectors. These are required to
dequantize the signed-normalized position data.

## Skinning

Each component embeds its own skeleton or skeleton subset. Bone indices in the `Skin` blob refer to
that component-local table, so a garment replacement should preserve the garment skeleton table and
its index mapping.

Tested `Skin` buffers use format 34 (`R16G16_FLOAT`). Each four-byte pair stores a half-float weight
and a half-float bone index. The stride determines the number of influences:

| Mesh | Stride | Influences | Observed weight sum |
| --- | ---: | ---: | ---: |
| Face | 16 | 4 | 0.999649 to 1.000404 |
| Body | 16 | 4 | 0.999634 to 1.000366 |
| Shirt | 16 | 4 | 0.999634 to 1.000366 |
| Eyes/teeth/simple hair | 4 | 1 | 1.0 |

The head skeleton contains named facial bones for brows, cheeks, eyes, jaw, tongue, and lips. It is
not just a conventional body rig.

## Remaining work

The next useful milestone is a geometry exporter that:

1. Resolves each mesh's index range and vertex buffer bindings.
2. Applies mesh scale/translation to position data.
3. Decodes packed normals, tangents, UVs, and colors.
4. Maps component-local skin indices to bone names.
5. Writes glTF with meshes, joints, inverse bind matrices, and LOD metadata.

Material payloads and morph buffers have not yet been decoded in this project. The current parser
reports their blob metadata but deliberately does not guess their FH6 layout.
