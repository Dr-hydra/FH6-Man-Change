#!/usr/bin/env python3
"""Build a seven-draw Helmet structural donor with a sclera material slot."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import inspect_modelbin as inspector
from modelbin_bundle import (
    BlobSpec,
    MetadataSpec,
    blob_spec,
    parse_bundle,
    rebuild_with_blob_sequence,
)
from patch_fh6_material_profile import materials_by_id, shader_info
from patch_fh6_racesuit_materials import material_id


SCLERA_MATERIAL_ID = 6
SCLERA_RENDER_PASS = 0x19
SCLERA_TEMPLATE_MATERIAL_ID = 2
SCLERA_MESH_TEMPLATE_MATERIAL_ID = 3


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("helmet", type=Path)
    parser.add_argument("alice", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def with_metadata(spec: BlobSpec, **values: bytes) -> BlobSpec:
    unknown = set(values)
    metadata = []
    for item in spec.metadata:
        value = values.get(item.tag, item.value)
        metadata.append(MetadataSpec(item.tag, item.version, value))
        unknown.discard(item.tag)
    if unknown:
        raise ValueError(f"Template is missing metadata fields: {sorted(unknown)}")
    return replace(spec, metadata=tuple(metadata))


def patch_mesh_identity(data: bytes, material_id_value: int, render_pass: int) -> bytes:
    if len(data) < 18:
        raise ValueError("Mesh payload is too small")
    result = bytearray(data)
    struct.pack_into("<h", result, 2, material_id_value)
    struct.pack_into("<H", result, 14, render_pass)
    return bytes(result)


def patch_model_counts(data: bytes, meshes: int, materials: int) -> bytes:
    if len(data) < 8:
        raise ValueError("Modl payload is too small")
    result = bytearray(data)
    struct.pack_into("<h", result, 0, meshes)
    struct.pack_into("<h", result, 6, materials)
    return bytes(result)


def main() -> int:
    args = arguments()
    helmet_path = args.helmet.resolve(strict=True)
    alice_path = args.alice.resolve(strict=True)
    output_path = args.output.resolve()
    report_path = args.report.resolve()
    for path in (output_path, report_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    helmet_data = helmet_path.read_bytes()
    alice_data = alice_path.read_bytes()
    helmet = parse_bundle(helmet_data)
    alice = parse_bundle(alice_data)
    helmet_materials = materials_by_id(helmet)
    alice_materials = materials_by_id(alice)
    if set(helmet_materials) != set(range(6)):
        raise ValueError(f"Helmet donor material IDs differ from 0..5: {sorted(helmet_materials)}")
    if SCLERA_TEMPLATE_MATERIAL_ID not in alice_materials:
        raise ValueError("Alice donor has no Mat_Eyes material")
    sclera_template = alice_materials[SCLERA_TEMPLATE_MATERIAL_ID]
    shader, atst, _nested, _mtpr = shader_info(sclera_template.data)
    if shader != "charactereyeball" or atst != "0000":
        raise ValueError(f"Unexpected Alice eye shader {shader}/{atst}")

    helmet_inspection = inspector.inspect(helmet_path)
    if helmet_inspection["parsed"]["errors"]:
        raise ValueError(f"Helmet parser errors: {helmet_inspection['parsed']['errors']}")
    meshes = helmet_inspection["parsed"]["meshes"]
    mesh_blobs = [blob for blob in helmet.blobs if blob.tag == "Mesh"]
    if len(meshes) != 6 or len(mesh_blobs) != 6:
        raise ValueError("Helmet donor must have six Mesh blobs")
    mesh_by_material = {
        int(mesh["material_id"]): blob for mesh, blob in zip(meshes, mesh_blobs)
    }
    mesh_template = mesh_by_material[SCLERA_MESH_TEMPLATE_MATERIAL_ID]

    sclera_material = with_metadata(
        blob_spec(sclera_template),
        **{"Name": b"Mat_Eyes", "Id  ": struct.pack("<I", SCLERA_MATERIAL_ID)},
    )
    sclera_mesh = replace(
        blob_spec(mesh_template),
        data=patch_mesh_identity(
            mesh_template.data, SCLERA_MATERIAL_ID, SCLERA_RENDER_PASS
        ),
        trailing_size=None,
    )

    specs: list[BlobSpec] = []
    inserted_material = False
    inserted_mesh = False
    for blob in helmet.blobs:
        if blob.tag == "Mesh" and not inserted_material:
            specs.append(sclera_material)
            inserted_material = True
        if blob.tag != "Mesh" and inserted_material and not inserted_mesh:
            specs.append(sclera_mesh)
            inserted_mesh = True
        spec = blob_spec(blob)
        if blob.tag == "Modl":
            spec = replace(
                spec,
                data=patch_model_counts(spec.data, meshes=7, materials=7),
                trailing_size=None,
            )
        specs.append(spec)
    if not inserted_material or not inserted_mesh:
        raise ValueError("Could not locate Helmet MatI/Mesh insertion points")

    output_data = rebuild_with_blob_sequence(helmet, specs)
    output_path.write_bytes(output_data)
    output_bundle = parse_bundle(output_data)
    output_inspection = inspector.inspect(output_path)
    parsed = output_inspection["parsed"]
    if parsed["errors"]:
        raise ValueError(f"Output parser errors: {parsed['errors']}")
    output_materials = materials_by_id(output_bundle)
    output_meshes = parsed["meshes"]
    model = parsed["model"][0]
    sclera_meshes = [
        mesh for mesh in output_meshes if int(mesh["material_id"]) == SCLERA_MATERIAL_ID
    ]
    if set(output_materials) != set(range(7)):
        raise ValueError(f"Output material IDs differ from 0..6: {sorted(output_materials)}")
    if len(output_meshes) != 7 or len(sclera_meshes) != 1:
        raise ValueError("Output does not contain seven meshes and one sclera draw")
    if int(model["meshes"]) != 7 or int(model["materials"]) != 7:
        raise ValueError(f"Output Modl counts are inconsistent: {model}")
    sclera_shader, sclera_atst, _nested, sclera_mtpr = shader_info(
        output_materials[SCLERA_MATERIAL_ID].data
    )
    if sclera_shader != "charactereyeball" or sclera_atst != "0000":
        raise ValueError("Output sclera MatI changed shader identity")
    sclera_mesh_info = sclera_meshes[0]
    if (
        int(sclera_mesh_info["vertex_layout_id"]) != 0
        or int(sclera_mesh_info["render_pass"]) != SCLERA_RENDER_PASS
    ):
        raise ValueError(f"Output sclera Mesh binding is invalid: {sclera_mesh_info}")

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "purpose": "Seven-draw Helmet structural donor with a dedicated opaque sclera material.",
        "inputs": {
            "helmet": {
                "path": str(helmet_path),
                "bytes": len(helmet_data),
                "sha256": sha256_path(helmet_path),
            },
            "alice": {
                "path": str(alice_path),
                "bytes": len(alice_data),
                "sha256": sha256_path(alice_path),
            },
        },
        "output": {
            "path": str(output_path),
            "bytes": len(output_data),
            "sha256": sha256_bytes(output_data),
            "blob_tags": output_bundle.blob_tags,
            "model": model,
        },
        "sclera": {
            "material_id": SCLERA_MATERIAL_ID,
            "shader": sclera_shader,
            "atst": sclera_atst,
            "parameter_count": int(sclera_mtpr.data[0]),
            "mesh_blob_index": int(sclera_mesh_info["blob_index"]),
            "vertex_layout_id": int(sclera_mesh_info["vertex_layout_id"]),
            "render_pass": int(sclera_mesh_info["render_pass"]),
            "source_material_payload_sha256": sha256_bytes(sclera_template.data),
            "output_material_payload_sha256": sha256_bytes(
                output_materials[SCLERA_MATERIAL_ID].data
            ),
        },
        "validation": {
            "outer_bundle": True,
            "modelbin_parser": True,
            "mesh_count": len(output_meshes),
            "material_count": len(output_materials),
            "material_ids": sorted(output_materials),
            "game_validated": False,
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FH6_HEAD7_DONOR=" + json.dumps(report["output"], separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
