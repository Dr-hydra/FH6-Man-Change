#!/usr/bin/env python3
"""Import an FH6 character donor modelbin into Blender as a read-only baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from inspect_modelbin import parse_bundle, parse_known_blobs


FORMAT_SIZES = {
    13: 8,  # R16G16B16A16_SNORM
    24: 4,  # R10G10B10A2_UNORM
    28: 4,  # R8G8B8A8_UNORM
    34: 4,  # R16G16_FLOAT
    35: 4,  # R16G16_UNORM
    37: 4,  # R16G16_SNORM
    57: 2,  # R16_UINT
}


def arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelbin", required=True, type=Path)
    parser.add_argument("--blend", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--preview", required=True, type=Path)
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clear_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)
    for material in list(bpy.data.materials):
        bpy.data.materials.remove(material)


def snorm16(data: bytes, offset: int) -> float:
    value = struct.unpack_from("<h", data, offset)[0]
    return max(-1.0, value / 32767.0)


def matrix_multiply(left: list[float], right: list[float]) -> list[float]:
    result = [0.0] * 16
    for row in range(4):
        for column in range(4):
            result[row * 4 + column] = sum(
                left[row * 4 + inner] * right[inner * 4 + column]
                for inner in range(4)
            )
    return result


def axis_convert(point: tuple[float, float, float] | list[float]) -> tuple[float, float, float]:
    return (-point[0], -point[2], point[1])


def transform_vector_row(vector: list[float], matrix: list[float], translate: bool) -> list[float]:
    result = [0.0, 0.0, 0.0]
    for output_axis in range(3):
        result[output_axis] = sum(vector[input_axis] * matrix[input_axis * 4 + output_axis] for input_axis in range(3))
        if translate:
            result[output_axis] += matrix[12 + output_axis]
    return result


def skeleton_world_matrices(bones: list[dict]) -> list[list[float]]:
    matrices: list[list[float]] = []
    for index, bone in enumerate(bones):
        local = [float(value) for value in bone["matrix"]]
        parent = bone["parent"]
        if parent >= index:
            raise ValueError(f"bone {index} has non-previous parent {parent}")
        matrices.append(matrix_multiply(local, matrices[parent]) if parent >= 0 else local)
    return matrices


def create_armature(bones: list[dict], world_matrices: list[list[float]], collection: bpy.types.Collection) -> tuple[bpy.types.Object, list[str]]:
    names = [bone["name"] for bone in bones]
    if len(set(names)) != len(names):
        duplicates = [name for name, count in Counter(names).items() if count > 1]
        raise ValueError(f"duplicate donor bone names: {duplicates}")

    armature_data = bpy.data.armatures.new("Upper_Shirt_Tucked_N_Driver_Skeleton")
    armature = bpy.data.objects.new("Upper_Shirt_Tucked_N_Driver_Skeleton", armature_data)
    collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")

    heads = [Vector(axis_convert((matrix[12], matrix[13], matrix[14]))) for matrix in world_matrices]
    edit_bones: list[bpy.types.EditBone] = []
    for index, bone in enumerate(bones):
        edit_bone = armature_data.edit_bones.new(bone["name"])
        head = heads[index]
        candidate_tail: Vector | None = None
        child = bone["first_child"]
        if 0 <= child < len(heads) and (heads[child] - head).length > 1e-5:
            candidate_tail = heads[child]
        if candidate_tail is None:
            parent = bone["parent"]
            if 0 <= parent < len(heads) and (head - heads[parent]).length > 1e-5:
                direction = (head - heads[parent]).normalized()
                candidate_tail = head + direction * max(0.02, min(0.12, (head - heads[parent]).length * 0.35))
        if candidate_tail is None:
            candidate_tail = head + Vector((0.0, 0.0, 0.04))
        edit_bone.head = head
        edit_bone.tail = candidate_tail
        edit_bone.use_connect = False
        edit_bones.append(edit_bone)

    for index, bone in enumerate(bones):
        if bone["parent"] >= 0:
            edit_bones[index].parent = edit_bones[bone["parent"]]
    bpy.ops.object.mode_set(mode="OBJECT")
    armature.select_set(False)

    for index, bone in enumerate(bones):
        data_bone = armature_data.bones[bone["name"]]
        data_bone["fh6_index"] = index
        data_bone["fh6_parent_index"] = bone["parent"]
        data_bone["fh6_local_matrix"] = [float(value) for value in bone["matrix"]]
    armature["fh6_display_skeleton"] = True
    armature["fh6_bind_matrix_reconstruction_pending"] = True
    armature.show_in_front = True
    return armature, names


def layout_elements(layout: dict) -> dict[str, dict]:
    offsets: defaultdict[int, int] = defaultdict(int)
    result: dict[str, dict] = {}
    for element in layout["elements"]:
        fmt = element["format"]
        if fmt not in FORMAT_SIZES:
            raise ValueError(f"unsupported vertex format {fmt}")
        slot = element["input_slot"]
        key = f"{element['semantic']}{element['semantic_index']}"
        result[key] = {**element, "byte_offset": offsets[slot]}
        offsets[slot] += FORMAT_SIZES[fmt]
    return result


def buffer_maps(data: bytes, blobs: list, parsed: dict) -> tuple[dict[int, dict], dict[int, dict], dict[int, dict]]:
    blob_by_index = {blob.index: blob for blob in blobs}

    def build(items: list[dict]) -> dict[int, dict]:
        result = {}
        for item in items:
            if item["id"] in result:
                raise ValueError(f"duplicate buffer id {item['id']}")
            result[item["id"]] = {**item, "blob": blob_by_index[item["blob_index"]]}
        return result

    return build(parsed["index_buffers"]), build(parsed["vertex_buffers"]), build(parsed["skin_buffers"])


def decode_skin(data: bytes, skin: dict, source_vertex: int, bone_count: int) -> list[tuple[int, float]]:
    if skin.get("format") != 34 or skin["stride"] % 4:
        raise ValueError(f"unsupported skin buffer format/stride: {skin.get('format')}/{skin['stride']}")
    if not 0 <= source_vertex < skin["count"]:
        raise ValueError(f"skin vertex {source_vertex} outside 0..{skin['count'] - 1}")
    offset = skin["payload_offset"] + source_vertex * skin["stride"]
    influences = []
    for influence_index in range(skin["stride"] // 4):
        weight, raw_bone_index = struct.unpack_from("<ee", data, offset + influence_index * 4)
        if weight <= 0:
            continue
        bone_index = int(round(raw_bone_index))
        if abs(raw_bone_index - bone_index) > 1e-3 or not 0 <= bone_index < bone_count:
            raise ValueError(f"invalid donor bone index {raw_bone_index} at vertex {source_vertex}")
        influences.append((bone_index, float(weight)))
    return influences


def create_materials(blobs: list) -> dict[int, bpy.types.Material]:
    result = {}
    palette = [(0.24, 0.42, 0.66, 1.0), (0.08, 0.10, 0.13, 1.0), (0.55, 0.55, 0.55, 1.0)]
    for blob in blobs:
        if blob.tag != "MatI" or blob.identifier is None:
            continue
        material = bpy.data.materials.new(blob.name or f"MatI_{blob.identifier}")
        material.diffuse_color = palette[blob.identifier % len(palette)]
        result[blob.identifier] = material
    return result


def create_mesh_object(
    data: bytes,
    mesh_report: dict,
    layout_report: dict,
    index_buffers: dict[int, dict],
    vertex_buffers: dict[int, dict],
    skin_buffers: dict[int, dict],
    world_matrices: list[list[float]],
    bone_names: list[str],
    armature: bpy.types.Object,
    material: bpy.types.Material | None,
    collection: bpy.types.Collection,
) -> dict:
    elements = layout_elements(layout_report)
    position_element = elements.get("POSITION0")
    normal_element = elements.get("NORMAL0")
    uv_element = elements.get("TEXCOORD0")
    if not position_element or position_element["format"] != 13:
        raise ValueError("donor POSITION0 is not R16G16B16A16_SNORM")
    if not normal_element or normal_element["format"] != 37:
        raise ValueError("donor NORMAL0 is not R16G16_SNORM")
    if not uv_element or uv_element["format"] != 35:
        raise ValueError("donor TEXCOORD0 is not R16G16_UNORM")

    bindings = {binding["input_slot"]: binding for binding in mesh_report["vertex_buffers"]}
    for element in (position_element, normal_element, uv_element):
        if element["input_slot"] not in bindings:
            raise ValueError(f"missing binding for input slot {element['input_slot']}")
        binding = bindings[element["input_slot"]]
        if binding["id"] not in vertex_buffers:
            raise ValueError(f"missing vertex buffer {binding['id']}")
        if element["byte_offset"] + FORMAT_SIZES[element["format"]] > binding["stride"]:
            raise ValueError(f"attribute {element['semantic']} exceeds stride {binding['stride']}")

    index_buffer = index_buffers[mesh_report["index_buffer_id"]]
    if index_buffer.get("format") != 57 or index_buffer["stride"] != 2:
        raise ValueError("donor index buffer is not R16_UINT")
    indices = [
        struct.unpack_from("<H", data, index_buffer["payload_offset"] + (mesh_report["start_index"] + index) * 2)[0]
        for index in range(mesh_report["index_count"])
    ]
    if len(indices) % 3:
        raise ValueError(f"mesh {mesh_report['name']} index count is not divisible by three")
    source_min = min(indices)
    source_max = max(indices)
    source_vertices = [mesh_report["base_vertex"] + index for index in range(source_min, source_max + 1)]
    if min(source_vertices) < 0:
        raise ValueError("negative source vertex index")

    bone_transform = world_matrices[mesh_report["bone_index"]]
    scale = mesh_report["scale"]
    translate = mesh_report["translate"]
    if scale is None or translate is None:
        raise ValueError("mesh lacks position scale/translation")
    uv_transform = mesh_report["uv_transforms"][0]
    vertices = []
    normals = []
    uvs = []

    def attribute_offset(element: dict, source_vertex: int) -> int:
        binding = bindings[element["input_slot"]]
        buffer = vertex_buffers[binding["id"]]
        if source_vertex >= buffer["count"]:
            raise ValueError(f"vertex {source_vertex} outside buffer {binding['id']}")
        return buffer["payload_offset"] + binding["offset"] + source_vertex * binding["stride"] + element["byte_offset"]

    for source_vertex in source_vertices:
        position_offset = attribute_offset(position_element, source_vertex)
        local_position = [
            snorm16(data, position_offset + axis * 2) * scale[axis] + translate[axis]
            for axis in range(3)
        ]
        packed_normal_x = snorm16(data, position_offset + 6)
        normal_offset = attribute_offset(normal_element, source_vertex)
        local_normal = [packed_normal_x, snorm16(data, normal_offset), snorm16(data, normal_offset + 2)]
        transformed_position = transform_vector_row(local_position, bone_transform, True)
        transformed_normal = transform_vector_row(local_normal, bone_transform, False)
        length = math.sqrt(sum(component * component for component in transformed_normal))
        if length > 1e-12:
            transformed_normal = [component / length for component in transformed_normal]
        vertices.append(axis_convert(transformed_position))
        normals.append(axis_convert(transformed_normal))

        uv_offset = attribute_offset(uv_element, source_vertex)
        raw_u, raw_v = struct.unpack_from("<HH", data, uv_offset)
        u = (raw_u / 65535.0) * uv_transform[1] + uv_transform[0]
        v = (raw_v / 65535.0) * uv_transform[3] + uv_transform[2]
        uvs.append((u, 1.0 - v))

    faces = [
        (indices[index] - source_min, indices[index + 2] - source_min, indices[index + 1] - source_min)
        for index in range(0, len(indices), 3)
    ]
    mesh_data = bpy.data.meshes.new(mesh_report["name"])
    mesh_data.from_pydata(vertices, [], faces)
    mesh_data.update()
    obj = bpy.data.objects.new(mesh_report["name"], mesh_data)
    collection.objects.link(obj)

    if material:
        mesh_data.materials.append(material)
    for polygon in mesh_data.polygons:
        polygon.use_smooth = True

    uv_layer = mesh_data.uv_layers.new(name="TEXCOORD0")
    for loop in mesh_data.loops:
        uv_layer.data[loop.index].uv = uvs[loop.vertex_index]

    source_index_attribute = mesh_data.attributes.new("fh6_source_vertex_index", "INT", "POINT")
    source_index_attribute.data.foreach_set("value", source_vertices)
    normal_attribute = mesh_data.attributes.new("fh6_source_normal", "FLOAT_VECTOR", "POINT")
    normal_attribute.data.foreach_set("vector", [component for normal in normals for component in normal])
    if hasattr(mesh_data, "normals_split_custom_set_from_vertices"):
        mesh_data.normals_split_custom_set_from_vertices(normals)

    skin = skin_buffers[mesh_report["skinning_data_buffer_id"]]
    weights_per_vertex = [decode_skin(data, skin, source_vertex, len(bone_names)) for source_vertex in source_vertices]
    used_bones = sorted({bone_index for weights in weights_per_vertex for bone_index, _ in weights})
    group_for_bone = {bone_index: obj.vertex_groups.new(name=bone_names[bone_index]) for bone_index in used_bones}
    bm = bmesh.new()
    bm.from_mesh(mesh_data)
    deform = bm.verts.layers.deform.verify()
    for vertex, weights in zip(bm.verts, weights_per_vertex):
        for bone_index, weight in weights:
            vertex[deform][group_for_bone[bone_index].index] = weight
    bm.to_mesh(mesh_data)
    bm.free()

    modifier = obj.modifiers.new(name="FH6 donor armature", type="ARMATURE")
    modifier.object = armature
    obj["fh6_mesh_blob_index"] = mesh_report["blob_index"]
    obj["fh6_material_id"] = mesh_report["material_id"]
    obj["fh6_lod_flags"] = mesh_report["lod_flags"]
    obj["fh6_render_pass"] = mesh_report["render_pass"]
    obj["fh6_start_index"] = mesh_report["start_index"]
    obj["fh6_index_count"] = mesh_report["index_count"]
    obj["fh6_base_vertex"] = mesh_report["base_vertex"]
    obj["fh6_source_vertex_min"] = source_vertices[0]
    obj["fh6_source_vertex_max"] = source_vertices[-1]

    influence_counts = Counter(len(weights) for weights in weights_per_vertex)
    weight_sums = [sum(weight for _, weight in weights) for weights in weights_per_vertex]
    return {
        "name": obj.name,
        "lod_flags": mesh_report["lod_flags"],
        "material_id": mesh_report["material_id"],
        "source_vertex_min": source_vertices[0],
        "source_vertex_max": source_vertices[-1],
        "vertices": len(vertices),
        "polygons": len(faces),
        "indices": len(indices),
        "used_bones": len(used_bones),
        "influence_histogram": dict(sorted(influence_counts.items())),
        "weight_sum_min": min(weight_sums),
        "weight_sum_max": max(weight_sums),
    }


def setup_preview(scene: bpy.types.Scene, visible_objects: list[bpy.types.Object], preview: Path) -> None:
    points = [obj.matrix_world @ vertex.co for obj in visible_objects for vertex in obj.data.vertices]
    minimum = Vector((min(point.x for point in points), min(point.y for point in points), min(point.z for point in points)))
    maximum = Vector((max(point.x for point in points), max(point.y for point in points), max(point.z for point in points)))
    center = (minimum + maximum) * 0.5
    extent = maximum - minimum
    distance = max(extent.x, extent.z, extent.y) * 2.3
    camera_data = bpy.data.cameras.new("Donor Preview Camera")
    camera = bpy.data.objects.new("Donor Preview Camera", camera_data)
    scene.collection.objects.link(camera)
    camera.location = center + Vector((0.0, -max(distance, 0.5), extent.z * 0.05))
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera_data.lens = 58
    scene.camera = camera

    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "WORLD"
    scene.render.resolution_x = 768
    scene.render.resolution_y = 768
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.render.filepath = str(preview)


def main() -> None:
    args = arguments()
    source = args.modelbin.resolve()
    blend = args.blend.resolve()
    metadata_path = args.metadata.resolve()
    preview = args.preview.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    for output in (blend, metadata_path, preview):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite baseline output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

    data = source.read_bytes()
    header, blobs = parse_bundle(data)
    parsed = parse_known_blobs(data, blobs)
    if parsed["errors"]:
        raise ValueError(f"modelbin parser errors: {parsed['errors']}")
    if len(parsed["skeleton"]) != 1:
        raise ValueError("expected exactly one skeleton")

    layouts = {layout["id"]: layout for layout in parsed["vertex_layouts"]}
    index_buffers, vertex_buffers, skin_buffers = buffer_maps(data, blobs, parsed)
    bones = parsed["skeleton"][0]["bones"]
    world_matrices = skeleton_world_matrices(bones)

    clear_scene()
    root = bpy.data.collections.new("FH6_DONOR_SOURCE")
    lod_streaming = bpy.data.collections.new("LOD_STREAMING")
    lod0 = bpy.data.collections.new("LOD0")
    bpy.context.scene.collection.children.link(root)
    root.children.link(lod_streaming)
    root.children.link(lod0)
    lod_streaming.hide_viewport = True
    lod_streaming.hide_render = True

    armature, bone_names = create_armature(bones, world_matrices, root)
    materials = create_materials(blobs)
    mesh_records = []
    visible_objects = []
    for mesh_report in parsed["meshes"]:
        target_collection = lod0 if mesh_report["lod_flags"] & 0x2 else lod_streaming
        record = create_mesh_object(
            data,
            mesh_report,
            layouts[mesh_report["vertex_layout_id"]],
            index_buffers,
            vertex_buffers,
            skin_buffers,
            world_matrices,
            bone_names,
            armature,
            materials.get(mesh_report["material_id"]),
            target_collection,
        )
        mesh_records.append(record)
        if target_collection == lod0:
            visible_objects.append(bpy.data.objects[record["name"]])

    scene = bpy.context.scene
    scene["baseline_kind"] = "immutable_fh6_donor_source"
    scene["source_modelbin"] = str(source)
    scene["source_modelbin_sha256"] = sha256(source)
    scene["geometry_edited"] = False
    scene["weights_edited"] = False
    scene["materials_are_placeholders"] = True
    scene["armature_is_display_reconstruction"] = True
    setup_preview(scene, visible_objects, preview)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend), compress=False, check_existing=False)
    bpy.ops.render.render(write_still=True)

    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Read-only FH6 donor inspection baseline; no geometry or weight edits.",
        "source": {"modelbin": str(source), "sha256": sha256(source), "bytes": len(data)},
        "output": {
            "blend": str(blend),
            "blend_sha256": sha256(blend),
            "preview": str(preview),
            "preview_sha256": sha256(preview),
        },
        "software": {"blender": bpy.app.version_string, "python": sys.version.split()[0]},
        "bundle": header,
        "bone_count": len(bones),
        "materials": {str(identifier): material.name for identifier, material in materials.items()},
        "meshes": mesh_records,
        "notes": {
            "geometry_edited": False,
            "weights_edited": False,
            "materials_are_placeholders": True,
            "armature_is_display_reconstruction": True,
            "lod_streaming_hidden_by_default": True,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("FH6_DONOR_BASELINE=" + json.dumps({
        "blend": str(blend),
        "preview": str(preview),
        "metadata": str(metadata_path),
        "bones": len(bones),
        "mesh_objects": len(mesh_records),
        "visible_mesh_objects": len(visible_objects),
        "source_vertices": max(buffer["count"] for buffer in vertex_buffers.values()),
        "source_indices": max(buffer["count"] for buffer in index_buffers.values()),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
