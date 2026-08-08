#!/usr/bin/env python3
"""Create a reduced-vertex page-preview milestone from the retargeted blend."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy


def arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--object", required=True)
    parser.add_argument("--ratio", type=float, default=0.82)
    return parser.parse_args(argv)


def main() -> None:
    args = arguments()
    source = args.input.resolve(strict=True)
    output = args.output.resolve()
    if Path(bpy.data.filepath).resolve() != source:
        raise RuntimeError(f"Opened blend {bpy.data.filepath!r} does not match --input {source}")
    if not 0.0 < args.ratio < 1.0:
        raise ValueError("--ratio must be between zero and one")
    obj = bpy.data.objects.get(args.object)
    if obj is None or obj.type != "MESH":
        raise RuntimeError(f"Mesh object {args.object!r} not found")
    duplicate = obj.copy()
    duplicate.data = obj.data.copy()
    duplicate.name = f"{obj.name}_Budget32K"
    duplicate.data.name = f"{obj.data.name}_Budget32K_Mesh"
    obj.users_collection[0].objects.link(duplicate)
    modifier = duplicate.modifiers.new(name="Budget decimation", type="DECIMATE")
    modifier.decimate_type = "COLLAPSE"
    modifier.ratio = args.ratio
    modifier.use_collapse_triangulate = True
    bpy.context.view_layer.objects.active = duplicate
    duplicate.select_set(True)
    for other in bpy.context.selected_objects:
        if other != duplicate:
            other.select_set(False)
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    for vertex in duplicate.data.vertices:
        assignments = sorted(
            ((group.group, float(group.weight)) for group in vertex.groups if group.weight > 0.001),
            key=lambda item: (-item[1], item[0]),
        )[:4]
        total = sum(weight for _, weight in assignments)
        if total <= 0.0:
            raise RuntimeError(f"Decimation produced an unweighted vertex {vertex.index}")
        for group in vertex.groups:
            group.weight = 0.0
        for group_index, weight in assignments:
            duplicate.vertex_groups[group_index].add([vertex.index], weight / total, "REPLACE")
    bpy.data.objects.remove(obj, do_unlink=True)
    if len(duplicate.data.vertices) >= 32_000:
        raise RuntimeError(f"Budget decimation left {len(duplicate.data.vertices)} vertices")
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    print(f"FH6_BUDGET_MILESTONE={{\"output\":{str(output)!r},\"object\":{duplicate.name!r},\"vertices\":{len(duplicate.data.vertices)},\"polygons\":{len(duplicate.data.polygons)}}}")


if __name__ == "__main__":
    main()
