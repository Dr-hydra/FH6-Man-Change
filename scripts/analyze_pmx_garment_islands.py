#!/usr/bin/env python3
"""Find connected topology islands in Si's garment material slots."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import bpy


GARMENT_MATERIALS = {"Cloth1", "Cloth2", "Cloth1Alpha"}


def arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--weld-epsilon", type=float, default=1e-5)
    return parser.parse_args(argv)


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def island_bounds(mesh: bpy.types.Mesh, vertices: set[int]) -> dict:
    minimum = [min(mesh.vertices[index].co[axis] for index in vertices) for axis in range(3)]
    maximum = [max(mesh.vertices[index].co[axis] for index in vertices) for axis in range(3)]
    return {
        "min": minimum,
        "max": maximum,
        "center": [(minimum[axis] + maximum[axis]) * 0.5 for axis in range(3)],
        "extent": [maximum[axis] - minimum[axis] for axis in range(3)],
    }


def island_bones(obj: bpy.types.Object, vertices: set[int], deform_groups: set[int]) -> dict:
    weight = Counter()
    vertex_count = Counter()
    for vertex_index in vertices:
        for element in obj.data.vertices[vertex_index].groups:
            if element.group in deform_groups and element.weight > 1e-8:
                weight[element.group] += element.weight
                vertex_count[element.group] += 1
    ranked = [
        {
            "group": obj.vertex_groups[group_index].name,
            "vertices": vertex_count[group_index],
            "total_weight": weight[group_index],
        }
        for group_index in sorted(weight, key=lambda index: (-weight[index], -vertex_count[index], obj.vertex_groups[index].name))
    ]
    return {"bone_group_count": len(ranked), "top_bones": ranked[:15]}


def main() -> None:
    args = arguments()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    obj = next(item for item in bpy.context.scene.objects if item.type == "MESH")
    armature = next(item for item in bpy.context.scene.objects if item.type == "ARMATURE")
    mesh = obj.data
    deform_names = {bone.name for bone in armature.data.bones if bone.use_deform}
    deform_groups = {group.index for group in obj.vertex_groups if group.name in deform_names}
    material_index_by_name = {
        slot.material.name: index
        for index, slot in enumerate(obj.material_slots)
        if slot.material and slot.material.name in GARMENT_MATERIALS
    }

    records = []
    for material_name in sorted(material_index_by_name):
        material_index = material_index_by_name[material_name]
        polygons = [polygon for polygon in mesh.polygons if polygon.material_index == material_index]
        union_find = UnionFind(len(mesh.vertices))
        material_vertices = {vertex for polygon in polygons for vertex in polygon.vertices}
        spatial_buckets: defaultdict[tuple[int, int, int], list[int]] = defaultdict(list)
        for vertex_index in material_vertices:
            coordinate = mesh.vertices[vertex_index].co
            key = tuple(round(coordinate[axis] / args.weld_epsilon) for axis in range(3))
            spatial_buckets[key].append(vertex_index)
        for bucket in spatial_buckets.values():
            for index in range(1, len(bucket)):
                union_find.union(bucket[0], bucket[index])
        for polygon in polygons:
            vertices = list(polygon.vertices)
            for index in range(1, len(vertices)):
                union_find.union(vertices[0], vertices[index])

        vertices_by_root: defaultdict[int, set[int]] = defaultdict(set)
        polygons_by_root: Counter[int] = Counter()
        for polygon in polygons:
            root = union_find.find(polygon.vertices[0])
            vertices_by_root[root].update(polygon.vertices)
            polygons_by_root[root] += 1

        islands = []
        ordered = sorted(vertices_by_root, key=lambda root: (-len(vertices_by_root[root]), -polygons_by_root[root], root))
        for rank, root in enumerate(ordered):
            vertices = vertices_by_root[root]
            islands.append(
                {
                    "rank": rank,
                    "seed_vertex": min(vertices),
                    "vertices": len(vertices),
                    "polygons": polygons_by_root[root],
                    "bounds": island_bounds(mesh, vertices),
                    "bone_usage": island_bones(obj, vertices, deform_groups),
                }
            )
        records.append(
            {
                "material": material_name,
                "material_index": material_index,
                "island_count": len(islands),
                "vertices": sum(item["vertices"] for item in islands),
                "polygons": sum(item["polygons"] for item in islands),
                "islands": islands,
            }
        )

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "blend_file": bpy.data.filepath,
        "mesh": obj.name,
        "connectivity_weld_epsilon": args.weld_epsilon,
        "materials": records,
    }
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("FH6_GARMENT_ISLANDS=" + json.dumps({
        "output": str(output),
        "materials": {
            record["material"]: {
                "islands": record["island_count"],
                "vertices": record["vertices"],
                "top": [
                    {
                        "rank": island["rank"],
                        "seed": island["seed_vertex"],
                        "vertices": island["vertices"],
                        "polygons": island["polygons"],
                        "center": island["bounds"]["center"],
                        "extent": island["bounds"]["extent"],
                        "bones": island["bone_usage"]["bone_group_count"],
                    }
                    for island in record["islands"][:12]
                ],
            }
            for record in records
        },
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
