#!/usr/bin/env python3
"""Compare neck top-ring normals and weights to the nearest face surface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import validate_si_fbx_display_seams as base


def arguments():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def reset(armature):
    armature.data.pose_position = "POSE"
    for bone in armature.pose.bones:
        bone.rotation_mode = "XYZ"
        bone.rotation_euler = (0, 0, 0)
        bone.location = (0, 0, 0)
        bone.scale = (1, 1, 1)


def normal_map(obj):
    sums = {}
    counts = {}
    for loop in obj.data.loops:
        n = Vector(obj.data.corner_normals[loop.index].vector)
        sums[loop.vertex_index] = sums.get(loop.vertex_index, Vector()) + n
        counts[loop.vertex_index] = counts.get(loop.vertex_index, 0) + 1
    matrix = obj.matrix_world.to_3x3().inverted().transposed()
    return {i: (matrix @ (v / counts[i])).normalized() for i, v in sums.items()}


def evaluated_normal_map(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        sums = {}
        counts = {}
        for loop in mesh.loops:
            normal = Vector(mesh.corner_normals[loop.index].vector)
            sums[loop.vertex_index] = sums.get(loop.vertex_index, Vector()) + normal
            counts[loop.vertex_index] = counts.get(loop.vertex_index, 0) + 1
        matrix = obj.matrix_world.to_3x3().inverted().transposed()
        return {i: (matrix @ (value / counts[i])).normalized() for i, value in sums.items()}
    finally:
        evaluated.to_mesh_clear()


def interpolated_triangle_normal(point, triangle, coordinates, normals):
    first, second, third = (coordinates[index] for index in triangle)
    edge_a = second - first
    edge_b = third - first
    offset = point - first
    aa = edge_a.dot(edge_a)
    ab = edge_a.dot(edge_b)
    bb = edge_b.dot(edge_b)
    oa = offset.dot(edge_a)
    ob = offset.dot(edge_b)
    denominator = aa * bb - ab * ab
    if abs(denominator) <= 1.0e-16:
        return normals[triangle[0]].copy()
    second_weight = (bb * oa - ab * ob) / denominator
    third_weight = (aa * ob - ab * oa) / denominator
    first_weight = 1.0 - second_weight - third_weight
    weights = [max(0.0, min(1.0, value)) for value in (first_weight, second_weight, third_weight)]
    weight_sum = sum(weights)
    if weight_sum <= 1.0e-12:
        return normals[triangle[0]].copy()
    result = sum(
        (normals[index] * (weight / weight_sum) for index, weight in zip(triangle, weights)),
        Vector(),
    )
    return result.normalized() if result.length > 1.0e-8 else normals[triangle[0]].copy()


def main():
    args = arguments()
    head = bpy.data.objects[base.HEAD_MESH_NAME]
    body = bpy.data.objects[base.BODY_MESH_NAME]
    ha = bpy.data.objects[base.HEAD_ARMATURE_NAME]
    ba = bpy.data.objects[base.BODY_ARMATURE_NAME]
    row = body.data.attributes["fh6_neck_bridge_row"]
    col = body.data.attributes["fh6_neck_bridge_column"]
    top = {int(col.data[i].value): i for i in range(len(row.data)) if row.data[i].value == 3}
    hp = base.material_polygons(head, "面")
    reset(ha); reset(ba); bpy.context.view_layer.update()
    hr = base.evaluated_coordinates(head); br = base.evaluated_coordinates(body)
    hn = evaluated_normal_map(head); bn = evaluated_normal_map(body)
    tree = BVHTree.FromPolygons(hr, hp, all_triangles=False)
    group_names = {g.index:g.name for g in body.vertex_groups}
    rows=[]
    for c,i in sorted(top.items()):
        near=tree.find_nearest(br[i])
        weights={group_names[x.group]:float(x.weight) for x in body.data.vertices[i].groups if x.weight>1e-8}
        shading_normal = interpolated_triangle_normal(near[0], hp[near[2]], hr, hn) if near else None
        rows.append({'column':c,'vertex':i,'point':list(br[i]),'dot':float(bn[i].dot(shading_normal)) if shading_normal else None,'nearest_geometric_normal':list(near[1]) if near else None,'nearest_shading_normal':list(shading_normal) if shading_normal else None,'body_normal':list(bn[i]),'weights':weights})
    base.apply_diagnostic_pose((ha,ba)); bpy.context.view_layer.update()
    hp2=base.evaluated_coordinates(head); bp2=base.evaluated_coordinates(body)
    hn2=evaluated_normal_map(head); bn2=evaluated_normal_map(body); tree2=BVHTree.FromPolygons(hp2,hp,all_triangles=False)
    for item in rows:
        i=item['vertex']; near=tree2.find_nearest(bp2[i]); shading_normal=interpolated_triangle_normal(near[0], hp[near[2]], hp2, hn2) if near else None; item['pose_dot']=float(bn2[i].dot(shading_normal)) if shading_normal else None
    out={'blend':str(Path(bpy.data.filepath).resolve()),'rows':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print('SI_NECK_NORMAL_AUDIT='+str(args.output.resolve()))

if __name__=='__main__': main()
