#!/usr/bin/env python3
"""Pure-Python REST segment frames and endpoint-preserving affine solves.

This module deliberately has no Blender or NumPy dependency.  Frames are
derived from explicit REST points and anatomical hints; callers must not feed
``matrix_local`` into this API.
"""

from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


Vec3 = tuple[float, float, float]
Mat3 = tuple[Vec3, Vec3, Vec3]

DEFAULT_MIN_SEGMENT_LENGTH_M = 5.0e-4
DEFAULT_PROJECTION_EPSILON = math.sin(math.radians(3.0))
DEFAULT_DETERMINANT_EPSILON = 1.0e-9


class SegmentGeometryError(ValueError):
    """Base class for invalid segment geometry."""


class DegenerateSegmentError(SegmentGeometryError):
    """Raised when a segment is too short to define an axis or scale."""


class InvalidFrameError(SegmentGeometryError):
    """Raised when a right-handed orthonormal frame cannot be constructed."""


class TransformQualityError(SegmentGeometryError):
    """Raised when a transform reflects, collapses, or exceeds its condition gate."""


class ArcLengthError(SegmentGeometryError):
    """Raised when a polyline cannot support arc-length sampling."""


def as_vec3(value: Sequence[float]) -> Vec3:
    if len(value) != 3:
        raise ValueError(f"Expected a 3-vector, got {len(value)} values")
    vector = (float(value[0]), float(value[1]), float(value[2]))
    if not all(math.isfinite(component) for component in vector):
        raise ValueError(f"Vector contains a non-finite value: {vector}")
    return vector


def add(left: Sequence[float], right: Sequence[float]) -> Vec3:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def sub(left: Sequence[float], right: Sequence[float]) -> Vec3:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def scale(vector: Sequence[float], scalar: float) -> Vec3:
    return (vector[0] * scalar, vector[1] * scalar, vector[2] * scalar)


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def cross(left: Sequence[float], right: Sequence[float]) -> Vec3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def norm(vector: Sequence[float]) -> float:
    return math.sqrt(max(0.0, dot(vector, vector)))


def normalize(vector: Sequence[float], epsilon: float = 1.0e-12) -> Vec3:
    magnitude = norm(vector)
    if magnitude <= epsilon:
        raise DegenerateSegmentError(f"Cannot normalize vector of length {magnitude:.12g}")
    return scale(vector, 1.0 / magnitude)


def distance(left: Sequence[float], right: Sequence[float]) -> float:
    return norm(sub(left, right))


def project_to_plane(vector: Sequence[float], plane_normal: Sequence[float]) -> Vec3:
    normal = normalize(plane_normal)
    return sub(vector, scale(normal, dot(vector, normal)))


def matrix_from_columns(columns: Sequence[Sequence[float]]) -> Mat3:
    if len(columns) != 3:
        raise ValueError("A 3x3 matrix requires three columns")
    converted = tuple(as_vec3(column) for column in columns)
    return tuple(tuple(converted[column][row] for column in range(3)) for row in range(3))  # type: ignore[return-value]


def transpose(matrix: Sequence[Sequence[float]]) -> Mat3:
    return tuple(tuple(float(matrix[column][row]) for column in range(3)) for row in range(3))  # type: ignore[return-value]


def matmul(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> Mat3:
    return tuple(
        tuple(sum(left[row][index] * right[index][column] for index in range(3)) for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> Vec3:
    return tuple(sum(matrix[row][column] * vector[column] for column in range(3)) for row in range(3))  # type: ignore[return-value]


def determinant3(matrix: Sequence[Sequence[float]]) -> float:
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def _symmetric_eigenvalues_3x3(matrix: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    """Return eigenvalues of a symmetric 3x3 matrix via Jacobi rotations."""

    work = [[float(matrix[row][column]) for column in range(3)] for row in range(3)]
    for _iteration in range(48):
        row, column = max(((0, 1), (0, 2), (1, 2)), key=lambda pair: abs(work[pair[0]][pair[1]]))
        off_diagonal = work[row][column]
        if abs(off_diagonal) <= 1.0e-15:
            break
        angle = 0.5 * math.atan2(2.0 * off_diagonal, work[column][column] - work[row][row])
        cosine = math.cos(angle)
        sine = math.sin(angle)

        row_row = work[row][row]
        column_column = work[column][column]
        work[row][row] = cosine * cosine * row_row - 2.0 * sine * cosine * off_diagonal + sine * sine * column_column
        work[column][column] = sine * sine * row_row + 2.0 * sine * cosine * off_diagonal + cosine * cosine * column_column
        work[row][column] = 0.0
        work[column][row] = 0.0
        for other in range(3):
            if other in (row, column):
                continue
            old_row = work[other][row]
            old_column = work[other][column]
            work[other][row] = cosine * old_row - sine * old_column
            work[row][other] = work[other][row]
            work[other][column] = sine * old_row + cosine * old_column
            work[column][other] = work[other][column]
    return tuple(sorted((work[0][0], work[1][1], work[2][2])))  # type: ignore[return-value]


def condition_number_3x3(matrix: Sequence[Sequence[float]], epsilon: float = 1.0e-15) -> float:
    gram = matmul(transpose(matrix), matrix)
    eigenvalues = _symmetric_eigenvalues_3x3(gram)
    smallest = max(0.0, eigenvalues[0])
    largest = max(0.0, eigenvalues[-1])
    if smallest <= epsilon:
        return math.inf
    return math.sqrt(largest / smallest)


@dataclass(frozen=True)
class TransformQuality:
    determinant: float
    condition_number: float


def validate_linear_transform(
    matrix: Sequence[Sequence[float]],
    *,
    max_condition: float,
    determinant_epsilon: float = DEFAULT_DETERMINANT_EPSILON,
) -> TransformQuality:
    determinant = determinant3(matrix)
    if not math.isfinite(determinant) or determinant <= determinant_epsilon:
        raise TransformQualityError(
            f"Transform determinant must be positive and > {determinant_epsilon:g}; got {determinant:.12g}"
        )
    condition = condition_number_3x3(matrix)
    if not math.isfinite(condition) or condition > max_condition:
        raise TransformQualityError(f"Transform condition {condition:.9g} exceeds gate {max_condition:.9g}")
    return TransformQuality(determinant=determinant, condition_number=condition)


def _least_parallel_axis(axis: Sequence[float]) -> Vec3:
    unit_axis = normalize(axis)
    canonical = min(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), key=lambda item: abs(dot(unit_axis, item)))
    return normalize(project_to_plane(canonical, unit_axis))


@dataclass(frozen=True)
class SegmentFrame:
    axis: Vec3
    roll: Vec3
    binormal: Vec3
    basis: str
    projection_strength: float

    @property
    def matrix(self) -> Mat3:
        return matrix_from_columns((self.axis, self.roll, self.binormal))

    @property
    def determinant(self) -> float:
        return determinant3(self.matrix)


def derive_segment_frame(
    head: Sequence[float],
    end: Sequence[float],
    *,
    anatomical_hint: Sequence[float],
    previous_roll: Sequence[float] | None = None,
    bend_direction: Sequence[float] | None = None,
    min_length_m: float = DEFAULT_MIN_SEGMENT_LENGTH_M,
    projection_epsilon: float = DEFAULT_PROJECTION_EPSILON,
) -> SegmentFrame:
    head_vector = as_vec3(head)
    end_vector = as_vec3(end)
    segment = sub(end_vector, head_vector)
    segment_length = norm(segment)
    if segment_length < min_length_m:
        raise DegenerateSegmentError(
            f"Segment length {segment_length:.9g} m is below minimum {min_length_m:.9g} m"
        )
    axis = normalize(segment)

    candidates: list[tuple[str, Vec3]] = [("anatomical world hint", as_vec3(anatomical_hint))]
    if previous_roll is not None:
        candidates.append(("parallel-transported previous roll", as_vec3(previous_roll)))
    if bend_direction is not None:
        candidates.append(("bend direction", as_vec3(bend_direction)))

    roll: Vec3 | None = None
    basis = "least-parallel canonical axis"
    projection_strength = 0.0
    for candidate_basis, candidate in candidates:
        projected = project_to_plane(candidate, axis)
        strength = norm(projected) / max(norm(candidate), 1.0e-15)
        if strength >= projection_epsilon:
            roll = normalize(projected)
            basis = candidate_basis
            projection_strength = strength
            break
    if roll is None:
        roll = _least_parallel_axis(axis)

    if previous_roll is not None:
        transported = project_to_plane(previous_roll, axis)
        if norm(transported) >= 1.0e-12 and dot(roll, transported) < 0.0:
            roll = scale(roll, -1.0)

    binormal = normalize(cross(axis, roll))
    roll = normalize(cross(binormal, axis))
    frame = SegmentFrame(axis=axis, roll=roll, binormal=binormal, basis=basis, projection_strength=projection_strength)
    if frame.determinant <= 1.0 - 1.0e-8:
        raise InvalidFrameError(f"Frame is not right-handed orthonormal; determinant={frame.determinant:.12g}")
    return frame


def derive_chain_frames(
    points: Sequence[Sequence[float]],
    *,
    anatomical_hint: Sequence[float],
    min_length_m: float = DEFAULT_MIN_SEGMENT_LENGTH_M,
) -> tuple[SegmentFrame, ...]:
    if len(points) < 2:
        raise DegenerateSegmentError("A chain requires at least two points")
    frames: list[SegmentFrame] = []
    previous_roll: Vec3 | None = None
    for index in range(len(points) - 1):
        bend_direction = None
        if index + 2 < len(points):
            bend_direction = sub(points[index + 2], points[index + 1])
        frame = derive_segment_frame(
            points[index],
            points[index + 1],
            anatomical_hint=anatomical_hint,
            previous_roll=previous_roll,
            bend_direction=bend_direction,
            min_length_m=min_length_m,
        )
        frames.append(frame)
        previous_roll = frame.roll
    return tuple(frames)


@dataclass(frozen=True)
class SegmentAffine:
    linear: Mat3
    translation: Vec3
    axial_scale: float
    radial_scale: float
    determinant: float
    condition_number: float
    source_length_m: float
    target_length_m: float

    def apply(self, point: Sequence[float]) -> Vec3:
        return add(matvec(self.linear, point), self.translation)


def solve_segment_affine(
    source_head: Sequence[float],
    source_end: Sequence[float],
    source_frame: SegmentFrame,
    target_head: Sequence[float],
    target_end: Sequence[float],
    target_frame: SegmentFrame,
    *,
    radial_scale: float = 1.0,
    max_condition: float = 2.5,
    min_length_m: float = DEFAULT_MIN_SEGMENT_LENGTH_M,
) -> SegmentAffine:
    source_head_vec = as_vec3(source_head)
    source_end_vec = as_vec3(source_end)
    target_head_vec = as_vec3(target_head)
    target_end_vec = as_vec3(target_end)
    source_length = distance(source_head_vec, source_end_vec)
    target_length = distance(target_head_vec, target_end_vec)
    if source_length < min_length_m or target_length < min_length_m:
        raise DegenerateSegmentError(
            f"Affine endpoints are degenerate: source={source_length:.9g} m, target={target_length:.9g} m"
        )
    if not math.isfinite(radial_scale) or radial_scale <= 0.0:
        raise TransformQualityError(f"Radial scale must be finite and positive; got {radial_scale!r}")

    source_direction = normalize(sub(source_end_vec, source_head_vec))
    target_direction = normalize(sub(target_end_vec, target_head_vec))
    if dot(source_direction, source_frame.axis) < 1.0 - 1.0e-7:
        raise InvalidFrameError("Source frame axis does not match source segment direction")
    if dot(target_direction, target_frame.axis) < 1.0 - 1.0e-7:
        raise InvalidFrameError("Target frame axis does not match target segment direction")

    axial_scale = target_length / source_length
    scale_matrix: Mat3 = (
        (axial_scale, 0.0, 0.0),
        (0.0, radial_scale, 0.0),
        (0.0, 0.0, radial_scale),
    )
    linear = matmul(matmul(target_frame.matrix, scale_matrix), transpose(source_frame.matrix))
    quality = validate_linear_transform(linear, max_condition=max_condition)
    translation = sub(target_head_vec, matvec(linear, source_head_vec))
    affine = SegmentAffine(
        linear=linear,
        translation=translation,
        axial_scale=axial_scale,
        radial_scale=radial_scale,
        determinant=quality.determinant,
        condition_number=quality.condition_number,
        source_length_m=source_length,
        target_length_m=target_length,
    )
    endpoint_tolerance = max(1.0e-10, target_length * 1.0e-9)
    if distance(affine.apply(source_head_vec), target_head_vec) > endpoint_tolerance:
        raise TransformQualityError("Solved affine does not preserve the target head")
    if distance(affine.apply(source_end_vec), target_end_vec) > endpoint_tolerance:
        raise TransformQualityError("Solved affine does not preserve the target end")
    return affine


@dataclass(frozen=True)
class RigidZAlignment:
    rotation_z_degrees: float
    translation: Vec3
    scale: float = 1.0

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "RigidZAlignment":
        alignment = mapping.get("alignment", mapping)
        return cls(
            rotation_z_degrees=float(alignment["rotation_z_degrees"]),
            translation=as_vec3(alignment["translation"]),
            scale=float(alignment.get("scale", 1.0)),
        )

    def apply_point(self, point: Sequence[float]) -> Vec3:
        radians = math.radians(self.rotation_z_degrees)
        cosine = math.cos(radians)
        sine = math.sin(radians)
        source = as_vec3(point)
        rotated = (
            self.scale * (cosine * source[0] - sine * source[1]),
            self.scale * (sine * source[0] + cosine * source[1]),
            self.scale * source[2],
        )
        return add(rotated, self.translation)

    def apply_vector(self, vector: Sequence[float]) -> Vec3:
        return sub(self.apply_point(vector), self.apply_point((0.0, 0.0, 0.0)))


@dataclass(frozen=True)
class ArcSample:
    point: Vec3
    fraction: float
    distance: float
    total_length: float
    segment_index: int
    segment_t: float


def cumulative_arc_lengths(points: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if len(points) < 2:
        raise ArcLengthError("A polyline requires at least two points")
    converted = tuple(as_vec3(point) for point in points)
    cumulative = [0.0]
    for first, second in zip(converted, converted[1:]):
        cumulative.append(cumulative[-1] + distance(first, second))
    if cumulative[-1] < DEFAULT_MIN_SEGMENT_LENGTH_M:
        raise ArcLengthError(f"Polyline total length {cumulative[-1]:.9g} m is too short")
    return tuple(cumulative)


def normalized_arc_fractions(points: Sequence[Sequence[float]]) -> tuple[float, ...]:
    cumulative = cumulative_arc_lengths(points)
    total = cumulative[-1]
    return tuple(value / total for value in cumulative)


def sample_polyline(points: Sequence[Sequence[float]], fraction: float) -> ArcSample:
    if not math.isfinite(fraction) or fraction < 0.0 or fraction > 1.0:
        raise ArcLengthError(f"Arc fraction must be in [0, 1]; got {fraction!r}")
    converted = tuple(as_vec3(point) for point in points)
    cumulative = cumulative_arc_lengths(converted)
    total = cumulative[-1]
    requested = fraction * total
    if fraction == 1.0:
        return ArcSample(converted[-1], fraction, total, total, len(converted) - 2, 1.0)
    index = max(0, min(len(converted) - 2, bisect.bisect_right(cumulative, requested) - 1))
    while index < len(converted) - 2 and cumulative[index + 1] <= cumulative[index] + 1.0e-15:
        index += 1
    segment_length = cumulative[index + 1] - cumulative[index]
    if segment_length <= 1.0e-15:
        raise ArcLengthError("Unable to sample a zero-length terminal polyline segment")
    segment_t = (requested - cumulative[index]) / segment_length
    point = add(converted[index], scale(sub(converted[index + 1], converted[index]), segment_t))
    return ArcSample(point, fraction, requested, total, index, segment_t)


@dataclass(frozen=True)
class ArcCorrespondence:
    target_index: int
    target_fraction: float
    target_point: Vec3
    source_sample: ArcSample
    inserted: bool


def map_target_knots_to_source(
    source_points: Sequence[Sequence[float]],
    target_points: Sequence[Sequence[float]],
    *,
    inserted_target_indices: Iterable[int] = (),
) -> tuple[ArcCorrespondence, ...]:
    target_converted = tuple(as_vec3(point) for point in target_points)
    fractions = normalized_arc_fractions(target_converted)
    inserted = set(int(index) for index in inserted_target_indices)
    invalid = sorted(index for index in inserted if index <= 0 or index >= len(target_converted) - 1)
    if invalid:
        raise ArcLengthError(f"Inserted target indices must be interior nodes; got {invalid}")
    return tuple(
        ArcCorrespondence(
            target_index=index,
            target_fraction=fraction,
            target_point=target_converted[index],
            source_sample=sample_polyline(source_points, fraction),
            inserted=index in inserted,
        )
        for index, fraction in enumerate(fractions)
    )


@dataclass(frozen=True)
class FrameAnchor:
    anchor_id: str
    source_start: str
    source_end: str
    target_start: str
    target_end: str
    family: str
    roll_hint: Vec3
    affects_weight_transfer: bool


def load_frame_anchor_metadata(path: Path | str) -> dict[str, Any]:
    metadata_path = Path(path)
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("schema_version") != 1:
        raise ValueError(f"Unsupported frame-anchor schema: {metadata.get('schema_version')!r}")
    for container_name, container in metadata.get("containers", {}).items():
        for anchor in container.get("anchors", []):
            if anchor.get("kind") != "frame_only" or anchor.get("affects_weight_transfer") is not False:
                raise ValueError(f"{container_name}/{anchor.get('id')} is not a frame-only anchor")
    return metadata


def iter_frame_anchors(metadata: Mapping[str, Any], container_name: str) -> Iterator[FrameAnchor]:
    containers = metadata.get("containers", {})
    if container_name not in containers:
        raise KeyError(f"Unknown frame-anchor container {container_name!r}")
    for anchor in containers[container_name].get("anchors", []):
        yield FrameAnchor(
            anchor_id=str(anchor["id"]),
            source_start=str(anchor["source_start"]),
            source_end=str(anchor["source_end"]),
            target_start=str(anchor["target_start"]),
            target_end=str(anchor["target_end"]),
            family=str(anchor["family"]),
            roll_hint=as_vec3(anchor["roll_hint"]),
            affects_weight_transfer=bool(anchor["affects_weight_transfer"]),
        )

