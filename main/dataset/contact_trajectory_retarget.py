"""Contact-aware geometric post-retargeting for dexterous-hand trajectories.

This module deliberately runs before Stage 1.  It keeps wrist orientation and
all finger DOFs unchanged, and solves only for a right-hand translation in the
tool-local frame.  Corrected PKLs mirror ``data/retargeting`` below
``data/retargeting_contact`` and remain compatible with ``load_retargeted_data``.

The implementation has no Isaac Gym dependency.  Hand collision surfaces and
forward kinematics are reconstructed directly from the dex-hand URDF, which
makes the geometric correction and its validation runnable as an offline data
preprocessing step.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import pickle
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import trimesh
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


SCHEMA_VERSION = 1
DEFAULT_OAKINK_ROOT = "/home/ps/data/ylj/ManipTrans/data/OakInk-v2"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _as_mesh(scene_or_mesh) -> trimesh.Trimesh:
    if isinstance(scene_or_mesh, trimesh.Scene):
        geometries = tuple(scene_or_mesh.geometry.values())
        if not geometries:
            raise ValueError("Mesh scene contains no geometry")
        return trimesh.util.concatenate(geometries)
    if not isinstance(scene_or_mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a trimesh mesh, got {type(scene_or_mesh)!r}")
    return scene_or_mesh


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _urdf_bundle_sha256(urdf_path: Path) -> str:
    """Hash a URDF together with every local mesh it references."""

    urdf_path = urdf_path.resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"Object collision URDF does not exist: {urdf_path}")
    asset_paths = [urdf_path]
    root = ET.fromstring(urdf_path.read_text(encoding="utf-8").lstrip())
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if not filename or filename.startswith("package://"):
            raise ValueError(f"Unsupported object URDF mesh URI {filename!r} in {urdf_path}")
        mesh_path = Path(filename[7:] if filename.startswith("file://") else filename)
        if not mesh_path.is_absolute():
            mesh_path = urdf_path.parent / mesh_path
        mesh_path = mesh_path.resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Object collision mesh does not exist: {mesh_path}")
        asset_paths.append(mesh_path)
    digest = hashlib.sha256()
    for asset_path in sorted(set(asset_paths), key=str):
        digest.update(asset_path.name.encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(asset_path)))
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _atomic_pickle_dump(data: Mapping, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=output_path.parent, prefix=output_path.name + ".", delete=False
    ) as stream:
        temp_path = Path(stream.name)
        pickle.dump(data, stream)
    os.replace(temp_path, output_path)


def _atomic_json_dump(data: Mapping, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output_path.parent, prefix=output_path.name + ".", delete=False
    ) as stream:
        temp_path = Path(stream.name)
        json.dump(_jsonable(data), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    os.replace(temp_path, output_path)


def mirror_retarget_output_path(input_path: Path, output_root: Path) -> Path:
    """Mirror the path below ``data/retargeting`` into the requested root."""

    input_path = input_path.resolve()
    source_root = (PROJECT_ROOT / "data" / "retargeting").resolve()
    try:
        relative = input_path.relative_to(source_root)
    except ValueError as exc:
        raise ValueError(
            f"Input PKL must be below {source_root} when --output-pkl is omitted; got {input_path}"
        ) from exc
    return output_root.resolve() / relative


def sequence_artifact_path(pkl_path: Path, suffix: str) -> Path:
    """Place non-PKL artifacts in a sibling directory named after the sequence PKL."""

    pkl_path = Path(pkl_path)
    if pkl_path.suffix.lower() != ".pkl":
        raise ValueError(f"Expected a .pkl trajectory path, got {pkl_path}")
    sequence_dir = pkl_path.parent / pkl_path.stem
    return sequence_dir / f"{pkl_path.stem}{suffix}"


def _origin_matrix(origin: Optional[ET.Element]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    if origin is None:
        return transform
    xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ", dtype=np.float64)
    rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ", dtype=np.float64)
    if xyz.size != 3 or rpy.size != 3:
        raise ValueError(f"Invalid URDF origin: {origin.attrib}")
    transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = xyz
    return transform


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _deterministic_mesh_samples(mesh: trimesh.Trimesh, count: int) -> np.ndarray:
    """Area-weighted deterministic surface samples, independent of RNG state."""

    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    if len(triangles) == 0:
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if len(vertices) == 0:
            return np.empty((0, 3), dtype=np.float64)
        indices = np.linspace(0, len(vertices) - 1, count).astype(np.int64)
        return vertices[indices]
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        indices = np.linspace(0, len(triangles) - 1, count).astype(np.int64)
    else:
        cumulative = np.cumsum(areas)
        targets = (np.arange(count, dtype=np.float64) + 0.5) / count * cumulative[-1]
        indices = np.searchsorted(cumulative, targets, side="left")
    selected = triangles[np.clip(indices, 0, len(triangles) - 1)]
    sample_ids = np.arange(count, dtype=np.float64)
    u = np.mod(sample_ids * 0.6180339887498949 + 0.2113248654051871, 1.0)
    v = np.mod(sample_ids * 0.4142135623730950 + 0.3660254037844386, 1.0)
    reflect = (u + v) > 1.0
    u[reflect] = 1.0 - u[reflect]
    v[reflect] = 1.0 - v[reflect]
    return selected[:, 0] + u[:, None] * (selected[:, 1] - selected[:, 0]) + v[:, None] * (
        selected[:, 2] - selected[:, 0]
    )


def _primitive_samples(geometry: ET.Element, count: int) -> np.ndarray:
    box = geometry.find("box")
    if box is not None:
        extents = np.fromstring(box.attrib["size"], sep=" ", dtype=np.float64)
        return _deterministic_mesh_samples(trimesh.creation.box(extents=extents), count)
    sphere = geometry.find("sphere")
    if sphere is not None:
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=float(sphere.attrib["radius"]))
        return _deterministic_mesh_samples(mesh, count)
    cylinder = geometry.find("cylinder")
    if cylinder is not None:
        mesh = trimesh.creation.cylinder(
            radius=float(cylinder.attrib["radius"]), height=float(cylinder.attrib["length"]), sections=24
        )
        return _deterministic_mesh_samples(mesh, count)
    return np.empty((0, 3), dtype=np.float64)


@dataclass(frozen=True)
class URDFJoint:
    name: str
    parent: str
    child: str
    joint_type: str
    origin: np.ndarray
    axis: np.ndarray
    mimic_joint: Optional[str]
    mimic_multiplier: float
    mimic_offset: float


class URDFCollisionKinematics:
    """Minimal URDF FK plus deterministic samples from every collision link."""

    def __init__(
        self,
        urdf_path: Path,
        dof_names: Sequence[str],
        points_per_link: int = 32,
    ):
        self.urdf_path = urdf_path.resolve()
        self.dof_names = list(dof_names)
        root = ET.parse(self.urdf_path).getroot()
        links = [element.attrib["name"] for element in root.findall("link")]
        self.joints: List[URDFJoint] = []
        child_links = set()
        for element in root.findall("joint"):
            parent = element.find("parent").attrib["link"]
            child = element.find("child").attrib["link"]
            axis_element = element.find("axis")
            axis = np.fromstring(
                "0 0 1" if axis_element is None else axis_element.attrib.get("xyz", "0 0 1"),
                sep=" ",
                dtype=np.float64,
            )
            norm = np.linalg.norm(axis)
            if norm > 0:
                axis = axis / norm
            mimic_element = element.find("mimic")
            self.joints.append(
                URDFJoint(
                    name=element.attrib["name"],
                    parent=parent,
                    child=child,
                    joint_type=element.attrib.get("type", "fixed"),
                    origin=_origin_matrix(element.find("origin")),
                    axis=axis,
                    mimic_joint=None if mimic_element is None else mimic_element.attrib["joint"],
                    mimic_multiplier=1.0
                    if mimic_element is None
                    else float(mimic_element.attrib.get("multiplier", 1.0)),
                    mimic_offset=0.0
                    if mimic_element is None
                    else float(mimic_element.attrib.get("offset", 0.0)),
                )
            )
            child_links.add(child)
        roots = [link for link in links if link not in child_links]
        if len(roots) != 1:
            raise ValueError(f"Expected one URDF root link, found {roots}")
        self.root_link = roots[0]
        self.children: Dict[str, List[URDFJoint]] = {}
        for joint in self.joints:
            self.children.setdefault(joint.parent, []).append(joint)

        collision_points: Dict[str, List[np.ndarray]] = {}
        for link in root.findall("link"):
            link_name = link.attrib["name"]
            collision_elements = link.findall("collision")
            if not collision_elements:
                continue
            per_collision = max(4, int(np.ceil(points_per_link / len(collision_elements))))
            for collision in collision_elements:
                geometry = collision.find("geometry")
                if geometry is None:
                    continue
                mesh_element = geometry.find("mesh")
                if mesh_element is not None:
                    mesh_path = self._resolve_mesh_path(mesh_element.attrib["filename"])
                    mesh = _as_mesh(trimesh.load(mesh_path, process=False, force="mesh"))
                    scale = np.fromstring(mesh_element.attrib.get("scale", "1 1 1"), sep=" ")
                    if scale.size != 3:
                        raise ValueError(f"Invalid mesh scale in {self.urdf_path}: {mesh_element.attrib}")
                    mesh = mesh.copy()
                    mesh.vertices = np.asarray(mesh.vertices) * scale[None]
                    points = _deterministic_mesh_samples(mesh, per_collision)
                else:
                    points = _primitive_samples(geometry, per_collision)
                if len(points):
                    points = _transform_points(points, _origin_matrix(collision.find("origin")))
                    collision_points.setdefault(link_name, []).append(points)
        self.link_points: Dict[str, np.ndarray] = {
            link: np.concatenate(parts, axis=0)[:points_per_link]
            for link, parts in collision_points.items()
        }
        if not self.link_points:
            raise ValueError(f"No collision geometry was sampled from {self.urdf_path}")
        self.point_link_names: List[str] = []
        self.point_link_ids: List[int] = []
        points = []
        for link_id, (link, link_points) in enumerate(self.link_points.items()):
            self.point_link_names.append(link)
            self.point_link_ids.extend([link_id] * len(link_points))
            points.append(link_points)
        self.local_points = np.concatenate(points, axis=0)
        self.point_link_ids_array = np.asarray(self.point_link_ids, dtype=np.int64)

    def _resolve_mesh_path(self, filename: str) -> Path:
        if filename.startswith("package://"):
            filename = filename[len("package://") :]
        candidate = Path(filename)
        if not candidate.is_absolute():
            candidate = self.urdf_path.parent / candidate
        candidate = candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"URDF collision mesh does not exist: {candidate}")
        return candidate

    def _joint_value(self, joint: URDFJoint, q_map: Mapping[str, float]) -> float:
        # Isaac Gym exposes mimic joints as regular DOFs for the Inspire URDF,
        # and the project stores those values explicitly in opt_dof_pos.
        if joint.name in q_map:
            return q_map[joint.name]
        if joint.mimic_joint is not None:
            return q_map.get(joint.mimic_joint, 0.0) * joint.mimic_multiplier + joint.mimic_offset
        return q_map.get(joint.name, 0.0)

    def link_transforms(self, dof: np.ndarray) -> Dict[str, np.ndarray]:
        q_map = {name: float(value) for name, value in zip(self.dof_names, dof)}
        transforms = {self.root_link: np.eye(4, dtype=np.float64)}
        stack = [self.root_link]
        while stack:
            parent = stack.pop()
            for joint in self.children.get(parent, []):
                motion = np.eye(4, dtype=np.float64)
                value = self._joint_value(joint, q_map)
                if joint.joint_type in ("revolute", "continuous"):
                    motion[:3, :3] = Rotation.from_rotvec(joint.axis * value).as_matrix()
                elif joint.joint_type == "prismatic":
                    motion[:3, 3] = joint.axis * value
                transforms[joint.child] = transforms[parent] @ joint.origin @ motion
                stack.append(joint.child)
        return transforms

    def root_frame_surface_points(self, dof: np.ndarray) -> np.ndarray:
        transforms = self.link_transforms(dof)
        result = []
        for link in self.point_link_names:
            if link not in transforms:
                raise ValueError(f"Collision link {link!r} is disconnected in {self.urdf_path}")
            result.append(_transform_points(self.link_points[link], transforms[link]))
        return np.concatenate(result, axis=0)

    def world_surface_trajectory(
        self,
        wrist_pos: np.ndarray,
        wrist_rotvec: np.ndarray,
        dof: np.ndarray,
        frame_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if frame_ids is None:
            frame_ids = np.arange(len(wrist_pos), dtype=np.int64)
        wrist_rotations = Rotation.from_rotvec(wrist_rotvec[frame_ids]).as_matrix()
        frames = []
        for local_idx, frame_id in enumerate(frame_ids):
            root_points = self.root_frame_surface_points(dof[frame_id])
            frames.append(root_points @ wrist_rotations[local_idx].T + wrist_pos[frame_id])
        return np.stack(frames, axis=0)


@dataclass
class OakInkContext:
    data_idx: str
    annotation_path: Path
    mesh_path: Path
    collision_urdf_path: Path
    object_id: str
    object_trajectory: np.ndarray
    frame_ids: List[int]


def _mujoco_to_gym_transform() -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec([0.0, 0.0, -np.pi / 2]).as_matrix() @ Rotation.from_rotvec(
        [np.pi / 2, 0.0, 0.0]
    ).as_matrix()
    transform[:3, 3] = np.array([0.0, 0.0, 0.415], dtype=np.float64)
    return transform


def load_oakink_context(
    data_idx: str,
    side: str,
    data_root: Path,
    mesh_override: Optional[Path] = None,
    collision_urdf_override: Optional[Path] = None,
    skip: int = 2,
) -> OakInkContext:
    if "@" not in data_idx:
        raise ValueError("Contact trajectory retargeting currently requires an OakInk2 index such as 083f7@0")
    sequence_hash, stage_text = data_idx.rsplit("@", 1)
    stage = int(stage_text)
    annotation_paths = sorted((data_root / "anno_preview").glob("*.pkl"))
    matches = [
        path
        for path in annotation_paths
        if len(path.name.split("_")) > 5 and path.name.split("_")[5].startswith(sequence_hash)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one annotation matching {sequence_hash!r}, found {matches}")
    annotation_path = matches[0]
    with annotation_path.open("rb") as stream:
        annotation = pickle.load(stream)
    program_path = data_root / "program" / "program_info" / f"{annotation_path.stem}.json"
    with program_path.open("r", encoding="utf-8") as stream:
        raw_program = json.load(stream)
    range_keys = [ast.literal_eval(key) for key in raw_program]
    if not 0 <= stage < len(range_keys):
        raise IndexError(f"Stage {stage} is outside [0, {len(range_keys)}) for {annotation_path.name}")
    selected_ranges = range_keys[stage]
    side_index = 1 if side == "right" else 0
    selected_range = selected_ranges[side_index]
    other_range = selected_ranges[1 - side_index]
    if selected_range is None:
        raise ValueError(f"{data_idx} has no {side}-hand range")
    if other_range is not None:
        selected_range = (
            max(selected_range[0], other_range[0]),
            min(selected_range[1], other_range[1]),
        )
        if selected_range[0] >= selected_range[1]:
            raise ValueError(f"The left/right ranges do not overlap for {data_idx}")
    frame_ids = [
        int(frame_id)
        for frame_id in annotation["mocap_frame_id_list"][::skip]
        if selected_range[0] <= frame_id <= selected_range[1]
    ]
    selected_program = raw_program[list(raw_program.keys())[stage]]
    object_key = "obj_list_rh" if side == "right" else "obj_list_lh"
    object_id = selected_program[object_key][0]
    raw_trajectory = np.stack([np.asarray(annotation["obj_transf"][object_id][frame]) for frame in frame_ids])
    object_trajectory = _mujoco_to_gym_transform()[None] @ raw_trajectory
    object_dir = data_root / "object_preview" / "align_ds" / object_id
    candidates = sorted(
        path for path in object_dir.iterdir() if path.suffix.lower() in (".ply", ".obj")
    )
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one mesh in {object_dir}, found {candidates}")
    dataset_mesh_path = candidates[0]
    mesh_path = dataset_mesh_path if mesh_override is None else mesh_override.resolve()
    collision_urdf_path = (
        data_root
        / "coacd_object_preview"
        / "align_ds"
        / object_id
        / dataset_mesh_path.with_suffix(".urdf").name
        if collision_urdf_override is None
        else collision_urdf_override.resolve()
    )
    _urdf_bundle_sha256(collision_urdf_path)
    return OakInkContext(
        data_idx=data_idx,
        annotation_path=annotation_path,
        mesh_path=mesh_path,
        collision_urdf_path=collision_urdf_path,
        object_id=object_id,
        object_trajectory=object_trajectory.astype(np.float64),
        frame_ids=frame_ids,
    )


@dataclass
class SemanticTool:
    mesh: trimesh.Trimesh
    red_points: np.ndarray
    black_points: np.ndarray
    red_tree: cKDTree
    black_tree: cKDTree
    red_vertex_count: int
    black_vertex_count: int


def load_semantic_tool(mesh_path: Path) -> SemanticTool:
    mesh = _as_mesh(trimesh.load(mesh_path, process=False, force="mesh"))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    colors = getattr(getattr(mesh, "visual", None), "vertex_colors", None)
    if colors is None or len(colors) != len(vertices):
        raise ValueError(f"Tool mesh must have per-vertex red/black colors: {mesh_path}")
    colors = np.asarray(colors[:, :3], dtype=np.float64)
    if colors.max(initial=0.0) <= 1.0:
        colors *= 255.0
    red_mask = (colors[:, 0] > 150.0) & (colors[:, 1] < 100.0) & (colors[:, 2] < 100.0)
    black_mask = (colors[:, 0] < 80.0) & (colors[:, 1] < 80.0) & (colors[:, 2] < 80.0)
    if not red_mask.any() or not black_mask.any():
        raise ValueError(
            f"Both red and black vertices are required; red={red_mask.sum()}, black={black_mask.sum()} in {mesh_path}"
        )
    faces = np.asarray(mesh.faces, dtype=np.int64)
    red_faces = faces[red_mask[faces].sum(axis=1) >= 2]
    black_faces = faces[black_mask[faces].sum(axis=1) >= 2]
    red_points = [vertices[red_mask]]
    black_points = [vertices[black_mask]]
    if len(red_faces):
        red_points.append(vertices[red_faces].mean(axis=1))
    if len(black_faces):
        black_points.append(vertices[black_faces].mean(axis=1))
    red_points_array = np.concatenate(red_points, axis=0)
    black_points_array = np.concatenate(black_points, axis=0)
    return SemanticTool(
        mesh=mesh,
        red_points=red_points_array,
        black_points=black_points_array,
        red_tree=cKDTree(red_points_array),
        black_tree=cKDTree(black_points_array),
        red_vertex_count=int(red_mask.sum()),
        black_vertex_count=int(black_mask.sum()),
    )


def world_to_tool_points(world_points: np.ndarray, object_trajectory: np.ndarray) -> np.ndarray:
    rotations = object_trajectory[:, :3, :3]
    translations = object_trajectory[:, :3, 3]
    return np.einsum("fji,fpj->fpi", rotations, world_points - translations[:, None])


def tool_to_world_offsets(offset_tool: np.ndarray, object_trajectory: np.ndarray) -> np.ndarray:
    return np.einsum("fij,j->fi", object_trajectory[:, :3, :3], offset_tool)


def smoothstep5(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0.0, 1.0)
    return value**3 * (value * (value * 6.0 - 15.0) + 10.0)


def build_offset_schedule(
    length: int,
    start_frame: int,
    end_frame: int,
    ramp_in_frames: int,
    ramp_out_frames: int,
) -> np.ndarray:
    if not 0 <= start_frame <= end_frame < length:
        raise ValueError(f"Invalid correction interval [{start_frame}, {end_frame}] for length {length}")
    alpha = np.zeros(length, dtype=np.float64)
    alpha[start_frame : end_frame + 1] = 1.0
    if ramp_in_frames > 0:
        ramp_start = max(0, start_frame - ramp_in_frames)
        count = start_frame - ramp_start
        if count:
            alpha[ramp_start:start_frame] = smoothstep5(np.arange(count, dtype=np.float64) / count)
    if ramp_out_frames > 0:
        ramp_end = min(length, end_frame + 1 + ramp_out_frames)
        count = ramp_end - (end_frame + 1)
        if count:
            alpha[end_frame + 1 : ramp_end] = 1.0 - smoothstep5(
                (np.arange(count, dtype=np.float64) + 1.0) / count
            )
    return alpha


def _close_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    result = mask.copy()
    true_ids = np.flatnonzero(result)
    for left, right in zip(true_ids[:-1], true_ids[1:]):
        if 1 < right - left <= max_gap + 1:
            result[left : right + 1] = True
    return result


def _longest_true_interval(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    best = None
    start = None
    for index, value in enumerate(np.r_[mask, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            candidate = (start, index - 1)
            if best is None or candidate[1] - candidate[0] > best[1] - best[0]:
                best = candidate
            start = None
    return best


@dataclass
class GeometrySettings:
    contact_distance: float = 0.006
    black_clearance: float = 0.006
    influence_distance: float = 0.02
    grasp_distance: float = 0.025
    min_red_link_ratio: float = 0.6
    min_red_frame_ratio: float = 0.8
    max_shift: float = 0.05
    max_black_violation_fraction: float = 0.0
    semantic_tie_epsilon: float = 1e-6


def _nearest_semantics(points: np.ndarray, tool: SemanticTool, tie_epsilon: float):
    flat = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    red_distance, red_index = tool.red_tree.query(flat, workers=1)
    black_distance, black_index = tool.black_tree.query(flat, workers=1)
    black_nearest = black_distance + tie_epsilon < red_distance
    shape = points.shape[:-1]
    return (
        red_distance.reshape(shape),
        black_distance.reshape(shape),
        black_nearest.reshape(shape),
        red_index.reshape(shape),
        black_index.reshape(shape),
    )


def contact_geometry_metrics(
    points_tool: np.ndarray,
    point_link_ids: np.ndarray,
    contact_link_ids: np.ndarray,
    tool: SemanticTool,
    settings: GeometrySettings,
) -> Dict[str, object]:
    red_distance, black_distance, black_nearest, _, _ = _nearest_semantics(
        points_tool, tool, settings.semantic_tie_epsilon
    )
    surface_distance = np.minimum(red_distance, black_distance)
    black_near = black_nearest & (surface_distance < settings.influence_distance)
    black_violation = black_nearest & (black_distance < settings.black_clearance)
    contact_link_set = set(int(value) for value in np.asarray(contact_link_ids).tolist())
    link_red_contact = np.zeros((len(points_tool), len(contact_link_set)), dtype=bool)
    link_red_distance = np.full((len(points_tool), len(contact_link_set)), np.inf, dtype=np.float64)
    for output_index, link_id in enumerate(sorted(contact_link_set)):
        link_mask = point_link_ids == link_id
        if not link_mask.any():
            continue
        link_red_contact[:, output_index] = (
            (~black_nearest[:, link_mask]) & (red_distance[:, link_mask] <= settings.contact_distance)
        ).any(axis=1)
        semantic_red_distance = red_distance[:, link_mask] + np.maximum(
            red_distance[:, link_mask] - black_distance[:, link_mask] + settings.semantic_tie_epsilon,
            0.0,
        )
        link_red_distance[:, output_index] = semantic_red_distance.min(axis=1)
    red_link_ratio_frame = link_red_contact.mean(axis=1) if link_red_contact.shape[1] else np.zeros(len(points_tool))
    black_count_frame = black_violation.sum(axis=1)
    black_near_count_frame = black_near.sum(axis=1)
    black_distances = black_distance[black_nearest]
    return {
        "red_link_ratio_frame": red_link_ratio_frame,
        "red_link_distance_frame": link_red_distance,
        "black_violation_count_frame": black_count_frame,
        "black_near_count_frame": black_near_count_frame,
        "surface_min_distance_frame": surface_distance.min(axis=1),
        "red_link_ratio_mean": float(red_link_ratio_frame.mean()),
        "red_frame_pass_ratio": float((red_link_ratio_frame >= settings.min_red_link_ratio).mean()),
        "black_violation_count": int(black_violation.sum()),
        "black_violation_fraction": float(black_violation.mean()),
        "black_near_fraction": float(black_near.mean()),
        "black_min_distance": float(black_distances.min()) if black_distances.size else None,
        "min_surface_distance": float(surface_distance.min()),
    }


def _summary_metrics(metrics: Mapping[str, object]) -> Dict[str, object]:
    return {
        key: value
        for key, value in metrics.items()
        if not isinstance(value, np.ndarray)
    }


def detect_grasp_interval(
    points_tool: np.ndarray,
    point_link_ids: np.ndarray,
    contact_link_ids: np.ndarray,
    tool: SemanticTool,
    settings: GeometrySettings,
    max_gap: int = 10,
    min_frames: int = 8,
) -> Tuple[int, int]:
    red_distance, black_distance, _, _, _ = _nearest_semantics(
        points_tool, tool, settings.semantic_tie_epsilon
    )
    surface_distance = np.minimum(red_distance, black_distance)
    active_per_link = []
    for link_id in sorted(set(int(value) for value in contact_link_ids.tolist())):
        link_mask = point_link_ids == link_id
        active_per_link.append((surface_distance[:, link_mask].min(axis=1) <= settings.grasp_distance))
    active = np.stack(active_per_link, axis=1).mean(axis=1) >= (1.0 / max(1, len(active_per_link)))
    active = _close_short_gaps(active, max_gap=max_gap)
    interval = _longest_true_interval(active)
    if interval is None or interval[1] - interval[0] + 1 < min_frames:
        raise ValueError(
            "Could not detect a stable grasp interval from the new mesh. "
            "Pass --start-frame and --end-frame explicitly after inspecting the preview."
        )
    return interval


def _optimization_objective(
    delta: np.ndarray,
    points_tool: np.ndarray,
    point_link_ids: np.ndarray,
    contact_link_ids: np.ndarray,
    tool: SemanticTool,
    settings: GeometrySettings,
) -> float:
    shifted = points_tool + np.asarray(delta, dtype=np.float64)[None, None]
    metrics = contact_geometry_metrics(shifted, point_link_ids, contact_link_ids, tool, settings)
    link_distances = np.asarray(metrics["red_link_distance_frame"])
    required = max(1, int(np.ceil(settings.min_red_link_ratio * link_distances.shape[1])))
    best_required = np.partition(link_distances, required - 1, axis=1)[:, :required]
    red_loss = np.mean(np.square(np.clip(best_required / settings.contact_distance, 0.0, 10.0)))

    red_distance, black_distance, black_nearest, _, _ = _nearest_semantics(
        shifted, tool, settings.semantic_tie_epsilon
    )
    black_depth = np.where(
        black_nearest,
        np.maximum(settings.black_clearance - black_distance, 0.0) / settings.black_clearance,
        0.0,
    )
    black_loss = float(np.mean(black_depth**2) + np.mean(np.max(black_depth, axis=1) ** 2))
    shift_loss = float(np.dot(delta, delta) / max(settings.max_shift**2, 1e-12))
    return float(red_loss + 80.0 * black_loss + 0.02 * shift_loss)


def _red_direction_seed(
    points_tool: np.ndarray,
    point_link_ids: np.ndarray,
    contact_link_ids: np.ndarray,
    tool: SemanticTool,
    max_shift: float,
) -> np.ndarray:
    contact_mask = np.isin(point_link_ids, contact_link_ids)
    contact_points = points_tool[:, contact_mask].reshape(-1, 3)
    distances, indices = tool.red_tree.query(contact_points, workers=1)
    if not len(distances):
        return np.zeros(3, dtype=np.float64)
    keep = distances <= np.quantile(distances, 0.1)
    vectors = tool.red_points[indices[keep]] - contact_points[keep]
    seed = np.median(vectors, axis=0)
    return np.clip(seed, -max_shift, max_shift)


def optimize_tool_translation(
    points_tool: np.ndarray,
    point_link_ids: np.ndarray,
    contact_link_ids: np.ndarray,
    tool: SemanticTool,
    settings: GeometrySettings,
    seed: int = 42,
    max_iterations: int = 30,
    initial_delta: Optional[np.ndarray] = None,
    global_search: bool = True,
) -> Tuple[np.ndarray, Dict[str, object]]:
    bounds = [(-settings.max_shift, settings.max_shift)] * 3
    objective = lambda delta: _optimization_objective(
        delta, points_tool, point_link_ids, contact_link_ids, tool, settings
    )
    red_seed = _red_direction_seed(
        points_tool, point_link_ids, contact_link_ids, tool, settings.max_shift
    )
    candidates = [
        (np.zeros(3, dtype=np.float64), objective(np.zeros(3, dtype=np.float64))),
        (red_seed, objective(red_seed)),
    ]
    if initial_delta is not None:
        initial_delta = np.clip(np.asarray(initial_delta, dtype=np.float64), -settings.max_shift, settings.max_shift)
        candidates.append((initial_delta, objective(initial_delta)))
    result = None
    if global_search:
        result = differential_evolution(
            objective,
            bounds=bounds,
            seed=seed,
            maxiter=max_iterations,
            popsize=6,
            tol=1e-5,
            polish=False,
            updating="immediate",
            workers=1,
        )
        candidates.append((np.asarray(result.x), float(result.fun)))
    best_seed, _ = min(candidates, key=lambda item: item[1])
    polished = minimize(
        objective,
        best_seed,
        method="Powell",
        bounds=bounds,
        options={"maxiter": 120, "xtol": 1e-5, "ftol": 1e-6},
    )
    candidates.append((np.asarray(polished.x), float(polished.fun)))
    delta, score = min(candidates, key=lambda item: item[1])
    return delta.astype(np.float64), {
        "objective": float(score),
        "global_search": bool(global_search),
        "differential_evolution_success": None if result is None else bool(result.success),
        "differential_evolution_message": None if result is None else str(result.message),
        "powell_success": bool(polished.success),
        "powell_message": str(polished.message),
        "red_direction_seed": red_seed,
    }


@dataclass(frozen=True)
class OfflineHandSpec:
    name: str
    side: str
    urdf_path: Path
    dof_names: Tuple[str, ...]
    contact_body_names: Tuple[str, ...]

    def __str__(self) -> str:
        return f"{self.name}_{'rh' if self.side == 'right' else 'lh'}"


def _urdf_actuated_joint_names(urdf_path: Path) -> Tuple[str, ...]:
    root = ET.parse(urdf_path).getroot()
    result = []
    for joint in root.findall("joint"):
        if joint.attrib.get("type", "fixed") in ("fixed", "floating"):
            continue
        # Mimic joints are functions of an actuated joint and do not consume a
        # column in the project's opt_dof_pos arrays.
        if joint.find("mimic") is not None:
            continue
        result.append(joint.attrib["name"])
    return tuple(result)


def _resolve_dexhand(name: str, side: str) -> OfflineHandSpec:
    """Resolve hand metadata without importing Isaac/PyTorch dataset modules."""

    suffix = "right" if side == "right" else "left"
    urdf_relpaths = {
        ("artimano", "right"): "physgraph_envs/assets/mano_urdf/rh_mano.urdf",
        ("artimano", "left"): "physgraph_envs/assets/mano_urdf/lh_mano.urdf",
        ("inspire", "right"): "physgraph_envs/assets/inspire_hand/inspire_hand_right.urdf",
        ("inspire", "left"): "physgraph_envs/assets/inspire_hand/inspire_hand_left.urdf",
        ("allegro", "right"): "physgraph_envs/assets/allegro_hand/allegro_hand_right.urdf",
        ("allegro", "left"): "physgraph_envs/assets/allegro_hand/allegro_hand_left.urdf",
        ("shadow", "right"): "physgraph_envs/assets/shadow_hand/shadow_hand_right_woarm.urdf",
        ("shadow", "left"): "physgraph_envs/assets/shadow_hand/shadow_hand_left_woarm.urdf",
    }
    artimano_dofs = (
        "j_index1y", "j_index1z", "j_index2", "j_index3",
        "j_middle1y", "j_middle1z", "j_middle2", "j_middle3",
        "j_pinky1y", "j_pinky1z", "j_pinky2", "j_pinky3",
        "j_ring1y", "j_ring1z", "j_ring2", "j_ring3",
        "j_thumb1x", "j_thumb1y", "j_thumb1z", "j_thumb2y", "j_thumb2z", "j_thumb3",
    )
    inspire_base_dofs = (
        "thumb_proximal_yaw_joint", "thumb_proximal_pitch_joint",
        "index_proximal_joint", "index_intermediate_joint",
        "middle_proximal_joint", "middle_intermediate_joint",
        "pinky_proximal_joint", "pinky_intermediate_joint",
        "ring_proximal_joint", "ring_intermediate_joint",
        "thumb_intermediate_joint", "thumb_distal_joint",
    )
    allegro_dofs = (
        "joint_0.0", "joint_1.0", "joint_2.0", "joint_3.0",
        "joint_12.0", "joint_13.0", "joint_14.0", "joint_15.0",
        "joint_4.0", "joint_5.0", "joint_6.0", "joint_7.0",
        "joint_8.0", "joint_9.0", "joint_10.0", "joint_11.0",
    )
    shadow_dofs = (
        "FFJ4", "FFJ3", "FFJ2", "FFJ1",
        "LFJ5", "LFJ4", "LFJ3", "LFJ2", "LFJ1",
        "MFJ4", "MFJ3", "MFJ2", "MFJ1",
        "RFJ4", "RFJ3", "RFJ2", "RFJ1",
        "THJ5", "THJ4", "THJ3", "THJ2", "THJ1",
    )
    dof_names = {
        ("artimano", "right"): artimano_dofs,
        ("artimano", "left"): artimano_dofs,
        ("inspire", "right"): tuple("R_" + value for value in inspire_base_dofs),
        ("inspire", "left"): tuple("L_" + value for value in inspire_base_dofs),
        ("allegro", "right"): allegro_dofs,
        ("allegro", "left"): allegro_dofs,
        ("shadow", "right"): shadow_dofs,
        ("shadow", "left"): shadow_dofs,
    }
    contact_names = {
        ("artimano", "right"): ("thumb3", "index3", "middle3", "ring3", "pinky3"),
        ("artimano", "left"): ("thumb3", "index3", "middle3", "ring3", "pinky3"),
        ("inspire", "right"): (
            "R_thumb_distal",
            "R_index_intermediate",
            "R_middle_intermediate",
            "R_ring_intermediate",
            "R_pinky_intermediate",
        ),
        ("inspire", "left"): (
            "L_thumb_distal",
            "L_index_intermediate",
            "L_middle_intermediate",
            "L_ring_intermediate",
            "L_pinky_intermediate",
        ),
        ("allegro", "right"): ("link_15.0", "link_3.0", "link_7.0", "link_11.0"),
        ("allegro", "left"): ("link_15.0", "link_11.0", "link_7.0", "link_3.0"),
        ("shadow", "right"): ("thdistal", "ffdistal", "mfdistal", "rfdistal", "lfdistal"),
        ("shadow", "left"): ("thdistal", "ffdistal", "mfdistal", "rfdistal", "lfdistal"),
    }
    key = (name.lower(), suffix)
    if key not in urdf_relpaths:
        raise ValueError(f"Unsupported offline dex hand {name!r}/{side!r}")
    urdf_path = (PROJECT_ROOT / urdf_relpaths[key]).resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"Dex-hand URDF does not exist: {urdf_path}")
    default_contacts = contact_names.get(key)
    if default_contacts is None:
        raise ValueError(
            f"{name}/{side} requires explicit contact-link metadata; add it to _resolve_dexhand before use"
        )
    return OfflineHandSpec(
        name=name.lower(),
        side=side,
        urdf_path=urdf_path,
        dof_names=dof_names.get(key, _urdf_actuated_joint_names(urdf_path)),
        contact_body_names=default_contacts,
    )


def _resolve_source_pkl(data_idx: str, dexhand_name: str, side: str) -> Path:
    suffix = "rh" if side == "right" else "lh"
    directory = PROJECT_ROOT / "data" / "retargeting" / "OakInk-v2" / f"mano2{dexhand_name}_{suffix}"
    sequence_hash, stage = data_idx.rsplit("@", 1)
    matches = sorted(directory.glob(f"*{sequence_hash}*@{stage}.pkl"))
    if len(matches) != 1:
        raise ValueError(f"Expected one source retarget PKL for {data_idx} in {directory}, found {matches}")
    return matches[0]


def _load_retarget_pkl(path: Path) -> Dict:
    with path.open("rb") as stream:
        data = pickle.load(stream)
    required = ("opt_wrist_pos", "opt_wrist_rot", "opt_dof_pos")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Retarget PKL {path} is missing required fields {missing}")
    lengths = {key: len(np.asarray(data[key])) for key in required}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Retarget fields have inconsistent lengths in {path}: {lengths}")
    return data


def _contact_link_ids(model: URDFCollisionKinematics, contact_link_names: Sequence[str]) -> np.ndarray:
    name_to_id = {name: index for index, name in enumerate(model.point_link_names)}
    missing = [name for name in contact_link_names if name not in name_to_id]
    if missing:
        raise ValueError(f"Contact links have no sampled collision geometry: {missing}")
    return np.asarray([name_to_id[name] for name in contact_link_names], dtype=np.int64)


def _select_optimization_frames(start: int, end: int, maximum: int = 48) -> np.ndarray:
    count = end - start + 1
    if count <= maximum:
        return np.arange(start, end + 1, dtype=np.int64)
    return np.unique(np.linspace(start, end, maximum).round().astype(np.int64))


def _is_feasible(metrics: Mapping[str, object], settings: GeometrySettings) -> bool:
    return bool(
        float(metrics["red_frame_pass_ratio"]) >= settings.min_red_frame_ratio
        and float(metrics["black_violation_fraction"]) <= settings.max_black_violation_fraction
    )


def _invariant_report(source: Mapping, corrected: Mapping, delta_world: np.ndarray) -> Dict[str, object]:
    source_rot = np.asarray(source["opt_wrist_rot"])
    corrected_rot = np.asarray(corrected["opt_wrist_rot"])
    source_dof = np.asarray(source["opt_dof_pos"])
    corrected_dof = np.asarray(corrected["opt_dof_pos"])
    source_pos = np.asarray(source["opt_wrist_pos"])
    corrected_pos = np.asarray(corrected["opt_wrist_pos"])
    result = {
        "rotation_max_abs_error": float(np.max(np.abs(corrected_rot - source_rot))),
        "dof_max_abs_error": float(np.max(np.abs(corrected_dof - source_dof))),
        "wrist_translation_max_abs_error": float(
            np.max(np.abs((corrected_pos - source_pos) - delta_world))
        ),
        "all_finite": bool(
            np.isfinite(corrected_pos).all()
            and np.isfinite(corrected_rot).all()
            and np.isfinite(corrected_dof).all()
        ),
        "length": int(len(corrected_pos)),
    }
    if "opt_joints_pos" in source and "opt_joints_pos" in corrected:
        source_joints = np.asarray(source["opt_joints_pos"])
        corrected_joints = np.asarray(corrected["opt_joints_pos"])
        result["joint_rigid_translation_max_abs_error"] = float(
            np.max(np.abs((corrected_joints - source_joints) - delta_world[:, None]))
        )
    result["passed"] = bool(
        result["all_finite"]
        and result["rotation_max_abs_error"] <= 1e-10
        and result["dof_max_abs_error"] <= 1e-10
        and result["wrist_translation_max_abs_error"] <= 1e-6
        and result.get("joint_rigid_translation_max_abs_error", 0.0) <= 1e-6
    )
    return result


def _write_preview(
    output_path: Path,
    tool: SemanticTool,
    points_before: np.ndarray,
    points_after: np.ndarray,
    point_link_ids: np.ndarray,
    contact_link_ids: np.ndarray,
    alpha: np.ndarray,
    before_full: Mapping[str, object],
    after_full: Mapping[str, object],
    preview_frame: int,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/physgraph-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(16, 9))
    axes = [figure.add_subplot(2, 3, index + 1, projection="3d") for index in range(3)]
    contact_mask = np.isin(point_link_ids, contact_link_ids)
    red = tool.red_points[:: max(1, len(tool.red_points) // 800)]
    black = tool.black_points[:: max(1, len(tool.black_points) // 800)]
    for axis, hand_points, title in (
        (axes[0], points_before[preview_frame], "Before correction"),
        (axes[1], points_after[preview_frame], "After correction"),
    ):
        axis.scatter(red[:, 0], red[:, 1], red[:, 2], c="red", s=2, alpha=0.35)
        axis.scatter(black[:, 0], black[:, 1], black[:, 2], c="black", s=2, alpha=0.2)
        axis.scatter(hand_points[:, 0], hand_points[:, 1], hand_points[:, 2], c="tab:blue", s=3)
        axis.scatter(
            hand_points[contact_mask, 0],
            hand_points[contact_mask, 1],
            hand_points[contact_mask, 2],
            c="limegreen",
            s=6,
        )
        axis.set_title(f"{title}, frame={preview_frame}")
        axis.set_box_aspect((1, 1, 1))
    axes[2].scatter(red[:, 0], red[:, 1], red[:, 2], c="red", s=3, alpha=0.5)
    axes[2].scatter(black[:, 0], black[:, 1], black[:, 2], c="black", s=3, alpha=0.3)
    axes[2].set_title("Tool semantic regions (tool frame)")
    axes[2].set_box_aspect((1, 1, 1))

    time_axis = figure.add_subplot(2, 1, 2)
    time_axis.plot(alpha, label="translation alpha", color="tab:gray")
    time_axis.plot(before_full["red_link_ratio_frame"], label="red link ratio before", color="tab:orange")
    time_axis.plot(after_full["red_link_ratio_frame"], label="red link ratio after", color="tab:green")
    before_bad = np.asarray(before_full["black_violation_count_frame"])
    after_bad = np.asarray(after_full["black_violation_count_frame"])
    scale = max(1.0, float(max(before_bad.max(initial=0), after_bad.max(initial=0))))
    time_axis.plot(before_bad / scale, "--", label="black violation before (normalized)", color="tab:red")
    time_axis.plot(after_bad / scale, "--", label="black violation after (normalized)", color="tab:blue")
    time_axis.set_xlabel("Frame")
    time_axis.set_ylim(bottom=-0.03)
    time_axis.grid(alpha=0.2)
    time_axis.legend(loc="best", ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def run_correction(args: argparse.Namespace) -> Dict[str, object]:
    hand = _resolve_dexhand(args.dexhand, args.side)
    input_path = Path(args.input_pkl).resolve() if args.input_pkl else _resolve_source_pkl(
        args.data_idx, args.dexhand, args.side
    )
    output_root = Path(args.output_root)
    output_path = (
        Path(args.output_pkl).resolve()
        if args.output_pkl
        else mirror_retarget_output_path(input_path, output_root)
    )
    if output_path == input_path:
        raise ValueError("Corrected output must not overwrite the source retarget PKL")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Corrected PKL already exists: {output_path}. Pass --overwrite to replace it."
        )
    context = load_oakink_context(
        args.data_idx,
        args.side,
        Path(args.data_root),
        None if args.mesh is None else Path(args.mesh),
        None if args.collision_urdf is None else Path(args.collision_urdf),
        skip=args.skip,
    )
    source = _load_retarget_pkl(input_path)
    if "contact_retarget" in source:
        raise ValueError(
            f"Input is already contact-corrected: {input_path}. Use its original source PKL instead."
        )
    length = len(source["opt_wrist_pos"])
    if len(context.object_trajectory) != length:
        raise ValueError(
            f"Object trajectory length {len(context.object_trajectory)} != retarget PKL length {length}. "
            "Check --skip, data index, and source PKL."
        )
    if np.asarray(source["opt_dof_pos"]).shape[1] != len(hand.dof_names):
        raise ValueError(
            f"PKL DOF count {np.asarray(source['opt_dof_pos']).shape[1]} != {hand} DOF count {len(hand.dof_names)}"
        )
    tool = load_semantic_tool(context.mesh_path)
    kinematics = URDFCollisionKinematics(
        Path(hand.urdf_path), hand.dof_names, points_per_link=args.points_per_link
    )
    contact_names = list(hand.contact_body_names)
    if args.contact_links:
        contact_names = [name.strip() for name in args.contact_links.split(",") if name.strip()]
    contact_ids = _contact_link_ids(kinematics, contact_names)

    world_before = kinematics.world_surface_trajectory(
        np.asarray(source["opt_wrist_pos"], dtype=np.float64),
        np.asarray(source["opt_wrist_rot"], dtype=np.float64),
        np.asarray(source["opt_dof_pos"], dtype=np.float64),
    )
    points_before = world_to_tool_points(world_before, context.object_trajectory)
    settings = GeometrySettings(
        contact_distance=args.contact_distance,
        black_clearance=args.black_clearance,
        influence_distance=args.influence_distance,
        grasp_distance=args.grasp_distance,
        min_red_link_ratio=args.min_red_link_ratio,
        min_red_frame_ratio=args.min_red_frame_ratio,
        max_shift=args.max_shift,
        max_black_violation_fraction=args.max_black_violation_fraction,
    )
    if (args.start_frame is None) != (args.end_frame is None):
        raise ValueError("--start-frame and --end-frame must be provided together")
    if args.start_frame is None:
        start_frame, end_frame = detect_grasp_interval(
            points_before,
            kinematics.point_link_ids_array,
            contact_ids,
            tool,
            settings,
            max_gap=args.max_grasp_gap,
            min_frames=args.min_grasp_frames,
        )
    else:
        start_frame, end_frame = args.start_frame, args.end_frame
        if not 0 <= start_frame <= end_frame < length:
            raise ValueError(
                f"Invalid correction interval [{start_frame}, {end_frame}] for length {length}"
            )
    optimization_frames = _select_optimization_frames(start_frame, end_frame, args.max_optimization_frames)
    delta_tool, optimizer_report = optimize_tool_translation(
        points_before[optimization_frames],
        kinematics.point_link_ids_array,
        contact_ids,
        tool,
        settings,
        seed=args.seed,
        max_iterations=args.optimizer_iterations,
    )
    optimizer_history = [optimizer_report]

    # A sparse set keeps the global search tractable, but final acceptance is
    # based on every frame in the grasp interval. Add any missed failure frames
    # and locally refine until the full interval satisfies the constraints.
    hold_frame_ids = np.arange(start_frame, end_frame + 1, dtype=np.int64)
    for refinement_round in range(args.max_refinement_rounds):
        hold_metrics = contact_geometry_metrics(
            points_before[hold_frame_ids] + delta_tool[None, None],
            kinematics.point_link_ids_array,
            contact_ids,
            tool,
            settings,
        )
        if _is_feasible(hold_metrics, settings):
            break
        failed_local = np.flatnonzero(
            (np.asarray(hold_metrics["red_link_ratio_frame"]) < settings.min_red_link_ratio)
            | (np.asarray(hold_metrics["black_violation_count_frame"]) > 0)
        )
        if len(failed_local) == 0:
            break
        failed_frames = hold_frame_ids[failed_local]
        optimization_frames = np.unique(np.concatenate([optimization_frames, failed_frames]))
        delta_tool, refinement_report = optimize_tool_translation(
            points_before[optimization_frames],
            kinematics.point_link_ids_array,
            contact_ids,
            tool,
            settings,
            seed=args.seed + refinement_round + 1,
            max_iterations=args.optimizer_iterations,
            initial_delta=delta_tool,
            global_search=False,
        )
        refinement_report["round"] = refinement_round + 1
        refinement_report["added_failure_frames"] = failed_frames
        optimizer_history.append(refinement_report)
    alpha = build_offset_schedule(
        length,
        start_frame,
        end_frame,
        args.ramp_in_frames,
        args.ramp_out_frames,
    )
    delta_world = tool_to_world_offsets(delta_tool, context.object_trajectory) * alpha[:, None]
    corrected = copy.deepcopy(source)
    corrected["opt_wrist_pos"] = (
        np.asarray(source["opt_wrist_pos"], dtype=np.float64) + delta_world
    ).astype(np.asarray(source["opt_wrist_pos"]).dtype)
    if "opt_joints_pos" in source:
        corrected["opt_joints_pos"] = (
            np.asarray(source["opt_joints_pos"], dtype=np.float64) + delta_world[:, None]
        ).astype(np.asarray(source["opt_joints_pos"]).dtype)
    # Rotation and DOF arrays are copied without arithmetic so byte-level values
    # remain unchanged apart from normal pickle serialization.
    corrected["opt_wrist_rot"] = np.asarray(source["opt_wrist_rot"]).copy()
    corrected["opt_dof_pos"] = np.asarray(source["opt_dof_pos"]).copy()

    points_after = points_before + alpha[:, None, None] * delta_tool[None, None]
    before_hold = contact_geometry_metrics(
        points_before[hold_frame_ids],
        kinematics.point_link_ids_array,
        contact_ids,
        tool,
        settings,
    )
    after_hold = contact_geometry_metrics(
        points_after[hold_frame_ids],
        kinematics.point_link_ids_array,
        contact_ids,
        tool,
        settings,
    )
    before_full = contact_geometry_metrics(
        points_before,
        kinematics.point_link_ids_array,
        contact_ids,
        tool,
        settings,
    )
    after_full = contact_geometry_metrics(
        points_after,
        kinematics.point_link_ids_array,
        contact_ids,
        tool,
        settings,
    )
    invariant_report = _invariant_report(source, corrected, delta_world)
    geometry_feasible = _is_feasible(after_hold, settings)
    overall_passed = bool(geometry_feasible and invariant_report["passed"])
    report = {
        "schema_version": SCHEMA_VERSION,
        "passed": overall_passed,
        "geometry_feasible": geometry_feasible,
        "input_pkl": input_path,
        "output_pkl": output_path,
        "data_idx": args.data_idx,
        "side": args.side,
        "dexhand": args.dexhand,
        "annotation_path": context.annotation_path,
        "object_id": context.object_id,
        "tool_mesh": context.mesh_path,
        "source_pkl_sha256": _sha256(input_path),
        "tool_mesh_sha256": _sha256(context.mesh_path),
        "collision_urdf": context.collision_urdf_path,
        "collision_bundle_sha256": _urdf_bundle_sha256(context.collision_urdf_path),
        "red_vertex_count": tool.red_vertex_count,
        "black_vertex_count": tool.black_vertex_count,
        "hand_collision_link_count": len(kinematics.point_link_names),
        "hand_surface_point_count": len(kinematics.local_points),
        "contact_links": contact_names,
        "grasp_interval": [start_frame, end_frame],
        "optimization_frames": optimization_frames,
        "delta_tool": delta_tool,
        "delta_norm": float(np.linalg.norm(delta_tool)),
        "settings": asdict(settings),
        "optimizer": optimizer_history[-1],
        "optimizer_history": optimizer_history,
        "before_hold": _summary_metrics(before_hold),
        "after_hold": _summary_metrics(after_hold),
        "before_full": _summary_metrics(before_full),
        "after_full": _summary_metrics(after_full),
        "invariants": invariant_report,
        "physics_validation_required": True,
        "physics_validation_note": (
            "This offline pass validates kinematics and semantic surface clearance. "
            "Replay the saved PKL with the matching new collision URDF before Stage 1 training."
        ),
    }
    corrected["contact_retarget"] = {
        "schema_version": SCHEMA_VERSION,
        "source_pkl": str(input_path),
        "data_idx": args.data_idx,
        "side": args.side,
        "dexhand": args.dexhand,
        "tool_mesh": str(context.mesh_path),
        "tool_mesh_sha256": report["tool_mesh_sha256"],
        "collision_urdf": str(context.collision_urdf_path),
        "collision_bundle_sha256": report["collision_bundle_sha256"],
        "delta_tool": delta_tool.astype(np.float32),
        "alpha": alpha.astype(np.float32),
        "grasp_interval": np.asarray([start_frame, end_frame], dtype=np.int64),
        "geometry_feasible": geometry_feasible,
        "settings": asdict(settings),
        "validation_summary": {
            "before_hold": report["before_hold"],
            "after_hold": report["after_hold"],
            "invariants": invariant_report,
        },
    }
    report_path = (
        Path(args.report)
        if args.report
        else sequence_artifact_path(output_path, ".contact_report.json")
    )
    preview_path = (
        Path(args.preview)
        if args.preview
        else sequence_artifact_path(output_path, ".contact_preview.png")
    )
    _atomic_json_dump(report, report_path)
    preview_frame = int((start_frame + end_frame) // 2)
    _write_preview(
        preview_path,
        tool,
        points_before,
        points_after,
        kinematics.point_link_ids_array,
        contact_ids,
        alpha,
        before_full,
        after_full,
        preview_frame,
    )
    if overall_passed or args.allow_infeasible:
        _atomic_pickle_dump(corrected, output_path)
    else:
        raise RuntimeError(
            "Geometric correction did not satisfy the configured feasibility thresholds. "
            f"The PKL was not written. Inspect {report_path} and {preview_path}, adjust the interval/constraints, "
            "or pass --allow-infeasible only for diagnostic output."
        )
    return report


def run_verification(args: argparse.Namespace) -> Dict[str, object]:
    corrected_path = Path(args.input_pkl).resolve()
    corrected = _load_retarget_pkl(corrected_path)
    metadata = corrected.get("contact_retarget")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{corrected_path} does not contain contact_retarget metadata")
    source_path = Path(args.source_pkl or metadata["source_pkl"]).resolve()
    source = _load_retarget_pkl(source_path)
    data_idx = args.data_idx or metadata["data_idx"]
    side = args.side or metadata["side"]
    dexhand_name = args.dexhand or metadata["dexhand"]
    mesh_path = Path(args.mesh or metadata["tool_mesh"])
    collision_urdf_path = Path(args.collision_urdf or metadata["collision_urdf"])
    context = load_oakink_context(
        data_idx,
        side,
        Path(args.data_root),
        mesh_path,
        collision_urdf_path,
        skip=args.skip,
    )
    hand = _resolve_dexhand(dexhand_name, side)
    model = URDFCollisionKinematics(Path(hand.urdf_path), hand.dof_names, args.points_per_link)
    tool = load_semantic_tool(mesh_path)
    contact_names = list(hand.contact_body_names)
    contact_ids = _contact_link_ids(model, contact_names)
    stored_settings = metadata.get("settings")
    settings = (
        GeometrySettings(**stored_settings)
        if isinstance(stored_settings, Mapping)
        else GeometrySettings(
            contact_distance=args.contact_distance,
            black_clearance=args.black_clearance,
            influence_distance=args.influence_distance,
            min_red_link_ratio=args.min_red_link_ratio,
            min_red_frame_ratio=args.min_red_frame_ratio,
            max_black_violation_fraction=args.max_black_violation_fraction,
        )
    )
    if len(context.object_trajectory) != len(corrected["opt_wrist_pos"]):
        raise ValueError("Corrected PKL and object trajectory lengths differ")
    world_corrected = model.world_surface_trajectory(
        np.asarray(corrected["opt_wrist_pos"], dtype=np.float64),
        np.asarray(corrected["opt_wrist_rot"], dtype=np.float64),
        np.asarray(corrected["opt_dof_pos"], dtype=np.float64),
    )
    points_corrected = world_to_tool_points(world_corrected, context.object_trajectory)
    start_frame, end_frame = (int(v) for v in metadata["grasp_interval"])
    frame_ids = np.arange(start_frame, end_frame + 1, dtype=np.int64)
    metrics = contact_geometry_metrics(
        points_corrected[frame_ids], model.point_link_ids_array, contact_ids, tool, settings
    )
    delta_tool = np.asarray(metadata["delta_tool"], dtype=np.float64)
    alpha = np.asarray(metadata["alpha"], dtype=np.float64)
    delta_world = tool_to_world_offsets(delta_tool, context.object_trajectory) * alpha[:, None]
    invariants = _invariant_report(source, corrected, delta_world)
    collision_bundle_matches = bool(
        _urdf_bundle_sha256(collision_urdf_path)
        == metadata.get("collision_bundle_sha256")
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "passed": bool(
            _is_feasible(metrics, settings)
            and invariants["passed"]
            and collision_bundle_matches
        ),
        "corrected_pkl": corrected_path,
        "source_pkl": source_path,
        "tool_mesh": mesh_path,
        "collision_urdf": collision_urdf_path,
        "collision_bundle_matches": collision_bundle_matches,
        "geometry": _summary_metrics(metrics),
        "invariants": invariants,
        "physics_validation_required": True,
    }
    report_path = (
        Path(args.report)
        if args.report
        else sequence_artifact_path(corrected_path, ".verify_report.json")
    )
    _atomic_json_dump(report, report_path)
    return report


def _add_common_geometry_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", default=DEFAULT_OAKINK_ROOT)
    parser.add_argument("--skip", type=int, default=2)
    parser.add_argument("--points-per-link", type=int, default=32)
    parser.add_argument("--contact-distance", type=float, default=0.006)
    parser.add_argument("--black-clearance", type=float, default=0.006)
    parser.add_argument("--influence-distance", type=float, default=0.02)
    parser.add_argument("--min-red-link-ratio", type=float, default=0.6)
    parser.add_argument("--min-red-frame-ratio", type=float, default=0.8)
    parser.add_argument("--max-black-violation-fraction", type=float, default=0.0)
    parser.add_argument("--max-optimization-frames", type=int, default=48)
    parser.add_argument("--report")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rigidly translate a retargeted dex hand into a colored tool's red contact region"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    correct = subparsers.add_parser("correct", help="solve, validate, and save a corrected retarget PKL")
    correct.add_argument("--data-idx", required=True, help="OakInk2 index, e.g. 083f7@0")
    correct.add_argument("--dexhand", default="inspire")
    correct.add_argument("--side", choices=("right", "left"), default="right")
    correct.add_argument("--input-pkl")
    correct.add_argument("--output-root", default=str(PROJECT_ROOT / "data" / "retargeting_contact"))
    correct.add_argument("--output-pkl")
    correct.add_argument("--overwrite", action="store_true")
    correct.add_argument("--mesh", help="colored new tool mesh; defaults to the dataset-selected tool")
    correct.add_argument(
        "--collision-urdf",
        help="new tool collision URDF; defaults to the dataset-selected coacd URDF",
    )
    correct.add_argument("--contact-links", help="comma-separated URDF links; defaults to hand contact_body_names")
    correct.add_argument("--start-frame", type=int)
    correct.add_argument("--end-frame", type=int)
    correct.add_argument("--ramp-in-frames", type=int, default=20)
    correct.add_argument("--ramp-out-frames", type=int, default=20)
    correct.add_argument("--max-grasp-gap", type=int, default=10)
    correct.add_argument("--min-grasp-frames", type=int, default=8)
    correct.add_argument("--grasp-distance", type=float, default=0.025)
    correct.add_argument("--max-shift", type=float, default=0.05)
    correct.add_argument("--optimizer-iterations", type=int, default=30)
    correct.add_argument("--max-refinement-rounds", type=int, default=4)
    correct.add_argument("--seed", type=int, default=42)
    correct.add_argument("--preview")
    correct.add_argument(
        "--allow-infeasible",
        action="store_true",
        help="write a diagnostic PKL even when constraints fail; never use it for Stage 1 without review",
    )
    _add_common_geometry_arguments(correct)

    verify = subparsers.add_parser("verify", help="independently revalidate a corrected PKL")
    verify.add_argument("--input-pkl", required=True)
    verify.add_argument("--source-pkl")
    verify.add_argument("--data-idx")
    verify.add_argument("--dexhand")
    verify.add_argument("--side", choices=("right", "left"))
    verify.add_argument("--mesh")
    verify.add_argument("--collision-urdf")
    _add_common_geometry_arguments(verify)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "correct":
        report = run_correction(args)
    else:
        report = run_verification(args)
    print(json.dumps(_jsonable(report), indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
