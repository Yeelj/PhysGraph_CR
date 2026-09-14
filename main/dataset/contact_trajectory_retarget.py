"""Contact-aware geometric post-retargeting for dexterous-hand trajectories.

This module deliberately runs before Stage 1.  It keeps wrist orientation and
all finger DOFs unchanged, maps the old red-region center to the new one, and
locally refines only a tool-local translation.  Corrected right-hand PKLs are
stored by data id and replacement-tool name below ``data/retargeting_contact``
and remain compatible with ``load_retargeted_data``.

The optimizer itself has no Isaac Gym dependency.  Hand collision surfaces and
forward kinematics are reconstructed directly from the dex-hand URDF.  After
the semantic geometry check passes, an optional isolated subprocess loads
Isaac Gym/PhysX for collision-response validation and visualization.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import trimesh
from scipy.optimize import differential_evolution, minimize, minimize_scalar
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


SCHEMA_VERSION = 9


def _physics_validation_required(metadata: Mapping[str, object]) -> bool:
    """Return the configured PhysX gate, preserving strict legacy behavior."""

    if "physics_validation_required" in metadata:
        return bool(metadata["physics_validation_required"])
    return int(metadata.get("schema_version", 0)) >= 3


def _stored_validation_tiers_passed(
    metadata: Mapping[str, object],
) -> bool:
    if int(metadata.get("schema_version", 0)) < 3:
        return True
    tiers = metadata.get("validation_tiers")
    if not isinstance(tiers, Mapping):
        return False
    required_names = ["semantic_geometry"]
    if int(metadata.get("schema_version", 0)) >= 9:
        required_names.append("collision_mesh_penetration")
    if _physics_validation_required(metadata):
        required_names.append("isaac_gym_physics")
        if int(metadata.get("schema_version", 0)) >= 8:
            required_names.extend(("physical_optimization", "mano_target_sync"))
    return all(
        isinstance(tiers.get(name), Mapping)
        and bool(tiers[name].get("passed", False))
        for name in required_names
    )


DEFAULT_OAKINK_ROOT = "/home/ps/data/ylj/PhysGraph/data/OakInk-v2"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main.new_object_paths import resolve_new_object_assets
from main.contact_retarget_memory import (
    bank_sequence_dir,
    build_mesh_descriptor,
    corrected_retarget_path,
    load_ranked_entries,
    require_compatible_sequence,
    stable_mapping_hash,
    write_memory_entry,
)


DEFAULT_BANK_ROOT = PROJECT_ROOT / "data" / "retargeting_bank" / "OakInk-v2"


def _log(message: str) -> None:
    print(f"[contact-retarget] {message}", flush=True)


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


def _primitive_mesh(geometry: ET.Element) -> Optional[trimesh.Trimesh]:
    box = geometry.find("box")
    if box is not None:
        extents = np.fromstring(box.attrib["size"], sep=" ", dtype=np.float64)
        return trimesh.creation.box(extents=extents)
    sphere = geometry.find("sphere")
    if sphere is not None:
        return trimesh.creation.icosphere(
            subdivisions=2, radius=float(sphere.attrib["radius"])
        )
    cylinder = geometry.find("cylinder")
    if cylinder is not None:
        return trimesh.creation.cylinder(
            radius=float(cylinder.attrib["radius"]),
            height=float(cylinder.attrib["length"]),
            sections=24,
        )
    return None


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
        # Some generated object URDFs contain a leading blank line before the
        # XML declaration. ElementTree rejects that layout unless it is stripped.
        root = ET.fromstring(self.urdf_path.read_text(encoding="utf-8").lstrip())
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
        self.parent_joint_by_child: Dict[str, URDFJoint] = {}
        for joint in self.joints:
            self.children.setdefault(joint.parent, []).append(joint)
            self.parent_joint_by_child[joint.child] = joint

        collision_points: Dict[str, List[np.ndarray]] = {}
        collision_meshes: Dict[str, List[trimesh.Trimesh]] = {}
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
                else:
                    mesh = _primitive_mesh(geometry)
                if mesh is not None and len(mesh.vertices):
                    mesh = mesh.copy()
                    mesh.apply_transform(_origin_matrix(collision.find("origin")))
                    points = _deterministic_mesh_samples(mesh, per_collision)
                    collision_points.setdefault(link_name, []).append(points)
                    collision_meshes.setdefault(link_name, []).append(mesh)
        self.link_collision_meshes = collision_meshes
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

    def actuated_ancestor_dof_indices(self, link_name: str) -> np.ndarray:
        """Return only stored DOFs that can kinematically move ``link_name``."""

        indices = []
        current = str(link_name)
        visited = set()
        while current != self.root_link:
            if current in visited or current not in self.parent_joint_by_child:
                break
            visited.add(current)
            joint = self.parent_joint_by_child[current]
            controlling_name = (
                joint.name
                if joint.name in self.dof_names
                else joint.mimic_joint
            )
            if controlling_name in self.dof_names:
                indices.append(self.dof_names.index(controlling_name))
            current = joint.parent
        return np.asarray(sorted(set(indices)), dtype=np.int64)

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

    def world_link_origins_trajectory(
        self,
        link_names: Sequence[str],
        wrist_pos: np.ndarray,
        wrist_rotvec: np.ndarray,
        dof: np.ndarray,
    ) -> np.ndarray:
        """Return URDF link origins in the exact order requested by Stage 1."""

        wrist_pos = np.asarray(wrist_pos, dtype=np.float64)
        wrist_rotations = Rotation.from_rotvec(
            np.asarray(wrist_rotvec, dtype=np.float64)
        ).as_matrix()
        dof = np.asarray(dof, dtype=np.float64)
        frames = []
        for frame in range(len(wrist_pos)):
            transforms = self.link_transforms(dof[frame])
            missing = [name for name in link_names if name not in transforms]
            if missing:
                raise ValueError(f"Stage-1 body links are absent from the hand URDF: {missing}")
            local = np.stack([transforms[name][:3, 3] for name in link_names])
            frames.append(local @ wrist_rotations[frame].T + wrist_pos[frame])
        return np.stack(frames)


def _collision_sdf_proxy(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, Dict[str, object]]:
    """Return a closed mesh suitable for a signed query and describe any proxying."""

    original = mesh.copy()
    original_watertight = bool(original.is_watertight)
    original_winding = bool(original.is_winding_consistent)
    use_original = bool(
        original_watertight
        and original_winding
        and np.isfinite(original.volume)
        and abs(float(original.volume)) > 1e-12
    )
    if use_original:
        proxy = original
        proxy_kind = "original_collision_mesh"
    else:
        proxy = original.convex_hull
        proxy_kind = "convex_hull_for_non_watertight_collision_mesh"
    proxy.fix_normals()
    if not proxy.is_watertight:
        raise ValueError("Could not construct a watertight collision SDF proxy")
    return proxy, {
        "proxy_kind": proxy_kind,
        "original_watertight": original_watertight,
        "original_winding_consistent": original_winding,
        "proxy_watertight": bool(proxy.is_watertight),
        "original_vertex_count": int(len(original.vertices)),
        "proxy_vertex_count": int(len(proxy.vertices)),
    }


def _union_signed_distance(
    meshes: Sequence[trimesh.Trimesh], points: np.ndarray
) -> np.ndarray:
    """Signed distance to a mesh union using trimesh's positive-inside convention."""

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not len(meshes):
        raise ValueError("Signed-distance query requires at least one collision mesh")
    values = []
    chunk_size = 100_000
    for mesh in meshes:
        chunks = []
        for start in range(0, len(points), chunk_size):
            chunks.append(
                trimesh.proximity.signed_distance(
                    mesh, points[start : start + chunk_size]
                )
            )
        values.append(np.concatenate(chunks) if chunks else np.empty(0))
    # Each component is positive inside and negative outside. The union is
    # therefore represented by the maximum component signed distance.
    return np.max(np.stack(values, axis=0), axis=0)


def _union_signed_distance_gradient(
    meshes: Sequence[trimesh.Trimesh], points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return union SDF, its query-point gradient, and closest surface points.

    Trimesh uses positive values inside. Away from medial-axis ambiguities, the
    gradient is the signed point-to-surface direction. Physical correction only
    consumes positive-penetration rows, for which this points farther inside the
    queried volume; the caller chooses the corresponding hand escape direction.
    """

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not len(meshes):
        raise ValueError("Signed-distance gradient requires collision geometry")
    component_values = np.stack(
        [_union_signed_distance((mesh,), points) for mesh in meshes], axis=0
    )
    component_ids = np.argmax(component_values, axis=0)
    signed = component_values[component_ids, np.arange(len(points))]
    gradients = np.zeros_like(points)
    closest_points = np.zeros_like(points)
    for component_id, mesh in enumerate(meshes):
        rows = np.flatnonzero(component_ids == component_id)
        if not len(rows):
            continue
        closest, distance, _ = trimesh.proximity.closest_point(mesh, points[rows])
        closest_points[rows] = closest
        delta = points[rows] - closest
        valid = np.asarray(distance, dtype=np.float64) > 1e-12
        if valid.any():
            # signed = +distance inside and -distance outside.
            gradients[rows[valid]] = (
                np.sign(signed[rows[valid]])[:, None]
                * delta[valid]
                / np.asarray(distance, dtype=np.float64)[valid, None]
            )
    return signed, gradients, closest_points


class CollisionMeshSDFValidator:
    """Signed penetration and gradient evaluator for URDF collision geometry."""

    def __init__(
        self,
        hand_kinematics: URDFCollisionKinematics,
        object_urdf: Path,
        *,
        surface_samples_per_link: int = 64,
        penetration_points_per_link: int = 2,
        maximum_points_per_frame: int = 16,
        cluster_radius: float = 0.0015,
        cluster_normal_degrees: float = 20.0,
    ):
        self.hand = hand_kinematics
        self.surface_samples_per_link = max(8, int(surface_samples_per_link))
        self.penetration_points_per_link = max(
            1, int(penetration_points_per_link)
        )
        self.maximum_points_per_frame = max(1, int(maximum_points_per_frame))
        self.cluster_radius = max(0.0, float(cluster_radius))
        self.cluster_normal_cosine = float(
            np.cos(np.deg2rad(float(cluster_normal_degrees)))
        )
        object_dofs = _urdf_actuated_joint_names(Path(object_urdf))
        if object_dofs:
            raise ValueError(
                "Read-only collision SDF currently requires a rigid object URDF; "
                f"found movable joints {object_dofs}"
            )
        self.object = URDFCollisionKinematics(
            Path(object_urdf), (), points_per_link=self.surface_samples_per_link
        )
        object_transforms = self.object.link_transforms(np.empty(0, dtype=np.float64))

        self.hand_sdf_meshes: Dict[str, List[trimesh.Trimesh]] = {}
        self.hand_surface_points: Dict[str, np.ndarray] = {}
        self.object_sdf_meshes: List[trimesh.Trimesh] = []
        object_surface_parts = []
        component_reports = []

        for link_name, components in self.hand.link_collision_meshes.items():
            proxies = []
            samples = []
            per_component = max(
                8, int(np.ceil(self.surface_samples_per_link / len(components)))
            )
            for component_index, component in enumerate(components):
                proxy, proxy_report = _collision_sdf_proxy(component)
                proxies.append(proxy)
                samples.append(_deterministic_mesh_samples(proxy, per_component))
                component_reports.append(
                    {
                        "owner": "hand",
                        "link_name": link_name,
                        "component_index": component_index,
                        **proxy_report,
                    }
                )
            self.hand_sdf_meshes[link_name] = proxies
            self.hand_surface_points[link_name] = np.concatenate(samples)[
                : self.surface_samples_per_link
            ]

        for link_name, components in self.object.link_collision_meshes.items():
            link_transform = object_transforms[link_name]
            per_component = max(
                8, int(np.ceil(self.surface_samples_per_link / len(components)))
            )
            for component_index, component in enumerate(components):
                root_component = component.copy()
                root_component.apply_transform(link_transform)
                proxy, proxy_report = _collision_sdf_proxy(root_component)
                self.object_sdf_meshes.append(proxy)
                object_surface_parts.append(
                    _deterministic_mesh_samples(proxy, per_component)
                )
                component_reports.append(
                    {
                        "owner": "object",
                        "link_name": link_name,
                        "component_index": component_index,
                        **proxy_report,
                    }
                )
        if not self.hand_sdf_meshes or not self.object_sdf_meshes:
            raise ValueError("Collision SDF requires both hand and object collision meshes")
        self.object_surface_points = np.concatenate(object_surface_parts)
        self.component_reports = component_reports

        digest = hashlib.sha256()
        for link_name in sorted(self.hand_sdf_meshes):
            digest.update(link_name.encode("utf-8"))
            for mesh in self.hand_sdf_meshes[link_name]:
                digest.update(np.asarray(mesh.vertices, dtype=np.float64).tobytes())
                digest.update(np.asarray(mesh.faces, dtype=np.int64).tobytes())
        self.hand_proxy_geometry_sha256 = digest.hexdigest()

    def write_physx_aligned_hand_urdf(
        self, output_directory: Path
    ) -> Tuple[Path, Dict[str, object]]:
        """Materialize the exact closed hand proxies used by SDF for PhysX.

        Collision origins are baked into the exported meshes and reset to the
        identity in the derived URDF. Visual mesh references are made absolute
        so relocating the URDF into an isolated validation directory does not
        change Viewer behavior.
        """

        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        root = ET.fromstring(
            self.hand.urdf_path.read_text(encoding="utf-8").lstrip()
        )
        exported = []
        for link in root.findall("link"):
            link_name = link.attrib["name"]
            proxies = self.hand_sdf_meshes.get(link_name, ())
            proxy_index = 0
            safe_link = "".join(
                character if character.isalnum() else "_"
                for character in link_name
            )
            for collision in link.findall("collision"):
                geometry = collision.find("geometry")
                if geometry is None or not any(
                    geometry.find(kind) is not None
                    for kind in ("mesh", "box", "sphere", "cylinder")
                ):
                    continue
                if proxy_index >= len(proxies):
                    raise ValueError(
                        f"Collision proxy mapping is incomplete for hand link {link_name!r}"
                    )
                proxy_path = output_directory / (
                    f"collision_{safe_link}_{proxy_index}.obj"
                )
                proxies[proxy_index].export(proxy_path)
                geometry.clear()
                ET.SubElement(
                    geometry,
                    "mesh",
                    {"filename": proxy_path.name, "scale": "1 1 1"},
                )
                origin = collision.find("origin")
                if origin is None:
                    origin = ET.SubElement(collision, "origin")
                origin.attrib.clear()
                origin.attrib.update({"xyz": "0 0 0", "rpy": "0 0 0"})
                exported.append(
                    {
                        "link_name": link_name,
                        "component_index": proxy_index,
                        "mesh_file": proxy_path.name,
                    }
                )
                proxy_index += 1
            if proxy_index != len(proxies):
                raise ValueError(
                    f"Collision proxy mapping count mismatch for hand link {link_name!r}: "
                    f"URDF={proxy_index}, SDF={len(proxies)}"
                )

        for visual_mesh in root.findall(".//visual/geometry/mesh"):
            visual_mesh.attrib["filename"] = str(
                self.hand._resolve_mesh_path(visual_mesh.attrib["filename"])
            )
        derived_urdf = output_directory / "hand_sdf_physx_aligned.urdf"
        ET.ElementTree(root).write(
            derived_urdf, encoding="utf-8", xml_declaration=True
        )
        return derived_urdf, {
            "mode": "shared_watertight_sdf_proxy_urdf",
            "component_count": len(exported),
            "geometry_sha256": self.hand_proxy_geometry_sha256,
            "components": exported,
        }

    def validate(
        self,
        corrected: Mapping,
        object_trajectory: np.ndarray,
        frame_ids: np.ndarray,
        *,
        maximum_penetration: float,
        collect_constraints: bool = False,
    ) -> Dict[str, object]:
        frame_ids = np.asarray(frame_ids, dtype=np.int64)
        wrist_pos = np.asarray(corrected["opt_wrist_pos"], dtype=np.float64)[frame_ids]
        wrist_rot = Rotation.from_rotvec(
            np.asarray(corrected["opt_wrist_rot"], dtype=np.float64)[frame_ids]
        ).as_matrix()
        dof = np.asarray(corrected["opt_dof_pos"], dtype=np.float64)[frame_ids]
        object_pose = np.asarray(object_trajectory, dtype=np.float64)[frame_ids]
        object_rot = object_pose[:, :3, :3]
        object_pos = object_pose[:, :3, 3]
        frame_count = len(frame_ids)
        depth_frame = np.zeros(frame_count, dtype=np.float64)
        direction_frame = np.full(frame_count, "none", dtype=object)
        link_frame = np.full(frame_count, "", dtype=object)
        # Zero is a serialization-safe placeholder when a frame has no positive
        # penetration. ``penetration_depth_frame`` determines whether the point
        # is meaningful and whether the Viewer draws it.
        point_world_frame = np.zeros((frame_count, 3), dtype=np.float64)
        closest_world_frame = np.zeros((frame_count, 3), dtype=np.float64)
        sdf_gradient_world_frame = np.zeros((frame_count, 3), dtype=np.float64)
        hand_escape_world_frame = np.zeros((frame_count, 3), dtype=np.float64)
        hand_contact_link_frame = np.zeros((frame_count, 3), dtype=np.float64)
        gradient_valid_frame = np.zeros(frame_count, dtype=bool)
        raw_constraints_frame: List[List[Dict[str, object]]] = [
            [] for _ in range(frame_count)
        ]
        positive_tolerance = 1e-7
        constraint_points_per_link = (
            self.penetration_points_per_link if collect_constraints else 0
        )

        transforms_by_link = {name: [] for name in self.hand_sdf_meshes}
        for frame_dof in dof:
            transforms = self.hand.link_transforms(frame_dof)
            for link_name in transforms_by_link:
                transforms_by_link[link_name].append(transforms[link_name])
        transforms_by_link = {
            name: np.stack(values) for name, values in transforms_by_link.items()
        }

        # Direction A: hand collision surfaces inside the object collision volume.
        for link_name, local_points in self.hand_surface_points.items():
            link_transform = transforms_by_link[link_name]
            root_points = np.einsum(
                "fij,pj->fpi", link_transform[:, :3, :3], local_points
            ) + link_transform[:, None, :3, 3]
            world_points = np.einsum("fij,fpj->fpi", wrist_rot, root_points) + wrist_pos[:, None]
            object_points = np.einsum(
                "fji,fpj->fpi", object_rot, world_points - object_pos[:, None]
            )
            signed = _union_signed_distance(
                self.object_sdf_meshes, object_points.reshape(-1, 3)
            ).reshape(frame_count, -1)
            candidate_rows = []
            candidate_samples = []
            for frame_local in range(frame_count):
                positive = np.flatnonzero(
                    signed[frame_local] > positive_tolerance
                )
                if not len(positive):
                    continue
                ordered = positive[
                    np.argsort(signed[frame_local, positive])[::-1]
                ][:constraint_points_per_link]
                candidate_rows.extend([frame_local] * len(ordered))
                candidate_samples.extend(ordered.tolist())
            if candidate_rows:
                candidate_rows_array = np.asarray(candidate_rows, dtype=np.int64)
                candidate_samples_array = np.asarray(
                    candidate_samples, dtype=np.int64
                )
                query_object = object_points[
                    candidate_rows_array, candidate_samples_array
                ]
                candidate_depth, candidate_gradient_object, candidate_closest_object = (
                    _union_signed_distance_gradient(
                        self.object_sdf_meshes, query_object
                    )
                )
                candidate_gradient_world = np.einsum(
                    "fij,fj->fi",
                    object_rot[candidate_rows_array],
                    candidate_gradient_object,
                )
                candidate_closest_world = np.einsum(
                    "fij,fj->fi",
                    object_rot[candidate_rows_array],
                    candidate_closest_object,
                ) + object_pos[candidate_rows_array]
                for candidate_index, frame_local in enumerate(candidate_rows):
                    gradient_world = candidate_gradient_world[candidate_index]
                    if np.linalg.norm(gradient_world) <= 0.5:
                        continue
                    sample_index = candidate_samples[candidate_index]
                    raw_constraints_frame[frame_local].append(
                        {
                            "link_name": link_name,
                            "direction": "hand_surface_inside_object",
                            "depth": float(candidate_depth[candidate_index]),
                            "point_world": world_points[
                                frame_local, sample_index
                            ],
                            "closest_surface_point_world": candidate_closest_world[
                                candidate_index
                            ],
                            "sdf_gradient_world": gradient_world,
                            "hand_escape_direction_world": -gradient_world,
                            "hand_contact_point_link": local_points[sample_index],
                        }
                    )
            local_index = np.argmax(signed, axis=1)
            local_depth = signed[np.arange(frame_count), local_index]
            update = local_depth > depth_frame
            if update.any():
                update_rows = np.flatnonzero(update)
                sample_ids = local_index[update]
                query_object = object_points[update_rows, sample_ids]
                _, gradient_object, closest_object = _union_signed_distance_gradient(
                    self.object_sdf_meshes, query_object
                )
                gradient_world = np.einsum(
                    "fij,fj->fi", object_rot[update], gradient_object
                )
                depth_frame[update] = local_depth[update]
                direction_frame[update] = "hand_surface_inside_object"
                link_frame[update] = link_name
                point_world_frame[update] = world_points[update_rows, sample_ids]
                closest_world_frame[update] = np.einsum(
                    "fij,fj->fi", object_rot[update], closest_object
                ) + object_pos[update]
                sdf_gradient_world_frame[update] = gradient_world
                hand_escape_world_frame[update] = -gradient_world
                hand_contact_link_frame[update] = local_points[sample_ids]
                gradient_valid_frame[update] = (
                    np.linalg.norm(gradient_world, axis=1) > 0.5
                )

        # Direction B: object collision surfaces inside individual hand links.
        object_world_points = np.einsum(
            "fij,pj->fpi", object_rot, self.object_surface_points
        ) + object_pos[:, None]
        for link_name, link_meshes in self.hand_sdf_meshes.items():
            link_transform = transforms_by_link[link_name]
            link_world_rot = np.einsum(
                "fij,fjk->fik", wrist_rot, link_transform[:, :3, :3]
            )
            link_world_pos = (
                np.einsum("fij,fj->fi", wrist_rot, link_transform[:, :3, 3])
                + wrist_pos
            )
            link_points = np.einsum(
                "fji,fpj->fpi",
                link_world_rot,
                object_world_points - link_world_pos[:, None],
            )
            signed = _union_signed_distance(
                link_meshes, link_points.reshape(-1, 3)
            ).reshape(frame_count, -1)
            candidate_rows = []
            candidate_samples = []
            for frame_local in range(frame_count):
                positive = np.flatnonzero(
                    signed[frame_local] > positive_tolerance
                )
                if not len(positive):
                    continue
                ordered = positive[
                    np.argsort(signed[frame_local, positive])[::-1]
                ][:constraint_points_per_link]
                candidate_rows.extend([frame_local] * len(ordered))
                candidate_samples.extend(ordered.tolist())
            if candidate_rows:
                candidate_rows_array = np.asarray(candidate_rows, dtype=np.int64)
                candidate_samples_array = np.asarray(
                    candidate_samples, dtype=np.int64
                )
                query_link = link_points[
                    candidate_rows_array, candidate_samples_array
                ]
                candidate_depth, candidate_gradient_link, candidate_closest_link = (
                    _union_signed_distance_gradient(link_meshes, query_link)
                )
                candidate_gradient_world = np.einsum(
                    "fij,fj->fi",
                    link_world_rot[candidate_rows_array],
                    candidate_gradient_link,
                )
                candidate_closest_world = np.einsum(
                    "fij,fj->fi",
                    link_world_rot[candidate_rows_array],
                    candidate_closest_link,
                ) + link_world_pos[candidate_rows_array]
                for candidate_index, frame_local in enumerate(candidate_rows):
                    gradient_world = candidate_gradient_world[candidate_index]
                    if np.linalg.norm(gradient_world) <= 0.5:
                        continue
                    sample_index = candidate_samples[candidate_index]
                    raw_constraints_frame[frame_local].append(
                        {
                            "link_name": link_name,
                            "direction": "object_surface_inside_hand",
                            "depth": float(candidate_depth[candidate_index]),
                            "point_world": object_world_points[
                                frame_local, sample_index
                            ],
                            "closest_surface_point_world": candidate_closest_world[
                                candidate_index
                            ],
                            "sdf_gradient_world": gradient_world,
                            "hand_escape_direction_world": gradient_world,
                            "hand_contact_point_link": candidate_closest_link[
                                candidate_index
                            ],
                        }
                    )
            local_index = np.argmax(signed, axis=1)
            local_depth = signed[np.arange(frame_count), local_index]
            update = local_depth > depth_frame
            if update.any():
                update_rows = np.flatnonzero(update)
                sample_ids = local_index[update]
                query_link = link_points[update_rows, sample_ids]
                _, gradient_link, closest_link = _union_signed_distance_gradient(
                    link_meshes, query_link
                )
                gradient_world = np.einsum(
                    "fij,fj->fi", link_world_rot[update], gradient_link
                )
                depth_frame[update] = local_depth[update]
                direction_frame[update] = "object_surface_inside_hand"
                link_frame[update] = link_name
                point_world_frame[update] = object_world_points[update_rows, sample_ids]
                closest_world_frame[update] = np.einsum(
                    "fij,fj->fi", link_world_rot[update], closest_link
                ) + link_world_pos[update]
                sdf_gradient_world_frame[update] = gradient_world
                # The query point is fixed on the object. Translating the hand
                # along +grad moves its SDF volume away from that point.
                hand_escape_world_frame[update] = gradient_world
                hand_contact_link_frame[update] = closest_link
                gradient_valid_frame[update] = (
                    np.linalg.norm(gradient_world, axis=1) > 0.5
                )

        constraints_frame = []
        for raw_constraints in raw_constraints_frame:
            selected = []
            for constraint in sorted(
                raw_constraints,
                key=lambda item: float(item["depth"]),
                reverse=True,
            ):
                point = np.asarray(constraint["point_world"], dtype=np.float64)
                normal = np.asarray(
                    constraint["hand_escape_direction_world"], dtype=np.float64
                )
                normal /= max(float(np.linalg.norm(normal)), 1e-12)
                duplicate = False
                for existing in selected:
                    if existing["link_name"] != constraint["link_name"]:
                        continue
                    existing_point = np.asarray(
                        existing["point_world"], dtype=np.float64
                    )
                    existing_normal = np.asarray(
                        existing["hand_escape_direction_world"], dtype=np.float64
                    )
                    existing_normal /= max(
                        float(np.linalg.norm(existing_normal)), 1e-12
                    )
                    if (
                        np.linalg.norm(point - existing_point)
                        <= self.cluster_radius
                        and float(normal @ existing_normal)
                        >= self.cluster_normal_cosine
                    ):
                        duplicate = True
                        break
                if not duplicate:
                    selected.append(constraint)
                if len(selected) >= self.maximum_points_per_frame:
                    break
            constraints_frame.append(selected)

        depth_frame = np.maximum(depth_frame, 0.0)
        finite = bool(np.isfinite(depth_frame).all())
        maximum_depth = float(depth_frame.max(initial=0.0))
        worst_local = int(np.argmax(depth_frame)) if frame_count else 0
        penetrating = depth_frame > positive_tolerance
        violating = depth_frame > float(maximum_penetration)
        proxy_count = sum(
            item["proxy_kind"] != "original_collision_mesh"
            for item in self.component_reports
        )
        passed = bool(finite and not violating.any())
        return {
            "name": "collision_mesh_penetration",
            "status": "passed" if passed else "failed",
            "passed": passed,
            "required": True,
            "read_only": False,
            "affects_correction": True,
            "correction_role": "sdf_point_and_gradient_source_when_physx_enabled",
            "affects_overall_acceptance": True,
            "method": "bidirectional_triangle_signed_distance_with_reported_proxies",
            "sign_convention": "positive_inside_negative_outside",
            "maximum_penetration": maximum_depth,
            "maximum_penetration_threshold": float(maximum_penetration),
            "penetrating_frame_count": int(penetrating.sum()),
            "penetrating_frame_ratio": float(penetrating.mean()) if frame_count else 0.0,
            "threshold_exceeding_frame_count": int(violating.sum()),
            "threshold_exceeding_frame_ratio": float(violating.mean()) if frame_count else 0.0,
            "worst_frame": int(frame_ids[worst_local]) if frame_count else None,
            "worst_link": str(link_frame[worst_local]) if frame_count else None,
            "worst_direction": str(direction_frame[worst_local]) if frame_count else None,
            "worst_point_world": (
                point_world_frame[worst_local]
                if frame_count and maximum_depth > positive_tolerance
                else None
            ),
            "worst_closest_surface_point_world": (
                closest_world_frame[worst_local]
                if frame_count and maximum_depth > positive_tolerance
                else None
            ),
            "worst_sdf_gradient_world": (
                sdf_gradient_world_frame[worst_local]
                if frame_count and maximum_depth > positive_tolerance
                else None
            ),
            "worst_hand_escape_direction_world": (
                hand_escape_world_frame[worst_local]
                if frame_count and maximum_depth > positive_tolerance
                else None
            ),
            "worst_hand_contact_point_link": (
                hand_contact_link_frame[worst_local]
                if frame_count and maximum_depth > positive_tolerance
                else None
            ),
            "worst_gradient_valid": bool(
                frame_count and gradient_valid_frame[worst_local]
            ),
            "frame_ids": frame_ids,
            "penetration_depth_frame": depth_frame,
            "worst_link_frame": link_frame.tolist(),
            "worst_direction_frame": direction_frame.tolist(),
            "worst_point_world_frame": point_world_frame,
            "closest_surface_point_world_frame": closest_world_frame,
            "sdf_gradient_world_frame": sdf_gradient_world_frame,
            "hand_escape_direction_world_frame": hand_escape_world_frame,
            "hand_contact_point_link_frame": hand_contact_link_frame,
            "gradient_valid_frame": gradient_valid_frame,
            "penetration_constraints_frame": constraints_frame,
            "penetration_constraint_count_frame": [
                len(constraints) for constraints in constraints_frame
            ],
            "penetration_points_per_link": self.penetration_points_per_link,
            "maximum_points_per_frame": self.maximum_points_per_frame,
            "constraints_collected": bool(collect_constraints),
            "cluster_radius": self.cluster_radius,
            "cluster_normal_degrees": float(
                np.rad2deg(np.arccos(np.clip(self.cluster_normal_cosine, -1.0, 1.0)))
            ),
            "surface_samples_per_link": self.surface_samples_per_link,
            "component_count": len(self.component_reports),
            "convex_hull_proxy_count": int(proxy_count),
            "uses_proxy_geometry": bool(proxy_count),
            "geometry_reliability": "proxy" if proxy_count else "exact",
            "hand_proxy_geometry_sha256": self.hand_proxy_geometry_sha256,
            "physx_alignment_mode": "shared_watertight_sdf_proxy_urdf",
            "component_geometry": self.component_reports,
            "all_finite": finite,
            "scope_note": (
                "Independent collision-mesh penetration gate. During a requested PhysX repair, "
                "its penetration points and gradients drive local finger/palm correction; PhysX "
                "still gives the final contact-force verdict. Failure blocks normal PKL saving "
                "and schema-v9 Stage-1 loading. Non-watertight URDF collision meshes use an "
                "explicitly reported convex-hull proxy for a well-defined signed volume."
            ),
        }


def _collision_mesh_penetration_report(
    validator: Optional[CollisionMeshSDFValidator],
    initialization_error: Optional[Exception],
    corrected: Mapping,
    object_trajectory: np.ndarray,
    frame_ids: np.ndarray,
    maximum_penetration: float,
    *,
    collect_constraints: bool = False,
) -> Dict[str, object]:
    """Convert SDF setup/query errors into an explicit failed validation tier."""

    try:
        if validator is None:
            if initialization_error is None:
                raise RuntimeError("collision SDF validator was not initialized")
            raise initialization_error
        return validator.validate(
            corrected,
            object_trajectory,
            frame_ids,
            maximum_penetration=maximum_penetration,
            collect_constraints=collect_constraints,
        )
    except Exception as error:
        return {
            "name": "collision_mesh_penetration",
            "status": "error",
            "passed": False,
            "required": True,
            "read_only": False,
            "affects_correction": True,
            "correction_role": "required_but_unavailable",
            "affects_overall_acceptance": True,
            "maximum_penetration_threshold": float(maximum_penetration),
            "error": f"{type(error).__name__}: {error}",
        }


def _compact_collision_mesh_penetration_report(
    report: Mapping[str, object],
) -> Dict[str, object]:
    """Keep required Stage-1 gate metadata small while preserving its verdict."""

    per_frame_fields = {
        "frame_ids",
        "penetration_depth_frame",
        "worst_link_frame",
        "worst_direction_frame",
        "worst_point_world_frame",
        "closest_surface_point_world_frame",
        "sdf_gradient_world_frame",
        "hand_escape_direction_world_frame",
        "hand_contact_point_link_frame",
        "gradient_valid_frame",
        "penetration_constraints_frame",
        "penetration_constraint_count_frame",
        "component_geometry",
    }
    return {
        key: value for key, value in report.items() if key not in per_frame_fields
    }


@dataclass
class OakInkContext:
    data_idx: str
    annotation_path: Path
    source_mesh_path: Path
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
    new_obj: Optional[str] = None,
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
    if new_obj is not None:
        if mesh_override is not None or collision_urdf_override is not None:
            raise ValueError("--new-obj cannot be combined with --mesh or --collision-urdf")
        replacement = resolve_new_object_assets(new_obj)
        mesh_path = replacement.mesh_path
        collision_urdf_path = replacement.urdf_path
    else:
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
        source_mesh_path=dataset_mesh_path,
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
    semantic_report: Optional[Dict[str, object]] = None


def validate_semantic_mesh(mesh_path: Path, label: str = "tool") -> Dict[str, object]:
    """Fail fast unless every mesh vertex is explicitly colored red or black."""

    mesh = _as_mesh(trimesh.load(mesh_path, process=False, force="mesh"))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if not len(vertices):
        raise ValueError(f"{label} mesh contains no vertices: {mesh_path}")
    colors = getattr(getattr(mesh, "visual", None), "vertex_colors", None)
    if colors is None or len(colors) != len(vertices):
        raise ValueError(
            f"{label} mesh has not been semantically segmented: expected one red/black color "
            f"per vertex in {mesh_path}"
        )
    colors = np.asarray(colors[:, :3], dtype=np.float64)
    if colors.max(initial=0.0) <= 1.0:
        colors *= 255.0
    red_mask = (colors[:, 0] > 150.0) & (colors[:, 1] < 100.0) & (colors[:, 2] < 100.0)
    black_mask = (colors[:, 0] < 80.0) & (colors[:, 1] < 80.0) & (colors[:, 2] < 80.0)
    unclassified_mask = ~(red_mask | black_mask)
    if not red_mask.any() or not black_mask.any() or unclassified_mask.any():
        raise ValueError(
            f"{label} mesh semantic segmentation is invalid: red={red_mask.sum()}, "
            f"black={black_mask.sum()}, unclassified={unclassified_mask.sum()}, "
            f"vertices={len(vertices)} in {mesh_path}. Every vertex must be red or black."
        )
    return {
        "label": label,
        "path": mesh_path.resolve(),
        "passed": True,
        "vertex_count": int(len(vertices)),
        "red_vertex_count": int(red_mask.sum()),
        "black_vertex_count": int(black_mask.sum()),
        "unclassified_vertex_count": int(unclassified_mask.sum()),
        "red_fraction": float(red_mask.mean()),
        "black_fraction": float(black_mask.mean()),
    }


def load_semantic_tool(mesh_path: Path, label: str = "tool") -> SemanticTool:
    semantic_report = validate_semantic_mesh(mesh_path, label=label)
    mesh = _as_mesh(trimesh.load(mesh_path, process=False, force="mesh"))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    colors = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.float64)
    if colors.max(initial=0.0) <= 1.0:
        colors *= 255.0
    red_mask = (colors[:, 0] > 150.0) & (colors[:, 1] < 100.0) & (colors[:, 2] < 100.0)
    black_mask = (colors[:, 0] < 80.0) & (colors[:, 1] < 80.0) & (colors[:, 2] < 80.0)
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
        semantic_report=semantic_report,
    )


def world_to_tool_points(world_points: np.ndarray, object_trajectory: np.ndarray) -> np.ndarray:
    rotations = object_trajectory[:, :3, :3]
    translations = object_trajectory[:, :3, 3]
    return np.einsum("fji,fpj->fpi", rotations, world_points - translations[:, None])


def tool_to_world_offsets(offset_tool: np.ndarray, object_trajectory: np.ndarray) -> np.ndarray:
    return np.einsum("fij,j->fi", object_trajectory[:, :3, :3], offset_tool)


def apply_tool_transform_points(points_tool: np.ndarray, transforms_tool: np.ndarray) -> np.ndarray:
    """Apply one tool-local rigid transform per frame to point trajectories."""

    points_tool = np.asarray(points_tool, dtype=np.float64)
    transforms_tool = np.asarray(transforms_tool, dtype=np.float64)
    if points_tool.ndim != 3 or points_tool.shape[-1] != 3:
        raise ValueError(f"Expected [frames, points, 3], got {points_tool.shape}")
    if transforms_tool.shape != (len(points_tool), 4, 4):
        raise ValueError(
            f"Expected transform schedule {(len(points_tool), 4, 4)}, got {transforms_tool.shape}"
        )
    return (
        np.einsum("fij,fpj->fpi", transforms_tool[:, :3, :3], points_tool)
        + transforms_tool[:, None, :3, 3]
    )


def build_transform_schedule(
    transform_tool: np.ndarray,
    alpha: np.ndarray,
    pivot_tool: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Smoothly interpolate identity to a fixed tool-local SE(3) transform.

    Rotation follows the shortest rotation-vector interpolation and translation
    follows the existing quintic schedule.  The resulting matrices are stored
    in the corrected PKL so correction, verification, and Stage 1 consume the
    exact same trajectory.
    """

    transform_tool = np.asarray(transform_tool, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    if transform_tool.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 contact transform, got {transform_tool.shape}")
    if alpha.ndim != 1:
        raise ValueError(f"Expected a one-dimensional schedule, got {alpha.shape}")
    rotation = Rotation.from_matrix(transform_tool[:3, :3]).as_rotvec()
    schedule = np.repeat(np.eye(4, dtype=np.float64)[None], len(alpha), axis=0)
    schedule[:, :3, :3] = Rotation.from_rotvec(alpha[:, None] * rotation[None]).as_matrix()
    if pivot_tool is None:
        schedule[:, :3, 3] = alpha[:, None] * transform_tool[:3, 3][None]
    else:
        pivot_tool = np.asarray(pivot_tool, dtype=np.float64)
        if pivot_tool.shape != (3,):
            raise ValueError(f"Expected a three-dimensional SE(3) pivot, got {pivot_tool.shape}")
        final_pivot = transform_tool[:3, :3] @ pivot_tool + transform_tool[:3, 3]
        pivot_path = (
            (1.0 - alpha)[:, None] * pivot_tool[None]
            + alpha[:, None] * final_pivot[None]
        )
        schedule[:, :3, 3] = pivot_path - np.einsum(
            "fij,j->fi", schedule[:, :3, :3], pivot_tool
        )
    return schedule


def transform_hand_trajectory(
    wrist_pos: np.ndarray,
    wrist_rotvec: np.ndarray,
    object_trajectory: np.ndarray,
    transforms_tool: np.ndarray,
    joints_world: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Apply a tool-local SE(3) schedule to wrist pose and optional hand joints."""

    wrist_pos = np.asarray(wrist_pos, dtype=np.float64)
    wrist_rotvec = np.asarray(wrist_rotvec, dtype=np.float64)
    object_trajectory = np.asarray(object_trajectory, dtype=np.float64)
    transforms_tool = np.asarray(transforms_tool, dtype=np.float64)
    length = len(wrist_pos)
    if wrist_rotvec.shape != (length, 3):
        raise ValueError(f"Expected wrist rotations {(length, 3)}, got {wrist_rotvec.shape}")
    if object_trajectory.shape != (length, 4, 4):
        raise ValueError(f"Expected object trajectory {(length, 4, 4)}, got {object_trajectory.shape}")
    if transforms_tool.shape != (length, 4, 4):
        raise ValueError(f"Expected transform schedule {(length, 4, 4)}, got {transforms_tool.shape}")

    object_rotation = object_trajectory[:, :3, :3]
    object_translation = object_trajectory[:, :3, 3]
    wrist_tool = np.einsum(
        "fji,fj->fi", object_rotation, wrist_pos - object_translation
    )
    corrected_wrist_tool = (
        np.einsum("fij,fj->fi", transforms_tool[:, :3, :3], wrist_tool)
        + transforms_tool[:, :3, 3]
    )
    corrected_wrist_pos = (
        np.einsum("fij,fj->fi", object_rotation, corrected_wrist_tool)
        + object_translation
    )

    wrist_rotation = Rotation.from_rotvec(wrist_rotvec).as_matrix()
    corrected_wrist_rotation = (
        object_rotation
        @ transforms_tool[:, :3, :3]
        @ np.swapaxes(object_rotation, 1, 2)
        @ wrist_rotation
    )
    corrected_wrist_rotvec = Rotation.from_matrix(corrected_wrist_rotation).as_rotvec()

    corrected_joints = None
    if joints_world is not None:
        joints_world = np.asarray(joints_world, dtype=np.float64)
        if joints_world.ndim != 3 or joints_world.shape[0] != length or joints_world.shape[2] != 3:
            raise ValueError(f"Expected joint positions [frames, joints, 3], got {joints_world.shape}")
        joints_tool = np.einsum(
            "fji,fpj->fpi",
            object_rotation,
            joints_world - object_translation[:, None],
        )
        corrected_joints_tool = apply_tool_transform_points(joints_tool, transforms_tool)
        corrected_joints = (
            np.einsum("fij,fpj->fpi", object_rotation, corrected_joints_tool)
            + object_translation[:, None]
        )
    return corrected_wrist_pos, corrected_wrist_rotvec, corrected_joints


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
    min_red_link_ratio: float = 0.1 # 接触link的红色点比例阈值
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
    red_distance, black_distance, black_nearest, _, _ = _nearest_semantics(
        points_tool, tool, settings.semantic_tie_epsilon
    )
    surface_distance = np.minimum(red_distance, black_distance)
    red_active_per_link = []
    surface_active_per_link = []
    for link_id in sorted(set(int(value) for value in contact_link_ids.tolist())):
        link_mask = point_link_ids == link_id
        red_active_per_link.append(
            (
                (~black_nearest[:, link_mask])
                & (red_distance[:, link_mask] <= settings.grasp_distance)
            ).any(axis=1)
        )
        surface_active_per_link.append(
            surface_distance[:, link_mask].min(axis=1) <= settings.grasp_distance
        )
    active = (
        np.stack(red_active_per_link, axis=1).mean(axis=1)
        >= settings.min_red_link_ratio
    )
    active = _close_short_gaps(active, max_gap=max_gap)
    interval = _longest_true_interval(active)
    if interval is None or interval[1] - interval[0] + 1 < min_frames:
        # Coarse source meshes can place the semantic samples farther from the
        # collision surface than the red-region threshold. Retain the original
        # surface-proximity detector only as a compatibility fallback.
        active = np.stack(surface_active_per_link, axis=1).mean(axis=1) >= (
            1.0 / max(1, len(surface_active_per_link))
        )
        active = _close_short_gaps(active, max_gap=max_gap)
        interval = _longest_true_interval(active)
    if interval is None or interval[1] - interval[0] + 1 < min_frames:
        raise ValueError(
            "Could not detect a stable grasp interval from the source mesh. "
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
    geometry_loss = _transformed_geometry_objective(
        shifted, point_link_ids, contact_link_ids, tool, settings
    )
    shift_loss = float(np.dot(delta, delta) / max(settings.max_shift**2, 1e-12))
    return float(geometry_loss + 0.02 * shift_loss)


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


def estimate_semantic_axis(tool: SemanticTool, roi_distance: float = 0.03) -> Tuple[np.ndarray, Dict[str, object]]:
    """Estimate a tool-local center axis directed from black toward red.

    Only black surface samples near the red region participate when possible,
    which prevents a large unrelated part of a tool (for example, a spoon
    bowl) from dominating the axis.  PCA supplies a mesh-aligned center axis;
    the red/black centroids determine its sign.
    """

    if roi_distance <= 0.0:
        raise ValueError(f"axis ROI distance must be positive, got {roi_distance}")
    red_points = np.asarray(tool.red_points, dtype=np.float64)
    black_points = np.asarray(tool.black_points, dtype=np.float64)
    if red_points.ndim != 2 or red_points.shape[1] != 3 or len(red_points) == 0:
        raise ValueError("Semantic-axis estimation requires non-empty red 3D points")
    if black_points.ndim != 2 or black_points.shape[1] != 3 or len(black_points) == 0:
        raise ValueError("Semantic-axis estimation requires non-empty black 3D points")

    black_to_red_distance, _ = tool.red_tree.query(black_points, workers=1)
    local_mask = black_to_red_distance <= roi_distance
    fallback_used = not bool(local_mask.any())
    if fallback_used:
        # Always retain a local semantic boundary even when the configured ROI
        # is narrower than the red/black separation in a coarse mesh.
        fallback_count = min(len(black_points), max(3, int(np.ceil(0.1 * len(black_points)))))
        local_indices = np.argsort(black_to_red_distance)[:fallback_count]
        local_black = black_points[local_indices]
    else:
        local_black = black_points[local_mask]

    red_center = np.median(red_points, axis=0)
    black_center = np.median(local_black, axis=0)
    semantic_vector = red_center - black_center
    semantic_norm = float(np.linalg.norm(semantic_vector))
    if semantic_norm <= 1e-10:
        raise ValueError(
            "Cannot orient the semantic axis because the red and local-black centers coincide"
        )
    semantic_direction = semantic_vector / semantic_norm

    # Balance the two semantic classes before PCA so vertex density or region
    # size does not give either color a disproportionate weight.
    red_centered = red_points - red_center
    black_centered = local_black - black_center
    covariance = (
        (red_centered.T @ red_centered) / max(1, len(red_centered))
        + (black_centered.T @ black_centered) / max(1, len(black_centered))
        + np.outer(semantic_vector, semantic_vector)
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    alignments = np.abs(eigenvectors.T @ semantic_direction)
    axis_index = int(np.argmax(alignments))
    axis_direction = eigenvectors[:, axis_index]
    if float(np.dot(axis_direction, semantic_vector)) < 0.0:
        axis_direction = -axis_direction
    axis_direction = axis_direction / np.linalg.norm(axis_direction)
    axis_origin = 0.5 * (red_center + black_center)

    return axis_direction.astype(np.float64), {
        "method": "semantic_pca",
        "direction_tool": axis_direction,
        "origin_tool": axis_origin,
        "red_center_tool": red_center,
        "local_black_center_tool": black_center,
        "semantic_center_distance": semantic_norm,
        "pca_alignment": float(alignments[axis_index]),
        "pca_eigenvalues": eigenvalues,
        "roi_distance": float(roi_distance),
        "local_black_point_count": int(len(local_black)),
        "local_black_fallback_used": fallback_used,
    }


def _axis_constraint_report(delta: np.ndarray, direction: np.ndarray) -> Dict[str, object]:
    delta = np.asarray(delta, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    direction_norm = float(np.linalg.norm(direction))
    if direction.shape != (3,) or direction_norm <= 1e-10:
        raise ValueError(f"Invalid semantic-axis direction: {direction}")
    unit_direction = direction / direction_norm
    signed_distance = float(np.dot(delta, unit_direction))
    perpendicular = delta - signed_distance * unit_direction
    perpendicular_norm = float(np.linalg.norm(perpendicular))
    tolerance = 1e-6
    return {
        "signed_distance": signed_distance,
        "perpendicular_norm": perpendicular_norm,
        "direction_norm": direction_norm,
        "passed": bool(signed_distance >= -tolerance and perpendicular_norm <= tolerance),
    }


def optimize_tool_translation_along_axis(
    points_tool: np.ndarray,
    point_link_ids: np.ndarray,
    contact_link_ids: np.ndarray,
    tool: SemanticTool,
    settings: GeometrySettings,
    direction: np.ndarray,
    grid_samples: int = 201,
    initial_delta: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Optimize a non-negative distance along one fixed tool-local axis."""

    direction = np.asarray(direction, dtype=np.float64)
    direction_norm = float(np.linalg.norm(direction))
    if direction.shape != (3,) or direction_norm <= 1e-10:
        raise ValueError(f"Invalid semantic-axis direction: {direction}")
    if grid_samples < 3:
        raise ValueError(f"axis grid samples must be at least 3, got {grid_samples}")
    direction = direction / direction_norm

    def objective_distance(distance: float) -> float:
        return _optimization_objective(
            float(distance) * direction,
            points_tool,
            point_link_ids,
            contact_link_ids,
            tool,
            settings,
        )

    distances = np.linspace(0.0, settings.max_shift, grid_samples, dtype=np.float64)
    scores = np.asarray([objective_distance(distance) for distance in distances])
    candidates = [(float(distance), float(score)) for distance, score in zip(distances, scores)]
    if initial_delta is not None:
        initial_distance = float(
            np.clip(np.dot(np.asarray(initial_delta, dtype=np.float64), direction), 0.0, settings.max_shift)
        )
        candidates.append((initial_distance, objective_distance(initial_distance)))

    best_grid_index = int(np.argmin(scores))
    lower_index = max(0, best_grid_index - 1)
    upper_index = min(len(distances) - 1, best_grid_index + 1)
    polished = None
    if distances[upper_index] > distances[lower_index]:
        polished = minimize_scalar(
            objective_distance,
            bounds=(float(distances[lower_index]), float(distances[upper_index])),
            method="bounded",
            options={"xatol": 1e-7, "maxiter": 160},
        )
        candidates.append((float(polished.x), float(polished.fun)))

    distance, score = min(candidates, key=lambda item: item[1])
    delta = distance * direction
    return delta.astype(np.float64), {
        "mode": "semantic_axis",
        "objective": float(score),
        "axis_direction_tool": direction,
        "axis_distance": float(distance),
        "axis_grid_samples": int(grid_samples),
        "axis_grid_best_distance": float(distances[best_grid_index]),
        "bounded_success": None if polished is None else bool(polished.success),
        "bounded_message": None if polished is None else str(polished.message),
        "axis_constraint": _axis_constraint_report(delta, direction),
    }


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
        "mode": "free3d",
        "objective": float(score),
        "global_search": bool(global_search),
        "differential_evolution_success": None if result is None else bool(result.success),
        "differential_evolution_message": None if result is None else str(result.message),
        "powell_success": bool(polished.success),
        "powell_message": str(polished.message),
        "red_direction_seed": red_seed,
    }


def contact_translation_transfer(
    source_tool: SemanticTool,
    target_tool: SemanticTool,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Map the robust old red-region center to the new one with translation only."""

    source_center = np.median(np.asarray(source_tool.red_points, dtype=np.float64), axis=0)
    target_center = np.median(np.asarray(target_tool.red_points, dtype=np.float64), axis=0)
    delta = target_center - source_center
    return delta, {
        "method": "red_center_translation",
        "source_red_center_tool": source_center,
        "target_red_center_tool": target_center,
        "initial_delta_tool": delta,
        "initial_delta_norm": float(np.linalg.norm(delta)),
        "rotation_degrees": 0.0,
    }


def extract_source_contact_intent(
    points_tool: np.ndarray,
    point_link_ids: np.ndarray,
    contact_link_ids: np.ndarray,
    source_tool: SemanticTool,
    settings: GeometrySettings,
) -> Dict[str, object]:
    """Summarize which configured links actually approach the old red patch.

    The summary informs and documents contact-region transfer, but acceptance is
    intentionally still performed with the configured fixed contact-link ratio.
    """

    red_distance, _, black_nearest, red_index, _ = _nearest_semantics(
        points_tool, source_tool, settings.semantic_tie_epsilon
    )
    per_link = []
    anchors = []
    for link_id in sorted(set(int(value) for value in np.asarray(contact_link_ids).tolist())):
        link_mask = point_link_ids == link_id
        if not link_mask.any():
            continue
        link_red_distance = red_distance[:, link_mask]
        link_black_nearest = black_nearest[:, link_mask]
        best_local = np.argmin(link_red_distance, axis=1)
        frame_index = np.arange(len(points_tool), dtype=np.int64)
        best_distance = link_red_distance[frame_index, best_local]
        best_is_red = ~link_black_nearest[frame_index, best_local]
        active = best_is_red & (best_distance <= settings.grasp_distance)
        link_red_indices = red_index[:, link_mask][frame_index, best_local]
        if active.any():
            link_anchors = source_tool.red_points[link_red_indices[active]]
            anchor = np.median(link_anchors, axis=0)
            anchors.append(anchor)
        else:
            anchor = None
        per_link.append(
            {
                "link_id": int(link_id),
                "active_frame_ratio": float(active.mean()),
                "minimum_red_distance": float(best_distance.min()),
                "anchor_tool": anchor,
            }
        )
    return {
        "method": "old_red_nearest_collision_surface",
        "configured_link_count": int(len(set(int(v) for v in contact_link_ids.tolist()))),
        "active_link_count": int(sum(item["anchor_tool"] is not None for item in per_link)),
        "contact_anchor_center_tool": None if not anchors else np.median(np.stack(anchors), axis=0),
        "per_link": per_link,
    }


def _transformed_geometry_objective(
    transformed: np.ndarray,
    point_link_ids: np.ndarray,
    contact_link_ids: np.ndarray,
    tool: SemanticTool,
    settings: GeometrySettings,
) -> float:
    red_distance, black_distance, black_nearest, _, _ = _nearest_semantics(
        transformed, tool, settings.semantic_tie_epsilon
    )
    link_distances = []
    for link_id in sorted(set(int(value) for value in np.asarray(contact_link_ids).tolist())):
        link_mask = point_link_ids == link_id
        semantic_red_distance = red_distance[:, link_mask] + np.maximum(
            red_distance[:, link_mask]
            - black_distance[:, link_mask]
            + settings.semantic_tie_epsilon,
            0.0,
        )
        link_distances.append(semantic_red_distance.min(axis=1))
    link_distances = np.stack(link_distances, axis=1)
    required = max(1, int(np.ceil(settings.min_red_link_ratio * link_distances.shape[1])))
    best_required = np.partition(link_distances, required - 1, axis=1)[:, :required]
    red_loss = np.mean(
        np.square(np.clip(best_required / settings.contact_distance, 0.0, 10.0))
    )
    black_depth = np.where(
        black_nearest,
        np.maximum(settings.black_clearance - black_distance, 0.0) / settings.black_clearance,
        0.0,
    )
    black_loss = float(
        np.mean(black_depth**2) + np.mean(np.max(black_depth, axis=1) ** 2)
    )
    return float(red_loss + 80.0 * black_loss)


def optimize_contact_translation(
    points_tool: np.ndarray,
    point_link_ids: np.ndarray,
    contact_link_ids: np.ndarray,
    tool: SemanticTool,
    settings: GeometrySettings,
    initial_delta: np.ndarray,
    max_residual_shift: float = 0.01,
    max_iterations: int = 30,
    initial_residual: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Locally refine red-center translation without rotating the grasp."""

    if max_residual_shift <= 0.0:
        raise ValueError("Contact-translation residual bound must be positive")
    initial_delta = np.asarray(initial_delta, dtype=np.float64)
    if initial_delta.shape != (3,):
        raise ValueError(f"Expected a three-dimensional initial delta, got {initial_delta.shape}")
    bounds = [(-max_residual_shift, max_residual_shift)] * 3

    def objective(residual: np.ndarray) -> float:
        delta = initial_delta + np.asarray(residual, dtype=np.float64)
        transformed = points_tool + delta[None, None]
        geometry_loss = _transformed_geometry_objective(
            transformed, point_link_ids, contact_link_ids, tool, settings
        )
        translation_prior = float(np.dot(residual, residual)) / max(
            max_residual_shift**2, 1e-12
        )
        return float(geometry_loss + 0.05 * translation_prior)

    seed = (
        np.zeros(3, dtype=np.float64)
        if initial_residual is None
        else np.asarray(initial_residual, dtype=np.float64)
    )
    seed = np.asarray(
        [np.clip(value, low, high) for value, (low, high) in zip(seed, bounds)],
        dtype=np.float64,
    )
    candidates = [(seed, objective(seed))]
    if np.linalg.norm(seed) > 0.0:
        zero = np.zeros(3, dtype=np.float64)
        candidates.append((zero, objective(zero)))
    polished = minimize(
        objective,
        seed,
        method="Powell",
        bounds=bounds,
        options={"maxiter": max_iterations, "xtol": 1e-5, "ftol": 1e-6},
    )
    candidates.append((np.asarray(polished.x), float(polished.fun)))
    residual, score = min(candidates, key=lambda item: item[1])
    delta = initial_delta + residual
    return delta, {
        "mode": "contact_translation",
        "objective": float(score),
        "residual": residual,
        "residual_translation_norm": float(np.linalg.norm(residual)),
        "initial_delta_tool": initial_delta,
        "final_delta_tool": delta,
        "final_translation_norm": float(np.linalg.norm(delta)),
        "final_rotation_degrees": 0.0,
        "powell_success": bool(polished.success),
        "powell_message": str(polished.message),
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
    root = ET.fromstring(Path(urdf_path).read_text(encoding="utf-8").lstrip())
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


def _bounded_frame_sample(frame_ids: np.ndarray, maximum: int) -> np.ndarray:
    """Keep refinement cost bounded even when every hold frame fails."""

    frame_ids = np.unique(np.asarray(frame_ids, dtype=np.int64))
    if maximum <= 0:
        raise ValueError(f"Maximum optimization frames must be positive, got {maximum}")
    if len(frame_ids) <= maximum:
        return frame_ids
    indices = np.linspace(0, len(frame_ids) - 1, maximum).round().astype(np.int64)
    return frame_ids[indices]


def _is_feasible(metrics: Mapping[str, object], settings: GeometrySettings) -> bool:
    return bool(
        float(metrics["red_frame_pass_ratio"]) >= settings.min_red_frame_ratio
        and float(metrics["black_violation_fraction"]) <= settings.max_black_violation_fraction
    )


def _invariant_report(
    source: Mapping,
    corrected: Mapping,
    delta_world: Optional[np.ndarray] = None,
    expected_wrist_pos: Optional[np.ndarray] = None,
    expected_wrist_rot: Optional[np.ndarray] = None,
    expected_joints_pos: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Check that correction is exactly the scheduled rigid transform.

    ``delta_world`` preserves the legacy translation-only API used by older
    callers.  SE(3) correction supplies explicit expected pose trajectories.
    Finger DOFs remain a strict invariant in both modes.
    """

    source_rot = np.asarray(source["opt_wrist_rot"])
    corrected_rot = np.asarray(corrected["opt_wrist_rot"])
    source_dof = np.asarray(source["opt_dof_pos"])
    corrected_dof = np.asarray(corrected["opt_dof_pos"])
    source_pos = np.asarray(source["opt_wrist_pos"])
    corrected_pos = np.asarray(corrected["opt_wrist_pos"])
    if expected_wrist_pos is None:
        if delta_world is None:
            raise ValueError("Invariant validation requires delta_world or expected wrist poses")
        expected_wrist_pos = source_pos + np.asarray(delta_world)
    if expected_wrist_rot is None:
        expected_wrist_rot = source_rot
    result = {
        "rotation_max_abs_error": float(
            np.max(np.abs(corrected_rot - np.asarray(expected_wrist_rot)))
        ),
        "dof_max_abs_error": float(np.max(np.abs(corrected_dof - source_dof))),
        "wrist_translation_max_abs_error": float(
            np.max(np.abs(corrected_pos - np.asarray(expected_wrist_pos)))
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
        if expected_joints_pos is None:
            if delta_world is None:
                raise ValueError("SE(3) joint invariant validation requires expected joint positions")
            expected_joints_pos = source_joints + np.asarray(delta_world)[:, None]
        result["joint_rigid_translation_max_abs_error"] = float(
            np.max(np.abs(corrected_joints - np.asarray(expected_joints_pos)))
        )
    result["passed"] = bool(
        result["all_finite"]
        # Corrected trajectories and stored SE(3) schedules use float32. Axis-angle
        # round trips near pi can differ by roughly 1e-7 while representing the
        # same scheduled wrist rotation.
        and result["rotation_max_abs_error"] <= 1e-6
        and result["dof_max_abs_error"] <= 1e-10
        and result["wrist_translation_max_abs_error"] <= 1e-6
        and result.get("joint_rigid_translation_max_abs_error", 0.0) <= 1e-6
    )
    return result


def _evenly_spaced_frame_ids(
    frame_ids: np.ndarray, maximum_frames: int
) -> np.ndarray:
    frame_ids = np.asarray(frame_ids, dtype=np.int64)
    if not len(frame_ids):
        return frame_ids
    sample_count = min(len(frame_ids), max(1, int(maximum_frames)))
    sample_indices = np.linspace(0, len(frame_ids) - 1, sample_count).round().astype(np.int64)
    return np.unique(frame_ids[sample_indices])


def run_isaac_gym_validation(
    corrected: Mapping,
    hand: "OfflineHandSpec",
    context: OakInkContext,
    contact_names: Sequence[str],
    hold_frame_ids: np.ndarray,
    headless: bool,
    maximum_frames: int,
    settle_steps: int,
    min_contact_frame_ratio: float,
    max_contact_force: float,
    max_initial_overlap: float,
    timeout: float,
    collision_sdf_report: Optional[Mapping[str, object]] = None,
    collision_sdf_validator: Optional[CollisionMeshSDFValidator] = None,
) -> Dict[str, object]:
    """Run tier-three PhysX contact replay in an isolated Isaac Gym process."""

    if not headless and not os.environ.get("DISPLAY"):
        result = {
            "name": "isaac_gym_physics",
            "status": "error",
            "passed": False,
            "error": (
                "Viewer mode requires an X11 DISPLAY, but DISPLAY is empty. "
                "Run from a graphical desktop/X-forwarded shell or add --headless."
            ),
        }
        if collision_sdf_report is not None:
            result["collision_mesh_penetration"] = collision_sdf_report
        return result
    helper_path = Path(__file__).with_name("contact_trajectory_physics_validate.py")
    if not helper_path.is_file():
        raise FileNotFoundError(f"Isaac Gym validation helper is missing: {helper_path}")
    frame_ids = _evenly_spaced_frame_ids(hold_frame_ids, maximum_frames)
    request = {
        "hand_urdf": str(hand.urdf_path),
        "hand_dof_names": tuple(hand.dof_names),
        "contact_body_names": tuple(contact_names),
        "collision_urdf": str(context.collision_urdf_path),
        "frame_ids": frame_ids,
        "wrist_pos": np.asarray(corrected["opt_wrist_pos"], dtype=np.float64)[frame_ids],
        "wrist_rot": np.asarray(corrected["opt_wrist_rot"], dtype=np.float64)[frame_ids],
        "dof_pos": np.asarray(corrected["opt_dof_pos"], dtype=np.float64)[frame_ids],
        "object_trajectory": np.asarray(context.object_trajectory, dtype=np.float64)[frame_ids],
        "headless": bool(headless),
        "settle_steps": int(settle_steps),
        "min_contact_frame_ratio": float(min_contact_frame_ratio),
        "max_contact_force": float(max_contact_force),
        "max_initial_overlap": float(max_initial_overlap),
    }
    if not headless:
        request.update(
            {
                "replay_wrist_pos": np.asarray(corrected["opt_wrist_pos"], dtype=np.float64),
                "replay_wrist_rot": np.asarray(corrected["opt_wrist_rot"], dtype=np.float64),
                "replay_dof_pos": np.asarray(corrected["opt_dof_pos"], dtype=np.float64),
                "replay_object_trajectory": np.asarray(context.object_trajectory, dtype=np.float64),
            }
        )
        if collision_sdf_report is not None:
            request.update(
                {
                    "replay_sdf_frame_ids": np.asarray(
                        collision_sdf_report.get("frame_ids", ()), dtype=np.int64
                    ),
                    "replay_sdf_penetration_depth": np.asarray(
                        collision_sdf_report.get("penetration_depth_frame", ()),
                        dtype=np.float64,
                    ),
                    "replay_sdf_worst_points_world": np.asarray(
                        collision_sdf_report.get("worst_point_world_frame", ()),
                        dtype=np.float64,
                    ),
                    "replay_sdf_hand_escape_direction_world": np.asarray(
                        collision_sdf_report.get(
                            "hand_escape_direction_world_frame", ()
                        ),
                        dtype=np.float64,
                    ),
                    "replay_sdf_maximum_penetration": float(
                        collision_sdf_report.get("maximum_penetration", 0.0)
                    ),
                    "replay_sdf_threshold": float(
                        collision_sdf_report.get("maximum_penetration_threshold", 0.0)
                    ),
                    "replay_sdf_worst_frame": collision_sdf_report.get("worst_frame"),
                    "replay_sdf_worst_link": collision_sdf_report.get("worst_link"),
                    "replay_sdf_uses_proxy_geometry": bool(
                        collision_sdf_report.get("uses_proxy_geometry", False)
                    ),
                }
            )
    with tempfile.TemporaryDirectory(prefix="contact-physics-") as temporary_directory:
        temporary_directory = Path(temporary_directory)
        request_path = temporary_directory / "request.pkl"
        result_path = temporary_directory / "result.json"
        alignment_report = None
        if collision_sdf_validator is not None:
            aligned_hand_urdf, alignment_report = (
                collision_sdf_validator.write_physx_aligned_hand_urdf(
                    temporary_directory / "hand_collision_proxy"
                )
            )
            request["hand_urdf"] = str(aligned_hand_urdf)
            request["hand_collision_geometry_alignment"] = alignment_report
        _atomic_pickle_dump(request, request_path)
        environment = os.environ.copy()
        python_library = str(Path(sys.prefix) / "lib")
        current_library_path = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = (
            python_library if not current_library_path else python_library + os.pathsep + current_library_path
        )
        environment["PYTHONUNBUFFERED"] = "1"
        process = subprocess.run(
            [sys.executable, str(helper_path), "--request", str(request_path), "--result", str(result_path)],
            cwd=str(PROJECT_ROOT),
            env=environment,
            text=True,
            stdout=subprocess.PIPE if headless else None,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        if not result_path.is_file():
            result = {
                "name": "isaac_gym_physics",
                "status": "error",
                "passed": False,
                "returncode": int(process.returncode),
                "output": "" if process.stdout is None else process.stdout[-12000:],
            }
            if collision_sdf_report is not None:
                result["collision_mesh_penetration"] = collision_sdf_report
            if alignment_report is not None:
                result["hand_collision_geometry_alignment"] = alignment_report
            return result
        with result_path.open("r", encoding="utf-8") as stream:
            result = json.load(stream)
        result["returncode"] = int(process.returncode)
        result["output"] = "" if process.stdout is None else process.stdout[-12000:]
        result["passed"] = bool(result.get("passed", False) and process.returncode == 0)
        if not result["passed"] and result.get("status") == "passed":
            result["status"] = "error"
        if collision_sdf_report is not None:
            result["collision_mesh_penetration"] = collision_sdf_report
        if alignment_report is not None:
            result["hand_collision_geometry_alignment"] = alignment_report
        return result


def _physics_failure_score(report: Mapping[str, object]) -> float:
    """Continuous score used only to accept monotonic PhysX repair steps."""

    if report.get("status") == "error" or not bool(report.get("all_finite", True)):
        return float("inf")
    thresholds = report.get("thresholds", {})
    min_ratio = max(float(thresholds.get("min_contact_frame_ratio", 0.5)), 1e-9)
    max_force = max(float(thresholds.get("max_contact_force", 1.0)), 1e-9)
    max_overlap = max(float(thresholds.get("max_initial_overlap", 0.01)), 1e-9)
    object_ratio = float(report.get("object_contact_frame_ratio", 0.0))
    intended_ratio = float(report.get("intended_link_contact_frame_ratio", 0.0))
    force = float(report.get("maximum_contact_force", float("inf")))
    overlap = float(report.get("maximum_initial_overlap", float("inf")))
    return float(
        max(0.0, (min_ratio - object_ratio) / min_ratio)
        + max(0.0, (min_ratio - intended_ratio) / min_ratio)
        + max(0.0, force / max_force - 1.0)
        + max(0.0, overlap / max_overlap - 1.0)
    )


def _sdf_penetration_objective(report: Mapping[str, object]) -> float:
    """Aggregate sampled violations while retaining pressure on the worst frame."""

    try:
        depth = np.asarray(report["penetration_depth_frame"], dtype=np.float64)
        threshold = float(report["maximum_penetration_threshold"])
    except (KeyError, TypeError, ValueError):
        return float("inf")
    if depth.ndim != 1 or not np.isfinite(depth).all() or not np.isfinite(threshold):
        return float("inf")
    excess = np.maximum(depth - threshold, 0.0)
    if not len(excess):
        return 0.0
    return float(excess.max(initial=0.0) + np.sqrt(np.mean(excess**2)))


def _finger_name(body_name: Optional[str]) -> Optional[str]:
    value = "" if body_name is None else str(body_name).lower()
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        if finger in value:
            return finger
    return None


def _interpolate_sample_correction(
    length: int,
    sample_frames: np.ndarray,
    sample_values: np.ndarray,
    *,
    sigma: float = 3.0,
) -> np.ndarray:
    """Interpolate sparse PhysX corrections without per-frame discontinuities."""

    sample_frames = np.asarray(sample_frames, dtype=np.int64)
    sample_values = np.asarray(sample_values, dtype=np.float64)
    if sample_values.ndim != 2 or len(sample_values) != len(sample_frames):
        raise ValueError("Sparse physical correction arrays have inconsistent shapes")
    result = np.zeros((length, sample_values.shape[1]), dtype=np.float64)
    timeline = np.arange(length, dtype=np.float64)
    for column in range(sample_values.shape[1]):
        result[:, column] = np.interp(
            timeline,
            sample_frames.astype(np.float64),
            sample_values[:, column],
            left=0.0,
            right=0.0,
        )
    if sigma > 0 and length > 2:
        result = gaussian_filter1d(result, sigma=sigma, axis=0, mode="nearest")
    return result


def _sdf_gradient_ik_delta(
    kinematics: URDFCollisionKinematics,
    body_name: str,
    dof_position: np.ndarray,
    wrist_rotvec: np.ndarray,
    active_indices: np.ndarray,
    contact_point_link: np.ndarray,
    escape_normal_world: np.ndarray,
    escape_distance: float,
) -> np.ndarray:
    """Move the exact SDF collision point along its hand escape gradient."""

    dof_position = np.asarray(dof_position, dtype=np.float64)
    active_indices = np.asarray(active_indices, dtype=np.int64)
    normal = np.asarray(escape_normal_world, dtype=np.float64)
    normal_norm = float(np.linalg.norm(normal))
    result = np.zeros_like(dof_position)
    if not len(active_indices) or normal_norm <= 1e-9:
        return result
    transforms = kinematics.link_transforms(dof_position)
    if body_name not in transforms or body_name not in kinematics.link_points:
        return result
    root_rotation = Rotation.from_rotvec(
        np.asarray(wrist_rotvec, dtype=np.float64)
    ).as_matrix()
    contact_point_link = np.asarray(contact_point_link, dtype=np.float64)
    if contact_point_link.shape != (3,) or not np.isfinite(contact_point_link).all():
        return result
    base_local = _transform_points(
        contact_point_link[None], transforms[body_name]
    )[0]
    normal_unit = normal / normal_norm
    jacobian = np.zeros((3, len(active_indices)), dtype=np.float64)
    epsilon = 1e-4
    for column, dof_index in enumerate(active_indices):
        perturbed = dof_position.copy()
        perturbed[dof_index] += epsilon
        moved_transform = kinematics.link_transforms(perturbed)[body_name]
        moved_local = _transform_points(
            contact_point_link[None],
            moved_transform,
        )[0]
        jacobian[:, column] = root_rotation @ ((moved_local - base_local) / epsilon)
    desired = normal_unit * float(escape_distance)
    damping = 2e-4
    try:
        active_delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping * np.eye(3), desired
        )
    except np.linalg.LinAlgError:
        return result
    active_delta = np.clip(active_delta, -np.deg2rad(3.0), np.deg2rad(3.0))
    result[active_indices] = active_delta
    return result


def _sdf_multi_constraint_ik_delta(
    kinematics: URDFCollisionKinematics,
    dof_position: np.ndarray,
    wrist_rotvec: np.ndarray,
    constraints: Sequence[Mapping[str, object]],
    maximum_penetration: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Jointly solve directional IK residuals at multiple collision points."""

    dof_position = np.asarray(dof_position, dtype=np.float64)
    result = np.zeros_like(dof_position)
    transforms = kinematics.link_transforms(dof_position)
    root_rotation = Rotation.from_rotvec(
        np.asarray(wrist_rotvec, dtype=np.float64)
    ).as_matrix()
    rows = []
    targets = []
    weights = []
    active_union = set()
    epsilon = 1e-4
    for constraint in constraints:
        link_name = str(constraint.get("link_name", ""))
        if link_name not in transforms:
            continue
        active_indices = kinematics.actuated_ancestor_dof_indices(link_name)
        if not len(active_indices):
            continue
        contact_point = np.asarray(
            constraint.get("hand_contact_point_link"), dtype=np.float64
        )
        escape = np.asarray(
            constraint.get("hand_escape_direction_world"), dtype=np.float64
        )
        escape_norm = float(np.linalg.norm(escape))
        if (
            contact_point.shape != (3,)
            or not np.isfinite(contact_point).all()
            or escape_norm <= 1e-9
        ):
            continue
        escape /= escape_norm
        base_local = _transform_points(
            contact_point[None], transforms[link_name]
        )[0]
        row = np.zeros(len(dof_position), dtype=np.float64)
        for dof_index in active_indices:
            perturbed = dof_position.copy()
            perturbed[dof_index] += epsilon
            moved_transform = kinematics.link_transforms(perturbed)[link_name]
            moved_local = _transform_points(
                contact_point[None], moved_transform
            )[0]
            point_jacobian_world = root_rotation @ (
                (moved_local - base_local) / epsilon
            )
            row[dof_index] = float(escape @ point_jacobian_world)
        if np.linalg.norm(row) <= 1e-9:
            continue
        depth = float(constraint.get("depth", 0.0))
        target = min(
            0.003,
            max(0.00025, depth - float(maximum_penetration) + 0.0005),
        )
        rows.append(row)
        targets.append(target)
        weights.append(
            np.sqrt(
                np.clip(
                    depth / max(float(maximum_penetration), 0.00025),
                    1.0,
                    10.0,
                )
            )
        )
        active_union.update(int(index) for index in active_indices)
    if not rows:
        return result, np.empty(0, dtype=np.int64)
    jacobian = np.stack(rows)
    target = np.asarray(targets, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    weighted_jacobian = jacobian * weight[:, None]
    weighted_target = target * weight
    damping = 2e-4
    normal_matrix = weighted_jacobian.T @ weighted_jacobian
    rhs = weighted_jacobian.T @ weighted_target
    try:
        delta = np.linalg.solve(
            normal_matrix + damping * np.eye(len(dof_position)), rhs
        )
    except np.linalg.LinAlgError:
        delta = np.linalg.lstsq(
            normal_matrix + damping * np.eye(len(dof_position)),
            rhs,
            rcond=None,
        )[0]
    active_indices = np.asarray(sorted(active_union), dtype=np.int64)
    result[active_indices] = np.clip(
        delta[active_indices], -np.deg2rad(3.0), np.deg2rad(3.0)
    )
    return result, active_indices


def _physical_invariant_report(
    semantic: Mapping,
    corrected: Mapping,
    dof_lower: np.ndarray,
    dof_upper: np.ndarray,
    active_dof_names: Sequence[str],
    hand: "OfflineHandSpec",
) -> Dict[str, object]:
    semantic_pos = np.asarray(semantic["opt_wrist_pos"], dtype=np.float64)
    corrected_pos = np.asarray(corrected["opt_wrist_pos"], dtype=np.float64)
    semantic_rot = np.asarray(semantic["opt_wrist_rot"], dtype=np.float64)
    corrected_rot = np.asarray(corrected["opt_wrist_rot"], dtype=np.float64)
    semantic_dof = np.asarray(semantic["opt_dof_pos"], dtype=np.float64)
    corrected_dof = np.asarray(corrected["opt_dof_pos"], dtype=np.float64)
    delta_dof = corrected_dof - semantic_dof
    delta_wrist = corrected_pos - semantic_pos
    active = {str(name) for name in active_dof_names}
    inactive_indices = [index for index, name in enumerate(hand.dof_names) if name not in active]
    inactive_error = 0.0 if not inactive_indices else float(
        np.max(np.abs(delta_dof[:, inactive_indices]))
    )
    lower_violation = np.maximum(np.asarray(dof_lower)[None] - corrected_dof, 0.0)
    upper_violation = np.maximum(corrected_dof - np.asarray(dof_upper)[None], 0.0)
    result = {
        "all_finite": bool(
            np.isfinite(corrected_pos).all()
            and np.isfinite(corrected_rot).all()
            and np.isfinite(corrected_dof).all()
        ),
        "rotation_max_abs_error": float(np.max(np.abs(corrected_rot - semantic_rot))),
        "inactive_dof_max_abs_error": inactive_error,
        "dof_max_abs_correction": float(np.max(np.abs(delta_dof))),
        "wrist_max_correction": float(np.max(np.linalg.norm(delta_wrist, axis=1))),
        "dof_limit_max_violation": float(max(lower_violation.max(), upper_violation.max())),
        "dof_delta_max_step": float(np.max(np.abs(np.diff(delta_dof, axis=0))))
        if len(delta_dof) > 1
        else 0.0,
        "wrist_delta_max_step": float(np.max(np.linalg.norm(np.diff(delta_wrist, axis=0), axis=1)))
        if len(delta_wrist) > 1
        else 0.0,
        "active_dof_names": list(active_dof_names),
    }
    result["passed"] = bool(
        result["all_finite"]
        and result["rotation_max_abs_error"] <= 1e-7
        and result["inactive_dof_max_abs_error"] <= 1e-7
        and result["dof_limit_max_violation"] <= 1e-7
        and result["dof_max_abs_correction"] <= np.deg2rad(15.0) + 1e-7
        and result["wrist_max_correction"] <= 0.01 + 1e-7
        and result["dof_delta_max_step"] <= np.deg2rad(5.0)
        and result["wrist_delta_max_step"] <= 0.003
    )
    return result


def _build_corrected_mano_body_targets(
    semantic: Mapping,
    corrected: Mapping,
    kinematics: URDFCollisionKinematics,
    body_names: Sequence[str],
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Move MANO-derived Stage-1 targets by the final dex-hand FK displacement."""

    body_names = tuple(str(name) for name in body_names)
    semantic_targets = np.asarray(semantic.get("opt_joints_pos"), dtype=np.float64)
    if semantic_targets.ndim != 3 or semantic_targets.shape[1:] != (len(body_names), 3):
        raise ValueError(
            "Cannot synchronize Stage-1 MANO targets: opt_joints_pos shape "
            f"{semantic_targets.shape} does not match {len(body_names)} PhysX bodies"
        )
    semantic_fk = kinematics.world_link_origins_trajectory(
        body_names,
        semantic["opt_wrist_pos"],
        semantic["opt_wrist_rot"],
        semantic["opt_dof_pos"],
    )
    corrected_fk = kinematics.world_link_origins_trajectory(
        body_names,
        corrected["opt_wrist_pos"],
        corrected["opt_wrist_rot"],
        corrected["opt_dof_pos"],
    )
    targets = semantic_targets + corrected_fk - semantic_fk
    passed = bool(np.isfinite(targets).all())
    report = {
        "name": "mano_target_sync",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "method": "dexhand_fk_delta_on_mano_body_targets",
        "body_names": list(body_names),
        "body_count": len(body_names),
        "maximum_target_shift": float(
            np.max(np.linalg.norm(targets - semantic_targets, axis=-1))
        ),
    }
    return targets.astype(np.float32), report


def _legacy_single_step_physical_correction(
    semantic: Mapping,
    initial_physics: Mapping[str, object],
    hand: "OfflineHandSpec",
    context: OakInkContext,
    contact_names: Sequence[str],
    hold_frame_ids: np.ndarray,
    kinematics: URDFCollisionKinematics,
    tool: SemanticTool,
    settings: GeometrySettings,
    args: argparse.Namespace,
    collision_sdf_validator: Optional[CollisionMeshSDFValidator] = None,
    collision_sdf_initialization_error: Optional[Exception] = None,
) -> Tuple[Dict, Dict[str, object], Dict[str, object], Dict[str, object]]:
    """Repair SDF penetrations by local IK, then retain PhysX as the final gate."""

    baseline = copy.deepcopy(semantic)
    current = copy.deepcopy(semantic)
    current_physics = dict(initial_physics)
    initial_sdf = current_physics.get("collision_mesh_penetration")
    last_sdf = dict(initial_sdf) if isinstance(initial_sdf, Mapping) else None
    history = [
        {
            "round": 0,
            "physics_score": _physics_failure_score(current_physics),
            "sdf_maximum_penetration": (
                None
                if last_sdf is None
                else last_sdf.get("maximum_penetration")
            ),
        }
    ]
    active_dofs = set()
    active_links = set()
    total_wrist_delta = np.zeros_like(np.asarray(current["opt_wrist_pos"], dtype=np.float64))
    total_dof_delta = np.zeros_like(np.asarray(current["opt_dof_pos"], dtype=np.float64))
    limits_lower = np.asarray(current_physics.get("dof_lower_limits"), dtype=np.float64)
    limits_upper = np.asarray(current_physics.get("dof_upper_limits"), dtype=np.float64)
    if limits_lower.shape != (len(hand.dof_names),) or limits_upper.shape != (len(hand.dof_names),):
        limits_lower = np.full(len(hand.dof_names), -np.inf, dtype=np.float64)
        limits_upper = np.full(len(hand.dof_names), np.inf, dtype=np.float64)

    for round_index in range(1, 5):
        per_frame = list(current_physics.get("per_frame", ()))
        sample_frames = np.asarray(current_physics.get("frame_ids", ()), dtype=np.int64)
        if len(per_frame) != len(sample_frames) or len(sample_frames) == 0:
            break
        current_sdf = _collision_mesh_penetration_report(
            collision_sdf_validator,
            collision_sdf_initialization_error,
            current,
            context.object_trajectory,
            sample_frames,
            args.collision_sdf_max_penetration,
            collect_constraints=True,
        )
        last_sdf = current_sdf
        if current_sdf.get("status") == "error":
            history.append(
                {
                    "round": round_index,
                    "accepted": False,
                    "reason": "collision_sdf_unavailable",
                    "error": current_sdf.get("error"),
                }
            )
            break
        sdf_frame_ids = np.asarray(current_sdf["frame_ids"], dtype=np.int64)
        sdf_depth = np.asarray(
            current_sdf["penetration_depth_frame"], dtype=np.float64
        )
        sdf_constraints = list(current_sdf["penetration_constraints_frame"])
        if not (
            np.array_equal(sdf_frame_ids, sample_frames)
            and sdf_depth.shape == (len(sample_frames),)
            and len(sdf_constraints) == len(sample_frames)
        ):
            history.append(
                {
                    "round": round_index,
                    "accepted": False,
                    "reason": "collision_sdf_constraint_shape_mismatch",
                }
            )
            break
        sdf_lookup = {
            int(frame): local for local, frame in enumerate(sdf_frame_ids)
        }
        thresholds = current_physics.get("thresholds", {})
        max_force = float(thresholds.get("max_contact_force", args.physics_max_contact_force))
        max_overlap = float(thresholds.get("max_initial_overlap", args.physics_max_initial_overlap))
        sparse_dof = np.zeros((len(sample_frames), len(hand.dof_names)), dtype=np.float64)
        sparse_wrist = np.zeros((len(sample_frames), 3), dtype=np.float64)
        current_dof = np.asarray(current["opt_dof_pos"], dtype=np.float64)
        candidate_found = False
        for record in per_frame:
            frame = int(record["frame"])
            sdf_index = sdf_lookup.get(frame)
            sdf_failed = bool(
                sdf_index is not None
                and sdf_depth[sdf_index]
                > float(args.collision_sdf_max_penetration)
            )
            force_failed = float(record.get("maximum_contact_force", 0.0)) > max_force
            overlap_failed = float(record.get("maximum_initial_overlap", 0.0)) > max_overlap
            if not (force_failed or overlap_failed or sdf_failed):
                continue
            if sdf_index is None or sdf_depth[sdf_index] <= 1e-7:
                continue
            constraints = [
                constraint
                for constraint in sdf_constraints[sdf_index]
                if float(constraint.get("depth", 0.0)) > 1e-7
                and (
                    not sdf_failed
                    or float(constraint.get("depth", 0.0))
                    > float(args.collision_sdf_max_penetration)
                )
            ]
            if not constraints:
                continue
            finger_constraints = [
                constraint
                for constraint in constraints
                if len(
                    kinematics.actuated_ancestor_dof_indices(
                        str(constraint.get("link_name", ""))
                    )
                )
            ]
            raw_delta, frame_active_indices = _sdf_multi_constraint_ik_delta(
                kinematics,
                current_dof[frame],
                np.asarray(current["opt_wrist_rot"], dtype=np.float64)[frame],
                finger_constraints,
                args.collision_sdf_max_penetration,
            )
            if np.max(np.abs(raw_delta), initial=0.0) >= 1e-6:
                sparse_dof[sdf_index] = raw_delta
                active_dofs.update(
                    hand.dof_names[index]
                    for index in frame_active_indices
                    if abs(float(raw_delta[index])) >= 1e-8
                )
                candidate_found = True
            palm_steps = []
            palm_weights = []
            for constraint in constraints:
                link_name = str(constraint.get("link_name", ""))
                if not (
                    link_name == kinematics.root_link
                    or any(
                        token in link_name.lower()
                        for token in ("palm", "base", "wrist")
                    )
                ):
                    continue
                escape = np.asarray(
                    constraint.get("hand_escape_direction_world"),
                    dtype=np.float64,
                )
                escape_norm = float(np.linalg.norm(escape))
                if escape_norm <= 1e-9:
                    continue
                depth = float(constraint.get("depth", 0.0))
                distance = min(
                    0.002,
                    max(
                        0.00025,
                        depth
                        - float(args.collision_sdf_max_penetration)
                        + 0.0005,
                    ),
                )
                palm_steps.append(escape / escape_norm * distance)
                palm_weights.append(
                    np.clip(
                        depth
                        / max(float(args.collision_sdf_max_penetration), 0.00025),
                        1.0,
                        10.0,
                    )
                )
            if palm_steps:
                wrist_delta = np.average(
                    np.stack(palm_steps),
                    axis=0,
                    weights=np.asarray(palm_weights),
                )
                wrist_norm = float(np.linalg.norm(wrist_delta))
                if wrist_norm > 0.002:
                    wrist_delta *= 0.002 / wrist_norm
                sparse_wrist[sdf_index] = wrist_delta
                candidate_found = True
            active_links.update(
                str(constraint.get("link_name", ""))
                for constraint in constraints
                if constraint.get("link_name")
            )
        if not candidate_found:
            history.append(
                {
                    "round": round_index,
                    "accepted": False,
                    "reason": "no_failed_frame_with_reachable_sdf_constraints",
                    "sdf_maximum_penetration": current_sdf.get(
                        "maximum_penetration"
                    ),
                }
            )
            break

        dof_step = _interpolate_sample_correction(len(current_dof), sample_frames, sparse_dof)
        wrist_step = _interpolate_sample_correction(len(current_dof), sample_frames, sparse_wrist)
        sdf_candidates = []
        sdf_candidate_errors = []
        for step_scale in (1.0, 0.5, 0.25, 0.125):
            proposed_total_dof = np.clip(
                total_dof_delta + step_scale * dof_step,
                -np.deg2rad(15.0),
                np.deg2rad(15.0),
            )
            proposed_total_wrist = total_wrist_delta + step_scale * wrist_step
            wrist_norm = np.linalg.norm(proposed_total_wrist, axis=1)
            too_large = wrist_norm > 0.01
            if too_large.any():
                proposed_total_wrist[too_large] *= (
                    0.01 / wrist_norm[too_large]
                )[:, None]

            candidate = copy.deepcopy(baseline)
            candidate["opt_dof_pos"] = np.clip(
                np.asarray(baseline["opt_dof_pos"], dtype=np.float64)
                + proposed_total_dof,
                limits_lower[None],
                limits_upper[None],
            ).astype(np.asarray(baseline["opt_dof_pos"]).dtype)
            candidate["opt_wrist_pos"] = (
                np.asarray(baseline["opt_wrist_pos"], dtype=np.float64)
                + proposed_total_wrist
            ).astype(np.asarray(baseline["opt_wrist_pos"]).dtype)
            candidate["opt_wrist_rot"] = np.asarray(
                baseline["opt_wrist_rot"]
            ).copy()
            candidate_sdf = _collision_mesh_penetration_report(
                collision_sdf_validator,
                collision_sdf_initialization_error,
                candidate,
                context.object_trajectory,
                sample_frames,
                args.collision_sdf_max_penetration,
            )
            if candidate_sdf.get("status") == "error":
                sdf_candidate_errors.append(candidate_sdf.get("error"))
                continue
            sdf_candidates.append(
                (
                    float(candidate_sdf["maximum_penetration"]),
                    step_scale,
                    candidate,
                    candidate_sdf,
                )
            )
        if not sdf_candidates:
            history.append(
                {
                    "round": round_index,
                    "accepted": False,
                    "reason": "proposal_collision_sdf_unavailable",
                    "errors": sdf_candidate_errors,
                }
            )
            break
        semantic_rejections = []
        proposal = None
        proposal_sdf = None
        selected_step_scale = None
        semantic_metrics = None
        for _, step_scale, candidate, candidate_sdf in sorted(
            sdf_candidates, key=lambda item: item[0]
        ):
            world_points = kinematics.world_surface_trajectory(
                candidate["opt_wrist_pos"],
                candidate["opt_wrist_rot"],
                candidate["opt_dof_pos"],
            )
            tool_points = world_to_tool_points(
                world_points, context.object_trajectory
            )
            candidate_semantic = contact_geometry_metrics(
                tool_points[hold_frame_ids],
                kinematics.point_link_ids_array,
                _contact_link_ids(kinematics, contact_names),
                tool,
                settings,
            )
            if _is_feasible(candidate_semantic, settings):
                proposal = candidate
                proposal_sdf = candidate_sdf
                selected_step_scale = step_scale
                semantic_metrics = candidate_semantic
                break
            semantic_rejections.append(
                {
                    "step_scale": step_scale,
                    "semantic": _summary_metrics(candidate_semantic),
                }
            )
        if proposal is None or proposal_sdf is None or semantic_metrics is None:
            history.append(
                {
                    "round": round_index,
                    "accepted": False,
                    "reason": "semantic_geometry_regressed_for_all_sdf_steps",
                    "candidates": semantic_rejections,
                }
            )
            break
        proposal_physics = run_isaac_gym_validation(
            proposal,
            hand,
            context,
            contact_names,
            hold_frame_ids,
            headless=args.headless,
            maximum_frames=args.physics_max_frames,
            settle_steps=args.physics_settle_steps,
            min_contact_frame_ratio=args.physics_min_contact_frame_ratio,
            max_contact_force=args.physics_max_contact_force,
            max_initial_overlap=args.physics_max_initial_overlap,
            timeout=args.physics_timeout,
            collision_sdf_report=proposal_sdf,
            collision_sdf_validator=collision_sdf_validator,
        )
        old_score = _physics_failure_score(current_physics)
        new_score = _physics_failure_score(proposal_physics)
        old_sdf_depth = float(current_sdf.get("maximum_penetration", float("inf")))
        new_sdf_depth = float(proposal_sdf.get("maximum_penetration", float("inf")))
        physics_improved = bool(new_score + 1e-9 < old_score)
        physics_not_worse = bool(new_score <= old_score + 1e-9)
        sdf_improved = bool(new_sdf_depth + 1e-7 < old_sdf_depth)
        sdf_not_worse = bool(new_sdf_depth <= old_sdf_depth + 1e-7)
        final_gates_passed = bool(
            proposal_physics.get("passed", False)
            and proposal_sdf.get("passed", False)
        )
        # SDF is the inner repair objective. PhysX is deliberately not required
        # to improve monotonically at every bounded step; it remains mandatory
        # in ``final_gates_passed`` and in the final physical tier verdict.
        accepted = bool(
            final_gates_passed
            or sdf_improved
            or (physics_improved and sdf_not_worse)
        )
        history.append(
            {
                "round": round_index,
                "accepted": accepted,
                "score_before": old_score,
                "score_after": new_score,
                "sdf_maximum_penetration_before": old_sdf_depth,
                "sdf_maximum_penetration_after": new_sdf_depth,
                "physics_improved": physics_improved,
                "physics_not_worse": physics_not_worse,
                "sdf_improved": sdf_improved,
                "final_gates_passed": final_gates_passed,
                "selected_sdf_step_scale": selected_step_scale,
                "sdf_line_search": [
                    {
                        "step_scale": step_scale,
                        "maximum_penetration": maximum_penetration,
                    }
                    for maximum_penetration, step_scale, _, _ in sdf_candidates
                ],
                "semantic": _summary_metrics(semantic_metrics),
            }
        )
        if not accepted:
            break
        current = proposal
        current_physics = proposal_physics
        last_sdf = proposal_sdf
        total_dof_delta = (
            np.asarray(current["opt_dof_pos"], dtype=np.float64)
            - np.asarray(baseline["opt_dof_pos"], dtype=np.float64)
        )
        total_wrist_delta = (
            np.asarray(current["opt_wrist_pos"], dtype=np.float64)
            - np.asarray(baseline["opt_wrist_pos"], dtype=np.float64)
        )
        if bool(
            current_physics.get("passed", False)
            and proposal_sdf.get("passed", False)
        ):
            break

    invariants = _physical_invariant_report(
        baseline, current, limits_lower, limits_upper, sorted(active_dofs), hand
    )
    sdf_passed = bool(
        isinstance(last_sdf, Mapping) and last_sdf.get("passed", False)
    )
    passed = bool(
        current_physics.get("passed", False)
        and sdf_passed
        and invariants["passed"]
    )
    report = {
        "name": "physical_optimization",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "attempted": True,
        "method": "collision_sdf_multi_point_ancestor_chain_ik_physx_final_gate",
        "sdf_penetration_passed_on_physx_frames": sdf_passed,
        "sdf_maximum_penetration_on_physx_frames": (
            None if last_sdf is None else last_sdf.get("maximum_penetration")
        ),
        "sdf_objective_on_physx_frames": (
            None
            if not isinstance(last_sdf, Mapping)
            else _sdf_penetration_objective(last_sdf)
        ),
        "round_count": len(history) - 1,
        "active_link_names": sorted(active_links),
        "active_dof_names": sorted(active_dofs),
        "history": history,
        "invariants": invariants,
    }
    return current, current_physics, report, invariants


def run_physical_correction(
    semantic: Mapping,
    initial_physics: Mapping[str, object],
    hand: "OfflineHandSpec",
    context: OakInkContext,
    contact_names: Sequence[str],
    hold_frame_ids: np.ndarray,
    kinematics: URDFCollisionKinematics,
    tool: SemanticTool,
    settings: GeometrySettings,
    args: argparse.Namespace,
    collision_sdf_validator: Optional[CollisionMeshSDFValidator] = None,
    collision_sdf_initialization_error: Optional[Exception] = None,
) -> Tuple[Dict, Dict[str, object], Dict[str, object], Dict[str, object]]:
    """Run several SDF-only IK steps before each final PhysX outer check."""

    baseline = copy.deepcopy(semantic)
    current = copy.deepcopy(semantic)
    current_physics = dict(initial_physics)
    active_dofs = set()
    active_links = set()
    outer_history = []
    total_inner_iterations = 0
    last_sdf = None
    trust_scale = 1.0
    stopped_reason = None
    best_physx_state = copy.deepcopy(current) if current_physics.get("passed", False) else None
    best_physx_report = dict(current_physics) if current_physics.get("passed", False) else None
    best_physx_sdf = (
        current_physics.get("collision_mesh_penetration")
        if current_physics.get("passed", False)
        else None
    )
    limits_lower = np.asarray(
        current_physics.get("dof_lower_limits"), dtype=np.float64
    )
    limits_upper = np.asarray(
        current_physics.get("dof_upper_limits"), dtype=np.float64
    )
    if (
        limits_lower.shape != (len(hand.dof_names),)
        or limits_upper.shape != (len(hand.dof_names),)
    ):
        limits_lower = np.full(len(hand.dof_names), -np.inf, dtype=np.float64)
        limits_upper = np.full(len(hand.dof_names), np.inf, dtype=np.float64)

    for outer_round in range(1, args.physics_max_outer_rounds + 1):
        sample_frames = np.asarray(
            current_physics.get("frame_ids", ()), dtype=np.int64
        )
        per_frame = list(current_physics.get("per_frame", ()))
        if len(sample_frames) == 0 or len(per_frame) != len(sample_frames):
            outer_history.append(
                {
                    "outer_round": outer_round,
                    "status": "stopped",
                    "reason": "missing_physx_frame_diagnostics",
                }
            )
            break
        physics_by_frame = {
            int(record["frame"]): record for record in per_frame
        }
        thresholds = current_physics.get("thresholds", {})
        max_force = float(
            thresholds.get("max_contact_force", args.physics_max_contact_force)
        )
        max_overlap = float(
            thresholds.get(
                "max_initial_overlap", args.physics_max_initial_overlap
            )
        )
        inner_history = []
        accepted_inner_steps = 0
        stalled_iterations = 0

        for inner_iteration in range(1, args.sdf_max_inner_iterations + 1):
            total_inner_iterations += 1
            current_sdf = _collision_mesh_penetration_report(
                collision_sdf_validator,
                collision_sdf_initialization_error,
                current,
                context.object_trajectory,
                sample_frames,
                args.collision_sdf_max_penetration,
                collect_constraints=True,
            )
            last_sdf = current_sdf
            if current_sdf.get("status") == "error":
                inner_history.append(
                    {
                        "inner_iteration": inner_iteration,
                        "accepted": False,
                        "reason": "collision_sdf_unavailable",
                        "error": current_sdf.get("error"),
                    }
                )
                break
            constraints_frame = list(
                current_sdf.get("penetration_constraints_frame", ())
            )
            depth_frame = np.asarray(
                current_sdf.get("penetration_depth_frame", ()),
                dtype=np.float64,
            )
            if (
                len(constraints_frame) != len(sample_frames)
                or depth_frame.shape != (len(sample_frames),)
            ):
                inner_history.append(
                    {
                        "inner_iteration": inner_iteration,
                        "accepted": False,
                        "reason": "collision_sdf_constraint_shape_mismatch",
                    }
                )
                break
            if current_sdf.get("passed", False) and current_physics.get(
                "passed", False
            ):
                break

            current_dof = np.asarray(current["opt_dof_pos"], dtype=np.float64)
            sparse_dof = np.zeros(
                (len(sample_frames), len(hand.dof_names)), dtype=np.float64
            )
            sparse_wrist = np.zeros((len(sample_frames), 3), dtype=np.float64)
            step_active_indices = set()
            step_active_links = set()
            constraint_count = 0
            for sample_index, frame in enumerate(sample_frames):
                record = physics_by_frame.get(int(frame), {})
                force_failed = (
                    float(record.get("maximum_contact_force", 0.0)) > max_force
                )
                overlap_failed = (
                    float(record.get("maximum_initial_overlap", 0.0))
                    > max_overlap
                )
                sdf_failed = (
                    depth_frame[sample_index]
                    > float(args.collision_sdf_max_penetration)
                )
                if not (force_failed or overlap_failed or sdf_failed):
                    continue
                constraints = [
                    constraint
                    for constraint in constraints_frame[sample_index]
                    if float(constraint.get("depth", 0.0)) > 1e-7
                    and (
                        not sdf_failed
                        or float(constraint.get("depth", 0.0))
                        > float(args.collision_sdf_max_penetration)
                    )
                ]
                if not constraints:
                    continue
                constraint_count += len(constraints)
                finger_constraints = [
                    constraint
                    for constraint in constraints
                    if len(
                        kinematics.actuated_ancestor_dof_indices(
                            str(constraint.get("link_name", ""))
                        )
                    )
                ]
                frame_delta, frame_active = _sdf_multi_constraint_ik_delta(
                    kinematics,
                    current_dof[int(frame)],
                    np.asarray(current["opt_wrist_rot"], dtype=np.float64)[
                        int(frame)
                    ],
                    finger_constraints,
                    args.collision_sdf_max_penetration,
                )
                if np.max(np.abs(frame_delta), initial=0.0) >= 1e-7:
                    sparse_dof[sample_index] = frame_delta
                    step_active_indices.update(int(index) for index in frame_active)

                palm_steps = []
                palm_weights = []
                for constraint in constraints:
                    link_name = str(constraint.get("link_name", ""))
                    if not (
                        link_name == kinematics.root_link
                        or any(
                            token in link_name.lower()
                            for token in ("palm", "base", "wrist")
                        )
                    ):
                        continue
                    escape = np.asarray(
                        constraint.get("hand_escape_direction_world"),
                        dtype=np.float64,
                    )
                    norm = float(np.linalg.norm(escape))
                    if norm <= 1e-9:
                        continue
                    depth = float(constraint.get("depth", 0.0))
                    distance = min(
                        0.002,
                        max(
                            0.00025,
                            depth
                            - float(args.collision_sdf_max_penetration)
                            + 0.0005,
                        ),
                    )
                    palm_steps.append(escape / norm * distance)
                    palm_weights.append(
                        np.clip(
                            depth
                            / max(
                                float(args.collision_sdf_max_penetration),
                                0.00025,
                            ),
                            1.0,
                            10.0,
                        )
                    )
                if palm_steps:
                    wrist_delta = np.average(
                        np.stack(palm_steps),
                        axis=0,
                        weights=np.asarray(palm_weights),
                    )
                    norm = float(np.linalg.norm(wrist_delta))
                    if norm > 0.002:
                        wrist_delta *= 0.002 / norm
                    sparse_wrist[sample_index] = wrist_delta
                step_active_links.update(
                    str(constraint.get("link_name", ""))
                    for constraint in constraints
                    if constraint.get("link_name")
                )

            if constraint_count == 0 or not (
                np.max(np.abs(sparse_dof), initial=0.0) >= 1e-7
                or np.max(np.abs(sparse_wrist), initial=0.0) >= 1e-7
            ):
                inner_history.append(
                    {
                        "inner_iteration": inner_iteration,
                        "accepted": False,
                        "reason": "no_reachable_sdf_constraints",
                    }
                )
                break

            dof_step = _interpolate_sample_correction(
                len(current_dof), sample_frames, sparse_dof
            )
            wrist_step = _interpolate_sample_correction(
                len(current_dof), sample_frames, sparse_wrist
            )
            total_dof = current_dof - np.asarray(
                baseline["opt_dof_pos"], dtype=np.float64
            )
            total_wrist = np.asarray(
                current["opt_wrist_pos"], dtype=np.float64
            ) - np.asarray(baseline["opt_wrist_pos"], dtype=np.float64)
            candidates = []
            for base_step_scale in (1.0, 0.5, 0.25, 0.125):
                step_scale = trust_scale * base_step_scale
                candidate_total_dof = np.clip(
                    total_dof + step_scale * dof_step,
                    -np.deg2rad(15.0),
                    np.deg2rad(15.0),
                )
                candidate_total_wrist = total_wrist + step_scale * wrist_step
                wrist_norm = np.linalg.norm(candidate_total_wrist, axis=1)
                too_large = wrist_norm > 0.01
                if too_large.any():
                    candidate_total_wrist[too_large] *= (
                        0.01 / wrist_norm[too_large]
                    )[:, None]
                candidate = copy.deepcopy(baseline)
                candidate["opt_dof_pos"] = np.clip(
                    np.asarray(baseline["opt_dof_pos"], dtype=np.float64)
                    + candidate_total_dof,
                    limits_lower[None],
                    limits_upper[None],
                ).astype(np.asarray(baseline["opt_dof_pos"]).dtype)
                candidate["opt_wrist_pos"] = (
                    np.asarray(baseline["opt_wrist_pos"], dtype=np.float64)
                    + candidate_total_wrist
                ).astype(np.asarray(baseline["opt_wrist_pos"]).dtype)
                candidate["opt_wrist_rot"] = np.asarray(
                    baseline["opt_wrist_rot"]
                ).copy()
                candidate_sdf = _collision_mesh_penetration_report(
                    collision_sdf_validator,
                    collision_sdf_initialization_error,
                    candidate,
                    context.object_trajectory,
                    sample_frames,
                    args.collision_sdf_max_penetration,
                )
                if candidate_sdf.get("status") != "error":
                    candidates.append(
                        (
                            _sdf_penetration_objective(candidate_sdf),
                            float(candidate_sdf["maximum_penetration"]),
                            step_scale,
                            candidate,
                            candidate_sdf,
                        )
                    )
            selected = None
            semantic_rejections = []
            for _, _, step_scale, candidate, candidate_sdf in sorted(
                candidates, key=lambda item: item[0]
            ):
                world_points = kinematics.world_surface_trajectory(
                    candidate["opt_wrist_pos"],
                    candidate["opt_wrist_rot"],
                    candidate["opt_dof_pos"],
                )
                tool_points = world_to_tool_points(
                    world_points, context.object_trajectory
                )
                semantic_metrics = contact_geometry_metrics(
                    tool_points[hold_frame_ids],
                    kinematics.point_link_ids_array,
                    _contact_link_ids(kinematics, contact_names),
                    tool,
                    settings,
                )
                if _is_feasible(semantic_metrics, settings):
                    selected = (
                        step_scale,
                        candidate,
                        candidate_sdf,
                        semantic_metrics,
                    )
                    break
                semantic_rejections.append(step_scale)
            if selected is None:
                stopped_reason = "semantic_constraint_conflict"
                inner_history.append(
                    {
                        "inner_iteration": inner_iteration,
                        "accepted": False,
                        "reason": "no_semantic_feasible_sdf_step",
                        "rejected_step_scales": semantic_rejections,
                    }
                )
                break
            step_scale, proposal, proposal_sdf, semantic_metrics = selected
            old_depth = float(current_sdf["maximum_penetration"])
            new_depth = float(proposal_sdf["maximum_penetration"])
            old_objective = _sdf_penetration_objective(current_sdf)
            new_objective = _sdf_penetration_objective(proposal_sdf)
            improvement = old_objective - new_objective
            accepted = bool(
                proposal_sdf.get("passed", False)
                or improvement > float(args.sdf_improvement_tolerance)
            )
            inner_history.append(
                {
                    "inner_iteration": inner_iteration,
                    "accepted": accepted,
                    "constraint_count": constraint_count,
                    "selected_step_scale": step_scale,
                    "maximum_penetration_before": old_depth,
                    "maximum_penetration_after": new_depth,
                    "objective_before": old_objective,
                    "objective_after": new_objective,
                    "objective_improvement": improvement,
                    "semantic": _summary_metrics(semantic_metrics),
                }
            )
            if not accepted:
                stalled_iterations += 1
                trust_scale *= 0.5
                if stalled_iterations >= args.sdf_stall_patience:
                    stopped_reason = "local_sdf_minimum_or_conflicting_constraints"
                    break
                continue
            stalled_iterations = 0
            trust_scale = min(1.0, trust_scale * 1.25)
            current = proposal
            last_sdf = proposal_sdf
            accepted_inner_steps += 1
            active_links.update(step_active_links)
            active_dofs.update(hand.dof_names[index] for index in step_active_indices)
            if proposal_sdf.get("passed", False):
                break

        physics_before = _physics_failure_score(current_physics)
        if accepted_inner_steps == 0:
            if stopped_reason is None:
                stopped_reason = "no_accepted_sdf_inner_step"
            outer_history.append(
                {
                    "outer_round": outer_round,
                    "status": "stalled",
                    "physics_score_before": physics_before,
                    "inner_history": inner_history,
                }
            )
            break
        proposal_physics = run_isaac_gym_validation(
            current,
            hand,
            context,
            contact_names,
            hold_frame_ids,
            headless=args.headless,
            maximum_frames=args.physics_max_frames,
            settle_steps=args.physics_settle_steps,
            min_contact_frame_ratio=args.physics_min_contact_frame_ratio,
            max_contact_force=args.physics_max_contact_force,
            max_initial_overlap=args.physics_max_initial_overlap,
            timeout=args.physics_timeout,
            collision_sdf_report=last_sdf,
            collision_sdf_validator=collision_sdf_validator,
        )
        physics_after = _physics_failure_score(proposal_physics)
        current_physics = proposal_physics
        if current_physics.get("passed", False):
            previous_best_objective = (
                float("inf")
                if not isinstance(best_physx_sdf, Mapping)
                else _sdf_penetration_objective(best_physx_sdf)
            )
            current_objective = (
                float("inf")
                if not isinstance(last_sdf, Mapping)
                else _sdf_penetration_objective(last_sdf)
            )
            if current_objective < previous_best_objective:
                best_physx_state = copy.deepcopy(current)
                best_physx_report = dict(current_physics)
                best_physx_sdf = dict(last_sdf)
        outer_history.append(
            {
                "outer_round": outer_round,
                "status": "accepted",
                "accepted_inner_steps": accepted_inner_steps,
                "physics_score_before": physics_before,
                "physics_score_after": physics_after,
                "physics_passed": bool(current_physics.get("passed", False)),
                "sdf_passed": bool(last_sdf and last_sdf.get("passed", False)),
                "inner_history": inner_history,
            }
        )
        if bool(
            current_physics.get("passed", False)
            and last_sdf
            and last_sdf.get("passed", False)
        ):
            break

    if (
        not current_physics.get("passed", False)
        and best_physx_state is not None
        and best_physx_report is not None
    ):
        current = best_physx_state
        current_physics = best_physx_report
        last_sdf = best_physx_sdf
        stopped_reason = stopped_reason or "restored_best_physx_passing_candidate"

    invariants = _physical_invariant_report(
        baseline, current, limits_lower, limits_upper, sorted(active_dofs), hand
    )
    sdf_passed = bool(last_sdf and last_sdf.get("passed", False))
    passed = bool(
        current_physics.get("passed", False)
        and sdf_passed
        and invariants["passed"]
    )
    report = {
        "name": "physical_optimization",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "attempted": True,
        "method": "adaptive_multi_point_sdf_inner_loop_physx_outer_gate",
        "round_count": len(outer_history),
        "sdf_inner_iteration_count": total_inner_iterations,
        "final_trust_scale": trust_scale,
        "stopped_reason": stopped_reason,
        "best_physx_candidate_available": best_physx_state is not None,
        "active_link_names": sorted(active_links),
        "active_dof_names": sorted(active_dofs),
        "sdf_penetration_passed_on_physx_frames": sdf_passed,
        "sdf_maximum_penetration_on_physx_frames": (
            None if last_sdf is None else last_sdf.get("maximum_penetration")
        ),
        "sdf_objective_on_physx_frames": (
            None
            if not isinstance(last_sdf, Mapping)
            else _sdf_penetration_objective(last_sdf)
        ),
        "history": outer_history,
        "invariants": invariants,
    }
    return current, current_physics, report, invariants


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
    semantic_axis: Optional[Mapping[str, object]] = None,
    delta_tool: Optional[np.ndarray] = None,
    contact_transfer: Optional[Mapping[str, object]] = None,
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

    def draw_semantic_axis(axis) -> None:
        if semantic_axis is None:
            return
        origin = np.asarray(semantic_axis["origin_tool"], dtype=np.float64)
        direction = np.asarray(semantic_axis["direction_tool"], dtype=np.float64)
        axis_length = max(
            0.02,
            0.0 if delta_tool is None else float(np.linalg.norm(delta_tool)),
        )
        axis.quiver(
            origin[0],
            origin[1],
            origin[2],
            direction[0],
            direction[1],
            direction[2],
            length=axis_length,
            normalize=True,
            color="magenta",
            linewidth=2.0,
        )

    def draw_contact_transfer(axis) -> None:
        if contact_transfer is None:
            return
        source_origin = np.asarray(
            contact_transfer["source_red_center_tool"], dtype=np.float64
        )
        target_origin = np.asarray(
            contact_transfer["target_red_center_tool"], dtype=np.float64
        )
        displacement = target_origin - source_origin
        if float(np.linalg.norm(displacement)) > 1e-10:
            axis.quiver(
                *source_origin,
                *displacement,
                length=1.0,
                normalize=False,
                color="magenta",
                linewidth=2.0,
            )

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
        draw_semantic_axis(axis)
        draw_contact_transfer(axis)
    axes[2].scatter(red[:, 0], red[:, 1], red[:, 2], c="red", s=3, alpha=0.5)
    axes[2].scatter(black[:, 0], black[:, 1], black[:, 2], c="black", s=3, alpha=0.3)
    axes[2].set_title("Tool semantic regions (tool frame)")
    axes[2].set_box_aspect((1, 1, 1))
    draw_semantic_axis(axes[2])
    draw_contact_transfer(axes[2])

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
    _log(f"开始修正 data_idx={args.data_idx}, dexhand={args.dexhand}, side={args.side}")
    if args.side != "right":
        raise ValueError("Contact trajectory correction and its memory bank are right-hand only")
    if args.new_obj is None:
        raise ValueError("Right-hand contact correction requires --new-obj to name the target tool")
    if args.bank and args.translation_mode != "contact-translation":
        raise ValueError("--bank currently requires --translation-mode contact-translation")
    if args.bank and args.output_pkl:
        raise ValueError("--bank cannot be combined with --output-pkl; use the canonical bank-backed path")
    if args.physics_max_outer_rounds < 1 or args.sdf_max_inner_iterations < 1:
        raise ValueError("Physical outer rounds and SDF inner iterations must be positive")
    if args.sdf_stall_patience < 1 or args.sdf_improvement_tolerance < 0.0:
        raise ValueError("SDF stall patience must be positive and tolerance non-negative")
    if args.physx and not args.headless and not os.environ.get("DISPLAY"):
        raise RuntimeError(
            "Isaac Gym Viewer was requested because --headless was omitted, but DISPLAY is empty. "
            "Run from a graphical desktop/X-forwarded shell, or add --headless for non-visual execution."
        )
    hand = _resolve_dexhand(args.dexhand, args.side)
    input_path = Path(args.input_pkl).resolve() if args.input_pkl else _resolve_source_pkl(
        args.data_idx, args.dexhand, args.side
    )
    output_root = Path(args.output_root)
    output_path = (
        Path(args.output_pkl).resolve()
        if args.output_pkl
        else corrected_retarget_path(
            output_root,
            input_path.parent.name,
            args.data_idx,
            args.new_obj,
            input_path.name,
        )
    )
    if output_path == input_path:
        raise ValueError("Corrected output must not overwrite the source retarget PKL")
    output_existed = output_path.exists()
    if output_existed:
        _log(f"检测到已有修正 PKL；本次验证通过后将原子覆盖：{output_path}")
    else:
        _log(f"本次将生成新的修正 PKL：{output_path}")
    _log("读取原始 retarget、新旧工具语义 mesh、collision URDF 和工具轨迹")
    context = load_oakink_context(
        args.data_idx,
        args.side,
        Path(args.data_root),
        None if args.mesh is None else Path(args.mesh),
        None if args.collision_urdf is None else Path(args.collision_urdf),
        skip=args.skip,
        new_obj=args.new_obj,
    )
    _log("修正前执行新旧 mesh 红/黑逐顶点语义分割预检")
    source_tool = load_semantic_tool(context.source_mesh_path, label="old/source")
    tool = load_semantic_tool(context.mesh_path, label="new/target")
    semantic_precheck = {
        "passed": True,
        "source": source_tool.semantic_report,
        "target": tool.semantic_report,
    }
    _log(
        "语义预检通过：旧 mesh={}红/{}黑，新 mesh={}红/{}黑".format(
            source_tool.red_vertex_count,
            source_tool.black_vertex_count,
            tool.red_vertex_count,
            tool.black_vertex_count,
        )
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
    kinematics = URDFCollisionKinematics(
        Path(hand.urdf_path), hand.dof_names, points_per_link=args.points_per_link
    )
    collision_sdf_validator = None
    collision_sdf_initialization_error = None
    try:
        collision_sdf_validator = CollisionMeshSDFValidator(
            kinematics,
            context.collision_urdf_path,
            surface_samples_per_link=args.collision_sdf_surface_samples,
            penetration_points_per_link=args.collision_sdf_points_per_link,
            maximum_points_per_frame=args.collision_sdf_max_points_per_frame,
            cluster_radius=args.collision_sdf_cluster_radius,
            cluster_normal_degrees=args.collision_sdf_cluster_normal_degrees,
        )
    except Exception as error:
        collision_sdf_initialization_error = error
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
    validation_settings = {
        "geometry": asdict(settings),
        "translation_mode": args.translation_mode,
        "max_residual_shift": args.max_residual_shift,
        "ramp_in_frames": args.ramp_in_frames,
        "ramp_out_frames": args.ramp_out_frames,
        "physics_required": bool(args.physx),
        "physics_max_frames": args.physics_max_frames,
        "physics_max_outer_rounds": args.physics_max_outer_rounds,
        "sdf_max_inner_iterations": args.sdf_max_inner_iterations,
        "sdf_stall_patience": args.sdf_stall_patience,
        "sdf_improvement_tolerance": args.sdf_improvement_tolerance,
        "physics_settle_steps": args.physics_settle_steps,
        "physics_min_contact_frame_ratio": args.physics_min_contact_frame_ratio,
        "physics_max_contact_force": args.physics_max_contact_force,
        "physics_max_initial_overlap": args.physics_max_initial_overlap,
        "collision_mesh_penetration_required": True,
        "collision_sdf_max_penetration": args.collision_sdf_max_penetration,
        "collision_sdf_surface_samples": args.collision_sdf_surface_samples,
        "collision_sdf_points_per_link": args.collision_sdf_points_per_link,
        "collision_sdf_max_points_per_frame": args.collision_sdf_max_points_per_frame,
        "collision_sdf_cluster_radius": args.collision_sdf_cluster_radius,
        "collision_sdf_cluster_normal_degrees": args.collision_sdf_cluster_normal_degrees,
    }
    validation_settings_hash = stable_mapping_hash(validation_settings)
    if (args.start_frame is None) != (args.end_frame is None):
        raise ValueError("--start-frame and --end-frame must be provided together")
    if args.start_frame is None:
        start_frame, end_frame = detect_grasp_interval(
            points_before,
            kinematics.point_link_ids_array,
            contact_ids,
            source_tool,
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
    _log(f"抓握修正区间：frame {start_frame}–{end_frame}，轨迹总长度={length}")
    hold_frame_ids = np.arange(start_frame, end_frame + 1, dtype=np.int64)
    _log(f"[1/2] 开始语义几何优化（translation_mode={args.translation_mode}）")
    optimization_frames = _select_optimization_frames(start_frame, end_frame, args.max_optimization_frames)
    semantic_axis_report = None
    semantic_axis_direction = None
    contact_transfer_report = None
    contact_transform_tool = None
    contact_initial_delta = None
    source_contact_intent = extract_source_contact_intent(
        points_before[hold_frame_ids],
        kinematics.point_link_ids_array,
        contact_ids,
        source_tool,
        settings,
    )
    memory_sequence_directory = bank_sequence_dir(DEFAULT_BANK_ROOT, args.data_idx)
    memory_descriptor = build_mesh_descriptor(tool) if args.bank else None
    memory_sequence_metadata = {
        "data_id": args.data_idx,
        "side": args.side,
        "dexhand": args.dexhand,
        "source_pkl_sha256": _sha256(input_path),
        "source_tool_mesh_sha256": _sha256(context.source_mesh_path),
        "annotation_sha256": _sha256(context.annotation_path),
        "trajectory_length": int(length),
        "skip": int(args.skip),
        "hand_urdf_bundle_sha256": _urdf_bundle_sha256(Path(hand.urdf_path)),
        "contact_links": contact_names,
    }
    memory_retrieval_report = {
        "enabled": bool(args.bank),
        "bank_sequence_directory": memory_sequence_directory,
        "sequence_found": False,
        "candidates": [],
        "selected_tool_mesh_name": None,
        "direct_validation_passed": False,
        "used_as_optimizer_warm_start": False,
        "physical_transfer_policy": (
            "reuse geometry residual only; never transfer stored joint/wrist physical deltas; "
            "always rerun the configured PhysX gate and local repair"
        ),
    }
    if args.translation_mode == "contact-translation":
        contact_initial_delta, contact_transfer_report = contact_translation_transfer(
            source_tool, tool
        )
        ranked_entries = []
        if args.bank and require_compatible_sequence(
            memory_sequence_directory, memory_sequence_metadata
        ):
            memory_retrieval_report["sequence_found"] = True
            ranked_entries = load_ranked_entries(
                memory_sequence_directory, memory_descriptor, top_k=3
            )

        evaluated_candidates = []
        for entry in ranked_entries:
            residual = np.asarray(
                entry.get("correction", {}).get("residual_tool"), dtype=np.float64
            )
            if residual.shape != (3,) or not np.isfinite(residual).all():
                continue
            residual = np.clip(
                residual, -args.max_residual_shift, args.max_residual_shift
            )
            candidate_delta = contact_initial_delta + residual
            candidate_metrics = contact_geometry_metrics(
                points_before[hold_frame_ids] + candidate_delta[None, None],
                kinematics.point_link_ids_array,
                contact_ids,
                tool,
                settings,
            )
            candidate_objective = _transformed_geometry_objective(
                points_before[optimization_frames] + candidate_delta[None, None],
                kinematics.point_link_ids_array,
                contact_ids,
                tool,
                settings,
            )
            candidate_record = {
                "tool_mesh_name": entry.get("tool_mesh_name"),
                "similarity": entry["similarity"],
                "bank_schema_version": int(entry.get("bank_schema_version", 1)),
                "capabilities": entry.get("capabilities", {}),
                "delta_tool": candidate_delta,
                "residual_tool": residual,
                "geometry_objective": float(candidate_objective),
                "red_frame_pass_ratio": float(candidate_metrics["red_frame_pass_ratio"]),
                "black_violation_fraction": float(candidate_metrics["black_violation_fraction"]),
                "passed": _is_feasible(candidate_metrics, settings),
            }
            evaluated_candidates.append(candidate_record)

        memory_retrieval_report["candidates"] = evaluated_candidates
        passing_candidates = [item for item in evaluated_candidates if item["passed"]]
        if passing_candidates:
            selected = min(passing_candidates, key=lambda item: item["geometry_objective"])
            delta_tool = np.asarray(selected["delta_tool"], dtype=np.float64)
            optimizer_report = {
                "mode": "memory_direct",
                "objective": float(selected["geometry_objective"]),
                "residual": np.asarray(selected["residual_tool"], dtype=np.float64),
                "residual_translation_norm": float(np.linalg.norm(selected["residual_tool"])),
                "initial_delta_tool": contact_initial_delta,
                "final_delta_tool": delta_tool,
                "final_translation_norm": float(np.linalg.norm(delta_tool)),
                "final_rotation_degrees": 0.0,
                "source_tool_mesh_name": selected["tool_mesh_name"],
            }
            memory_retrieval_report["selected_tool_mesh_name"] = selected["tool_mesh_name"]
            memory_retrieval_report["direct_validation_passed"] = True
            _log(
                "Memory bank 直接候选通过语义几何预验收：{}".format(
                    selected["tool_mesh_name"]
                )
            )
        else:
            initial_residual = None
            if evaluated_candidates:
                selected = min(
                    evaluated_candidates, key=lambda item: item["geometry_objective"]
                )
                initial_residual = np.asarray(selected["residual_tool"], dtype=np.float64)
                memory_retrieval_report["selected_tool_mesh_name"] = selected["tool_mesh_name"]
                memory_retrieval_report["used_as_optimizer_warm_start"] = True
                _log(
                    "Memory bank 候选未直接通过；使用 {} 的修正残差作为优化初值".format(
                        selected["tool_mesh_name"]
                    )
                )
            delta_tool, optimizer_report = optimize_contact_translation(
                points_before[optimization_frames],
                kinematics.point_link_ids_array,
                contact_ids,
                tool,
                settings,
                contact_initial_delta,
                max_residual_shift=args.max_residual_shift,
                max_iterations=args.optimizer_iterations,
                initial_residual=initial_residual,
            )
    elif args.translation_mode == "semantic-axis":
        semantic_axis_direction, semantic_axis_report = estimate_semantic_axis(
            tool, roi_distance=args.axis_roi_distance
        )
        delta_tool, optimizer_report = optimize_tool_translation_along_axis(
            points_before[optimization_frames],
            kinematics.point_link_ids_array,
            contact_ids,
            tool,
            settings,
            semantic_axis_direction,
            grid_samples=args.axis_grid_samples,
        )
    else:
        delta_tool, optimizer_report = optimize_tool_translation(
            points_before[optimization_frames],
            kinematics.point_link_ids_array,
            contact_ids,
            tool,
            settings,
            seed=args.seed,
            max_iterations=args.optimizer_iterations,
        )
    if contact_transform_tool is None:
        contact_transform_tool = np.eye(4, dtype=np.float64)
        contact_transform_tool[:3, 3] = delta_tool
    optimizer_history = [optimizer_report]

    # A bounded representative set keeps refinement tractable, while final acceptance
    # is still evaluated on every frame in the complete grasp interval.
    for refinement_round in range(args.max_refinement_rounds):
        hold_transform = np.repeat(
            contact_transform_tool[None], len(hold_frame_ids), axis=0
        )
        hold_metrics = contact_geometry_metrics(
            apply_tool_transform_points(points_before[hold_frame_ids], hold_transform),
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
        next_optimization_frames = _bounded_frame_sample(
            np.concatenate([optimization_frames, failed_frames]),
            args.max_optimization_frames,
        )
        if np.array_equal(next_optimization_frames, optimization_frames):
            optimizer_history[-1]["refinement_stopped_reason"] = (
                "all failed frames reduce to the same bounded representative set"
            )
            optimizer_history[-1]["failed_frame_count"] = int(len(failed_frames))
            _log(
                "停止 refinement：失败帧经硬上限采样后未引入新的代表帧，"
                "继续优化只会重复相同目标"
            )
            break
        optimization_frames = next_optimization_frames
        _log(
            "第 {} 轮局部 refinement：失败帧={}，本轮优化帧={}（硬上限={}）".format(
                refinement_round + 1,
                len(failed_frames),
                len(optimization_frames),
                args.max_optimization_frames,
            )
        )
        if args.translation_mode == "contact-translation":
            delta_tool, refinement_report = optimize_contact_translation(
                points_before[optimization_frames],
                kinematics.point_link_ids_array,
                contact_ids,
                tool,
                settings,
                contact_initial_delta,
                max_residual_shift=args.max_residual_shift,
                max_iterations=args.optimizer_iterations,
                initial_residual=np.asarray(optimizer_history[-1]["residual"]),
            )
        elif semantic_axis_direction is not None:
            delta_tool, refinement_report = optimize_tool_translation_along_axis(
                points_before[optimization_frames],
                kinematics.point_link_ids_array,
                contact_ids,
                tool,
                settings,
                semantic_axis_direction,
                grid_samples=args.axis_grid_samples,
                initial_delta=delta_tool,
            )
        else:
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
        contact_transform_tool = np.eye(4, dtype=np.float64)
        contact_transform_tool[:3, 3] = delta_tool
        refinement_report["round"] = refinement_round + 1
        refinement_report["failed_frame_count"] = int(len(failed_frames))
        refinement_report["optimization_frames"] = optimization_frames
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
    ).astype(
        np.asarray(source["opt_wrist_pos"]).dtype
    )
    if "opt_joints_pos" in source:
        corrected["opt_joints_pos"] = (
            np.asarray(source["opt_joints_pos"], dtype=np.float64) + delta_world[:, None]
        ).astype(
            np.asarray(source["opt_joints_pos"]).dtype
        )
    corrected["opt_wrist_rot"] = np.asarray(source["opt_wrist_rot"]).copy()
    corrected["opt_dof_pos"] = np.asarray(source["opt_dof_pos"]).copy()
    semantic_corrected = copy.deepcopy(corrected)

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
    semantic_invariant_report = _invariant_report(source, semantic_corrected, delta_world)
    invariant_report = semantic_invariant_report
    final_axis_constraint = (
        _axis_constraint_report(delta_tool, semantic_axis_direction)
        if semantic_axis_direction is not None
        else None
    )
    geometry_feasible = _is_feasible(after_hold, settings)
    semantic_geometry_passed = bool(
        geometry_feasible
        and semantic_invariant_report["passed"]
        and (final_axis_constraint is None or final_axis_constraint["passed"])
    )
    semantic_geometry_tier = {
        "name": "semantic_geometry_optimization",
        "status": "passed" if semantic_geometry_passed else "failed",
        "passed": semantic_geometry_passed,
        "geometry_feasible": geometry_feasible,
        "invariants_passed": bool(semantic_invariant_report["passed"]),
        "axis_constraint_passed": bool(
            final_axis_constraint is None or final_axis_constraint["passed"]
        ),
    }
    final_rotation_degrees = 0.0
    _log(
        "[1/2] 语义几何优化{}：红区合格帧={:.2%}，黑区违规比例={:.6%}，"
        "工具局部平移={:.2f} mm".format(
            "通过" if semantic_geometry_passed else "失败",
            float(after_hold["red_frame_pass_ratio"]),
            float(after_hold["black_violation_fraction"]),
            float(np.linalg.norm(delta_tool)) * 1000.0,
        )
    )
    collision_sdf_tier = _collision_mesh_penetration_report(
        collision_sdf_validator,
        collision_sdf_initialization_error,
        corrected,
        context.object_trajectory,
        hold_frame_ids,
        args.collision_sdf_max_penetration,
    )
    if collision_sdf_tier["status"] == "error":
        _log(
            "[collision mesh/SDF 验收] 无法计算：{}（本次必需验收失败）".format(
                collision_sdf_tier.get("error", "unknown error")
            )
        )
    else:
        _log(
            "[collision mesh/SDF 验收] {}：最大穿透={:.3f} mm，超阈值帧={}/{}，"
            "worst frame={}，link={}，凸包代理组件={}"
            "（启用 PhysX 时可驱动局部修正）".format(
                "通过" if collision_sdf_tier["passed"] else "失败",
                float(collision_sdf_tier["maximum_penetration"]) * 1000.0,
                int(collision_sdf_tier["threshold_exceeding_frame_count"]),
                len(collision_sdf_tier["frame_ids"]),
                collision_sdf_tier["worst_frame"],
                collision_sdf_tier["worst_link"],
                collision_sdf_tier["convex_hull_proxy_count"],
            )
        )
    physical_optimization_tier = {
        "name": "physical_optimization",
        "status": "not_requested",
        "passed": False,
        "required": False,
        "attempted": False,
    }
    mano_target_sync_tier = {
        "name": "mano_target_sync",
        "status": "not_requested",
        "passed": False,
        "required": False,
    }
    initial_physics_tier = None
    if not args.physx:
        physics_tier = {
            "name": "isaac_gym_physics",
            "status": "not_requested",
            "passed": False,
            "required": False,
            "reason": "PhysX validation was not enabled; pass --physx to require it",
        }
    elif not semantic_geometry_tier["passed"]:
        physics_tier = {
            "name": "isaac_gym_physics",
            "status": "skipped",
            "passed": False,
            "required": True,
            "reason": "semantic geometry validation did not pass",
        }
        physical_optimization_tier.update(
            status="skipped", required=True,
            reason="semantic geometry validation did not pass",
        )
        mano_target_sync_tier.update(
            status="skipped", required=True,
            reason="semantic geometry validation did not pass",
        )
    else:
        if args.headless:
            _log("[2/2] 开始 Isaac Gym/PhysX 无窗口接触验证")
        else:
            _log("[2/2] 开始 Isaac Gym/PhysX 接触验证；随后打开修正轨迹 Viewer")
            _log("Viewer：唯一工具 + 修正绿手；关闭窗口后继续保存")
        try:
            initial_physics_tier = run_isaac_gym_validation(
                corrected,
                hand,
                context,
                contact_names,
                hold_frame_ids,
                headless=args.headless,
                maximum_frames=args.physics_max_frames,
                settle_steps=args.physics_settle_steps,
                min_contact_frame_ratio=args.physics_min_contact_frame_ratio,
                max_contact_force=args.physics_max_contact_force,
                max_initial_overlap=args.physics_max_initial_overlap,
                timeout=args.physics_timeout,
                collision_sdf_report=collision_sdf_tier,
                collision_sdf_validator=collision_sdf_validator,
            )
        except Exception as error:
            initial_physics_tier = {
                "name": "isaac_gym_physics",
                "status": "error",
                "passed": False,
                "required": True,
                "error": f"{type(error).__name__}: {error}",
            }
        initial_physics_tier.setdefault("required", True)
        physics_tier = initial_physics_tier
        initial_physics_passed = bool(initial_physics_tier.get("passed", False))
        initial_sdf_passed = bool(collision_sdf_tier.get("passed", False))
        if initial_physics_passed and initial_sdf_passed:
            physical_optimization_tier = {
                "name": "physical_optimization",
                "status": "not_needed",
                "passed": True,
                "required": True,
                "attempted": False,
                "reason": "initial collision SDF and PhysX validation passed",
            }
            mano_target_sync_tier = {
                "name": "mano_target_sync",
                "status": "not_needed",
                "passed": True,
                "required": True,
                "reason": "no physical trajectory change was required",
            }
        elif (
            initial_physics_tier.get("status") in ("passed", "failed")
            and collision_sdf_tier.get("status") in ("passed", "failed")
        ):
            failed_gates = []
            if not initial_sdf_passed:
                failed_gates.append("collision mesh/SDF")
            if not initial_physics_passed:
                failed_gates.append("PhysX")
            _log(
                "[2/3] 初次 {} 验收失败；开始 SDF 穿透点/梯度驱动的"
                "局部手指链或掌部修正".format(" + ".join(failed_gates))
            )
            try:
                corrected, physics_tier, physical_optimization_tier, invariant_report = (
                    run_physical_correction(
                        semantic_corrected,
                        initial_physics_tier,
                        hand,
                        context,
                        contact_names,
                        hold_frame_ids,
                        kinematics,
                        tool,
                        settings,
                        args,
                        collision_sdf_validator,
                        collision_sdf_initialization_error,
                    )
                )
            except Exception as error:
                corrected = copy.deepcopy(semantic_corrected)
                physics_tier = initial_physics_tier
                physical_optimization_tier = {
                    "name": "physical_optimization",
                    "status": "error",
                    "passed": False,
                    "required": True,
                    "attempted": True,
                    "error": f"{type(error).__name__}: {error}",
                }
                invariant_report = semantic_invariant_report
            if bool(physical_optimization_tier.get("passed", False)):
                try:
                    body_names = physics_tier.get("hand_body_names", ())
                    corrected_mano_targets, mano_target_sync_tier = (
                        _build_corrected_mano_body_targets(
                            semantic_corrected, corrected, kinematics, body_names
                        )
                    )
                    corrected["corrected_mano_body_targets"] = corrected_mano_targets
                    corrected["corrected_mano_body_names"] = list(body_names)
                    corrected["opt_joints_pos"] = corrected_mano_targets.copy()
                    mano_target_sync_tier["required"] = True
                except Exception as error:
                    mano_target_sync_tier = {
                        "name": "mano_target_sync",
                        "status": "error",
                        "passed": False,
                        "required": True,
                        "error": f"{type(error).__name__}: {error}",
                    }
            else:
                mano_target_sync_tier = {
                    "name": "mano_target_sync",
                    "status": "skipped",
                    "passed": False,
                    "required": True,
                    "reason": "physical optimization did not pass",
                }
        else:
            physical_optimization_tier = {
                "name": "physical_optimization",
                "status": "skipped",
                "passed": False,
                "required": True,
                "attempted": False,
                "reason": "initial PhysX validation could not produce repair diagnostics",
            }
            mano_target_sync_tier = {
                "name": "mano_target_sync",
                "status": "skipped",
                "passed": False,
                "required": True,
                "reason": "physical optimization was not run",
            }

    # The SDF payload attached to a proposal Viewer may cover only the PhysX
    # sample frames. Keep it out of the PhysX tier and produce one complete
    # final report below if physical correction changed the trajectory.
    physics_tier.pop("collision_mesh_penetration", None)
    if isinstance(initial_physics_tier, dict):
        initial_physics_tier.pop("collision_mesh_penetration", None)
    if physical_optimization_tier.get("attempted", False):
        collision_sdf_tier = _collision_mesh_penetration_report(
            collision_sdf_validator,
            collision_sdf_initialization_error,
            corrected,
            context.object_trajectory,
            hold_frame_ids,
            args.collision_sdf_max_penetration,
        )
        final_world = kinematics.world_surface_trajectory(
            np.asarray(corrected["opt_wrist_pos"], dtype=np.float64),
            np.asarray(corrected["opt_wrist_rot"], dtype=np.float64),
            np.asarray(corrected["opt_dof_pos"], dtype=np.float64),
        )
        points_after = world_to_tool_points(final_world, context.object_trajectory)
        after_hold = contact_geometry_metrics(
            points_after[hold_frame_ids],
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
        geometry_feasible = _is_feasible(after_hold, settings)
        semantic_geometry_passed = bool(
            geometry_feasible
            and invariant_report["passed"]
            and (final_axis_constraint is None or final_axis_constraint["passed"])
        )
        semantic_geometry_tier.update(
            status="passed" if semantic_geometry_passed else "failed",
            passed=semantic_geometry_passed,
            geometry_feasible=geometry_feasible,
            invariants_passed=bool(invariant_report["passed"]),
        )
    else:
        invariant_report = semantic_invariant_report
    if physics_tier["status"] == "passed":
        _log(
            "[2/2] PhysX 验证通过：工具接触帧={:.2%}，指定末端接触帧={:.2%}，最大接触包络={:.2f} mm".format(
                float(physics_tier["object_contact_frame_ratio"]),
                float(physics_tier["intended_link_contact_frame_ratio"]),
                float(physics_tier["maximum_initial_overlap"]) * 1000.0,
            )
        )
    elif physics_tier["status"] == "not_requested":
        _log("[2/2] 未启用 PhysX 验收；如需启用请在命令中添加 --physx")
    elif physics_tier["status"] == "skipped":
        _log("[2/2] 语义几何验收未通过，跳过 PhysX 验收")
    elif physics_tier["status"] != "skipped":
        _log(f"[2/2] PhysX 验证未通过：{physics_tier.get('error', physics_tier['status'])}")
    validation_tiers = {
        "semantic_geometry": semantic_geometry_tier,
        "collision_mesh_penetration": collision_sdf_tier,
        "isaac_gym_physics": physics_tier,
        "physical_optimization": physical_optimization_tier,
        "mano_target_sync": mano_target_sync_tier,
    }
    overall_passed = bool(
        semantic_geometry_tier["passed"]
        and collision_sdf_tier["passed"]
        and (
            not args.physx
            or (
                physics_tier["passed"]
                and physical_optimization_tier["passed"]
                and mano_target_sync_tier["passed"]
            )
        )
    )
    physical_wrist_delta = (
        np.asarray(corrected["opt_wrist_pos"], dtype=np.float64)
        - np.asarray(semantic_corrected["opt_wrist_pos"], dtype=np.float64)
    )
    physical_dof_delta = (
        np.asarray(corrected["opt_dof_pos"], dtype=np.float64)
        - np.asarray(semantic_corrected["opt_dof_pos"], dtype=np.float64)
    )
    physical_correction_metadata = {
        "enabled": bool(args.physx),
        "attempted": bool(physical_optimization_tier.get("attempted", False)),
        "method": physical_optimization_tier.get("method"),
        "active_link_names": physical_optimization_tier.get("active_link_names", []),
        "active_dof_names": physical_optimization_tier.get("active_dof_names", []),
        "wrist_rotation_locked": True,
        "maximum_wrist_translation": float(
            np.max(np.linalg.norm(physical_wrist_delta, axis=1))
        ),
        "maximum_dof_correction": float(np.max(np.abs(physical_dof_delta))),
        "mano_target_source": (
            "corrected_mano_body_targets"
            if "corrected_mano_body_targets" in corrected
            else "semantic_rigid_transform"
        ),
    }
    corrected["physical_wrist_delta"] = physical_wrist_delta.astype(np.float32)
    corrected["physical_dof_delta"] = physical_dof_delta.astype(np.float32)
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
        "new_obj": args.new_obj,
        "semantic_precheck": semantic_precheck,
        "source_tool_mesh": context.source_mesh_path,
        "source_tool_mesh_sha256": _sha256(context.source_mesh_path),
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
        "translation_mode": args.translation_mode,
        "source_contact_intent": source_contact_intent,
        "contact_transfer": contact_transfer_report,
        "contact_transform_tool": contact_transform_tool,
        "semantic_axis": semantic_axis_report,
        "axis_constraint": final_axis_constraint,
        "delta_tool": delta_tool,
        "delta_norm": float(np.linalg.norm(delta_tool)),
        "rotation_degrees": final_rotation_degrees,
        "settings": asdict(settings),
        "optimizer": optimizer_history[-1],
        "optimizer_history": optimizer_history,
        "memory_retrieval": memory_retrieval_report,
        "validation_settings_hash": validation_settings_hash,
        "before_hold": _summary_metrics(before_hold),
        "after_hold": _summary_metrics(after_hold),
        "before_full": _summary_metrics(before_full),
        "after_full": _summary_metrics(after_full),
        "invariants": invariant_report,
        "semantic_invariants": semantic_invariant_report,
        "validation_tiers": validation_tiers,
        "initial_physics_validation": initial_physics_tier,
        "physical_correction": physical_correction_metadata,
        "collision_mesh_penetration_required": True,
        "physics_validation_required": bool(args.physx),
        "physics_validation_note": (
            "When enabled, PhysX first validates the semantic trajectory, then drives bounded "
            "collision-SDF point/gradient finger IK (or palm-only wrist translation) if that "
            "gate fails. Multiple SDF-only inner steps run before each PhysX outer check. PhysX "
            "remains the final contact-force gate; whole-task stability is still validated by "
            "Stage 1."
        ),
    }
    corrected["contact_retarget"] = {
        "schema_version": SCHEMA_VERSION,
        "source_pkl": str(input_path),
        "data_idx": args.data_idx,
        "side": args.side,
        "dexhand": args.dexhand,
        "new_obj": args.new_obj,
        "tool_mesh": str(context.mesh_path),
        "tool_mesh_sha256": report["tool_mesh_sha256"],
        "collision_urdf": str(context.collision_urdf_path),
        "collision_bundle_sha256": report["collision_bundle_sha256"],
        "source_tool_mesh": str(context.source_mesh_path),
        "source_tool_mesh_sha256": report["source_tool_mesh_sha256"],
        "semantic_precheck": semantic_precheck,
        "translation_mode": args.translation_mode,
        "source_contact_intent": source_contact_intent,
        "contact_transfer": contact_transfer_report,
        "contact_transform_tool": contact_transform_tool.astype(np.float32),
        "semantic_axis": semantic_axis_report,
        "axis_constraint": final_axis_constraint,
        "delta_tool": delta_tool.astype(np.float32),
        "alpha": alpha.astype(np.float32),
        "grasp_interval": np.asarray([start_frame, end_frame], dtype=np.int64),
        "geometry_feasible": geometry_feasible,
        "collision_mesh_penetration_required": True,
        "physics_validation_required": bool(args.physx),
        "validation_tiers": {
            **validation_tiers,
            "collision_mesh_penetration": _compact_collision_mesh_penetration_report(
                collision_sdf_tier
            ),
        },
        "physical_correction": physical_correction_metadata,
        "settings": asdict(settings),
        "memory_retrieval": memory_retrieval_report,
        "validation_settings_hash": validation_settings_hash,
        "validation_summary": {
            "before_hold": report["before_hold"],
            "after_hold": report["after_hold"],
            "invariants": invariant_report,
            "semantic_invariants": semantic_invariant_report,
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
    _log(f"检查报告已写入：{report_path}")
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
        semantic_axis_report,
        delta_tool,
        contact_transfer_report,
    )
    if overall_passed or args.allow_infeasible:
        _atomic_pickle_dump(corrected, output_path)
        _log(f"修正 PKL 已原子{'覆盖' if output_existed else '写入'}：{output_path}")
        if args.bank and overall_passed:
            center_delta = np.asarray(
                contact_transfer_report["initial_delta_tool"], dtype=np.float64
            )
            memory_entry = {
                "bank_schema_version": 2,
                "data_id": args.data_idx,
                "tool_mesh_name": args.new_obj,
                "tool_mesh_sha256": report["tool_mesh_sha256"],
                "collision_bundle_sha256": report["collision_bundle_sha256"],
                "descriptor_file": "descriptor.npz",
                "correction": {
                    "delta_tool": _jsonable(delta_tool),
                    "center_delta_tool": _jsonable(center_delta),
                    "residual_tool": _jsonable(delta_tool - center_delta),
                    "grasp_interval": [int(start_frame), int(end_frame)],
                    "ramp_in_frames": int(args.ramp_in_frames),
                    "ramp_out_frames": int(args.ramp_out_frames),
                },
                "validation_settings_hash": validation_settings_hash,
                "validation": {
                    "overall_passed": True,
                    "semantic_geometry_passed": bool(semantic_geometry_tier["passed"]),
                    "collision_mesh_penetration_passed": bool(
                        collision_sdf_tier["passed"]
                    ),
                    "collision_mesh_maximum_penetration": float(
                        collision_sdf_tier["maximum_penetration"]
                    ),
                    "physics_required": bool(args.physx),
                    "physics_passed": bool(physics_tier["passed"]),
                    "red_frame_pass_ratio": float(after_hold["red_frame_pass_ratio"]),
                    "red_link_ratio_mean": float(after_hold["red_link_ratio_mean"]),
                    "black_violation_fraction": float(after_hold["black_violation_fraction"]),
                    "invariants_passed": bool(invariant_report["passed"]),
                },
                "capabilities": {
                    "semantic_geometry": bool(semantic_geometry_tier["passed"]),
                    "collision_mesh_penetration": bool(
                        collision_sdf_tier["passed"]
                    ),
                    "physics_validated": bool(args.physx and physics_tier["passed"]),
                    "physical_joint_correction": bool(
                        physical_optimization_tier.get("attempted", False)
                        and np.max(np.abs(physical_dof_delta)) > 0.0
                    ),
                    "physical_wrist_correction": bool(
                        physical_optimization_tier.get("attempted", False)
                        and np.max(np.abs(physical_wrist_delta)) > 0.0
                    ),
                    "mano_target_sync": bool(mano_target_sync_tier["passed"]),
                },
                "physical_correction": physical_correction_metadata,
                "corrected_pkl": str(output_path.resolve()),
                "corrected_pkl_sha256": _sha256(output_path),
                "report_path": str(report_path.resolve()),
            }
            write_memory_entry(
                memory_sequence_directory,
                memory_sequence_metadata,
                args.new_obj,
                memory_descriptor,
                memory_entry,
            )
            _log(f"Memory bank 已更新：{memory_sequence_directory / args.new_obj}")
        _log(f"预览图已写入：{preview_path}")
        if overall_passed:
            _log(
                "几何轨迹修正完成，语义几何、collision mesh/SDF 与 PhysX 验收全部通过"
                if args.physx
                else "几何轨迹修正完成，语义几何与 collision mesh/SDF 验收通过（未启用 PhysX）"
            )
        else:
            _log("已写入诊断 PKL（未通过全部已配置验证）")
    else:
        raise RuntimeError(
            "Contact trajectory correction did not pass all configured validation tiers. "
            f"The corrected PKL was not replaced. Inspect {report_path} and {preview_path}, adjust the interval/constraints, "
            "or pass --allow-infeasible only for diagnostic output."
        )
    return report


def run_verification(args: argparse.Namespace) -> Dict[str, object]:
    corrected_path = Path(args.input_pkl).resolve()
    _log(f"开始独立复验：{corrected_path}")
    corrected = _load_retarget_pkl(corrected_path)
    metadata = corrected.get("contact_retarget")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{corrected_path} does not contain contact_retarget metadata")
    source_path = Path(args.source_pkl or metadata["source_pkl"]).resolve()
    source = _load_retarget_pkl(source_path)
    data_idx = args.data_idx or metadata["data_idx"]
    side = args.side or metadata["side"]
    dexhand_name = args.dexhand or metadata["dexhand"]
    if args.new_obj is not None:
        if args.mesh is not None or args.collision_urdf is not None:
            raise ValueError("--new-obj cannot be combined with --mesh or --collision-urdf")
        replacement = resolve_new_object_assets(args.new_obj)
        mesh_path = replacement.mesh_path
        collision_urdf_path = replacement.urdf_path
    else:
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
    source_tool = load_semantic_tool(context.source_mesh_path, label="old/source")
    tool = load_semantic_tool(mesh_path, label="new/target")
    semantic_precheck = {
        "passed": True,
        "source": source_tool.semantic_report,
        "target": tool.semantic_report,
    }
    contact_names = list(hand.contact_body_names)
    contact_ids = _contact_link_ids(model, contact_names)
    stored_settings = metadata.get("settings")
    settings = (
        GeometrySettings(
            **{
                name: stored_settings[name]
                for name in GeometrySettings.__dataclass_fields__
                if name in stored_settings
            }
        )
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
    stored_tiers = metadata.get("validation_tiers")
    stored_collision_tier = (
        stored_tiers.get("collision_mesh_penetration")
        if isinstance(stored_tiers, Mapping)
        else None
    )
    if not isinstance(stored_collision_tier, Mapping):
        stored_collision_tier = {}
    verification_sdf_samples = int(
        args.collision_sdf_surface_samples
        if args.collision_sdf_surface_samples is not None
        else stored_collision_tier.get("surface_samples_per_link", 64)
    )
    verification_sdf_threshold = float(
        args.collision_sdf_max_penetration
        if args.collision_sdf_max_penetration is not None
        else stored_collision_tier.get("maximum_penetration_threshold", 0.001)
    )
    verification_sdf_points_per_link = int(
        args.collision_sdf_points_per_link
        if args.collision_sdf_points_per_link is not None
        else stored_collision_tier.get("penetration_points_per_link", 2)
    )
    verification_sdf_max_points = int(
        args.collision_sdf_max_points_per_frame
        if args.collision_sdf_max_points_per_frame is not None
        else stored_collision_tier.get("maximum_points_per_frame", 16)
    )
    verification_sdf_cluster_radius = float(
        args.collision_sdf_cluster_radius
        if args.collision_sdf_cluster_radius is not None
        else stored_collision_tier.get("cluster_radius", 0.0015)
    )
    verification_sdf_cluster_normal_degrees = float(
        args.collision_sdf_cluster_normal_degrees
        if args.collision_sdf_cluster_normal_degrees is not None
        else stored_collision_tier.get("cluster_normal_degrees", 20.0)
    )
    verification_sdf_validator = None
    verification_sdf_error = None
    try:
        verification_sdf_validator = CollisionMeshSDFValidator(
            model,
            collision_urdf_path,
            surface_samples_per_link=verification_sdf_samples,
            penetration_points_per_link=verification_sdf_points_per_link,
            maximum_points_per_frame=verification_sdf_max_points,
            cluster_radius=verification_sdf_cluster_radius,
            cluster_normal_degrees=verification_sdf_cluster_normal_degrees,
        )
    except Exception as error:
        verification_sdf_error = error
    collision_sdf_tier = _collision_mesh_penetration_report(
        verification_sdf_validator,
        verification_sdf_error,
        corrected,
        context.object_trajectory,
        frame_ids,
        verification_sdf_threshold,
    )
    collision_penetration_required = int(metadata.get("schema_version", 0)) >= 9
    collision_sdf_tier["required"] = collision_penetration_required
    collision_sdf_tier["affects_overall_acceptance"] = (
        collision_penetration_required
    )
    metrics = contact_geometry_metrics(
        points_corrected[frame_ids],
        model.point_link_ids_array,
        contact_ids,
        tool,
        settings,
    )
    translation_mode = metadata.get("translation_mode", "free3d")
    delta_tool = np.asarray(metadata["delta_tool"], dtype=np.float64)
    alpha = np.asarray(metadata["alpha"], dtype=np.float64)
    semantic_expected = copy.deepcopy(source)
    if translation_mode == "contact-se3":
        transform_schedule_tool = np.asarray(
            metadata.get("transform_schedule_tool"), dtype=np.float64
        )
        if transform_schedule_tool.shape != (len(corrected["opt_wrist_pos"]), 4, 4):
            raise ValueError(
                "contact-se3 correction metadata has an invalid transform_schedule_tool"
            )
        expected_wrist_pos, expected_wrist_rot, expected_joints_pos = transform_hand_trajectory(
            np.asarray(source["opt_wrist_pos"], dtype=np.float64),
            np.asarray(source["opt_wrist_rot"], dtype=np.float64),
            context.object_trajectory,
            transform_schedule_tool,
            None
            if "opt_joints_pos" not in source
            else np.asarray(source["opt_joints_pos"], dtype=np.float64),
        )
        semantic_expected["opt_wrist_pos"] = expected_wrist_pos
        semantic_expected["opt_wrist_rot"] = expected_wrist_rot
        if expected_joints_pos is not None:
            semantic_expected["opt_joints_pos"] = expected_joints_pos
        semantic_expected["opt_dof_pos"] = np.asarray(source["opt_dof_pos"]).copy()
        semantic_invariants = _invariant_report(
            source,
            semantic_expected,
            expected_wrist_pos=expected_wrist_pos,
            expected_wrist_rot=expected_wrist_rot,
            expected_joints_pos=expected_joints_pos,
        )
    else:
        delta_world = tool_to_world_offsets(delta_tool, context.object_trajectory) * alpha[:, None]
        semantic_expected["opt_wrist_pos"] = (
            np.asarray(source["opt_wrist_pos"], dtype=np.float64) + delta_world
        )
        semantic_expected["opt_wrist_rot"] = np.asarray(source["opt_wrist_rot"]).copy()
        semantic_expected["opt_dof_pos"] = np.asarray(source["opt_dof_pos"]).copy()
        if "opt_joints_pos" in source:
            semantic_expected["opt_joints_pos"] = (
                np.asarray(source["opt_joints_pos"], dtype=np.float64)
                + delta_world[:, None]
            )
        semantic_invariants = _invariant_report(source, semantic_expected, delta_world)

    physical_metadata = metadata.get("physical_correction", {})
    physical_attempted = bool(
        int(metadata.get("schema_version", 0)) >= 8
        and isinstance(physical_metadata, Mapping)
        and physical_metadata.get("attempted", False)
    )
    physical_delta_consistency = {
        "passed": True,
        "wrist_max_abs_error": 0.0,
        "dof_max_abs_error": 0.0,
    }
    mano_target_consistency = {"passed": True, "status": "not_needed"}
    if physical_attempted:
        stored_wrist_delta = np.asarray(
            corrected.get("physical_wrist_delta"), dtype=np.float64
        )
        stored_dof_delta = np.asarray(
            corrected.get("physical_dof_delta"), dtype=np.float64
        )
        expected_wrist_delta = (
            np.asarray(corrected["opt_wrist_pos"], dtype=np.float64)
            - np.asarray(semantic_expected["opt_wrist_pos"], dtype=np.float64)
        )
        expected_dof_delta = (
            np.asarray(corrected["opt_dof_pos"], dtype=np.float64)
            - np.asarray(semantic_expected["opt_dof_pos"], dtype=np.float64)
        )
        wrist_error = (
            float("inf")
            if stored_wrist_delta.shape != expected_wrist_delta.shape
            else float(np.max(np.abs(stored_wrist_delta - expected_wrist_delta)))
        )
        dof_error = (
            float("inf")
            if stored_dof_delta.shape != expected_dof_delta.shape
            else float(np.max(np.abs(stored_dof_delta - expected_dof_delta)))
        )
        physical_delta_consistency = {
            "passed": bool(wrist_error <= 1e-6 and dof_error <= 1e-6),
            "wrist_max_abs_error": wrist_error,
            "dof_max_abs_error": dof_error,
        }
        invariants = _physical_invariant_report(
            semantic_expected,
            corrected,
            np.full(len(hand.dof_names), -np.inf),
            np.full(len(hand.dof_names), np.inf),
            physical_metadata.get("active_dof_names", ()),
            hand,
        )
        stored_body_names = corrected.get("corrected_mano_body_names", ())
        try:
            expected_mano_targets, _ = _build_corrected_mano_body_targets(
                semantic_expected, corrected, model, stored_body_names
            )
            stored_mano_targets = np.asarray(
                corrected.get("corrected_mano_body_targets"), dtype=np.float64
            )
            mano_error = (
                float("inf")
                if stored_mano_targets.shape != expected_mano_targets.shape
                else float(
                    np.max(
                        np.abs(
                            stored_mano_targets
                            - np.asarray(expected_mano_targets, dtype=np.float64)
                        )
                    )
                )
            )
            opt_joints_error = (
                float("inf")
                if np.asarray(corrected.get("opt_joints_pos")).shape
                != expected_mano_targets.shape
                else float(
                    np.max(
                        np.abs(
                            np.asarray(corrected["opt_joints_pos"], dtype=np.float64)
                            - expected_mano_targets
                        )
                    )
                )
            )
            mano_target_consistency = {
                "passed": bool(mano_error <= 1e-6 and opt_joints_error <= 1e-6),
                "status": "checked",
                "maximum_target_error": mano_error,
                "opt_joints_maximum_error": opt_joints_error,
                "body_count": len(stored_body_names),
            }
        except Exception as error:
            mano_target_consistency = {
                "passed": False,
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
            }
    else:
        invariants = _invariant_report(
            source,
            corrected,
            expected_wrist_pos=np.asarray(semantic_expected["opt_wrist_pos"]),
            expected_wrist_rot=np.asarray(semantic_expected["opt_wrist_rot"]),
            expected_joints_pos=(
                None
                if "opt_joints_pos" not in semantic_expected
                else np.asarray(semantic_expected["opt_joints_pos"])
            ),
        )
    axis_constraint = None
    if translation_mode == "semantic-axis":
        stored_axis = metadata.get("semantic_axis")
        if not isinstance(stored_axis, Mapping) or "direction_tool" not in stored_axis:
            raise ValueError("semantic-axis correction metadata is missing direction_tool")
        axis_constraint = _axis_constraint_report(
            delta_tool, np.asarray(stored_axis["direction_tool"], dtype=np.float64)
        )
    collision_bundle_matches = bool(
        _urdf_bundle_sha256(collision_urdf_path)
        == metadata.get("collision_bundle_sha256")
    )
    mesh_matches = bool(_sha256(mesh_path) == metadata.get("tool_mesh_sha256"))
    source_mesh_matches = bool(
        int(metadata.get("schema_version", 0)) < 4
        or _sha256(context.source_mesh_path) == metadata.get("source_tool_mesh_sha256")
    )
    stored_validation_tiers = metadata.get("validation_tiers")
    physics_validation_required = _physics_validation_required(metadata)
    stored_tiers_passed = _stored_validation_tiers_passed(metadata)
    report = {
        "schema_version": SCHEMA_VERSION,
        "passed": bool(
            _is_feasible(metrics, settings)
            and invariants["passed"]
            and mesh_matches
            and source_mesh_matches
            and collision_bundle_matches
            and stored_tiers_passed
            and (
                not collision_penetration_required
                or collision_sdf_tier["passed"]
            )
            and semantic_invariants["passed"]
            and physical_delta_consistency["passed"]
            and mano_target_consistency["passed"]
            and (axis_constraint is None or axis_constraint["passed"])
        ),
        "corrected_pkl": corrected_path,
        "source_pkl": source_path,
        "new_obj": args.new_obj or metadata.get("new_obj"),
        "tool_mesh": mesh_path,
        "source_tool_mesh": context.source_mesh_path,
        "semantic_precheck": semantic_precheck,
        "collision_urdf": collision_urdf_path,
        "tool_mesh_matches": mesh_matches,
        "source_tool_mesh_matches": source_mesh_matches,
        "collision_bundle_matches": collision_bundle_matches,
        "stored_validation_tiers": stored_validation_tiers,
        "stored_validation_tiers_passed": stored_tiers_passed,
        "translation_mode": translation_mode,
        "source_contact_intent": metadata.get("source_contact_intent"),
        "contact_transfer": metadata.get("contact_transfer"),
        "contact_transform_tool": metadata.get("contact_transform_tool"),
        "semantic_axis": metadata.get("semantic_axis"),
        "axis_constraint": axis_constraint,
        "geometry": _summary_metrics(metrics),
        "collision_mesh_penetration": collision_sdf_tier,
        "collision_mesh_penetration_required": collision_penetration_required,
        "invariants": invariants,
        "semantic_invariants": semantic_invariants,
        "physical_delta_consistency": physical_delta_consistency,
        "mano_target_consistency": mano_target_consistency,
        "physical_correction": physical_metadata,
        "physics_validation_required": physics_validation_required,
    }
    report_path = (
        Path(args.report)
        if args.report
        else sequence_artifact_path(corrected_path, ".verify_report.json")
    )
    _atomic_json_dump(report, report_path)
    _log(f"独立复验{'通过' if report['passed'] else '失败'}；报告已写入：{report_path}")
    return report


def _add_common_geometry_arguments(
    parser: argparse.ArgumentParser, *, verification: bool = False
) -> None:
    parser.add_argument("--data-root", default=DEFAULT_OAKINK_ROOT)
    parser.add_argument("--skip", type=int, default=2)
    parser.add_argument("--points-per-link", type=int, default=32)
    parser.add_argument("--contact-distance", type=float, default=0.006)
    parser.add_argument("--black-clearance", type=float, default=0.006)
    parser.add_argument("--influence-distance", type=float, default=0.02)
    parser.add_argument("--min-red-link-ratio", type=float, default=0.1)
    parser.add_argument("--min-red-frame-ratio", type=float, default=0.8)
    # parser.add_argument("--max-black-violation-fraction", type=float, default=0.0)
    parser.add_argument("--max-black-violation-fraction", type=float, default=0.2)
    parser.add_argument("--max-optimization-frames", type=int, default=48)
    parser.add_argument(
        "--collision-sdf-max-penetration",
        type=float,
        default=None if verification else 0.001,
        help=(
            "required collision-mesh signed-penetration threshold in meters; failure blocks "
            "normal PKL saving and schema-v9 Stage-1 loading (correction default: 0.001; "
            "verification default: stored value)"
        ),
    )
    parser.add_argument(
        "--collision-sdf-surface-samples",
        type=int,
        default=None if verification else 64,
        help=(
            "deterministic bidirectional collision-surface samples per URDF link "
            "(correction default: 64; verification default: stored value)"
        ),
    )
    parser.add_argument(
        "--collision-sdf-points-per-link",
        type=int,
        default=None if verification else 2,
        help="deep SDF constraints retained per collision link and direction",
    )
    parser.add_argument(
        "--collision-sdf-max-points-per-frame",
        type=int,
        default=None if verification else 16,
        help="maximum clustered SDF constraints jointly optimized per frame",
    )
    parser.add_argument(
        "--collision-sdf-cluster-radius",
        type=float,
        default=None if verification else 0.0015,
        help="world-space radius in meters for deduplicating nearby SDF constraints",
    )
    parser.add_argument(
        "--collision-sdf-cluster-normal-degrees",
        type=float,
        default=None if verification else 20.0,
        help="maximum escape-normal angle for clustering nearby SDF constraints",
    )
    parser.add_argument("--report")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rigidly transfer a retargeted dex hand between colored semantic contact regions"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    correct = subparsers.add_parser("correct", help="solve, validate, and save a corrected retarget PKL")
    correct.add_argument("--data-idx", required=True, help="OakInk2 index, e.g. 083f7@0")
    correct.add_argument("--dexhand", default="inspire")
    correct.add_argument("--side", choices=("right", "left"), default="right")
    correct.add_argument("--input-pkl")
    correct.add_argument("--output-root", default=str(PROJECT_ROOT / "data" / "retargeting_contact"))
    correct.add_argument("--output-pkl")
    correct.add_argument(
        "--bank",
        action="store_true",
        help=(
            "retrieve verified corrections from data/retargeting_bank/OakInk-v2 "
            "and update the bank after successful validation"
        ),
    )
    correct.add_argument(
        "--overwrite",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    correct.add_argument("--mesh", help="colored new tool mesh; defaults to the dataset-selected tool")
    correct.add_argument(
        "--new-obj",
        help=(
            "replacement mesh identifier under "
            "/home/ps/data/ylj/PhysGraph/data/OakInk-v2/new_obj; selects its mesh and URDF"
        ),
    )
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
    correct.add_argument(
        "--translation-mode",
        choices=("contact-translation", "free3d", "semantic-axis"),
        default="contact-translation",
        help=(
            "contact-translation maps the old red center to the new one and locally refines "
            "translation only; free3d and semantic-axis retain the legacy solvers"
        ),
    )
    correct.add_argument(
        "--max-residual-shift",
        type=float,
        default=0.01,
        help="maximum per-axis translation refinement around the old-to-new red-center offset",
    )
    correct.add_argument(
        "--axis-roi-distance",
        type=float,
        default=0.03,
        help="tool-local radius used to select black surface points near the red region",
    )
    correct.add_argument(
        "--axis-grid-samples",
        type=int,
        default=201,
        help="number of coarse distance samples for semantic-axis one-dimensional optimization",
    )
    correct.add_argument("--optimizer-iterations", type=int, default=30)
    correct.add_argument("--max-refinement-rounds", type=int, default=4)
    correct.add_argument("--seed", type=int, default=42)
    correct.add_argument(
        "--headless",
        action="store_true",
        help="when --physx is enabled, run validation without opening a viewer",
    )
    correct.add_argument("--physics-max-frames", type=int, default=48)
    correct.add_argument("--physics-max-outer-rounds", type=int, default=8)
    correct.add_argument("--sdf-max-inner-iterations", type=int, default=20)
    correct.add_argument("--sdf-stall-patience", type=int, default=4)
    correct.add_argument(
        "--sdf-improvement-tolerance",
        type=float,
        default=1e-6,
        help="minimum maximum-penetration decrease accepted by an SDF inner step",
    )
    correct.add_argument("--physics-settle-steps", type=int, default=8)
    correct.add_argument("--physics-min-contact-frame-ratio", type=float, default=0.5)
    correct.add_argument(
        "--physics-max-contact-force",
        type=float,
        default=50000.0,
        help="fixed-pose PhysX solver-force sanity ceiling",
    )
    correct.add_argument("--physics-max-initial-overlap", type=float, default=0.01)
    correct.add_argument("--physics-timeout", type=float, default=180.0)
    physics_group = correct.add_mutually_exclusive_group()
    physics_group.add_argument(
        "--physx",
        action="store_true",
        help=(
            "run PhysX validation, repair failed penetrations with collision-SDF gradient IK, "
            "and require the final SDF/physical/Stage-1 target checks to pass before saving"
        ),
    )
    physics_group.add_argument(
        "--skip-physics-validation",
        dest="physx",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    correct.set_defaults(physx=False)
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
    verify.add_argument("--new-obj")
    _add_common_geometry_arguments(verify, verification=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "correct":
        report = run_correction(args)
    else:
        report = run_verification(args)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
