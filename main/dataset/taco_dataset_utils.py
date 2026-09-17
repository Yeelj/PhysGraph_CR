import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation, Slerp


TACO_SEQUENCE_ID_PATTERN = re.compile(r"^[a-z0-9]{2}_[a-z0-9_]+_[0-9]{7}$")
TACO_SOURCE_FPS = 30
TACO_OUTPUT_FPS = 60
TACO_MESH_SCALE = 0.01
TACO_SEMANTIC_POINT_COUNT = 512


@dataclass(frozen=True)
class TacoObjectAssets:
    object_id: str
    semantic_mesh_path: str
    collision_mesh_path: str
    urdf_path: str


@dataclass(frozen=True)
class TacoSequenceRecord:
    sequence_id: str
    triplet: str
    sequence_name: str
    hand_pose_dir: str
    object_pose_dir: str
    tool_id: str
    target_id: str
    tool_pose_path: str
    target_pose_path: str
    tool_mesh_path: str
    target_mesh_path: str


def _slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return re.sub(r"_+", "_", value).strip("_")


def parse_taco_triplet(triplet: str) -> Tuple[str, str, str]:
    """Parse TACO's unquoted directory form, e.g. ``(brush, brush, bowl)``."""
    if not (triplet.startswith("(") and triplet.endswith(")")):
        raise ValueError(f"Invalid TACO triplet directory name: {triplet}")
    fields = tuple(part.strip() for part in triplet[1:-1].split(","))
    if len(fields) != 3 or any(not field for field in fields):
        raise ValueError(f"Invalid TACO triplet directory name: {triplet}")
    return fields


def make_taco_sequence_id(triplet: str, sequence_name: str) -> str:
    """Create the globally unique compact ID used by ManipTrans.

    The convention requested for this integration is::

        (brush, brush, bowl)/20230919_036 -> bb_bowl_0919036

    The first two letters are the action/tool initials, followed by the full
    target slug and ``MMDD + sequence number``. TACO V1 sequence names are
    globally unique; the triplet prefix also makes IDs human-readable.
    """
    action, tool, target = parse_taco_triplet(triplet)
    action_slug = _slug(action)
    tool_slug = _slug(tool)
    target_slug = _slug(target)
    if not action_slug or not tool_slug or not target_slug:
        raise ValueError(f"Cannot build an ID from TACO triplet: {triplet}")
    if re.fullmatch(r"20[0-9]{6}_[0-9]{3}", sequence_name) is None:
        raise ValueError(f"Invalid TACO sequence name: {sequence_name}")
    compact_sequence = sequence_name[4:].replace("_", "")
    return f"{action_slug[0]}{tool_slug[0]}_{target_slug}_{compact_sequence}"


def is_taco_sequence_id(index) -> bool:
    return isinstance(index, str) and TACO_SEQUENCE_ID_PATTERN.fullmatch(index) is not None


def normalize_taco_object_id(object_id) -> str:
    value = str(object_id).strip()
    if not value.isdigit() or len(value) > 3:
        raise ValueError(f"Invalid TACO object ID: {object_id!r}")
    return value.zfill(3)


def _find_role_pose(object_pose_dir: str, role: str) -> Tuple[str, str]:
    prefix = f"{role}_"
    candidates = sorted(
        filename
        for filename in os.listdir(object_pose_dir)
        if filename.startswith(prefix) and filename.endswith(".npy")
    )
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one {role} pose in {object_pose_dir}, found {candidates}"
        )
    filename = candidates[0]
    object_id = filename[len(prefix) : -len(".npy")]
    if re.fullmatch(r"[0-9]{3}", object_id) is None:
        raise ValueError(f"Invalid TACO object ID in {filename}: {object_id!r}")
    return object_id, os.path.join(object_pose_dir, filename)


@lru_cache(maxsize=None)
def build_taco_sequence_index(data_dir: str) -> Dict[str, TacoSequenceRecord]:
    data_dir = os.path.abspath(data_dir)
    hand_root = os.path.join(data_dir, "Hand_Poses")
    object_pose_root = os.path.join(data_dir, "Object_Poses")
    object_model_root = os.path.join(data_dir, "object_models_released")
    for required_dir in (hand_root, object_pose_root, object_model_root):
        if not os.path.isdir(required_dir):
            raise FileNotFoundError(f"Required TACO directory does not exist: {required_dir}")

    records: Dict[str, TacoSequenceRecord] = {}
    for triplet in sorted(os.listdir(hand_root)):
        triplet_hand_dir = os.path.join(hand_root, triplet)
        if not os.path.isdir(triplet_hand_dir):
            continue
        parse_taco_triplet(triplet)
        for sequence_name in sorted(os.listdir(triplet_hand_dir)):
            hand_pose_dir = os.path.join(triplet_hand_dir, sequence_name)
            if not os.path.isdir(hand_pose_dir):
                continue
            sequence_id = make_taco_sequence_id(triplet, sequence_name)
            object_pose_dir = os.path.join(object_pose_root, triplet, sequence_name)
            if not os.path.isdir(object_pose_dir):
                raise FileNotFoundError(f"Missing TACO object poses: {object_pose_dir}")

            tool_id, tool_pose_path = _find_role_pose(object_pose_dir, "tool")
            target_id, target_pose_path = _find_role_pose(object_pose_dir, "target")
            tool_mesh_path = os.path.join(object_model_root, f"{tool_id}_cm.obj")
            target_mesh_path = os.path.join(object_model_root, f"{target_id}_cm.obj")
            for mesh_path in (tool_mesh_path, target_mesh_path):
                if not os.path.isfile(mesh_path):
                    raise FileNotFoundError(f"Missing TACO object model: {mesh_path}")

            record = TacoSequenceRecord(
                sequence_id=sequence_id,
                triplet=triplet,
                sequence_name=sequence_name,
                hand_pose_dir=hand_pose_dir,
                object_pose_dir=object_pose_dir,
                tool_id=tool_id,
                target_id=target_id,
                tool_pose_path=tool_pose_path,
                target_pose_path=target_pose_path,
                tool_mesh_path=tool_mesh_path,
                target_mesh_path=target_mesh_path,
            )
            if sequence_id in records:
                other = records[sequence_id]
                raise ValueError(
                    f"Duplicate TACO sequence ID {sequence_id}: "
                    f"{other.triplet}/{other.sequence_name} and {triplet}/{sequence_name}"
                )
            records[sequence_id] = record

    if not records:
        raise RuntimeError(f"No TACO sequences found under {hand_root}")
    return records


def taco_asset_paths(data_dir: str, object_id: str) -> Tuple[str, str, str]:
    object_id = normalize_taco_object_id(object_id)
    object_dir = os.path.join(data_dir, "object_models", object_id)
    coacd_dir = os.path.join(data_dir, "coacd_object_models", object_id)
    mesh_path = os.path.join(object_dir, f"{object_id}.ply")
    coacd_mesh_path = os.path.join(coacd_dir, f"{object_id}.ply")
    urdf_path = os.path.join(coacd_dir, f"{object_id}.urdf")
    return mesh_path, coacd_mesh_path, urdf_path


def resolve_taco_object_assets(data_dir: str, object_id: str) -> TacoObjectAssets:
    """Resolve one existing meter-scale TACO semantic/collision asset bundle."""

    object_id = normalize_taco_object_id(object_id)
    mesh_path, collision_mesh_path, urdf_path = taco_asset_paths(data_dir, object_id)
    missing = [
        path
        for path in (mesh_path, collision_mesh_path, urdf_path)
        if not os.path.isfile(path)
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing processed TACO assets for object {object_id}: {missing}"
        )
    return TacoObjectAssets(
        object_id=object_id,
        semantic_mesh_path=os.path.abspath(mesh_path),
        collision_mesh_path=os.path.abspath(collision_mesh_path),
        urdf_path=os.path.abspath(urdf_path),
    )


def load_taco_meter_mesh(mesh_path: str) -> trimesh.Trimesh:
    """Load a processed TACO mesh which is already expressed in meters."""

    mesh = trimesh.load(mesh_path, force="mesh", process=False, skip_materials=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a triangle mesh at {mesh_path}, got {type(mesh).__name__}")
    return mesh


def load_taco_source_mesh(mesh_path: str) -> trimesh.Trimesh:
    """Load a released ``*_cm.obj`` mesh and convert centimeters to meters."""

    mesh = load_taco_meter_mesh(mesh_path).copy()
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float32) * TACO_MESH_SCALE
    return mesh


def _fixed_semantic_points(points: np.ndarray, point_count: int) -> np.ndarray:
    if point_count <= 0:
        raise ValueError(f"point_count must be positive, got {point_count}")
    if len(points) == 0:
        raise ValueError("Cannot sample an empty TACO semantic region")
    indices = np.linspace(0, len(points) - 1, point_count, dtype=np.int64)
    return np.asarray(points[indices], dtype=np.float32)


def extract_taco_red_black_regions(
    mesh: trimesh.Trimesh,
    point_count: int = TACO_SEMANTIC_POINT_COUNT,
) -> Dict[str, np.ndarray]:
    """Decode TACO's per-vertex red/black grasp semantics.

    TACO intentionally owns this decoder instead of importing the OakInk
    helper.  Both datasets expose the same Stage-1 field contract.
    """

    colors = getattr(getattr(mesh, "visual", None), "vertex_colors", None)
    vertices = np.asarray(mesh.vertices)
    if colors is None:
        raise ValueError("TACO tool mesh has no per-vertex red/black colors")
    colors = np.asarray(colors)
    if colors.ndim != 2 or colors.shape[0] != len(vertices) or colors.shape[1] < 3:
        raise ValueError(
            f"Invalid TACO vertex color array {colors.shape}; expected ({len(vertices)}, >=3)"
        )
    rgb = colors[:, :3].astype(np.int16, copy=False)
    red_mask = (rgb[:, 0] >= 200) & (rgb[:, 1] <= 40) & (rgb[:, 2] <= 40)
    black_mask = np.max(rgb, axis=1) <= 40
    unclassified = ~(red_mask | black_mask)
    if not red_mask.any() or not black_mask.any() or unclassified.any():
        raise ValueError(
            "TACO tool mesh must be completely segmented into red and black vertices; "
            f"red={int(red_mask.sum())}, black={int(black_mask.sum())}, "
            f"unclassified={int(unclassified.sum())}, vertices={len(vertices)}"
        )
    return {
        "tool_red_points": _fixed_semantic_points(vertices[red_mask], point_count),
        "tool_black_points": _fixed_semantic_points(vertices[black_mask], point_count),
        "tool_red_vertex_count": int(red_mask.sum()),
        "tool_black_vertex_count": int(black_mask.sum()),
    }


def resample_taco_pose_trajectory(
    poses: np.ndarray,
    source_fps: int = TACO_SOURCE_FPS,
    target_fps: int = TACO_OUTPUT_FPS,
) -> np.ndarray:
    """Resample a rigid TACO trajectory while preserving its duration."""

    poses = np.asarray(poses, dtype=np.float64)
    if source_fps == target_fps:
        return poses.copy()
    if target_fps % source_fps != 0:
        raise ValueError(f"Only integer upsampling is supported, got {source_fps} -> {target_fps} FPS")
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 2:
        raise ValueError(f"Expected at least two [4,4] TACO poses, got {poses.shape}")
    target_length = (len(poses) - 1) * (target_fps // source_fps) + 1
    source_time = np.arange(len(poses), dtype=np.float64) / source_fps
    target_time = np.arange(target_length, dtype=np.float64) / target_fps
    translations = np.stack(
        [np.interp(target_time, source_time, poses[:, axis, 3]) for axis in range(3)],
        axis=1,
    )
    rotations = Slerp(source_time, Rotation.from_matrix(poses[:, :3, :3]))(
        target_time
    ).as_matrix()
    result = np.repeat(np.eye(4, dtype=np.float64)[None], target_length, axis=0)
    result[:, :3, :3] = rotations
    result[:, :3, 3] = translations
    return result


def taco_normalization_transform(
    tool_trajectory: np.ndarray,
    target_trajectory: np.ndarray,
    tool_mesh: trimesh.Trimesh,
    target_mesh: trimesh.Trimesh,
    *,
    workspace_center=(-0.1, 0.0),
    floor_clearance: float = 0.003,
) -> np.ndarray:
    """Return the stable sequence transform shared by retarget, correct and Stage 1."""

    first_frame_count = min(5, len(tool_trajectory))
    object_xy = np.concatenate(
        [
            np.asarray(tool_trajectory)[:first_frame_count, :2, 3],
            np.asarray(target_trajectory)[:first_frame_count, :2, 3],
        ],
        axis=0,
    )
    xy_anchor = np.median(object_xy, axis=0)
    min_z = np.inf
    for trajectory, mesh in (
        (tool_trajectory, tool_mesh),
        (target_trajectory, target_mesh),
    ):
        for pose in np.asarray(trajectory)[:first_frame_count]:
            vertices = np.asarray(mesh.vertices) @ pose[:3, :3].T + pose[:3, 3]
            min_z = min(min_z, float(vertices[:, 2].min()))
    workspace_center = np.asarray(workspace_center, dtype=np.float64)
    if workspace_center.shape != (2,):
        raise ValueError(f"workspace_center must contain two values, got {workspace_center}")
    transform = np.eye(4, dtype=np.float64)
    transform[:2, 3] = workspace_center - xy_anchor
    transform[2, 3] = float(floor_clearance) - min_z
    return transform
