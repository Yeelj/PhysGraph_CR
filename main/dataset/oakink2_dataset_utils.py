import trimesh
import os
from typing import List
import numpy as np


SEMANTIC_REGION_POINT_COUNT = 512


def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        assert len(scene_or_mesh.geometry) != 0
        mesh = trimesh.util.concatenate(
            tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces) for g in scene_or_mesh.geometry.values())
        )
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh


def load_obj_map(obj_prefix: str, obj_list: List[str]):
    res = {}
    for obj_id in obj_list:
        obj_filedir = os.path.join(obj_prefix, obj_id)
        candidate_list = [el for el in os.listdir(obj_filedir) if os.path.splitext(el)[-1] in [".obj", ".ply"]]
        assert len(candidate_list) == 1
        obj_filename = candidate_list[0]
        obj_filepath = os.path.join(obj_filedir, obj_filename)
        mesh = load_obj_mesh(obj_filepath)
        res[obj_id] = (mesh, obj_filepath)
    return res


def load_obj_mesh(obj_filepath: str):
    suffix = os.path.splitext(str(obj_filepath))[-1].lower()
    if suffix == ".obj":
        return as_mesh(
            trimesh.load_mesh(obj_filepath, process=False, skip_materials=True, force="mesh")
        )
    return as_mesh(trimesh.load(obj_filepath, process=False, force="mesh"))


def _fixed_sample_points(points: np.ndarray, point_count: int) -> np.ndarray:
    """Deterministically sample a fixed-size point set without changing geometry."""
    if point_count <= 0:
        raise ValueError(f"point_count must be positive, got {point_count}")
    if len(points) == 0:
        raise ValueError("Cannot sample an empty semantic region")
    indices = np.linspace(0, len(points) - 1, point_count, dtype=np.int64)
    return np.asarray(points[indices], dtype=np.float32)


def extract_red_black_regions(mesh: trimesh.Trimesh, point_count: int = SEMANTIC_REGION_POINT_COUNT):
    """Extract fixed red/black vertex samples from a color-annotated tool mesh.

    Red and black are deliberately decoded independently, so both relations can
    coexist in downstream graph construction. Missing color semantics is a data
    error rather than a silent fallback to the complete mesh.
    """
    if not hasattr(mesh.visual, "vertex_colors"):
        raise ValueError("Tool mesh has no per-vertex colors for red/black contact semantics")
    colors = np.asarray(mesh.visual.vertex_colors)
    vertices = np.asarray(mesh.vertices)
    if colors.ndim != 2 or colors.shape[0] != vertices.shape[0] or colors.shape[1] < 3:
        raise ValueError(
            f"Invalid vertex color array {colors.shape}; expected ({vertices.shape[0]}, >=3)"
        )

    rgb = colors[:, :3].astype(np.int16, copy=False)
    red_mask = (rgb[:, 0] >= 200) & (rgb[:, 1] <= 40) & (rgb[:, 2] <= 40)
    black_mask = np.max(rgb, axis=1) <= 40
    red_count = int(red_mask.sum())
    black_count = int(black_mask.sum())
    if red_count == 0 or black_count == 0:
        raise ValueError(
            "Tool mesh must contain both semantic regions; "
            f"decoded red={red_count}, black={black_count} vertices"
        )

    return {
        "tool_red_points": _fixed_sample_points(vertices[red_mask], point_count),
        "tool_black_points": _fixed_sample_points(vertices[black_mask], point_count),
        "tool_red_vertex_count": red_count,
        "tool_black_vertex_count": black_count,
    }


oakink2_obj_scale = {
    "O02@0206@00001": 1.15,
    "O02@0206@00002": 0.95,
    "O02@0029@00011": 0.9,
    "O02@0029@00012": 1.2,
    "O02@0015@00020": 0.98,
    "O02@0015@00019": 1.02,
}
oakink2_obj_mass = {
    "O02@0015@00002": 0.101,
    "O02@0015@00001": 0.027,
    "C12001": 0.114,
    "O02@0030@00002": 0.0144,
    "O02@0033@00002": 0.014,
    "O02@0011@00003": 0.12,
    "O02@0206@00002": 0.163,
}  # TODO: add more objects
