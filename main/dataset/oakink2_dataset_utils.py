import trimesh
import os
from typing import List
import numpy as np


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
        if os.path.splitext(obj_filename)[-1] == ".obj":
            mesh = as_mesh(trimesh.load_mesh(obj_filepath, process=False, skip_materials=True, force="mesh"))
        else:
            mesh = trimesh.load(obj_filepath, process=False, force="mesh")
        res[obj_id] = (mesh, obj_filepath)
    return res


def _fixed_sample_points(points, n_points: int):
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        return np.zeros((n_points, 3), dtype=np.float32)
    if len(points) >= n_points:
        idx = np.linspace(0, len(points) - 1, n_points).astype(np.int64)
        return points[idx]
    repeat = int(np.ceil(n_points / len(points)))
    return np.tile(points, (repeat, 1))[:n_points]


def sample_contact_region_vertices(mesh, n_points: int = 512):
    """Sample the Stage-2 red/black tool regions from PLY vertex colors."""
    colors = getattr(getattr(mesh, "visual", None), "vertex_colors", None)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if colors is None or len(colors) != len(vertices):
        return (
            _fixed_sample_points(vertices, n_points),
            np.zeros((n_points, 3), dtype=np.float32),
            False,
            False,
            0,
            0,
        )

    colors = np.asarray(colors[:, :3], dtype=np.float32)
    if colors.max() <= 1.0:
        colors = colors * 255.0
    red_mask = (colors[:, 0] > 150.0) & (colors[:, 1] < 100.0) & (colors[:, 2] < 100.0)
    black_mask = (colors[:, 0] < 80.0) & (colors[:, 1] < 80.0) & (colors[:, 2] < 80.0)

    has_region = bool(red_mask.any())
    contact_vertices = vertices[red_mask] if has_region else vertices
    noncontact_vertices = vertices[black_mask]
    return (
        _fixed_sample_points(contact_vertices, n_points),
        _fixed_sample_points(noncontact_vertices, n_points),
        has_region,
        bool(black_mask.any()),
        int(red_mask.sum()),
        int(black_mask.sum()),
    )


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
