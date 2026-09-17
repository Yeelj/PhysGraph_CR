#!/usr/bin/env python3
import argparse
import importlib.util
import os
import tempfile
from pathlib import Path

import numpy as np
import trimesh


# Load the lightweight index helpers without importing main.dataset.__init__.
# That package auto-registers every dataset and therefore requires Isaac Gym
# and MANO even though object preprocessing itself does not need either one.
_UTILS_PATH = Path(__file__).resolve().parents[1] / "main" / "dataset" / "taco_dataset_utils.py"
_UTILS_SPEC = importlib.util.spec_from_file_location("taco_asset_preprocess_utils", _UTILS_PATH)
_UTILS = importlib.util.module_from_spec(_UTILS_SPEC)
_UTILS_SPEC.loader.exec_module(_UTILS)
build_taco_sequence_index = _UTILS.build_taco_sequence_index
taco_asset_paths = _UTILS.taco_asset_paths


MESH_SCALE = 0.01
URDF_TEMPLATE_PATH = Path(__file__).resolve().parents[1] / "assets" / "obj_urdf_example.urdf"
URDF_TEMPLATE_MESH_NAME = "102_obj.obj"


def _load_scaled_mesh(source_path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(source_path, force="mesh", process=False, skip_materials=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a triangle mesh at {source_path}, got {type(mesh).__name__}")
    mesh = mesh.copy()
    # Keep the original object origin: TACO object poses are defined in this
    # exact local frame. Only convert centimeters to meters.
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) * MESH_SCALE
    return mesh


def _write_urdf(urdf_path: str, mesh_path: str) -> None:
    content = URDF_TEMPLATE_PATH.read_text(encoding="utf-8")
    if content.count(URDF_TEMPLATE_MESH_NAME) != 2:
        raise ValueError(
            f"Unexpected OakInk-v2 URDF template contents in {URDF_TEMPLATE_PATH}: "
            f"expected two {URDF_TEMPLATE_MESH_NAME!r} mesh references"
        )
    content = content.replace(URDF_TEMPLATE_MESH_NAME, os.path.basename(mesh_path))
    Path(urdf_path).write_text(content, encoding="utf-8")


def preprocess_object(data_dir: str, object_id: str, args) -> None:
    source_path = os.path.join(data_dir, "object_models_released", f"{object_id}_cm.obj")
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Missing TACO object model: {source_path}")
    mesh_path, coacd_mesh_path, urdf_path = taco_asset_paths(data_dir, object_id)
    os.makedirs(os.path.dirname(mesh_path), exist_ok=True)
    os.makedirs(os.path.dirname(coacd_mesh_path), exist_ok=True)

    if not args.force and all(os.path.isfile(path) for path in (mesh_path, coacd_mesh_path, urdf_path)):
        print(f"[skip] {object_id}")
        return

    if args.force or not os.path.isfile(mesh_path):
        mesh = _load_scaled_mesh(source_path)
        mesh.export(mesh_path)

    coacd_updated = args.force or not os.path.isfile(coacd_mesh_path)
    if coacd_updated:
        if args.skip_coacd:
            mesh = trimesh.load(mesh_path, force="mesh", process=False, skip_materials=True)
            mesh.export(coacd_mesh_path)
        else:
            from physgraph_envs.lib.utils.coacd_process import coacd_process

            # Keep the temporary output on the same filesystem as the final
            # asset so the atomic rename cannot fail with EXDEV.
            with tempfile.TemporaryDirectory(
                prefix=f".taco_coacd_{object_id}_",
                dir=os.path.dirname(coacd_mesh_path),
            ) as temporary_dir:
                coacd_output_path = os.path.join(temporary_dir, f"{object_id}.ply")
                coacd_process(
                    mesh_path,
                    coacd_output_path,
                    quiet=args.quiet,
                    threshold=args.threshold,
                    max_convex_hull=args.max_convex_hull,
                    mcts_iteration=args.mcts_iteration,
                    mcts_max_depth=args.mcts_max_depth,
                    seed=args.seed,
                )
                if not os.path.isfile(coacd_output_path):
                    raise RuntimeError(f"CoACD failed for TACO object {object_id}")
                os.replace(coacd_output_path, coacd_mesh_path)
    if args.force or coacd_updated or not os.path.isfile(urdf_path):
        _write_urdf(urdf_path, coacd_mesh_path)
    print(f"[done] {object_id}: {urdf_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare meter-scale TACO CoACD meshes and OakInk-style URDFs")
    parser.add_argument("--data-dir", default="data/TACO")
    parser.add_argument("--object-id", action="append", default=[], help="Process one numeric TACO object ID")
    parser.add_argument(
        "--sequence-id",
        action="append",
        default=[],
        help="Process the tool and target used by a compact sequence ID, e.g. bb_bowl_0919036",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-coacd",
        action="store_true",
        help="Store the scaled source mesh without convex decomposition (debug only)",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.07)
    parser.add_argument("--max-convex-hull", type=int, default=32)
    parser.add_argument("--mcts-iteration", type=int, default=2000)
    parser.add_argument("--mcts-max-depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    records = build_taco_sequence_index(args.data_dir)
    object_ids = set(args.object_id)
    for sequence_id in args.sequence_id:
        if sequence_id not in records:
            raise KeyError(f"Unknown TACO sequence ID: {sequence_id}")
        object_ids.update((records[sequence_id].tool_id, records[sequence_id].target_id))
    if not object_ids:
        for record in records.values():
            object_ids.update((record.tool_id, record.target_id))
    for object_id in sorted(object_ids):
        preprocess_object(args.data_dir, object_id, args)


if __name__ == "__main__":
    main()
