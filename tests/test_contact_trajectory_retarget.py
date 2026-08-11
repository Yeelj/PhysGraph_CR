import runpy
import hashlib
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import trimesh
import torch
from scipy.spatial import cKDTree


MODULE = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "main" / "dataset" / "contact_trajectory_retarget.py"),
    run_name="contact_trajectory_retarget",
)


def load_base_module_without_training_dependencies():
    """Load base.py with tiny stubs for optional rendering dependencies."""

    transform = types.ModuleType("main.dataset.transform")
    transform.aa_to_rotmat = lambda value: value
    transform.rotmat_to_aa = lambda value: value
    transform.caculate_align_mat = lambda value: value
    pytorch3d_ops = types.ModuleType("pytorch3d.ops")
    pytorch3d_ops.sample_points_from_meshes = lambda *args, **kwargs: None
    termcolor = types.ModuleType("termcolor")
    termcolor.cprint = lambda *args, **kwargs: None
    replacements = {
        "main.dataset.transform": transform,
        "pytorch3d.ops": pytorch3d_ops,
        "termcolor": termcolor,
    }
    previous = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    try:
        return runpy.run_path(
            str(Path(__file__).resolve().parents[1] / "main" / "dataset" / "base.py"),
            run_name="dataset_base_for_test",
        )
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


class SquaredChamfer:
    def __call__(self, source, target):
        squared = ((source[:, :, None] - target[:, None]) ** 2).sum(dim=-1)
        return squared.min(dim=-1).values, None, None, None


class ContactTrajectoryRetargetTest(unittest.TestCase):
    def test_non_pkl_artifacts_use_sequence_directory(self):
        pkl_path = Path("data/retargeting_contact/OakInk-v2/mano2artimano_rh/demo@0.pkl")
        report_path = MODULE["sequence_artifact_path"](pkl_path, ".contact_report.json")
        preview_path = MODULE["sequence_artifact_path"](pkl_path, ".contact_preview.png")
        verify_path = MODULE["sequence_artifact_path"](pkl_path, ".verify_report.json")
        expected_dir = pkl_path.parent / "demo@0"
        self.assertEqual(report_path, expected_dir / "demo@0.contact_report.json")
        self.assertEqual(preview_path, expected_dir / "demo@0.contact_preview.png")
        self.assertEqual(verify_path, expected_dir / "demo@0.verify_report.json")

    def test_schedule_is_smooth_and_has_hold_interval(self):
        alpha = MODULE["build_offset_schedule"](30, 10, 19, 5, 5)
        self.assertTrue(np.all(alpha[:5] == 0.0))
        self.assertTrue(np.all(alpha[10:20] == 1.0))
        self.assertTrue(np.all(alpha[25:] == 0.0))
        self.assertTrue(np.all(np.diff(alpha[5:11]) >= 0.0))
        self.assertTrue(np.all(np.diff(alpha[19:25]) <= 0.0))

    def test_optimizer_moves_contact_from_black_to_red(self):
        red = np.array(
            [[0.02, y, z] for y in (-0.005, 0.0, 0.005) for z in (-0.005, 0.0, 0.005)]
        )
        black = np.array(
            [[-0.02, y, z] for y in (-0.005, 0.0, 0.005) for z in (-0.005, 0.0, 0.005)]
        )
        semantic_tool = MODULE["SemanticTool"](
            mesh=trimesh.creation.box(),
            red_points=red,
            black_points=black,
            red_tree=cKDTree(red),
            black_tree=cKDTree(black),
            red_vertex_count=len(red),
            black_vertex_count=len(black),
        )
        one_frame = np.array(
            [[-0.019, y, z] for y in (-0.002, 0.0, 0.002) for z in (-0.002, 0.0, 0.002)]
            + [[0.0, 0.04, 0.0]]
        )
        points = np.repeat(one_frame[None], 8, axis=0)
        point_link_ids = np.array([0] * 9 + [1])
        contact_link_ids = np.array([0])
        settings = MODULE["GeometrySettings"](
            contact_distance=0.004,
            black_clearance=0.004,
            influence_distance=0.01,
            min_red_link_ratio=1.0,
            min_red_frame_ratio=0.8,
            max_shift=0.05,
        )
        delta, _ = MODULE["optimize_tool_translation"](
            points,
            point_link_ids,
            contact_link_ids,
            semantic_tool,
            settings,
            max_iterations=8,
        )
        metrics = MODULE["contact_geometry_metrics"](
            points + delta[None, None],
            point_link_ids,
            contact_link_ids,
            semantic_tool,
            settings,
        )
        self.assertGreater(delta[0], 0.03)
        self.assertTrue(MODULE["_is_feasible"](metrics, settings))

    def test_artimano_urdf_samples_all_collision_links(self):
        hand = MODULE["_resolve_dexhand"]("artimano", "right")
        model = MODULE["URDFCollisionKinematics"](
            hand.urdf_path, hand.dof_names, points_per_link=8
        )
        MODULE["_contact_link_ids"](model, hand.contact_body_names)
        points = model.root_frame_surface_points(np.zeros(len(hand.dof_names)))
        self.assertEqual(points.shape[1], 3)
        self.assertGreaterEqual(len(model.link_points), len(hand.contact_body_names))
        self.assertTrue(np.isfinite(points).all())

    def test_invariants_require_only_rigid_translation(self):
        source = {
            "opt_wrist_pos": np.zeros((4, 3), dtype=np.float32),
            "opt_wrist_rot": np.ones((4, 3), dtype=np.float32),
            "opt_dof_pos": np.ones((4, 2), dtype=np.float32),
            "opt_joints_pos": np.zeros((4, 3, 3), dtype=np.float32),
        }
        delta = np.arange(12, dtype=np.float32).reshape(4, 3) * 0.001
        corrected = {
            "opt_wrist_pos": source["opt_wrist_pos"] + delta,
            "opt_wrist_rot": source["opt_wrist_rot"].copy(),
            "opt_dof_pos": source["opt_dof_pos"].copy(),
            "opt_joints_pos": source["opt_joints_pos"] + delta[:, None],
        }
        report = MODULE["_invariant_report"](source, corrected, delta)
        self.assertTrue(report["passed"])

    def test_stage1_targets_receive_exact_same_rigid_offset(self):
        base_module = load_base_module_without_training_dependencies()
        class TestManipData(base_module["ManipData"]):
            def __getitem__(self, index):
                raise NotImplementedError

        dataset = object.__new__(TestManipData)
        dataset.dexhand = types.SimpleNamespace(side="rh")
        dataset.device = "cpu"
        dataset.skip = 2
        dataset.ch_dist = SquaredChamfer()

        length = 4
        trajectory = torch.eye(4).repeat(length, 1, 1)
        trajectory[:, :3, :3] = torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]],
                [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            ]
        )
        joint_names = [
            "thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip",
            "index_proximal",
        ]
        original_wrist = torch.zeros(length, 3)
        data = {
            "obj_trajectory": trajectory,
            "obj_verts": torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
            "wrist_pos": original_wrist.clone(),
            "wrist_velocity": original_wrist.clone(),
            "mano_joints": {name: original_wrist.clone() for name in joint_names},
            "mano_joints_velocity": {name: original_wrist.clone() for name in joint_names},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            mesh_file = temp_dir / "tool.ply"
            mesh_file.write_bytes(b"same mesh used by correction and stage1")
            collision_mesh = temp_dir / "collision.ply"
            collision_mesh.write_bytes(b"matching physical collision mesh")
            collision_urdf = temp_dir / "collision.urdf"
            collision_urdf.write_text(
                '<robot name="tool"><link name="base"><collision><geometry>'
                '<mesh filename="collision.ply"/></geometry></collision></link></robot>'
            )
            data["obj_mesh_path"] = str(mesh_file)
            data["obj_urdf_path"] = str(collision_urdf)
            mesh_hash = hashlib.sha256(mesh_file.read_bytes()).hexdigest()
            metadata = {
                "geometry_feasible": True,
                "side": "right",
                "tool_mesh": str(mesh_file),
                "tool_mesh_sha256": mesh_hash,
                "collision_bundle_sha256": dataset._urdf_bundle_sha256(collision_urdf),
                "delta_tool": np.array([0.01, 0.0, 0.0], dtype=np.float32),
                "alpha": np.array([0.0, 0.5, 1.0, 0.0], dtype=np.float32),
            }
            dataset._apply_contact_retarget_targets(data, metadata, "corrected.pkl")

        expected = torch.matmul(
            trajectory[:, :3, :3], torch.tensor([[0.01], [0.0], [0.0]])
        ).squeeze(-1) * torch.tensor([0.0, 0.5, 1.0, 0.0])[:, None]
        self.assertTrue(torch.allclose(data["wrist_pos"], expected))
        for name in joint_names:
            self.assertTrue(torch.allclose(data["mano_joints"][name], expected))
        self.assertEqual(data["tips_distance"].shape, (length, 5))


if __name__ == "__main__":
    unittest.main()
