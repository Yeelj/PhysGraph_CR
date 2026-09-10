import runpy
import hashlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import trimesh
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from main.new_object_paths import object_asset_cache_key, replacement_object_for_side


MODULE = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "main" / "dataset" / "contact_trajectory_retarget.py"),
    run_name="contact_trajectory_retarget",
)


def load_base_module_without_training_dependencies():
    """Load base.py with tiny stubs for optional rendering dependencies."""

    transform = types.ModuleType("main.dataset.transform")
    def aa_to_rotmat(value):
        shape = tuple(value.shape[:-1])
        matrices = Rotation.from_rotvec(
            value.detach().cpu().numpy().reshape(-1, 3)
        ).as_matrix().reshape(*shape, 3, 3)
        return torch.as_tensor(matrices, dtype=value.dtype, device=value.device)

    def rotmat_to_aa(value):
        shape = tuple(value.shape[:-2])
        vectors = Rotation.from_matrix(
            value.detach().cpu().numpy().reshape(-1, 3, 3)
        ).as_rotvec().reshape(*shape, 3)
        return torch.as_tensor(vectors, dtype=value.dtype, device=value.device)

    transform.aa_to_rotmat = aa_to_rotmat
    transform.rotmat_to_aa = rotmat_to_aa
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
    def test_stage1_rejects_swapped_original_and_corrected_trajectories(self):
        base_module = load_base_module_without_training_dependencies()
        validate_source = base_module["_validate_retargeting_source"]
        metadata = {"side": "right"}

        self.assertEqual(
            validate_source(metadata, "right.pkl", "contact_corrected"),
            "contact_corrected",
        )
        self.assertEqual(validate_source(None, "left.pkl", "original"), "original")
        with self.assertRaisesRegex(ValueError, "Expected contact_corrected"):
            validate_source(None, "right.pkl", "contact_corrected")
        with self.assertRaisesRegex(ValueError, "Expected original"):
            validate_source(metadata, "left.pkl", "original")

    def test_stage1_replacement_is_scoped_to_right_hand_tool(self):
        self.assertEqual(replacement_object_for_side("spoon_01", "right"), "spoon_01")
        self.assertEqual(replacement_object_for_side("spoon_01", "rh"), "spoon_01")
        self.assertIsNone(replacement_object_for_side("spoon_01", "left"))
        self.assertIsNone(replacement_object_for_side("spoon_01", "lh"))
        with self.assertRaisesRegex(ValueError, "Unsupported hand side"):
            replacement_object_for_side("spoon_01", "centre")

    def test_isaac_asset_cache_separates_same_id_with_different_urdfs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            right_urdf = root / "tool.urdf"
            left_urdf = root / "object.urdf"
            right_urdf.write_text("<robot name='tool'/>", encoding="utf-8")
            left_urdf.write_text("<robot name='object'/>", encoding="utf-8")
            self.assertNotEqual(
                object_asset_cache_key("shared-id", right_urdf),
                object_asset_cache_key("shared-id", left_urdf),
            )

    def test_semantic_mesh_precheck_requires_complete_red_black_coloring(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            mesh_path = Path(temp_dir) / "semantic.ply"
            mesh = trimesh.creation.box()
            colors = np.zeros((len(mesh.vertices), 4), dtype=np.uint8)
            colors[:, 3] = 255
            colors[np.asarray(mesh.vertices)[:, 0] >= 0, 0] = 255
            mesh.visual.vertex_colors = colors
            mesh.export(mesh_path)
            report = MODULE["validate_semantic_mesh"](mesh_path, label="test")
            self.assertTrue(report["passed"])
            self.assertEqual(report["unclassified_vertex_count"], 0)

            colors[0, :3] = 128
            mesh.visual.vertex_colors = colors
            mesh.export(mesh_path)
            with self.assertRaisesRegex(ValueError, "unclassified=1"):
                MODULE["validate_semantic_mesh"](mesh_path, label="test")

    def test_contact_acceptance_uses_all_configured_contact_links(self):
        red = np.array([[0.0, 0.0, 0.0], [0.0, 0.02, 0.0]])
        black = np.array([[0.1, 0.0, 0.0]])
        tool = MODULE["SemanticTool"](
            mesh=trimesh.creation.box(),
            red_points=red,
            black_points=black,
            red_tree=cKDTree(red),
            black_tree=cKDTree(black),
            red_vertex_count=len(red),
            black_vertex_count=len(black),
        )
        frames = 10
        points = np.repeat(
            np.array(
                [[[0.0, 0.0, 0.001], [0.0, 0.02, 0.001],
                  [0.0, 0.05, 0.02], [0.0, 0.07, 0.02], [0.0, 0.09, 0.02]]]
            ),
            frames,
            axis=0,
        )
        settings = MODULE["GeometrySettings"](
            contact_distance=0.006,
            min_red_link_ratio=0.6,
            min_red_frame_ratio=0.8,
        )
        metrics = MODULE["contact_geometry_metrics"](
            points,
            np.arange(5),
            np.arange(5),
            tool,
            settings,
        )
        self.assertAlmostEqual(metrics["red_link_ratio_mean"], 0.4)
        self.assertEqual(metrics["red_frame_pass_ratio"], 0.0)
        self.assertFalse(MODULE["_is_feasible"](metrics, settings))

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

    def test_refinement_frame_sampling_has_a_hard_cost_bound(self):
        sampled = MODULE["_bounded_frame_sample"](np.arange(1300), 48)
        self.assertEqual(len(sampled), 48)
        self.assertEqual(sampled[0], 0)
        self.assertEqual(sampled[-1], 1299)

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

    def test_semantic_axis_points_from_local_black_toward_red(self):
        red = np.array(
            [[0.02, y, z] for y in (-0.005, 0.0, 0.005) for z in (-0.005, 0.0, 0.005)]
        )
        black = np.array(
            [[-0.02, y, z] for y in (-0.005, 0.0, 0.005) for z in (-0.005, 0.0, 0.005)]
            + [[-0.20, y, z] for y in (-0.08, 0.08) for z in (-0.08, 0.08)]
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
        direction, report = MODULE["estimate_semantic_axis"](
            semantic_tool, roi_distance=0.05
        )
        self.assertGreater(direction[0], 0.99)
        self.assertLess(np.linalg.norm(direction[1:]), 1e-6)
        self.assertEqual(report["local_black_point_count"], 9)
        self.assertFalse(report["local_black_fallback_used"])

    def test_semantic_axis_optimizer_uses_only_nonnegative_axis_distance(self):
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
        direction, _ = MODULE["estimate_semantic_axis"](
            semantic_tool, roi_distance=0.05
        )
        delta, optimizer_report = MODULE["optimize_tool_translation_along_axis"](
            points,
            point_link_ids,
            contact_link_ids,
            semantic_tool,
            settings,
            direction,
            grid_samples=101,
        )
        metrics = MODULE["contact_geometry_metrics"](
            points + delta[None, None],
            point_link_ids,
            contact_link_ids,
            semantic_tool,
            settings,
        )
        self.assertGreater(delta[0], 0.03)
        self.assertLessEqual(np.linalg.norm(delta), settings.max_shift + 1e-9)
        self.assertLess(np.linalg.norm(delta[1:]), 1e-8)
        self.assertGreaterEqual(optimizer_report["axis_distance"], 0.0)
        self.assertTrue(optimizer_report["axis_constraint"]["passed"])
        self.assertTrue(MODULE["_is_feasible"](metrics, settings))

    def test_contact_translation_moves_old_red_center_to_new_red_center(self):
        old_red = np.array(
            [[x, y, 0.02] for x in (-0.02, 0.0, 0.02) for y in (-0.006, 0.006)]
        )
        new_red = np.array(
            [[x, y, -0.03] for x in (-0.02, 0.0, 0.02) for y in (-0.006, 0.006)]
        )
        black = np.array([[0.2, 0.0, 0.0]])

        def semantic_tool(red):
            return MODULE["SemanticTool"](
                mesh=trimesh.creation.box(extents=(0.08, 0.04, 0.04)),
                red_points=red,
                black_points=black,
                red_tree=cKDTree(red),
                black_tree=cKDTree(black),
                red_vertex_count=len(red),
                black_vertex_count=len(black),
            )

        delta, report = MODULE["contact_translation_transfer"](
            semantic_tool(old_red), semantic_tool(new_red)
        )
        self.assertTrue(np.allclose(delta, [0.0, 0.0, -0.05], atol=1e-8))
        self.assertEqual(report["method"], "red_center_translation")
        self.assertEqual(report["rotation_degrees"], 0.0)
        old_hand_point = np.array([0.0, 0.0, 0.021])
        self.assertTrue(np.allclose(old_hand_point + delta, [0.0, 0.0, -0.029]))

    def test_se3_schedule_transforms_wrist_and_joints_but_not_articulation(self):
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("x", 180, degrees=True).as_matrix()
        transform[:3, 3] = np.array([0.0, 0.0, -0.01])
        alpha = np.array([0.0, 1.0])
        schedule = MODULE["build_transform_schedule"](transform, alpha)
        object_trajectory = np.repeat(np.eye(4)[None], 2, axis=0)
        wrist_pos = np.array([[0.0, 0.0, 0.02], [0.0, 0.0, 0.02]])
        wrist_rot = np.zeros((2, 3))
        joints = np.array(
            [[[0.0, 0.0, 0.03]], [[0.0, 0.0, 0.03]]], dtype=np.float64
        )
        corrected_pos, corrected_rot, corrected_joints = MODULE[
            "transform_hand_trajectory"
        ](wrist_pos, wrist_rot, object_trajectory, schedule, joints)
        self.assertTrue(np.allclose(corrected_pos[0], wrist_pos[0]))
        self.assertTrue(np.allclose(corrected_pos[1], [0.0, 0.0, -0.03]))
        self.assertAlmostEqual(np.linalg.norm(corrected_rot[1]), np.pi, places=7)
        self.assertTrue(np.allclose(corrected_joints[1, 0], [0.0, 0.0, -0.04]))

    def test_contact_translation_optimizer_preserves_fixed_contact_link_acceptance(self):
        old_red = np.array(
            [[x, y, 0.02] for x in (-0.02, 0.0, 0.02) for y in (-0.006, 0.006)]
        )
        new_red = np.array(
            [[x, y, -0.03] for x in (-0.02, 0.0, 0.02) for y in (-0.006, 0.006)]
        )
        black = np.array([[0.2, 0.0, 0.0]])

        def semantic_tool(red):
            return MODULE["SemanticTool"](
                mesh=trimesh.creation.box(extents=(0.08, 0.04, 0.04)),
                red_points=red,
                black_points=black,
                red_tree=cKDTree(red),
                black_tree=cKDTree(black),
                red_vertex_count=len(red),
                black_vertex_count=len(black),
            )

        source_tool = semantic_tool(old_red)
        target_tool = semantic_tool(new_red)
        initial, _ = MODULE["contact_translation_transfer"](source_tool, target_tool)
        source_points = np.repeat(old_red[:5][None], 4, axis=0)
        settings = MODULE["GeometrySettings"](
            contact_distance=0.002,
            black_clearance=0.002,
            min_red_link_ratio=0.6,
            min_red_frame_ratio=0.8,
        )
        delta, _ = MODULE["optimize_contact_translation"](
            source_points,
            np.arange(5),
            np.arange(5),
            target_tool,
            settings,
            initial,
            max_iterations=4,
        )
        transformed = source_points + delta[None, None]
        metrics = MODULE["contact_geometry_metrics"](
            transformed, np.arange(5), np.arange(5), target_tool, settings
        )
        self.assertEqual(metrics["red_link_ratio_mean"], 1.0)
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

    def test_correct_command_recomputes_without_overwrite_flag(self):
        args = MODULE["build_parser"]().parse_args(
            [
                "correct",
                "--data-idx",
                "9fc3e@0",
                "--dexhand",
                "artimano",
                "--new-obj",
                "spoon_01",
                "--headless",
            ]
        )
        self.assertFalse(args.overwrite)
        self.assertFalse(args.bank)
        self.assertEqual(args.new_obj, "spoon_01")
        self.assertEqual(args.translation_mode, "contact-translation")
        self.assertFalse(hasattr(args, "max_residual_rotation_deg"))
        self.assertFalse(hasattr(args, "precise_max_frames"))
        self.assertEqual(args.physics_max_contact_force, 50000.0)
        self.assertFalse(args.physx)
        self.assertFalse(hasattr(args, "show_original_hand"))

        physx_args = MODULE["build_parser"]().parse_args(
            ["correct", "--data-idx", "9fc3e@0", "--physx", "--headless"]
        )
        self.assertTrue(physx_args.physx)

        bank_args = MODULE["build_parser"]().parse_args(
            [
                "correct",
                "--data-idx",
                "9fc3e@0",
                "--new-obj",
                "kni@1",
                "--bank",
            ]
        )
        self.assertTrue(bank_args.bank)

    def test_optional_physx_is_not_part_of_configured_acceptance_gate(self):
        tiers = {
            "semantic_geometry": {"passed": True},
            "isaac_gym_physics": {
                "status": "not_requested",
                "passed": False,
                "required": False,
            },
        }
        optional = {
            "schema_version": 5,
            "physics_validation_required": False,
            "validation_tiers": tiers,
        }
        required = {
            "schema_version": 5,
            "physics_validation_required": True,
            "validation_tiers": tiers,
        }
        legacy = {
            "schema_version": 4,
            "physics_validation_required": False,
            "validation_tiers": tiers,
        }
        self.assertTrue(MODULE["_stored_validation_tiers_passed"](optional))
        self.assertFalse(MODULE["_stored_validation_tiers_passed"](required))
        self.assertTrue(MODULE["_stored_validation_tiers_passed"](legacy))

        base_module = load_base_module_without_training_dependencies()
        stage1_gate = base_module["_contact_retarget_validation_tiers_passed"]
        self.assertTrue(stage1_gate(optional))
        self.assertFalse(stage1_gate(required))
        self.assertTrue(stage1_gate(legacy))

    def test_schema8_physx_requires_physical_and_mano_tiers(self):
        metadata = {
            "schema_version": 8,
            "physics_validation_required": True,
            "validation_tiers": {
                "semantic_geometry": {"passed": True},
                "isaac_gym_physics": {"passed": True},
            },
        }
        self.assertFalse(MODULE["_stored_validation_tiers_passed"](metadata))
        base_module = load_base_module_without_training_dependencies()
        self.assertFalse(
            base_module["_contact_retarget_validation_tiers_passed"](metadata)
        )
        metadata["validation_tiers"].update(
            physical_optimization={"passed": True},
            mano_target_sync={"passed": True},
        )
        self.assertTrue(MODULE["_stored_validation_tiers_passed"](metadata))
        self.assertTrue(
            base_module["_contact_retarget_validation_tiers_passed"](metadata)
        )

    def test_removed_show_original_hand_option_is_rejected(self):
        with self.assertRaises(SystemExit):
            MODULE["build_parser"]().parse_args(
                [
                    "correct",
                    "--data-idx",
                    "9fc3e@0",
                    "--dexhand",
                    "artimano",
                    "--show-original-hand",
                    "true",
                ]
            )

    def test_viewer_mode_reports_missing_display_explicitly(self):
        previous = os.environ.pop("DISPLAY", None)
        try:
            report = MODULE["run_isaac_gym_validation"](
                corrected={},
                hand=None,
                context=None,
                contact_names=(),
                hold_frame_ids=np.array([0]),
                headless=False,
                maximum_frames=1,
                settle_steps=1,
                min_contact_frame_ratio=0.5,
                max_contact_force=1.0,
                max_initial_overlap=0.01,
                timeout=1.0,
            )
        finally:
            if previous is not None:
                os.environ["DISPLAY"] = previous
        self.assertFalse(report["passed"])
        self.assertEqual(report["status"], "error")
        self.assertIn("DISPLAY", report["error"])

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
                "schema_version": 5,
                "geometry_feasible": True,
                "side": "right",
                "semantic_precheck": {"passed": True},
                "physics_validation_required": True,
                "tool_mesh": str(mesh_file),
                "tool_mesh_sha256": mesh_hash,
                "collision_bundle_sha256": dataset._urdf_bundle_sha256(collision_urdf),
                "delta_tool": np.array([0.01, 0.0, 0.0], dtype=np.float32),
                "alpha": np.array([0.0, 0.5, 1.0, 0.0], dtype=np.float32),
                "validation_tiers": {
                    name: {"passed": True}
                    for name in ("semantic_geometry", "isaac_gym_physics")
                },
            }
            dataset._apply_contact_retarget_targets(data, metadata, "corrected.pkl")

        expected = torch.matmul(
            trajectory[:, :3, :3], torch.tensor([[0.01], [0.0], [0.0]])
        ).squeeze(-1) * torch.tensor([0.0, 0.5, 1.0, 0.0])[:, None]
        self.assertTrue(torch.allclose(data["wrist_pos"], expected))
        for name in joint_names:
            self.assertTrue(torch.allclose(data["mano_joints"][name], expected))
        self.assertEqual(data["tips_distance"].shape, (length, 5))

    def test_stage1_targets_receive_contact_se3_schedule(self):
        base_module = load_base_module_without_training_dependencies()

        class TestManipData(base_module["ManipData"]):
            def __getitem__(self, index):
                raise NotImplementedError

        dataset = object.__new__(TestManipData)
        dataset.dexhand = types.SimpleNamespace(side="rh")
        dataset.device = "cpu"
        dataset.skip = 2
        dataset.ch_dist = SquaredChamfer()
        length = 3
        trajectory = torch.eye(4).repeat(length, 1, 1)
        wrist = torch.tensor([[0.0, 0.0, 0.02]]).repeat(length, 1)
        joint_names = [
            "thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"
        ]
        data = {
            "obj_trajectory": trajectory,
            "obj_verts": torch.tensor([[0.0, 0.0, -0.03], [0.01, 0.0, -0.03]]),
            "wrist_pos": wrist.clone(),
            "wrist_rot": torch.zeros(length, 3),
            "wrist_velocity": torch.zeros_like(wrist),
            "mano_joints": {name: wrist.clone() for name in joint_names},
            "mano_joints_velocity": {name: torch.zeros_like(wrist) for name in joint_names},
        }
        final_transform = np.eye(4)
        final_transform[:3, :3] = Rotation.from_euler("x", 180, degrees=True).as_matrix()
        final_transform[:3, 3] = np.array([0.0, 0.0, -0.01])
        schedule = MODULE["build_transform_schedule"](
            final_transform, np.array([0.0, 0.5, 1.0])
        )
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
            metadata = {
                "schema_version": 6,
                "geometry_feasible": True,
                "side": "right",
                "semantic_precheck": {"passed": True},
                "physics_validation_required": False,
                "tool_mesh": str(mesh_file),
                "tool_mesh_sha256": hashlib.sha256(mesh_file.read_bytes()).hexdigest(),
                "collision_bundle_sha256": dataset._urdf_bundle_sha256(collision_urdf),
                "alpha": np.array([0.0, 0.5, 1.0], dtype=np.float32),
                "transform_schedule_tool": schedule.astype(np.float32),
                "validation_tiers": {"semantic_geometry": {"passed": True}},
            }
            dataset._apply_contact_retarget_targets(data, metadata, "corrected.pkl")

        self.assertTrue(torch.allclose(data["wrist_pos"][0], wrist[0]))
        self.assertTrue(
            torch.allclose(data["wrist_pos"][-1], torch.tensor([0.0, 0.0, -0.03]), atol=1e-6)
        )
        self.assertAlmostEqual(float(torch.linalg.norm(data["wrist_rot"][-1])), np.pi, places=5)
        for name in joint_names:
            self.assertTrue(torch.allclose(data["mano_joints"][name], data["wrist_pos"]))

    def test_stage1_uses_physics_corrected_body_targets(self):
        base_module = load_base_module_without_training_dependencies()

        class TestManipData(base_module["ManipData"]):
            def __getitem__(self, index):
                raise NotImplementedError

        body_names = [
            "palm", "thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"
        ]
        mapping = {name: ("wrist" if name == "palm" else name) for name in body_names}
        dataset = object.__new__(TestManipData)
        dataset.dexhand = types.SimpleNamespace(
            side="rh",
            body_names=body_names,
            to_hand=lambda name: (mapping[name],),
        )
        dataset.device = "cpu"
        dataset.skip = 2
        dataset.ch_dist = SquaredChamfer()
        length = 3
        original = torch.zeros(length, 3)
        targets = torch.arange(length * len(body_names) * 3, dtype=torch.float32).reshape(
            length, len(body_names), 3
        ) * 0.001
        final_wrist = torch.tensor([[0.01, 0.0, 0.0]]).repeat(length, 1)
        data = {
            "obj_trajectory": torch.eye(4).repeat(length, 1, 1),
            "obj_verts": torch.tensor([[0.0, 0.0, 0.0]]),
            "wrist_pos": original.clone(),
            "wrist_rot": original.clone(),
            "opt_wrist_pos": final_wrist,
            "opt_wrist_rot": original.clone(),
            "mano_joints": {
                name: original.clone() for name in body_names if name != "palm"
            },
            "mano_joints_velocity": {
                name: original.clone() for name in body_names if name != "palm"
            },
            "corrected_mano_body_targets": targets,
            "corrected_mano_body_names": body_names,
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
            metadata = {
                "schema_version": 8,
                "geometry_feasible": True,
                "side": "right",
                "semantic_precheck": {"passed": True},
                "physics_validation_required": True,
                "tool_mesh": str(mesh_file),
                "tool_mesh_sha256": hashlib.sha256(mesh_file.read_bytes()).hexdigest(),
                "collision_bundle_sha256": dataset._urdf_bundle_sha256(collision_urdf),
                "delta_tool": np.zeros(3, dtype=np.float32),
                "alpha": np.ones(length, dtype=np.float32),
                "validation_tiers": {
                    name: {"passed": True}
                    for name in (
                        "semantic_geometry", "isaac_gym_physics",
                        "physical_optimization", "mano_target_sync",
                    )
                },
            }
            dataset._apply_contact_retarget_targets(data, metadata, "corrected.pkl")

        self.assertTrue(torch.equal(data["wrist_pos"], final_wrist))
        for body_index, body_name in enumerate(body_names[1:], start=1):
            self.assertTrue(
                torch.equal(data["mano_joints"][body_name], targets[:, body_index])
            )
        self.assertEqual(
            tuple(data["corrected_mano_body_targets_velocity"].shape),
            tuple(targets.shape),
        )


if __name__ == "__main__":
    unittest.main()
