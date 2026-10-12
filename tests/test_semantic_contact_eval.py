import ast
import contextlib
import io
import json
import math
from pathlib import Path
import pickle
import sys
import tempfile
import types
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from main.semantic_contact_eval import (
    CollisionSurfaceContactEvaluator,
    SemanticContactEpisodeAccumulator,
    SemanticContactEvalConfig,
    classify_surface_contacts,
    contact_required_mask_from_metadata,
    format_semantic_episode_summary,
    load_contact_required_masks,
    summarize_semantic_episodes,
)


ROOT = Path(__file__).resolve().parents[1]


def load_method(path, class_name, method_name, namespace):
    """Exercise actual integration methods without optional simulator imports."""
    tree = ast.parse((ROOT / path).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(module, str(ROOT / path), "exec"), namespace)
    return namespace[method_name]


class SemanticContactEvalTest(unittest.TestCase):
    def config(self, **kwargs):
        return SemanticContactEvalConfig(min_contact_frames=1, **kwargs)

    def test_thresholds_boundaries_and_nonfinite_forces(self):
        red = torch.tensor([[0.001, 0.008, 0.005, 0.001, 0.2, 0.001]])
        black = torch.tensor([[0.008, 0.001, 0.005, 0.008, 0.3, 0.008]])
        force = torch.tensor([[1.0, 2.0, 2.0, 0.99, 2.0, float("nan")]])
        r, b = classify_surface_contacts(red, black, force, self.config())
        self.assertEqual(r.tolist(), [[True, False, False, False, False, False]])
        self.assertEqual(b.tolist(), [[False, True, True, False, False, False]])
        self.assertFalse(bool((r & b).any()))

    def test_surface_geometry_detects_contact_far_from_link_origin_and_includes_palm(self):
        config = self.config(env_chunk_size=1)
        evaluator = CollisionSurfaceContactEvaluator(
            {"finger": [[-0.2, 0, 0]], "palm": [[0, 0, 0]]},
            {"finger": 0, "palm": 1},
            torch.tensor([[[0.0, 0, 0]], [[0.0, 0, 0]]]),
            torch.tensor([[[0.02, 0, 0]], [[0.02, 0, 0]]]),
            torch.tensor([1.0, 2.0]), config,
        )
        state = torch.zeros(2, 2, 13)
        state[:, :, 6] = 1
        state[0, 0, 0] = 1.2
        state[0, 1, 0] = 1.02
        state[1, 0, 0] = 1.2
        state[1, 1, 0] = 1.04
        force = torch.zeros(2, 2, 3)
        force[:, :, 0] = 2
        tool_position = torch.tensor([[1.0, 0, 0], [1.0, 0, 0]])
        tool_rotation = torch.tensor([[0.0, 0, 0, 1.0]]).expand(2, -1)
        rng_before = torch.get_rng_state().clone()
        r, b = evaluator.detect(state, force, tool_position, tool_rotation)
        self.assertEqual(r.tolist(), [[True, False], [True, False]])
        self.assertEqual(b.tolist(), [[False, True], [False, True]])
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
        # A link pressed against something far away from the tool is excluded.
        state[:, 1, 0] += 1
        _, b = evaluator.detect(state, force, tool_position, tool_rotation)
        self.assertFalse(bool(b.any()))

    def test_tool_and_hand_rotations_use_xyzw_and_current_tool_pose(self):
        evaluator = CollisionSurfaceContactEvaluator(
            {"finger": [[0.02, 0, 0]]}, {"finger": 0},
            torch.tensor([[[0.02, 0, 0]]]), torch.tensor([[[-0.02, 0, 0]]]),
            torch.tensor([1.0]), self.config(),
        )
        quaternion = torch.tensor([[0.0, 0, math.sqrt(0.5), math.sqrt(0.5)]])
        state = torch.zeros(1, 1, 13)
        state[0, 0, :3] = torch.tensor([3.0, 4.0, 5.0])
        state[:, :, 3:7] = quaternion[:, None]
        r, b = evaluator.detect(
            state, torch.ones(1, 1, 3), state[:, 0, :3], quaternion
        )
        self.assertTrue(r.item())
        self.assertFalse(b.item())

    def test_terminal_black_contact_is_counted_and_episode_reset_is_independent(self):
        acc = SemanticContactEpisodeAccumulator(torch.tensor([1, 2]), self.config())
        red = torch.tensor([[True, True], [True, False]])
        black = torch.zeros_like(red)
        acc.update(red, black, torch.tensor([0, 0]))
        acc.update(torch.tensor([[True, False], [True, False]]),
                   torch.tensor([[False, True], [False, False]]), torch.tensor([1, 1]))
        terminal = acc.snapshot(torch.tensor([True, False]))
        self.assertEqual(terminal["red_contacts"].tolist(), [3, 2])
        self.assertEqual(terminal["black_contacts"].tolist(), [1, 0])
        self.assertEqual(terminal["black_frames"].tolist(), [1, 0])
        self.assertEqual(terminal["semantic_success"].tolist(), [False, False])
        acc.reset(torch.tensor([0]))
        self.assertEqual(acc.frames.tolist(), [0, 2])
        self.assertEqual(terminal["frames"].tolist(), [2, 2])
        acc.update(red, black, torch.tensor([0, 2]))
        result = acc.snapshot(torch.tensor([False, True]))
        self.assertEqual(result["frames"].tolist(), [1, 3])
        self.assertEqual(result["semantic_success"].tolist(), [False, True])

    def test_early_failure_retains_quality_but_cannot_be_semantic_success(self):
        acc = SemanticContactEpisodeAccumulator(torch.tensor([99]), self.config())
        for frame in range(40):
            acc.update(torch.tensor([[True, True]]), torch.tensor([[False, False]]), torch.tensor([frame]))
        result = acc.snapshot(torch.tensor([False]))
        self.assertEqual(result["red_contact_ratio"].item(), 1)
        self.assertEqual(result["black_violation_frame_ratio"].item(), 0)
        self.assertEqual(result["frames"].item(), 40)
        self.assertFalse(result["semantic_success"].item())

    def test_partial_start_skipped_frames_and_duplicate_updates_cannot_fake_completion(self):
        red, black = torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 1, dtype=torch.bool)
        acc = SemanticContactEpisodeAccumulator(torch.tensor([4]), self.config())
        for frame in (2, 3, 4, 4):
            acc.update(red, black, torch.tensor([frame]))
        self.assertEqual(acc.frames.item(), 3)
        self.assertFalse(acc.snapshot(torch.tensor([True]))["semantic_success"].item())
        acc.reset(torch.tensor([0]))
        for frame in (0, 2, 3, 4):
            acc.update(red, black, torch.tensor([frame]))
        self.assertFalse(acc.snapshot(torch.tensor([True]))["complete"].item())

    def test_no_contact_and_minimum_sustained_contact(self):
        config = SemanticContactEvalConfig(min_contact_frames=3)
        acc = SemanticContactEpisodeAccumulator(torch.tensor([2, 2]), config)
        for frame in range(3):
            red = torch.tensor([[False], [frame == 0]])
            acc.update(red, torch.zeros_like(red), torch.tensor([frame, frame]))
        result = acc.snapshot(torch.tensor([True, True]))
        self.assertTrue(math.isnan(result["red_contact_ratio"][0].item()))
        self.assertTrue(math.isnan(result["black_contact_ratio"][0].item()))
        self.assertEqual(result["black_violation_frame_ratio"][0].item(), 0)
        self.assertFalse(bool(result["semantic_success"].any()))
        summary = summarize_semantic_episodes(result, torch.tensor([[0]]))
        self.assertIn("planned_semantic_contact_coverage: 0.0000", format_semantic_episode_summary(summary))
        self.assertEqual(summary["no_contact_episodes"], 1)

    def test_batch_summary_pools_events_and_counts_failures_without_filtering(self):
        acc = SemanticContactEpisodeAccumulator(torch.tensor([0, 0]), self.config())
        red = torch.tensor([[True] * 9, [False] * 9])
        black = torch.tensor([[False] * 9, [True] + [False] * 8])
        acc.update(red, black, torch.tensor([0, 0]))
        summary = summarize_semantic_episodes(acc.snapshot(torch.tensor([True, False])), torch.tensor([[0], [1]]))
        self.assertAlmostEqual(summary["red_contact_ratio"], 0.9)
        self.assertAlmostEqual(summary["black_contact_ratio"], 0.1)
        self.assertEqual(summary["black_violation_frame_ratio"], 0.5)
        self.assertEqual(summary["semantic_task_success_rate"], 0.5)
        self.assertEqual(summary["episodes"], 2)

    def test_invalid_config_fails_before_simulation(self):
        for kwargs in ({"force_threshold": 0}, {"contact_distance": float("nan")},
                       {"points_per_link": 0}, {"min_red_contact_ratio": 1.1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SemanticContactEvalConfig(**kwargs)

    def test_invalid_geometry_cannot_silently_change_contact_statistics(self):
        red = torch.zeros(1, 1, 3)
        for region, handle in ((torch.zeros(1, 0, 3), 0),
                               (torch.full((1, 1, 3), float("nan")), 0), (red, -1)):
            with self.subTest(handle=handle, shape=region.shape), self.assertRaises(ValueError):
                CollisionSurfaceContactEvaluator(
                    {"finger": [[0, 0, 0]]}, {"finger": handle},
                    region, red, torch.ones(1), self.config(),
                )

    def test_training_and_disabled_eval_skip_geometry_initialization(self):
        method = load_method(
            "physgraph_envs/lib/envs/tasks/dexhandmanip_bih.py", "DexHandManipBiHEnv",
            "_init_semantic_contact_evaluation", {},
        )
        # No data/simulator/URDF attributes are needed on the training path.
        for config in (None, SemanticContactEvalConfig(enabled=False)):
            with self.subTest(config=config):
                env = SimpleNamespace(semantic_contact_eval_config=config)
                with patch("builtins.__import__", side_effect=AssertionError("Unexpected import")):
                    method(env)

    def test_post_physics_collects_terminal_frame_before_progress_increment(self):
        method = load_method(
            "physgraph_envs/lib/envs/tasks/dexhandmanip_bih.py", "DexHandManipBiHEnv", "post_physics_step", {}
        )
        acc = SemanticContactEpisodeAccumulator(torch.tensor([0]), self.config())
        red, black = torch.tensor([[False]]), torch.tensor([[True]])
        env = SimpleNamespace(
            compute_observations=lambda: None, compute_reward=lambda actions: None,
            actions=None, _semantic_eval_detector=SimpleNamespace(detect=lambda *args: (red, black)),
            _semantic_eval_accumulator=acc, _rigid_body_state=None, net_cf=None,
            rh_states={"manip_obj_pos": None, "manip_obj_quat": None},
            progress_buf=torch.tensor([0]), running_progress_buf=torch.tensor([0]),
            randomize_buf=torch.tensor([0]),
        )
        method(env)
        self.assertEqual(acc.last_frame.item(), 0)
        self.assertEqual(acc.black_contacts.item(), 1)
        self.assertEqual(env.progress_buf.item(), 1)
        # Training/disabled evaluation retains the original method behaviour.
        env._semantic_eval_detector = None
        method(env)
        self.assertEqual(acc.frames.item(), 1)
        self.assertEqual(env.progress_buf.item(), 2)

    def test_player_appends_metrics_and_resets_only_finished_environments(self):
        namespace = {
            "torch": torch, "time": None,
            "summarize_semantic_episodes": summarize_semantic_episodes,
            "format_semantic_episode_summary": format_semantic_episode_summary,
        }
        run = load_method("lib/rl/player.py", "MyBasePlayer", "run", namespace)
        config = self.config()
        acc = SemanticContactEpisodeAccumulator(torch.tensor([0, 1]), config)

        class StopTest(Exception):
            pass

        class FakeEnv:
            def __init__(self):
                self.step_index = 0
                self.success_buf = torch.zeros(2, dtype=torch.bool)
                self.previous_done = torch.zeros(2, dtype=torch.bool)
                self.reset_ids = []

            def reset_done(self):
                ids = self.previous_done.nonzero().flatten()
                self.reset_ids.append(ids.tolist())
                acc.reset(ids)
                return torch.zeros(2, 1), ids

            def step(self, actions):
                if self.step_index == 2:
                    raise StopTest()
                if self.step_index == 0:
                    red, black = torch.tensor([[True], [True]]), torch.tensor([[False], [False]])
                    frames, done = torch.tensor([0, 0]), torch.tensor([True, False])
                else:
                    red, black = torch.tensor([[False], [True]]), torch.tensor([[True], [False]])
                    frames, done = torch.tensor([0, 1]), torch.tensor([True, True])
                acc.update(red, black, frames)
                self.success_buf = done.clone()
                self.previous_done = done.clone()
                self.step_index += 1
                return torch.zeros(2, 1), torch.ones(2), done, {"semantic_contact_eval": acc.snapshot(done)}

        env = FakeEnv()
        player = SimpleNamespace(
            render_env=False, is_deterministic=True, env=env, save_rollouts=False,
            wait_for_checkpoint=lambda: None, env_reset=lambda env: torch.zeros(2, 1),
            get_batch_size=lambda obs, initial: 2, get_action=lambda *args: torch.zeros(2, 1),
            env_step=lambda env, action: env.step(action), print_stats=True,
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(StopTest):
            run(player)
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("reward: 1.00 steps: 1.0 operation_completion_ratio: 1.0000"))
        self.assertIn("planned_semantic_contact_coverage: 1.0000", lines[0])
        self.assertIn("semantic_task_success_rate: 1.0000", lines[0])
        self.assertTrue(lines[1].startswith("reward: 1.50 steps: 1.5 operation_completion_ratio: 1.0000"))
        self.assertIn("planned_semantic_contact_coverage: 0.6667", lines[1])
        self.assertIn("black_violation_frame_ratio: 0.3333", lines[1])
        self.assertIn("semantic_task_success_rate: 0.5000", lines[1])
        self.assertIn("semantic_sr_total: 0.6667", lines[1])
        self.assertEqual(env.reset_ids, [[0], [0, 1]])

    def test_player_keeps_original_log_when_env_has_no_semantic_eval(self):
        run = load_method("lib/rl/player.py", "MyBasePlayer", "run", {"torch": torch})

        class StopTest(Exception):
            pass

        class FakeEnv:
            def __init__(self):
                self.step_index = 0
                self.success_buf = torch.tensor([True])

            def reset_done(self):
                return torch.zeros(1, 1), torch.tensor([0])

            def step(self, action):
                if self.step_index:
                    raise StopTest()
                self.step_index += 1
                return torch.zeros(1, 1), torch.tensor([42.0]), torch.tensor([True]), {}

        env = FakeEnv()
        player = SimpleNamespace(
            render_env=False, is_deterministic=True, env=env, save_rollouts=False,
            wait_for_checkpoint=lambda: None, env_reset=lambda env: torch.zeros(1, 1),
            get_batch_size=lambda obs, initial: 1, get_action=lambda *args: torch.zeros(1, 1),
            env_step=lambda env, action: env.step(action), print_stats=True,
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(StopTest):
            run(player)
        self.assertEqual(output.getvalue(), "reward: 42.00 steps: 1.0\n")

    def test_short_perfect_red_episode_does_not_receive_full_plan_credit(self):
        acc = SemanticContactEpisodeAccumulator(torch.tensor([299]), self.config())
        for frame in range(36):
            acc.update(torch.tensor([[True]]), torch.tensor([[False]]), torch.tensor([frame]))
        result = acc.snapshot(torch.tensor([False]))
        self.assertEqual(result["red_contact_ratio"].item(), 1)
        self.assertAlmostEqual(result["operation_completion_ratio"].item(), 0.12)
        self.assertAlmostEqual(result["planned_semantic_contact_coverage"].item(), 0.12)
        self.assertEqual(result["missing_contact_frames"].item(), 0)
        self.assertEqual(result["unexecuted_required_frames"].item(), 264)
        self.assertFalse(result["semantic_success"].item())

    def test_normal_approach_and_release_do_not_reduce_required_contact_coverage(self):
        required = torch.zeros(1, 100, dtype=torch.bool)
        required[:, 20:80] = True
        acc = SemanticContactEpisodeAccumulator(torch.tensor([99]), self.config(), required)
        for frame in range(100):
            acc.update(required[:, frame:frame + 1], torch.tensor([[False]]), torch.tensor([frame]))
        result = acc.snapshot(torch.tensor([True]))
        self.assertEqual(result["operation_completion_ratio"].item(), 1)
        self.assertEqual(result["planned_semantic_contact_coverage"].item(), 1)
        self.assertEqual(result["planned_required_frames"].item(), 60)
        self.assertEqual(result["missing_contact_frames"].item(), 0)
        self.assertTrue(result["semantic_success"].item())

    def test_sustained_contact_gate_rejects_sparse_red_contact_despite_original_success(self):
        required = torch.zeros(1, 100, dtype=torch.bool)
        required[:, 20:80] = True
        acc = SemanticContactEpisodeAccumulator(torch.tensor([99]), self.config(), required)
        for frame in range(100):
            red = required[:, frame:frame + 1] & (not 35 <= frame < 50)
            acc.update(red, torch.tensor([[False]]), torch.tensor([frame]))
        result = acc.snapshot(torch.tensor([True]))
        self.assertEqual(result["red_contact_ratio"].item(), 1)
        self.assertEqual(result["operation_completion_ratio"].item(), 1)
        self.assertEqual(result["planned_semantic_contact_coverage"].item(), 0.75)
        self.assertEqual(result["missing_contact_frames"].item(), 15)
        self.assertFalse(result["semantic_success"].item())

    def test_missing_black_and_unexecuted_required_frames_are_separate(self):
        required = torch.zeros(1, 100, dtype=torch.bool)
        required[:, 20:80] = True
        acc = SemanticContactEpisodeAccumulator(torch.tensor([99]), self.config(), required)
        for frame in range(40):
            red = torch.tensor([[20 <= frame < 30]])
            black = torch.tensor([[30 <= frame < 35]])
            acc.update(red, black, torch.tensor([frame]))
        result = acc.snapshot(torch.tensor([False]))
        self.assertAlmostEqual(result["operation_completion_ratio"].item(), 0.4)
        self.assertAlmostEqual(result["planned_semantic_contact_coverage"].item(), 10 / 60)
        self.assertEqual(result["compliant_required_frames"].item(), 10)
        self.assertEqual(result["black_required_frames"].item(), 5)
        self.assertEqual(result["missing_contact_frames"].item(), 5)
        self.assertEqual(result["unexecuted_required_frames"].item(), 40)
        self.assertEqual(result["black_frames"].item(), 5)
        self.assertEqual(sum(result[name].item() for name in (
            "compliant_required_frames", "black_required_frames", "missing_contact_frames",
            "unexecuted_required_frames",
        )), result["planned_required_frames"].item())

    def test_full_plan_batch_denominators_are_not_averaged_per_episode(self):
        required = torch.tensor([[1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)
        acc = SemanticContactEpisodeAccumulator(torch.tensor([3, 5]), self.config(), required)
        for frame in range(3):
            acc.update(torch.tensor([[frame == 0], [True]]), torch.zeros(2, 1, dtype=torch.bool),
                       torch.tensor([frame if frame < 2 else 1, frame]))
        result = acc.snapshot(torch.tensor([False, False]))
        summary = summarize_semantic_episodes(result, torch.tensor([0, 1]))
        self.assertAlmostEqual(summary["operation_completion_ratio"], 5 / 10)
        self.assertAlmostEqual(summary["planned_semantic_contact_coverage"], 4 / 8)
        self.assertEqual(summary["planned_frames"], 10)
        self.assertEqual(summary["planned_required_frames"], 8)
        acc.reset(torch.tensor([0]))
        self.assertEqual(acc.planned_required_frames.tolist(), [2, 6])
        self.assertEqual(result["planned_required_frames"].tolist(), [2, 6])

    def test_no_required_frames_is_na_and_cannot_produce_semantic_success(self):
        acc = SemanticContactEpisodeAccumulator(
            torch.tensor([0]), self.config(), torch.zeros(1, 1, dtype=torch.bool)
        )
        acc.update(torch.tensor([[True]]), torch.tensor([[False]]), torch.tensor([0]))
        result = acc.snapshot(torch.tensor([True]))
        self.assertTrue(math.isnan(result["planned_semantic_contact_coverage"].item()))
        self.assertFalse(result["semantic_success"].item())
        summary = summarize_semantic_episodes(result, torch.tensor([0]))
        self.assertIn("planned_semantic_contact_coverage: NA", format_semantic_episode_summary(summary))

    def test_source_annotation_interval_is_aligned_to_full_operation_frames(self):
        metadata = {
            "data_idx": "motion@0", "grasp_interval": [2, 4],
            "source_contact_intent": {"contact_frame_mask": [[1, 0], [0, 0], [0, 1]]},
        }
        mask = contact_required_mask_from_metadata(metadata, 8, "motion@0")
        self.assertEqual(mask.tolist(), [False, False, True, False, True, False, False, False])
        clipped = contact_required_mask_from_metadata(metadata, 4, "motion@0")
        self.assertEqual(clipped.tolist(), [False, False, True, False])
        with self.assertRaises(ValueError):
            contact_required_mask_from_metadata(metadata, 8, "other@0")
        metadata["grasp_interval"] = [2, 5]
        with self.assertRaises(ValueError):
            contact_required_mask_from_metadata(metadata, 8, "motion@0")

    def test_shared_json_overrides_different_checkpoint_references_and_pkls(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.json"
            path.write_text(json.dumps({"motion@0": {"required_intervals": [[1, 3]]}}))
            config = self.config(contact_required_mask_file=str(path))
            distances = torch.tensor([[[0.01]] * 5, [[0.5]] * 5])
            mask, sources = load_contact_required_masks(
                ["motion@0", "motion@0"], torch.tensor([4, 4]), distances, config,
                [Path(directory) / "missing.pkl", None],
            )
            self.assertEqual(mask.tolist(), [[False, True, True, True, False]] * 2)
            self.assertTrue(all(source.startswith("shared_json:") for source in sources))
            path.write_text(json.dumps({"motion@0": [False, True, False, True, False]}))
            mask, _ = load_contact_required_masks(["motion@0"], torch.tensor([4]), None, config)
            self.assertEqual(mask.tolist(), [[False, True, False, True, False]])
            with self.assertRaises(ValueError):
                load_contact_required_masks(["missing@0"], torch.tensor([4]), distances, config)

    def test_common_source_pkl_has_priority_over_checkpoint_tip_distances(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.pkl"
            with path.open("wb") as stream:
                pickle.dump({"contact_retarget": {
                    "data_idx": "motion@0", "grasp_interval": [1, 2],
                    "source_contact_intent": {"contact_frame_mask": [[True], [True]]},
                }}, stream)
            mask, sources = load_contact_required_masks(
                ["motion@0"] * 2, torch.tensor([3, 3]), torch.full((2, 4, 5), 0.5),
                self.config(), [path, path],
            )
            self.assertEqual(mask.tolist(), [[False, True, True, False]] * 2)
            self.assertTrue(all(source.startswith("source_contact_intent:") for source in sources))

    def test_legacy_corrected_pkl_uses_shared_grasp_interval_without_retraining(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pkl"
            with path.open("wb") as stream:
                pickle.dump({"contact_retarget": {
                    "data_idx": "motion@0", "grasp_interval": [1, 3],
                    "source_contact_intent": {"active_link_count": 3},
                }}, stream)
            mask, sources = load_contact_required_masks(
                ["motion@0"] * 2, torch.tensor([4, 4]), torch.tensor([[[0.0]] * 5, [[0.5]] * 5]),
                self.config(), [path, path],
            )
            self.assertEqual(mask.tolist(), [[False, True, True, True, False]] * 2)
            self.assertTrue(all(source.startswith("source_grasp_interval:") for source in sources))

    def test_reference_fallback_is_explicit_and_nonfinite_distances_are_rejected(self):
        distances = torch.tensor([[[0.1], [0.01], [0.02], [0.1]]])
        mask, sources = load_contact_required_masks(["motion@0"], torch.tensor([3]), distances, self.config())
        self.assertEqual(mask.tolist(), [[False, True, True, False]])
        self.assertEqual(sources, ["reference_tips_distance"])
        distances[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            load_contact_required_masks(["motion@0"], torch.tensor([3]), distances, self.config())

    def test_original_and_corrected_policy_references_use_same_source_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "data/retargeting/OakInk-v2/mano2artimano_rh/motion@0.pkl"
            corrected = root / "data/retargeting_contact/OakInk-v2/mano2artimano_rh/motion@0/tool/motion@0.pkl"
            corrected.parent.mkdir(parents=True)
            with corrected.open("wb") as stream:
                pickle.dump({"contact_retarget": {
                    "data_idx": "motion@0", "grasp_interval": [1, 3],
                    "source_contact_intent": {"contact_frame_mask": [[True]] * 3},
                }}, stream)

            class Hand:
                urdf_path = "unused.urdf"
                dof_names = ()

                def __str__(self):
                    return "artimano_rh"

            geometry_module = types.ModuleType("main.dataset.contact_trajectory_retarget")
            geometry_module.PROJECT_ROOT = root
            geometry_module.URDFCollisionKinematics = lambda *args, **kwargs: SimpleNamespace(
                link_points={"finger": [[0, 0, 0]]}
            )
            namespace = {
                "torch": torch, "Path": Path,
                "CollisionSurfaceContactEvaluator": CollisionSurfaceContactEvaluator,
                "SemanticContactEpisodeAccumulator": SemanticContactEpisodeAccumulator,
                "load_contact_required_masks": load_contact_required_masks,
            }
            method = load_method("physgraph_envs/lib/envs/tasks/dexhandmanip_bih.py", "DexHandManipBiHEnv",
                                 "_init_semantic_contact_evaluation", namespace)
            masks = []
            for correct, trajectory in ((False, original), (True, corrected)):
                env = SimpleNamespace(
                    semantic_contact_eval_config=self.config(), dexhand_rh=Hand(), envs=[None],
                    gym=SimpleNamespace(find_actor_handle=lambda *args: 0, find_actor_rigid_body_handle=lambda *args: 0),
                    max_episode_length=100, num_envs=1, dataIndices=["motion@0"], correct=correct, new_obj="tool",
                    random_state_init=False, rollout_begin=None, rollout_len=None,
                    demo_data_lh={"seq_len": torch.tensor([8])},
                    demo_data_rh={
                        "seq_len": torch.tensor([8]), "tool_red_points": torch.zeros(1, 1, 3),
                        "tool_black_points": torch.ones(1, 1, 3), "tool_semantic_scale": torch.ones(1),
                        "retargeted_data_path": [str(trajectory)], "replacement_obj_id": ["tool"],
                        "dataset_name": ["OakInk-v2"], "tips_distance": torch.full((1, 8, 5), 0 if correct else 0.5),
                    },
                )
                with patch.dict(sys.modules, {geometry_module.__name__: geometry_module}), contextlib.redirect_stdout(io.StringIO()):
                    method(env)
                masks.append(env._semantic_eval_accumulator.contact_required.clone())
                self.assertEqual(env.demo_data_rh["retargeted_data_path"], [str(trajectory)])
            self.assertTrue(torch.equal(masks[0], masks[1]))
            self.assertEqual(masks[0].tolist(), [[False, True, True, True, False]])


if __name__ == "__main__":
    unittest.main()
