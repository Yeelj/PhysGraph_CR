"""Run the real replay methods with CPU tensors and a recording Gym adapter.

The AST loader skips optional dataset/rendering imports. Asset loading, actor
selection, material options, tensor indexing, camera selection and the playback
loop all execute the script's source.
"""

import argparse
import ast
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from itertools import product
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
try:
    from isaacgym import gymapi as _optional_gymapi  # noqa: F401; must precede torch
except ImportError:
    pass
import torch
from scipy.spatial.transform import Rotation


SOURCE_PATH = Path(__file__).resolve().parents[1] / "scripts/visualize_taco_retarget.py"


def aa_to_quat(value):
    xyzw = Rotation.from_rotvec(value.numpy()).as_quat()
    return torch.as_tensor(xyzw[:, [3, 0, 1, 2]], dtype=value.dtype)


def rotmat_to_quat(value):
    xyzw = Rotation.from_matrix(value.numpy()).as_quat()
    return torch.as_tensor(xyzw[:, [3, 0, 1, 2]], dtype=value.dtype)


def parse_custom_arguments(description, headless, custom_parameters):
    """Use argparse on the production custom-parameter definitions."""
    parser = argparse.ArgumentParser(description=description)
    for option in custom_parameters:
        parser.add_argument(option["name"], **{k: v for k, v in option.items() if k != "name"})
    return parser.parse_args()


GYM_API = SimpleNamespace(
    AssetOptions=SimpleNamespace, Transform=SimpleNamespace, CameraProperties=SimpleNamespace,
    Vec3=lambda *values: tuple(float(value) for value in values),
    COMPUTE_PER_VERTEX=1, DOF_MODE_NONE=0, DOMAIN_SIM=0, MESH_VISUAL=1, INVALID_HANDLE=-1,
    KEY_ESCAPE=1, KEY_SPACE=2, KEY_R=3,
)


def load_viewer_source():
    tree = ast.parse(SOURCE_PATH.read_text())
    names = {"TABLE_WIDTH_OFFSET", "TABLE_HEIGHT", "TABLE_POSITION", "TABLE_SURFACE_Z", "REPLAY_COMPONENTS"}
    selected = [node for node in tree.body if (
        isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)
    ) or (
        isinstance(node, ast.FunctionDef) and node.name in ("_parse_color", "parse_args")
    ) or (
        isinstance(node, ast.ClassDef) and node.name == "TacoRetargetViewer"
    )]
    namespace = {
        "argparse": argparse, "math": math, "os": os, "np": np, "torch": torch,
        "gymapi": GYM_API,
        "gymtorch": SimpleNamespace(wrap_tensor=lambda tensor: tensor, unwrap_tensor=lambda tensor: tensor),
        "gymutil": SimpleNamespace(parse_arguments=parse_custom_arguments),
        "aa_to_quat": aa_to_quat, "rotmat_to_quat": rotmat_to_quat,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    return namespace


SOURCE = load_viewer_source()
Viewer = SOURCE["TacoRetargetViewer"]
COMPONENTS = SOURCE["REPLAY_COMPONENTS"]


class RecordingGym:
    def __init__(self, hands):
        self.hands = {Path(hand.urdf_path).name: hand for hand in hands}
        self.loaded_assets = []
        self.actors = {}
        self.colors = {}
        self.textures = {}
        self.texture_buffers = []
        self.dof_acquired = 0
        self.root_writes = []
        self.dof_writes = []
        self.draws = 0
        self.max_draws = 10
        self.event_batches = []

    def load_asset(self, sim, root, filename, options):
        hand = self.hands.get(filename)
        asset = SimpleNamespace(
            filename=filename, use_mesh_materials=options.use_mesh_materials,
            dof_names=hand.dof_names if hand else (), bodies=3,
        )
        self.loaded_assets.append(asset)
        return asset

    def get_asset_dof_names(self, asset):
        return asset.dof_names

    def create_box(self, sim, width, depth, height, options):
        return SimpleNamespace(filename="table", dof_names=(), bodies=1)

    def create_env(self, sim, lower, upper, rows):
        return object()

    def create_actor(self, env, asset, pose, name, group, replay_filter):
        # Actor handles deliberately differ from the simulator tensor indices.
        handle = 10 + len(self.actors)
        self.actors[handle] = SimpleNamespace(
            name=name, asset=asset, index=len(self.actors), replay_filter=replay_filter,
        )
        return handle

    def get_actor_rigid_body_count(self, env, actor):
        return self.actors[actor].asset.bodies

    def create_texture_from_buffer(self, sim, width, height, pixels):
        self.texture_buffers.append((width, height, pixels.copy()))
        return 500

    def set_rigid_body_texture(self, env, actor, body, mesh, texture):
        assert mesh == GYM_API.MESH_VISUAL
        self.textures[self.actors[actor].name, body] = texture

    def set_rigid_body_color(self, env, actor, body, mesh, color):
        assert mesh == GYM_API.MESH_VISUAL
        self.colors[self.actors[actor].name, body] = color

    def acquire_actor_root_state_tensor(self, sim):
        if len(self.actors) == 1:
            raise AssertionError("The table-only scene must skip the root tensor API")
        return torch.full((len(self.actors), 13), -7.0)

    def get_actor_index(self, env, actor, domain):
        return self.actors[actor].index

    def get_sim_dof_count(self, sim):
        return sum(len(actor.asset.dof_names) for actor in self.actors.values())

    def acquire_dof_state_tensor(self, sim):
        count = self.get_sim_dof_count(sim)
        if not count:
            raise AssertionError("A scene without hands must skip the DOF tensor API")
        self.dof_acquired += 1
        return torch.full((count, 2), -3.0)

    def set_actor_root_state_tensor(self, sim, tensor):
        self.root_writes.append(tensor.clone())

    def set_dof_state_tensor(self, sim, tensor):
        self.dof_writes.append(tensor.clone())

    def create_viewer(self, sim, properties):
        return object()

    def subscribe_viewer_keyboard_event(self, viewer, key, action):
        pass

    def viewer_camera_look_at(self, viewer, env, position, target):
        self.camera_position, self.camera_target = position, target

    def query_viewer_has_closed(self, viewer):
        return self.draws >= self.max_draws

    def query_viewer_action_events(self, viewer):
        actions = self.event_batches.pop(0) if self.event_batches else []
        return [SimpleNamespace(action=action, value=1) for action in actions]

    def simulate(self, sim):
        pass

    def fetch_results(self, sim, wait):
        pass

    def step_graphics(self, sim):
        pass

    def draw_viewer(self, viewer, sim, sync):
        self.draws += 1

    def sync_frame_time(self, sim):
        pass


def fixture(hidden=(), **colors):
    viewer = Viewer.__new__(Viewer)
    viewer.args = SimpleNamespace(start_frame=0, once=True, **colors)
    for component in hidden:
        setattr(viewer.args, f"hide_{component}", True)
    viewer.sim = object()
    viewer.device = torch.device("cpu")
    viewer.frame_count = 3
    # Unequal hand DOF counts expose incorrect slices when either hand is hidden.
    viewer.dexhand_rh = SimpleNamespace(urdf_path="rh.urdf", dof_names=("r0", "r1", "r2"), n_dofs=3)
    viewer.dexhand_lh = SimpleNamespace(urdf_path="lh.urdf", dof_names=("l0", "l1"), n_dofs=2)
    viewer.gym = RecordingGym((viewer.dexhand_rh, viewer.dexhand_lh))
    for side, offset, hand in (("rh", 0, viewer.dexhand_rh), ("lh", 100, viewer.dexhand_lh)):
        trajectory = torch.eye(4).repeat(viewer.frame_count, 1, 1)
        trajectory[:, :3, :3] = torch.as_tensor(Rotation.from_euler(
            "z", np.arange(viewer.frame_count) * 0.1
        ).as_matrix(), dtype=torch.float32)
        trajectory[:, :3, 3] = torch.arange(viewer.frame_count * 3).reshape(-1, 3) + offset + 20
        setattr(viewer, f"{side}_data", {
            "obj_urdf_path": f"{side}_object.urdf",
            "opt_wrist_pos": torch.arange(viewer.frame_count * 3).reshape(-1, 3).float() + offset,
            "opt_wrist_rot": torch.arange(viewer.frame_count * 3).reshape(-1, 3).float() * 0.05,
            "opt_dof_pos": torch.arange(viewer.frame_count * hand.n_dofs).reshape(-1, hand.n_dofs).float() + offset,
            "obj_trajectory": trajectory,
        })
    viewer._create_scene()
    viewer._acquire_state_tensors()
    viewer._prepare_trajectories()
    return viewer


class RetargetViewerTest(unittest.TestCase):
    def test_cli_colors_visibility_and_existing_flags(self):
        argv = ["viewer", "--data_idx", "083f7@0", "--new_obj", "kni@2", "--correct", "--once"]
        for component, _, _ in COMPONENTS.values():
            argv += [f"--{component}_color", "0.1,0.5,1", f"--hide_{component}"]
        with patch.object(sys, "argv", argv):
            args = SOURCE["parse_args"]()
        self.assertEqual((args.data_idx, args.new_obj, args.correct, args.once), ("083f7@0", "kni@2", True, True))
        for component, _, _ in COMPONENTS.values():
            self.assertEqual(getattr(args, f"{component}_color"), (0.1, 0.5, 1))
            self.assertTrue(getattr(args, f"hide_{component}"))
        with patch.object(sys, "argv", ["viewer"]):
            args = SOURCE["parse_args"]()
        for component, _, _ in COMPONENTS.values():
            self.assertIsNone(getattr(args, f"{component}_color"))
            self.assertFalse(getattr(args, f"hide_{component}"))
        self.assertEqual(SOURCE["_parse_color"](" URDF "), "urdf")
        for value in ("red", "0.5", "0,1", "0,0,0,1", "-1,0,0", "0,0,255", "nan,0,0", "0,inf,0"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                SOURCE["_parse_color"](value)
        with patch.object(sys, "argv", ["viewer", "--tool_color", "nan,0,0"]), redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as error:
                SOURCE["parse_args"]()
        self.assertEqual(error.exception.code, 2)

    def test_all_visibility_combinations_preserve_frame_mapping_and_hand_dofs(self):
        component_names = [component for component, _, _ in COMPONENTS.values()]
        for flags in product((False, True), repeat=4):
            hidden = [component for component, hide in zip(component_names, flags) if hide]
            with self.subTest(hidden=hidden):
                viewer = fixture(hidden)
                expected_keys = {key for key, (component, _, _) in COMPONENTS.items() if component not in hidden}
                self.assertEqual(set(viewer.actor_handles), expected_keys)
                self.assertEqual(set(viewer.actor_indices), expected_keys)
                self.assertEqual(len(viewer.gym.loaded_assets), len(expected_keys))
                table_index = viewer.gym.get_actor_index(viewer.env, viewer.table_actor, GYM_API.DOMAIN_SIM)
                table_state = viewer.root_state[table_index].clone() if expected_keys else None
                originals = [{key: value.clone() for key, value in data.items() if torch.is_tensor(value)}
                             for data in (viewer.rh_data, viewer.lh_data)]
                for frame in range(viewer.frame_count):
                    viewer._set_frame(frame)
                    for key, index in viewer.actor_indices.items():
                        torch.testing.assert_close(viewer.root_state[index, :3], getattr(viewer, f"{key}_pos")[frame])
                        torch.testing.assert_close(viewer.root_state[index, 3:7], getattr(viewer, f"{key}_quat")[frame])
                        torch.testing.assert_close(viewer.root_state[index, 7:], torch.zeros(6))
                    expected_dofs = [getattr(viewer, f"{key}_dof")[frame] for key in ("hand_rh", "hand_lh") if key in expected_keys]
                    if expected_dofs:
                        torch.testing.assert_close(viewer.dof_state[:, 0], torch.cat(expected_dofs))
                        torch.testing.assert_close(viewer.dof_state[:, 1], torch.zeros(viewer.dof_state.shape[0]))
                    else:
                        self.assertIsNone(viewer.dof_state)
                        self.assertFalse(viewer.gym.dof_writes)
                        self.assertEqual(viewer.gym.dof_acquired, 0)
                if expected_keys:
                    torch.testing.assert_close(viewer.root_state[table_index], table_state)
                else:
                    self.assertIsNone(viewer.root_state)
                    self.assertFalse(viewer.gym.root_writes)
                for data, original in zip((viewer.rh_data, viewer.lh_data), originals):
                    for key, value in original.items():
                        torch.testing.assert_close(data[key], value)

    def test_legacy_colors_and_textures_remain_the_default(self):
        viewer = fixture()
        self.assertEqual([actor.name for actor in viewer.gym.actors.values()],
                         ["dexhand_r", "dexhand_l", "table", "manip_obj_rh", "manip_obj_lh"])
        self.assertTrue(all(asset.use_mesh_materials for asset in viewer.gym.loaded_assets))
        self.assertFalse(viewer.gym.texture_buffers)
        self.assertFalse(viewer.gym.textures)
        expected = {("table", 0): (0.12, 0.12, 0.12)}
        for key, (_, actor_name, color) in COMPONENTS.items():
            for body in range(viewer.gym.get_actor_rigid_body_count(viewer.env, viewer.actor_handles[key])):
                expected[actor_name, body] = color
        self.assertEqual(viewer.gym.colors, expected)
        self.assertTrue(all(actor.replay_filter == 1 for actor in viewer.gym.actors.values()))

    def test_rgb_and_urdf_modes_are_independent_for_all_four_components(self):
        for modes in product(("rgb", "urdf"), repeat=4):
            options = {}
            expected_colors = {}
            for index, ((key, (component, _, _)), mode) in enumerate(zip(COMPONENTS.items(), modes)):
                color = (index * 0.2, 0.5, 1.0) if mode == "rgb" else "urdf"
                options[f"{component}_color"] = color
                expected_colors[key] = color
            with self.subTest(modes=modes):
                viewer = fixture(**options)
                solid_keys = {key for key, value in expected_colors.items() if value != "urdf"}
                self.assertEqual(len(viewer.gym.texture_buffers), int(bool(solid_keys)))
                if solid_keys:
                    width, height, pixels = viewer.gym.texture_buffers[0]
                    self.assertEqual((width, height, pixels.shape, pixels.dtype), (1, 1, (1, 4), np.uint8))
                    np.testing.assert_array_equal(pixels, 255)
                for key, (_, actor_name, _) in COMPONENTS.items():
                    actor = viewer.gym.actors[viewer.actor_handles[key]]
                    color = expected_colors[key]
                    self.assertEqual(actor.asset.use_mesh_materials, color != "urdf")
                    for body in range(actor.asset.bodies):
                        if color == "urdf":
                            self.assertNotIn((actor_name, body), viewer.gym.colors)
                            self.assertNotIn((actor_name, body), viewer.gym.textures)
                        else:
                            self.assertEqual(viewer.gym.colors[actor_name, body], color)
                            self.assertEqual(viewer.gym.textures[actor_name, body], 500)
        viewer = fixture(hidden=("tool",), tool_color=(1, 0, 0))
        self.assertFalse(viewer.gym.texture_buffers)

    def test_camera_uses_visible_trajectories_and_falls_back_to_table_when_empty(self):
        for hidden in ((), ("right_hand", "left_hand", "object"), ("right_hand", "left_hand", "object", "tool")):
            with self.subTest(hidden=hidden):
                viewer = fixture(hidden)
                with redirect_stdout(StringIO()):
                    viewer._create_viewer()
                if viewer.actor_indices:
                    positions = torch.cat([getattr(viewer, f"{key}_pos") for key in viewer.actor_indices])
                    expected = (positions.amin(0) + positions.amax(0)) / 2
                else:
                    expected = (SOURCE["TABLE_POSITION"][0], SOURCE["TABLE_POSITION"][1], SOURCE["TABLE_SURFACE_Z"])
                np.testing.assert_allclose(viewer.gym.camera_target, expected)
                self.assertTrue(np.isfinite(viewer.gym.camera_position).all())

    def test_once_loop_pause_restart_and_quit_still_work_with_or_without_hands(self):
        cases = (
            (0, True, [], 10, [0, 1, 2]),
            (1, True, [], 10, [1, 2]),
            (0, False, [], 5, [0, 1, 2, 0, 1]),
            (0, False, [[], [], ["restart"], []], 4, [0, 1, 0, 1]),
            (1, False, [[], [], ["restart"], []], 4, [1, 2, 1, 2]),
            (0, False, [["pause"], [], ["pause"], []], 4, [0, 1]),
            (0, False, [["quit"]], 10, []),
        )
        for hidden in ((), ("right_hand", "left_hand", "tool", "object")):
            for start_frame, once, events, max_draws, expected_frames in cases:
                with self.subTest(hidden=hidden, start_frame=start_frame, once=once, events=events):
                    viewer = fixture(hidden)
                    viewer.viewer = object()
                    viewer.args.start_frame, viewer.args.once = start_frame, once
                    viewer.gym.max_draws, viewer.gym.event_batches = max_draws, list(events)
                    frames = []
                    set_frame = viewer._set_frame

                    def record_frame(frame):
                        frames.append(frame)
                        set_frame(frame)

                    viewer._set_frame = record_frame
                    viewer.run()
                    self.assertEqual(frames, expected_frames)


if __name__ == "__main__":
    unittest.main()
