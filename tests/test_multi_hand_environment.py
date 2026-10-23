"""Exercise exact environment methods with synthetic named simulator tensors.

The AST loader avoids dataset/renderer imports and constructor side effects.
Only the simulator I/O and rotation adapters are substituted; the observation,
reset, action, limit-ordering and reward methods execute their real source.
"""

import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
try:
    from isaacgym import gymapi as _optional_gymapi  # noqa: F401; must precede torch
except ImportError:
    pass
import torch
from scipy.spatial.transform import Rotation

from main.dexhand_runtime import joint_order_indices, validate_direct_control
from main.dexhand_spec import ObservationLayout, SUPPORTED_HANDS, get_hand_spec


SOURCE_PATH = Path(__file__).resolve().parents[1] / "physgraph_envs/lib/envs/tasks/dexhandmanip_bih.py"


def aa_to_quat(value):
    quat = Rotation.from_rotvec(value.detach().numpy().reshape(-1, 3)).as_quat()
    return torch.as_tensor(quat[:, [3, 0, 1, 2]], dtype=value.dtype).reshape(*value.shape[:-1], 4)


def rotmat_to_quat(value):
    quat = Rotation.from_matrix(value.detach().numpy().reshape(-1, 3, 3)).as_quat()
    return torch.as_tensor(quat[:, [3, 0, 1, 2]], dtype=value.dtype).reshape(*value.shape[:-2], 4)


def quat_to_rotmat(value):
    matrix = Rotation.from_quat(value.detach().numpy().reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix()
    return torch.as_tensor(matrix, dtype=value.dtype).reshape(*value.shape[:-1], 3, 3)


def quat_conjugate(value):
    result = value.clone()
    result[..., :3] *= -1
    return result


def quat_mul(left, right):
    product = Rotation.from_quat(left.detach().numpy().reshape(-1, 4)) * Rotation.from_quat(right.detach().numpy().reshape(-1, 4))
    return torch.as_tensor(product.as_quat(), dtype=left.dtype).reshape(left.shape)


def clamp_tensor(value, lower, upper):
    return torch.maximum(torch.minimum(value, upper), lower)


class RewardCapture:
    target = None

    def __call__(self, reset, progress, running, actions, states, target, *unused):
        self.target = target
        scalar = torch.zeros_like(progress, dtype=torch.float32)
        return scalar, reset, reset, reset, {}, reset, scalar, scalar, scalar, scalar


REWARD_CAPTURE = RewardCapture()


def load_environment_methods():
    source = SOURCE_PATH.read_text()
    tree = ast.parse(source)
    environment = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DexHandManipBiHEnv")
    wanted = {"_configure_hand_joint_order", "_update_states", "_reset_default_side", "pre_physics_step", "compute_observations_side", "compute_reward_side"}
    methods = [deepcopy(node) for node in environment.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if len(methods) != len(wanted):
        raise AssertionError("Environment methods changed; update the integration fixture")

    create_envs = next(node for node in environment.body if isinstance(node, ast.FunctionDef) and node.name == "_create_envs")
    limit_loop = next(
        node for node in create_envs.body
        if isinstance(node, ast.For)
        and "self.limit_info[side][bound]" in ast.get_source_segment(source, node)
        and "index_select(0, order)" in ast.get_source_segment(source, node)
    )
    # Run the same limit conversion block used after Gym asset import.
    limit_method = ast.parse("def canonicalize_limits(self):\n    pass\n").body[0]
    limit_method.body = [deepcopy(limit_loop)]
    methods.append(limit_method)
    fixture_class = ast.ClassDef(name="EnvironmentMethods", bases=[], keywords=[], body=methods, decorator_list=[])
    module = ast.Module(body=[fixture_class], type_ignores=[])
    namespace = {
        "torch": torch,
        "np": np,
        "joint_order_indices": joint_order_indices,
        "torch_jit_utils": SimpleNamespace(tensor_clamp=clamp_tensor, scale=lambda x, low, high: low + (x + 1) * 0.5 * (high - low)),
        "gymtorch": SimpleNamespace(unwrap_tensor=lambda x: x),
        "gymapi": SimpleNamespace(ENV_SPACE=0),
        "aa_to_quat": aa_to_quat,
        "rotmat_to_quat": rotmat_to_quat,
        "quat_to_rotmat": quat_to_rotmat,
        "quat_conjugate": quat_conjugate,
        "quat_mul": quat_mul,
        "compute_imitation_reward": REWARD_CAPTURE,
    }
    exec(compile(ast.fix_missing_locations(module), str(SOURCE_PATH), "exec"), namespace)
    return namespace["EnvironmentMethods"]


EnvironmentMethods = load_environment_methods()


class FakeGym:
    def __init__(self, assets):
        self.assets = assets
        self.last_targets = None
        self.last_forces = None

    def get_asset_dof_names(self, asset):
        return self.assets[asset]["dof_names"]

    def get_asset_rigid_body_names(self, asset):
        return self.assets[asset]["body_names"]

    def apply_rigid_body_force_tensors(self, sim, force, torque, space):
        self.last_forces = force.clone(), torque.clone()

    def set_dof_position_target_tensor(self, sim, targets):
        self.last_targets = targets.clone()

    def get_sim_params(self, sim):
        return SimpleNamespace(gravity=SimpleNamespace(z=-9.81))

    def get_frame_count(self, sim):
        return 0


def fixture(name, future_length=1):
    env = EnvironmentMethods()
    env.num_envs, frames = 3, 5
    env.device = env.sim_device = torch.device("cpu")
    env.hand_specs = {side: get_hand_spec(name, side) for side in ("rh", "lh")}
    right = env.hand_specs["rh"]
    dofs, bodies = right.n_dofs, right.n_bodies
    env.num_dexhand_rh_dofs = dofs
    env.num_dexhand_lh_dofs = dofs
    assets = {}
    for side, shift in (("rh", 3), ("lh", 5)):
        spec = env.hand_specs[side]
        # Body order is independent of DOF order and both differ from policy.
        assets[side] = {
            "dof_names": spec.dof_names[shift:] + spec.dof_names[:shift],
            "body_names": spec.body_names[::-1],
        }
        setattr(env, "dexhand_" + side, SimpleNamespace(
            n_dofs=dofs, n_bodies=bodies, dof_names=spec.dof_names, body_names=spec.body_names,
            contact_body_names=spec.contact_body_names, weight_idx={}, to_dex=lambda _: (right.body_names[0],),
        ))
    # Bind side-specific wrist lookup (Inspire has side prefixes).
    for side in ("rh", "lh"):
        getattr(env, "dexhand_" + side).to_dex = lambda _, side=side: (env.hand_specs[side].body_names[0],)
    env.gym, env.sim = FakeGym(assets), object()
    for side in ("rh", "lh"):
        env._configure_hand_joint_order(side, side)

    env.limit_info = {}
    env.expected_limits = {}
    for side, offset in (("rh", 0.0), ("lh", 0.15)):
        low = torch.arange(dofs).float() * 0.01 - 1.0 + offset
        high = low + 1.5 + torch.arange(dofs).float() * 0.02
        speed = 0.1 + torch.arange(dofs).float() * 0.01
        effort = 2.0 + torch.arange(dofs).float()
        env.expected_limits[side] = low, high, speed, effort
        order = getattr(env, f"_{side}_sim_from_policy")
        for field, value in ((f"dexhand_{side}_dof_lower_limits", low), (f"dexhand_{side}_dof_upper_limits", high), (f"_dexhand_{side}_dof_speed_limits", speed), (f"_dexhand_{side}_effort_limits", effort)):
            setattr(env, field, value[order])
        env.limit_info[side] = {"lower": low[order].numpy().copy(), "upper": high[order].numpy().copy()}
    env.canonicalize_limits()

    env._q = torch.zeros(env.num_envs, 2 * dofs)
    env._qd, env._pos_control = env._q.clone(), env._q.clone()
    env.prev_targets = env._q.clone()
    env._rigid_body_state = torch.arange(env.num_envs * (2 * bodies + 2) * 13).reshape(env.num_envs, 2 * bodies + 2, 13).float() * 0.001
    env.net_cf = torch.arange(env.num_envs * (2 * bodies + 2) * 3).reshape(env.num_envs, 2 * bodies + 2, 3).float() * 0.001
    env.apply_forces = torch.zeros_like(env.net_cf)
    env.apply_torque = torch.zeros_like(env.net_cf)
    env.dof_force = torch.zeros_like(env._q)
    env.obs_future_length = future_length
    env.progress_buf = torch.tensor([0, 3, 5])
    env.running_progress_buf = torch.zeros(env.num_envs, dtype=torch.long)
    env.reset_buf = torch.zeros(env.num_envs, dtype=torch.bool)
    env.total_rew_buf = torch.zeros(env.num_envs)
    env.training, env.headless = True, True
    env.max_episode_length = frames
    env.use_pid_control = False
    env.act_moving_average = 0.4
    env.dt, env.translation_scale, env.orientation_scale = 0.02, 1.3, 0.7
    env.tighten_method, env.rollout_len = "None", None
    env.semantic_contact_enabled = False
    env._obs_keys = ("q", "cos_q", "sin_q", "base_state")
    env._privileged_obs_keys = ("dq", "manip_obj_pos", "manip_obj_quat", "manip_obj_vel", "manip_obj_ang_vel", "tip_force", "manip_obj_com", "manip_obj_weight")
    for side, offset in (("rh", 0), ("lh", bodies)):
        setattr(env, f"dexhand_{side}_handles", {body: offset + assets[side]["body_names"].index(body) for body in env.hand_specs[side].body_names})
        base = torch.zeros(env.num_envs, 13)
        base[:, :3], base[:, 6], base[:, 7:13] = 0.25, 1.0, 0.05
        setattr(env, f"_{side}_base_state", base)
        obj = base.clone()
        obj[:, :3], obj[:, 7:13] = 0.75, 0.15
        setattr(env, f"_manip_obj_{side}_root_state", obj)
        setattr(env, f"{side}_states", {})
        setattr(env, f"{side}_tips_contact_history", torch.zeros(env.num_envs, 2, 5, dtype=torch.bool))
        setattr(env, f"manip_obj_{side}_com", torch.zeros(env.num_envs, 3))
        setattr(env, f"manip_obj_{side}_mass", torch.arange(env.num_envs).float() + 1.0)
        setattr(env, f"obj_bps_{side}", torch.arange(128).float().repeat(env.num_envs, 1) + (1000 if side == "lh" else 0))

        def labels(width, field_offset):
            return (torch.arange(env.num_envs)[:, None, None] * 10000 + torch.arange(frames)[None, :, None] * 100 + torch.arange(width)[None, None] + field_offset).float()

        trajectories = torch.eye(4).repeat(env.num_envs, frames, 1, 1)
        rotations = torch.arange(env.num_envs * frames * 3).reshape(env.num_envs, frames, 3).float() * 0.002
        trajectories[:, :, :3, :3] = torch.as_tensor(
            Rotation.from_rotvec(rotations.reshape(-1, 3).numpy()).as_matrix(), dtype=torch.float32,
        ).reshape(env.num_envs, frames, 3, 3)
        trajectories[:, :, :3, 3] = labels(3, 1000)
        data = {
            "seq_len": torch.tensor([1, 3, 5]),
            "wrist_pos": labels(3, 10), "wrist_velocity": labels(3, 20),
            "wrist_rot": rotations + 0.01, "wrist_angular_velocity": labels(3, 30),
            "mano_joints": labels(3 * (bodies - 1), 40), "mano_joints_velocity": labels(3 * (bodies - 1), 50),
            "obj_trajectory": trajectories, "obj_velocity": labels(3, 60), "obj_angular_velocity": labels(3, 70),
            "tips_distance": labels(5, 80), "opt_dof_pos": torch.zeros(env.num_envs, frames, dofs),
            "opt_dof_velocity": torch.zeros(env.num_envs, frames, dofs),
            "opt_wrist_pos": labels(3, 90), "opt_wrist_rot": torch.zeros(env.num_envs, frames, 3),
            "opt_wrist_velocity": labels(3, 100), "opt_wrist_angular_velocity": labels(3, 110),
        }
        setattr(env, f"demo_data_{side}", data)
    env.actions = torch.zeros(env.num_envs, 12 + 2 * dofs)
    env._update_states()
    return env


class MultiHandEnvironmentTests(unittest.TestCase):
    def test_asset_orders_states_and_limits_are_converted_by_name(self):
        for name in SUPPORTED_HANDS:
            with self.subTest(hand=name):
                env = fixture(name)
                dofs = env.num_dexhand_rh_dofs
                for side, offset in (("rh", 0), ("lh", dofs)):
                    expected_q = torch.arange(dofs).float().repeat(env.num_envs, 1) * 0.1 + offset
                    expected_dq = expected_q + 0.5
                    order = getattr(env, f"_{side}_sim_from_policy")
                    self.assertFalse(torch.equal(order, torch.arange(dofs)))
                    env._q[:, offset:offset + dofs] = expected_q[:, order]
                    env._qd[:, offset:offset + dofs] = expected_dq[:, order]
                    env._update_states()
                    states = getattr(env, f"{side}_states")
                    torch.testing.assert_close(states["q"], expected_q)
                    torch.testing.assert_close(states["dq"], expected_dq)
                    torch.testing.assert_close(states["cos_q"], expected_q.cos())
                    torch.testing.assert_close(states["sin_q"], expected_q.sin())
                    handles = getattr(env, f"dexhand_{side}_handles")
                    expected_bodies = torch.stack([env._rigid_body_state[:, handles[body], :10] for body in env.hand_specs[side].body_names], dim=1)
                    torch.testing.assert_close(states["joints_state"], expected_bodies)
                    low, high, speed, effort = env.expected_limits[side]
                    for field, expected in ((f"dexhand_{side}_dof_lower_limits", low), (f"dexhand_{side}_dof_upper_limits", high), (f"_dexhand_{side}_dof_speed_limits", speed), (f"_dexhand_{side}_effort_limits", effort)):
                        torch.testing.assert_close(getattr(env, field), expected)
                    np.testing.assert_array_equal(env.limit_info[side]["lower"], low.numpy())
                    np.testing.assert_array_equal(env.limit_info[side]["upper"], high.numpy())

    def test_reset_clamps_in_policy_order_then_writes_simulator_order(self):
        for name in SUPPORTED_HANDS:
            with self.subTest(hand=name):
                env = fixture(name)
                dofs = env.num_dexhand_rh_dofs
                env_ids, frame_ids = torch.tensor([0, 2]), torch.tensor([0, 4])
                for side, offset in (("rh", 0), ("lh", dofs)):
                    data = getattr(env, f"demo_data_{side}")
                    data["opt_dof_pos"][:] = torch.linspace(-3, 3, dofs)
                    data["opt_dof_velocity"][:] = torch.linspace(-1, 1, dofs)
                    env._reset_default_side(env_ids, frame_ids, side)
                    low, high, speed, _ = env.expected_limits[side]
                    expected_q = torch.linspace(-3, 3, dofs).maximum(low).minimum(high).repeat(2, 1)
                    expected_dq = torch.linspace(-1, 1, dofs).maximum(-speed).minimum(speed).repeat(2, 1)
                    order = getattr(env, f"_{side}_sim_from_policy")
                    torch.testing.assert_close(env._q[env_ids, offset:offset + dofs], expected_q[:, order])
                    torch.testing.assert_close(env._qd[env_ids, offset:offset + dofs], expected_dq[:, order])
                    torch.testing.assert_close(env._pos_control[env_ids, offset:offset + dofs], expected_q[:, order])
                    self.assertTrue(torch.equal(env._q[1, offset:offset + dofs], torch.zeros(dofs)))
                    env._update_states()
                    torch.testing.assert_close(getattr(env, f"{side}_states")["q"][env_ids], expected_q)

    def test_full_actions_scale_and_smooth_canonical_targets_before_simulator_io(self):
        for name in SUPPORTED_HANDS:
            with self.subTest(hand=name):
                env = fixture(name)
                dofs = env.num_dexhand_rh_dofs
                actions = torch.linspace(-1.4, 1.4, 12 + 2 * dofs).repeat(env.num_envs, 1)
                expected = []
                for side, offset, action_offset in (("rh", 0, 6), ("lh", dofs, 12 + dofs)):
                    low, high, _, _ = env.expected_limits[side]
                    old = low + (high - low) * 0.35
                    order = getattr(env, f"_{side}_sim_from_policy")
                    env.prev_targets[:, offset:offset + dofs] = old[order]
                    normalized = actions[:, action_offset:action_offset + dofs].clamp(-1, 1)
                    scaled = low + (normalized + 1) * 0.5 * (high - low)
                    smoothed = 0.4 * scaled + 0.6 * old
                    expected.append(smoothed[:, order])
                env.pre_physics_step(actions)
                torch.testing.assert_close(env.gym.last_targets, torch.cat(expected, dim=1))
                torch.testing.assert_close(env._pos_control, env.gym.last_targets)
                for side, action_offset in (("rh", 0), ("lh", 6 + dofs)):
                    root = getattr(env, f"dexhand_{side}_handles")[env.hand_specs[side].body_names[0]]
                    torch.testing.assert_close(env.gym.last_forces[0][:, root], 0.4 * actions[:, action_offset:action_offset + 3] * env.dt * env.translation_scale * 500)
                    torch.testing.assert_close(env.gym.last_forces[1][:, root], 0.4 * actions[:, action_offset + 3:action_offset + 6] * env.dt * env.orientation_scale * 200)

    def test_actual_target_generation_matches_layout_and_clamps_future_frames(self):
        for name in SUPPORTED_HANDS:
            for k in (1, 3):
                for side in ("rh", "lh"):
                    with self.subTest(hand=name, future=k, side=side):
                        env = fixture(name, future_length=k)
                        layout = ObservationLayout(env.hand_specs[side], future_length=k)
                        result = env.compute_observations_side(side)
                        self.assertEqual(tuple(result["proprioception"].shape), (3, layout.hand_proprioception_dim))
                        self.assertEqual(tuple(result["privileged"].shape), (3, layout.hand_privileged_dim))
                        self.assertEqual(tuple(result["target"].shape), (3, layout.hand_target_dim))
                        fields = layout.split_target(result["target"])
                        data, states = getattr(env, f"demo_data_{side}"), getattr(env, f"{side}_states")
                        # Env0 is before frame zero; env1/2 reach their final
                        # valid frame, including every requested future frame.
                        self.assertEqual(data["seq_len"].tolist(), [1, 3, 5])
                        for batch, final_frame in enumerate((0, 2, 4)):
                            wrist = data["wrist_pos"][batch, final_frame].repeat(k, 1) - states["base_state"][batch, :3]
                            torch.testing.assert_close(fields["delta_wrist_pos"][batch].reshape(k, 3), wrist)
                            for field, source in (("wrist_vel", "wrist_velocity"), ("wrist_ang_vel", "wrist_angular_velocity"), ("manip_obj_vel", "obj_velocity"), ("manip_obj_ang_vel", "obj_angular_velocity")):
                                torch.testing.assert_close(fields[field][batch].reshape(k, 3), data[source][batch, final_frame].repeat(k, 1))
                            for field, source, current in (
                                ("delta_wrist_vel", "wrist_velocity", states["base_state"][batch, 7:10]),
                                ("delta_wrist_ang_vel", "wrist_angular_velocity", states["base_state"][batch, 10:13]),
                                ("delta_manip_obj_vel", "obj_velocity", states["manip_obj_vel"][batch]),
                                ("delta_manip_obj_ang_vel", "obj_angular_velocity", states["manip_obj_ang_vel"][batch]),
                            ):
                                expected = data[source][batch, final_frame].repeat(k, 1) - current
                                torch.testing.assert_close(fields[field][batch].reshape(k, 3), expected)
                            expected_obj_delta = data["obj_trajectory"][batch, final_frame, :3, 3] - states["manip_obj_pos"][batch]
                            torch.testing.assert_close(fields["delta_manip_obj_pos"][batch].reshape(k, 3), expected_obj_delta.repeat(k, 1))
                            for field, source, current in (("delta_joints_pos", "mano_joints", states["joints_state"][batch, 1:, :3]), ("delta_joints_vel", "mano_joints_velocity", states["joints_state"][batch, 1:, 7:10])):
                                expected = data[source][batch, final_frame].reshape(-1, 3) - current
                                torch.testing.assert_close(fields[field][batch].reshape(k, -1, 3), expected.repeat(k, 1, 1))
                            torch.testing.assert_close(fields["joints_vel"][batch].reshape(k, -1, 3), data["mano_joints_velocity"][batch, final_frame].reshape(-1, 3).repeat(k, 1, 1))
                            target_wrist_rotation = Rotation.from_rotvec(data["wrist_rot"][batch, final_frame].numpy())
                            target_object_rotation = Rotation.from_matrix(data["obj_trajectory"][batch, final_frame, :3, :3].numpy())
                            for field, rotation in (("wrist_quat", target_wrist_rotation), ("manip_obj_quat", target_object_rotation)):
                                expected_quat = torch.as_tensor(rotation.as_quat(), dtype=torch.float32).repeat(k, 1)
                                torch.testing.assert_close(fields[field][batch].reshape(k, 4), expected_quat)
                            for field, current_quat, target_rotation in (
                                ("delta_wrist_quat", states["base_state"][batch, 3:7], target_wrist_rotation),
                                ("delta_manip_obj_quat", states["manip_obj_quat"][batch], target_object_rotation),
                            ):
                                delta = Rotation.from_quat(current_quat.numpy()) * target_rotation.inv()
                                expected_quat = torch.as_tensor(delta.as_quat(), dtype=torch.float32).repeat(k, 1)
                                actual_quat = fields[field][batch].reshape(k, 4)
                                # SciPy versions choose either sign for inv();
                                # q and -q encode the same physical rotation.
                                alignment = torch.where((actual_quat * expected_quat).sum(-1, keepdim=True) < 0, -1.0, 1.0)
                                torch.testing.assert_close(actual_quat, expected_quat * alignment)
                            torch.testing.assert_close(fields["gt_tips_distance"][batch].reshape(k, 5), data["tips_distance"][batch, final_frame].repeat(k, 1))
                        expected_distance = torch.linalg.vector_norm(states["manip_obj_pos"][:, None] - states["joints_state"][:, :, :3], dim=-1)
                        torch.testing.assert_close(fields["obj_to_joints"], expected_distance)
                        torch.testing.assert_close(fields["bps"], getattr(env, f"obj_bps_{side}"))
                        # Base translation is suppressed only in proprioception.
                        dofs = env.hand_specs[side].n_dofs
                        self.assertTrue(torch.equal(result["proprioception"][:, 3 * dofs:3 * dofs + 3], torch.zeros(3, 3)))

    def test_multiple_future_frames_preserve_frame_and_link_order_before_endpoint(self):
        env = fixture("inspire", future_length=3)
        env.progress_buf = torch.tensor([1, 1, 2])
        layout = ObservationLayout(env.hand_specs["rh"], future_length=3)
        fields = layout.split_target(env.compute_observations_side("rh")["target"])
        for batch, frames in enumerate(((0, 0, 0), (0, 1, 2), (1, 2, 3))):
            for future, frame in enumerate(frames):
                torch.testing.assert_close(fields["wrist_vel"][batch].reshape(3, 3)[future], env.demo_data_rh["wrist_velocity"][batch, frame])
                torch.testing.assert_close(fields["joints_vel"][batch].reshape(3, -1, 3)[future], env.demo_data_rh["mano_joints_velocity"][batch, frame].reshape(-1, 3))

    def test_reward_power_pairs_canonical_velocities_with_matching_joint_forces(self):
        for name in SUPPORTED_HANDS:
            with self.subTest(hand=name):
                env = fixture(name)
                env.progress_buf = torch.tensor([0, 1, 2])
                env.headless = False  # Permit the capture callback instead of TorchScript reward.
                dofs = env.num_dexhand_rh_dofs
                for side, offset in (("rh", 0), ("lh", dofs)):
                    canonical_velocity = torch.arange(1, dofs + 1).float().repeat(3, 1)
                    canonical_force = torch.arange(dofs, 0, -1).float().repeat(3, 1)
                    order = getattr(env, f"_{side}_sim_from_policy")
                    env._qd[:, offset:offset + dofs] = canonical_velocity[:, order]
                    env.dof_force[:, offset:offset + dofs] = canonical_force[:, order]
                    env._update_states()
                    env.compute_reward_side(env.actions, side)
                    torch.testing.assert_close(REWARD_CAPTURE.target["power"], (canonical_velocity * canonical_force).sum(dim=-1))

    def test_unknown_or_missing_asset_joints_and_invalid_controls_fail_early(self):
        env = fixture("shadow")
        with self.assertRaisesRegex(RuntimeError, "Failed to load"):
            env._configure_hand_joint_order(None, "rh")
        env.gym.assets["rh"]["dof_names"] = env.gym.assets["rh"]["dof_names"][:-1] + ("wrong_joint",)
        with self.assertRaisesRegex(ValueError, "DOFs differ"):
            env._configure_hand_joint_order("rh", "rh")
        for bad in ({"usePIDControl": True}, {"useQuatRot": True}, {"policyMode": "residual"}, {"bimanual_mode": "independent"}):
            with self.subTest(config=bad), self.assertRaises(ValueError):
                validate_direct_control(bad)
        with self.assertRaisesRegex(ValueError, "Expected complete bimanual actions"):
            env.pre_physics_step(torch.zeros(3, 54))


if __name__ == "__main__":
    unittest.main()
