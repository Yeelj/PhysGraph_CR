"""Isolated Isaac Gym/PhysX contact replay for corrected hand trajectories.

Isaac Gym must be imported before packages which import torch.  Keeping this
validator in a subprocess prevents that global import-order requirement from
leaking into the offline geometric optimizer.
"""

from __future__ import annotations

import argparse
import json
import pickle
import traceback
from pathlib import Path
from typing import Dict, Sequence

from isaacgym import gymapi
import numpy as np
from scipy.spatial.transform import Rotation


def _pose(position: np.ndarray, rotation: np.ndarray) -> gymapi.Transform:
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(*[float(value) for value in position])
    quaternion = Rotation.from_matrix(rotation).as_quat()
    pose.r = gymapi.Quat(*[float(value) for value in quaternion])
    return pose


def _configure_shape_friction(gym, asset, friction: float = 1.0) -> None:
    properties = gym.get_asset_rigid_shape_properties(asset)
    for item in properties:
        item.friction = friction
        item.rolling_friction = 0.0
        item.torsion_friction = 0.0
        # Isaac Gym defaults to a 2 cm contact offset per shape, which makes
        # PhysX report roughly 4 cm of ``initialOverlap`` for merely nearby
        # surfaces.  Match the offline 6 mm contact envelope instead.
        item.contact_offset = 0.003
        item.rest_offset = 0.0
    gym.set_asset_rigid_shape_properties(asset, properties)


def _set_actor_root_pose(gym, environment, actor, position, rotation) -> None:
    states = gym.get_actor_rigid_body_states(environment, actor, gymapi.STATE_ALL)
    quaternion = Rotation.from_matrix(rotation).as_quat()
    states["pose"]["p"]["x"][0] = float(position[0])
    states["pose"]["p"]["y"][0] = float(position[1])
    states["pose"]["p"]["z"][0] = float(position[2])
    states["pose"]["r"]["x"][0] = float(quaternion[0])
    states["pose"]["r"]["y"][0] = float(quaternion[1])
    states["pose"]["r"]["z"][0] = float(quaternion[2])
    states["pose"]["r"]["w"][0] = float(quaternion[3])
    states["vel"]["linear"][0] = (0.0, 0.0, 0.0)
    states["vel"]["angular"][0] = (0.0, 0.0, 0.0)
    gym.set_actor_rigid_body_states(environment, actor, states, gymapi.STATE_ALL)


def _color_actor(gym, environment, actor, color: gymapi.Vec3) -> None:
    for body_index in range(gym.get_actor_rigid_body_count(environment, actor)):
        gym.set_rigid_body_color(
            environment,
            actor,
            body_index,
            gymapi.MESH_VISUAL,
            color,
        )


def _vector3(value) -> list:
    """Convert an Isaac Gym contact vector to a JSON-safe xyz list."""

    array = np.asarray(value)
    if array.dtype.names and all(name in array.dtype.names for name in ("x", "y", "z")):
        return [float(array[name]) for name in ("x", "y", "z")]
    flat = np.asarray(value, dtype=np.float64).reshape(-1)
    if len(flat) < 3:
        return [0.0, 0.0, 0.0]
    return [float(item) for item in flat[:3]]


def replay_corrected_trajectory(request: Dict) -> Dict:
    """Kinematically replay one corrected green hand and one moving tool."""

    wrist_pos = np.asarray(request["replay_wrist_pos"], dtype=np.float64)
    wrist_rot = np.asarray(request["replay_wrist_rot"], dtype=np.float64)
    dof_pos = np.asarray(request["replay_dof_pos"], dtype=np.float64)
    object_trajectory = np.asarray(request["replay_object_trajectory"], dtype=np.float64)
    length = len(wrist_pos)
    if not (
        wrist_rot.shape == (length, 3)
        and dof_pos.ndim == 2
        and len(dof_pos) == length
        and object_trajectory.shape == (length, 4, 4)
    ):
        raise ValueError("Corrected Viewer replay arrays have inconsistent shapes")

    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 1
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
    sim_params.use_gpu_pipeline = False
    sim_params.physx.use_gpu = False
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError(
            "Isaac Gym failed to create a graphics simulation. Check DISPLAY and the NVIDIA driver."
        )
    viewer = None
    try:
        hand_urdf = Path(request["hand_urdf"]).resolve()
        hand_options = gymapi.AssetOptions()
        hand_options.fix_base_link = False
        hand_options.disable_gravity = True
        hand_options.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
        hand_options.collapse_fixed_joints = False
        hand_asset = gym.load_asset(sim, str(hand_urdf.parent), hand_urdf.name, hand_options)

        object_urdf = Path(request["collision_urdf"]).resolve()
        object_options = gymapi.AssetOptions()
        object_options.fix_base_link = False
        object_options.disable_gravity = True
        object_options.vhacd_enabled = False
        object_asset = gym.load_asset(
            sim, str(object_urdf.parent), object_urdf.name, object_options
        )
        if hand_asset is None or object_asset is None:
            raise RuntimeError("Isaac Gym failed to load Viewer replay assets")

        environment = gym.create_env(
            sim,
            gymapi.Vec3(-1.0, -1.0, -1.0),
            gymapi.Vec3(1.0, 1.0, 1.0),
            1,
        )
        hand_actor = gym.create_actor(
            environment,
            hand_asset,
            _pose(
                wrist_pos[0],
                Rotation.from_rotvec(wrist_rot[0]).as_matrix(),
            ),
            "corrected_hand_green",
            0,
            0,
        )
        object_actor = gym.create_actor(
            environment,
            object_asset,
            _pose(object_trajectory[0, :3, 3], object_trajectory[0, :3, :3]),
            "tool",
            1,
            0,
        )
        actor_dof_names = gym.get_actor_dof_names(environment, hand_actor)
        source_name_to_index = {
            name: index for index, name in enumerate(request["hand_dof_names"])
        }
        missing = [name for name in actor_dof_names if name not in source_name_to_index]
        if missing:
            raise ValueError(f"Viewer hand asset has unmapped DOFs: {missing}")
        dof_order = np.asarray([source_name_to_index[name] for name in actor_dof_names])

        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            raise RuntimeError(
                "Isaac Gym failed to create Viewer. Check DISPLAY/X11 access and graphics drivers."
            )
        gym.prepare_sim(sim)
        trajectory_center = np.median(object_trajectory[:, :3, 3], axis=0)
        camera_position = trajectory_center + np.array([0.45, 0.45, 0.30])
        gym.viewer_camera_look_at(
            viewer,
            environment,
            gymapi.Vec3(*[float(value) for value in camera_position]),
            gymapi.Vec3(*[float(value) for value in trajectory_center]),
        )
        _color_actor(gym, environment, hand_actor, gymapi.Vec3(0.15, 0.85, 0.25))

        print(
            f"[Isaac Gym Viewer] 循环播放 {length} 帧：唯一工具 + 修正绿手；"
            "关闭 Viewer 结束回放。",
            flush=True,
        )
        rendered_frames = 0
        while not gym.query_viewer_has_closed(viewer):
            frame = rendered_frames % length
            _set_actor_root_pose(
                gym,
                environment,
                hand_actor,
                wrist_pos[frame],
                Rotation.from_rotvec(wrist_rot[frame]).as_matrix(),
            )
            _set_actor_root_pose(
                gym,
                environment,
                object_actor,
                object_trajectory[frame, :3, 3],
                object_trajectory[frame, :3, :3],
            )
            states = np.zeros(len(actor_dof_names), dtype=gymapi.DofState.dtype)
            states["pos"] = dof_pos[frame, dof_order].astype(np.float32)
            gym.set_actor_dof_states(
                environment, hand_actor, states, gymapi.STATE_ALL
            )
            # Viewer playback is purely kinematic.  The independent tier-three
            # validator already ran PhysX; simulating here would let contacts
            # alter the fixed PKL finger angles before rendering.
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)
            rendered_frames += 1
        return {
            "mode": "single_tool_corrected_green_kinematic_replay",
            "opened": True,
            "trajectory_length": int(length),
            "rendered_frames": int(rendered_frames),
            "closed_by_user": True,
            "corrected_hand_color": "green",
            "displayed_hand_actor_count": 1,
            "tool_actor_count": 1,
            "physics_simulation_during_replay": False,
        }
    finally:
        if viewer is not None:
            gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)


def run_validation(request: Dict) -> Dict:
    frame_ids = np.asarray(request["frame_ids"], dtype=np.int64)
    if len(frame_ids) == 0:
        raise ValueError("Physics validation requires at least one frame")
    wrist_pos = np.asarray(request["wrist_pos"], dtype=np.float64)
    wrist_rot = np.asarray(request["wrist_rot"], dtype=np.float64)
    dof_pos = np.asarray(request["dof_pos"], dtype=np.float64)
    object_trajectory = np.asarray(request["object_trajectory"], dtype=np.float64)
    expected_shapes = {
        "wrist_pos": (len(frame_ids), 3),
        "wrist_rot": (len(frame_ids), 3),
        "object_trajectory": (len(frame_ids), 4, 4),
    }
    for name, shape in expected_shapes.items():
        if np.asarray(request[name]).shape != shape:
            raise ValueError(f"Invalid {name} shape {np.asarray(request[name]).shape}; expected {shape}")
    if dof_pos.ndim != 2 or len(dof_pos) != len(frame_ids):
        raise ValueError(f"Invalid DOF trajectory shape {dof_pos.shape}")

    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.use_gpu_pipeline = False
    sim_params.physx.use_gpu = False
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 2
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("Isaac Gym failed to create a PhysX simulation")
    viewer = None
    try:
        hand_urdf = Path(request["hand_urdf"]).resolve()
        hand_options = gymapi.AssetOptions()
        hand_options.fix_base_link = True
        hand_options.disable_gravity = True
        hand_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
        hand_options.thickness = 0.001
        hand_options.collapse_fixed_joints = False
        hand_asset = gym.load_asset(sim, str(hand_urdf.parent), hand_urdf.name, hand_options)
        if hand_asset is None:
            raise RuntimeError(f"Isaac Gym failed to load hand asset {hand_urdf}")

        object_urdf = Path(request["collision_urdf"]).resolve()
        object_options = gymapi.AssetOptions()
        object_options.fix_base_link = True
        object_options.disable_gravity = True
        object_options.thickness = 0.001
        object_options.vhacd_enabled = False
        object_asset = gym.load_asset(
            sim, str(object_urdf.parent), object_urdf.name, object_options
        )
        if object_asset is None:
            raise RuntimeError(f"Isaac Gym failed to load object asset {object_urdf}")
        _configure_shape_friction(gym, hand_asset)
        _configure_shape_friction(gym, object_asset)

        spacing = 1.0
        lower = gymapi.Vec3(-0.5 * spacing, -0.5 * spacing, -0.5 * spacing)
        upper = gymapi.Vec3(0.5 * spacing, 0.5 * spacing, 0.5 * spacing)
        environments = []
        actors = []
        object_body_indices = []
        intended_body_indices = []
        hand_body_names_by_index = []
        canonical_hand_body_names = None
        source_dof_names: Sequence[str] = tuple(request["hand_dof_names"])
        source_name_to_index = {name: index for index, name in enumerate(source_dof_names)}
        num_per_row = max(1, int(np.ceil(np.sqrt(len(frame_ids)))))

        for local_index, frame_id in enumerate(frame_ids):
            environment = gym.create_env(sim, lower, upper, num_per_row)
            hand_rotation = Rotation.from_rotvec(wrist_rot[local_index]).as_matrix()
            hand_actor = gym.create_actor(
                environment,
                hand_asset,
                _pose(wrist_pos[local_index], hand_rotation),
                f"corrected_hand_{int(frame_id)}",
                local_index,
                0,
            )
            object_actor = gym.create_actor(
                environment,
                object_asset,
                _pose(
                    object_trajectory[local_index, :3, 3],
                    object_trajectory[local_index, :3, :3],
                ),
                f"tool_{int(frame_id)}",
                local_index,
                0,
            )

            actor_dof_names = gym.get_actor_dof_names(environment, hand_actor)
            missing = [name for name in actor_dof_names if name not in source_name_to_index]
            if missing:
                raise ValueError(f"Physics hand asset has unmapped DOFs: {missing}")
            targets = np.asarray(
                [dof_pos[local_index, source_name_to_index[name]] for name in actor_dof_names],
                dtype=np.float32,
            )
            properties = gym.get_actor_dof_properties(environment, hand_actor)
            properties["driveMode"].fill(int(gymapi.DOF_MODE_POS))
            properties["stiffness"].fill(400.0)
            properties["damping"].fill(40.0)
            gym.set_actor_dof_properties(environment, hand_actor, properties)
            states = np.zeros(len(actor_dof_names), dtype=gymapi.DofState.dtype)
            states["pos"] = targets
            gym.set_actor_dof_states(environment, hand_actor, states, gymapi.STATE_ALL)
            gym.set_actor_dof_position_targets(environment, hand_actor, targets)

            body_names = gym.get_actor_rigid_body_names(environment, hand_actor)
            if canonical_hand_body_names is None:
                canonical_hand_body_names = tuple(str(name) for name in body_names)
            body_name_by_index = {}
            for body_name in body_names:
                body_handle = gym.find_actor_rigid_body_handle(environment, hand_actor, body_name)
                body_index = gym.get_actor_rigid_body_index(
                    environment, hand_actor, body_handle, gymapi.DOMAIN_ENV
                )
                body_name_by_index[int(body_index)] = str(body_name)

            object_body_indices.append(
                gym.get_actor_rigid_body_index(
                    environment, object_actor, 0, gymapi.DOMAIN_ENV
                )
            )
            per_environment_intended = []
            for body_name in request.get("contact_body_names", ()):
                body_handle = gym.find_actor_rigid_body_handle(environment, hand_actor, body_name)
                if body_handle >= 0:
                    per_environment_intended.append(
                        gym.get_actor_rigid_body_index(
                            environment, hand_actor, body_handle, gymapi.DOMAIN_ENV
                        )
                    )
            intended_body_indices.append(set(per_environment_intended))
            hand_body_names_by_index.append(body_name_by_index)
            environments.append(environment)
            actors.append((hand_actor, object_actor))

        gym.prepare_sim(sim)
        for _ in range(max(1, int(request.get("settle_steps", 8)))):
            gym.simulate(sim)
            gym.fetch_results(sim, True)

        per_frame = []
        object_contact_frames = 0
        intended_contact_frames = 0
        maximum_overlap = 0.0
        maximum_force = 0.0
        finite = True
        min_contact_ratio = float(request.get("min_contact_frame_ratio", 0.5))
        max_force_limit = float(request.get("max_contact_force", 1000.0))
        max_overlap_limit = float(request.get("max_initial_overlap", 0.01))
        dof_lower = None
        dof_upper = None
        for local_index, (frame_id, environment) in enumerate(zip(frame_ids, environments)):
            contacts = gym.get_env_rigid_contacts(environment)
            if len(contacts):
                object_index = object_body_indices[local_index]
                object_mask = (contacts["body0"] == object_index) | (
                    contacts["body1"] == object_index
                )
                object_contacts = contacts[object_mask]
            else:
                object_contacts = contacts
            contact_count = int(len(object_contacts))
            if contact_count:
                object_contact_frames += 1
                overlaps = np.asarray(object_contacts["initialOverlap"], dtype=np.float64)
                forces = np.abs(np.asarray(object_contacts["lambda"], dtype=np.float64)) / sim_params.dt
                frame_overlap = float(np.max(np.maximum(overlaps, 0.0)))
                frame_force = float(np.max(forces))
                maximum_overlap = max(maximum_overlap, frame_overlap)
                maximum_force = max(maximum_force, frame_force)
                finite = bool(finite and np.isfinite(overlaps).all() and np.isfinite(forces).all())
                intended = intended_body_indices[local_index]
                hand_body = np.where(
                    object_contacts["body0"] == object_index,
                    object_contacts["body1"],
                    object_contacts["body0"],
                )
                intended_count = int(np.isin(hand_body, list(intended)).sum())
                if intended_count:
                    intended_contact_frames += 1

                worst_index = int(
                    np.argmax(forces)
                    if frame_force > max_force_limit
                    else np.argmax(np.maximum(overlaps, 0.0))
                )
                worst_contact = object_contacts[worst_index]
                worst_hand_body_index = int(hand_body[worst_index])
                worst_hand_body_name = hand_body_names_by_index[local_index].get(
                    worst_hand_body_index, f"body_{worst_hand_body_index}"
                )
                normal = np.asarray(_vector3(worst_contact["normal"]), dtype=np.float64)
                # PhysX reports the normal as the force direction on body0.  The
                # corresponding escape direction for the hand is therefore the
                # normal for body0 and its negation for body1.
                hand_escape_normal = (
                    normal
                    if int(worst_contact["body0"]) == worst_hand_body_index
                    else -normal
                )
            else:
                frame_overlap = 0.0
                frame_force = 0.0
                intended_count = 0
                worst_hand_body_name = None
                hand_escape_normal = np.zeros(3, dtype=np.float64)

            hand_actor = actors[local_index][0]
            if dof_lower is None:
                actor_dof_names = gym.get_actor_dof_names(environment, hand_actor)
                actor_properties = gym.get_actor_dof_properties(environment, hand_actor)
                dof_lower = np.full(len(source_dof_names), -np.inf, dtype=np.float64)
                dof_upper = np.full(len(source_dof_names), np.inf, dtype=np.float64)
                for actor_index, dof_name in enumerate(actor_dof_names):
                    source_index = source_name_to_index[dof_name]
                    dof_lower[source_index] = float(actor_properties["lower"][actor_index])
                    dof_upper[source_index] = float(actor_properties["upper"][actor_index])
            per_frame.append(
                {
                    "frame": int(frame_id),
                    "object_contact_count": contact_count,
                    "intended_link_contact_count": intended_count,
                    "maximum_initial_overlap": frame_overlap,
                    "maximum_contact_force": frame_force,
                    "worst_hand_body_name": worst_hand_body_name,
                    "hand_escape_normal": hand_escape_normal.tolist(),
                }
            )

        object_contact_ratio = float(object_contact_frames / len(frame_ids))
        intended_contact_ratio = float(intended_contact_frames / len(frame_ids))
        passed = bool(
            finite
            and object_contact_ratio >= min_contact_ratio
            and intended_contact_ratio >= min_contact_ratio
            and maximum_force <= max_force_limit # 控制接触力大小
            and maximum_overlap <= max_overlap_limit 
        )

        return {
            "name": "isaac_gym_physics",
            "mode": "fixed_tool_contact_replay",
            "status": "passed" if passed else "failed",
            "passed": passed,
            "headless": bool(request.get("headless", False)),
            "frame_ids": frame_ids.tolist(),
            "frame_count": int(len(frame_ids)),
            "object_contact_frame_count": int(object_contact_frames),
            "object_contact_frame_ratio": object_contact_ratio,
            "intended_link_contact_frame_count": int(intended_contact_frames),
            "intended_link_contact_frame_ratio": intended_contact_ratio,
            "maximum_initial_overlap": maximum_overlap,
            "maximum_contact_force": maximum_force,
            "all_finite": finite,
            "hand_body_names": list(canonical_hand_body_names or ()),
            "dof_lower_limits": None if dof_lower is None else dof_lower.tolist(),
            "dof_upper_limits": None if dof_upper is None else dof_upper.tolist(),
            "thresholds": {
                "min_contact_frame_ratio": min_contact_ratio,
                "max_contact_force": max_force_limit,
                "max_initial_overlap": max_overlap_limit,
            },
            "per_frame": per_frame,
            "scope_note": (
                "PhysX replays fixed hand/tool poses to validate collision response and contact-force "
                "sanity. Whole-task dynamic grasp stability remains a Stage-1 simulation property."
            ),
        }
    finally:
        if viewer is not None:
            gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    result_path = Path(args.result)
    try:
        with Path(args.request).open("rb") as stream:
            request = pickle.load(stream)
        result = run_validation(request)
        if not request.get("headless", False):
            result["viewer_replay"] = replay_corrected_trajectory(request)
    except Exception as error:
        result = {
            "name": "isaac_gym_physics",
            "status": "error",
            "passed": False,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return 0 if result.get("passed", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
