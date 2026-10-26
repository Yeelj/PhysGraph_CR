#!/usr/bin/env python3
"""Replay bimanual TACO or OakInk-V2 retargeting results in Isaac Gym.

The retargeting pickle contains only dexterous-hand states.  Object assets,
object trajectories, and the source-to-Gym transform are intentionally loaded
through the corresponding ``ManipDataFactory`` datasets so this viewer uses
the same data path as training.
"""

import argparse
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Isaac Gym must be imported before torch.
from isaacgym import gymapi, gymtorch, gymutil

import numpy as np
import torch

from main.contact_retarget_memory import right_hand_retarget_path
from main.dataset.factory import ManipDataFactory
from main.dataset.taco_dataset_utils import normalize_taco_object_id
from main.dataset.transform import aa_to_quat, aa_to_rotmat, rotmat_to_quat
from physgraph_envs.lib.envs.dexhands.factory import DexHandFactory


TABLE_WIDTH_OFFSET = 0.2
TABLE_HEIGHT = 0.03
TABLE_POSITION = (-TABLE_WIDTH_OFFSET / 2, 0.0, 0.4)
TABLE_SURFACE_Z = TABLE_POSITION[2] + TABLE_HEIGHT / 2
REQUIRED_RETARGET_KEYS = ("opt_wrist_pos", "opt_wrist_rot", "opt_dof_pos")
DATASET_NAMES = {"taco": "TACO", "oakink2": "OakInk-v2"}
# State key -> CLI component, actor name, legacy display color.
REPLAY_COMPONENTS = {
    "hand_rh": ("right_hand", "dexhand_r", (0.35, 0.75, 0.95)),
    "hand_lh": ("left_hand", "dexhand_l", (0.75, 0.45, 0.95)),
    "object_rh": ("tool", "manip_obj_rh", (1.0, 0.45, 0.08)),
    "object_lh": ("object", "manip_obj_lh", (0.2, 0.7, 0.35)),
}


def _parse_color(value: str):
    """Accept an explicit solid RGB override or the URDF material mode."""
    if value.strip().lower() == "urdf":
        return "urdf"
    message = "Expected urdf or three comma-separated RGB numbers in [0, 1], e.g. 0.3,0.6,1.0"
    try:
        color = tuple(float(channel) for channel in value.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(message) from None
    if len(color) != 3 or any(not math.isfinite(channel) or not 0 <= channel <= 1 for channel in color):
        raise argparse.ArgumentTypeError(message)
    return color


def _retarget_path(
    dexhand, data_idx: str, dataset_name: str = "TACO", *, filename: Optional[str] = None
) -> Path:
    return (
        PROJECT_ROOT / "data" / "retargeting" / dataset_name / f"mano2{dexhand}"
        / (filename if filename is not None else f"{data_idx}.pkl")
    )


def _right_retarget_path(
    dexhand,
    data_idx: str,
    new_obj: Optional[str],
    correct: bool,
    *,
    dataset_name: str = "TACO",
    filename: Optional[str] = None,
):
    if new_obj is not None and dataset_name == "TACO":
        new_obj = normalize_taco_object_id(new_obj)
    return right_hand_retarget_path(
        PROJECT_ROOT / "data" / "retargeting",
        PROJECT_ROOT / "data" / "retargeting_contact",
        f"mano2{dexhand}",
        data_idx,
        new_obj,
        filename if filename is not None else f"{data_idx}.pkl",
        correct=correct,
        dataset_name=dataset_name,
    )


def _load_and_validate_pickle(path: Path, expected_dofs: int, retarget_kind: str = "original"):
    if not path.is_file():
        if retarget_kind == "contact_corrected":
            hint = "Run main/dataset/contact_trajectory_retarget.py correct for this replacement mesh."
        else:
            hint = "Run main/dataset/mano2dexhand.py for this sequence, hand side, and dexhand first."
        raise FileNotFoundError(
            f"Missing {retarget_kind} retargeting result: {path}. {hint}"
        )

    with path.open("rb") as stream:
        data = pickle.load(stream)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a dict in {path}, got {type(data).__name__}")

    missing = [key for key in REQUIRED_RETARGET_KEYS if key not in data]
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}")

    expected_tail_shapes = {
        "opt_wrist_pos": (3,),
        "opt_wrist_rot": (3,),
        "opt_dof_pos": (expected_dofs,),
    }
    frame_count = None
    for key, tail_shape in expected_tail_shapes.items():
        value = np.asarray(data[key])
        if value.ndim != 2 or value.shape[1:] != tail_shape:
            raise ValueError(
                f"{path}: {key} must have shape [T, {', '.join(map(str, tail_shape))}], "
                f"got {value.shape}"
            )
        if value.shape[0] < 1:
            raise ValueError(f"{path}: {key} contains no frames")
        if not np.isfinite(value).all():
            raise ValueError(f"{path}: {key} contains NaN or Inf")
        if frame_count is None:
            frame_count = value.shape[0]
        elif value.shape[0] != frame_count:
            raise ValueError(
                f"{path}: inconsistent frame counts; expected {frame_count}, "
                f"but {key} has {value.shape[0]}"
            )
    return data, frame_count


def _mujoco_to_gym_transform(device: torch.device) -> torch.Tensor:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2], dtype=np.float32)) @ aa_to_rotmat(
        np.array([np.pi / 2, 0, 0], dtype=np.float32)
    )
    transform[:3, 3] = np.array([0, 0, TABLE_SURFACE_Z], dtype=np.float32)
    return torch.as_tensor(transform, dtype=torch.float32, device=device)


def _validate_loaded_data(
    data, side: str, frame_count: int, expected_dofs: int, dataset_name: str = "TACO"
) -> None:
    expected_shapes = {
        "opt_wrist_pos": (frame_count, 3),
        "opt_wrist_rot": (frame_count, 3),
        "opt_dof_pos": (frame_count, expected_dofs),
        "obj_trajectory": (frame_count, 4, 4),
    }
    for key, expected_shape in expected_shapes.items():
        if key not in data:
            raise KeyError(f"The {side}-hand {dataset_name} sample is missing {key!r}")
        if tuple(data[key].shape) != expected_shape:
            raise ValueError(
                f"The {side}-hand {dataset_name} sample has {key}.shape={tuple(data[key].shape)}, "
                f"expected {expected_shape}"
            )
        if not torch.isfinite(data[key]).all().item():
            raise ValueError(f"The {side}-hand {dataset_name} sample has NaN or Inf in {key}")

    urdf_path = Path(data["obj_urdf_path"])
    if not urdf_path.is_file():
        raise FileNotFoundError(f"Missing {side}-hand object URDF: {urdf_path}")


def _load_replay_data(args, device, dexhand_rh, dexhand_lh):
    if not args.data_idx:
        raise ValueError("--data_idx is required")
    dataset_type = ManipDataFactory.dataset_type(args.data_idx)
    if dataset_type not in DATASET_NAMES:
        raise ValueError(
            "Expected a TACO sequence ID (e.g. db_teapot_0930049) or an "
            f"OakInk-V2 sequence ID (e.g. 083f7@0), got {args.data_idx!r}"
        )
    dataset_name = DATASET_NAMES[dataset_type]
    data_idx = args.data_idx
    if dataset_type == "oakink2":
        parts = data_idx.split("@")
        if len(parts) != 2 or len(parts[0]) != 5 or not parts[1].isdigit():
            raise ValueError(f"Expected an OakInk-V2 ID in hash@stage form, e.g. 083f7@0, got {data_idx!r}")
        sequence_hash, stage_text = parts
        stage = int(stage_text)
        data_idx = f"{sequence_hash}@{stage}"
    if args.correct and not args.new_obj:
        raise ValueError("--correct requires --new_obj to select the right-hand corrected trajectory")

    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data" / dataset_name
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    transform = _mujoco_to_gym_transform(device)
    common_kwargs = {
        "data_dir": str(data_dir),
        "device": device,
        "mujoco2gym_transf": transform,
        "dexhand": None,
        "verbose": False,
    }
    rh_kwargs = {
        **common_kwargs,
        "dexhand": dexhand_rh,
        "new_obj": args.new_obj or None,
        "correct": args.correct,
    }
    if dataset_type == "oakink2":
        rh_kwargs["load_tool_semantics"] = False
    rh_dataset = ManipDataFactory.create_data(
        manipdata_type=dataset_type,
        side="right",
        **rh_kwargs,
    )
    lh_dataset = ManipDataFactory.create_data(
        manipdata_type=dataset_type,
        side="left",
        **{**common_kwargs, "dexhand": dexhand_lh},
    )
    if dataset_type == "oakink2":
        # OakInk PKLs use the full annotation filename, while the dataset and
        # corrected-trajectory directory use the compact hash@stage ID.
        if sequence_hash not in rh_dataset.seq_hashes:
            raise ValueError(f"OakInk-V2 sequence hash {sequence_hash!r} not found under {data_dir}")
        annotation_path = Path(rh_dataset.data_pathes[rh_dataset.seq_hashes[sequence_hash]])
        filename = f"{annotation_path.stem}@{stage}.pkl"
    else:
        filename = f"{data_idx}.pkl"
    rh_pickle, rh_retarget_kind = _right_retarget_path(
        dexhand_rh,
        data_idx,
        args.new_obj or None,
        args.correct,
        dataset_name=dataset_name,
        filename=filename,
    )
    lh_pickle = _retarget_path(dexhand_lh, data_idx, dataset_name, filename=filename)
    rh_retarget, rh_frames = _load_and_validate_pickle(
        rh_pickle, dexhand_rh.n_dofs, rh_retarget_kind
    )
    lh_retarget, lh_frames = _load_and_validate_pickle(lh_pickle, dexhand_lh.n_dofs)
    if rh_frames != lh_frames:
        raise ValueError(f"Right/left retarget frame mismatch: {rh_frames} vs {lh_frames}")

    rh_data = rh_dataset[data_idx]
    lh_data = lh_dataset[data_idx]
    # Replay the selected, validated PKLs directly. Dataset loading above also
    # preserves the training-time checks for corrected trajectories and meshes.
    for key in REQUIRED_RETARGET_KEYS:
        rh_data[key] = torch.as_tensor(rh_retarget[key], dtype=torch.float32, device=device)
        lh_data[key] = torch.as_tensor(lh_retarget[key], dtype=torch.float32, device=device)
    _validate_loaded_data(rh_data, "right", rh_frames, dexhand_rh.n_dofs, dataset_name)
    _validate_loaded_data(lh_data, "left", lh_frames, dexhand_lh.n_dofs, dataset_name)

    print(f"Dataset: {dataset_name}")
    print(f"Sequence: {data_idx}")
    print(f"Frames: {rh_frames} at {args.playback_fps:g} FPS")
    print(f"Right retarget ({rh_retarget_kind}): {rh_pickle}")
    print(f"Left retarget:  {lh_pickle}")
    print(f"Right tool: {rh_data.get('selected_obj_id', rh_data['obj_id'])} ({rh_data['obj_urdf_path']})")
    print(f"Left target: {lh_data['obj_id']} ({lh_data['obj_urdf_path']})")
    return rh_data, lh_data, rh_frames


class TacoRetargetViewer:
    """Shared viewer for TACO and OakInk-V2; keep the original public name."""

    def __init__(self, args):
        if args.physics_engine != gymapi.SIM_PHYSX:
            raise ValueError("This viewer supports the PhysX backend only")
        if args.playback_fps <= 0:
            raise ValueError(f"playback_fps must be positive, got {args.playback_fps}")
        if args.start_frame < 0:
            raise ValueError(f"start_frame must be non-negative, got {args.start_frame}")

        self.args = args
        self.gym = gymapi.acquire_gym()
        self.sim = None
        self.viewer = None
        self.device = torch.device(args.sim_device if args.use_gpu_pipeline else "cpu")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is required for device {self.device}, but torch.cuda.is_available() is false")

        try:
            self.dexhand_rh = DexHandFactory.create_hand(args.dexhand, "right")
            self.dexhand_lh = DexHandFactory.create_hand(args.dexhand, "left")
            self.rh_data, self.lh_data, self.frame_count = _load_replay_data(
                args, self.device, self.dexhand_rh, self.dexhand_lh
            )
            if args.start_frame >= self.frame_count:
                raise ValueError(
                    f"start_frame={args.start_frame} is outside a sequence with {self.frame_count} frames"
                )

            self._create_sim()
            self._create_scene()
            self.gym.prepare_sim(self.sim)
            self._acquire_state_tensors()
            self._prepare_trajectories()
            self._create_viewer()
        except Exception:
            self.close()
            raise

    def _create_sim(self):
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / self.args.playback_fps
        sim_params.substeps = 1
        sim_params.up_axis = gymapi.UP_AXIS_Z
        # Every pose is supplied by the recorded trajectory. Gravity is off so
        # the viewer does not introduce dynamics between consecutive frames.
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
        sim_params.use_gpu_pipeline = self.args.use_gpu_pipeline
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = self.args.num_threads
        sim_params.physx.use_gpu = self.args.use_gpu

        self.sim = self.gym.create_sim(
            self.args.compute_device_id,
            self.args.graphics_device_id,
            self.args.physics_engine,
            sim_params,
        )
        if self.sim is None:
            raise RuntimeError("Failed to create the Isaac Gym simulation")

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _load_hand_asset(self, dexhand, *, use_urdf_materials=False):
        options = gymapi.AssetOptions()
        options.thickness = 0.001
        options.fix_base_link = False
        options.disable_gravity = True
        options.flip_visual_attachments = False
        options.collapse_fixed_joints = False
        options.default_dof_drive_mode = int(gymapi.DOF_MODE_NONE)
        options.use_mesh_materials = not use_urdf_materials
        urdf_path = os.path.abspath(dexhand.urdf_path)
        asset = self.gym.load_asset(self.sim, os.path.dirname(urdf_path), os.path.basename(urdf_path), options)
        if asset is None:
            raise RuntimeError(f"Failed to load dexhand asset: {urdf_path}")

        asset_dof_names = list(self.gym.get_asset_dof_names(asset))
        if asset_dof_names != list(dexhand.dof_names):
            raise ValueError(
                f"DOF order mismatch for {dexhand}: asset has {asset_dof_names}, "
                f"retarget expects {dexhand.dof_names}"
            )
        return asset

    def _load_object_asset(self, urdf_path, *, use_urdf_materials=False):
        options = gymapi.AssetOptions()
        options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        options.thickness = 0.001
        options.fix_base_link = False
        options.disable_gravity = True
        options.collapse_fixed_joints = False
        options.use_mesh_materials = not use_urdf_materials
        urdf_path = os.path.abspath(urdf_path)
        asset = self.gym.load_asset(self.sim, os.path.dirname(urdf_path), os.path.basename(urdf_path), options)
        if asset is None:
            raise RuntimeError(f"Failed to load object asset: {urdf_path}")
        return asset

    def _create_scene(self):
        # Omitted arguments keep the legacy appearance and all four actors.
        self.colors = {
            key: getattr(self.args, f"{component}_color", None)
            for key, (component, _, _) in REPLAY_COMPONENTS.items()
        }
        visible = {
            key: not getattr(self.args, f"hide_{component}", False)
            for key, (component, _, _) in REPLAY_COMPONENTS.items()
        }
        self._solid_color_texture = None
        assets = {}
        for key, hand in (("hand_rh", self.dexhand_rh), ("hand_lh", self.dexhand_lh)):
            if visible[key]:
                assets[key] = self._load_hand_asset(hand, use_urdf_materials=self.colors[key] == "urdf")
        for key, data in (("object_rh", self.rh_data), ("object_lh", self.lh_data)):
            if visible[key]:
                assets[key] = self._load_object_asset(
                    data["obj_urdf_path"], use_urdf_materials=self.colors[key] == "urdf"
                )

        table_options = gymapi.AssetOptions()
        table_options.fix_base_link = True
        table_asset = self.gym.create_box(
            self.sim, 0.8 + TABLE_WIDTH_OFFSET, 1.6, TABLE_HEIGHT, table_options
        )

        lower = gymapi.Vec3(-1.0, -1.0, 0.0)
        upper = gymapi.Vec3(1.0, 1.0, 1.0)
        self.env = self.gym.create_env(self.sim, lower, upper, 1)
        identity = gymapi.Transform()
        # A shared non-zero filter bit suppresses contacts between replay
        # actors. Their poses should come solely from the recorded trajectory.
        replay_filter = 1
        self.actor_handles = {}
        for key in ("hand_rh", "hand_lh"):
            if visible[key]:
                self.actor_handles[key] = self.gym.create_actor(
                    self.env, assets[key], identity, REPLAY_COMPONENTS[key][1], 0, replay_filter
                )

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(*TABLE_POSITION)
        self.table_actor = self.gym.create_actor(
            self.env, table_asset, table_pose, "table", 0, replay_filter
        )
        for key in ("object_rh", "object_lh"):
            if visible[key]:
                self.actor_handles[key] = self.gym.create_actor(
                    self.env, assets[key], identity, REPLAY_COMPONENTS[key][1], 0, replay_filter
                )

        for key, (_, _, default_color) in REPLAY_COMPONENTS.items():
            actor = self.actor_handles.get(key)
            setattr(self, f"{key}_actor", actor)
            if actor is None or self.colors[key] == "urdf":
                continue
            color = self.colors[key]
            self._set_actor_color(
                actor, gymapi.Vec3(*(default_color if color is None else color)),
                solid=color is not None,
            )
        self._set_actor_color(self.table_actor, gymapi.Vec3(0.12, 0.12, 0.12))

    def _set_actor_color(self, actor, color, *, solid=False):
        if solid and self._solid_color_texture is None:
            # Gym owns this shared white texture until the simulation is destroyed.
            self._solid_color_texture = self.gym.create_texture_from_buffer(
                self.sim, 1, 1, np.full((1, 4), 255, dtype=np.uint8)
            )
            if self._solid_color_texture == gymapi.INVALID_HANDLE:
                raise RuntimeError("Failed to create the replay solid-color texture")
        body_count = self.gym.get_actor_rigid_body_count(self.env, actor)
        for body_index in range(body_count):
            if solid:
                self.gym.set_rigid_body_texture(
                    self.env, actor, body_index, gymapi.MESH_VISUAL, self._solid_color_texture
                )
            self.gym.set_rigid_body_color(self.env, actor, body_index, gymapi.MESH_VISUAL, color)

    def _acquire_state_tensors(self):
        self.actor_indices = {
            key: self.gym.get_actor_index(self.env, actor, gymapi.DOMAIN_SIM)
            for key, actor in self.actor_handles.items()
        }
        self.root_state = None
        if self.actor_indices:
            root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
            self.root_state = gymtorch.wrap_tensor(root_tensor).view(-1, 13)

        # Hands are created RH then LH, but either may be absent. Object/table
        # assets must remain rigid, as in the original bimanual replay.
        rh_dofs = self.dexhand_rh.n_dofs if self.hand_rh_actor is not None else 0
        lh_dofs = self.dexhand_lh.n_dofs if self.hand_lh_actor is not None else 0
        expected_dofs = rh_dofs + lh_dofs
        actual_dofs = self.gym.get_sim_dof_count(self.sim)
        if actual_dofs != expected_dofs:
            raise RuntimeError(
                f"Unexpected simulation DOF count: {actual_dofs}, expected {expected_dofs}"
            )
        self.rh_dof_slice = slice(0, rh_dofs) if rh_dofs else None
        self.lh_dof_slice = slice(rh_dofs, expected_dofs) if lh_dofs else None
        self.dof_state = None
        if expected_dofs:
            dof_tensor = self.gym.acquire_dof_state_tensor(self.sim)
            self.dof_state = gymtorch.wrap_tensor(dof_tensor).view(-1, 2)

    def _prepare_trajectories(self):
        self.hand_rh_pos = self.rh_data["opt_wrist_pos"].to(self.device, dtype=torch.float32)
        self.hand_lh_pos = self.lh_data["opt_wrist_pos"].to(self.device, dtype=torch.float32)
        self.hand_rh_quat = aa_to_quat(self.rh_data["opt_wrist_rot"]).to(self.device)[:, [1, 2, 3, 0]]
        self.hand_lh_quat = aa_to_quat(self.lh_data["opt_wrist_rot"]).to(self.device)[:, [1, 2, 3, 0]]
        self.hand_rh_dof = self.rh_data["opt_dof_pos"].to(self.device, dtype=torch.float32)
        self.hand_lh_dof = self.lh_data["opt_dof_pos"].to(self.device, dtype=torch.float32)

        rh_object = self.rh_data["obj_trajectory"].to(self.device, dtype=torch.float32)
        lh_object = self.lh_data["obj_trajectory"].to(self.device, dtype=torch.float32)
        self.object_rh_pos = rh_object[:, :3, 3]
        self.object_lh_pos = lh_object[:, :3, 3]
        self.object_rh_quat = rotmat_to_quat(rh_object[:, :3, :3])[:, [1, 2, 3, 0]]
        self.object_lh_quat = rotmat_to_quat(lh_object[:, :3, :3])[:, [1, 2, 3, 0]]

    def _create_viewer(self):
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise RuntimeError("Failed to create the Isaac Gym viewer")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "quit")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "pause")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "restart")

        positions = [getattr(self, f"{key}_pos") for key in self.actor_indices]
        if positions:
            all_positions = torch.cat(positions, dim=0)
            center = (all_positions.amin(dim=0) + all_positions.amax(dim=0)) / 2
            center = center.detach().cpu().numpy()
        else:
            center = np.array([TABLE_POSITION[0], TABLE_POSITION[1], TABLE_SURFACE_Z])
        camera_target = gymapi.Vec3(float(center[0]), float(center[1]), float(center[2]))
        camera_position = gymapi.Vec3(
            float(center[0] + 0.9), float(center[1] + 0.9), float(max(center[2] + 0.65, 0.8))
        )
        self.gym.viewer_camera_look_at(self.viewer, self.env, camera_position, camera_target)
        print("Controls: Space=pause/resume, R=restart, Esc=quit")

    def _write_root_pose(self, key, position, quaternion):
        if key not in self.actor_indices:
            return
        state = self.root_state[self.actor_indices[key]]
        state[:3] = position
        state[3:7] = quaternion
        state[7:13] = 0

    def _set_frame(self, frame: int):
        self._write_root_pose("hand_rh", self.hand_rh_pos[frame], self.hand_rh_quat[frame])
        self._write_root_pose("hand_lh", self.hand_lh_pos[frame], self.hand_lh_quat[frame])
        self._write_root_pose("object_rh", self.object_rh_pos[frame], self.object_rh_quat[frame])
        self._write_root_pose("object_lh", self.object_lh_pos[frame], self.object_lh_quat[frame])
        if self.dof_state is not None:
            self.dof_state[:, 1] = 0
            if self.rh_dof_slice is not None:
                self.dof_state[self.rh_dof_slice, 0] = self.hand_rh_dof[frame]
            if self.lh_dof_slice is not None:
                self.dof_state[self.lh_dof_slice, 0] = self.hand_lh_dof[frame]
        if self.root_state is not None:
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_state))
        if self.dof_state is not None:
            self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))

    def run(self):
        frame = self.args.start_frame
        paused = False
        quit_requested = False
        while not quit_requested and not self.gym.query_viewer_has_closed(self.viewer):
            restart_requested = False
            for event in self.gym.query_viewer_action_events(self.viewer):
                if event.value <= 0:
                    continue
                if event.action == "quit":
                    quit_requested = True
                elif event.action == "pause":
                    paused = not paused
                elif event.action == "restart":
                    restart_requested = True

            if quit_requested:
                break
            if restart_requested:
                frame = self.args.start_frame
                paused = False

            reached_end = False
            if not paused:
                self._set_frame(frame)
                self.gym.simulate(self.sim)
                self.gym.fetch_results(self.sim, True)
                reached_end = frame + 1 >= self.frame_count

            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)

            if not paused:
                if reached_end:
                    if self.args.once:
                        break
                    frame = self.args.start_frame
                else:
                    frame += 1

    def close(self):
        if self.viewer is not None:
            self.gym.destroy_viewer(self.viewer)
            self.viewer = None
        if self.sim is not None:
            self.gym.destroy_sim(self.sim)
            self.sim = None


def parse_args():
    return gymutil.parse_arguments(
        description="Replay bimanual TACO or OakInk-V2 retargeting results in Isaac Gym",
        headless=False,
        custom_parameters=[
            {
                "name": "--data_idx",
                "type": str,
                "help": "Sequence ID: TACO db_teapot_0930049 or OakInk-V2 083f7@0 (auto-detected)",
            },
            {
                "name": "--dexhand",
                "type": str,
                "default": "artimano",
                "help": "Dexhand embodiment used to create the retarget pickle",
            },
            {
                "name": "--data_dir",
                "type": str,
                "default": None,
                "help": "Dataset root (default: data/TACO or data/OakInk-v2, selected from --data_idx)",
            },
            {
                "name": "--new_obj",
                "type": str,
                "default": "",
                "help": "Right-hand tool replacement: TACO object ID (004) or OakInk-V2 new_obj name (kni@2)",
            },
            {
                "name": "--correct",
                "action": "store_true",
                "help": "Load the contact-corrected right-hand trajectory for --new_obj",
            },
            {
                "name": "--playback_fps",
                "type": float,
                "default": 60.0,
                "help": "Viewer playback rate",
            },
            {
                "name": "--start_frame",
                "type": int,
                "default": 0,
                "help": "First frame to play",
            },
            {
                "name": "--once",
                "action": "store_true",
                "help": "Exit after one playback instead of looping",
            },
        ] + [
            {
                "name": f"--{component}_color",
                "type": _parse_color,
                "default": None,
                "help": f"{component.replace('_', ' ').capitalize()} appearance: solid RGB r,g,b in [0,1], "
                        "or urdf for URDF materials (omitted: legacy color and textures)",
            }
            for component, _, _ in REPLAY_COMPONENTS.values()
        ] + [
            {
                "name": f"--hide_{component}",
                "action": "store_true",
                "help": f"Hide the {component.replace('_', ' ')} in this replay",
            }
            for component, _, _ in REPLAY_COMPONENTS.values()
        ],
    )


def main():
    args = parse_args()
    viewer = None
    try:
        viewer = TacoRetargetViewer(args)
        viewer.run()
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
