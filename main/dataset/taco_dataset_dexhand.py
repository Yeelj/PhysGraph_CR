import os
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from manotorch.manolayer import ManoLayer
from pytorch3d.structures import Meshes
from scipy.spatial.transform import Rotation, Slerp

from main.dataset.transform import aa_to_rotmat
from main.contact_retarget_memory import right_hand_retarget_path

from .base import ManipData
from .decorators import register_manipdata
from .taco_dataset_utils import (
    TACO_OUTPUT_FPS,
    TACO_SOURCE_FPS,
    build_taco_sequence_index,
    extract_taco_red_black_regions,
    load_taco_meter_mesh,
    load_taco_source_mesh,
    normalize_taco_object_id,
    resample_taco_pose_trajectory,
    resolve_taco_object_assets,
    taco_normalization_transform,
)


MANIPTRANS_FPS = TACO_OUTPUT_FPS
RETARGETING_ROOT_ORIGINAL = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "retargeting")
)
RETARGETING_ROOT_CORRECTED = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "retargeting_contact")
)


def _load_meter_mesh(mesh_path: str) -> trimesh.Trimesh:
    return load_taco_source_mesh(mesh_path)


def _linear_resample(data: torch.Tensor, source_fps: int, target_fps: int) -> torch.Tensor:
    if source_fps == target_fps:
        return data
    if target_fps % source_fps != 0:
        raise ValueError(f"Only integer upsampling is supported, got {source_fps} -> {target_fps} FPS")
    if data.shape[0] < 2:
        raise ValueError("A TACO sequence must contain at least two frames")
    target_length = (data.shape[0] - 1) * (target_fps // source_fps) + 1
    flattened = data.reshape(data.shape[0], -1).transpose(0, 1).unsqueeze(0)
    resampled = F.interpolate(flattened, size=target_length, mode="linear", align_corners=True)
    return resampled.squeeze(0).transpose(0, 1).reshape(target_length, *data.shape[1:])


def _rotation_resample(rotation_matrices: torch.Tensor, source_fps: int, target_fps: int) -> torch.Tensor:
    if source_fps == target_fps:
        return rotation_matrices
    if target_fps % source_fps != 0:
        raise ValueError(f"Only integer upsampling is supported, got {source_fps} -> {target_fps} FPS")
    source_length = rotation_matrices.shape[0]
    if source_length < 2:
        raise ValueError("A TACO sequence must contain at least two frames")
    target_length = (source_length - 1) * (target_fps // source_fps) + 1
    source_time = np.arange(source_length, dtype=np.float64) / source_fps
    target_time = np.arange(target_length, dtype=np.float64) / target_fps
    matrices = rotation_matrices.detach().cpu().numpy()
    interpolated = Slerp(source_time, Rotation.from_matrix(matrices))(target_time).as_matrix()
    return torch.as_tensor(interpolated, dtype=rotation_matrices.dtype, device=rotation_matrices.device)


def _validate_rigid_trajectory(name: str, trajectory: np.ndarray) -> None:
    if trajectory.ndim != 3 or trajectory.shape[1:] != (4, 4):
        raise ValueError(f"{name} must have shape [T, 4, 4], got {trajectory.shape}")
    if trajectory.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two frames")
    if not np.isfinite(trajectory).all():
        raise ValueError(f"{name} contains NaN or Inf")
    rotations = trajectory[:, :3, :3]
    identity = np.eye(3, dtype=trajectory.dtype)[None]
    if not np.allclose(rotations @ rotations.transpose(0, 2, 1), identity, atol=1e-4):
        raise ValueError(f"{name} contains non-orthonormal rotations")
    if not np.allclose(trajectory[:, 3], np.array([0, 0, 0, 1], dtype=trajectory.dtype), atol=1e-6):
        raise ValueError(f"{name} contains invalid homogeneous transforms")


class TacoDatasetDexHand(ManipData):
    side = None

    def __init__(
        self,
        *,
        data_dir: str = "data/TACO",
        split: str = "all",
        device="cuda:0",
        mujoco2gym_transf=None,
        max_seq_len=int(1e10),
        dexhand=None,
        mano_assets_root: str = "data/mano_v1_2",
        workspace_center=(-0.1, 0.0),
        floor_clearance: float = 0.003,
        max_hand_object_distance: float = 0.45,
        new_obj=None,
        correct=False,
        load_retargeted=True,
        **kwargs,
    ):
        if self.side not in ("left", "right"):
            raise ValueError("TacoDatasetDexHand must be specialized with side='left' or side='right'")
        super().__init__(
            data_dir=data_dir,
            split=split,
            skip=1,
            device=device,
            mujoco2gym_transf=mujoco2gym_transf,
            max_seq_len=max_seq_len,
            dexhand=dexhand,
            output_fps=MANIPTRANS_FPS,
            **kwargs,
        )
        self.sequence_index = build_taco_sequence_index(data_dir)
        self.data_pathes = [self.sequence_index[key].hand_pose_dir for key in sorted(self.sequence_index)]
        self._base_mujoco2gym_transf = mujoco2gym_transf.clone()
        self.workspace_center = np.asarray(workspace_center, dtype=np.float32)
        if self.workspace_center.shape != (2,):
            raise ValueError(f"workspace_center must contain two values, got {workspace_center}")
        self.floor_clearance = float(floor_clearance)
        self.max_hand_object_distance = float(max_hand_object_distance)
        self.new_obj = (
            None if new_obj is None else normalize_taco_object_id(new_obj)
        )
        self.correct = bool(correct)
        self.load_retargeted = bool(load_retargeted)
        if self.correct and self.side == "right" and self.new_obj is None:
            raise ValueError("TACO correct=true requires new_obj for the right-hand tool")

        self.manolayer = ManoLayer(
            rot_mode="axisang",
            side=self.side,
            center_idx=0,
            mano_assets_root=mano_assets_root,
            use_pca=False,
            flat_hand_mean=True,
        ).to(device)
        self.manolayer.eval()

        # ManipTrans's retargeter always applies this source-to-Gym rotation.
        # TACO is already z-up, so pre-applying its inverse keeps TACO's z axis
        # aligned with gravity both during standalone retargeting and RL loading.
        source_to_gym = aa_to_rotmat(np.array([0, 0, -np.pi / 2], dtype=np.float32)) @ aa_to_rotmat(
            np.array([np.pi / 2, 0, 0], dtype=np.float32)
        )
        correction = torch.eye(4, dtype=torch.float32, device=device)
        correction[:3, :3] = torch.as_tensor(source_to_gym.T, dtype=torch.float32, device=device)
        self._source_frame_correction = correction

    def __len__(self):
        return len(self.sequence_index)

    def _load_hand(self, record):
        pose_path = os.path.join(record.hand_pose_dir, f"{self.side}_hand.pkl")
        shape_path = os.path.join(record.hand_pose_dir, f"{self.side}_hand_shape.pkl")
        for path in (pose_path, shape_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Missing TACO hand annotation: {path}")

        with open(pose_path, "rb") as stream:
            hand_data = pickle.load(stream)
        with open(shape_path, "rb") as stream:
            shape_data = pickle.load(stream)
        frame_keys = sorted(hand_data)
        poses = torch.stack([hand_data[key]["hand_pose"].detach().float() for key in frame_keys]).to(self.device)
        translations = torch.stack([hand_data[key]["hand_trans"].detach().float() for key in frame_keys]).to(
            self.device
        )
        betas = shape_data["hand_shape"].detach().float().reshape(1, 10).to(self.device)
        betas = betas.repeat(poses.shape[0], 1)
        with torch.no_grad():
            mano_output = self.manolayer(poses, betas)
        # center_idx=0 returns a wrist-centered MANO hand; TACO hand_trans is
        # the metric world-space wrist translation.
        joints = mano_output.joints + translations[:, None]
        return pose_path, poses, translations, joints

    @staticmethod
    def _mano_joint_dict(joints: torch.Tensor):
        # manotorch SNAP order: wrist, thumb(4), index(4), middle(4),
        # ring(4), pinky(4).
        return {
            "thumb_proximal": joints[:, 1],
            "thumb_intermediate": joints[:, 2],
            "thumb_distal": joints[:, 3],
            "thumb_tip": joints[:, 4],
            "index_proximal": joints[:, 5],
            "index_intermediate": joints[:, 6],
            "index_distal": joints[:, 7],
            "index_tip": joints[:, 8],
            "middle_proximal": joints[:, 9],
            "middle_intermediate": joints[:, 10],
            "middle_distal": joints[:, 11],
            "middle_tip": joints[:, 12],
            "ring_proximal": joints[:, 13],
            "ring_intermediate": joints[:, 14],
            "ring_distal": joints[:, 15],
            "ring_tip": joints[:, 16],
            "pinky_proximal": joints[:, 17],
            "pinky_intermediate": joints[:, 18],
            "pinky_distal": joints[:, 19],
            "pinky_tip": joints[:, 20],
        }

    def _normalization_transform(self, tool_trajectory, target_trajectory, tool_mesh, target_mesh):
        normalization = taco_normalization_transform(
            tool_trajectory,
            target_trajectory,
            tool_mesh,
            target_mesh,
            workspace_center=self.workspace_center,
            floor_clearance=self.floor_clearance,
        )
        return torch.as_tensor(normalization, dtype=torch.float32, device=self.device)

    @lru_cache(maxsize=None)
    def __getitem__(self, index):
        if not isinstance(index, str) or index not in self.sequence_index:
            example = "bb_bowl_0919036"
            raise KeyError(f"Unknown TACO sequence ID {index!r}; expected an ID such as {example}")
        record = self.sequence_index[index]

        tool_trajectory_np = np.load(record.tool_pose_path).astype(np.float32)
        target_trajectory_np = np.load(record.target_pose_path).astype(np.float32)
        _validate_rigid_trajectory("tool trajectory", tool_trajectory_np)
        _validate_rigid_trajectory("target trajectory", target_trajectory_np)
        if tool_trajectory_np.shape != target_trajectory_np.shape:
            raise ValueError(
                f"TACO tool/target frame mismatch for {index}: "
                f"{tool_trajectory_np.shape} vs {target_trajectory_np.shape}"
            )

        data_path, hand_poses, hand_translations, mano_joints = self._load_hand(record)
        if len(hand_poses) != len(tool_trajectory_np):
            raise ValueError(
                f"TACO hand/object frame mismatch for {index}: {len(hand_poses)} vs {len(tool_trajectory_np)}"
            )

        role = "tool" if self.side == "right" else "target"
        selected_trajectory_np = tool_trajectory_np if role == "tool" else target_trajectory_np
        hand_object_distance = torch.linalg.norm(
            hand_translations - torch.as_tensor(selected_trajectory_np[:, :3, 3], device=self.device), dim=-1
        ).median().item()
        if hand_object_distance > self.max_hand_object_distance:
            raise ValueError(
                f"TACO sequence {index} is inconsistent: median {self.side}-hand/{role} distance is "
                f"{hand_object_distance:.3f} m. Exclude or repair this annotation before training."
            )

        tool_mesh = _load_meter_mesh(record.tool_mesh_path)
        target_mesh = _load_meter_mesh(record.target_mesh_path)
        normalization = self._normalization_transform(
            tool_trajectory_np, target_trajectory_np, tool_mesh, target_mesh
        )
        effective_transform = self._base_mujoco2gym_transf @ self._source_frame_correction @ normalization

        source_object_id = record.tool_id if role == "tool" else record.target_id
        selected_object_id = (
            self.new_obj
            if self.side == "right" and self.new_obj is not None
            else source_object_id
        )
        mano_joints = _linear_resample(mano_joints, TACO_SOURCE_FPS, MANIPTRANS_FPS)
        wrist_pos = mano_joints[:, 0]
        wrist_pos = wrist_pos - (mano_joints[:, 9] - wrist_pos) * 0.25
        wrist_pos = wrist_pos + torch.as_tensor(
            self.dexhand.relative_translation, dtype=wrist_pos.dtype, device=self.device
        )
        wrist_rot = aa_to_rotmat(hand_poses[:, :3])
        wrist_rot = _rotation_resample(wrist_rot, TACO_SOURCE_FPS, MANIPTRANS_FPS)
        wrist_rot = wrist_rot @ torch.as_tensor(
            self.dexhand.relative_rotation, dtype=wrist_rot.dtype, device=self.device
        )
        selected_trajectory = torch.as_tensor(
            resample_taco_pose_trajectory(selected_trajectory_np),
            dtype=torch.float32,
            device=self.device,
        )
        mano_joints_dict = self._mano_joint_dict(mano_joints)

        selected_assets = resolve_taco_object_assets(self.data_dir, selected_object_id)
        selected_mesh = load_taco_meter_mesh(selected_assets.semantic_mesh_path)
        mesh = Meshes(
            verts=torch.as_tensor(np.asarray(selected_mesh.vertices)[None], dtype=torch.float32),
            faces=torch.as_tensor(np.asarray(selected_mesh.faces)[None], dtype=torch.int64),
        )
        rs_verts_obj = self.random_sampling_pc(mesh)

        data = {
            "data_path": data_path,
            "sequence_id": index,
            "obj_id": f"taco_{selected_object_id}",
            "obj_mesh_path": selected_assets.semantic_mesh_path,
            "obj_verts": rs_verts_obj,
            "obj_urdf_path": selected_assets.urdf_path,
            "obj_trajectory": selected_trajectory,
            "scene_objs": [],
            "wrist_pos": wrist_pos,
            "wrist_rot": wrist_rot,
            "mano_joints": mano_joints_dict,
            "dataset_name": "TACO",
            "source_obj_id": source_object_id,
            "selected_obj_id": selected_object_id,
            "replacement_obj_id": self.new_obj if self.side == "right" else None,
        }
        if self.side == "right":
            semantic_regions = extract_taco_red_black_regions(selected_mesh)
            data.update(
                {
                    "tool_red_points": torch.from_numpy(
                        semantic_regions["tool_red_points"]
                    ).to(self.device),
                    "tool_black_points": torch.from_numpy(
                        semantic_regions["tool_black_points"]
                    ).to(self.device),
                    "tool_red_vertex_count": torch.tensor(
                        semantic_regions["tool_red_vertex_count"],
                        device=self.device,
                        dtype=torch.long,
                    ),
                    "tool_black_vertex_count": torch.tensor(
                        semantic_regions["tool_black_vertex_count"],
                        device=self.device,
                        dtype=torch.long,
                    ),
                    "tool_semantic_scale": torch.tensor(
                        1.0, device=self.device, dtype=torch.float32
                    ),
                }
            )

        original_transform = self.mujoco2gym_transf
        try:
            self.mujoco2gym_transf = effective_transform
            self.process_data(data, index, rs_verts_obj)
        finally:
            self.mujoco2gym_transf = original_transform

        if self.load_retargeted:
            hand_directory = f"mano2{str(self.dexhand)}"
            filename = f"{index}.pkl"
            if self.side == "right":
                opt_path, expected_kind = right_hand_retarget_path(
                    Path(RETARGETING_ROOT_ORIGINAL),
                    Path(RETARGETING_ROOT_CORRECTED),
                    hand_directory,
                    index,
                    self.new_obj,
                    filename,
                    correct=self.correct,
                    dataset_name="TACO",
                )
            else:
                opt_path = (
                    Path(RETARGETING_ROOT_ORIGINAL)
                    / "TACO"
                    / hand_directory
                    / filename
                )
                expected_kind = "original"
            if not opt_path.is_file():
                raise FileNotFoundError(
                    f"The TACO {self.side}-hand Stage1 dataset requires a "
                    f"{expected_kind} retarget PKL: {opt_path}"
                )
            self.load_retargeted_data(
                data,
                str(opt_path),
                expected_kind=expected_kind,
            )
        return data


@register_manipdata("taco_rh")
class TacoDatasetDexHandRH(TacoDatasetDexHand):
    side = "right"


@register_manipdata("taco_lh")
class TacoDatasetDexHandLH(TacoDatasetDexHand):
    side = "left"
