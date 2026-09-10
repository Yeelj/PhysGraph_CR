from abc import ABC, abstractmethod
import hashlib
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from scipy.ndimage import gaussian_filter1d
import numpy as np
import torch
from main.dataset.transform import aa_to_rotmat, caculate_align_mat, rotmat_to_aa
from torch.utils.data import Dataset
from pytorch3d.ops import sample_points_from_meshes
from termcolor import cprint
import pickle


def _contact_retarget_validation_tiers_passed(metadata):
    """Check only the correction tiers that were explicitly configured."""

    if int(metadata.get("schema_version", 0)) < 3:
        return True
    validation_tiers = metadata.get("validation_tiers")
    if not isinstance(validation_tiers, dict):
        return False
    required_tiers = ["semantic_geometry"]
    # Schema-v3/v4 PKLs written before this switch did not carry the field and
    # required PhysX, so absence deliberately remains strict.
    if bool(metadata.get("physics_validation_required", True)):
        required_tiers.append("isaac_gym_physics")
        if int(metadata.get("schema_version", 0)) >= 8:
            required_tiers.extend(("physical_optimization", "mano_target_sync"))
    return all(
        isinstance(validation_tiers.get(name), dict)
        and bool(validation_tiers[name].get("passed", False))
        for name in required_tiers
    )


def _validate_retargeting_source(metadata, retargeted_data_path, expected_kind):
    """Fail closed when Stage-1 is given the wrong hand-trajectory PKL."""

    if expected_kind is None:
        return "contact_corrected" if metadata is not None else "original"
    if expected_kind not in {"original", "contact_corrected"}:
        raise ValueError(f"Unsupported retargeting kind: {expected_kind!r}")
    actual_kind = "contact_corrected" if metadata is not None else "original"
    if actual_kind != expected_kind:
        raise ValueError(
            f"Expected {expected_kind} hand trajectory, but {retargeted_data_path} "
            f"contains {actual_kind} data"
        )
    return actual_kind


class ManipData(Dataset, ABC):
    def __init__(
        self,
        *,
        data_dir: str,
        split: str = "all",
        skip: int = 2,
        device="cuda:0",
        mujoco2gym_transf=None,
        max_seq_len=int(1e10),
        dexhand=None,
        verbose=True,
        **kwargs,
    ):
        self.data_dir = data_dir
        self.split = split
        self.skip = skip
        self.data_pathes = None

        self.dexhand = dexhand
        self.device = device

        self.verbose = verbose

        # ? modify this depending on the origin point
        self.transf_offset = torch.eye(4, dtype=torch.float32, device=mujoco2gym_transf.device)

        self.mujoco2gym_transf = mujoco2gym_transf
        self.max_seq_len = max_seq_len

        # caculate contact

        import chamfer_distance as chd

        self.ch_dist = chd.ChamferDistance()

    def __len__(self):
        return len(self.data_pathes)

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @staticmethod
    def compute_velocity(p, time_delta, guassian_filter=True):
        # [T, K, 3]
        velocity = np.gradient(p.cpu().numpy(), axis=0) / time_delta
        if guassian_filter:
            velocity = gaussian_filter1d(velocity, 2, axis=0, mode="nearest")
        return torch.from_numpy(velocity).to(p)

    @staticmethod
    def compute_angular_velocity(r, time_delta: float, guassian_filter=True):
        # [T, K, 3, 3]
        diff_r = r[1:] @ r[:-1].transpose(-1, -2)  # [T-1, K, 3, 3]
        diff_aa = rotmat_to_aa(diff_r).cpu().numpy()  # [T-1, K, 3]
        diff_angle = np.linalg.norm(diff_aa, axis=-1)  # [T-1, K]
        diff_axis = diff_aa / (diff_angle[:, :, None] + 1e-8)  # [T-1, K, 3]
        angular_velocity = diff_axis * diff_angle[:, :, None] / time_delta  # [T-1, K, 3]
        angular_velocity = np.concatenate([angular_velocity, angular_velocity[-1:]], axis=0)  # [T, K, 3]
        if guassian_filter:
            angular_velocity = gaussian_filter1d(angular_velocity, 2, axis=0, mode="nearest")
        return torch.from_numpy(angular_velocity).to(r)

    @staticmethod
    def compute_dof_velocity(dof, time_delta, guassian_filter=True):
        # [T, K]
        velocity = np.gradient(dof.cpu().numpy(), axis=0) / time_delta
        if guassian_filter:
            velocity = gaussian_filter1d(velocity, 2, axis=0, mode="nearest")
        return torch.from_numpy(velocity).to(dof)

    def random_sampling_pc(self, mesh):
        numpy_random_state = np.random.get_state()
        torch_random_state = torch.random.get_rng_state()
        cuda_available = torch.cuda.is_available()
        torch_random_state_cuda = torch.cuda.get_rng_state() if cuda_available else None
        np.random.seed(0)
        torch.manual_seed(0)
        if cuda_available:
            torch.cuda.manual_seed(0)
            torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        rs_verts_obj = sample_points_from_meshes(mesh, 1000, return_normals=False).to(self.device).squeeze(0)

        # reset seed
        np.random.set_state(numpy_random_state)
        torch.random.set_rng_state(torch_random_state)
        if cuda_available:
            torch.cuda.set_rng_state(torch_random_state_cuda)

        return rs_verts_obj

    @staticmethod
    def _sha256_file(path):
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @classmethod
    def _urdf_bundle_sha256(cls, urdf_path):
        urdf_path = Path(urdf_path).resolve()
        if not urdf_path.is_file():
            raise FileNotFoundError(f"Object collision URDF does not exist: {urdf_path}")
        asset_paths = [urdf_path]
        urdf_root = ET.fromstring(urdf_path.read_text(encoding="utf-8").lstrip())
        for mesh in urdf_root.findall(".//mesh"):
            filename = mesh.attrib.get("filename")
            if not filename or filename.startswith("package://"):
                raise ValueError(f"Unsupported object URDF mesh URI {filename!r} in {urdf_path}")
            mesh_path = Path(filename[7:] if filename.startswith("file://") else filename)
            if not mesh_path.is_absolute():
                mesh_path = urdf_path.parent / mesh_path
            mesh_path = mesh_path.resolve()
            if not mesh_path.is_file():
                raise FileNotFoundError(f"Object collision mesh does not exist: {mesh_path}")
            asset_paths.append(mesh_path)
        digest = hashlib.sha256()
        for asset_path in sorted(set(asset_paths), key=str):
            digest.update(asset_path.name.encode("utf-8"))
            digest.update(bytes.fromhex(cls._sha256_file(asset_path)))
        return digest.hexdigest()

    def _update_hand_object_distances(self, data):
        """Recompute trajectory-dependent distances after a rigid hand shift."""

        obj_verts_transf = (
            data["obj_trajectory"][:, :3, :3] @ data["obj_verts"].T[None]
        ).transpose(-1, -2) + data["obj_trajectory"][:, :3, 3][:, None]

        tip_list = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
        tips = torch.cat([data["mano_joints"][key][:, None] for key in tip_list], dim=1)
        tips_near, _, _, _ = self.ch_dist(tips, obj_verts_transf)
        data["tips_distance"] = torch.sqrt(tips_near)

        joints = torch.cat(
            [data["mano_joints"][key][:, None] for key in data["mano_joints"].keys()],
            dim=1,
        )
        joints_near, _, _, _ = self.ch_dist(joints, obj_verts_transf)
        data["joints_distance"] = torch.sqrt(joints_near)

    def _apply_contact_retarget_targets(self, data, metadata, retargeted_data_path):
        """Apply the PKL's tool-local rigid correction to every Stage-1 target.

        The corrected PKL already contains transformed dex-hand targets. Stage 1,
        however, also tracks MANO wrist/joints. Schema-v6 trajectories therefore
        store the exact per-frame tool-local SE(3) schedule and apply it here.
        Older trajectories retain their translation-only compatibility path.
        """

        if not isinstance(metadata, dict):
            raise TypeError(f"Invalid contact_retarget metadata in {retargeted_data_path}")
        if not bool(metadata.get("geometry_feasible", False)):
            raise ValueError(
                f"Refusing infeasible contact-retarget PKL: {retargeted_data_path}"
            )
        if int(metadata.get("schema_version", 0)) >= 3:
            if not _contact_retarget_validation_tiers_passed(metadata):
                raise ValueError(
                    "Stage-1 requires a schema-v3-or-newer contact trajectory that passed all "
                    "validation tiers configured during correction (semantic geometry and "
                    f"optional PhysX): {retargeted_data_path}"
                )
        if int(metadata.get("schema_version", 0)) >= 4:
            semantic_precheck = metadata.get("semantic_precheck")
            if not isinstance(semantic_precheck, dict) or not bool(
                semantic_precheck.get("passed", False)
            ):
                raise ValueError(
                    f"Schema-v4 contact trajectory lacks a passing mesh semantic precheck: {retargeted_data_path}"
                )

        expected_side = getattr(self.dexhand, "side", None)
        metadata_side = metadata.get("side")
        side_alias = {"right": "rh", "left": "lh", "rh": "rh", "lh": "lh"}
        if expected_side is not None and side_alias.get(metadata_side) != side_alias.get(expected_side):
            raise ValueError(
                f"Contact-retarget side {metadata_side!r} does not match dataset hand "
                f"{expected_side!r}: {retargeted_data_path}"
            )

        expected_mesh_hash = metadata.get("tool_mesh_sha256")
        current_mesh_path = data.get("obj_mesh_path")
        if not expected_mesh_hash or not current_mesh_path or not os.path.isfile(current_mesh_path):
            raise ValueError(
                "Contact-retarget loading requires a readable dataset obj_mesh_path and "
                f"tool_mesh_sha256 metadata: {retargeted_data_path}"
            )
        current_mesh_hash = self._sha256_file(current_mesh_path)
        if current_mesh_hash != expected_mesh_hash:
            raise ValueError(
                "The Stage-1 dataset mesh differs from the mesh used for geometric correction. "
                f"dataset={current_mesh_path}, corrected={metadata.get('tool_mesh')}. "
                "Install the same new mesh in the dataset before training."
            )

        expected_collision_hash = metadata.get("collision_bundle_sha256")
        current_collision_path = data.get("obj_urdf_path")
        if not expected_collision_hash or not current_collision_path:
            raise ValueError(
                f"Missing collision bundle metadata in {retargeted_data_path}"
            )
        current_collision_hash = self._urdf_bundle_sha256(current_collision_path)
        if current_collision_hash != expected_collision_hash:
            raise ValueError(
                "The Stage-1 collision URDF/mesh bundle differs from the bundle checked "
                f"during geometric correction: {current_collision_path}."
            )

        target_length = len(data["obj_trajectory"])
        alpha = torch.as_tensor(
            metadata.get("alpha"), dtype=data["wrist_pos"].dtype, device=self.device
        )
        transform_schedule_value = metadata.get("transform_schedule_tool")
        if transform_schedule_value is not None:
            transform_schedule = torch.as_tensor(
                transform_schedule_value,
                dtype=data["wrist_pos"].dtype,
                device=self.device,
            )
            if transform_schedule.ndim != 3 or transform_schedule.shape[1:] != (4, 4):
                raise ValueError(
                    "Invalid contact SE(3) schedule shape: "
                    f"{tuple(transform_schedule.shape)} in {retargeted_data_path}"
                )
            if len(transform_schedule) < target_length or alpha.ndim != 1 or len(alpha) < target_length:
                raise ValueError(
                    "Contact SE(3) schedule is shorter than the Stage-1 target: "
                    f"transforms={len(transform_schedule)}, alpha={len(alpha)}, target={target_length}"
                )
            transform_schedule = transform_schedule[:target_length]
            object_rotation = data["obj_trajectory"][:, :3, :3]
            object_translation = data["obj_trajectory"][:, :3, 3]
            correction_rotation = transform_schedule[:, :3, :3]
            correction_translation = transform_schedule[:, :3, 3]

            def transform_world_points(points):
                points_tool = torch.einsum(
                    "fji,fj->fi", object_rotation, points - object_translation
                )
                corrected_tool = (
                    torch.einsum("fij,fj->fi", correction_rotation, points_tool)
                    + correction_translation
                )
                return (
                    torch.einsum("fij,fj->fi", object_rotation, corrected_tool)
                    + object_translation
                )

            data["wrist_pos"] = transform_world_points(data["wrist_pos"])
            for key in data["mano_joints"].keys():
                data["mano_joints"][key] = transform_world_points(data["mano_joints"][key])

            wrist_rotation = aa_to_rotmat(data["wrist_rot"])
            correction_world_rotation = (
                object_rotation
                @ correction_rotation
                @ object_rotation.transpose(-1, -2)
            )
            data["wrist_rot"] = rotmat_to_aa(correction_world_rotation @ wrist_rotation)
        else:
            delta_tool = torch.as_tensor(
                metadata.get("delta_tool"), dtype=data["wrist_pos"].dtype, device=self.device
            )
            if delta_tool.shape != (3,) or alpha.ndim != 1 or len(alpha) < target_length:
                raise ValueError(
                    f"Invalid correction offset shapes: delta_tool={tuple(delta_tool.shape)}, "
                    f"alpha={tuple(alpha.shape)}, target_length={target_length}"
                )
            alpha = alpha[:target_length]
            delta_world = torch.matmul(
                data["obj_trajectory"][:, :3, :3], delta_tool[:, None]
            ).squeeze(-1) * alpha[:, None]

            data["wrist_pos"] = data["wrist_pos"] + delta_world
            for key in data["mano_joints"].keys():
                data["mano_joints"][key] = data["mano_joints"][key] + delta_world

        if alpha.ndim != 1 or len(alpha) < target_length:
            raise ValueError(
                f"Invalid correction alpha shape: {tuple(alpha.shape)}, target_length={target_length}"
            )

        corrected_body_targets = data.get("corrected_mano_body_targets")
        if corrected_body_targets is not None:
            body_names = data.get("corrected_mano_body_names")
            expected_body_names = list(self.dexhand.body_names)
            if list(body_names or ()) != expected_body_names:
                raise ValueError(
                    "Corrected MANO body ordering does not match the Stage-1 dex-hand: "
                    f"stored={body_names}, expected={expected_body_names}"
                )
            expected_shape = (len(expected_body_names), 3)
            if corrected_body_targets.ndim != 3 or tuple(
                corrected_body_targets.shape[1:]
            ) != expected_shape:
                raise ValueError(
                    "Invalid corrected MANO body target shape: "
                    f"{tuple(corrected_body_targets.shape)}"
                )
            if len(corrected_body_targets) < target_length:
                raise ValueError(
                    "Corrected MANO body targets are shorter than the Stage-1 target: "
                    f"{len(corrected_body_targets)} < {target_length}"
                )
            corrected_body_targets = corrected_body_targets[:target_length]
            data["corrected_mano_body_targets"] = corrected_body_targets
            grouped_targets = {}
            for body_index, body_name in enumerate(expected_body_names):
                mano_name = self.dexhand.to_hand(body_name)[0]
                if mano_name == "wrist":
                    continue
                grouped_targets.setdefault(mano_name, []).append(
                    corrected_body_targets[:, body_index]
                )
            for mano_name, values in grouped_targets.items():
                data["mano_joints"][mano_name] = torch.stack(values, dim=1).mean(dim=1)
            data["wrist_pos"] = data["opt_wrist_pos"][:target_length]
            data["wrist_rot"] = data["opt_wrist_rot"][:target_length]

        time_delta = 1 / (120 / self.skip)
        data["wrist_velocity"] = self.compute_velocity(
            data["wrist_pos"][:, None], time_delta, guassian_filter=True
        ).squeeze(1)
        for key in data["mano_joints"].keys():
            data["mano_joints_velocity"][key] = self.compute_velocity(
                data["mano_joints"][key], time_delta, guassian_filter=True
            )
        if corrected_body_targets is not None:
            data["corrected_mano_body_targets_velocity"] = self.compute_velocity(
                corrected_body_targets, time_delta, guassian_filter=True
            )
        if "wrist_rot" in data:
            data["wrist_angular_velocity"] = self.compute_angular_velocity(
                aa_to_rotmat(data["wrist_rot"][:, None]),
                time_delta,
                guassian_filter=True,
            ).squeeze(1)
        self._update_hand_object_distances(data)
        data["contact_retarget_applied"] = retargeted_data_path

    def process_data(self, data, idx, rs_verts_obj):
        data["obj_trajectory"] = self.mujoco2gym_transf @ data["obj_trajectory"]
        data["wrist_pos"] = (self.mujoco2gym_transf[:3, :3] @ data["wrist_pos"].T).T + self.mujoco2gym_transf[:3, 3]
        data["wrist_rot"] = rotmat_to_aa(self.mujoco2gym_transf[:3, :3] @ data["wrist_rot"])
        for k in data["mano_joints"].keys():
            data["mano_joints"][k] = (
                self.mujoco2gym_transf[:3, :3] @ data["mano_joints"][k].T
            ).T + self.mujoco2gym_transf[:3, 3]

        # calculate distance
        self._update_hand_object_distances(data)

        data["obj_velocity"] = self.compute_velocity(
            data["obj_trajectory"][:, None, :3, 3], 1 / (120 / self.skip), guassian_filter=True
        ).squeeze(1)
        data["obj_angular_velocity"] = self.compute_angular_velocity(
            data["obj_trajectory"][:, None, :3, :3], 1 / (120 / self.skip), guassian_filter=True
        ).squeeze(1)
        data["wrist_velocity"] = self.compute_velocity(
            data["wrist_pos"][:, None], 1 / (120 / self.skip), guassian_filter=True
        ).squeeze(1)
        data["wrist_angular_velocity"] = self.compute_angular_velocity(
            aa_to_rotmat(data["wrist_rot"][:, None]), 1 / (120 / self.skip), guassian_filter=True
        ).squeeze(1)
        data["mano_joints_velocity"] = {}
        for k in data["mano_joints"].keys():
            data["mano_joints_velocity"][k] = self.compute_velocity(
                data["mano_joints"][k], 1 / (120 / self.skip), guassian_filter=True
            )

        if len(data["obj_trajectory"]) > self.max_seq_len:
            cprint(
                f"WARN: {self.data_pathes[idx]} is too long : {len(data['obj_trajectory'])}, cut to {self.max_seq_len}",
                "yellow",
            )
            data["obj_trajectory"] = data["obj_trajectory"][: self.max_seq_len]
            data["obj_velocity"] = data["obj_velocity"][: self.max_seq_len]
            data["obj_angular_velocity"] = data["obj_angular_velocity"][: self.max_seq_len]
            data["wrist_pos"] = data["wrist_pos"][: self.max_seq_len]
            data["wrist_rot"] = data["wrist_rot"][: self.max_seq_len]
            for k in data["mano_joints"].keys():
                data["mano_joints"][k] = data["mano_joints"][k][: self.max_seq_len]
            data["wrist_velocity"] = data["wrist_velocity"][: self.max_seq_len]
            data["wrist_angular_velocity"] = data["wrist_angular_velocity"][: self.max_seq_len]
            for k in data["mano_joints_velocity"].keys():
                data["mano_joints_velocity"][k] = data["mano_joints_velocity"][k][: self.max_seq_len]
            data["tips_distance"] = data["tips_distance"][: self.max_seq_len]
            data["joints_distance"] = data["joints_distance"][: self.max_seq_len]

    def load_retargeted_data(self, data, retargeted_data_path, *, expected_kind=None):
        if not os.path.exists(retargeted_data_path):
            if expected_kind is not None:
                raise FileNotFoundError(
                    f"Required {expected_kind} hand trajectory does not exist: {retargeted_data_path}"
                )
            if self.verbose:
                cprint(f"\nWARNING: {retargeted_data_path} does not exist.", "red")
                cprint(f"WARNING: This may lead to a slower transfer process or even failure to converge.", "red")
                cprint(
                    f"WARNING: It is recommended to first execute the retargeting code to obtain initial values.\n",
                    "red",
                )
            data.update(
                {
                    "opt_wrist_pos": data["wrist_pos"],
                    "opt_wrist_rot": data["wrist_rot"],
                    "opt_dof_pos": torch.zeros([data["wrist_pos"].shape[0], self.dexhand.n_dofs], device=self.device),
                }
            )
        else:
            with open(retargeted_data_path, "rb") as stream:
                opt_params = pickle.load(stream)
            contact_metadata = opt_params.get("contact_retarget")
            retargeting_kind = _validate_retargeting_source(
                contact_metadata, retargeted_data_path, expected_kind
            )
            data.update(
                {
                    "opt_wrist_pos": torch.tensor(opt_params["opt_wrist_pos"], device=self.device),
                    "opt_wrist_rot": torch.tensor(opt_params["opt_wrist_rot"], device=self.device),
                    "opt_dof_pos": torch.tensor(opt_params["opt_dof_pos"], device=self.device),
                    # "opt_joints_pos": torch.tensor(opt_params["opt_joints_pos"], device=self.device), # ? only used for ablation study
                    "retargeted_data_path": os.path.abspath(retargeted_data_path),
                    "retargeting_kind": retargeting_kind,
                }
            )
            if "corrected_mano_body_targets" in opt_params:
                data["corrected_mano_body_targets"] = torch.tensor(
                    opt_params["corrected_mano_body_targets"], device=self.device
                )
                data["corrected_mano_body_names"] = list(
                    opt_params.get("corrected_mano_body_names", ())
                )
            if contact_metadata is not None:
                self._apply_contact_retarget_targets(
                    data, contact_metadata, retargeted_data_path
                )
        data["opt_wrist_velocity"] = self.compute_velocity(
            data["opt_wrist_pos"][:, None], 1 / (120 / self.skip), guassian_filter=True
        ).squeeze(1)
        data["opt_wrist_angular_velocity"] = self.compute_angular_velocity(
            aa_to_rotmat(data["opt_wrist_rot"][:, None]), 1 / (120 / self.skip), guassian_filter=True
        ).squeeze(1)
        data["opt_dof_velocity"] = self.compute_dof_velocity(
            data["opt_dof_pos"], 1 / (120 / self.skip), guassian_filter=True
        )
        # data["opt_joints_velocity"] = self.compute_velocity(
        #     data["opt_joints_pos"], 1 / (120 / self.skip), guassian_filter=True
        # ) # ? only used for ablation study
        if len(data["opt_wrist_pos"]) > self.max_seq_len:
            data["opt_wrist_pos"] = data["opt_wrist_pos"][: self.max_seq_len]
            data["opt_wrist_rot"] = data["opt_wrist_rot"][: self.max_seq_len]
            data["opt_wrist_velocity"] = data["opt_wrist_velocity"][: self.max_seq_len]
            data["opt_wrist_angular_velocity"] = data["opt_wrist_angular_velocity"][: self.max_seq_len]
            data["opt_dof_pos"] = data["opt_dof_pos"][: self.max_seq_len]
            data["opt_dof_velocity"] = data["opt_dof_velocity"][: self.max_seq_len]
            # ? only used for ablation study
            # data["opt_joints_pos"] = data["opt_joints_pos"][: self.max_seq_len]
            # data["opt_joints_velocity"] = data["opt_joints_velocity"][: self.max_seq_len]
        target_length = len(data["obj_trajectory"])
        for key in (
            "opt_wrist_pos",
            "opt_wrist_rot",
            "opt_wrist_velocity",
            "opt_wrist_angular_velocity",
            "opt_dof_pos",
            "opt_dof_velocity",
        ):
            if len(data[key]) != target_length:
                raise ValueError(
                    f"Retarget trajectory length mismatch for {key}: "
                    f"{len(data[key])} != object trajectory {target_length} "
                    f"({retargeted_data_path})"
                )
