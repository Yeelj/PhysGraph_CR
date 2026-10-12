"""Policy-independent, test-only semantic contact evaluation.

The GPU-compatible detector uses force-gated collision-surface samples. These
are contact proxies, not PhysX contact pairs or measurements of contact area.
The accumulator and log formatter do not depend on Isaac Gym or a policy.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import pickle
from typing import Mapping, Optional

import torch


@dataclass(frozen=True)
class SemanticContactEvalConfig:
    enabled: bool = True
    force_threshold: float = 1.0
    contact_distance: float = 0.01
    points_per_link: int = 32
    env_chunk_size: int = 16
    min_red_contact_ratio: float = 0.90
    max_black_contact_ratio: float = 0.05
    max_black_violation_frame_ratio: float = 0.05
    min_contact_frames: int = 3
    min_planned_semantic_contact_coverage: float = 0.80
    expected_contact_distance: float = 0.03
    contact_required_mask_file: Optional[str] = None

    @classmethod
    def from_mapping(cls, values: Mapping):
        return cls(
            enabled=bool(values.get("enabled", True)),
            force_threshold=float(values.get("forceThreshold", 1.0)),
            contact_distance=float(values.get("contactDistance", 0.01)),
            points_per_link=int(values.get("pointsPerLink", 32)),
            env_chunk_size=int(values.get("envChunkSize", 16)),
            min_red_contact_ratio=float(values.get("minRedContactRatio", 0.90)),
            max_black_contact_ratio=float(values.get("maxBlackContactRatio", 0.05)),
            max_black_violation_frame_ratio=float(
                values.get("maxBlackViolationFrameRatio", 0.05)
            ),
            min_contact_frames=int(values.get("minContactFrames", 3)),
            min_planned_semantic_contact_coverage=float(
                values.get("minPlannedSemanticContactCoverage", 0.80)
            ),
            expected_contact_distance=float(values.get("expectedContactDistance", 0.03)),
            contact_required_mask_file=values.get("contactRequiredMaskFile", None),
        )

    def __post_init__(self):
        for name in ("force_threshold", "contact_distance", "expected_contact_distance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("points_per_link", "env_chunk_size", "min_contact_frames"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "min_red_contact_ratio", "max_black_contact_ratio",
            "max_black_violation_frame_ratio",
            "min_planned_semantic_contact_coverage",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")


def contact_required_mask_from_metadata(metadata, frame_count, data_id):
    """Map the ORIGINAL tool's contact intent from its hold window onto G.

    The mask is extracted before geometric correction, so it can be shared by
    policies following either the original or corrected reference trajectory.
    Older PKLs can supply a shared coarse grasp_interval. PKLs with neither
    annotation return None; malformed annotations fail.
    """
    if not isinstance(metadata, Mapping):
        return None
    if metadata.get("data_idx", str(data_id)) != str(data_id):
        raise ValueError(f"Contact reference belongs to a different operation: {data_id}")
    intent = metadata.get("source_contact_intent")
    has_frame_mask = isinstance(intent, Mapping) and "contact_frame_mask" in intent
    interval = metadata.get("grasp_interval")
    if interval is None and not has_frame_mask:
        return None
    if interval is None or len(interval) != 2:
        raise ValueError("Source contact intent requires a grasp_interval")
    start, end = int(interval[0]), int(interval[1])
    if start < 0 or end < start:
        raise ValueError("Source contact interval must have non-negative ordered endpoints")
    required = torch.zeros(frame_count, dtype=torch.bool)
    stop = min(end + 1, frame_count)
    if not has_frame_mask:
        if start < stop:
            required[start:stop] = True
        return required
    mask = torch.as_tensor(intent["contact_frame_mask"])
    if (
        start < 0 or end < start or mask.ndim != 2 or mask.shape[1] == 0
        or len(mask) != end - start + 1
        or not bool(((mask == 0) | (mask == 1)).all())
    ):
        raise ValueError("Source contact mask does not match its grasp_interval")
    if start < stop:
        required[start:stop] = mask[:stop - start].bool().any(dim=1)
    return required


def _explicit_required_mask(value, frame_count, data_id):
    if isinstance(value, Mapping):
        intervals = value.get("required_intervals")
        if intervals is None:
            raise ValueError(f"{data_id}: expected required_intervals or a boolean frame mask")
        result = torch.zeros(frame_count, dtype=torch.bool)
        for interval in intervals:
            if len(interval) != 2:
                raise ValueError(f"{data_id}: a required interval must contain start and end")
            start, end = interval
            if (
                not isinstance(start, int) or not isinstance(end, int)
                or start < 0 or end < start
            ):
                raise ValueError(f"{data_id}: invalid required interval {interval}")
            result[start:min(end + 1, frame_count)] = True
        return result
    mask = torch.as_tensor(value)
    if (
        mask.ndim != 1 or len(mask) < frame_count
        or not bool(((mask == 0) | (mask == 1)).all())
    ):
        raise ValueError(f"{data_id}: expected at least {frame_count} binary frame entries")
    return mask[:frame_count].bool().clone()


def load_contact_required_masks(data_ids, end_frame, reference_distances, config, reference_pkls=None):
    """Build a fixed full-plan mask before any rollout; never use live contacts.

    Priority: shared JSON > common original-tool intent in a correction PKL >
    a shared coarse grasp interval > checkpoint-reference tip distances.
    Return provenance for the startup log.
    The final fallback is useful for diagnosis but is not a shared benchmark
    annotation when different methods follow different reference trajectories.
    """
    ends = end_frame.detach().cpu().long().reshape(-1)
    if len(data_ids) != len(ends):
        raise ValueError("Contact annotations require one operation ID per environment")
    reference_pkls = reference_pkls or [None] * len(ends)
    if len(reference_pkls) != len(ends):
        raise ValueError("Contact references require one path per environment")
    explicit = None
    if config.contact_required_mask_file:
        with Path(config.contact_required_mask_file).open() as stream:
            explicit = json.load(stream)
        if not isinstance(explicit, Mapping):
            raise ValueError("Contact mask JSON must map operation IDs to masks or intervals")
    max_frames = int(ends.max().item()) + 1
    masks = torch.zeros(len(ends), max_frames, dtype=torch.bool)
    cache, sources = {}, []
    for index, (data_id, end, reference_path) in enumerate(zip(data_ids, ends.tolist(), reference_pkls)):
        data_id = str(data_id)
        count = end + 1
        key = (data_id, count, str(reference_path))
        if key not in cache:
            mask, source = None, None
            if explicit is not None:
                if data_id not in explicit:
                    raise ValueError(f"Contact mask JSON has no annotation for {data_id}")
                mask = _explicit_required_mask(explicit[data_id], count, data_id)
                source = f"shared_json:{Path(config.contact_required_mask_file).resolve()}"
            elif reference_path is not None and Path(reference_path).is_file():
                with Path(reference_path).open("rb") as stream:
                    reference = pickle.load(stream)
                mask = contact_required_mask_from_metadata(reference.get("contact_retarget"), count, data_id)
                if mask is not None:
                    metadata = reference["contact_retarget"]
                    intent = metadata.get("source_contact_intent")
                    kind = (
                        "source_contact_intent"
                        if isinstance(intent, Mapping) and "contact_frame_mask" in intent
                        else "source_grasp_interval"
                    )
                    source = f"{kind}:{Path(reference_path).resolve()}"
            if mask is None:
                if reference_distances is None or reference_distances.ndim != 3 or reference_distances.shape[1] < count:
                    raise ValueError(f"{data_id}: no valid contact reference; supply contactRequiredMaskFile")
                distances = reference_distances[index, :count].detach().cpu()
                if not bool(torch.isfinite(distances).all()):
                    raise ValueError(f"{data_id}: non-finite reference contact distances")
                mask = (distances <= config.expected_contact_distance).any(dim=1)
                source = "reference_tips_distance"
            cache[key] = mask, source
        mask, source = cache[key]
        masks[index, :count] = mask
        sources.append(source)
    return masks.to(device=end_frame.device), sources


def _rotate_xyzw(quaternion, points):
    """Rotate with broadcastable Isaac Gym xyzw quaternions."""
    xyz, scalar = quaternion[..., :3], quaternion[..., 3:4]
    cross = 2.0 * torch.cross(xyz.expand_as(points), points, dim=-1)
    return points + scalar * cross + torch.cross(xyz.expand_as(points), cross, dim=-1)


def classify_surface_contacts(red_distance, black_distance, force_magnitude, config):
    """Partition force-gated near-tool samples; boundary ties belong to black."""
    finite = (
        torch.isfinite(red_distance) & torch.isfinite(black_distance)
        & torch.isfinite(force_magnitude)
    )
    contact = (
        finite & (force_magnitude >= config.force_threshold)
        & (torch.minimum(red_distance, black_distance) <= config.contact_distance)
    )
    # Conservative, deterministic classification; no red-priority suppression
    # from the policy's soft semantic observation is used by the evaluator.
    black_nearest = black_distance <= red_distance + 1e-6
    return contact & ~black_nearest, contact & black_nearest


class CollisionSurfaceContactEvaluator:
    backend = "collision_surface_proxy"

    def __init__(self, link_points, body_handles, red_points, black_points, scale, config):
        self.config = config
        self.red_points = red_points
        self.black_points = black_points
        self.scale = scale.reshape(-1, 1, 1)
        for name, region in (("red", red_points), ("black", black_points)):
            if (
                region.ndim != 3 or region.shape[0] != len(self.scale)
                or region.shape[1] == 0 or region.shape[2] != 3
            ):
                raise ValueError(f"{name} points must have shape [num_envs, nonempty_region, 3]")
            if not bool(torch.isfinite(region).all()):
                raise ValueError(f"{name} semantic region has non-finite coordinates")
        if not bool((torch.isfinite(self.scale) & (self.scale > 0)).all()):
            raise ValueError("Semantic tool scale must be finite and positive")
        self.link_names = tuple(link_points)
        missing = [name for name in self.link_names if name not in body_handles]
        if missing:
            raise ValueError(f"Collision links have no live rigid-body handle: {missing}")
        if any(body_handles[name] < 0 for name in self.link_names):
            raise ValueError("Collision rigid-body handles must be non-negative")
        points, handles = [], []
        for name in self.link_names:
            samples = torch.as_tensor(
                link_points[name], dtype=red_points.dtype, device=red_points.device
            )
            if samples.ndim != 2 or samples.shape[1] != 3 or not len(samples):
                raise ValueError(f"Invalid collision samples for {name}")
            if not bool(torch.isfinite(samples).all()):
                raise ValueError(f"Non-finite collision samples for {name}")
            points.append(samples)
            handles.extend([body_handles[name]] * len(samples))
        if not points:
            raise ValueError("Semantic evaluation requires right-hand collision surfaces")
        self.local_points = torch.cat(points)
        self.body_indices = torch.tensor(handles, dtype=torch.long, device=red_points.device)

    @torch.no_grad()
    def detect(self, rigid_body_state, net_contact_force, tool_position, tool_quaternion):
        """Transform full-hand collision samples into the current tool frame.

        Chunk by environment to bound the nearest-neighbour distance matrices.
        All geometry and thresholds are independent of semanticContact.enabled.
        """
        red_parts, black_parts = [], []
        for start in range(0, len(rigid_body_state), self.config.env_chunk_size):
            stop = start + self.config.env_chunk_size
            body = rigid_body_state[start:stop, self.body_indices]
            samples = self.local_points.unsqueeze(0).expand(len(body), -1, -1)
            world = _rotate_xyzw(body[:, :, 3:7], samples) + body[:, :, :3]
            inverse = tool_quaternion[start:stop].clone()
            inverse[:, :3] *= -1
            local = _rotate_xyzw(
                inverse[:, None, :], world - tool_position[start:stop, None, :]
            ) / self.scale[start:stop]
            red_distance = torch.cdist(local, self.red_points[start:stop]).amin(dim=-1)
            black_distance = torch.cdist(local, self.black_points[start:stop]).amin(dim=-1)
            metric_scale = self.scale[start:stop, :, 0]
            force = torch.linalg.vector_norm(
                net_contact_force[start:stop, self.body_indices], dim=-1
            )
            red, black = classify_surface_contacts(
                red_distance * metric_scale, black_distance * metric_scale, force, self.config
            )
            red_parts.append(red)
            black_parts.append(black)
        return torch.cat(red_parts), torch.cat(black_parts)


class SemanticContactEpisodeAccumulator:
    """Count one complete loaded operation per environment, without altering it."""

    COUNTER_NAMES = (
        "red_contacts", "black_contacts", "black_frames", "contact_frames", "frames",
        "compliant_required_frames", "executed_required_frames", "missing_contact_frames",
        "black_required_frames",
    )

    def __init__(self, end_frame, config, contact_required=None):
        self.end_frame = end_frame.long().reshape(-1).clone()
        if bool((self.end_frame < 0).any()):
            raise ValueError("Operation end frames must be non-negative")
        self.config = config
        self.planned_frames = self.end_frame + 1
        max_frames = int(self.planned_frames.max().item())
        if contact_required is None:
            contact_required = torch.ones(
                len(self.end_frame), max_frames, dtype=torch.bool, device=self.end_frame.device
            )
        if (
            contact_required.ndim != 2 or len(contact_required) != len(self.end_frame)
            or contact_required.shape[1] < max_frames
            or not bool(((contact_required == 0) | (contact_required == 1)).all())
        ):
            raise ValueError("Required contact mask must cover the full plan of every environment")
        self.contact_required = contact_required[:, :max_frames].to(
            device=self.end_frame.device, dtype=torch.bool
        ).clone()
        self.contact_required &= torch.arange(max_frames, device=self.end_frame.device)[None] <= self.end_frame[:, None]
        self.planned_required_frames = self.contact_required.sum(dim=1).long()
        for name in self.COUNTER_NAMES:
            setattr(self, name, torch.zeros_like(self.end_frame))
        self.first_frame = torch.full_like(self.end_frame, -1)
        self.last_frame = torch.full_like(self.end_frame, -1)

    def reset(self, env_ids):
        for name in self.COUNTER_NAMES:
            getattr(self, name)[env_ids] = 0
        self.first_frame[env_ids] = -1
        self.last_frame[env_ids] = -1

    @torch.no_grad()
    def update(self, red, black, frame_index):
        if red.shape != black.shape or red.ndim != 2 or len(red) != len(self.end_frame):
            raise ValueError("Contact masks must have shape [num_envs, num_samples]")
        frame_index = frame_index.long().reshape(-1)
        if frame_index.shape != self.end_frame.shape:
            raise ValueError("Frame indices must contain one entry per environment")
        # This also prevents counting a terminal state twice before a reset.
        valid = (
            (frame_index >= 0) & (frame_index <= self.end_frame)
            & (frame_index > self.last_frame)
        )
        self.first_frame = torch.where(
            valid & (self.frames == 0), frame_index, self.first_frame
        )
        self.last_frame = torch.where(valid, frame_index, self.last_frame)
        self.red_contacts += red.sum(dim=1).long() * valid
        self.black_contacts += black.sum(dim=1).long() * valid
        self.black_frames += black.any(dim=1).long() * valid
        self.contact_frames += (red | black).any(dim=1).long() * valid
        self.frames += valid.long()
        lookup = frame_index.clamp(min=0, max=self.contact_required.shape[1] - 1)
        required = self.contact_required[torch.arange(len(lookup), device=lookup.device), lookup] & valid
        has_red, has_black = red.any(dim=1), black.any(dim=1)
        has_contact = has_red | has_black
        self.executed_required_frames += required.long()
        self.compliant_required_frames += (required & has_red & ~has_black).long()
        self.missing_contact_frames += (required & ~has_contact).long()
        self.black_required_frames += (required & has_black).long()

    @torch.no_grad()
    def snapshot(self, task_success):
        contact_count = self.red_contacts + self.black_contacts
        red_ratio = _tensor_ratio(self.red_contacts, contact_count)
        black_ratio = _tensor_ratio(self.black_contacts, contact_count)
        black_frame_ratio = _tensor_ratio(self.black_frames, self.frames)
        completion_ratio = _tensor_ratio(self.frames, self.planned_frames)
        planned_coverage = _tensor_ratio(self.compliant_required_frames, self.planned_required_frames)
        complete = (
            (self.first_frame == 0) & (self.last_frame == self.end_frame)
            & (self.frames == self.end_frame + 1)
        )
        semantic_success = (
            task_success.bool().reshape(-1) & complete
            & (self.contact_frames >= self.config.min_contact_frames)
            & (contact_count > 0)
            & (red_ratio >= self.config.min_red_contact_ratio)
            & (black_ratio <= self.config.max_black_contact_ratio)
            & (black_frame_ratio <= self.config.max_black_violation_frame_ratio)
            & (self.planned_required_frames > 0)
            & (planned_coverage >= self.config.min_planned_semantic_contact_coverage)
        )
        # Copies survive environment resets and keep terminal records immutable.
        result = {name: getattr(self, name).clone() for name in self.COUNTER_NAMES}
        result.update(
            operation_completion_ratio=completion_ratio,
            planned_semantic_contact_coverage=planned_coverage,
            planned_frames=self.planned_frames.clone(),
            planned_required_frames=self.planned_required_frames.clone(),
            unexecuted_required_frames=(self.planned_required_frames - self.executed_required_frames).clone(),
            red_contact_ratio=red_ratio,
            black_contact_ratio=black_ratio,
            black_violation_frame_ratio=black_frame_ratio,
            semantic_success=semantic_success,
            complete=complete,
            first_frame=self.first_frame.clone(),
            last_frame=self.last_frame.clone(),
        )
        return result


def _tensor_ratio(numerator, denominator):
    value = numerator.double() / denominator.clamp_min(1).double()
    return torch.where(denominator > 0, value, torch.full_like(value, float("nan")))


def summarize_semantic_episodes(snapshot, env_ids):
    """Pool counts for exactly the episodes ending in the current player step."""
    device = snapshot["frames"].device
    env_ids = env_ids.reshape(-1).to(device=device, dtype=torch.long)
    if not len(env_ids):
        raise ValueError("Cannot summarize an empty set of completed episodes")
    counts = {
        name: int(snapshot[name][env_ids].sum().item())
        for name in SemanticContactEpisodeAccumulator.COUNTER_NAMES + (
            "planned_frames", "planned_required_frames", "unexecuted_required_frames",
        )
    }
    contacts = counts["red_contacts"] + counts["black_contacts"]
    successes = int(snapshot["semantic_success"][env_ids].sum().item())
    counts.update(
        operation_completion_ratio=counts["frames"] / counts["planned_frames"],
        planned_semantic_contact_coverage=(
            counts["compliant_required_frames"] / counts["planned_required_frames"]
            if counts["planned_required_frames"] else float("nan")
        ),
        red_contact_ratio=counts["red_contacts"] / contacts if contacts else float("nan"),
        black_contact_ratio=counts["black_contacts"] / contacts if contacts else float("nan"),
        black_violation_frame_ratio=(
            counts["black_frames"] / counts["frames"] if counts["frames"] else float("nan")
        ),
        semantic_task_success_rate=successes / len(env_ids),
        semantic_successes=successes,
        episodes=len(env_ids),
        complete_episodes=int(snapshot["complete"][env_ids].sum().item()),
        no_contact_episodes=int((
            snapshot["red_contacts"][env_ids] + snapshot["black_contacts"][env_ids] == 0
        ).sum().item()),
    )
    return counts


def format_semantic_episode_summary(summary):
    def number(value):
        return f"{value:.4f}" if math.isfinite(value) else "NA"

    return "".join(
        f" {name}: {number(summary[name])}"
        for name in (
            "operation_completion_ratio", "planned_semantic_contact_coverage",
            "black_violation_frame_ratio", "semantic_task_success_rate",
        )
    )
