"""Dependency-free hand and observation contracts shared by training and tools.

Names retain the ordering of the existing DexHand classes.  These orders also
match Isaac Gym's imported asset ordering (including Inspire's mimic joints),
which is not necessarily the order of the joints in the URDF XML.
"""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple
import xml.etree.ElementTree as ET


SUPPORTED_HANDS = ("artimano", "allegro", "inspire", "shadow")
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OBSERVATION_LAYOUT_VERSION = 1


@dataclass(frozen=True)
class HandSpec:
    name: str
    side: str
    urdf_path: Path
    dof_names: Tuple[str, ...]
    body_names: Tuple[str, ...]
    contact_body_names: Tuple[str, ...]
    graph_edges: Tuple[Tuple[int, int], ...]
    finger_groups: Tuple[Tuple[int, ...], ...]
    synergy_groups: Tuple[Tuple[int, ...], ...]
    root_body_id: int = 0

    @property
    def n_dofs(self) -> int:
        return len(self.dof_names)

    @property
    def n_bodies(self) -> int:
        return len(self.body_names)

    @property
    def contact_body_ids(self) -> Tuple[int, ...]:
        """Five semantic slots; Allegro intentionally repeats ring/pinky."""
        return tuple(self.body_names.index(name) for name in self.contact_body_names)

    @property
    def unique_contact_body_ids(self) -> Tuple[int, ...]:
        return tuple(dict.fromkeys(self.contact_body_ids))

    @property
    def legacy_isaac_dof_names(self) -> Tuple[str, ...]:
        """Original mano2dexhand exporter order, verified on the eight assets."""
        return self.dof_names

    def signature(self) -> Dict[str, Any]:
        """Portable, JSON-safe metadata suitable for checkpoint validation."""
        return {
            "name": self.name,
            "side": self.side,
            "dof_names": list(self.dof_names),
            "body_names": list(self.body_names),
            "contact_body_names": list(self.contact_body_names),
            "graph_edges": [list(edge) for edge in self.graph_edges],
            "finger_groups": [list(group) for group in self.finger_groups],
            "synergy_groups": [list(group) for group in self.synergy_groups],
            "root_body_id": self.root_body_id,
        }

    @property
    def signature_hash(self) -> str:
        payload = json.dumps(self.signature(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _reduced_graph(path: Path, bodies: Tuple[str, ...], dofs: Tuple[str, ...]):
    """Connect each selected body to its nearest selected URDF ancestor."""
    tree = ET.parse(path).getroot()
    links = {node.attrib["name"] for node in tree.findall("link")}
    joints = {node.attrib["name"]: node for node in tree.findall("joint")}
    missing_bodies = set(bodies) - links
    missing_dofs = set(dofs) - joints.keys()
    if missing_bodies or missing_dofs:
        raise ValueError(
            f"Hand metadata does not match {path}: "
            f"missing bodies={sorted(missing_bodies)}, joints={sorted(missing_dofs)}"
        )
    for name in dofs:
        if joints[name].attrib.get("type", "fixed") in ("fixed", "floating"):
            raise ValueError(f"Hand DOF {name!r} is not movable in {path}")
    parent_by_child = {
        node.find("child").attrib["link"]: node.find("parent").attrib["link"]
        for node in joints.values()
    }
    body_ids = {name: idx for idx, name in enumerate(bodies)}
    edges = []
    roots = []
    for child in bodies:
        parent = parent_by_child.get(child)
        visited = {child}
        while parent is not None and parent not in body_ids:
            if parent in visited:
                raise ValueError(f"Cyclic URDF ancestry at {child!r} in {path}")
            visited.add(parent)
            parent = parent_by_child.get(parent)
        if parent is None:
            roots.append(body_ids[child])
        else:
            edges.append((body_ids[parent], body_ids[child]))
    if roots != [0] or len(edges) != len(bodies) - 1:
        raise ValueError(f"Hand graph must be a tree rooted at body 0: {path}")
    return tuple(edges)


def get_hand_spec(name: str, side: str = "right") -> HandSpec:
    """Resolve one of the four supported hands without importing Isaac/PyTorch.

    ``rh``/``lh`` aliases and names such as ``shadow_lh`` are accepted to match
    existing configuration and dataset conventions.
    """
    name = str(name).lower()
    side = str(side).lower()
    for suffix, suffix_side in (("_rh", "right"), ("_lh", "left")):
        if name.endswith(suffix):
            name, side = name[:-len(suffix)], suffix_side
            break
    side = {"rh": "right", "lh": "left"}.get(side, side)
    if name not in SUPPORTED_HANDS:
        raise ValueError(f"Unsupported dex hand {name!r}; expected one of {SUPPORTED_HANDS}")
    if side not in ("right", "left"):
        raise ValueError(f"Unsupported hand side {side!r}; expected right/left or rh/lh")
    return _get_hand_spec(name, side)


@lru_cache(maxsize=8)
def _get_hand_spec(name: str, side: str) -> HandSpec:
    if name == "artimano":
        body_names = ("palm",) + tuple(
            body
            for finger in ("index", "middle", "pinky", "ring")
            for body in (finger + "1y", finger + "1z", finger + "2", finger + "3", finger + "_tip")
        ) + ("thumb1x", "thumb1y", "thumb1z", "thumb2y", "thumb2z", "thumb3", "thumb_tip")
        dof_names = tuple("j_" + body for body in body_names[1:] if not body.endswith("_tip"))
        finger_names = tuple(tuple(body for body in body_names if body.startswith(finger)) for finger in FINGER_NAMES)
        contacts = tuple(finger + "3" for finger in FINGER_NAMES)
        synergy = (
            ("index_tip", "middle_tip", "pinky_tip", "ring_tip", "thumb_tip"),
            ("index1y", "middle1y", "pinky1y", "ring1y", "thumb1x"),
            ("index2", "middle2", "pinky2", "ring2", "thumb2y"),
        )
        relpath = f"mano_urdf/{'rh' if side == 'right' else 'lh'}_mano.urdf"
    elif name == "allegro":
        starts = (0, 12, 4, 8)
        body_names = ("base_link",) + tuple(
            body
            for start in starts
            for body in tuple(f"link_{idx}.0" for idx in range(start, start + 4)) + (f"link_{start + 3}.0_tip",)
        )
        dof_names = tuple(f"joint_{idx}.0" for start in starts for idx in range(start, start + 4))
        index_start, ring_start = (0, 8) if side == "right" else (8, 0)
        semantic_starts = (12, index_start, 4, ring_start, ring_start)
        finger_names = tuple(
            tuple(f"link_{idx}.0" for idx in range(start, start + 4)) + (f"link_{start + 3}.0_tip",)
            for start in semantic_starts
        )
        contacts = tuple(f"link_{start + 3}.0" for start in semantic_starts)
        synergy = (
            tuple(f"link_{start + 3}.0_tip" for start in starts),
            tuple(f"link_{start}.0" for start in starts),
            tuple(f"link_{start + 2}.0" for start in starts),
        )
        relpath = f"allegro_hand/allegro_hand_{side}.urdf"
    elif name == "inspire":
        prefix = "R_" if side == "right" else "L_"
        body_names = ("hand_base_link",) + tuple(
            finger + "_" + part
            for finger in ("index", "middle", "pinky", "ring")
            for part in ("proximal", "intermediate", "tip")
        ) + ("thumb_proximal_base", "thumb_proximal", "thumb_intermediate", "thumb_distal", "thumb_tip")
        dof_names = tuple(
            finger + "_" + part + "_joint"
            for finger in ("index", "middle", "pinky", "ring")
            for part in ("proximal", "intermediate")
        ) + ("thumb_proximal_yaw_joint", "thumb_proximal_pitch_joint", "thumb_intermediate_joint", "thumb_distal_joint")
        finger_names = tuple(tuple(prefix + body for body in body_names if body.startswith(finger)) for finger in FINGER_NAMES)
        contacts = tuple(prefix + ("thumb_distal" if finger == "thumb" else finger + "_intermediate") for finger in FINGER_NAMES)
        synergy = (
            tuple(prefix + finger + "_tip" for finger in FINGER_NAMES),
            tuple(prefix + ("thumb_proximal_base" if finger == "thumb" else finger + "_proximal") for finger in FINGER_NAMES),
            tuple(prefix + finger + "_intermediate" for finger in FINGER_NAMES),
        )
        body_names = tuple(prefix + body for body in body_names)
        dof_names = tuple(prefix + dof for dof in dof_names)
        relpath = f"inspire_hand/inspire_hand_{side}.urdf"
    else:  # Shadow
        body_names = (
            "palm", "ffknuckle", "ffproximal", "ffmiddle", "ffdistal", "fftip",
            "lfmetacarpal", "lfknuckle", "lfproximal", "lfmiddle", "lfdistal", "lftip",
            "mfknuckle", "mfproximal", "mfmiddle", "mfdistal", "mftip",
            "rfknuckle", "rfproximal", "rfmiddle", "rfdistal", "rftip",
            "thbase", "thproximal", "thhub", "thmiddle", "thdistal", "thtip",
        )
        dof_names = tuple(finger + "J" + str(joint) for finger, count in (("FF", 4), ("LF", 5), ("MF", 4), ("RF", 4), ("TH", 5)) for joint in range(count, 0, -1))
        finger_prefixes = ("th", "ff", "mf", "rf", "lf")
        finger_names = tuple(tuple(body for body in body_names if body.startswith(prefix)) for prefix in finger_prefixes)
        contacts = tuple(prefix + "distal" for prefix in finger_prefixes)
        synergy = (
            tuple(prefix + "tip" for prefix in finger_prefixes),
            ("thbase", "ffknuckle", "mfknuckle", "rfknuckle", "lfknuckle"),
            ("thhub", "ffmiddle", "mfmiddle", "rfmiddle", "lfmiddle"),
        )
        relpath = f"shadow_hand/shadow_hand_{side}_woarm.urdf"
    urdf_path = PROJECT_ROOT / "physgraph_envs" / "assets" / relpath
    if not urdf_path.is_file():
        raise FileNotFoundError(f"Dex-hand URDF does not exist: {urdf_path}")
    body_ids = {body: idx for idx, body in enumerate(body_names)}
    return HandSpec(
        name=name,
        side=side,
        urdf_path=urdf_path,
        dof_names=dof_names,
        body_names=body_names,
        contact_body_names=contacts,
        graph_edges=_reduced_graph(urdf_path, body_names, dof_names),
        finger_groups=tuple(tuple(body_ids[body] for body in group) for group in finger_names),
        synergy_groups=tuple(tuple(body_ids[body] for body in group) for group in synergy),
    )


@dataclass(frozen=True)
class ObservationLayout:
    """One hand's flat observation contract, preserving field-major targets.

    Future frames are grouped *within each field*.  Object-to-body distances
    describe the current state and therefore occur once, whereas fingertip
    targets have one set of five values per future frame.
    """

    spec: HandSpec
    future_length: int = 1

    def __post_init__(self):
        if isinstance(self.future_length, bool) or not isinstance(self.future_length, int) or self.future_length < 1:
            raise ValueError("future_length must be a positive integer")

    @property
    def proprioception_dim(self) -> int:
        return 3 * self.spec.n_dofs + 13

    @property
    def privileged_dim(self) -> int:
        return self.spec.n_dofs + 37

    @property
    def target_dim(self) -> int:
        return 128 + self.spec.n_bodies + self.future_length * (46 + 9 * (self.spec.n_bodies - 1) + 5)

    hand_proprioception_dim = proprioception_dim
    hand_privileged_dim = privileged_dim
    hand_target_dim = target_dim

    @property
    def hand_token_dim(self) -> int:
        return 4 * self.spec.n_dofs + 49

    @property
    def curr_token_dim(self) -> int:
        return 46 * self.future_length + self.spec.n_bodies

    @property
    def link_token_dim(self) -> int:
        return 9 * self.future_length

    @property
    def target_field_shapes(self) -> Dict[str, Tuple[int, ...]]:
        k, links = self.future_length, self.spec.n_bodies - 1
        return {
            "delta_wrist_pos": (k, 3),
            "wrist_vel": (k, 3),
            "delta_wrist_vel": (k, 3),
            "wrist_quat": (k, 4),
            "delta_wrist_quat": (k, 4),
            "wrist_ang_vel": (k, 3),
            "delta_wrist_ang_vel": (k, 3),
            "delta_joints_pos": (k, links, 3),
            "joints_vel": (k, links, 3),
            "delta_joints_vel": (k, links, 3),
            "delta_manip_obj_pos": (k, 3),
            "manip_obj_vel": (k, 3),
            "delta_manip_obj_vel": (k, 3),
            "manip_obj_quat": (k, 4),
            "delta_manip_obj_quat": (k, 4),
            "manip_obj_ang_vel": (k, 3),
            "delta_manip_obj_ang_vel": (k, 3),
            "obj_to_joints": (self.spec.n_bodies,),
            "gt_tips_distance": (k, 5),
            "bps": (128,),
        }

    @property
    def target_slices(self) -> Dict[str, slice]:
        offset, result = 0, {}
        for name, shape in self.target_field_shapes.items():
            size = 1
            for dim in shape:
                size *= dim
            result[name] = slice(offset, offset + size)
            offset += size
        return result

    def split_target(self, target):
        """Slice NumPy/PyTorch-like arrays without importing their libraries."""
        if target.shape[-1] != self.target_dim:
            raise ValueError(f"Expected target dimension {self.target_dim}, got {target.shape[-1]}")
        return {name: target[..., index] for name, index in self.target_slices.items()}
