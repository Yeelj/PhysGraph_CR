"""Resolve explicitly selected OakInk-v2 replacement object assets.

The original OakInk ``object_preview`` and ``coacd_object_preview`` trees are
deliberately not touched here.  Replacement meshes and their matching URDFs
live under one fixed directory and are selected only when ``new_obj`` is
provided by a command.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


NEW_OBJECT_ROOT = Path("/workspace/PhysGraph/data/OakInk-v2/new_obj")
MESH_SUFFIXES = (".ply", ".obj", ".stl")


@dataclass(frozen=True)
class NewObjectAssets:
    identifier: str
    mesh_path: Path
    urdf_path: Path


def replacement_object_for_side(identifier: Optional[str], side: str) -> Optional[str]:
    """Return the replacement identifier for the tool-holding hand only.

    In the bimanual OakInk tasks the right hand holds the replaceable tool and
    the left hand holds the task object.  A single Hydra ``new_obj`` option is
    therefore intentionally scoped to the right-hand dataset.
    """

    side_alias = {
        "right": "right",
        "rh": "right",
        "left": "left",
        "lh": "left",
    }
    normalized_side = side_alias.get(str(side).lower())
    if normalized_side is None:
        raise ValueError(f"Unsupported hand side: {side!r}")
    return identifier if normalized_side == "right" else None


def object_asset_cache_key(obj_id, urdf_path) -> Tuple[str, str]:
    """Identify an Isaac Gym asset by both semantic id and physical bundle."""

    return str(obj_id), str(Path(urdf_path).resolve())


def _asset_candidates(identifier: str, root: Path) -> Tuple[list, list]:
    direct = root / identifier
    if direct.is_dir():
        files = sorted(path for path in direct.iterdir() if path.is_file())
        meshes = [path for path in files if path.suffix.lower() in MESH_SUFFIXES]
        urdfs = [path for path in files if path.suffix.lower() == ".urdf"]
        return meshes, urdfs

    meshes = [
        root / f"{identifier}{suffix}"
        for suffix in MESH_SUFFIXES
        if (root / f"{identifier}{suffix}").is_file()
    ]
    urdf = root / f"{identifier}.urdf"
    return meshes, [urdf] if urdf.is_file() else []


def resolve_new_object_assets(
    identifier: str,
    *,
    root: Optional[Path] = None,
) -> NewObjectAssets:
    """Resolve one replacement mesh/URDF pair from its asset identifier.

    Supported layouts are::

        new_obj/<identifier>.ply   + new_obj/<identifier>.urdf
        new_obj/<identifier>/one_mesh_file + one_urdf_file

    The directory form intentionally requires exactly one mesh and one URDF so
    the visual and collision assets can never be selected ambiguously.
    """

    value = str(identifier).strip()
    if not value:
        raise ValueError("new_obj identifier must not be empty")
    relative = Path(value)
    if relative.is_absolute() or len(relative.parts) != 1 or value in {".", ".."}:
        raise ValueError(
            f"new_obj must be a single asset identifier, not a path: {identifier!r}"
        )

    asset_root = Path(root) if root is not None else NEW_OBJECT_ROOT
    if not asset_root.is_dir():
        raise FileNotFoundError(f"Replacement-object directory does not exist: {asset_root}")
    meshes, urdfs = _asset_candidates(value, asset_root)
    if len(meshes) != 1 or len(urdfs) != 1:
        raise FileNotFoundError(
            "Expected exactly one replacement mesh and one URDF for "
            f"new_obj={value!r} under {asset_root}; found meshes={meshes}, urdfs={urdfs}. "
            "Use either <id>.ply/.obj/.stl + <id>.urdf, or <id>/ containing one of each."
        )
    return NewObjectAssets(
        identifier=value,
        mesh_path=meshes[0].resolve(),
        urdf_path=urdfs[0].resolve(),
    )
