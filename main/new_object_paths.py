"""Resolve explicitly selected OakInk-v2 replacement object assets.

Replacement assets mirror OakInk's original split layout: the semantic/surface
mesh lives under ``new_obj/object_preview`` while the physical collision mesh
and URDF live under ``new_obj/coacd_object_preview``.  They are selected only
when ``new_obj`` is provided by a command.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


NEW_OBJECT_ROOT = Path("data/OakInk-v2/new_obj")
MESH_SUFFIXES = (".ply", ".obj", ".stl")


@dataclass(frozen=True)
class NewObjectAssets:
    identifier: str
    mesh_path: Path
    collision_mesh_path: Path
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


def _named_mesh_candidates(directory: Path, identifier: str) -> list:
    return [
        directory / f"{identifier}{suffix}"
        for suffix in MESH_SUFFIXES
        if (directory / f"{identifier}{suffix}").is_file()
    ]


def resolve_new_object_assets(
    identifier: str,
    *,
    root: Optional[Path] = None,
) -> NewObjectAssets:
    """Resolve one replacement semantic mesh and physical collision bundle.

    The required layout mirrors the original OakInk asset trees::

        new_obj/object_preview/<identifier>/<identifier>.ply
        new_obj/coacd_object_preview/<identifier>/<identifier>.ply
        new_obj/coacd_object_preview/<identifier>/<identifier>.urdf

    ``mesh_path`` always identifies the semantic/surface mesh.  The CoACD mesh
    is exposed separately for validation, while existing callers continue to
    load the physical asset through ``urdf_path``.
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

    semantic_directory = asset_root / "object_preview" / value
    collision_directory = asset_root / "coacd_object_preview" / value
    semantic_meshes = _named_mesh_candidates(semantic_directory, value)
    collision_meshes = _named_mesh_candidates(collision_directory, value)
    urdf_path = collision_directory / f"{value}.urdf"
    urdfs = [urdf_path] if urdf_path.is_file() else []

    if len(semantic_meshes) != 1 or len(collision_meshes) != 1 or len(urdfs) != 1:
        raise FileNotFoundError(
            "Expected exactly one named semantic mesh and one named CoACD mesh/URDF for "
            f"new_obj={value!r} under {asset_root}; "
            f"semantic_meshes={semantic_meshes}, collision_meshes={collision_meshes}, "
            f"urdfs={urdfs}. Required layout: "
            "object_preview/<id>/<id>.<mesh> and "
            "coacd_object_preview/<id>/<id>.<mesh|urdf>."
        )
    return NewObjectAssets(
        identifier=value,
        mesh_path=semantic_meshes[0].resolve(),
        collision_mesh_path=collision_meshes[0].resolve(),
        urdf_path=urdfs[0].resolve(),
    )
