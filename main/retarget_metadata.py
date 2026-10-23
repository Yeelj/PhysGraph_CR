"""Validate retarget trajectories without importing the simulator or datasets.

Retarget DOF columns are in the *exporting actor's* order, not URDF XML
order. New files record that order. Legacy mano2dexhand exports used the
current assets' Isaac Gym order, which matches the canonical hand spec.
"""

import re
import warnings
from pathlib import Path

import numpy as np

from main.dexhand_spec import get_hand_spec


RETARGET_METADATA_VERSION = 1
_HAND_DIRECTORY = re.compile(r"mano2(artimano|allegro|inspire|shadow)_(rh|lh)$")
_SIDES = {"right": "right", "rh": "right", "left": "left", "lh": "left"}


def _side_name(side):
    try:
        return _SIDES[str(side).lower()]
    except KeyError:
        raise ValueError("Unsupported retarget hand side {!r}".format(side)) from None


def _names(value, field, path):
    if isinstance(value, str):
        raise ValueError("{} must be a sequence of names: {}".format(field, path))
    try:
        names = tuple(str(name) for name in value)
    except TypeError:
        raise ValueError("Invalid {} metadata: {}".format(field, path)) from None
    if len(names) != len(set(names)):
        raise ValueError("Duplicate {} in retarget trajectory: {}".format(field, path))
    return names


def _require_same_names(actual, expected, field, path):
    if set(actual) != set(expected) or len(actual) != len(expected):
        raise ValueError(
            "Retarget {} differ from the selected hand; missing={}, extra={}: {}".format(
                field, sorted(set(expected) - set(actual)),
                sorted(set(actual) - set(expected)), path,
            )
        )


def normalize_retarget_data(
    payload, hand_name, side, *, source_path=None, target_dof_names=None,
):
    """Return a validated shallow copy with explicit embodiment metadata.

    ``target_dof_names=None`` preserves the PKL's column order, for offline FK
    and rigid-correction invariants. Dataset loading passes canonical names to
    reorder columns before computing velocities. Unknown legacy filenames are
    supported, but their embodiment cannot be independently established.
    """

    hand_name = str(hand_name).lower()
    side = _side_name(side)
    spec = get_hand_spec(hand_name, side)
    path = str(source_path) if source_path is not None else "<in-memory retarget>"
    result = dict(payload)
    identities = [payload]
    correction = payload.get("contact_retarget")
    if isinstance(correction, dict):
        identities.append(correction)
    for metadata in identities:
        if metadata.get("dexhand") is not None:
            declared = str(metadata["dexhand"]).lower()
            if declared != hand_name:
                raise ValueError(
                    "Retarget dexhand {!r} does not match selected hand {!r}: {}".format(
                        declared, hand_name, path,
                    )
                )
        if metadata.get("side") is not None and _side_name(metadata["side"]) != side:
            raise ValueError(
                "Retarget side {!r} does not match selected side {!r}: {}".format(
                    metadata["side"], side, path,
                )
            )
    if source_path is not None:
        for component in Path(source_path).parts:
            match = _HAND_DIRECTORY.fullmatch(component)
            if match and (match.group(1) != hand_name or _side_name(match.group(2)) != side):
                raise ValueError(
                    "Retarget directory {} does not match selected hand {}/{}: {}".format(
                        component, hand_name, side, path,
                    )
                )

    dof = np.asarray(payload["opt_dof_pos"])
    if dof.ndim != 2 or dof.shape[1] != len(spec.dof_names):
        raise ValueError(
            "Retarget opt_dof_pos shape {} does not match {} DOFs for {}/{}: {}".format(
                dof.shape, len(spec.dof_names), hand_name, side, path,
            )
        )
    if not np.isfinite(dof).all():
        raise ValueError("Non-finite retarget DOF positions: {}".format(path))
    declared_names = payload.get("dof_names", payload.get("opt_dof_names"))
    if declared_names is None:
        # Validated for all eight shipped left/right assets using Isaac Gym,
        # with collapse_fixed_joints=False, as in the original exporter.
        source_names = tuple(spec.dof_names)
        warnings.warn(
            "Legacy retarget trajectory has no dof_names; assuming the original "
            "Isaac Gym {} / {} asset order: {}".format(hand_name, side, path),
            UserWarning, stacklevel=2,
        )
    else:
        source_names = _names(declared_names, "dof_names", path)
    _require_same_names(source_names, spec.dof_names, "dof_names", path)
    if "dof_names" in payload and "opt_dof_names" in payload:
        if source_names != _names(payload["opt_dof_names"], "opt_dof_names", path):
            raise ValueError("Conflicting DOF name metadata: {}".format(path))
    output_names = source_names
    if target_dof_names is not None:
        output_names = _names(target_dof_names, "target_dof_names", path)
        _require_same_names(output_names, spec.dof_names, "target_dof_names", path)
        if output_names != source_names:
            source_index = {name: index for index, name in enumerate(source_names)}
            dof = dof[:, [source_index[name] for name in output_names]]
    result["opt_dof_pos"] = dof
    result["dof_names"] = list(output_names)
    if "opt_dof_names" in result:
        result["opt_dof_names"] = list(output_names)

    if "body_names" in payload:
        body_names = _names(payload["body_names"], "body_names", path)
        _require_same_names(body_names, spec.body_names, "body_names", path)
    else:
        # opt_joints_pos has always been explicitly gathered in class order.
        body_names = tuple(spec.body_names)
    if "opt_joints_pos" in payload:
        joints_shape = np.asarray(payload["opt_joints_pos"]).shape
        expected_shape = (len(dof), len(body_names), 3)
        if joints_shape != expected_shape:
            raise ValueError(
                "Retarget opt_joints_pos shape {} != {}: {}".format(
                    joints_shape, expected_shape, path,
                )
            )
    result.update({
        "retarget_metadata_version": RETARGET_METADATA_VERSION,
        "dexhand": hand_name,
        "side": side,
        "body_names": list(body_names),
    })
    return result
