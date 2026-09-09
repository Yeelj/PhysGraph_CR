"""Persistent nearest-neighbour memory for right-hand contact retargeting."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
from scipy.spatial import cKDTree


GLOBAL_SAMPLE_COUNT = 256
RED_SAMPLE_COUNT = 128
CONTEXT_SAMPLE_COUNT = 128
CONTEXT_RADIUS = 0.03


def safe_path_component(value: str, label: str) -> str:
    result = str(value).strip()
    path = Path(result)
    if not result or path.is_absolute() or len(path.parts) != 1 or result in {".", ".."}:
        raise ValueError(f"{label} must be one non-empty path component, got {value!r}")
    return result


def corrected_retarget_path(
    output_root: Path,
    hand_directory: str,
    data_id: str,
    tool_mesh_name: str,
    filename: str,
) -> Path:
    return (
        Path(output_root)
        / "OakInk-v2"
        / safe_path_component(hand_directory, "hand directory")
        / safe_path_component(data_id, "data_id")
        / safe_path_component(tool_mesh_name, "tool mesh name")
        / safe_path_component(filename, "retarget PKL filename")
    )


def right_hand_retarget_path(
    original_root: Path,
    corrected_root: Path,
    hand_directory: str,
    data_id: str,
    tool_mesh_name: str | None,
    filename: str,
    *,
    correct: bool,
) -> tuple[Path, str]:
    if correct:
        if tool_mesh_name is None:
            raise ValueError("correct=true requires new_obj for the right-hand dataset")
        return (
            corrected_retarget_path(
                corrected_root,
                hand_directory,
                data_id,
                tool_mesh_name,
                filename,
            ),
            "contact_corrected",
        )
    return (
        Path(original_root)
        / "OakInk-v2"
        / safe_path_component(hand_directory, "hand directory")
        / safe_path_component(filename, "retarget PKL filename"),
        "original",
    )


def bank_sequence_dir(bank_root: Path, data_id: str) -> Path:
    return Path(bank_root) / safe_path_component(data_id, "data_id")


def _atomic_json_dump(data: Mapping, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", delete=False
    ) as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _atomic_descriptor_dump(descriptor: Mapping[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b", dir=path.parent, prefix=path.name + ".", delete=False
    ) as stream:
        np.savez_compressed(stream, **descriptor)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def stable_mapping_hash(data: Mapping) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resample_points(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        raise ValueError("Cannot describe an empty point set")
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    points = points[order]
    indices = np.linspace(0, len(points) - 1, count).round().astype(np.int64)
    return points[indices].astype(np.float32)


def build_mesh_descriptor(tool) -> Dict[str, np.ndarray]:
    vertices = np.asarray(tool.mesh.vertices, dtype=np.float64)
    faces = np.asarray(tool.mesh.faces, dtype=np.int64)
    global_points = [vertices]
    if len(faces):
        global_points.append(vertices[faces].mean(axis=1))
    global_points = np.concatenate(global_points, axis=0)

    red_points = np.asarray(tool.red_points, dtype=np.float64)
    black_points = np.asarray(tool.black_points, dtype=np.float64)
    red_tree = cKDTree(red_points)
    black_to_red, _ = red_tree.query(black_points, workers=1)
    context_points = black_points[black_to_red <= CONTEXT_RADIUS]
    if len(context_points) == 0:
        keep = max(1, min(len(black_points), int(np.ceil(0.1 * len(black_points)))))
        context_points = black_points[np.argsort(black_to_red)[:keep]]

    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    bbox_extent = bounds_max - bounds_min
    diagonal = max(float(np.linalg.norm(bbox_extent)), 1e-12)
    global_center = np.median(global_points, axis=0)
    red_center = np.median(red_points, axis=0)
    red_centered = red_points - red_center
    covariance = red_centered.T @ red_centered / max(1, len(red_centered))
    red_pca_values = np.sort(np.linalg.eigvalsh(covariance))[::-1] / (diagonal**2)
    red_count = float(getattr(tool, "red_vertex_count", len(red_points)))
    black_count = float(getattr(tool, "black_vertex_count", len(black_points)))

    return {
        "global_points": _resample_points(global_points, GLOBAL_SAMPLE_COUNT),
        "red_points": _resample_points(red_points, RED_SAMPLE_COUNT),
        "context_black_points": _resample_points(context_points, CONTEXT_SAMPLE_COUNT),
        "bbox_extent": bbox_extent.astype(np.float32),
        "diagonal": np.asarray(diagonal, dtype=np.float32),
        "global_center": global_center.astype(np.float32),
        "red_center": red_center.astype(np.float32),
        "red_fraction": np.asarray(red_count / max(red_count + black_count, 1.0), dtype=np.float32),
        "red_pca_values": red_pca_values.astype(np.float32),
    }


def _normalised_points(descriptor: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    points = np.asarray(descriptor[key], dtype=np.float64)
    center = np.asarray(descriptor["global_center"], dtype=np.float64)
    diagonal = max(float(np.asarray(descriptor["diagonal"])), 1e-12)
    return (points - center) / diagonal


def _symmetric_chamfer(left: np.ndarray, right: np.ndarray) -> float:
    left_tree = cKDTree(left)
    right_tree = cKDTree(right)
    left_to_right, _ = right_tree.query(left, workers=1)
    right_to_left, _ = left_tree.query(right, workers=1)
    return float(0.5 * (left_to_right.mean() + right_to_left.mean()))


def descriptor_distance(query: Mapping[str, np.ndarray], stored: Mapping[str, np.ndarray]) -> Dict[str, float]:
    red = _symmetric_chamfer(
        _normalised_points(query, "red_points"),
        _normalised_points(stored, "red_points"),
    )
    context = _symmetric_chamfer(
        _normalised_points(query, "context_black_points"),
        _normalised_points(stored, "context_black_points"),
    )
    global_shape = _symmetric_chamfer(
        _normalised_points(query, "global_points"),
        _normalised_points(stored, "global_points"),
    )

    query_diagonal = max(float(np.asarray(query["diagonal"])), 1e-12)
    stored_diagonal = max(float(np.asarray(stored["diagonal"])), 1e-12)
    query_extent = np.asarray(query["bbox_extent"], dtype=np.float64) / query_diagonal
    stored_extent = np.asarray(stored["bbox_extent"], dtype=np.float64) / stored_diagonal
    statistics = float(
        np.mean(np.abs(query_extent - stored_extent))
        + abs(np.log(query_diagonal / stored_diagonal))
        + abs(float(np.asarray(query["red_fraction"])) - float(np.asarray(stored["red_fraction"])))
        + np.mean(
            np.abs(
                np.asarray(query["red_pca_values"], dtype=np.float64)
                - np.asarray(stored["red_pca_values"], dtype=np.float64)
            )
        )
    )
    total = 0.40 * red + 0.30 * context + 0.20 * global_shape + 0.10 * statistics
    return {
        "total": float(total),
        "red": red,
        "context": context,
        "global": global_shape,
        "statistics": statistics,
    }


def load_sequence_metadata(sequence_directory: Path) -> Dict | None:
    path = Path(sequence_directory) / "sequence.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid memory-bank sequence metadata: {path}")
    return value


def require_compatible_sequence(sequence_directory: Path, expected: Mapping) -> bool:
    stored = load_sequence_metadata(sequence_directory)
    if stored is None:
        return False
    mismatches = {
        key: (stored.get(key), value)
        for key, value in expected.items()
        if stored.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Memory-bank sequence context is incompatible at {sequence_directory}: {mismatches}"
        )
    return True


def load_ranked_entries(
    sequence_directory: Path,
    query_descriptor: Mapping[str, np.ndarray],
    *,
    top_k: int = 3,
) -> list[Dict]:
    ranked = []
    for entry_path in sorted(Path(sequence_directory).glob("*/entry.json")):
        with entry_path.open("r", encoding="utf-8") as stream:
            entry = json.load(stream)
        validation = entry.get("validation", {})
        if not bool(validation.get("overall_passed", False)):
            continue
        corrected_path = Path(entry.get("corrected_pkl", ""))
        descriptor_path = entry_path.parent / entry.get("descriptor_file", "descriptor.npz")
        if not corrected_path.is_file() or not descriptor_path.is_file():
            continue
        expected_pkl_hash = entry.get("corrected_pkl_sha256")
        if not expected_pkl_hash or _file_sha256(corrected_path) != expected_pkl_hash:
            continue
        with np.load(descriptor_path) as archive:
            descriptor = {key: archive[key] for key in archive.files}
        distance = descriptor_distance(query_descriptor, descriptor)
        entry["entry_path"] = str(entry_path.resolve())
        entry["similarity"] = distance
        ranked.append(entry)
    ranked.sort(key=lambda item: item["similarity"]["total"])
    return ranked[: max(0, int(top_k))]


def write_memory_entry(
    sequence_directory: Path,
    sequence_metadata: Mapping,
    tool_mesh_name: str,
    descriptor: Mapping[str, np.ndarray],
    entry: Mapping,
) -> None:
    sequence_directory = Path(sequence_directory)
    stored = load_sequence_metadata(sequence_directory)
    if stored is not None and dict(stored) != dict(sequence_metadata):
        raise ValueError(
            f"Refusing to overwrite incompatible memory-bank sequence metadata: {sequence_directory}"
        )
    if stored is None:
        _atomic_json_dump(sequence_metadata, sequence_directory / "sequence.json")

    entry_directory = sequence_directory / safe_path_component(tool_mesh_name, "tool mesh name")
    _atomic_descriptor_dump(descriptor, entry_directory / "descriptor.npz")
    _atomic_json_dump(entry, entry_directory / "entry.json")
