"""Pure tensor utilities shared by Stage-1 semantic-contact training and tests."""

from __future__ import annotations

import torch


SEMANTIC_FEATURE_DIM = 8
(
    ACTUAL_CONTACT,
    EXPECTED_CONTACT,
    RED_PROXIMITY,
    RED_CONTACT,
    BLACK_PROXIMITY,
    BLACK_VIOLATION,
    FORCE_NORMALIZED,
    CURRICULUM_WEIGHT,
) = range(SEMANTIC_FEATURE_DIM)


def smoothstep01(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def semantic_curriculum_weight(
    step,
    total_steps: int,
    pure_imitation_fraction: float = 0.05,
    imitation_dominant_fraction: float = 0.30,
    full_semantic_fraction: float = 0.60,
    dominant_semantic_weight: float = 0.25,
):
    """0--5%: 0; 5--30%: 0->0.25; 30--60%: 0.25->1; then 1."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not (0.0 <= pure_imitation_fraction < imitation_dominant_fraction < full_semantic_fraction <= 1.0):
        raise ValueError("Curriculum fractions must satisfy 0 <= pure < dominant < full <= 1")

    if not torch.is_tensor(step):
        step = torch.as_tensor(step, dtype=torch.float32)
    progress = step.float() / float(total_steps)
    first = smoothstep01(
        (progress - pure_imitation_fraction)
        / (imitation_dominant_fraction - pure_imitation_fraction)
    ) * dominant_semantic_weight
    second = dominant_semantic_weight + smoothstep01(
        (progress - imitation_dominant_fraction)
        / (full_semantic_fraction - imitation_dominant_fraction)
    ) * (1.0 - dominant_semantic_weight)
    return torch.where(
        progress < imitation_dominant_fraction,
        first,
        torch.where(progress < full_semantic_fraction, second, torch.ones_like(progress)),
    )


def semantic_reward_terms(
    actual_contact: torch.Tensor,
    expected_contact: torch.Tensor,
    red_quality: torch.Tensor,
    black_violation: torch.Tensor,
    force_magnitude: torch.Tensor,
    force_safe: float,
    eps: float = 1e-6,
):
    """Return normalized red reward and black/missing/force penalties per env."""
    expected_sum = expected_contact.sum(dim=-1).clamp_min(eps)
    actual_sum = actual_contact.sum(dim=-1).clamp_min(eps)
    red_reward = (expected_contact * actual_contact * red_quality).sum(dim=-1) / expected_sum
    black_penalty = black_violation.sum(dim=-1) / actual_sum
    missing_penalty = (expected_contact * (1.0 - actual_contact)).sum(dim=-1) / expected_sum
    force_penalty = (
        actual_contact * torch.relu(force_magnitude / float(force_safe) - 1.0).square()
    ).sum(dim=-1) / actual_sum
    raw = red_reward - 2.0 * black_penalty - 0.2 * missing_penalty - 0.05 * force_penalty
    return red_reward, black_penalty, missing_penalty, force_penalty, raw
