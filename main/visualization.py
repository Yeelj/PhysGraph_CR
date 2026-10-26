"""Display-only options for bimanual policy testing (no simulator imports)."""

from dataclasses import dataclass, fields
import math
from numbers import Real
from typing import Optional, Tuple


RGB = Tuple[float, float, float]


@dataclass(frozen=True)
class TestVisualizationConfig:
    left_hand_color: Optional[RGB] = None
    right_hand_color: Optional[RGB] = None
    object_color: Optional[RGB] = None
    tool_color: Optional[RGB] = None
    show_mano: bool = True
    contact_highlight: bool = True

    @classmethod
    def from_mapping(cls, values=None, *, training=False):
        # Ignore test overrides entirely during training, including validation.
        if training or values is None:
            return cls()
        values = dict(values)
        unknown = values.keys() - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown test_visualization options: {sorted(unknown)}")
        for name in ("left_hand_color", "right_hand_color", "object_color", "tool_color"):
            color = values.get(name)
            if color is None:
                continue
            message = f"test_visualization.{name} must be null or three RGB numbers in [0, 1]"
            try:
                color = tuple(color)
            except TypeError:
                raise ValueError(message) from None
            if len(color) != 3 or any(
                isinstance(channel, bool)
                or not isinstance(channel, Real)
                or not math.isfinite(channel)
                or not 0 <= channel <= 1
                for channel in color
            ):
                raise ValueError(message)
            values[name] = tuple(float(channel) for channel in color)
        for name in ("show_mano", "contact_highlight"):
            if name in values and not isinstance(values[name], bool):
                raise ValueError(f"test_visualization.{name} must be true or false")
        return cls(**values)
