"""Camera observations: 2D affine map from base metres to camera units."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Camera:
    matrix: tuple[tuple[float, float], tuple[float, float]] = ((1.0, 0.0), (0.0, 1.0))
    offset: tuple[float, float] = (0.0, 0.0)

    def observe(self, tip_m):
        return np.asarray(self.matrix) @ tip_m + np.asarray(self.offset)
